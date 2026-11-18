from __future__ import annotations
import json
import random
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'
LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
HEAVY_ROOM = 0.02
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
BRIEF_BY = 120.0
BRIEF_GAP = 10.0
OUTAGE_BY = 150.0
SHOW_CHARS = 450
HUNT_STEPS = 24
ENV_GAP = 0.25
OPEN_GAP = 0.5
TABLE_KEYS = 18
CHEAPER_SHOWN = 4
OVERRIDE_MARK = r"\bnot (?:taken )?(?:from |to |by )?this\b|\brather than (?:to )?this\b|\binstead of this\b|\bfrom that (?:product|item|listing)\b|\b(?:another|a different) (?:product|listing|item)\b"  # OV: the sentence says the value is not this item's own
OVERRIDE_FOR = r"(?:for|as for|regarding) (?:its |the |this item's )?([a-z][a-z0-9 /\-]{1,40}?), (?:check|see|look at|look to|refer to|consult|go by|use|take it from|get it from)\b(.*)"  # OV: needs no marker
OVERRIDE_TAKE = r"(?:take|get) (?:its |the )?([a-z][a-z0-9 /\-]{1,40}?) from\b(.*)"  # OV: needs no marker
OVERRIDE_TAKEN = r"(?:its |the |this item's )?(?:whether it is )?([a-z][a-z0-9 /\-]{1,40}?) (?:comes|is to be taken|should be taken|must be taken|is taken|is copied|is borrowed|is drawn) from\b(.*)"  # OV
CHEAP_HOLDS = 2  # CS: checkouts held at most this often for a cheaper opened listing that shows every requirement
CHEAP_LIMIT = r'\d|\b(?:at least|at most|minimum|maximum|under|over|below|above|between|up to|no more|no less|not|unlike|except|without|other than|same as|like the|near)\b'  # CS: a requirement word-matching cannot prove
LIMIT_WORDS = r'\b(?:at least|at most|minimum|maximum|under|over|below|above|between|up to|no more|no less|not|unlike|except|without|other than|same as|like the|near|or more|or less|or fewer)\b'  # CW
COND_MARK = r'[,;]?\s+(?:only\s+)?(?:if|unless|except(?:\s+when|\s+if)?|provided(?:\s+that)?|in case|so long as|as long as)\s+(?:the|its|a|an|that|this)\b'  # CO
LIMIT_NUMBER = r'\b(?:at least|at most|minimum|maximum|min|max|under|over|below|above|no more than|no less than|up to|or more|or less)\b[^,;]*\d|\d[^,;]*\b(?:or more|or less|or fewer|at most|at least|maximum|minimum)\b'  # UL
SALVAGE_ROOM = 25.0  # SP: seconds of model time a salvage needs (one review call, then store calls)
SALVAGE_EACH = 4  # SP: reviewed listings per item tried in a set
QUIET_WORD = 4
OPEN_LATER = 2
PAGE_ATTR_CHARS = 700
PAGE_REST_CHARS = 300
NO_NAME = ('none', 'null', 'n/a', 'unknown')
ASK_TURNS = 2
MESSAGE_FIELD = '(?:^|_)(?:texts?|body|messages?|notes?|contents?|questions?|msg|say)(?:_|$)'
NOT_MESSAGE = '(?:^|_)(?:id|ids|type|format|code|lang|language|kind)$'
TRAIL_ROOM = 15
FIT_PAGES = 2
CRASH_TURNS = 3
OPEN_MIN_STEPS = 20
OPEN_ROWS = 5
OPEN_ITEMS = 2
OPEN_SHARE = 5
OPEN_EACH = 6
OPEN_NOT = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or|same|listing)\\b'
SHORT_KEEP = 8
SET_SHOWN = 2
STOCK_OUT = '^(?:no|none|false|0|n)$|out[ _-]?of[ _-]?stock|sold[ _-]?out|unavailable|not[ _-]?available'
HUNT_WHOLE = 49
HUNT_GRACE = 25.0
HUNT_VALUE_CHARS = 30
ASK_WORDS = '\\b(?:one|each|single|first|time|many|much|several|list|once|separate\\w*|split|short\\w*|questions?|ask\\w*|turn|lot|too|all)\\b'
HUNT_ROUNDS = 2
HUNT_KEEP = 10
HUNT_IDLE = 4
HUNT_PAGE_NEW = 5
HUNT_FIT_WORDS = 4
HUNT_FIT_OPENS = 4
HUNT_COMMON = 0.6
HUNT_BY = 200.0
TITLE_CHARS = 200
MIX_LABELS = 3
TEXT_CHARS = 3000
OBS_LIMIT = 15000
HISTORY_LIMIT = 90000  # CT
GATE_ROUNDS = 3
AUTO_PAGES = 6
CHEAPER_OPENS = 3
TASK_SECONDS = 300.0
TIME_MARGIN = 20.0
CALL_FLOOR = 6.0
ASK_BY = 165.0
QUESTION_BY = 210.0
WIND_DOWN = 200.0
LAST_TURN = 255.0
EMPTY_ROOM = 12.0
LATE_BY = 190.0
LATE_STEPS = 8
SCARCE_MESSAGES = 2
ASK_SINGLES = True  # AK: Division A (singles): one property of one item per message, best ranked first; several items take turns
LIST_SIZE = 12
LIST_MIXED = 8
LIST_STEPS = (1, 3)  # how many labels questions 1 and 2 carry before a list grows to its full size
LIST_STEPS_PAIR = (3, 8)  # PL: two items, one question each: the second names up to LIST_MIXED labels
LIST_STEPS_MIXED = (3, 3)  # same idea when the questions spread over several items (sim: as good as 3,3,8 or better than 1,3,8 for 2-3 items)
CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
CAP_NOT = 'result|row|page|count|size|num|item|step'
CAP_PRICED = 'price|cost|budget|spend|money|amount'  # name of a limit argument while no search has yet shown it working
CAP_LOW = '(?:^|_)(?:min|minimum|low|lowest|least|from|start|offset|floor)(?:_|$)'
EARLY_TRAILS = 3  # rounds of chain following allowed early on; checkout has a separate budget
CAP_CHANCE = 0.05  # a value counts as a held limit only if random rows from its labels would rarely all fall below it
WALK_ROOM = 10  # extra steps past the finish for each listing walk after the first, so a refused checkout can still swap
WALK_CALLS = 4  # at most this many shelf listings per cart line in one checkout (race 16: shelves up to 456 rows, 6 per page)
SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
LOOP_WAIT = 22.0  # LM: seconds a light-model loop turn may take before the spare model answers it
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short: dict[int, ProxyClient] = {}
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = """THE JOB
Your name here is Cairn. You shop inside an online store for a single shopper, and nothing happens except through calls to that store's tools. Grading is binary: 1 for a flawless finish, otherwise 0. A flawless finish is one of two things. (a) An order holding every item asked for, in the quantity asked, where each one satisfies all requirements active at checkout time and the total stays within budget. (b) If this store genuinely has nothing meeting all of that, closing the episode with the no-purchase tool in the manner the request specifies.

HOW TO FINISH
- Get the cart to hold precisely your chosen lines (take out anything extra), then make the checkout call by itself.
- Ending without buying requires that a real search turned up no listing meeting everything. Running low on steps proves no such thing; when steps run short, check out the best cart you have. Phrase that ending the way the request tells you; if some remark is called for, make it brief and mention no listing.
- Every reply needs at least one tool call. Write little: the facts so far, then the move you make next.

WHICH QUALIFYING LISTING TO TAKE
- "Cheapest" requests (lowest total, smallest price): your order needs the lowest-priced option among ALL qualifying listings. Collect that whole qualifying set by paging (rows often arrive unsorted by price; for a short result Cairn pulls further pages on its own), test candidates from cheapest upward, and keep the first one that passes on every point. Where a search lets you set a price ceiling, set it below what your current best pick costs, so the category list shows only cheaper rivals still to test.
- "Most" or "least" requests on some measure: line up every qualifying listing on that measure and keep the extreme one; listings silent about the measure are out of the running.
- Anything else: whichever listing passes every requirement is right, so don't burn steps searching for a nicer one.
- Add an item to the cart as soon as an opened listing passes every requirement for it (for cheapest, most or least: once the qualifying set has been compared), and continue with the next item. Carted lines frequently go out of stock a little after being added; carting early keeps steps in hand to swap them.
- When several distinct items are asked for, each gets its own cart line at its own requested quantity.

DOES A LISTING QUALIFY?
Kind of product
- The category label decides what a listing is. Only a label naming the requested item in full counts; a nearby label never does. Anything filed under the requested label is that product, even when its title or description talks about something else, so just test it on requirements, price and stock. A label that itself expresses a requirement's value (say, a season or an audience) satisfies that requirement by itself.
Attribute values
- For a requirement on a property, the listing passes when its attribute for that property gives the value. Key formatting is irrelevant (spaces, underscores, camel case, prefixes). One exception: keys for a base or boost clock or frequency (e.g. base clock, base frequency, boost clock) never answer a requirement on processor speed or clock speed; likewise a GHz number in title or description is no answer when the attributes carry only that kind of base or boost number. Matching one entry of a list split by commas or slashes is enough.
- Values are compared on their words. Letter case, singular versus plural and plain synonyms make no difference, and a value that includes all of the required value's words passes ("deep teal" for teal, "extra roomy fit" for roomy). A value worded differently fails: abbreviations and codes do not equal the word they stand for, and "teal green" is not "sea green". Numbers get converted to common units first; boundaries are inclusive, and a range passes only if all of it lies inside the limit.
- A key that names the requirement's property (that word or a plain synonym) is final: when it gives another value, the listing fails no matter what the title or other keys suggest. When a key prefixes the property with the item's own name (a bag's "bag shape"), its value is accepted even if a plain key ("shape") disagrees. A key belonging to some other property, even a close one (handle colour is not the colour of the body), tells you nothing about the requirement. Cairn appends [evidence] lines to each opened listing, giving the keys whose names correspond to a requirement.
- When no attribute covers the property, the wanted value may come from a different attribute (a list of features or styles that contains it, for instance) or from title or description text. A number measuring something else does not count (weight is not capacity), and neither does a bare number whose text leaves its meaning unstated.
Requirements that point elsewhere
- Some requirements name another listing rather than a value (something like "X identical to that of the <property> item <shop> sells", or "X no more than N away from the one shown on ...'s page"). Resolve these first. Track the referenced listing down (search by its category, brand, shop or words), then open its complete page through the single-item or side-by-side tool, because search rows are not sufficient. Should that page in turn send you elsewhere for the property, track down and open that next listing too; the value comes from the last page. If the requirement asks for what accompanies that value (partner, counterpart) and the page says what the value goes with, then that companion value is the requirement. An order is valid only if each page along the chain was opened. When the requirement wants any value except that listing's, it excludes just listings carrying that exact value, and it alone never justifies ending without a purchase.
- Before buying a listing whose page sends you somewhere else for more details on it, go and look there.
Linked items
- When one item asked for must suit another (an accessory for a device), decide the pair jointly from the wording on the listings (a list of compatible products, a model or series code printed on both pages). If no accessory passing its own requirements suits the device you chose, change to some other device that passes its requirements and that a suitable accessory works with; while that change has not been tried, ending without buying is not allowed.
- A requirement that applies only depending on a different requested item ("only if", "unless", "except when" that item has some value) binds only when your pick for that item meets the condition. Either choose that other item so the condition fails, or satisfy the requirement. The condition is about your own cart and does not send you to look up some listing.
Further limits
- Exclusions and closed lists are binding. Wishes ("ideally", "would be nice") only break ties.
- Never buy something the store marks as sold out. The order total (sum over all lines of price times quantity) may equal the budget but not exceed it.

WORKING THE STORE
- Every tool call uses up one of a limited supply of steps; thinking is free. Tools, their argument names and their output formats differ from store to store, so read the description and schema of each tool and use the argument names and permitted values exactly as given.
- Searching may be restricted: maybe just a category filter, a keyword mode reading only the first few words of a query, pages of a handful of rows, a token or offset to get the following page. When your answer depends on having seen every row, weigh the total reported against the rows shown so far, and keep paging with exactly the token or offset the store handed back.
- Short keyword queries that open with the value work best (wanted value, then the product noun). The store may search attribute values and shop names along with titles and may demand all words, so two or three required values and the noun can surface nearly the entire qualifying set; a model or part number can come back empty, and one attempt is enough to know.
- An empty result from a filtered search usually means the filter parameter or its value is off. Repeat it with the label spelled exactly as the result rows spell it, or switch to another filter parameter.
- Search rows only summarise. How a listing's record is split varies by store: one view may give its attributes (properties), another only its offer and description, and a comparison view may give either. When Cairn has worked out which tool reveals attributes, the ledger says so. Decide each property requirement from those attributes, so read them for every listing you might purchase; a single call covering several listings saves steps if it reveals what you need. Do not reopen a listing you have already read unless something changed.

THE SHOPPER AND SIDE TASKS
- Cairn's interviewer handles every question to the shopper, so you never send the shopper anything yourself. Answers appear in the conversation and get added to the ledger.
- A request may ask for things along the way, such as an approval, a message, or something to read or check first; the ledger lists them. Those due immediately before the order or the ending are performed by Cairn at the moment you call checkout or the ending tool, so don't wait on them and never end without buying because of them. Handle the rest with the store's tools (questions about products go via Cairn's interviewer).

CHANGES MID-TASK
- The shopper or the store may alter the situation: a fresh budget, a requirement added or removed, a new price, a listing gone out of stock. Such news arrives inside tool output, often as a refusal at checkout; a note is valid whichever tool's output shows it, and the ledger shows the requirements currently in force. A refusal does not end the episode: take every change on board, fix the cart, then retry the checkout.
- Stock and prices may also move with no notice. When you call the checkout, Cairn re-reads each cart line and views the cart, so spend no turn on either."""

BRIEF = '''ROLE: task planner. Input below: a store's tool list and one buyer's request. Job: turn both into a structured plan a shopping agent will follow.

KEYS, WITH MEANINGS:
"roles": object mapping each job to the exact tool name that performs it; null for a job no tool covers. Jobs: "search"; "open" (reveals a single listing's attributes or properties; if two tools each show one listing, take the tool whose description mentions attributes, properties or specifications); "offer" (reveals a single listing's live price and availability, only when a tool separate from "open" does this); "compare" (full records of many listings at once); "ask" (writes to the buyer); "cart" (edits the cart); "cart_view" (displays the cart); "buy" (completes checkout); "refuse" (closes the session with no order).
"search_words": {"tool": name of the tool that looks listings up by free text (a category-browsing tool loses to one that accepts text), "field": the parameter receiving that text, "args": constant arguments that switch it into text-search mode (e.g. its mode value), else {}}.
"buy_args": constant arguments that make the "buy" tool perform checkout (e.g. its mode value), else {}.
"refuse_args": constant arguments that make the "refuse" tool close with no order, else {}.
"items": one entry per separate product to purchase. Entry shape:
  {"name": product type, phrased as the request phrases it;
   "shelf": a category label a store tool lists, copied verbatim; failing that, a category the request states ("boots from footwear" -> "footwear"); failing that, "";
   "quantity": integer count of this one listing; default 1; a pair or set sold as one listing = 1; a number written ahead of the product ("4 x mugs", "2 rugs") is the count even if each unit is a pack, box or set;
   "musts": hard requirements, each a string "property: constraint" carrying the precise value, bound, exclusion or fixed list of options}.
"budget": spending limit for the entire order, as a number in the unit the request uses; null if absent.
"budget_unit": that unit spelled out (e.g. "cents"); null if absent.
"budget_scope": "total" = ceiling applies to the order sum; "each" = ceiling applies per item; null = no ceiling.
"in_stock_only": true, except when the request plainly allows unavailable items.
"pick": "cheapest" for a lowest-price or lowest-total demand; "most" / "least" for a demand to maximise / minimise some measure; otherwise "any". Complaints about a tight wallet do not set a pick.
"pick_measure": the measure behind "most"/"least"; "" otherwise.
"lookups": musts (verbatim copies) whose value has to be fetched first from some other listing the request refers to (identified by seller, type or properties); else []. Musts conditioned on another requested item never go here.
"fits": [{"item": item number, "with": item number}] for every requested item that, per the request, has to match, fit or suit a DIFFERENT requested item (whatever the agent selects for it); [] if none.
"messages": integer cap on agent-to-buyer messages; null if unstated.
"steps": integer cap on steps, moves or tool calls; null if unstated.
"ask_first": points the request explicitly wants confirmed with the buyer before selection; [] if none.
"process": extra actions demanded beyond locating and purchasing the correct items (obtaining sign-off, informing or showing the buyer something, confirming a point with them, reading something beforehand, calling a specific tool). Entry: {"step": the action, 15 words max, "when": "choose" = before selecting products, "order" = immediately before checkout, "end" = immediately before closing with no order, "messages": number of buyer messages it costs (0 if none)}. [] if none demanded. Not process: item requirements, budget, stock, message/step caps, how to close when nothing qualifies.
"quiet_decline": true if the request forbids proposing or naming substitutes when nothing qualifies; false if it requests them; null if silent.
"rules": every working or closing instruction found in the request, 12 words max each.
"wishes": soft preferences only.

RULES FOR MUSTS:
1. Property name = the request's own term ("intended for a picnic occasion" -> "occasion: picnic"; "a 30-inch strap" -> "strap length: 30 inch"). Never substitute an adjacent property.
2. Price, budget, stock and quantity never appear in musts.
3. Value taken from another listing -> exactly ONE must that embeds the full description of that listing ("handle colour: unlike the handle colour of the teapot Brevik sells with a steel base"). Seller, type and any trailing "with <property> <value> and <property> <value>" stay inside that must; none of them becomes a must of this product.
4. Value conditioned on ANOTHER requested item, i.e. on what the agent chooses for it ("the strap must be leather should the case of the watch be gold", "a blue mat, except when the table is pine") -> one must on the constrained item, written "property: value, only if the <other item> has <its property> <values>" or "property: value, unless the <other item> has <its property> <values>"; preserve the request's logic.
5. Background (who will use it, where it goes, past failures, the reason for buying) is context, not a must, unless it states a property value or bound outright.
6. Words forming the product's own name (an audience or season attached to the product) go into "name", never into musts.
7. A use or occasion stated as a property of the product itself ("shoes for formal wear", "a skirt designed for dancing", "a hat, and it is for a funeral", "the occasion must be formal") = a must on that property ("occasion: formal"), unless the store's category label already says it. Context instead: a specific person's event, a recipient or a setting ("to wear at her graduation", "for my uncle"), and how it should fit, feel or look when used.
8. One phrase joining products with "and" (a single "I want ..." with a single quantity, often a store department name) = ONE item. Split into several items only when each product gets its own quantity or its own requirements.

OUTPUT: ONE JSON object (compact, no indentation, nothing else) holding every key above.
'''

PLAN = """Task: per requested item, pick which attributes the agent should ask the shopper about.

Background:
- Shoppers hide a handful of firm wishes, each tied to one attribute of one requested product.
- A wish surfaces only through a question naming exactly its attribute. Indifferent attributes get "don't mind"; ones never pondered get bewilderment. So every attribute mentioned eats into a message.
- Inputs: request; items with what the request already fixes for each; per category, numbered attribute keys gathered from opened listings, with carrying-listing counts and sample values. Hidden wishes may also concern attributes absent from these keys.

"label": bare attribute name. Listed keys keep every word, normally spelt ("lid colour" for lidColour, "rim finish" for rim_finish - never reduced to "colour" or "finish"). No sample values, brackets or second attribute.

Exclude:
- attributes already fixed for that item (some requirement names it). Exception: keys whose values merely echo a required value (e.g. pattern key showing the demanded decoration) are NOT fixed;
- synonyms of an already chosen attribute;
- keys where every value bundles several purchase options;
- identifiers, model names, warranty, other logistics.
Remember:
- pack size, count, quantity per pack: ordinary attributes;
- key tagged [not the stated ...] = different attribute from that requirement, whatever words its name shares;
- each item ranked independently - hints first, keys after.

Ranking order (per item):
0. ASK FIRST entries, if present: points the request wants confirmed with the shopper, each named by its attribute.
1. Hinted attributes. Stories often gesture at a hidden wish yet withhold its value - lines about: something beyond the shopper's judgement or delegated to you; whose taste decides; appearance or the setting it goes in; who uses or wears it; its contents. Per such line: one attribute it gestures at + which item, following story order.
   - Prefer that item's matching listed key: wearer hint -> gender/audience key; appearance hint -> whichever colour, pattern or finish key most of its listings carry.
   - Hint with no key on any of the item's listings -> rank below that item's three most-carried keys.
2. Listed keys next: first what buyers of that product kind choose between up front (typically type or style, form or fit, size/capacity/dimensions, material, colour or pattern - whichever drives that kind's purchase), plus keys most of its listings carry. After 5+ listings were read, avoid keys found on just one or two.

Answer with: ONE JSON object and nothing else; up to COUNT picks per item, best first, each item's picks in rank order:
{"ask": [{"item": item number, "label": "...", "hint": "max 8 story words it rests on, or an empty string"}]}"""
HEAR = """Task: read every shopper reply to the agent's questions and extract what it reveals.

Per field:
- "needs" - firm preferences stated in any reply, exact value/limit with unit ("at least 30 cm", "oval", "not wicker").
  - Property = whichever one the reply names (may differ from what was asked). Several replies on one property -> one need.
  - Value copied verbatim, nouns + units kept ("at least 3 shelves", not "at least 3"); only what a listing could state, starting at the wanted value or its limit word, running to its end; strip surrounding framing: "matching oval", "settled, with oval", "oval and nothing else" -> "oval"; "equal to at least 30 cm" -> "at least 30 cm".
  - "points": true if the value must be read off another listing, else false.
  - Reply pointing at another listing (value like / unlike / near what some shop's product has, that product singled out by a few of its properties): ONE need on the reply's named property, "points": true; value = reply's own words verbatim from comparison words through end of that description, including anything it says about a listing that page may name - no shortening, no rewording. Properties describing that other product are no needs of this item.
- "drop" - "property" holds a requirement the shopper now gives up.
- "free" - per question, the property the shopper said they don't mind about.
- "crowded" - numbers of questions whose reply complains the message covered too many things or wants them one at a time. Not replies merely saying they didn't follow or want simpler words.
- "budget" - new total limit as a number, or null.

Check:
- Folksy/roundabout wording naming a value still states it.
- Refusing to bargain over a value (won't trade, give up, change it) = that value wanted. "not X" only when a reply rejects X itself.

Answer with: ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true/false}], "drop": [{"item": item number, "property": "..."}], "free": [{"item": item number, "property": "..."}], "crowded": [question numbers], "budget": number or null}"""
NOTE = """Task: a store message surfaced mid-task (author: shopper or store). Classify it and pull out changes.

Answer with: ONE JSON object and nothing else -
- "summary": "one short sentence"
- "gone": [listing ids declared unavailable]
- "sold_out": true if a picked listing is declared gone, sold out or unbuyable - even unnamed
- "price_moved": true if some price shifted
- "drop": [{"item": item number, "need": "abandoned requirement"}]
- "add": [{"item": item number, "need": "property: constraint"}]
- "budget": fresh overall cap, numeric, request's unit; else null
- "kind": "change" (task altered: budget, demands, a price, availability, order refused) | "error" (merely flags a malformed or rejected call) | "other" (everything else)

Check: capture every change; a swapped requirement -> old under "drop", new under "add"."""
CHECK = """Task: final gate before checkout - from the records shown, is this cart exactly right?

Whole cart:
- each requested item sits on its own line at its requested quantity; no extras;
- no line displayed as out of stock;
- price x quantity summed across lines fits the budget;
- duplicate records of a listing -> latest one rules;
- pick rule cheapest / most / least: scan the other listings shown too; one plainly meeting all demands while beating the chosen line on that rule = cart wrong.

Each demand:
- Passes when the listing's attribute for it carries the value. Accept: any key spelling; one list element; case, singular/plural, plain synonyms; values containing the demanded words ("deep teal" for teal); converted units; edge-inclusive limits. Ranges must sit wholly inside.
- Same-property key rules; a different value there -> fail. Except: item's own name + property as key (a bag's "bag shape") counts even if the bare key ("shape") disagrees.
- Keys for some other property, however close (handle colour is no body colour), prove nothing.
- Attribute missing -> wanted value appearing under another attribute, in title or in description counts; never figures of another kind (weight is no capacity) nor bare figures not saying what they measure.
- Clock keys (base clock, base frequency, boost clock - any base or boost clock/frequency) never establish a processor speed or clock speed demand; nor do GHz figures in title/description when attributes give only such a base/boost value.
- Category label voicing a demand -> satisfied by that category; label must fit the requested item as a whole.
- "Unlike another listing's value" rules: satisfied unless records show that same value.
- Demands conditioned on another requested item (only if / unless / except when it has some value): binding only if that item's cart line meets the condition, per that line's record.

Answer with: ONE JSON object and nothing else:
{"ok": true or false, "problems": ["short, concrete problem", ...], "better": [replacement listing ids for a cart line, best first]}
ok=false solely for faults records plainly show, or demands no record covers."""
READING = """- Out-of-stock listings, and listings breaking the budget, never fit.
- Category label: a requirement it states is met by that category; the label must match the item.
- Key naming the same property decides (any key spelling; one list element suffices; synonyms and unit conversions count; values containing the required words count, "deep teal" for teal; limits inclusive of the edge).
- Item's own name + property as key (a bag's "bag shape") counts even if the bare key ("shape") says otherwise.
- Keys for another property, even closely related (handle colour is no body colour), decide nothing.
- No attribute naming that property -> wanted value under another attribute, in title or in description counts; bare figures not saying what they measure count for nothing.
- Clock keys (base clock, base frequency, boost clock - any base or boost clock/frequency) never establish a processor speed or clock speed requirement; neither do GHz figures in title/description of a listing whose attributes give only such a base/boost value.
- "Unlike another listing's value" rules: met unless the records show that same value.
- Requirement conditioned on another requested item (only if / unless / except when that item has some value): applies only when the listing chosen for that item meets the condition."""
AUDIT = """Task: the shopping agent plans to end without a purchase, believing nothing fits. Test the listing records shown against the requirements.

Reading rules:
""" + READING + """
- ROW records: seen only as search-result rows, page (attributes) unread -> on any property the row doesn't show they can neither pass nor fail.

Answer with: ONE JSON object and nothing else:
{"fits": {"item number": [listing ids apparently meeting every requirement, cheapest first, max 4]}, "open": [ROW listing ids of a requested kind, within budget, whose page could prove a fit, most promising first, max 6], "reason": "one sentence"}"""
UNMIX = """Task: flag which OTHER REQUIREMENTS below merely portray a second listing, going by the shopper's original wording.
Answer with: ONE JSON object and nothing else: {"describe_other": [flagged lines, verbatim; empty if none]}
Why this happens: certain demands parsed from that wording copy their value off a second listing it mentions, and phrases pinning down that listing (its kind, its shop, attributes named after it via "with ...") occasionally slip in as demands on the purchased product itself."""
TRAIL = """Task: checkout pre-check on borrowed values. Some demands listed below carry no value of their own; they copy it off another listing, whose page may in turn forward to yet another listing for that attribute. Ordering counts only once every listing along each such chain was fully opened.

Do: walk each chain hop by hop - begin at the demand's own description, continue through what each page itself says - using the records below. Tags: OPENED = agent opened this page; ROW = page unread, known merely as a search-result row or via its attributes.

Check:
- A record qualifies as a hop only if it matches that hop's whole description: shop or brand, product kind AND each attribute value given (e.g. "a 10 pieces multipack" excludes the same shop's listing with 10 capsules and no multipack).
- Nothing matches everything -> hop not found yet, even if that shop sells other listings of that kind; supply "find" for it: shop name + described values, as a few search words.
- Hops go in chain order; halt at the first hop not found or whose page isn't OPENED - its text is still unknown.

Answer with: ONE JSON object and nothing else:
{"chains": [{"need": demand copied verbatim, "links": [{"id": matching record's listing id, else "", "about": how the demand or preceding page portrays this listing, "find": "if not found: search words (its shop + described values)"}], "done": true when the final hop is an OPENED page stating that attribute's value itself (also when the demand wants something other than or unlike it: that page value is the benchmark), else false, "value": attribute value per that final page; if the demand wants whatever goes with it and the page names its pairing, that paired value; or ""}]}"""
GATE = """Task: before the agent may STAGE, confirm the request's process steps.
For each step owed before it can STAGE, rule from the numbered RECORD log whether it holds for the cart as it now stands.

Answer with: ONE JSON object and nothing else:
{"steps": [{"step": ..., "done": true or false, "lines": [proving RECORD numbers]}], "call": {"tool": "...", "args": {...}} or null, "hold": change a reply demands before continuing, or "", "needs": [{"item": item number, "need": "property: constraint"} per firm requirement that reply states]}

Check:
- Holds only with a RECORD line proving it happened; cart- or order-related steps also need that line later than the newest cart edit, plus a granting reply.
- Asking about product attributes never fulfils anything.
- Still owed -> propose exactly one next store call (a TOOLS entry, with args).
- Shopper contact uses the messaging tool: one brief plain sentence or question on that step only, mentioning no attribute or value it doesn't require.
- Propose nothing once every owed step holds, or if leftover messages/steps can't cover what's owed."""
SWAP = """Task: replacement search. A cart line's listing just sold out; look among the other listing records shown for listings that could take its place.

Reading rules:
""" + READING + """

Answer with: ONE JSON object and nothing else, keys:
- "reason": "one sentence"
- "fits": up to 6 entries {"id": listing id, "price": numeric price exactly as its record shows}, one per listing apparently satisfying every requirement of the sold-out line's item"""
SHAPE = """Task: map a store tool's output; name every part by the exact key it uses there or among the tool's parameters.

Answer with: ONE JSON object and nothing else:
- "rows": key holding the listings list
- "count": overall match-total key (not merely those displayed); null if absent
- "next": key whose value fetches the following page; null if absent
- "next_arg": tool parameter accepting that value; null if absent
- "price": per-row price key; null if absent
- "stock": per-row key telling whether buying is possible right now; null if absent"""

GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5}[ -]?(?:\\d{3,6}|\\d{1,5}[xX]{1,3})'
PART_CODE = '[A-Z]{1,5}\\d{0,3}-?\\d{3,7}'
MODEL_ALSO = '\\bmodel\\s+(' + FIT_CODE + ')\\s+also\\s+(?:takes|fits|accepts|works with|uses|supports)\\s+(?:part\\s+)?(?:series\\s+)?([A-Za-z]{1,5}\\d{1,3}(?:\\s*(?:,|and|or|/)\\s*[A-Za-z]{1,5}\\d{1,3})*)'  # FX
VALUE_ROOM = 12  # a value search starts only with this many steps to spare past the finish (race 16: cap 35, answers near step 15)
VALUE_KEEP = 8  # steps a value search keeps in reserve past the finish to open and buy its find
HUNT_FORMS = 2  # further spellings of an item's number-with-unit values one cheapest hunt searches, besides the value as written
GLUE_MAX = 3  # a unit glued to its number is short ("2tb", "16gb"); longer words and "in" stay apart
LOOK_ALIKES = 2  # keys held back in a single question (race 16: 1-2)
LINK_SEARCHES = 3  # per-item series tried in a decline hunt (race 16: as many as 9)
LINK_ROWS = 3  # rows with a series code an item needs before the codes are treated as a system (race 16: almost all rows)
LINK_FAMILIES = 12  # above this many distinct series they are model numbers rather than a system (race 16: 9)

_G1A_DATED = '[A-Z][a-z]+ (?:19|20)\\d\\d'
_G1A_CODE_RULES = ('\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(?!(?:no|number)\\b)(' + FIT_CODE + ')\\b(?!\\s+also\\b)',  # FX
                   '\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)')
_G1A_OWN_PART = '[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]'
_G1A_FITS_WITH = '\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)'
_G1A_PART_ALSO_TAIL = '\\bpart\\s+' + PART_CODE + '\\s+also\\s+$'
_G1A_FIT_EXCEPT = '(?i)\\b(?:except|excluding|but not)\\b'
_G1A_PART_ALSO_FITS = '\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)'
_G1A_MORE_PART = '\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')'
_G1A_SERIES_DROPS = ('\\b[A-Za-z]{1,5}\\d{1,3}-\\d{2,7}\\b',
                     '(?i)\\b(?:except|excluding|but not|not for|not compatible with|does not (?:fit|take))\\b[^.;]*')


def _codes(text: Any) -> list[str]:
    kept: list[str] = []
    for code in re.findall('\\b' + FIT_CODE + '\\b', str(text or '')):
        if re.fullmatch(_G1A_DATED, code) is not None:
            continue
        kept.append(re.sub('[\\s-]+', ' ', code).strip().lower())
    return kept

def _code_hit(pattern: str, code: str) -> bool:
    wide = re.sub('[\\s-]+', '', pattern)
    plain = re.sub('[\\s-]+', '', code)
    stem = wide.rstrip('x')
    if stem == wide or not stem[-1:].isdigit():
        return wide == plain
    return len(plain) == len(wide) and plain.startswith(stem)

def _g1a_upper_group(pattern: str, text: str) -> str:
    hit = re.search(pattern, text, re.I)
    return '' if hit is None else hit.group(1).upper()

def _g1a_fit_cuts(text: str):
    for hit in re.finditer(_G1A_FITS_WITH, text, re.I):
        start = hit.start()
        if re.search(_G1A_PART_ALSO_TAIL, text[max(0, start - 40):start], re.I) is None:
            yield re.split(_G1A_FIT_EXCEPT, hit.group(1), maxsplit=1)

def _fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    codes = [code for rule in _G1A_CODE_RULES for hit in re.finditer(rule, text, re.I) for code in _codes(hit.group(1))]
    works: list[str] = []
    bars: list[str] = []
    for kept, *barred in _g1a_fit_cuts(text):
        works.extend(_codes(kept))
        for piece in barred:
            bars.extend(_codes(piece))
    also = [(hit.group(1).upper(), _codes(hit.group(2))) for hit in re.finditer(_G1A_PART_ALSO_FITS, text, re.I)]
    fam, named = _series_marks(text)
    return dict(own=_g1a_upper_group(_G1A_OWN_PART, text), codes=codes, works=works, bars=bars, also=also,
                more=_g1a_upper_group(_G1A_MORE_PART, text), fam=fam, named=named)

def _series_marks(text: str) -> tuple[list[str], list[str]]:
    fam = set(map(str.upper, re.findall('\\b([A-Za-z]{1,5}\\d{1,3})-\\d{2,7}\\b', text)))
    rest = text
    for drop in _G1A_SERIES_DROPS:
        rest = re.sub(drop, ' ', rest)
    named = set(map(str.upper, re.findall('\\b[A-Za-z]{1,5}\\d{1,3}\\b', rest)))
    return sorted(fam), sorted(named)
PAIR_MARKS = (': ', ' = ', '=', ' -> ', ' - ')
FIELD_MARKS = (' | ', '\n', ' || ', ' ; ', '; ', '\t', ' // ', ' · ', ' • ', '|', ';', ', ')
FIELD_NAME = "[A-Za-z][A-Za-z0-9_ ./&'()-]{0,38}"
LIST_BULLET = '^\\s*(?:[-*•·]|\\d{1,3}[.)])\\s+'
RECORD_FIELDS = 3
NOTE_FIELDS = 5
FIELD_SHARE = 0.15
PAIR_SHARE = 0.7
BLOCK_NAME = 60
BLOCK_VALUE = 40
TAG_TEXT = 20000
PAIR_KEY = 'name|key|label|attr|prop|field|spec|title|^k$'
ID_FIELD = '(?:^|[_ -])(?:id|idx|sku|pid|ref|asin|ident\\w*)(?:$|[_ -])|^id(?-i:[A-Z0-9])|(?-i:Ident)|(?<![a-z])(?!(?:paid|valid|grid|solid|rigid|fluid|liquid|hybrid|rapid|void|acid|humid|android|said|laid)$)[a-z]*id$'
FAIL_KEY = 'error|fail|(?<!de)fault|trouble(?!shoot)|denied|refus|invalid'
_G1A_WORD_VALUES = {'true': True, 'false': False, 'none': None, 'null': None, '': None}
_G1A_MONEY = '[$€£]?\\s?((?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)'
_G1A_BLOCK_SPLIT = ';\\s+|;$|\\s\\|\\s|\\n'
_G1A_LOOSE_PAIR = '[\'"]?([A-Za-z][\\w ./-]{0,38})[\'"]?\\s*[:=]\\s*[\'"]?([^,\'"}\\]]*)'
_G1A_RULE_LINE = '[\\s|:+=-]+'
_G1A_TAG_STEPS = (('</?(?:b|i|em|strong|span|a|u|small)\\b[^<>]{0,40}>', '', re.I),
                  ('</t[dh]>\\s*<t[dh][^<>]{0,40}>', ': ', re.I),
                  ('\\s*(?:<[^<>]{1,40}>\\s*)+', '\n', 0))
_G1A_COMMA = '(?<!\\d),|,(?!\\d)'
_G1A_TOKEN_TRIES = ('[a-z0-9]{3,}', '[0-9]+[a-z]|[0-9]*[a-z][a-z0-9]+', '[0-9]+')

def _failed(observation: Any) -> bool:
    size = len(observation) if isinstance(observation, dict) else 0
    if size == 0 or size > 3:
        return False
    for key, value in observation.items():
        if value and re.search(FAIL_KEY, str(key), re.I):
            return True
    if size != 1:
        return False
    (only,) = observation.values()
    return isinstance(only, str) and 0 < len(only) <= 40 and ' ' not in only

def _plain(text: str) -> Any:
    bare = text.strip()
    low = bare.lower()
    if low in _G1A_WORD_VALUES:
        return _G1A_WORD_VALUES[low]
    money = re.fullmatch(_G1A_MONEY, bare)
    digits = money.group(1) if money else ''
    if not digits or re.match('0\\d', digits):
        return bare
    number = float(digits.replace(',', ''))
    if '.' in digits or not number.is_integer():
        return number
    return int(number)

def _block_value(text: str) -> Any:
    found = _plain(text)
    if found is None:
        return text.strip() or None
    if isinstance(found, bool):
        return text.strip()
    return found

def _pair(text: str, mark: str) -> tuple[str, str] | None:
    tail = mark.strip()
    if mark in text:
        head, _, rest = text.partition(mark)
    elif text.endswith(tail):
        head, rest = text[:-len(tail)], ''
    else:
        return None
    name = re.sub(LIST_BULLET, '', head).strip()
    if len(name.split()) > 5 or re.fullmatch(FIELD_NAME, name) is None:
        return None
    return name, rest.strip()

def _split_fields(text: str, between: str, mark: str) -> tuple[list[tuple[str, str]], int]:
    parts = [p for p in text.split(between) if p.strip()]
    if not parts or not _pair(parts[0].strip(), mark):
        return [], 0
    lined = between == '\n'
    fields: list[tuple[str, str]] = []
    named = 0
    open_head = False
    for part in parts:
        bare = part.strip()
        if open_head and (part[:1] in ' \t' or re.match(LIST_BULLET, part)):
            name, value = fields[-1]
            fields[-1] = (name, f'{value}\n{bare}' if value else bare)
            continue
        head = bare[:-1].strip()
        open_head = lined and bare.endswith(':') and re.fullmatch(FIELD_NAME, head) is not None
        pair = (head, '') if open_head else _pair(bare, mark)
        if pair:
            fields.append(pair)
            named += 1
        else:
            name, value = fields[-1]
            fields[-1] = (name, value + between + bare)
    return fields, named

def _g1a_json_block(bare: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(bare)
    except (ValueError, RecursionError):
        loose = dict(re.findall(_G1A_LOOSE_PAIR, bare))
        if len(loose) < 2:
            return None
        return {key.strip(): _plain(raw) for key, raw in loose.items()}
    return parsed if isinstance(parsed, dict) else None

def _g1a_eq_block(bare: str) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    last = ''
    for part in filter(str.strip, re.split(_G1A_BLOCK_SPLIT, bare)):
        name, eq, value = part.partition('=')
        key = name.strip()
        if eq and key and len(key) <= BLOCK_NAME and key not in out:
            last = key
            out[key] = _block_value(value)
            continue
        if not last:
            return None
        out[last] = f'{out[last]}; {part.strip()}'
    if len(out) >= 2:
        return out
    short = bool(out) and ':' not in last and len(last) <= 30 and len(bare) <= 90
    return out if short else None

def _g1a_bullet_pairs(bare: str) -> list[tuple[str, str]] | None:
    lines = [line for line in bare.splitlines() if line.strip()]
    if len(lines) < 2 or any(re.match(LIST_BULLET, line) is None for line in lines):
        return None
    pairs: list[tuple[str, str]] = []
    for line in lines:
        pair = _pair(line.strip(), ': ')
        if pair is None:
            return None
        pairs.append(pair)
    return pairs

def _g1a_joined_pairs(bare: str, between: str) -> list[tuple[str, str]] | None:
    if between not in bare or ': ' not in bare:
        return None
    pairs = [_pair(part.strip(), ': ') for part in bare.split(between)]
    if len(pairs) < 2 or None in pairs:
        return None
    if any('.' in name or len(value) > BLOCK_VALUE for name, value in pairs):
        return None
    return pairs

def _attr_block(text: str) -> dict[str, Any] | None:
    bare = text.strip()
    if len(bare) < 5:
        return None
    if bare[0] in '{[' and bare[-1] in '}]':
        return _g1a_json_block(bare)
    if '=' in bare:
        return _g1a_eq_block(bare)
    tries = (lambda: _g1a_bullet_pairs(bare), lambda: _g1a_joined_pairs(bare, ', '), lambda: _g1a_joined_pairs(bare, '; '))
    found = next((pairs for pairs in (attempt() for attempt in tries) if pairs is not None), None)
    if found is None:
        return None
    return {name: _plain(value) for name, value in found}

def _g1a_split_options(text: str, between: str):
    floor = (PAIR_SHARE if between == ', ' else FIELD_SHARE) * sum(1 for p in text.split(between) if p.strip())
    for mark in PAIR_MARKS:
        fields, named = _split_fields(text, between, mark)
        if named >= RECORD_FIELDS and named >= floor and 2 * len({name for name, _ in fields}) > len(fields):
            yield named, fields

def _record_text(text: str) -> dict[str, Any] | None:
    chosen = None
    for between in filter(text.__contains__, FIELD_MARKS):
        options = list(_g1a_split_options(text, between))
        if options:
            chosen = max(options, key=lambda option: option[0])[1]
            break
    if chosen is None:
        return None
    record: dict[str, Any] = {}
    for name, value in chosen:
        if name not in record:
            record[name] = _attr_block(value) or (value.strip() if re.search(ID_FIELD, name, re.I) else _plain(value))
    return record

def _g1a_grid(lines: list[str], between: str) -> list[dict[str, Any]] | None:
    cells = [[c.strip() for c in line.split(between)] for line in lines]
    head = cells[0]
    if len(head) < 3 or any(len(row) != len(head) for row in cells[1:]):
        return None
    if not all(re.fullmatch(FIELD_NAME, c) for c in head):
        return None
    return [dict(zip(head, map(_plain, row))) for row in cells[1:]]

def _table_text(text: str) -> list[dict[str, Any]] | None:
    lines: list[str] = []
    for raw in text.splitlines():
        if raw.strip() and re.fullmatch(_G1A_RULE_LINE, raw) is None:
            lines.append(raw.strip().strip('|').strip())
    if len(lines) < 2:
        return None
    return next((grid for grid in (_g1a_grid(lines, b) for b in ('|', '\t', ';', ',')) if grid is not None), None)

def _g1a_stacked(bare: str) -> list[dict[str, Any]] | None:
    for parts in (re.split('\\n\\s*\\n', bare), bare.splitlines()):
        if len(parts) < 2:
            continue
        good = [row for row in map(_record_text, parts) if row]
        if len(good) >= 2 and 5 * len(good) >= 4 * len(parts) and len({next(iter(row)) for row in good}) == 1:
            return good
    return None

def _records_text(text: str) -> Any:
    bare = text.strip()
    if len(bare) < 12:
        return None
    if bare[0] in '{[':
        try:
            return _restore(json.loads(bare))
        except (ValueError, RecursionError):
            pass
    stacked = _g1a_stacked(bare)
    if stacked is not None:
        return stacked
    record = _record_text(bare)
    found = _table_text(bare) if record is None and '\n' in bare else record
    if found is not None:
        return found
    if len(bare) <= TAG_TEXT and re.search('<[a-zA-Z/][^<>]{0,40}>', bare):
        plain = bare
        for pattern, swap, flags in _G1A_TAG_STEPS:
            plain = re.sub(pattern, swap, plain, flags=flags)
        plain = plain.strip()
        if plain != bare and len(plain) >= 12:
            return _records_text(plain)
    if '\n' in bare:
        spaced = '\n'.join(re.sub('(?:[ ]{2,}|\t)', ': ', line.strip(), count=1) for line in bare.splitlines() if line.strip())
        if spaced != bare:
            return _record_text(spaced)
    return None

def _block_text(text: str) -> Any:
    if len(text) >= 5 and '=' in text and re.search('; | \\| |\\n', text):
        parts = [p.strip() for p in re.split(_G1A_BLOCK_SPLIT, text.strip()) if p.strip()]
        if len(parts) >= 2 and all(re.fullmatch('[^=:]{1,40}=[^=]{0,200}', p) for p in parts):
            return _attr_block(text) or text
    return text

def _g1a_dict_pairs(rows: list[Any]) -> list[tuple[str, Any]] | None:
    keys = list(rows[0])
    shape = set(keys)
    if len(keys) != 2 or any(not isinstance(r, dict) or set(r) != shape for r in rows):
        return None
    if not re.search(PAIR_KEY, str(keys[0]), re.I):
        return None
    out: list[tuple[str, Any]] = []
    for row in rows:
        name, value = row[keys[0]], row[keys[1]]
        if not isinstance(name, str) or not 0 < len(name) <= BLOCK_NAME or isinstance(value, (dict, list)):
            return None
        out.append((name.strip(), value))
    return out

def _g1a_seq_pair_ok(row: Any) -> bool:
    if not isinstance(row, (list, tuple)) or len(row) != 2:
        return False
    return isinstance(row[0], str) and 0 < len(row[0]) <= BLOCK_NAME and not isinstance(row[1], (dict, list))

def _g1a_text_pairs(rows: list[str]) -> list[tuple[str, Any]] | None:
    out: list[tuple[str, Any]] = []
    for row in rows:
        bare = row.strip()
        pair = _pair(bare, ': ') or _pair(bare, '=')
        if not pair or len(pair[1]) > BLOCK_VALUE or len(pair[0].split()) > 3:
            return None
        out.append((pair[0], _plain(pair[1])))
    return out

def _pairs_block(rows: list[Any], in_record: bool) -> dict[str, Any] | None:
    if len(rows) < 2:
        return None
    first = rows[0]
    found = None
    if isinstance(first, dict):
        found = _g1a_dict_pairs(rows)
    elif isinstance(first, (list, tuple)):
        found = [(row[0].strip(), row[1]) for row in rows] if all(map(_g1a_seq_pair_ok, rows)) else None
    elif in_record and all(isinstance(r, str) for r in rows):
        found = _g1a_text_pairs(rows)
    return None if found is None else dict(found)

def _g1a_is_header(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 2 and all(isinstance(c, str) and re.fullmatch(FIELD_NAME, c) for c in value)

def _g1a_fits_header(rows: Any, width: int) -> bool:
    return isinstance(rows, list) and bool(rows) and all(isinstance(r, (list, tuple)) and len(r) == width for r in rows)

def _column_rows(node: dict[str, Any]) -> dict[str, Any]:
    heads = [key for key in node if _g1a_is_header(node[key])]
    match = next(((head, key) for head in heads for key in node if key != head and _g1a_fits_header(node[key], len(node[head]))), None)
    if match is None:
        return node
    head, body = match
    out = {k: v for k, v in node.items() if k != head}
    out[body] = [dict(zip(node[head], row)) for row in node[body]]
    return out

def _g1a_id_number(key: Any, value: Any) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return bool(re.search(ID_FIELD, key, re.I)) and abs(value) >= 100

def _id_texts(record: dict[str, Any]) -> dict[str, Any]:
    if next((True for v in record.values() if isinstance(v, str) and ' ' in v), False) is False:
        return record
    out = dict(record)
    for key, value in record.items():
        if _g1a_id_number(key, value):
            out[key] = str(value)
    return out

def _keyed_rows(node: dict[str, Any]) -> list[dict[str, Any]] | None:
    rows = list(node.values())
    if len(rows) < 2 or any(not isinstance(r, dict) or len(r) < RECORD_FIELDS for r in rows):
        return None
    if any(' ' in str(k) or not re.search('\\d', str(k)) for k in node):
        return None
    shared = set(rows[0])
    for other in rows[1:]:
        shared.intersection_update(other)
    if len(shared) < RECORD_FIELDS:
        return None
    out: list[dict[str, Any]] = []
    for key, row in node.items():
        out.append(row if _holds_id(row) else {'id': str(key), **row})
    return out

def _holds_id(node: dict[str, Any]) -> bool:
    for key in node:
        if re.search(ID_FIELD, str(key), re.I):
            return True
    return False

def _flat_block(node: dict[str, Any]) -> dict[str, Any]:
    if all(not isinstance(v, dict) for v in node.values()):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if not isinstance(value, dict):
            out[key] = value
            continue
        for inner, v in value.items():
            if not isinstance(v, (dict, list)):
                out[f'{key} {inner}'] = v
    return out

def _restore_field(value: Any, depth: int, carded: bool, page: bool=True) -> Any:
    found = _restore(value, depth + 1, carded, page)
    if not (isinstance(value, str) and isinstance(found, dict)) or len(found) >= NOTE_FIELDS or _holds_id(found):
        return found
    return found if any(map(_id_like, found.values())) else value

def _g1a_nested_card(value: Any) -> bool:
    if not isinstance(value, dict) or _holds_id(value):
        return False
    inner = list(value.values())
    if not inner or not isinstance(inner[0], (dict, str, int, float)):
        return False
    return any(isinstance(x, dict) for x in inner) and not any(isinstance(x, list) for x in inner)

def _g1a_restore_map(value: dict[str, Any], depth: int, page: bool) -> Any:
    node = _id_texts(_column_rows(value))
    carded = len(node) >= RECORD_FIELDS or _holds_id(node)
    node = {k: (_keyed_rows(v) if isinstance(v, dict) else None) or v for k, v in node.items()}
    listing = _holds_id(node) or any(map(_id_like, node.values()))
    roomy = len(node) > 2
    out: dict[str, Any] = {}
    for key, item in node.items():
        if roomy and isinstance(item, str):
            out[key] = _block_text(item) if listing else item
        else:
            out[key] = _restore_field(item, depth, carded, page)
    if depth <= 3 and len(node) >= RECORD_FIELDS:
        for key, item in list(out.items()):
            if _g1a_nested_card(item):
                out[key] = _flat_block(item)
    if len(out) != 1:
        return out
    (only,) = out.values()
    if isinstance(only, dict) and len(only) >= RECORD_FIELDS and isinstance(next(iter(value.values())), str):
        return only
    return out

def _restore(value: Any, depth: int=0, in_record: bool=False, page: bool=True) -> Any:
    if depth > 7:
        return value
    if isinstance(value, str):
        found = _records_text(value)
        return value if found is None else found
    if isinstance(value, list):
        return _pairs_block(value, in_record and page) or [_restore_field(v, depth, False, page) for v in value]
    if isinstance(value, dict):
        return _g1a_restore_map(value, depth, page)
    return value

_FIRST_SPELLING: dict[str, str] = {}

def _join_twins(held: str, value: str) -> str:
    have = {part.strip().lower() for part in re.split(_G1A_COMMA, held)}
    extra: list[str] = []
    for piece in map(str.strip, re.split(_G1A_COMMA, value)):
        if not piece or piece.lower() in have:
            continue
        have.add(piece.lower())
        extra.append(piece)
    if not extra:
        return held
    joined = ', '.join([held, *extra] if held.strip() else extra)
    _FIRST_SPELLING[joined] = _FIRST_SPELLING.get(held, held)
    return joined

def _glued(text: str) -> str:
    return re.sub('(\\d)\\s+([a-z]{1,3})(?![a-z0-9])', lambda hit: hit.group(1) + hit.group(2), text.lower())

def _text_words(text: str) -> set[str]:
    low = text.lower()
    words: set[str] = set()
    for form in (low, _glued(low)):
        words.update(re.findall('[a-z0-9]+', form))
    return words

def _value_tokens(value: str) -> list[str]:
    low = _glued(value)
    return next((found for found in (re.findall(rule, low) for rule in _G1A_TOKEN_TRIES) if found), [])

def _value_forms(value: str, floor: bool=False) -> list[str]:
    low = ' '.join(value.lower().split())
    match = re.fullmatch('(\\d+) ?([a-z]+|")', low)
    if match is None:
        return []
    number, unit = match.group(1), match.group(2)
    glue = ' ' in low and unit.isalpha() and len(unit) <= GLUE_MAX and unit != 'in'
    return [form for form, wanted in ((number + unit, glue), (number, floor and len(number) >= 2)) if wanted]

def _g1a_merge_twins(node: dict[str, Any], depth: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    owner: dict[str, Any] = {}
    for key, value in node.items():
        name = _words(key)
        first = owner.get(name)
        if first is None:
            owner[name] = key
        else:
            held = out[first]
            if held == value:
                continue
            if isinstance(held, str) and isinstance(value, str) and re.search(ID_FIELD, name, re.I) is None:
                out[first] = _join_twins(held, value)
                continue
        out[key] = _fold_twins(value, depth + 1)
    return out

def _fold_twins(node: Any, depth: int=0) -> Any:
    if depth > 5:
        return node
    if isinstance(node, list):
        return [_fold_twins(v, depth + 1) for v in node]
    if not isinstance(node, dict):
        return node
    merged = _g1a_merge_twins(node, depth)
    if len(merged) < 2 and len(node) >= 2:
        return {k: _fold_twins(v, depth + 1) for k, v in node.items()}
    return merged

def _dump(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        text = str(value)
    return text

def _clip(value: Any, limit: int, depth: int=0) -> Any:
    # iterative copy: each pending entry says where its clipped copy lands
    holder: list[Any] = [None]
    pending = [(value, depth, holder, 0)]
    while pending:
        node, level, into, slot = pending.pop()
        if level > 7:
            into[slot] = '...'
        elif isinstance(node, str):
            into[slot] = node[:limit] + '...' if len(node) > limit else node
        elif isinstance(node, dict):
            copy = dict.fromkeys(node)
            into[slot] = copy
            pending.extend(((child, level + 1, copy, key) for key, child in node.items()))
        elif isinstance(node, list):
            copy = [None] * len(node)
            into[slot] = copy
            pending.extend(((child, level + 1, copy, pos) for pos, child in enumerate(node)))
        else:
            into[slot] = node
    return holder[0]

def _g1b_json_texts(text: str) -> Any:
    yield text
    start, stop = (text.find('{'), text.rfind('}'))
    if 0 <= start < stop:
        yield text[start:stop + 1]

def _json_obj(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', body, re.S)
    body = fenced.group(1) if fenced else body
    for piece in _g1b_json_texts(body):
        try:
            parsed = json.loads(piece)
        except ValueError:
            continue
        return parsed if isinstance(parsed, dict) else None
    return None

_G1B_WORD_STEPS = (('([a-z0-9])([A-Z])', '\\1 \\2'), ('[\\s_.\\-]+', ' '))

def _words(key: str) -> str:
    text = str(key)
    for pattern, repl in _G1B_WORD_STEPS:
        text = re.sub(pattern, repl, text)
    return text.strip().lower()

def _snake(key: str) -> str:
    return '_'.join(_words(key).split(' '))

def _g1b_unplural(word: str, keep_ss: bool=True, keep_digit: bool=True) -> str:
    if len(word) <= 3 or not word.endswith('s'):
        return word
    if not keep_ss and word.endswith('ss'):
        return word
    if not keep_digit and word[-2].isdigit():
        return word
    return word[:-1]

def _stem(text: str) -> frozenset[str]:
    return frozenset(map(_g1b_unplural, _words(text).split()))

_G1B_ROOT_SKIP = frozenset(('and', 'for', 'the', 'with'))

def _g1b_root(word: str) -> str:
    if re.search('(?:ch|sh|ss|x|z)es$', word):
        return word[:-2]
    return _g1b_unplural(word, keep_ss=False)

def _roots(text: str) -> set[str]:
    found = map(_g1b_root, _words(text).split())
    return {w for w in found if len(w) >= 3 and w not in _G1B_ROOT_SKIP}

def _spelled(text: str) -> frozenset[str]:
    return frozenset((re.sub('our$', 'or', _g1b_unplural(w, keep_ss=False)) for w in _words(text).split()))

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not (prop and key):
        return False
    if prop <= key:
        return True
    return len(key) >= 2 and key <= prop

_G1B_TOKEN_SKIP = frozenset(('a', 'an', 'the', 'of', 'and', 'with', 'for'))

def _exact_tokens(text: str) -> frozenset[str]:
    raw = re.findall('[a-z0-9]+(?:[.][0-9]+)?[+]?', str(text).lower())
    return frozenset({_g1b_unplural(w, keep_ss=False, keep_digit=False) for w in raw} - _G1B_TOKEN_SKIP)

def _plain_value(value: str) -> str:
    text = str(value)
    mark = re.search(COND_MARK, text, re.I)
    head = text[:mark.start()] if mark else text
    return head.strip().rstrip(',;').strip()

def _conditional(text: str) -> bool:
    _, _, tail = str(text).partition(':')
    return re.search(COND_MARK, tail, re.I) is not None

def _same_prop(a: str, b: str) -> bool:
    left = _spelled(a)
    if not left:
        return False
    right = _spelled(b)
    if left == right:
        return True
    return min(len(left), len(right)) >= 2 and (left <= right or right <= left)

def _kinds(schema: dict[str, Any]) -> list[str]:
    declared = schema.get('type')
    pool = (declared,) if isinstance(declared, str) else declared or ()
    return [k for k in pool if isinstance(k, str)]

def _g1b_is_int(value: Any) -> bool:
    return isinstance(value, int) and (not isinstance(value, bool))

def _g1b_is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and (not isinstance(value, bool))

_G1B_KIND_TESTS = {'string': lambda v: isinstance(v, str), 'integer': _g1b_is_int, 'number': _g1b_is_num, 'boolean': lambda v: isinstance(v, bool), 'array': lambda v: isinstance(v, list), 'object': lambda v: isinstance(v, dict), 'null': lambda v: v is None}

def _fits_kind(value: Any, kind: str) -> bool:
    test = _G1B_KIND_TESTS.get(kind)
    return True if test is None else test(value)

_G1B_MISS = object()

def _g1b_whole_float(value: Any, kind: str) -> Any:
    return int(value) if isinstance(value, float) and value.is_integer() else _G1B_MISS

def _g1b_number_text(value: Any, kind: str) -> Any:
    if not (isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value)):
        return _G1B_MISS
    number = float(value)
    return int(number) if kind == 'integer' or number.is_integer() else number

def _g1b_number_as_text(value: Any, kind: str) -> Any:
    return str(value) if _g1b_is_num(value) else _G1B_MISS

def _g1b_flag_text(value: Any, kind: str) -> Any:
    if isinstance(value, str) and value.strip().lower() in ('true', 'false'):
        return value.strip().lower() == 'true'
    return _G1B_MISS

def _g1b_wrap_one(value: Any, kind: str) -> Any:
    return _G1B_MISS if isinstance(value, list) or value is None else [value]

_G1B_REPAIRS = {'integer': (_g1b_whole_float, _g1b_number_text), 'number': (_g1b_number_text,), 'string': (_g1b_number_as_text,), 'boolean': (_g1b_flag_text,), 'array': (_g1b_wrap_one,)}

def _g1b_fits_any(value: Any, kinds: list[str]) -> bool:
    return any((_fits_kind(value, k) for k in kinds))

def _g1b_repaired(value: Any, kinds: list[str]) -> Any:
    tries = (fix(value, kind) for kind in kinds for fix in _G1B_REPAIRS.get(kind, ()))
    return next((got for got in tries if got is not _G1B_MISS), value)

def _g1b_enum_stage(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
    options = schema.get('enum')
    if not isinstance(options, list) or value in options:
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        value = next((o for o in options if isinstance(o, str) and o.lower() == low), value)
    if value not in options:
        faults.append(f'{where}: allowed values are {options}')
    return value

def _g1b_text_stage(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
    if not isinstance(value, str):
        return value
    longest, shortest = (schema.get('maxLength'), schema.get('minLength'))
    if isinstance(longest, int) and len(value) > longest:
        value = value[:longest]
    if isinstance(shortest, int) and len(value) < shortest:
        faults.append(f'{where} is too short')
    return value

def _g1b_number_stage(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
    if not _g1b_is_num(value):
        return value
    low, high = (schema.get('minimum'), schema.get('maximum'))
    if isinstance(low, (int, float)) and value < low:
        faults.append(f"{where} is below {schema['minimum']}")
    if isinstance(high, (int, float)) and value > high:
        faults.append(f"{where} is above {schema['maximum']}")
    return value

def _g1b_list_stage(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
    if not isinstance(value, list):
        return value
    most, least, inner = (schema.get('maxItems'), schema.get('minItems'), schema.get('items'))
    if isinstance(most, int) and len(value) > most:
        faults.append(f"{where} allows at most {schema['maxItems']} entries")
    if isinstance(least, int) and len(value) < least:
        faults.append(f"{where} needs at least {schema['minItems']} entries")
    if not isinstance(inner, dict):
        return value
    shaped = []
    for pos, entry in enumerate(value):
        shaped.append(_shape(entry, inner, f'{where}[{pos}]', faults))
    return shaped

def _g1b_dict_stage(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
    if not isinstance(value, dict):
        return value
    props = schema['properties'] if isinstance(schema.get('properties'), dict) else {}
    if schema.get('additionalProperties') is False:
        faults.extend((f'{where}.{name} is not an allowed argument' for name in [n for n in value if n not in props]))
    faults.extend((f'{where}.{name} is required' for name in schema.get('required') or [] if name not in value))
    shaped = {}
    for key, entry in value.items():
        shaped[key] = _shape(entry, props[key], f'{where}.{key}', faults) if key in props else entry
    return shaped

_G1B_STAGES = (_g1b_enum_stage, _g1b_text_stage, _g1b_number_stage, _g1b_list_stage, _g1b_dict_stage)

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if isinstance(schema, dict):
        kinds = _kinds(schema)
        wrong = bool(kinds) and not _g1b_fits_any(value, kinds)
        if wrong:
            value = _g1b_repaired(value, kinds)
            wrong = not _g1b_fits_any(value, kinds)
        if wrong:
            faults.append(f"{where} should be {'/'.join(kinds)}")
        else:
            for stage in _G1B_STAGES:
                value = stage(value, schema, where, faults)
    return value

def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, (str, dict)) and value else []

def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        whole = isinstance(value, int) or value.is_integer()
        return int(value) if whole and isinstance(value, float) else value if whole else None
    matched = isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value)
    return int(float(value)) if matched else None

def _id_like(value: Any, named: bool=False) -> bool:
    bare = value.strip() if isinstance(value, str) else None
    if bare is None or ' ' in bare or not 3 <= len(bare) <= 64:
        return False
    return named or re.search('\\d', value) is not None

def _g1b_id_scores(rows: list[dict[str, Any]]) -> Any:
    for key in rows[0]:
        named = bool(re.search(ID_FIELD, str(key), re.I))
        raw = [row.get(key) for row in rows]
        if any((not _id_like(v, named) for v in raw)):
            continue
        clean = [v.strip() for v in raw]
        if len(set(clean)) != len(clean):
            continue
        yield ((named, sum(map(len, clean)) / len(clean)), key)

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    top = max(_g1b_id_scores(rows), key=lambda pair: pair[0], default=None)
    return None if top is None else top[1]

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    queue = [(node, depth)]
    while queue:
        here, level = queue.pop()
        if level > 5:
            continue
        if isinstance(here, dict):
            queue.extend(((v, level + 1) for v in here.values()))
            continue
        if not isinstance(here, list):
            continue
        for entry in here:
            if isinstance(entry, dict) and ident in [str(x).strip() for x in entry.values() if isinstance(x, str)]:
                return True
        queue.extend(((v, level + 1) for v in here))
    return False

def _llm_within(seconds: float) -> ProxyClient:
    if seconds >= 50:
        return _llm
    limit = max(5, int(seconds) // 5 * 5)
    client = _llm_short.get(limit)
    if client is None:
        client = _llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return client

def _g1b_scalar_sets(node: dict[str, Any]) -> Any:
    for value in node.values():
        if not isinstance(value, dict) or len(value) < 2:
            continue
        scalars = {k: v for k, v in value.items() if _g1b_is_num(v) or isinstance(v, str)}
        if len(scalars) >= 2:
            yield scalars

def _fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    return max(_g1b_scalar_sets(node), key=len, default=None)

def _g1b_crash_note(where: str, exc: BaseException) -> dict[str, Any]:
    return {'kind': 'soft_crash', 'where': where, 'error': f'{type(exc).__name__}: {exc}'[:300]}

def _soft(fallback: Any) -> Any:
    produce = fallback if callable(fallback) else lambda: fallback

    def wrap(method: Any) -> Any:

        def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:
                self.trace.append(_g1b_crash_note(method.__name__, exc))
                return produce()
        return guarded
    return wrap

class _Stamped(list):

    def __init__(self, clock: Any):
        list.__init__(self)
        self.clock = clock

    def append(self, entry: dict[str, Any]) -> None:
        stamp = round(self.clock(), 1)
        entry.setdefault('at', stamp)
        list.append(self, entry)

def _filler(spec: dict[str, Any]) -> Any:
    choices = spec.get('enum')
    if isinstance(choices, list) and choices:
        return choices[0]
    preset = next((w for w in ('const', 'default') if w in spec), None)
    if preset is not None:
        return spec[preset]
    kinds = set(_kinds(spec))
    if kinds & {'integer', 'number'}:
        low = spec.get('minimum')
        return low if isinstance(low, (int, float)) else 1
    if 'boolean' in kinds:
        return False
    if 'string' not in kinds:
        return None
    return 'question'[:spec.get('maxLength') or 8]

def _strings(node: Any) -> list[str]:
    found: list[str] = []
    todo = [node]
    while todo:
        here = todo.pop()
        if isinstance(here, dict):
            todo.extend(reversed(list(here.values())))
        elif isinstance(here, list):
            todo.extend(reversed(here))
        elif isinstance(here, str):
            found.append(here)
    return found

def _says(node: Any, text: str) -> bool:
    todo = [node]
    while todo:
        here = todo.pop()
        if isinstance(here, dict):
            todo.extend(here.values())
        elif isinstance(here, list):
            todo.extend(here)
        elif here == text:
            return True
    return False

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name, self.shelf, self.qty, self.musts = (name, shelf, qty, musts)
        self.told, self.dropped, self.free = ([], [], [])

class Cairn:

    _g2_STATE = (
        ('broke', False), ('no_credit', False), ('waited', 0), ('misses', 0), ('cost', 0.0), ('calls', 0),
        ('loop_calls', 0), ('brief', dict), ('roles', dict), ('items', list), ('budget', None), ('msg_limit', 0),
        ('msg_used', 0), ('process', list), ('reserve', 0), ('gate_msgs', 0), ('gate_ok', set), ('done_log', list),
        ('done_steps', list), ('asked', False), ('qa', list), ('rows', dict), ('full', dict), ('seen_order', list),
        ('heard', set), ('notes', list), ('version', 0), ('cart_touched', False), ('cart_at', -1),
        ('cart_seen_at', -1), ('cart_view', None), ('note_named', False), ('cart_lines', dict), ('checked', set),
        ('audited', False), ('offer_seen', dict), ('gone', dict), ('unavailable', set), ('sold_lines', set),
        ('note_sold', False), ('held', 0), ('id_args', set), ('cart_said', ''), ('refused_at', -1), ('leads', list),
        ('refuse_blocks', 0), ('pushed', False), ('built', list), ('dropped', set), ('late_held', False),
        ('lead_items', dict), ('verdicts', dict), ('confirmed', None), ('completed', False), ('link_tried', set),
        ('lane_misses', dict), ('adopted', dict), ('capped', set), ('factless', set), ('trail_room', None),
        ('id_key', None), ('facts_from', dict), ('queued', list), ('buys', 0), ('refusals', 0), ('recent', list),
        ('chat', list), ('opened', set), ('views', dict), ('fit_cached', None), ('owners_seen', dict), ('refs', list),
        ('ref_values', dict), ('anchor_done', False), ('trails', 0), ('trail_blocks', 0), ('compare_ok', 0),
        ('compare_bad', 0), ('tool_keys', dict), ('view_keys', dict), ('read_cache', dict), ('pages_read', 0),
        ('shapes', dict), ('price_key', None), ('stock_key', None), ('cheaper_marks', set), ('wider_held', False),
        ('paged_obs', list), ('plan_order', list), ('lanes', dict), ('lane_found', dict), ('lane_spent', dict),
        ('lane_used', dict), ('late_asked', False), ('one_product', set), ('list_asks', 0), ('apart', dict),
        ('follow_up', list), ('listing', not ASK_SINGLES), ('line_holds', 0), ('bound', list), ('fit_holds', 0),
        ('cheap_holds', 0), ('override_held', set), ('cheap_marks', set), ('searched', list), ('link_searches', 0),
        ('link_seen', dict), ('hunts', 0), ('env_spent', 0.0), ('env_count', 0), ('read_calls', dict),
        ('read_args', dict), ('value_rows', set), ('row_keys', set), ('refuse_trailed', False), ('env_at', 0.0),
        ('hunt_stem', dict), ('hunt_no_noun', set), ('bare_pages', dict), ('code_tries', list), ('codes_dead', False),
        ('hear_reads', 0), ('fit_hunted', False), ('hunt_mark', ''), ('hunt_tried', dict), ('page_args', dict),
        ('page_queries', dict), ('slow_models', set), ('last_model', ''), ('light_idle', 0), ('swapped_now', False),
        ('unstated_holds', 0), ('car_holds', 0), ('car_cache', None), ('salvaged', False), ('searches_seen', dict),
    )

    def __init__(self, problem: dict[str, Any]):
        env = problem['environment']
        self.binding = dict(env['binding'])
        self.view = view = env['policy_view']
        self.request = str(view.get('query') or '')
        self.tools = []
        self.schemas = {}
        for entry in view.get('tools') or []:
            spec = entry.get('function') if isinstance(entry, dict) else None
            if not isinstance(spec, dict) or not isinstance(spec.get('name'), str):
                continue
            self.tools.append(entry)
            params = spec.get('parameters')
            self.schemas[spec['name']] = params if isinstance(params, dict) else {'type': 'object', 'properties': {}}
        self.max_steps = self._g2_step_cap(view.get('max_steps'))
        self.started = time.monotonic()
        self.step = 0
        self.finished = bool(view.get('finished'))
        for attr, seed in self._g2_STATE:
            setattr(self, attr, seed() if isinstance(seed, type) else seed)
        self.trace = _Stamped(self.clock)
        self.env_gap = ENV_GAP
        self.call_cap = ALL_CALLS  # SP3

    @staticmethod
    def _g2_step_cap(raw: Any) -> int:
        cap = _count(raw)
        if cap is None and isinstance(raw, float):
            cap = int(raw)
        if not cap or cap <= 0:
            return 30
        return cap

    def left(self) -> int:
        remaining = self.max_steps - self.step
        return remaining

    def clock(self) -> float:
        now = time.monotonic()
        return now - self.started

    def model_room(self) -> float:
        spent = self.clock()
        return TASK_SECONDS - TIME_MARGIN - spent

    def rushed(self) -> bool:
        pace = 1.0
        if self.env_count:
            pace = max(self.env_spent / self.env_count, 1.0)
        return self.model_room() <= pace * 2

    def msgs_free(self) -> int:
        owed = self.reserve - self.gate_msgs
        if owed < 0:
            owed = 0
        return self.msg_limit - self.msg_used - owed

    def cart_mark(self) -> str:
        lines = sorted(self.cart_lines.items())
        return _dump(lines)

    def row_ids(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if all(self.roles.get(job) != name for job in ('open', 'offer', 'compare', 'cart')) or not self.rows:
            return args
        title = self.title_key()
        if not title:
            return args
        props = (self.schemas.get(name) or {}).get('properties') or {}
        fixed: dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, list):
                fixed[key] = [self._g2_row_match(part, key, props, title) for part in value]
            else:
                fixed[key] = self._g2_row_match(value, key, props, title)
        return fixed

    def _g2_row_match(self, value: Any, key: str, props: dict[str, Any], title: str) -> Any:
        if not isinstance(value, str):
            return value
        bare = value.strip()
        if bare in self.rows or not 3 <= len(bare) <= 30:
            return value
        if not re.search('\\d', value) or 'enum' in (props.get(key) or {}):
            return value
        pattern = '(?<![a-z0-9])' + re.escape(bare.lower()) + '(?![a-z0-9])'
        hits: list[str] = []
        for ident, row in self.rows.items():
            if not re.search(pattern, str(row.get(title) or '').lower()):
                continue
            hits.append(ident)
            if len(hits) > 1:
                return value
        if not hits:
            return value
        self.trace.append({'kind': 'row_id', 'sent': value, 'id': hits[0]})
        return hits[0]

    def page_tools(self) -> set[str]:
        picks = (self.roles.get('offer') or self.roles.get('open'), self.roles.get('compare'))
        return set(filter(None, picks))

    def facts_tool(self) -> str | None:
        for tool, ok in self.facts_from.items():
            if ok and self.single_target(tool):
                if tool:
                    return tool
                break
        opener = self.roles.get('open')
        if self.facts_from.get(opener or '') is False or not self.single_target(opener):
            return None
        return opener

    def compare_facts(self) -> bool | None:
        tool = self.roles.get('compare') or ''
        return self.facts_from.get(tool)

    def compare_bare(self) -> bool:
        seen = self.compare_facts()
        if seen is False:
            return True
        if seen is not None:
            return False
        return self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))

    def gate_due(self, stage: str) -> bool:
        if not any(entry['when'] == stage for entry in self.process):
            return False
        return (stage, f'{self.cart_mark()}#{self.version}') not in self.gate_ok

    def tail(self) -> int:
        for entry in self.process:
            if entry['when'] == 'order' and entry['messages'] > 0:
                break
        else:
            return 1
        if self.msg_used < self.msg_limit and self.gate_due('order'):
            return 2
        return 1

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        cost = self.cost
        if cost <= limit and cost + HEAVY_ROOM <= SPEND_HARD:
            picks = [HEAVY, LIGHT]
        elif cost <= SPEND_HARD:
            picks = [LIGHT]
        else:
            picks = []
        return picks

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None, cap: float | None=None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= self.call_cap:  # SP3
                return None
            if self.model_room() < CALL_FLOOR:
                self.trace.append({'kind': 'late_call_skipped', 'at': round(self.clock(), 1)})
                return None
            self.calls += 1
            limit = cap if cap and model == LIGHT else None  # LM
            message, status = self._g2_attempts(model, messages, max_tokens, tools, limit)
            if message is not None:
                self.last_model = model  # CL
                return message
            if status == 402:
                self.no_credit = True
                return None
        return None

    def _g2_attempts(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, wait: float | None) -> tuple[dict[str, Any] | None, Any]:
        tries = 0
        while True:
            message, status = self.call_model(model, messages, max_tokens, tools, wait)
            if message is not None:
                return message, status
            if wait and status in (None, 0, 408):  # LM: slow or empty loop turn -> next model now
                self.slow_models.add(model)
                self.trace.append({'kind': 'loop_model_slow', 'model': model, 'status': status, 'at': round(self.clock(), 1)})
                return None, status
            pause = self._g2_backoff(status, tries)
            if pause is None:
                return None, status
            self.waited += WAITS[tries]
            tries += 1
            time.sleep(pause)

    def _g2_backoff(self, status: Any, tries: int) -> float | None:
        retryable = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
        if not retryable:
            return None
        if tries >= (1 if status in (None, 408) else len(WAITS)):
            return None
        base = WAITS[tries]
        if self.waited + base > WAIT_TOTAL or self.model_room() - base * 1.2 < CALL_FLOOR:
            return None
        return base * (0.8 + 0.4 * random.random())

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, wait: float | None=None) -> tuple[dict[str, Any] | None, Any]:
        body = self._g2_body(model, messages, max_tokens, tools)
        room = self.model_room()
        if wait:
            room = min(room, wait)
        return self._g2_reply(model, _llm_within(room).post_verbose('/inference/chat/completions', body), tools)

    @staticmethod
    def _g2_body(model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            sent = list(map(dict, messages))
            for entry in (sent[0], sent[-1]):
                text = entry.get('content')
                if isinstance(text, str) and text:
                    entry['content'] = [{'type': 'text', 'text': text, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body.update(tools=tools, tool_choice='auto')
        return body

    def _g2_reply(self, model: str, result: Any, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            return self._g2_choice(model, data, tools)
        error = result.error or {}
        status = error.get('status')
        failure = data.get('error') if isinstance(data, dict) else None
        if isinstance(failure, dict):
            status = _count(failure.get('code'))
            error = {'body': failure.get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def _g2_choice(self, model: str, data: dict[str, Any], tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        spent = (data.get('usage') or {}).get('cost')
        if isinstance(spent, (int, float)):
            self.cost += float(spent)
        first = data['choices'][0]
        message = first.get('message') or {}
        if message.get('content') or message.get('tool_calls'):
            return (message, 200)
        if tools and first.get('finish_reason') == 'length':  # LM
            self.slow_models.add(model)
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
        return (None, 0)

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        pending = iter(models)
        model = next(pending, None)
        while model is not None:
            reply = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens) or {}
            parsed = _json_obj(reply.get('content'))
            if parsed is not None:
                return parsed
            if self.broke or self.no_credit or self.calls >= self.call_cap:  # SP3
                break
            model = next(pending, None)
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = {**self.binding, 'step': self.step + 1, 'action': {'name': name, 'args': args}}
        self._g2_space_out()
        attempt = 0
        while attempt < 4:
            attempt += 1
            result = self._g2_send(payload)
            if isinstance(result.data, dict):
                return self._g2_absorb(name, args, result.data)
            if not self._g2_env_retry(result, payload):
                return None
        return None

    def _g2_space_out(self) -> None:
        gap = self.env_at + self.env_gap - time.monotonic()
        if gap > 0:
            time.sleep(gap)
        self.env_at = time.monotonic()

    def _g2_send(self, payload: dict[str, Any]) -> Any:
        began = time.monotonic()
        result = _env.post_verbose('/environment/call', payload)
        self.env_spent += time.monotonic() - began
        self.env_count += 1
        return result

    def _g2_env_retry(self, result: Any, payload: dict[str, Any]) -> bool:
        error = result.error or {}
        status = error.get('status')
        body = str(error.get('body') or '')
        self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
        if status == 409:
            return self._g2_resync(body, payload)
        if status in (400, 401, 402, 403, 404, 422):
            if status == 402:
                self.broke = True
            return False
        return True

    def _g2_resync(self, body: str, payload: dict[str, Any]) -> bool:
        low = body.lower()
        if any(word in low for word in ('finish', 'quarantin', 'finaliz')):
            self.finished = True
            return False
        found = re.search('expected (\\d+)', body)
        if found:
            self.step = int(found.group(1)) - 1
        elif 'reuse' in low or 'different action' in low:
            self.step += 1
        else:
            return False
        payload['step'] = self.step + 1
        return True

    def _g2_absorb(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        stamped = data.get('step')
        self.step = stamped if isinstance(stamped, int) else self.step + 1
        if data.get('finished'):
            self.finished = True
        if name != self.roles.get('ask'):
            data['observation'] = self.unspelled(name, data.get('observation'))
        seen = data.get('observation')
        self.note_read(name, args, seen)
        broken = _failed(seen)
        if name == self.roles.get('search') and not broken:  # SR
            self.searches_seen[name + _dump(args)] = (self.version, self.step, _dump(_clip(seen, SHOW_CHARS))[:6000])
        if name == self.facts_tool() and isinstance(seen, dict) and not broken:
            key = self.ident(seen)
            if key and not _fact_block(seen):
                self.factless.add((name, key))
        if name in self.page_tools() and not (isinstance(seen, dict) and seen.get('error')) and not broken:
            self._g2_mark_opened(name, args, _dump(seen))
        return data

    def _g2_mark_opened(self, name: str, args: dict[str, Any], shown: str) -> None:
        for value in args.values():
            many = isinstance(value, list)
            for raw in (value if many else [value]):
                if not isinstance(raw, str):
                    continue
                ident = raw.strip()
                if many and ident not in shown:
                    continue
                self.opened.add(ident)
                self.views.setdefault(ident, set()).add(name)

    def unspelled(self, name: str, observation: Any) -> Any:
        views = {self.roles.get(job) for job in ('open', 'offer', 'compare')}
        as_page = name in views or name != self.roles.get('search')
        out = _fold_twins(_restore(observation, page=as_page))
        if name not in views:
            return out
        for ident, row in self.rows.items():
            if ident not in self.opened:
                self.row_keys.update(row)
        known = self.row_keys
        if not known or not isinstance(out, dict):
            return out
        if self.ident(out):
            return self._g2_gather(out, known)
        return {key: [self._g2_gather(part, known) for part in value] if isinstance(value, list) else value for key, value in out.items()}

    def _g2_gather(self, record: Any, known: set[str]) -> Any:
        if not isinstance(record, dict) or _fact_block(record) or not self.ident(record, True):
            return record
        extra: dict[str, Any] = {}
        for key, value in record.items():
            if key in known or isinstance(value, bool) or not isinstance(value, (str, int, float)):
                continue
            extra[key] = value
        if len(extra) < RECORD_FIELDS:
            return record
        kept = {key: value for key, value in record.items() if key not in extra}
        kept['attributes'] = extra
        return kept

    def read_request(self) -> None:
        catalogue = '\n'.join(f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools)
        prompt = BRIEF + '\nTOOLS:\n' + catalogue + '\n\nREQUEST:\n' + self.request
        opening = self.view.get('opening')
        if opening is not None:
            prompt += '\n\nWHAT THE STORE SAID FIRST:\n' + _dump(_clip(self.view.get('opening'), 1500))
        brief = self._g3_fetch_brief(prompt)
        self.brief = brief
        renumber: dict[int, int] = {}
        steps = (self._g3_roles, self._g3_mode_args, self._g3_items, self._g3_fits, self._g3_lookups,
                 lambda b, r: self.unmix(), self._g3_budget, self._g3_limits, self._g3_process)
        for normalise in steps:
            normalise(brief, renumber)

    @staticmethod
    def _g3_complete(brief: dict[str, Any]) -> bool:
        return bool(_items(brief.get('items')) and isinstance(brief.get('roles'), dict) and brief['roles'].get('buy'))

    def _g3_fetch_brief(self, prompt: str) -> dict[str, Any]:
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit:
            if not self._g3_complete(brief):
                brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
            elif self.shared_shelf(brief):
                second = self.ask_json(prompt, [HEAVY], 2500) or {}
                brief = second if self._g3_complete(second) else brief
        while not (brief or self.no_credit or self.calls >= ALL_CALLS or self.clock() >= BRIEF_BY):
            time.sleep(BRIEF_GAP)
            brief = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        if brief:
            self.waited = 0
        return brief

    def _g3_roles(self, brief: dict[str, Any], renumber: dict[int, int]) -> None:
        named = brief.get('roles')
        self.roles = {} if not isinstance(named, dict) else {job: tool for job, tool in named.items() if isinstance(tool, str) and tool in self.schemas}

    def _g3_mode_args(self, brief: dict[str, Any], renumber: dict[int, int]) -> None:
        brief.update({key: {} for key in ('buy_args', 'refuse_args') if not isinstance(brief.get(key), dict)})
        buy = self.roles.get('buy')
        if not buy or buy != self.roles.get('refuse'):
            return
        if brief['buy_args'] and brief['refuse_args'] and brief['buy_args'] != brief['refuse_args']:
            return
        ask = f'One store tool, {buy}, either places the order or closes the episode without buying, and its arguments decide which. Schema: ' + _dump(self.schemas.get(buy)) + '\nReply with nothing but one JSON object, {"buy_args": fixed arguments under which it checks out, "refuse_args": fixed arguments under which it closes with no purchase}, naming only the mode-choosing arguments.'
        fixed = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
        pair = (fixed.get('buy_args'), fixed.get('refuse_args'))
        if all(isinstance(p, dict) for p in pair) and pair[0] != pair[1]:
            brief['buy_args'], brief['refuse_args'] = pair

    @staticmethod
    def _g3_clash(kept: Item, item: Item) -> bool:
        ours = [(_words(a.partition(':')[0]), _words(a.partition(':')[2])) for a in kept.musts]
        theirs = [(_words(b.partition(':')[0]), _words(b.partition(':')[2])) for b in item.musts]
        return any(p == q and v != w for p, v in ours for q, w in theirs)

    def _g3_items(self, brief: dict[str, Any], renumber: dict[int, int]) -> None:
        for entry in filter(lambda e: isinstance(e, dict), _items(brief.get('items'))):
            quantity = _count(entry.get('quantity'))
            self.items.append(Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), quantity if quantity and quantity > 0 else 1, [str(m) for m in _items(entry.get('musts')) if m]))
        merged: list[Item] = []
        for old, item in enumerate(self.items, 1):
            sig = (item.qty, _words(item.shelf)) if item.shelf.strip() else None
            slot = 0 if sig is None else next((k for k, m in enumerate(merged, 1) if (m.qty, _words(m.shelf)) == sig and not self._g3_clash(m, item)), 0)
            if slot:
                twin = merged[slot - 1]
                twin.musts.extend([m for m in item.musts if m not in twin.musts])
                self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
            else:
                merged.append(item)
            renumber[old] = slot or len(merged)
        self.items = merged or [Item('the requested product', '', 1, [])]

    def _g3_fits(self, brief: dict[str, Any], renumber: dict[int, int]) -> None:
        span = range(1, len(self.items) + 1)
        for pair in _items(brief.get('fits')):
            if not isinstance(pair, dict):
                continue
            link = (renumber.get(_count(pair.get('item')) or 0), renumber.get(_count(pair.get('with')) or 0))
            if all(link) and link[0] != link[1] and link[0] in span and link[1] in span and link not in self.bound:
                self.bound.append(link)

    def _g3_anchor(self, text: str) -> tuple[int, str]:
        owners: dict[str, int] = {}
        for n, item in enumerate(self.items, 1):
            for m in item.musts:
                owners.setdefault(m, n)
        if text not in owners:
            key = _words(text.split(':')[0])
            text = next((m for m in owners if _words(m.split(':')[0]) == key), text)
        return owners.get(text, 1), text

    def _g3_lookups(self, brief: dict[str, Any], renumber: dict[int, int]) -> None:
        texts = [str(lookup).strip() for lookup in _items(brief.get('lookups'))]
        for text in texts:
            if not _conditional(text):  # CO2
                ref = self._g3_anchor(text)
                if ref[1] and ref not in self.refs:
                    self.refs.append(ref)
            else:
                self.trace.append({'kind': 'lookup_is_condition', 'text': text[:200]})

    def _g3_budget(self, brief: dict[str, Any], renumber: dict[int, int]) -> None:
        raw = brief.get('budget')
        if isinstance(raw, str):
            raw = float(raw) if re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', raw) else None
        usable = isinstance(raw, (int, float)) and not isinstance(raw, bool)
        if usable and raw > 0:
            self.budget = raw

    def _g3_limits(self, brief: dict[str, Any], renumber: dict[int, int]) -> None:
        limit = _count(brief.get('messages'))
        if limit is None or limit < 0:
            limit = 3 if self.roles.get('ask') else None
        if limit is not None:
            self.msg_limit = limit
        steps = _count(brief.get('steps'))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps

    @staticmethod
    def _g3_step_entry(entry: dict[str, Any]) -> dict[str, Any]:
        said = str(entry.get('when') or '').strip().lower()
        return dict(step=str(entry['step']).strip(), when=next((w for w in ('choose', 'order', 'end') if w == said), 'order'), messages=max(0, _count(entry.get('messages')) or 0))

    def _g3_process(self, brief: dict[str, Any], renumber: dict[int, int]) -> None:
        self.process.extend(self._g3_step_entry(e) for e in _items(brief.get('process')) if isinstance(e, dict) and str(e.get('step') or '').strip())
        if self.roles.get('ask'):
            peak = max((p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']), default=None)
            self.reserve = 0 if peak is None else min(peak, self.msg_limit)

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        met: set[str] = set()
        for entry in _items(brief.get('items')):
            label = _words(str(entry.get('shelf') or '')) if isinstance(entry, dict) else ''
            if label in met:
                return True
            if label:
                met.add(label)
        return False

    def unmix(self) -> None:
        todo = []
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if pointers and rest:
                todo.append((number, item, pointers, rest))
        for number, item, pointers, rest in todo:
            prompt = UNMIX + '\n\nTHE REQUEST:\n' + self.request + f'\n\nPRODUCT BEING BOUGHT: {item.name}\nREQUIREMENTS THAT TAKE THEIR VALUE FROM ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
            named = {str(x).strip() for x in _items(got.get('describe_other'))}
            moved = [m for m in rest if m in named]
            if not moved:
                continue
            lead = pointers[0]
            joined = lead + ' (that listing: ' + '; '.join(moved) + ')'
            item.musts = [joined if m == lead else m for m in item.musts if m not in moved]
            self.refs = [(n, joined) if (n, need) == (number, lead) else (n, need) for n, need in self.refs]
            self.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def described(self, number: int, need: str) -> bool:
        prop, _, value = need.partition(':')
        item = self.items[number - 1]
        vetoes = (
            lambda: need in [m for _, m in self.refs] or need in [str(x) for x in _items(self.brief.get('lookups'))],
            lambda: not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4,
            lambda: bool(re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I)),
            lambda: any({w for w in _stem(other.name) if len(w) >= 4} & _stem(value) for other in self.items if other is not item),
        )
        if any(veto() for veto in vetoes):
            return False
        cat_key = self.category_key()
        blocks = [b for b in (_fact_block(r) or {} for r in self.full.values() if not cat_key or self.owns(item, str(r.get(cat_key) or ''))) if b]
        if len(blocks) < 3:
            return False
        wanted = _stem(prop)
        return not any(_stem(_words(k)) & wanted for b in blocks for k in b)

    def _g3_item_lines(self) -> list[str]:
        out = []
        for number, item in enumerate(self.items, 1):
            told, needs = [], []
            for m in (m for m in item.musts + item.told if m not in item.dropped):
                if self.described(number, m):
                    told.append(m)
                else:
                    needs.append(m if m not in self.ref_values else f'{m} (value read from the page it points at: {self.ref_values[m]})')
            label = f' (category label in the store: {item.shelf})' if item.shelf else ''
            head = (f'item {number}: ' if len(self.items) > 1 else 'item: ') + item.name + label + f', quantity {item.qty}'
            line = head + '; must meet: ' + ('; '.join(needs) or 'nothing beyond being this kind of product')
            if told:
                line += "; worded the shopper's way and matching no attribute any listing carries (not deciding): " + '; '.join(told)
            out.append(line)
        return out

    def _g3_terms_lines(self) -> list[str]:
        brief = self.brief
        out = []
        if self.budget is not None:
            scope = 'per item' if brief.get('budget_scope') == 'each' else 'for the whole order'
            out.append(' '.join(str(x) for x in ('spending limit:', self.budget, brief.get('budget_unit') or '', scope) if x != ''))
        pick = brief.get('pick')
        picks = {'cheapest': 'choose: the lowest-priced listing that meets everything',
                 'most': f"choose: the fitting listing with the most {brief.get('pick_measure') or ''}".rstrip(),
                 'least': f"choose: the fitting listing with the least {brief.get('pick_measure') or ''}".rstrip()}
        out.append(picks.get(pick, 'choose: any listing that meets everything') if isinstance(pick, str) else 'choose: any listing that meets everything')
        if brief.get('in_stock_only', True) is not False:
            out.append('in-stock listings only')
        out += [f'nice to have, breaks ties only: {w}' for w in _items(brief.get('wishes'))]
        out += [f'instruction in the request: {r}' for r in _items(brief.get('rules'))]
        if brief.get('quiet_decline') is True:
            out.append('when nothing fits: end with no listing named or suggested')
        return out

    def _g3_process_lines(self) -> list[str]:
        moments = {'choose': 'before the products are chosen', 'order': 'right before the order (Cairn handles it when you call checkout)', 'end': 'when ending with no purchase (Cairn handles it at that moment)'}
        return [f"step required by the request, {moments[entry['when']]}: {entry['step']}" for entry in self.process]

    def _g3_source_lines(self) -> list[str]:
        facts = self.facts_tool()
        if not facts or not (self.facts_from.get(facts) is True or self.roles.get('offer')):
            return []
        bare = sorted(t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False)
        pages = sorted(t for t in self.page_tools() if t != facts)
        line = f'attributes of a listing are shown by {facts}'
        if bare:
            line += f"; {', '.join(bare)} carry no attributes"
        if pages:
            line += f"; {' or '.join(pages)} show a listing's page (the page meant by a page check or a chain of pages)"
        return [line]

    def ledger(self) -> str:
        sections = (self._g3_item_lines, self._g3_terms_lines, self._g3_process_lines, self._g3_source_lines)
        return '\n'.join('- ' + part for build in sections for part in build())

    def _g3_pace(self, left: int) -> str:
        nudges = (
            (lambda: left <= self.tail() + 1, ' LAST STEPS: with a cart line for every item, check out now; end with no purchase only if your searching proved some item has no fitting listing (a lack of steps proves nothing).'),
            (lambda: left <= 6 or (left <= len(self.items) + 5 and self.buys == 0), ' Only a few steps remain: put the chosen lines in the cart and check out now.'),
            (lambda: self.buys == 0 and left <= 12, ' Not many steps remain: fix your choices, put them in the cart and finish with the checkout soon.'),
            (lambda: self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2, " Half of the steps are gone with nothing in the cart: add each item's fitting listing now (a listing can still sell out)."),
        )
        text = f'[Cairn] {self.step} of {self.max_steps} steps used, {left} remaining.' + next((say for due, say in nudges if due()), '')
        if self.clock() >= WIND_DOWN and left > self.tail() + 1:
            text += ' TIME IS ALMOST GONE: put the chosen lines in the cart and check out within two turns; running out of time scores zero.'
        return text

    def _g3_open_lookups(self) -> str:
        live = {need for _, need in self.needs_now()}
        waiting = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if not waiting or self.buys:
            return ''
        return '[resolve first: open the listing each one points at, then every listing named on that page; the value is on the last page of the chain] ' + ' | '.join(waiting[:3])

    def status(self) -> str:
        head = self._g3_pace(self.left())
        sections = [
            self._g3_open_lookups(),
            '[review candidates] ' + ', '.join(self.leads) if self.leads else '',
            self.fit_plan() if self.bound and self.buys == 0 else '',
            self.car_plan() if self.buys == 0 else '',  # CF3
            '[changes so far] ' + ' | '.join(self.notes[-4:]) if self.notes else '',
        ]
        return head + ''.join('\n' + s for s in sections if s) + '\n[ledger]\n' + self.ledger()

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        key = self.id_key
        if not (key and _id_like(node.get(key), True)):
            key = None
            if in_list or depth == 0 or any(isinstance(v, dict) for v in node.values()):
                found = _learn_id([node])
                if found and ('id' in str(found).lower() or node[found].strip() in self.rows):
                    key = found
        return node[key].strip() if key else None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if 2 * len(rows) < len(node) or not rows:
            return False
        known = self.id_key
        if known and all(_id_like(r.get(known), True) for r in rows):
            return True
        fresh = _learn_id(rows)
        self.id_key = fresh or known
        return bool(fresh)

    def _g3_keep_row(self, ident: str, node: dict[str, Any]) -> None:
        if ident not in self.rows:
            self.seen_order.append(ident)
        old = self.rows.setdefault(ident, {})

        def flipped(before: bool, after: bool) -> bool:
            return any(old.get(k) is before and v is after for k, v in node.items())
        if flipped(True, False):
            self.unavailable.add(ident)
        elif flipped(False, True):
            self.unavailable.discard(ident)
        old.update(node)
        if _fact_block(node):
            self.full[ident] = dict(old)
        elif ident in self.full:
            self.full[ident].update(node)

    def absorb(self, observation: Any) -> None:
        stack: list[tuple[Any, int, bool]] = [(observation, 0, False)]
        while stack:
            node, depth, in_list = stack.pop()
            if depth > 6:
                continue
            if isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    self._g3_keep_row(ident, node)
                else:
                    stack.extend((v, depth + 1, False) for v in reversed(list(node.values())))
            elif isinstance(node, list):
                rows = self.learn_rows(node)
                stack.extend((v, depth + 1, rows) for v in reversed(node))

    def labels(self) -> set[str]:
        pending = [(schema, 0) for schema in self.schemas.values()]
        found: set[str] = set()
        while pending:
            node, depth = pending.pop()
            if depth > 6 or not isinstance(node, dict):
                continue
            found.update(option.strip().lower() for option in node.get('enum') or [] if isinstance(option, str))
            pending.extend((child, depth + 1) for child in node.values() if isinstance(child, dict))
        return found

    def category_key(self) -> str | None:
        spellings: dict[str, set[str]] = {}
        carried: dict[str, int] = {}
        short = ((key, value) for row in self.rows.values() for key, value in row.items()
                 if isinstance(value, str) and 0 < len(value) <= 40 and not re.search('\\d{3}', value))
        for key, value in short:
            spellings.setdefault(key, set()).add(value)
            carried[key] = carried.get(key, 0) + 1
        floor = 0.8 * max(1, len(self.rows))
        known = self.labels()

        def rank(key: str) -> tuple[float, int, int, float]:
            vals = spellings[key]
            share = sum(1 for v in vals if v.strip().lower() in known) / len(vals)
            return (-share, len(vals), -carried[key], -(sum(len(v) for v in vals) / len(vals)))
        return min((key for key in spellings if carried[key] >= floor and len(spellings[key]) <= 40), key=rank, default=None)

    def _g4_wanted(self) -> list[tuple[str, str, set[str]]]:
        return [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]

    def _g4_ours(self, label: str, wanted: list[tuple[str, str, set[str]]]) -> bool:
        plain = _words(label)
        roots = {w for w in _stem(label) if len(w) >= 4}
        return any(plain in (shelf, name) or roots & long for shelf, name, long in wanted) or bool(self.owner_of(label))

    def _g4_fact_groups(self, cat_key: str | None) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for ident, record in ((i, self.full.get(i)) for i in self.seen_order):
            if not record:
                continue
            group = groups.setdefault(str(record.get(cat_key) or 'listings') if cat_key else 'listings', {'n': 0, 'keys': {}})
            group['n'] += 1
            firsts: dict[str, Any] = {}
            for key, value in (_fact_block(record) or {}).items():
                firsts.setdefault(_words(key), value)
            firsts.pop('', None)
            for name, value in firsts.items():
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}, 'plain': {}})
                text = str(value).strip().lower()
                slot['n'] += 1
                slot['per'][ident] = text
                slot['plain'][ident] = _FIRST_SPELLING.get(str(value).strip(), text).strip().lower()
                slot['values'] += [] if text in slot['values'] else [text]
        return groups

    @staticmethod
    def _g4_unfit_key(values: list[str]) -> bool:
        if not values:
            return True
        mean = sum(len(v) for v in values) / len(values)
        if mean > 40 or sum(' / ' in v for v in values) * 2 >= len(values):
            return True
        if len(values) < 4 or mean <= 12 or len(set(values)) != len(values):
            return False

        def terms(text: str) -> set[str]:
            return set(re.findall('[a-z]{3,}', text))
        echoed = sum(1 for v in values if terms(v) & terms(' '.join(o for o in values if o is not v)))
        return echoed * 2 < len(values)

    @staticmethod
    def _g4_fold(into: dict[str, Any], slot: dict[str, Any]) -> None:
        for ident, text in slot['per'].items():
            into['per'].setdefault(ident, text)
            into['plain'].setdefault(ident, slot['plain'][ident])
        into['values'] = list(dict.fromkeys(into['values'] + list(slot['per'].values())))
        into['n'] = len(into['per'])

    def _g4_kept_keys(self, keys: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        kept: list[tuple[str, dict[str, Any]]] = []
        for name, slot in sorted(keys.items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
            if self._g4_unfit_key(list(slot['plain'].values())):
                continue
            twin = next((spec for other, spec in kept if self.twins(other, spec, name, slot)), None)
            if twin is None:
                kept.append((name, slot))
            else:
                self._g4_fold(twin, slot)
        return sorted(kept, key=lambda kv: -kv[1]['n'])

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        groups = self._g4_fact_groups(self.category_key())
        wanted = self._g4_wanted()
        ours = [label for label in groups if self._g4_ours(label, wanted)]
        chosen = [(label, groups[label]) for label in ours] if ours else list(groups.items())
        return [(label, name, slot['n'], [v[:40] for v in slot['values'][:4]])
                for label, group in sorted(chosen, key=lambda pair: -pair[1]['n'])
                for name, slot in self._g4_kept_keys(group['keys'])[:TABLE_KEYS]]

    @staticmethod
    def _g4_other_prop(prop: str, value: str, name: str, shown: set[str]) -> bool:
        if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
            return False
        if not prop.strip() or _words(prop) == name or not _stem(prop) & _stem(name):
            return False
        words = set(re.findall('[a-z0-9]{3,}', value.lower()))
        return bool(words) and not words <= shown

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_of(label)
        if not number:
            return ''
        shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
        split = (must.partition(':') for must in self.items[number - 1].musts)
        clash = next((prop.strip() for prop, _, value in split if self._g4_other_prop(prop, value, name, shown)), '')
        return f' [not the stated {clash}: another property]' if clash else ''

    @staticmethod
    def _g4_agree(x: str, y: str) -> bool:
        if x == y:
            return True
        short, long = sorted((re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower())), key=len)
        return bool(short) and set(short) <= set(long) and len(long) <= 2 * len(short) + 2

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = set(name.split()), set(other.split())
        if not a & b:
            return False
        mine, theirs = spec['plain'], slot['plain']
        both = [i for i in mine if i in theirs and mine[i].strip() and theirs[i].strip()]
        same = sum(Cairn._g4_agree(mine[i], theirs[i]) for i in both)
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        return same * 2 > len(both) if both else bool((set(spec['values']) & set(slot['values'])) - {''})

    def item_groups(self) -> list[list[str]]:
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            groups.setdefault(str(self.rows[ident].get(cat_key) or '') if cat_key else '', []).append(ident)
        wanted = self._g4_wanted()
        pool = [ids for label, ids in groups.items() if self._g4_ours(label, wanted)] or list(groups.values())
        return sorted(pool, key=len, reverse=True)[:max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        if not tool:
            return None
        schema = self.schemas.get(tool) or {}
        required = schema.get('required') or []
        texts = [k for k, v in (schema.get('properties') or {}).items() if isinstance(v, dict) and 'enum' not in v and 'string' in _kinds(v)]
        target = next(filter(texts.__contains__, required), None) or (texts[0] if len(texts) == 1 else None)
        return target if target and all(k == target for k in required) else None

    @staticmethod
    def _g4_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def price_of(self, ident: str) -> float | None:
        if not self.price_key:
            return None
        value = (self.rows.get(ident) or {}).get(self.price_key)
        if isinstance(value, dict):
            value = next(filter(self._g4_number, value.values()), None)
        elif isinstance(value, str):
            figure = re.fullmatch('\\s*[^\\d\\s]{0,3}\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s*[A-Za-z]{0,6}\\.?\\s*', value)
            return float(figure.group(1).replace(',', '')) if figure else None
        return float(value) if self._g4_number(value) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        if isinstance(value, str):
            return not re.search(STOCK_OUT, value.strip(), re.I)
        return value is not False and not (self._g4_number(value) and value <= 0)

    def _g4_listed(self, rows: list[Any]) -> Any:
        for row in rows:
            ident = self.ident(row, True) if isinstance(row, dict) else None
            if ident:
                yield ident, row

    def _g4_nested(self, value: Any) -> Any:
        if isinstance(value, list):
            yield from self._g4_listed(value)
            return
        if not isinstance(value, dict):
            return
        inner = self.ident(value, False, 1)
        if inner:
            yield inner, value
            return
        for part in value.values():
            if isinstance(part, list):
                yield from self._g4_listed(part)

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, list):
            return list(self._g4_listed(observation))
        if not isinstance(observation, dict):
            return []
        ident = self.ident(observation)
        if ident:
            return [(ident, observation)]
        return [pair for value in observation.values() for pair in self._g4_nested(value)]

    def _g4_current_view(self, ident: str) -> set[str]:
        version, have = self.view_keys.get(ident, (-1, set()))
        return have if version == self.version else set()

    def note_view(self, name: str, observation: Any) -> None:
        pairs = self.records_in(observation)
        if pairs:
            self.tool_keys.setdefault(name, set()).update(*(set(record) for _, record in pairs))
        for ident, record in pairs:
            self.view_keys[ident] = (self.version, self._g4_current_view(ident) | set(record))

    def shown_already(self, name: str, ident: str) -> bool:
        keys = self.tool_keys.get(name)
        if not keys or ident in self.gone:
            return False
        return keys <= self._g4_current_view(ident)

    @staticmethod
    def _g4_whole(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def _g4_parse_shape(self, name: str, observation: dict[str, Any], rows: str, got: dict[str, Any]) -> dict[str, Any]:
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get('properties') or {}
        homes = {'count': observation, 'next': observation, 'next_arg': props, 'price': first, 'stock': first}
        shape: dict[str, Any] = {'rows': rows}
        for field, where in homes.items():
            key = got.get(field)
            shape[field] = key if isinstance(key, str) and key in where else None
        if shape['count'] and not self._g4_whole(observation[shape['count']]):
            shape['count'] = None
        return shape

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = (SHAPE + '\n\nStore tool: ' + name + '; its parameters: ' + _dump(self.schemas.get(name) or {})
                  + '\n\nWhat it returned (each list cut to its first rows):\n' + _dump(_clip(sample, 120))[:4000])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        rows = got.get('rows')
        if rows not in lists or not rows:
            return None
        shape = self.shapes[name] = self._g4_parse_shape(name, observation, rows, got)
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.read_shape(name, observation)
        if not shape or not isinstance(observation, dict) or not all(shape[field] for field in ('count', 'next', 'next_arg')):
            return None
        rows = observation.get(shape['rows'])
        count = observation.get(shape['count'])
        token = observation.get(shape['next'])
        usable = isinstance(rows, list) and bool(rows) and self._g4_whole(count) and (token is None or (isinstance(token, (str, int)) and not isinstance(token, bool)))
        return (count, len(rows), shape['next_arg'], token) if usable else None

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        if len(prices) < 3 or None in prices:
            return False
        return all(prices[k] <= prices[k + 1] for k in range(len(prices) - 1))

    @staticmethod
    def _g4_past_first(given: Any) -> bool:
        if isinstance(given, int) and not isinstance(given, bool):
            return given not in (0, 1)
        return bool(str(given or '').strip())

    def _g4_room(self, pages: int) -> bool:
        return self.pages_read + pages <= AUTO_PAGES and self.left() - self.tail() - pages >= 12

    def _g4_page_budget(self, count: int, shown: int) -> int:
        need = -(-(count - shown) // shown)
        if need > 2:
            return 2 if self._g4_room(2) else 0
        return need if self._g4_room(need) else 0

    def _g4_page_walk(self, name: str, args: dict[str, Any], param: str, last: list[Any], pages: int) -> Any:
        """Read up to `pages` further pages, yielding (token, call, observation); last[0] tracks the next token."""
        for _ in range(pages):
            faults: list[str] = []
            call = _shape({**args, param: last[0]}, self.schemas.get(name), 'args', faults)
            if faults:
                return
            reply = self.act(name, call)
            if reply is None or self.finished:
                return
            self.pages_read += 1
            seen = reply.get('observation')
            self.observe(name, seen)
            self.paged_obs.append(seen)
            yield last[0], call, seen
            after = self.paging(name, seen)
            last[0] = after[3] if after else None
            if last[0] is None:
                return

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        if pick not in ('cheapest', 'most', 'least'):
            return ''
        plan = self.paging(name, observation)
        if not plan:
            return ''
        count, shown, param, token = plan
        if token is None or self._g4_past_first(args.get(param)) or count <= shown:
            return ''
        if pick == 'cheapest' and self.price_sorted(observation):
            return ''
        pages = self._g4_page_budget(count, shown)
        if not pages:
            return (f"\n[Cairn] {shown} of the {count} matching listings came back, not sorted by price. Since the pick weighs every listing that fits, "
                    f"either narrow the query to the requirement's value or ask for the next page with {param}={_dump(token)}.")
        last = [token]
        texts: list[str] = []
        for used, call, seen in self._g4_page_walk(name, args, param, last, pages):
            shown_text = self.show(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f'{param}={_dump(used)}: {shown_text}')
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        if not texts:
            return ''
        note = '\n[Cairn fetched the following pages of this search on its own; each page cost one step]\n' + '\n'.join(texts)
        if last[0] is None:
            return note
        return note + f"\n[Cairn] So far {shown * (len(texts) + 1)} rows of the {count} matching listings are read; to cover the remainder, page further or narrow the query to the requirement's value."

    @staticmethod
    def _g4_priced(props: dict[str, Any], key: str) -> bool:
        spec = props.get(key)
        about = str(spec.get('description') or '') if isinstance(spec, dict) else ''
        return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + about, re.I))

    def _g4_caps(self, name: str, args: dict[str, Any], props: dict[str, Any]) -> list[str]:
        limits = self.cap_keys(name)
        lone = len(limits) == 1

        def is_cap(key: str) -> bool:
            if re.search('max|most|under|below|limit|cap|upto|up_to', key, re.I) and self._g4_priced(props, key):
                return True
            return lone and key in limits
        return [k for k, v in args.items() if k in props and self._g4_number(v) and is_cap(k)]

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.schemas.get(name) or {}).get('properties') or {}
        caps = self._g4_caps(name, args, props)
        if not caps or not self.price_key:
            return ''
        key = min(caps, key=args.__getitem__)
        cap = args[key]
        over = sum(1 for i, _ in self.records_in(observation) if (self.price_of(i) or 0) > cap)
        if not over:
            return ''
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        swaps = [f'{k}={_dump(o[0])}' for k, o in others.items() if o]
        hint = f'; to get only cheaper rows, browse with {", ".join(swaps)} while keeping {key} as it is' if swaps else ''
        return f'\n[Cairn] The store ignored {key}={cap:g} on this search: {over} returned rows are priced above it{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        ranked = sorted((price, i) for i in found if self.buyable(i) for price in [self.price_of(i)] if price is not None)
        if len(ranked) < 2:
            return ''
        return '\n[Cairn] cheapest buyable rows seen here, low to high: ' + ', '.join(f'{i} ({p:g})' for p, i in ranked[:6])

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get('pick') != 'cheapest':
            return ''
        if self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        held = '; '.join(f'{i} at {p:g}' for i in self.cart_ids() for p in [self.price_of(i)] if p is not None)
        return (f"Not run yet (no step used). There are {self.left()} steps to go and the pick rule is the cheapest. A search may match on only part of a listing, "
                "so searching a requirement's value catches just some of the fitting listings, and the rows are not sorted by price. Before the checkout, search again "
                "for every item under other names for the product (a plain one-word name, synonyms, words taken from titles of fitting listings) and consider only "
                "buyable rows cheaper than what the cart holds" + (f' ({held})' if held else '')
                + '; open the cheapest of those rows to read their attributes, swap in any that fits, and then check out.')

    def note_facts(self, tool: str, shown: bool) -> None:
        if shown:
            self.facts_from[tool] = True
            return
        if self.facts_from.get(tool) is True:
            return
        bare = self.bare_pages[tool] = self.bare_pages.get(tool, 0) + 1
        if bare >= 2:
            self.facts_from[tool] = False

    def lists_things(self, tool: str) -> bool:
        snake = _snake(self.single_target(tool) or '')
        return any(re.search(pattern, snake, re.I) for pattern in (SHELF_ARG, QUERY_WORDS)) or tool in self.shapes

    def _g5_fetch(self, tool: str, wanted: dict[str, Any]) -> tuple[bool, Any]:
        faults: list[str] = []
        args = _shape(wanted, self.schemas.get(tool), 'args', faults)
        reply = None if faults else self.act(tool, args)
        if reply is None:
            return (False, None)
        seen = reply.get('observation')
        self.observe(tool, seen)
        self.note_view(tool, seen)
        return (True, seen)

    def _g5_may_compare(self, side: Any, many: Any, waiting: int, keep: int) -> bool:
        gates = (lambda: side and many, lambda: self.compare_facts() is True, lambda: waiting >= 2, lambda: self.left() > keep, lambda: not self.finished, lambda: not self.rushed())
        return all(gate() for gate in gates)

    def _g5_compare_pass(self, wanted: list[str], keep: int, texts: list[str]) -> tuple[list[str], list[str]]:
        side = self.roles.get('compare')
        many = self.compare_target()
        unread: list[str] = []
        while self._g5_may_compare(side, many, len(wanted), keep):
            room = self.compare_room()
            batch = wanted[:room]
            wanted = wanted[room:]
            ok, seen = self._g5_fetch(side, {many: batch})
            if not ok:
                return (unread, batch + wanted)
            rows = self.records_in(seen)
            texts.extend(f'{ident}: ' + self.show(record)[:1500] for ident, record in rows)
            if rows:
                self.compare_ok = max(self.compare_ok, len(rows))
                unread.extend([i for i in batch if i not in self.full])
                continue
            self.compare_bad = min(self.compare_bad or 999, len(batch))
            wanted = batch + wanted
            if self.compare_room() >= len(batch):
                break
        return (unread, wanted)

    def _g5_single_pass(self, tool: str, target: str, order: list[str], keep: int, texts: list[str]) -> None:
        stop = lambda: self.left() <= keep or self.finished or self.facts_from.get(tool) is False or self.rushed()
        for ident in order:
            if stop():
                return
            if (tool, ident) in self.factless:
                continue
            ok, seen = self._g5_fetch(tool, {target: ident})
            if not ok:
                return
            if isinstance(seen, dict) and self.ident(seen):
                self.note_facts(tool, bool(_fact_block(seen)))
            texts.append(f'{ident}: ' + self.show(seen)[:1500])

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        direct = bool(tool and target)
        if not direct and self.compare_facts() is not True:
            return []
        texts: list[str] = []
        pending = [i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))]
        unread, pending = self._g5_compare_pass(pending, keep, texts)
        if not direct:
            return texts
        self._g5_single_pass(tool, target, unread + pending, keep, texts)
        if texts:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def _g5_link_code(self, ident: str) -> Any:
        marks = self.fit_index()['marks']
        code = (marks.get(ident) or {}).get('more')
        taken = bool(code) and (code in self.link_tried or any(mark.get('own') == code for mark in marks.values()))
        return None if not code or taken else code

    def show_links(self, ids: list[str], keep: int) -> None:
        plan = self.value_template()
        if not plan or (len(self.items) < 2 and len(ids) < 2):
            return
        tool, base, field = plan
        for ident, code in ((i, self._g5_link_code(i)) for i in ids):
            if code is None:
                continue
            if self.left() <= keep or self.finished or self.rushed():
                return
            self.link_tried.add(code)
            faults: list[str] = []
            args = _shape({**base, field: code}, self.schemas.get(tool), 'args', faults)
            reply = None if faults else self.act(tool, args)
            if reply is None:
                return
            self.observe(tool, reply.get('observation'))
            self.trace.append({'kind': 'show_link', 'line': ident, 'code': code})
            owned = [i for i, mark in self.fit_index()['marks'].items() if mark.get('own') == code]
            self.fill_facts(owned, max(keep, self.tail() + 6))

    def _g5_cap_guess(self, key: str, options: list[Any], props: dict[str, Any], shelf: str, single: bool) -> Any:
        texty = lambda k, v: isinstance(v, str) and v.strip() and re.search(QUERY_WORDS, _snake(k), re.I)
        loose = lambda k: not (isinstance(props.get(k), dict) and 'enum' in props[k])
        plain = lambda a: shelf in a and key in a and not any(k != shelf and loose(k) and texty(k, v) for k, v in a.items())
        recent = next((a[key] for a in reversed(self.searched) if plain(a)), None)
        named = [o for o in options if isinstance(o, str) and re.search(LIST_MODE, o, re.I)]
        worded = {a[key] for a in self.searched if key in a and any(texty(k, v) for k, v in a.items())}
        unworded = [o for o in options if o not in worded]
        if recent is not None:
            return recent
        if len(named) == 1:
            return named[0]
        return unworded[0] if single and worded and len(unworded) == 1 else None

    def _g5_cap_pick(self, key: str, options: list[Any], learned: list[tuple[dict[str, Any], bool]], props: dict[str, Any], shelf: str, single: bool) -> Any:
        pick = next((mode[key] for mode, held in learned if held and key in mode), None)
        if pick is None:
            pick = self._g5_cap_guess(key, options, props, shelf, single)
        spoiled = {mode[key] for mode, held in learned if not held and key in mode} if single else set()
        if not spoiled or (pick is not None and pick not in spoiled):
            return pick
        rest = [o for o in options if o not in spoiled]
        return rest[0] if len(rest) == 1 else None

    def _g5_cap_base(self, tool: str, spec: dict[str, Any], props: dict[str, Any], shelf: str, cap: str) -> tuple[str, dict[str, Any], str, str] | None:
        learned = [(dict(mode), held) for (key, mode), held in self.cap_modes(tool).items() if key == cap]
        menus = {k: (v.get('enum') if isinstance(v, dict) else None) for k, v in props.items()}
        choosers = [k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and len(v['enum']) >= 2]
        single = len(choosers) == 1 and choosers[0] in (spec.get('required') or [])
        base: dict[str, Any] = {}
        for key, options in menus.items():
            if not isinstance(options, list) or len(options) < 2:
                continue
            pick = self._g5_cap_pick(key, options, learned, props, shelf, single)
            if pick is None:
                return None
            base[key] = pick
        return (tool, base, shelf, cap)

    def cap_template(self) -> tuple[str, dict[str, Any], str, str] | None:
        shelf_like = lambda k, v: isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and re.search(SHELF_ARG, _snake(k), re.I)
        for tool, schema in self.schemas.items():
            spec = schema or {}
            props = spec.get('properties') or {}
            caps = self.cap_keys(tool)
            shelves = [k for k, v in props.items() if shelf_like(k, v)]
            if len(caps) == 1 and len(shelves) == 1:
                return self._g5_cap_base(tool, spec, props, shelves[0], caps[0])
        return None

    def cap_keys(self, tool: str) -> list[str]:
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        numeric = lambda k, v: isinstance(v, dict) and {'integer', 'number'} & set(_kinds(v)) and not re.search(CAP_NOT, k, re.I) and not re.search(CAP_LOW, _snake(k), re.I)
        numbers = [k for k, v in props.items() if numeric(k, v)]
        held = {key for (key, _), ok in self.cap_modes(tool).items() if ok}
        named = [k for k in numbers if re.search(CAP_ARG, _snake(k), re.I)]
        priced = [k for k in numbers if re.search(CAP_PRICED, _snake(k), re.I)]
        tiers = ([k for k in numbers if k in held], [k for k in named if k in priced], priced, named)
        return next((tier for tier in tiers if tier), [])

    def _g5_kind(self, ident: str, cat_key: Any) -> str:
        return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ''

    def _g5_price_pool(self, cat_key: Any) -> dict[str, list[float]]:
        pool: dict[str, list[float]] = {}
        for ident in self.rows:
            price = self.price_of(ident)
            if price is not None:
                pool.setdefault(self._g5_kind(ident, cat_key), []).append(price)
        return pool

    def _g5_cap_judge(self, out: dict[Any, bool], props: dict[str, Any], cat_key: Any, prices: dict[str, list[float]], mark: Any, args: dict[str, Any]) -> None:
        rows = [i for i, _ in self.records_in(self.read_calls.get(mark))]
        shown = [self.price_of(i) for i in rows]
        if not rows or None in shown:
            return
        mode = tuple(sorted((k, v) for k, v in args.items() if isinstance(props.get(k), dict) and isinstance(props[k].get('enum'), list) and isinstance(v, (str, int, float))))
        labels = {self._g5_kind(i, cat_key) for i in rows} if cat_key else set()
        pool = [p for label in labels for p in prices.get(label, [])]
        top = max(shown)
        capped = [(k, v) for k, v in args.items() if k in props and not isinstance(v, bool) and isinstance(v, (int, float))]
        for key, value in capped:
            slot = (key, mode)
            if top > value:
                out[slot] = False
                continue
            if out.get(slot) is False or not pool:
                continue
            if (sum(p <= value for p in pool) / len(pool)) ** len(rows) < CAP_CHANCE:
                out[slot] = True

    def cap_modes(self, tool: str) -> dict[tuple[str, tuple[tuple[str, Any], ...]], bool]:
        props = (self.schemas.get(tool) or {}).get('properties') or {}
        cat_key = self.category_key()
        out: dict[tuple[str, tuple[tuple[str, Any], ...]], bool] = {}
        prices = self._g5_price_pool(cat_key)
        for mark, (name, args) in self.read_args.items():
            if name == tool:
                self._g5_cap_judge(out, props, cat_key, prices, mark, args)
        return out

    @staticmethod
    def _g5_under(value: float) -> int:
        return int(value) - 1 if float(value).is_integer() else int(value)

    def _g5_last_page(self, tool: str, args: dict[str, Any], page: Any) -> None:
        if page[3] is None:
            return
        faults: list[str] = []
        call = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
        if faults or call is None or self.left() <= self.tail() + 7 or self.rushed():
            return
        more = self.act(tool, call)
        if more is not None:
            self.observe(tool, more.get('observation'))

    def _g5_walk_step(self, plan: tuple[str, dict[str, Any], str, str], label: str, line: str, price: float, limit: int) -> int | bool | None:
        tool, base, shelf, cap = plan
        faults: list[str] = []
        args = _shape({**base, shelf: label, cap: limit}, self.schemas.get(tool), 'args', faults)
        reply = None if faults else self.act(tool, args)
        if reply is None:
            return False
        seen = reply.get('observation')
        self.observe(tool, seen)
        rows = self.records_in(seen)
        self.trace.append({'kind': 'list_below', 'line': line, 'price': price, 'limit': limit, 'rows': len(rows)})
        page = self.paging(tool, seen)
        over = lambda: any((self.price_of(i) or 0) > limit for i, _ in rows)
        if not rows or over() or not page or page[0] <= page[1] or self.price_sorted(seen):
            return None
        if page[0] <= 2 * page[1]:
            self._g5_last_page(tool, args, page)
            return None
        low = min((p for p in (self.price_of(i) for i, _ in rows) if p is not None), default=None)
        return None if low is None else self._g5_under(low)

    def _g5_walk_line(self, plan: tuple[str, dict[str, Any], str, str], cat_key: str, line: str) -> bool:
        price = self.price_of(line)
        label = str((self.rows.get(line) or {}).get(cat_key) or '')
        if price is None or price < 2 or not label or (line, price) in self.capped:
            return True
        self.capped.add((line, price))
        limit = self._g5_under(price)
        tried: set[int] = set()
        for _ in range(WALK_CALLS):
            if self.left() <= self.tail() + (WALK_ROOM if tried else 7) or self.finished or self.rushed():
                return False
            if limit < 1 or limit in tried:
                return True
            tried.add(limit)
            step = self._g5_walk_step(plan, label, line, price, limit)
            if step is False:
                return False
            if step is None:
                return True
            limit = step
        return True

    def list_below(self, cart: list[str]) -> None:
        plan = self.cap_template()
        cat_key = self.category_key()
        if not plan or not cat_key:
            return
        all(self._g5_walk_line(plan, cat_key, line) for line in cart)

    def _g5_below(self, kind: str, price: float, cart: list[str], cat_key: Any) -> list[str]:
        def asking(i: str) -> float:
            own = self.price_of(i)
            return own if own is not None else float('inf')
        keep = lambda i: i not in cart and i not in self.full and self.buyable(i) and self._g5_kind(i, cat_key) == kind and asking(i) < price
        return list(filter(keep, self.seen_order))

    def _g5_cheaper_for(self, line: str, cart: list[str], cat_key: str, room: int) -> list[str]:
        price = self.price_of(line)
        kind = self._g5_kind(line, cat_key)
        if price is None or not kind:
            return []
        pool = self._g5_below(kind, price, cart, cat_key)
        number = self.owner_of(str((self.rows.get(line) or {}).get(cat_key) or ''))  # CR
        wanted = [r for r in (_roots(v) for v, _ in self.value_words(number)) if r] if number else []

        def shows(i: str) -> int:
            have = _roots(' '.join(str(v) for v in (self.rows.get(i) or {}).values() if isinstance(v, str)))
            return sum(w <= have for w in wanted)
        ranked = sorted(pool, key=lambda i: (-shows(i), -(i in self.value_rows), self.price_of(i) or 0.0))
        return ranked[:room or 3]

    def _g5_cheaper_key(self) -> Any:
        if self.brief.get('pick') != 'cheapest':
            return None
        mark = self.cart_mark() + f'#{self.version}'
        cat_key = self.category_key()
        fresh = bool(cat_key) and mark not in self.cheaper_marks
        if fresh:
            self.cheaper_marks.add(mark)
        return cat_key if fresh else None

    def cheaper_rows(self) -> list[str]:
        cat_key = self._g5_cheaper_key()
        if cat_key is None:
            return []
        cart = self.cart_ids()
        self.list_below(cart)
        room = self.compare_room() if self.compare_facts() is True else 0
        picks: list[str] = []
        for top in (self._g5_cheaper_for(line, cart, cat_key, room) for line in cart):
            picks.extend([i for i in top if i not in picks])
        cap = room * len(cart) if room else min(6, CHEAPER_OPENS * max(1, len(cart)))
        return self.fill_facts(picks[:cap], self.tail() + 6)

    @_soft('')
    def below_cart(self) -> str:
        cat_key = self.category_key()
        cart = self.cart_ids()
        found: list[str] = []
        for line in cart:
            price = self.price_of(line)
            kind = self._g5_kind(line, cat_key)
            if price is None or not kind:
                continue
            cheap = sorted(self._g5_below(kind, price, cart, cat_key), key=lambda i: self.price_of(i) or 0.0)
            found.extend(f'{i} at {self.price_of(i):g}' for i in cheap[:CHEAPER_SHOWN])
        return f"Cheaper rows you have not opened yet, under a cart line's price: {', '.join(found)}." if found else ''

    def title_key(self) -> str | None:
        cat_key = self.category_key()
        texts: dict[str, list[str]] = {}
        for key, value in ((k, v) for row in self.rows.values() for k, v in row.items()):
            if key != cat_key and isinstance(value, str) and len(value) <= TITLE_CHARS:
                texts.setdefault(key, []).append(value)
        total = len(self.rows)
        stats = [(key, len(vals), sum(len(v.split()) for v in vals), sum(map(len, vals)), len(set(vals))) for key, vals in texts.items()]
        kept = [(chars / count, key, distinct) for key, count, words, chars, distinct in stats if count * 5 >= total * 4 and words >= count * 2 and distinct * 2 >= count]
        most = max((distinct for _, _, distinct in kept), default=0)
        ranked = [(mean, key) for mean, key, distinct in kept if distinct * 10 >= most * 9]
        return min(ranked)[1] if ranked else None

    def hunt_words(self, number: int) -> tuple[str, list[str]]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        title = self.title_key()
        label_of = lambda record: str(record.get(cat_key) or '')
        mine = [row for row in self.rows.values() if self.owns(item, label_of(row))] if cat_key else []
        labels = [label_of(row) for row in mine]
        names = [str(row.get(title) or '') for row in mine] if title else []
        if cat_key:
            short = lambda record: (str(v) for v in (_fact_block(record) or {}).values() if isinstance(v, str) and len(v) <= HUNT_VALUE_CHARS)
            names.extend([' '.join(short(record)) for record in self.full.values() if self.owns(item, label_of(record))])
        others: set[str] = set()
        for other in (o for n, o in enumerate(self.items, 1) if n != number):
            others.update(re.findall('[a-z]{4,}', ((other.shelf or '') + ' ' + other.name).lower()))
        label = (max(labels, key=labels.count) if labels else item.shelf or item.name).lower()
        nouns = sorted((w for w in re.findall('[a-z]{4,}', label) if w not in others), key=len, reverse=True)
        noun = nouns[0] if nouns and number not in self.hunt_no_noun else ''
        if not names:
            return (noun, [])
        counts: dict[str, int] = {}
        spoken = (w for name in names for w in dict.fromkeys(x.lower() for x in re.findall("[A-Za-z][A-Za-z'-]{2,}", name)))  # DET
        for word in spoken:
            counts[word] = counts.get(word, 0) + 1
        own = _roots(item.shelf or '') | _roots(item.name)
        ceiling = max(2, int(len(names) * HUNT_COMMON))
        fits = lambda w, n: 2 <= n <= ceiling and w != noun and _roots(w) and not _roots(w) & own
        return (noun, sorted((w for w, n in counts.items() if fits(w, n)), key=lambda w: (-counts[w], w)))

    def rows_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        if not cat_key:
            return len(self.rows)
        return len([row for row in self.rows.values() if self.owns(item, str(row.get(cat_key) or ''))])

    @_soft('')
    def hunt_cheaper(self) -> str:
        mark = _dump([self.needs_now(), self.budget, sorted(self.gone)])
        spent_rules = (lambda: self.brief.get('pick') != 'cheapest', lambda: self.hunts >= HUNT_ROUNDS, lambda: self.hunt_mark == mark)
        if any(rule() for rule in spent_rules):
            return ''
        plan = self.value_template()
        late_rules = (lambda: not plan, lambda: not self.price_key, lambda: self.clock() > HUNT_BY)
        if any(rule() for rule in late_rules):
            return ''
        tool, base, field = plan
        room = min(HUNT_STEPS, self.left() - self.tail() - HUNT_KEEP - 2 * len(self.items))
        if room < 3:
            return ''
        self.hunts += 1
        self.hunt_mark = mark
        cart = self.cart_ids()
        numbers = range(1, len(self.items) + 1)
        share = max(3, room // max(1, len(numbers)))
        tally = [0, 0]
        for number in numbers:
            if tally[0] >= room or self.finished:
                break
            used, gained = self.hunt_item(number, tool, base, field, min(share, room - tally[0]))
            tally = [tally[0] + used, tally[1] + gained]
        spent, fresh = tally
        self.trace.append({'kind': 'hunt', 'steps': spent, 'new_rows': fresh, 'cart': cart})
        if not spent:
            return ''
        self.wider_held = True
        return f'Price hunt done: Cairn used {spent} extra steps on new searches (one step per search), which turned up {fresh} rows not seen before.'

    def hunt_item(self, number: int, tool: str, base: dict[str, Any], field: str, room: int) -> tuple[int, int]:
        hunt: dict[str, Any] = {'tried': self.hunt_tried.setdefault(number, set()), 'number': number, 'tool': tool,
                                'base': base, 'field': field, 'room': room, 'spent': 0, 'gained': 0, 'idle': 0,
                                'issued': False, 'refused': False}
        hunt['noun'], _unused = self.hunt_words(number)
        hunt['values'] = [value for value, _ in self.value_words(number)]
        hunt['forms'] = self.value_forms(number)
        stem_key = (number, tuple(hunt['values']))
        stem = self.hunt_stem.get(stem_key)
        if stem is None:
            stem = self._g6_find_stem(hunt)
            if stem is None:
                return (hunt['spent'], hunt['gained'])
            self.hunt_stem[stem_key] = stem
        noun = hunt['noun']
        tail = [noun] if noun and (not stem or noun in stem.split()) else []
        idle_before = hunt['idle']
        self._g6_sweep(hunt, self._g6_form_queries(hunt, tail), True)
        self._g6_sweep(hunt, self._g6_rest_queries(hunt, tail), False)
        hunt['idle'] = idle_before
        while self._g6_can_widen(hunt):
            _, extra = self.hunt_words(number)
            lead = next((w for w in extra if f'{w} {stem}'.strip().lower() not in hunt['tried'] and w not in stem.lower().split()), None)
            if lead is None:
                break
            self._g6_hunt_round(hunt, f'{lead} {stem}'.strip())
        return (hunt['spent'], hunt['gained'])

    def _g6_find_stem(self, hunt: dict[str, Any]) -> str | None:
        values, noun = hunt['values'], hunt['noun']
        openers: list[str] = []
        if values:
            openers.extend([' '.join(values + [noun]).strip(), ' '.join(values)])
        if len(values) >= 2:
            openers.append(' '.join(values[:-1] + [noun]).strip())
        if noun:
            openers.append(noun)
        for query in dict.fromkeys(openers):
            hunt['issued'] = False
            if self._g6_hunt_round(hunt, query):
                return query
            if not hunt['issued']:
                return None
            if query == hunt['noun']:
                self.hunt_no_noun.add(hunt['number'])
                hunt['noun'] = ''
        return ''

    def _g6_form_queries(self, hunt: dict[str, Any], tail: list[str]) -> list[str]:
        values, forms = hunt['values'], hunt['forms']
        out: list[str] = []
        for i, value in enumerate(values):
            out.extend(' '.join(values[:i] + [form] + values[i + 1:] + tail) for form in forms.get(value, []))
        return out[:HUNT_FORMS]

    def _g6_rest_queries(self, hunt: dict[str, Any], tail: list[str]) -> Any:
        values, forms = hunt['values'], hunt['forms']
        rest = [v for v in values if v not in self.floor_values(hunt['number'])]
        if not rest or len(rest) >= len(values):
            return {}
        first_forms = [(forms.get(v) or [v])[0] for v in rest]
        return dict.fromkeys([' '.join(first_forms + tail), ' '.join(rest + tail)])

    def _g6_sweep(self, hunt: dict[str, Any], queries: Any, whole: bool) -> None:
        for query in queries:
            if hunt['spent'] >= hunt['room'] or hunt['refused'] or self.finished:
                return
            if query.lower() not in hunt['tried']:
                self._g6_hunt_round(hunt, query, whole)

    def _g6_can_widen(self, hunt: dict[str, Any]) -> bool:
        if hunt['spent'] >= hunt['room'] or hunt['idle'] >= HUNT_IDLE or hunt['refused'] or self.finished:
            return False
        return self.clock() <= HUNT_BY + HUNT_GRACE

    def _g6_hunt_round(self, hunt: dict[str, Any], query: str, whole: bool = True) -> bool:
        tool = hunt['tool']
        hunt['tried'].add(query.lower())
        faults: list[str] = []
        args = _shape({**hunt['base'], hunt['field']: query}, self.schemas.get(tool), 'args', faults)
        hunt['refused'] = hunt['refused'] or bool(faults)
        if faults or hunt['spent'] >= hunt['room'] or self.finished:
            return False
        hunt['issued'] = True
        lowered = query.lower()
        forms = hunt['forms']
        covers = whole and bool(hunt['values']) and all(any(f.lower() in lowered for f in [v] + forms.get(v, [])) for v in hunt['values'])
        found, walked = False, set()
        while not self.finished and self.clock() <= HUNT_BY + HUNT_GRACE:
            mark = self.call_mark(tool, args)
            if mark in walked:
                break
            walked.add(mark)
            page = self._g6_hunt_open(hunt, mark, args)
            if page is None:
                break
            seen, new = page
            found = self._g6_hunt_judge(seen, covers) or found
            args = self._g6_hunt_next(tool, args, seen, new)
            if args is None:
                break
        return found

    def _g6_hunt_open(self, hunt: dict[str, Any], mark: str, args: dict[str, Any]) -> tuple[Any, int] | None:
        if mark in self.read_calls:
            return (self.read_calls[mark], HUNT_PAGE_NEW)
        if hunt['spent'] >= hunt['room']:
            return None
        number = hunt['number']
        before = self.rows_of_item(number)
        reply = self.act(hunt['tool'], args)
        hunt['spent'] += 1
        if reply is None:
            return None
        seen = reply.get('observation')
        self.observe(hunt['tool'], seen)
        new = self.rows_of_item(number) - before
        hunt['gained'] += new
        hunt['idle'] = hunt['idle'] + 1 if not new else 0
        return (seen, new)

    def _g6_hunt_judge(self, seen: Any, covers: bool) -> bool:
        rows = self.records_in(seen)
        if covers:
            self.value_rows.update(ident for ident, _ in rows)
        return bool(rows)

    def _g6_hunt_next(self, tool: str, args: dict[str, Any], seen: Any, new: int) -> dict[str, Any] | None:
        page = self.paging(tool, seen)
        if not page or page[3] is None:
            return None
        if new < HUNT_PAGE_NEW and page[0] > HUNT_WHOLE:
            return None
        faults: list[str] = []
        nxt = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
        return None if faults else nxt

    @_soft(list)
    def hunt_fits(self) -> list[str]:
        if self.fit_hunted or self.clock() > HUNT_BY:
            return []
        plan = self.value_template()
        if not plan:
            return []
        tool, base, field = plan
        keep = self.tail() + len(self.items) + 1
        spare = self.left() - keep
        if spare < 3:
            return []
        self.fit_hunted = True
        cat_key = self.category_key()
        numbers = self._g6_fit_numbers(cat_key)
        wanted = {n: {w for value, _ in self.value_words(n) for w in _value_tokens(value)} for n in numbers}
        paged = self.unread_pages(tool, wanted, min(FIT_PAGES, spare // 2))
        room = min(HUNT_STEPS, self.left() - self.tail() - HUNT_KEEP)
        spent = self._g6_fit_searches(tool, base, field, numbers, room) if room >= 3 else 0
        if spent is None:
            return []
        picks = [ident for n in numbers for ident in self._g6_fit_pool(n, cat_key, wanted.get(n) or set())[:HUNT_FIT_OPENS]]
        read: list[str] = []
        for start in range(0, len(picks), HUNT_FIT_OPENS):
            if self.clock() > HUNT_BY + HUNT_GRACE:
                break
            read += self.fill_facts(picks[start:start + HUNT_FIT_OPENS], keep)
        self.trace.append({'kind': 'hunt_fits', 'pages': paged, 'searches': spent, 'opened': len(read)})
        return read

    def _g6_fit_numbers(self, cat_key: Any) -> list[int]:
        held: list[str] = []
        if cat_key:
            for ident in self.cart_ids():
                if (self.held_units(ident) or 0) > 0:
                    held.append(str((self.rows.get(ident) or {}).get(cat_key) or ''))
        open_items = [n for n, item in enumerate(self.items, 1) if all(not self.owns(item, label) for label in held)]
        return open_items or list(range(1, len(self.items) + 1))

    def _g6_fit_tries(self, number: int) -> list[str]:
        pairs = self.value_words(number)
        noun, words = self.hunt_words(number)
        plain = [value for value, _ in pairs]
        tries = [f'{family} {value} {noun}'.strip() for family in self.series_for(number) for value in (plain[:1] or [''])]
        tries.extend(f'{value} {noun}'.strip() for value in plain)
        if noun:
            tries.extend(plain)
        tries.extend(f'{word} {noun}'.strip() for word in words[:HUNT_FIT_WORDS])
        if noun:
            tries.append(noun)
        return tries

    def _g6_fit_searches(self, tool: str, base: dict[str, Any], field: str, numbers: list[int], room: int) -> int | None:
        spent = 0
        for number in numbers:
            tries = self._g6_fit_tries(number)
            share = max(2, room // max(1, len(numbers)))
            used = 0
            for query in dict.fromkeys(tries):
                if spent >= room or used >= share or self.finished or self.clock() > HUNT_BY + HUNT_GRACE:
                    break
                faults: list[str] = []
                args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
                if faults:
                    continue
                reply = self.act(tool, args)
                spent, used = spent + 1, used + 1
                if reply is None:
                    return None
                self.observe(tool, reply.get('observation'))
        return spent

    def _g6_fit_pool(self, number: int, cat_key: Any, words: set[str]) -> list[str]:
        item = self.items[number - 1]
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        pool: list[str] = []
        for ident in self.seen_order:
            if ident in self.full or not self.buyable(ident):
                continue
            if cat_key and not self.owns(item, str((self.rows.get(ident) or {}).get(cat_key) or '')):
                continue
            pool.append(ident)
        if cap is not None:
            pool = [ident for ident in pool if (self.price_of(ident) or 0) <= cap]

        def rank(ident: str) -> tuple[int, float]:
            price = self.price_of(ident)
            return (-len(words & _text_words(_dump(self.rows.get(ident) or {}))), price if price is not None else float('inf'))
        return sorted(pool, key=rank)

    def unread_pages(self, tool: str, wanted: dict[int, set[str]], room: int) -> int:
        tally = {'spent': 0}
        cat_key = self.category_key()
        share = max(1, room // max(1, len(wanted)))
        for limit, (number, words) in ((lim, entry) for lim in (share, room) for entry in wanted.items()):
            stop = min(room, tally['spent'] + limit)
            for mark, (name, args) in list(self.read_args.items()):
                if name != tool or not words & _text_words(' '.join(v for v in args.values() if isinstance(v, str)).lower()):
                    continue
                seen = self.read_calls[mark]
                if cat_key:
                    owners = {self.owner_of(str(record.get(cat_key) or '')) for _, record in self.records_in(seen)}
                    if owners and 0 not in owners and number not in owners:
                        continue
                if not self._g6_walk_unread(tool, args, mark, seen, stop, tally):
                    return tally['spent']
        return tally['spent']

    def _g6_walk_unread(self, tool: str, call: dict[str, Any], mark: str, seen: Any, stop: int, tally: dict[str, int]) -> bool:
        walked = {mark}
        while tally['spent'] < stop and not self.finished:
            page = self.paging(tool, seen)
            if not page or page[3] is None:
                return True
            faults: list[str] = []
            call = _shape({**call, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
            if faults:
                return True
            here = self.call_mark(tool, call)
            if here in walked:
                return True
            walked.add(here)
            if here in self.read_calls:
                seen = self.read_calls[here]
                continue
            reply = self.act(tool, call)
            tally['spent'] += 1
            if reply is None:
                return False
            seen = reply.get('observation')
            self.observe(tool, seen)
        return True

    def call_mark(self, name: str, args: dict[str, Any]) -> str:
        props = (self.schemas.get(name) or {}).get('properties') or {}
        page_key = self.page_args.get(name)

        def default_page(key: str, value: Any) -> bool:
            if not isinstance(value, int) or isinstance(value, bool):
                return False
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            if value in (spec.get('minimum'), spec.get('default')):
                return True
            return value == 1 and key == page_key and spec.get('minimum') is None and spec.get('default') is None
        kept = {key: (' '.join(value.lower().split()) if isinstance(value, str) else value)
                for key, value in sorted(args.items()) if not default_page(key, value)}
        return name + _dump(kept)

    def note_read(self, name: str, args: dict[str, Any], seen: Any) -> None:
        if name in self.page_tools() | {self.roles.get(r) for r in ('ask', 'cart', 'cart_view', 'buy', 'refuse')}:
            return
        self.query_is_new(name, args)
        if not isinstance(seen, (dict, list)) or _failed(seen):
            return
        mark = self.call_mark(name, args)
        self.read_calls[mark] = seen
        self.read_args[mark] = (name, dict(args))

    def query_is_new(self, name: str, args: dict[str, Any]) -> bool:
        words_only = {k: v for k, v in args.items() if isinstance(v, bool) or not isinstance(v, int)}
        bucket = self.page_queries.setdefault(name, set())
        size = len(bucket)
        bucket.add(_dump(words_only))
        return len(bucket) > size

    @staticmethod
    def _g6_floor(spec: Any) -> int:
        low = spec.get('minimum')
        return low if isinstance(low, int) else 1

    @_soft('')
    def last_page_first(self, name: str, args: dict[str, Any], observation: Any) -> str:
        if name in self.page_args or not isinstance(observation, dict):
            return ''
        props = (self.schemas.get(name) or {}).get('properties') or {}
        numeric = [k for k, v in args.items() if isinstance(v, int) and not isinstance(v, bool) and k in props and v > self._g6_floor(props[k])]
        if len(numeric) != 1 or self.left() <= self.tail() + 4:
            return ''
        shape = self.read_shape(name, observation)
        if not self._g6_ends_short(shape, observation):
            return ''
        rows = observation.get(shape['rows'])
        key = numeric[0]
        low = self._g6_floor(props[key])
        faults: list[str] = []
        call = _shape({**args, key: low}, self.schemas.get(name), 'args', faults)
        if faults:
            return ''
        reply = self.act(name, call)
        if reply is None:
            return ''
        seen = reply.get('observation')
        self.observe(name, seen)
        fresh = seen.get(shape['rows']) if isinstance(seen, dict) else None
        if not isinstance(fresh, list) or len(fresh) < len(rows):
            return ''
        if {i for i, _ in self.records_in(seen)} & {i for i, _ in self.records_in(observation)}:
            return ''
        self.page_args[name] = key
        self.trace.append({'kind': 'page_arg', 'tool': name, 'arg': key})
        return (f'\n[Cairn] That call landed on the final page of results; {key} chooses which page comes back. '
                f'Here is page {low} of the very same search:\n') + self.show(seen)

    @staticmethod
    def _g6_ends_short(shape: Any, observation: dict[str, Any]) -> bool:
        if not shape or not shape.get('count') or shape.get('next') is None:
            return False
        if observation.get(shape['next']) is not None:
            return False
        rows, count = observation.get(shape['rows']), observation.get(shape['count'])
        return isinstance(rows, list) and isinstance(count, int) and count > len(rows)

    @_soft(False)
    def names_code(self, args: dict[str, Any]) -> bool:
        said = {c for v in args.values() if isinstance(v, str) for c in _codes(v)}
        if not said:
            return False
        known = {c for mark in self.fit_index()['marks'].values() for part in ('codes', 'works') for c in (mark.get(part) or [])}
        return not said.isdisjoint(known)

    def first_look(self, name: str, args: dict[str, Any], fresh: bool) -> dict[str, Any]:
        key = self.page_args.get(name)
        if key and fresh and isinstance(args.get(key), int):
            props = (self.schemas.get(name) or {}).get('properties') or {}
            return {**args, key: self._g6_floor(props.get(key, {}))}
        return args

    @_soft('')
    def shorter_search(self, name: str, args: dict[str, Any], observation: Any) -> str:
        plan = self.value_template()
        if not plan or plan[0] != name or self.left() <= self.tail() + SHORT_KEEP or self.finished or not isinstance(observation, dict):
            return ''
        text = args.get(plan[2])
        if not isinstance(text, str) or self.records_in(observation) or len(text.split()) < 3:
            return ''
        if any(self._g6_has_content(value) for value in observation.values()):
            return ''
        short = ' '.join([w for w in text.split() if not _codes(w)][:2])
        if not short:
            return ''
        faults: list[str] = []
        loose = {k: v for k, v in {**args, plan[2]: short}.items() if isinstance(v, bool) or not isinstance(v, int)}
        call = _shape(loose, self.schemas.get(name), 'args', faults)
        if faults:
            return ''
        known = self.searches_seen.get(name + _dump(call))  # SR
        if known and known[0] == self.version:
            return (f'\n[Cairn] No listing matched every word. The shorter search {short!r} already ran at step {known[1]}, '
                    f'so no step was spent; its result:\n') + known[2][:3000]
        reply = self.act(name, call)
        if reply is None:
            return ''
        seen = reply.get('observation')
        self.observe(name, seen)
        self.trace.append({'kind': 'shorter', 'from': text, 'to': short, 'rows': len(self.records_in(seen))})
        return f'\n[Cairn] No listing matched every word. Retried with just {short!r}, which returned:\n' + self.show(seen)

    @staticmethod
    def _g6_has_content(value: Any) -> bool:
        if isinstance(value, (list, str)):
            return bool(value)
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0

    @_soft('')
    def first_moves(self) -> str:
        plan = self.value_template()
        if not plan or self.max_steps < OPEN_MIN_STEPS or len(self.items) < OPEN_ITEMS:
            return ''
        tool, base, field = plan
        self.env_gap = OPEN_GAP
        opening = {'plan': (tool, base, field), 'shown': [],
                   'stop_at': self.step + self.left() - self.tail() - self.msgs_free() - OPEN_LATER * len(self.items) - HUNT_KEEP // 2}
        moves = (self._g6_move_search_items, self._g6_move_widen, self._g6_move_wrap)
        for move in moves:
            if not move(opening):
                return '\n'.join(opening['shown'])
        return opening['text']

    def _g6_item_queries(self, item: Any) -> list[str]:
        told = [need for _, need in self.refs]
        wanted = [_plain_value(m.split(':', 1)[1].strip()) for m in item.musts if ':' in m and m not in told]  # CO3
        wanted = [w for w in wanted if w and len(w.split()) <= 3 and not re.search(OPEN_NOT, w, re.I)]
        parts = item.name.split()
        noun = item.name if len(parts) <= 3 else ' '.join(parts[-2:])
        lead = [' '.join(wanted[:2] + [noun])] if wanted else []
        return list(dict.fromkeys(lead + [noun]))

    def _g6_move_search_items(self, opening: dict[str, Any]) -> bool:
        tool, base, field = opening['plan']
        shown, stop_at = opening['shown'], opening['stop_at']
        for number, item in enumerate(self.items, 1):
            for query in self._g6_item_queries(item):
                faults: list[str] = []
                args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
                if faults or self.finished or self.step >= stop_at:
                    break
                reply = self.act(tool, args)
                if reply is None:
                    return False
                seen = reply.get('observation')
                self.observe(tool, seen)
                if not self.records_in(seen):
                    shown.append(f"{query}: {'the store refused the call: ' + self.show(seen) if _failed(seen) else 'no rows'}")
                    continue
                self.searched = (self.searched + [dict(args)])[-8:]
                shown.append(f'{query}: ' + self.show(seen))
                shown.extend(self.more_rows_for_keys(tool, args, seen, stop_at, number))
                if self.rows_of_item(number) >= OPEN_EACH:
                    break
        return True

    def _g6_move_widen(self, opening: dict[str, Any]) -> bool:
        if not (self.rows and self.roles.get('ask') and self.msgs_free() > 0):
            return True
        start = self.step
        found = [n for n in range(1, len(self.items) + 1) if self.rows_of_item(n)]
        limit = self.max_steps // OPEN_SHARE
        while self.step - start < limit and self.step < opening['stop_at'] and not self.finished:
            if min((self.records_of_item(n) for n in found), default=OPEN_EACH) >= OPEN_EACH:
                break
            had = len(self.full)
            self.widen()
            if len(self.full) == had:
                break
        return True

    def _g6_move_wrap(self, opening: dict[str, Any]) -> bool:
        self.env_gap = ENV_GAP
        self.trace.append({'kind': 'first_moves', 'rows': len(self.rows), 'opened': len(self.full)})
        lines = opening['shown'] + ['read in full: ' + self.page_text(ident, record) for ident, record in self.full.items()]
        proof = self.evidence(list(self.full.values()))
        if proof:
            lines.append(proof)
        opening['text'] = '\n'.join(lines)
        return True

    def page_text(self, ident: str, record: dict[str, Any]) -> str:
        block = _fact_block(record) or {}
        flat = {k: v for k, v in record.items() if not isinstance(v, (dict, list)) and v is not block}
        attrs = _dump(_clip(block, 120))[:PAGE_ATTR_CHARS]
        other = _dump(_clip(flat, 200))[:PAGE_REST_CHARS]
        return '{}: attributes {} | {}'.format(ident, attrs, other)

    def more_rows_for_keys(self, tool: str, args: dict[str, Any], seen: Any, stop_at: int, number: int) -> list[str]:
        page = self.paging(tool, seen)
        ready = page and page[3] is not None and self.rows_of_item(number) < OPEN_EACH and not self.finished and self.step < stop_at
        if not ready:
            return []
        faults: list[str] = []
        call = _shape({**args, page[2]: page[3]}, self.schemas.get(tool), 'args', faults)
        reply = None if faults else self.act(tool, call)
        if reply is None:
            return []
        self.observe(tool, reply.get('observation'))
        return ['next page: ' + self.show(reply.get('observation'))]

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        found = self.loose_lines(observation)
        meaning = self.take_note(found) if found else ''
        if meaning:
            self.queued.append(f'({name} also showed) {meaning}')
        return meaning

    @staticmethod
    def _g7_spots(lanes: list[list[str]], depth: int | None = None) -> list[tuple[int, int]]:
        # (rank, lane) pairs in round-robin order: every lane's first entry, then every lane's second, ...
        return sorted((k, j) for j, lane in enumerate(lanes) for k in range(len(lane) if depth is None else min(depth, len(lane))))

    def _g7_compare_slot(self, tool: str | None, schema: dict[str, Any]) -> str | None:
        props = schema.get('properties') or {}
        slot = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not tool or not slot:
            return None
        if any((k != slot for k in schema.get('required') or [])) or self.compare_bare():
            return None
        return slot

    def _g7_compare_heard(self, tool: str, seen: Any, picks: list[str], before: int) -> bool:
        self.note_view(tool, seen)
        rows = self.records_in(seen)
        if not rows:
            self.compare_bad = min(self.compare_bad or 999, len(picks))
        else:
            self.facts_from[tool] = self.facts_from.get(tool, False) or any((_fact_block(r) for _, r in rows))
            self.compare_ok = max(self.compare_ok, len(rows))
        self.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.full) - before})
        if rows and self.facts_from.get(tool) is False:
            self.widen_singly()
            return False
        return len(self.full) == before and len(picks) > 2 and self.left() > 12

    def widen(self) -> None:
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or '') or {}
        slot = self._g7_compare_slot(tool, schema)
        if slot is None:
            self.widen_singly()
            return
        if self.left() <= 12:
            return
        limit = schema['properties'][slot].get('maxItems')
        cap = min(limit if isinstance(limit, int) else 4, self.compare_room())
        groups = sorted(self.item_groups(), key=lambda ids: len([i for i in ids if i in self.full]))
        lanes = [[i for i in ids if i not in self.full] for ids in groups]
        picks = [lanes[j][k] for k, j in self._g7_spots(lanes, cap)][:cap]
        if len(picks) < 2:
            self.fill_facts(picks, 12)
            return
        before = len(self.full)
        for retry, batch in enumerate((picks, picks[:2])):
            faults: list[str] = []
            args = _shape({slot: batch}, schema, 'args', faults)
            if faults:
                return
            reply = self.act(tool, args)
            if reply is None:
                return
            seen = reply.get('observation')
            self.observe(tool, seen)
            if retry:
                self.compare_ok = max(self.compare_ok, len(self.records_in(seen)))
            elif not self._g7_compare_heard(tool, seen, batch, before):
                return

    def _g7_need_words(self) -> set[str]:
        words: set[str] = set()
        for _, need in self.needs_now():
            words.update(re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower()))
        return words

    def _g7_single_lane(self, ids: list[str], wanted: set[str], each: int) -> tuple[list[str], int]:
        opened = len([i for i in ids if i in self.full])
        cheapest = self.brief.get('pick') == 'cheapest'

        def rank(i: str) -> tuple[int, float]:
            overlap = len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower())))
            if not cheapest:
                return (-overlap, 0.0)
            price = self.price_of(i)
            return (-overlap, price if price is not None else float('inf'))
        pool = sorted((i for i in ids if i not in self.full and self.buyable(i)), key=rank)
        return (pool[:max(0, each - opened)], max(0, 4 - opened))

    def widen_singly(self) -> None:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not (tool and target):
            return
        wanted = self._g7_need_words()
        each = 4 if len(self.items) > 2 else WIDEN_EACH
        built = [self._g7_single_lane(ids, wanted, each) for ids in self.item_groups()]
        lanes = [lane for lane, _ in built]
        heads = [head for _, head in built]
        spots = self._g7_spots(lanes)
        first = [lanes[j][k] for k, j in spots if k < heads[j]]
        more = [lanes[j][k] for k, j in spots if k >= heads[j]]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.fill_facts(first[:min(cap, 8)], 22)
        self.fill_facts((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get('ask'):
            return False
        if self.msgs_free() <= 0 or self.left() <= 8 or self.clock() > ASK_BY:
            return False
        unread = any((not self.records_of_item(n) for n in range(1, len(self.items) + 1)))
        if self.loop_calls < ASK_TURNS and unread:
            return False
        return len(self.full) >= 3 * len(self.items) + 1 or self.step >= max(8, self.max_steps // 4)

    @staticmethod
    def _g7_nested_slot(props: dict[str, Any], text: str) -> dict[str, Any]:
        for key, spec in props.items():
            if not isinstance(spec, dict):
                continue
            if 'array' in _kinds(spec):
                return {key: [text]}
            inner = next((k for k, v in (spec.get('properties') or {}).items() if isinstance(v, dict) and 'string' in _kinds(v)), None)
            if inner:
                return {key: {inner: text}}
        return {}

    def message_args(self, text: str) -> dict[str, Any] | None:
        schema = self.schemas.get(self.roles.get('ask') or '') or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        plain = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and 'const' not in v]
        pool = [k for k in required if k in plain] or plain

        def fitness(k: str) -> tuple[bool, bool, int]:
            snake = _snake(k)
            return (not re.search(NOT_MESSAGE, snake, re.I), bool(re.search(MESSAGE_FIELD, snake, re.I)), props[k].get('maxLength') or 10 ** 6)
        args: dict[str, Any] = {max(pool, key=fitness): text} if pool else self._g7_nested_slot(props, text)
        if not args:
            return None
        for key in (k for k in required if k not in args):
            value = _filler(props.get(key) or {})
            args[key] = value
            if value is None:
                return None
        faults: list[str] = []
        shaped = _shape(args, schema, 'args', faults)
        return shaped if not faults else None

    def _g7_open_lanes(self, size: int) -> None:
        self.lanes = self.plan_lanes(size)
        self.lane_found = dict.fromkeys(self.lanes, 0)
        self.lane_spent = dict.fromkeys(self.lanes, 0)
        self.lane_used = {n: set() for n in self.lanes}

    def _g7_plan_interview(self) -> list[list[int]]:
        left_now = self.msg_limit - self.msg_used
        self._g7_open_lanes(max(left_now, self.list_cap()) if self.listing or ASK_SINGLES else left_now)  # AK: Giyu's ranking, walked one by one
        self.trace.append({'kind': 'ask_policy', 'division': 'A singles' if ASK_SINGLES else 'lists', 'messages': left_now, 'free': self.msgs_free(), 'items': len(self.items), 'order': [n for n in self.plan_order if n in self.lanes] + [n for n in self.lanes if n not in self.plan_order], 'hinted': list(self.plan_order)})  # AK
        self.follow_up = []
        return self.groups_for(self.msgs_free()) if self.listing else []

    def _g7_may_question(self) -> bool:
        return self.msgs_free() > 0 and not self.finished and self.left() > 6 and self.clock() < QUESTION_BY

    def _g7_select_and_send(self, groups: list[list[int]], told: list[str]) -> list[tuple[int, str, str]]:
        batch: list[tuple[int, str, str]] = []
        while not batch and groups and self.listing:
            batch = self.ask_group(groups.pop(0), told)
        return batch or self.ask_next(told)

    def _g7_hear(self, groups: list[list[int]], told: list[str]):
        while self._g7_may_question():
            batch = self._g7_select_and_send(groups, told)
            if not batch:
                return
            yield from batch

    def _g7_pointed(self, told: list[str]) -> list[str]:
        return [need for _, need in self.refs if any((need in t for t in told))]

    def _g7_apply(self, sent: list[tuple[int, str, str]], told: list[str]) -> str:
        found_items = [n for n, hits in self.lane_found.items() if hits > 0]
        chain = self.early_trail()
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        parts = ['The shopper was asked these questions by the Cairn interviewer (one step per question):\n' + '\n'.join((f'Q: {q}\nA: {a}' for _, q, a in sent)),
                 '\nFirm requirements learned from the replies: ' + ('; '.join(told) if told else 'none') + '. Every pick must meet them (run new searches with the new values when needed).']
        pointed = self._g7_pointed(told)
        if pointed:
            parts.append(' The values for these come from a different listing: ' + ' | '.join(pointed) + '. Locate that listing, read its whole page, then follow its references from page to page; the page where they end holds the value, and the order only counts once every one of them is opened.')
        parts += ['\n' + extra for extra in (chain, searched) if extra]
        if self.queued:
            parts.append('\n[notes seen meanwhile] ' + ' | '.join(self.queued))
            self.queued = []
        return ''.join(parts)

    def interview(self) -> str:
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        self.widen()
        groups = self._g7_plan_interview()
        told: list[str] = []
        sent = list(self._g7_hear(groups, told))
        return self._g7_apply(sent, told)

    def _g7_late_blocked(self) -> bool:
        gates = (lambda: self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit,
                 lambda: any((p['when'] == 'order' and p['messages'] > 0 for p in self.process)),
                 lambda: self.left() < LATE_STEPS or self.clock() > LATE_BY or not self.judges())
        return any((gate() for gate in gates))

    def late_question(self) -> str:
        if self._g7_late_blocked():
            return ''
        self.late_asked = True
        if not self.lanes:
            self._g7_open_lanes(self.list_cap() if self.listing or ASK_SINGLES else 1)  # AK
        told: list[str] = []
        money = self.budget
        batch = self.ask_next(told, LATE_STEPS - 2)
        if not batch:
            self.trace.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
            return ''
        if money != self.budget:
            told.append(f'budget now {self.budget}')
        number, question, answer = batch[0]
        self.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
        if not told:
            return ''
        self.version += 1
        text = ('Before checkout, Cairn put one more question to the shopper (one step):\n'
                f'Q: {question}\nA: {answer}\nFirm requirements learned from this reply: ' + '; '.join(told)
                + '. Checkout has not happened yet: test every cart line against them (run new searches with the new values when needed), correct the cart, and only then check out.')
        pointed = self._g7_pointed(told)
        if pointed:
            text += ' The values for these come from a different listing: ' + ' | '.join(pointed) + '. Locate that listing, read its whole page, then follow its references from page to page; the page where they end holds the value.'
        searched = self.value_search([number])
        return f'{text}\n{searched}' if searched else text

    def _g7_shelf_words(self, cat_key: str | None) -> set[str]:
        shelves = {_words(item.shelf or '') for item in self.items}
        shelves.update((_words(item.name) for item in self.items))
        if cat_key:
            shelves.update((_words(str(row.get(cat_key) or '')) for row in self.rows.values()))
        return shelves

    def _g7_names_shelf(self, value: Any, shelves: set[str]) -> bool:
        if not isinstance(value, str):
            return False
        if _words(value) in shelves:
            return True
        roots = {w for w in _stem(value) if len(w) >= 4}
        return any((roots & {w for w in _stem(s) if len(w) >= 4} for s in shelves if s))

    def _g7_template_field(self, args: dict[str, Any], named: list[str], texts: list[str]) -> str | None:
        fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
        if named:
            return fields[0] if fields else None
        cat_key = self.category_key()
        labels = {str(r.get(cat_key) or '').strip().lower() for r in self.rows.values()} - {''} if cat_key else set()
        for k in fields:
            used = [a.get(k) for a in self.searched]
            if len({str(u) for u in used if u}) < 2:
                continue
            if not any((len(str(u).split()) >= 2 for u in used)):
                continue
            if any((str(u or '').strip().lower() in labels for u in used)):
                continue
            return k
        return None

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        if not tool:
            return None
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search(QUERY_WORDS, _snake(k), re.I)]
        hit = next(((args, field) for args in reversed(self.searched) for field in [self._g7_template_field(args, named, texts)] if field), None)
        if hit is None:
            return None
        args, field = hit
        listed = lambda k: isinstance((props.get(k) or {}).get('enum'), list)
        shelves = self._g7_shelf_words(self.category_key())
        keep = {k: v for k, v in args.items() if k != field and (listed(k) or k in required) and (k in required or not self._g7_names_shelf(v, shelves))}
        worded = [a for a in self.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
        for k in [k for k in keep if listed(k)]:
            firsts: dict[str, Any] = {}
            for a in worded:
                if k in a:
                    firsts.setdefault(_dump(a[k]), a[k])
            if firsts:
                tally = [_dump(a[k]) for a in worded if k in a]
                keep[k] = firsts[max(firsts, key=tally.count)]
        return (tool, keep, field)

    def _g7_item_needs(self, number: int):
        for n, need in self.needs_now():
            if n == number and ':' in need:
                yield (need, need.split(':', 1)[1].strip())

    def value_words(self, number: int) -> list[tuple[str, bool]]:
        told = self.items[number - 1].told
        linked = [m for _, m in self.refs]
        out: list[tuple[str, bool]] = []
        for need, rest in self._g7_item_needs(number):
            if need in linked:
                continue
            value = _plain_value(rest)  # CO3
            edge = re.search(FLOOR, value, re.I)
            if edge is not None:
                out += [(edge.group(1).strip(), True)] if need in told else []
            elif value and not re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                out.append((value, need in told))
        return out

    def value_forms(self, number: int) -> dict[str, list[str]]:
        floors = self.floor_values(number)
        return {value: _value_forms(value, value in floors) for value, _ in self.value_words(number)}

    def floor_values(self, number: int) -> set[str]:
        edges = (re.search(FLOOR, rest, re.I) for _, rest in self._g7_item_needs(number))
        return {edge.group(1).strip() for edge in edges if edge}

    def _g7_stated_template(self, tool: str) -> tuple[str, dict[str, Any], str] | None:
        told = self.brief.get('search_words')
        told = told if isinstance(told, dict) else {}
        field = told.get('field')
        field = field if isinstance(field, str) else None
        fixed = told.get('args')
        fixed = fixed if isinstance(fixed, dict) else {}
        chosen = told.get('tool')
        chosen = chosen if isinstance(chosen, str) and chosen in self.schemas else tool
        spec = (self.schemas.get(chosen) or {}).get('properties') or {}
        if field not in spec or 'string' not in _kinds(spec[field]) or 'enum' in spec[field]:
            return None
        faults: list[str] = []
        base = _shape({**fixed, field: 'x'}, self.schemas.get(chosen), 'args', faults)
        if faults:
            return None
        base.pop(field, None)
        return (chosen, base, field)

    @staticmethod
    def _g7_mode(key: str, options: list[Any], last: dict[str, Any], required: list[str]) -> tuple[str, Any] | None:
        reads = [o for o in options if isinstance(o, str) and re.search(QUERY_WORDS, o, re.I)]
        if len(reads) == 1:
            return (key, reads[0])
        others = [o for o in options if o != last.get(key)]
        if key in required and len(options) == 2 and len(others) == 1:
            return (key, others[0])
        return None

    def value_template(self) -> tuple[str, dict[str, Any], str] | None:
        plan = self.search_template()
        tool = self.roles.get('search')
        if plan or not tool:
            return plan
        stated = self._g7_stated_template(tool)
        if stated:
            return stated
        schema = self.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        named = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and 'enum' not in v and re.search(QUERY_WORDS, _snake(k), re.I)]
        last = self.searched[-1] if self.searched else {}
        enums = {k: v['enum'] for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and len(v['enum']) >= 2}
        modes = [m for m in (self._g7_mode(k, opts, last, required) for k, opts in enums.items()) if m]
        if len(named) != 1 or len(modes) > 1 or (enums and not modes):
            return None
        query = named[0]
        base = {k: last[k] for k in required if k in last and k != query}
        base.update(modes)
        return (tool, base, query)

    def _g7_value_queries(self, number: int) -> list[list[str]]:
        pairs = self.value_words(number)
        fresh = [v for v, told in pairs if told]
        if not fresh:
            return []
        lead = [v for v in fresh if len(v.split()) == 1 or re.fullmatch(AMOUNT, v, re.I)]
        order = lead + [v for v, _ in pairs if v not in lead]
        if len(order) == 1:
            tries = [[order[0], self.items[number - 1].name], order] if lead else [order]
        else:
            stated = [i for i, v in enumerate(order) if v not in fresh]
            cut = stated[-1] if stated else 0
            tries = [order, order[:cut] + order[cut + 1:]]
        return [words for words in tries if words]

    def _g7_money_cap(self, found: list[str]) -> float | None:
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        if cap is None:
            return None
        prices = [self.price_of(i) for i in found]
        if prices and all((p is not None and p > 10 * cap for p in prices)):
            return None
        return cap

    def _g7_fitting_rows(self, item: Any, number: int, observation: Any) -> list[str]:
        cat_key = self.category_key()
        found = [i for i, _ in self.records_in(observation)]
        cap = self._g7_money_cap(found)
        label = lambda i: str((self.rows.get(i) or {}).get(cat_key) or '')

        def fits(i: str) -> bool:
            if not self.buyable(i):
                return False
            if cat_key and not self.owns(item, label(i)):
                return False
            return cap is None or (self.price_of(i) or 0) <= cap
        rows = [i for i in found if fits(i)]
        home = self.home_label(number)
        if cat_key and home:
            rows.sort(key=lambda i: _words(label(i)) != home)
        return rows

    def _g7_search_values(self, plan: tuple[str, dict[str, Any], str], number: int, queries: list[list[str]], budget: int, seen: list[str]) -> int:
        tool, base, field = plan
        item = self.items[number - 1]
        for words in queries:
            faults: list[str] = []
            args = _shape({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', faults)
            if faults or self.finished:
                break
            reply = self.act(tool, args)
            budget -= 1
            if reply is None:
                break
            observation = reply.get('observation')
            self.observe(tool, observation)
            rows = self._g7_fitting_rows(item, number, observation)
            seen.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or 'nothing of that kind'))
            if rows:
                opened = self.fill_facts([i for i in rows if i not in self.full][:min(3, budget)], self.tail() + VALUE_KEEP)
                if opened:
                    seen.append('opened: ' + ' | '.join(opened)[:2400])
                return budget - len(opened)
        return budget

    def value_search(self, numbers: list[int]) -> str:
        plan = self.value_template()
        if not plan or self.left() < self.tail() + VALUE_ROOM:
            return ''
        seen: list[str] = []
        budget = min(8, self.left() - self.tail() - VALUE_KEEP)
        for number in numbers:
            queries = self._g7_value_queries(number)
            if queries and budget > 2:
                budget = self._g7_search_values(plan, number, queries, budget, seen)
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Cairn ran one search per item with that item's requirement values combined, then read the first rows of the right kind] " + '\n'.join(seen)

    def _g7_label_counts(self, pool: dict[str, Any], item: Any, cat_key: str) -> dict[str, int]:
        labels = [str(record.get(cat_key) or '') for record in pool.values()]
        kept = [_words(label) for label in labels if label and self.owns(item, label)]
        counts = dict.fromkeys(kept, 0)
        for label in kept:
            counts[label] += 1
        return counts

    def home_label(self, number: int) -> str:
        cat_key = self.category_key()
        if not cat_key:
            return ''
        item = self.items[number - 1]
        for pool in (self.full, self.rows):
            counts = self._g7_label_counts(pool, item, cat_key)
            named = next((w for w in (_words(item.shelf), _words(item.name)) if w in counts), None)
            if named is not None:
                return named
            if counts:
                return max(counts, key=counts.get)
        return ''

    def groups_for(self, free: int) -> list[list[int]]:
        numbers = [n for n in self.plan_order if self.lanes.get(n)] + [n for n in self.lanes if n not in self.plan_order and self.lanes[n]]
        if free <= 0 or len(numbers) <= free:
            return []
        out: list[list[int]] = []
        for slots in range(free, 0, -1):
            size = -(-len(numbers) // slots)
            out.append(numbers[:size])
            numbers = numbers[size:]
        return out

    def ask_group(self, group: list[int], told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        parts = self._g8_group_parts(group)
        if not parts:
            return []
        if len(parts) == 1:
            return self._g8_ask_alone(parts[0], told, floor)
        parts, text = self._g8_phrase_group(parts, self._g8_group_share(len(group)))
        if not parts:
            return []
        lead, lead_labels = parts[0]
        batch = self.send([(lead, lead_labels[0], text)], floor)
        if batch:
            self._g8_settle_group(parts, text, batch, told)
        return batch

    def _g8_group_share(self, width: int) -> int:
        cap = self.list_cap()
        if self.list_now() >= cap:
            return max(MIX_LABELS, cap // width)
        return max(1, self.list_now() // width)

    def _g8_open_labels(self, number: int) -> list[str]:
        pool = self.lanes.get(number) or []
        if not pool:
            return []
        used = self.lane_used.setdefault(number, set())
        return [label for label in pool if _words(label) not in used and not self.told_about(number, label)]

    def _g8_group_parts(self, group: list[int]) -> list[tuple[int, list[str]]]:
        share = self._g8_group_share(len(group))
        picked = ((number, self._g8_open_labels(number)) for number in group)
        return [(number, left[:share]) for number, left in picked if left]

    def _g8_ask_alone(self, part: tuple[int, list[str]], told: list[str], floor: int) -> list[tuple[int, str, str]]:
        number, labels = part
        fitted, text = self.fit_ask(number, labels, False)
        if not fitted:
            return []
        batch = self.send([(number, labels[0], text)], floor)
        if batch:
            self.finish_ask(number, fitted, False, batch, told)
        return batch

    def _g8_group_text(self, parts: list[tuple[int, list[str]]], size: int) -> str:
        clauses = []
        for number, labels in parts:
            head = labels[:size]
            piece = head[0] if len(head) < 2 else f"{', '.join(head[:-1])} or {head[-1]}"
            clauses.append(f'what {piece} for the {self.item_said(number)}')
        joined = ', and '.join(clauses)
        return 'W' + joined[1:].replace(' for the ', ' would you like for the ', 1) + '?'

    def _g8_fits(self, text: str) -> bool:
        args = self.message_args(text)
        return bool(args and _says(args, text))

    def _g8_phrase_group(self, parts: list[tuple[int, list[str]]], share: int) -> tuple[list[tuple[int, list[str]]], str]:
        while parts:
            size = next((s for s in range(share, 0, -1) if self._g8_fits(self._g8_group_text(parts, s))), 0)
            if size:
                return [(number, labels[:size]) for number, labels in parts], self._g8_group_text(parts, size)
            parts = parts[:-1]
        return [], ''

    def _g8_bump(self, counter: dict[int, int], number: int) -> None:
        counter[number] = counter.get(number, 0) + 1

    def _g8_told_sizes(self) -> list[int]:
        return [len(item.told) for item in self.items]

    def _g8_crowded(self, heard: dict) -> bool:
        return any(_count(x) == 1 for x in _items(heard.get('crowded')))

    def _g8_lists_off(self, number: int) -> None:
        self.listing = False
        self.trace.append({'kind': 'lists_off', 'item': number})

    def _g8_settle_group(self, parts: list[tuple[int, list[str]]], text: str, batch: list[tuple[int, str, str]], told: list[str]) -> None:
        for number, _ in parts:
            self._g8_bump(self.lane_spent, number)
        self.list_asks += 1
        before = self._g8_told_sizes()
        width = sum(len(labels) for _, labels in parts)
        heard = self.hear([(0, text, batch[0][2])], (LIGHT, HEAVY), width)
        heard.pop('free', None)
        self.take_heard(heard, told)
        answered = False
        for number, labels in parts:
            fresh = self.items[number - 1].told[before[number - 1]:]
            if not fresh:
                continue
            answered = True
            self._g8_bump(self.lane_found, number)
            self.settle(number, labels, False, list(fresh), False)
        lead, lead_labels = parts[0]
        if not answered:
            if self._g8_crowded(heard):
                self._g8_lists_off(lead)
                return
            self.lane_used.setdefault(lead, set()).add(_words(lead_labels[0]))
        self.trace.append({'kind': 'mixed_ask', 'parts': parts, 'answered': answered})

    def finish_ask(self, number: int, labels: list[str], shown: bool, batch: list[tuple[int, str, str]], told: list[str]) -> None:
        self._g8_bump(self.lane_spent, number)
        self.list_asks += 1
        before = self._g8_told_sizes()
        heard = self.hear(batch, (LIGHT, HEAVY), len(labels))
        if len(labels) > 1:
            heard.pop('free', None)
        self.take_heard(heard, told)
        gained = [(n, item.told[size:]) for n, (item, size) in enumerate(zip(self.items, before), 1)]
        fresh: list[str] = []
        for n, extra in gained:
            fresh.extend(extra)
            if extra:
                self._g8_bump(self.lane_found, n)
        crowded = self._g8_crowded(heard)
        self.trace.append({'kind': 'heard', 'item': number, 'labels': labels, 'new': fresh, 'free': [str(x.get('property')) for x in _items(heard.get('free')) if isinstance(x, dict)][:4], 'crowded': crowded})  # AK
        self.settle(number, labels, shown, fresh, crowded)

    def list_cap(self) -> int:
        if len(self.items) == 1:
            return LIST_SIZE
        return LIST_MIXED

    def list_now(self) -> int:
        steps = {1: LIST_STEPS, 2: LIST_STEPS_PAIR}.get(len(self.items), LIST_STEPS_MIXED)
        cap = self.list_cap()
        if self.list_asks >= len(steps):
            return cap
        return min(steps[self.list_asks], cap)

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        rank = {n: i for i, n in enumerate(self.plan_order)}  # AK: hinted / ask-first items first, as Giyu
        scarce = self.msg_limit <= SCARCE_MESSAGES and not self.listing and not ASK_SINGLES  # AK: singles take the items in turn

        def priority(n: int) -> tuple:
            return (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), rank.get(n, len(rank) + n))
        width = self.list_now() if self.listing else 1
        remaining = ((n, [label for label in lanes[n] if _words(label) not in used[n] and not self.told_about(n, label)]) for n in sorted(lanes, key=priority))
        return next(((n, left[:width]) for n, left in remaining if left), (0, []))

    def told_about(self, number: int, label: str) -> bool:
        mine = _spelled(label)
        heads = (_spelled(said.partition(':')[0]) for said in self.items[number - 1].told)
        return any(_names(mine, head) for head in heads)

    def _g8_fact_words(self, number: int) -> dict[str, set[str]]:
        seen: dict[str, set[str]] = {}
        for ident in self.item_ids(number):
            facts = _fact_block(self.full.get(ident) or {}) or {}
            for key, value in facts.items():
                bucket = seen.setdefault(_words(key), set())
                bucket.update(w for w in _roots(str(value)) if len(w) > 1)
        return seen

    def look_alikes(self, number: int, label: str) -> list[str]:
        mine = _words(label)
        seen = self._g8_fact_words(number)
        split = {name: set(name.split()) for name in seen}
        plain = {w for w in mine.split() if sum(w in words for words in split.values()) <= 2}
        own_values = seen.get(mine, set())
        ranked = []
        for order, (name, values) in enumerate(seen.items()):
            if not name or name == mine:
                continue
            weight = 3 * len(plain & split[name]) + min(len(own_values & values), 3)
            if weight:
                ranked.append((-weight, order, name))
        ranked.sort()
        return [name for _, _, name in ranked[:LOOK_ALIKES]]

    def ask_text(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) < 2:
            label = labels[0]
            alike = self.apart.get(label, {}).get(number)
            if not alike:
                return self.question(number, label, self.samples(number, label) if shown else None)
            return f"Apart from the {' and the '.join(alike)}, which {label} would you like for the {self.item_said(number)}?"
        head = ', '.join(labels[:-1])
        return f'What {head} or {labels[-1]} would you like for the {self.item_said(number)}?'

    def fit_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        drafts = ((labels[:size], self.ask_text(number, labels[:size], shown)) for size in range(len(labels), 0, -1))
        return next(((head, text) for head, text in drafts if self._g8_fits(text)), ([], ''))

    def _g8_pick(self) -> tuple[int, list[str], bool]:
        self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
        if self.follow_up:
            number, labels, shown = self.follow_up.pop(0)
            return number, labels, shown
        number, labels = self.next_ask(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
        return number, labels, False

    def _g8_phrase(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        fitted, text = self.fit_ask(number, labels, shown)
        if fitted or not shown:
            return fitted, text
        return self.fit_ask(number, labels, False)

    def _g8_plan_one(self) -> tuple[int, list[str], bool, str] | None:
        for _ in range(LIST_SIZE):
            number, labels, shown = self._g8_pick()
            if not labels:
                return None
            fitted, text = self._g8_phrase(number, labels, shown)
            if fitted:
                return number, fitted, shown, text
            self.lane_used.setdefault(number, set()).add(_words(labels[0]))
            self.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
        return None

    def ask_next(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        planned = self._g8_plan_one()
        if planned is None:
            return []
        number, labels, shown, text = planned
        batch = self.send([(number, labels[0], text)], floor)
        if batch:
            self.finish_ask(number, labels, shown, batch, told)
        return batch or []

    def _g8_retry_one_product(self, number: int, labels: list[str], shown: bool) -> bool:
        if number in self.one_product or not self.name_parts(number):
            return False
        self.one_product.add(number)
        self.trace.append({'kind': 'one_product', 'item': number, 'said': self.item_said(number)})
        self.follow_up.insert(0, (number, labels, shown))
        return True

    def _g8_retry_apart(self, number: int, labels: list[str], shown: bool) -> bool:
        if self.apart:
            return False
        alike = self.look_alikes(number, labels[0])
        if not alike:
            return False
        self.apart.setdefault(labels[0], {})[number] = alike
        self.trace.append({'kind': 'apart', 'item': number, 'label': labels[0], 'from': alike})
        self.follow_up.insert(0, (number, labels, False))
        return True

    def _g8_settle_single(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool, used: set[str]) -> None:
        used.add(_words(labels[0]))
        if crowded and self.listing:
            self._g8_lists_off(number)
        if new or not crowded:
            return
        for retry in (self._g8_retry_one_product, self._g8_retry_apart):
            if retry(number, labels, shown):
                return

    def _g8_settle_list(self, number: int, labels: list[str], new: list[str], used: set[str]) -> None:
        hit = self.answered_label(labels, new) if new else -1
        used.update(_words(label) for label in labels[:max(hit, 0) + 1])
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        misses = self.lane_misses.get(number, 0) + 1 if hit < 0 else 0
        self.lane_misses[number] = misses
        if hit >= 0:
            ahead = labels[hit + 1:]
        else:
            ahead = labels[:0:-1]
        if misses and misses % 2 == 0:
            middle = (len(ahead) - 1) / 2
            ahead = [ahead[i] for i in sorted(range(len(ahead)), key=lambda i: (abs(i - middle), i))]
            self.lanes[number] = rest + ahead
        else:
            self.lanes[number] = ahead + rest
        self.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

    def settle(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool) -> None:
        used = self.lane_used.setdefault(number, set())
        if len(labels) == 1:
            self._g8_settle_single(number, labels, shown, new, crowded, used)
        elif crowded and not new:
            self._g8_lists_off(number)
            self.follow_up.insert(0, (number, labels[:1], False))
        else:
            self._g8_settle_list(number, labels, new, used)

    def _g8_need_slot(self, labels: list[str], need: str, vocab: dict[str, set[str]]) -> int | None:
        prop, _, value = need.partition(':')
        scores = [len(_spelled(label) & _spelled(prop)) for label in labels]
        if not any(scores):
            scores = [len(vocab.get(_words(label), set()) & _roots(value)) for label in labels]
        top = max(scores, default=0)
        return scores.index(top) if top > 0 else None

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        vocab: dict[str, set[str]] = {}
        for _, key, _, values in self.key_table():
            vocab.setdefault(_words(key), set()).update(w for v in values for w in _roots(v))
        slots = [slot for slot in (self._g8_need_slot(labels, need, vocab) for need in new) if slot is not None]
        return min(slots, default=-1)

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        settled = [_stem(m.partition(':')[0]) for m in self.items[number - 1].musts]
        numeric = re.compile('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$')
        measured = {}
        for _, key, _, values in table:
            measured[key] = 2 * len([v for v in values if numeric.match(v)]) > len(values)
        rows = [row for row in table if self.owner_of(row[0]) == number]
        if not rows and len(self.items) == 1:
            rows = list(table)
        rows = sorted(rows, key=lambda r: (-r[2], measured.get(r[1], False), -len(r[3])))
        every = [key for _, key, _, _ in rows if not any(_names(prop, _stem(key)) for prop in settled)]

        def tier(key: str) -> int:
            if re.search(NOT_PROPERTY, key) and not re.search('\\w[ _]models?\\b', key):
                return 2
            return 1 if re.search(CATCH_ALL, key) else 0
        return sorted(every, key=tier)

    def adopt_labels(self) -> None:
        cat_key = self.category_key()
        if len(self.items) < 2 or not cat_key:
            return
        self.part_shelves()
        labels = {str(record.get(cat_key) or '') for record in self.full.values()} - {''}
        owners = {label: self.owner_of(label) for label in labels}
        strays = [label for label in labels if not owners[label]]
        claimed = {owners[label] for label in labels if label not in self.adopted}
        orphans = [n for n in range(1, len(self.items) + 1) if n not in claimed]
        if not strays or len(orphans) != 1:
            return
        home = orphans[0]
        self.adopted.update(dict.fromkeys(strays, home))
        self.trace.append({'kind': 'adopted', 'item': home, 'labels': strays})

    def _g8_owner_score(self, words: str, long: set[str], item: Item) -> float:
        if words == _words(item.shelf or ''):
            return 3.0
        if words == _words(item.name):
            return 2.0
        mine = _roots(item.shelf or '') | _roots(item.name)
        return len(long & mine) / max(1, len(long | mine))

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        if label in self.adopted:
            return self.adopted[label]
        memo = (label, len(self.items))
        if self.owners_seen.get(memo) is not None:
            return self.owners_seen[memo]
        words, long = _words(label), _roots(label)
        scores = [self._g8_owner_score(words, long, item) for item in self.items]
        top = max(scores, default=0.0)
        best = scores.index(top) + 1 if top > 0 else 0
        self.owners_seen[memo] = best
        return best

    def owns(self, item: Item, label: str) -> bool:
        if not label or item not in self.items:
            return False
        if self.owner_of(label) == self.items.index(item) + 1:
            return True
        return _words(label) == _words(item.shelf or '')

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        if not cat_key:
            return 0
        return len([record for record in self.full.values() if self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)])

    def _g8_plan_menu(self, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        menu = []
        heads = [None] + [row[0] for row in table]
        for index, row, previous in zip(range(1, len(table) + 1), table, heads):
            label, name, carried, values = row
            if label != previous:
                menu.append('shelf {!r}:'.format(label))
            notes = [' | '.join(values), self.not_stated(label, name, values)]
            if re.search(CATCH_ALL, name):
                notes.append(' [bundles several features: put it last]')
            menu.append('  {}. {} (on {}): {}'.format(index, name, carried, ''.join(notes)))
        return menu

    def _g8_plan_prompt(self, count: int, table: list[tuple[str, str, int, list[str]]]) -> str:
        needs = []
        for n, item in zip(range(1, len(self.items) + 1), self.items):
            fixed = '; '.join(item.musts) or 'nothing'
            needs.append('item {}: {} - already fixed by the request: {}'.format(n, item.name, fixed))
        menu = self._g8_plan_menu(table)
        blocks = [PLAN.replace('COUNT', str(max(1, count))), 'THE REQUEST:\n' + self.request, 'THE ITEMS:\n' + '\n'.join(needs)]
        first = list(filter(str.strip, map(str, _items(self.brief.get('ask_first')))))
        if first:
            blocks.append('ASK FIRST (the request wants the shopper asked about these):\n' + '\n'.join(first))
        blocks.append('ATTRIBUTE KEYS ON OPENED LISTINGS:\n' + ('\n'.join(menu) or '(no listing has been opened in full yet)'))
        return '\n\n'.join(blocks)

    def _g8_read_pick(self, pick: object) -> tuple[int, str, bool] | None:
        if not isinstance(pick, dict):
            return None
        raw = str(pick.get('label') or '')
        label = re.split('\\s+(?:and|or)\\s+|[/,;(]', raw)[0].strip().rstrip('?.')
        if not label or len(label.split()) > 5:
            return None
        wanted = _count(pick.get('item') or 1) or 1
        return min(max(1, wanted), len(self.items)), label, bool(str(pick.get('hint') or '').strip())

    def _g8_plan_picks(self, plan: dict) -> tuple[dict[int, list[str]], dict[int, set[str]]]:
        numbers = range(1, len(self.items) + 1)
        lanes: dict[int, list[str]] = dict(zip(numbers, ([] for _ in numbers)))
        hinted: dict[int, set[str]] = dict(zip(numbers, (set() for _ in numbers)))
        reads = [read for read in map(self._g8_read_pick, _items(plan.get('ask'))) if read]
        for number, label, hint in reads:
            words = _words(label)
            lane = lanes[number]
            if words not in set(map(_words, lane)):
                lane.append(label)
            if hint:
                hinted[number].add(words)
                if number not in self.plan_order:
                    self.plan_order.append(number)
        return lanes, hinted

    def _g8_guard_place(self, lane: list[str], key: str, weak: int | None) -> list[str]:
        placed = [other for other in lane if _words(other) != _words(key)]
        if weak is None:
            placed.append(key)
        elif len(self.items) > 1:
            moved = placed[weak]
            placed[weak] = key
            placed.insert(min(3, len(placed)), moved)
        else:
            placed.insert(min(len(placed), weak), key)
        return placed

    def _g8_owned_shares(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> dict[str, float]:
        read = max(1, self.records_of_item(number))
        pairs = [(key, carried / read) for label, key, carried, _ in table if self.owner_of(label) == number]
        return {key: max([0.0] + [s for k, s in pairs if k == key]) for key in dict.fromkeys(k for k, _ in pairs)}

    def _g8_share(self, own: dict[str, float], label: str) -> float:
        exact = _words(label)
        if exact in own:
            return own[exact]
        stem = _stem(label)
        return max((s for k, s in own.items() if _names(stem, _stem(k)) or _names(_stem(k), stem)), default=0.0)

    def _g8_weak_slot(self, front: list[str], hinted: set[str], need: float, own: dict[str, float]) -> int | None:
        for i, other in enumerate(front):
            if _words(other) in hinted:
                continue
            if self._g8_share(own, other) + GUARD_GAP <= need:
                return i
        return None

    def _g8_build_lane(self, number: int, table: list[tuple[str, str, int, list[str]]], picks: list[str], hints: set[str]) -> list[str]:
        keys = self.lane_keys(number, table)
        own = self._g8_owned_shares(number, table)
        squash = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
        canon: dict[str, str] = {}
        for k in own:
            canon.setdefault(squash(k), k)
        spelled = lambda label: canon.get(squash(label), label)
        hinted = set(map(_words, map(spelled, hints)))
        firsts: dict[str, str] = {}
        for pick in map(spelled, picks):
            firsts.setdefault(_words(pick), pick)
        lane = list(firsts.values())
        guards = [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and not re.search(CATCH_ALL, k)][:GUARD_KEYS]
        for key in guards:
            front = lane[:3]
            shown = set(map(_words, front))
            if _words(key) in shown:
                continue
            weak = self._g8_weak_slot(front, hinted, own.get(key, 0.0), own)
            if weak is None and len(lane) >= 3:
                continue
            placed = self._g8_guard_place(lane, key, weak)
            if (shown & hinted) - set(map(_words, placed[:3])):
                continue
            lane = placed
        extra: dict[str, str] = {}
        for key in keys:
            extra.setdefault(_words(key), key)
        taken = set(map(_words, lane))
        lane += [key for words, key in extra.items() if words not in taken]
        return sorted(lane, key=lambda x: bool(re.search(CATCH_ALL, _words(x))))

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        self.adopt_labels()
        table = self.key_table()
        plan = self.ask_json(self._g8_plan_prompt(count, table), [HEAVY, LIGHT], 3500) or {}
        lanes, hinted = self._g8_plan_picks(plan)
        built = {number: self._g8_build_lane(number, table, picks, hinted[number]) for number, picks in lanes.items()}
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in built.items()}})
        return built

    def name_parts(self, number: int) -> list[str]:
        name = self.items[number - 1].name
        if not re.search('\\band\\b', name, re.I):
            return []
        parts = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()]
        if len(parts) < 2:
            return []
        plural = re.compile('[a-z]{3}(?<![su])s$', re.I)
        return parts if all(plural.search(p) for p in parts) else []

    def item_said(self, number: int) -> str:
        item = self.items[number - 1]
        parts = self.name_parts(number)
        if not parts:
            return item.name
        if number not in self.one_product:
            return re.sub('\\band\\b', 'or', item.name, flags=re.I)
        cat_key = self.category_key()
        named = []
        for row in self.rows.values():
            if cat_key and not self.owns(item, str(row.get(cat_key) or '')):
                continue
            named.append(_roots(' '.join(str(v) for k, v in row.items() if k != cat_key and isinstance(v, str))))
        counts = [len([words for words in named if _roots(p.split()[-1]) & words]) for p in parts]
        return parts[counts.index(max(counts))]

    def question(self, number: int, label: str, samples: list[str] | None=None) -> str:
        tail = f', for example {samples[0]} or {samples[1]}?' if samples else '?'
        return f'What {label} would you like for the {self.item_said(number)}' + tail

    def samples(self, number: int, label: str) -> list[str]:
        want = _words(label)
        settled = ' '.join(self.items[number - 1].musts).lower()
        values = next((values for _, name, _, values in self.key_table() if name == want), None)
        if values is None:
            return []
        return [v for v in values if v and v not in settled and len(v) <= 30][:2]

    def _g9_slot(self, raw: Any) -> int:
        # 0-based index of the item a model entry names, clamped into range
        wanted = _count(raw) or 1
        return min(max(1, wanted), len(self.items)) - 1

    def _g9_may_send(self, floor: int) -> bool:
        return not (self.finished or self.left() <= floor or self.msg_used >= self.msg_limit)

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
        out: list[tuple[int, str, str]] = []
        queue = list(questions)
        while queue and self._g9_may_send(floor):
            number, label, text = queue.pop(0)
            args = self.message_args(text)
            reply = self.act(tool, args) if args else None
            if reply is None:
                return out
            self.msg_used += 1
            answer = reply.get('observation')
            words = self.spoken(answer)
            out.append((number, text, words))
            mark = self.cart_mark()
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {mark})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'item': number, 'label': label, 'q': text, 'a': words[:300]})  # AK
            failed = isinstance(answer, dict) and answer.get('error')
            if failed:
                return out
        return out

    def _g9_heard_need(self, entry: dict[str, Any], told: list[str]) -> None:
        need = str(entry.get('need') or '').strip()
        if not need:
            return
        slot = self._g9_slot(entry.get('item'))
        need = str(entry['need']).strip()
        bucket = self.items[slot].told
        if need not in bucket:
            bucket.append(need)
            told.append(need if len(self.items) <= 1 else f'item {slot + 1}: {need}')
        pair = (slot + 1, need)
        if entry.get('points') is True and pair not in self.refs:
            self.refs.append(pair)

    def _g9_heard_drop(self, entry: dict[str, Any], told: list[str]) -> None:
        if not str(entry.get('property') or '').strip():
            return
        item = self.items[self._g9_slot(entry.get('item'))]
        gone = _words(str(entry['property']).split(':')[0])
        hit = list(filter(lambda m: m not in item.dropped and _words(m.split(':')[0]) == gone, item.musts + item.told))
        item.dropped.extend(hit)
        told.extend(map('dropped {}'.format, hit))

    def _g9_heard_free(self, entry: dict[str, Any], told: list[str]) -> None:
        if entry.get('property'):
            self.items[self._g9_slot(entry.get('item'))].free.append(str(entry['property']))

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        table = (('needs', self._g9_heard_need), ('drop', self._g9_heard_drop), ('free', self._g9_heard_free))
        for field, apply in table:
            for entry in filter(lambda e: isinstance(e, dict), _items(heard.get(field))):
                apply(entry, told)
        budget = heard.get('budget')
        if isinstance(budget, bool) or not isinstance(budget, (int, float)):
            return
        if heard['budget'] > 0:
            self.budget = heard['budget']

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT), named: int=1) -> dict[str, Any]:
        # a reply to a list naming 2+ labels is always read: it may ask for one question at a time in words ASK_WORDS lacks
        skip = named < 2 and self.quiet_reply(sent)
        if skip:
            self.trace.append({'kind': 'quiet_reply', 'a': sent[0][2][:80]})
            return {}
        self.hear_reads += 1
        roster = [f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)]
        pairs = []
        for i, (n, q, a) in enumerate(sent, 1):
            about = f'about item {n}' if n else 'about several items'
            pairs.append(f'{i}. ({about}) Q: {q}\nA: {a}')
        prompt = ''.join((HEAR, '\n\nITEMS:\n', '\n'.join(roster), '\n\nQUESTIONS AND ANSWERS:\n', '\n'.join(pairs)))
        return self.ask_json(prompt, list(models), 2500) or {}

    @staticmethod
    def _g9_alnum(text: str) -> frozenset[str]:
        return _spelled(' '.join(re.findall('[A-Za-z0-9]+', text)))

    def _g9_known_words(self) -> set[str]:
        pool: set[str] = set()
        for item in self.items:
            pool.update(_spelled(item.name), _spelled(item.shelf or ''))
            pool.update(*(self._g9_alnum(need.split(':', 1)[0]) for need in item.musts + item.told))
        for label, name, _, _ in self.key_table():
            pool.update(_spelled(name), _spelled(label))
        facts = (v for record in self.full.values() for v in (_fact_block(record) or {}).values())
        pool.update(w for v in facts for w in self._g9_alnum(str(v)) if len(w) >= QUIET_WORD)
        return pool

    @_soft(False)
    def quiet_reply(self, sent: list[tuple[int, str, str]]) -> bool:
        if self.hear_reads < 1:
            return False
        known = self._g9_known_words()

        def talks(question: str, answer: str) -> bool:
            if re.search('\\d', answer):
                return True
            asked = re.sub('\\b(?:what|which|would|you|like|for|the|please|tell|me|and|or)\\b', ' ', question, flags=re.I)
            if self._g9_alnum(answer) & (known | self._g9_alnum(asked)):
                return True
            return bool(re.search(ASK_WORDS, answer, re.I))
        return not any(talks(q, a) for _, q, a in sent)

    def _g9_gate_prompt(self, stage: str, steps: list[dict], tools: list[str]) -> str:
        buying = stage == 'order'
        record = '\n'.join(f'{i}. {line}' for i, line in enumerate(self.done_log, 1)) or '(nothing yet)'
        parts = [GATE.replace('STAGE', 'place the order' if buying else 'end without a purchase'),
                 '\n\nREQUEST:\n', self.request,
                 '\n\nSTEPS DUE NOW:\n', '\n'.join(f"- {p['step']}" for p in steps),
                 '\n\nLEDGER:\n', self.ledger(),
                 '\n\nCART NOW: ', self.cart_text(),
                 '\n\nRECORD (messages and step calls so far):\n', record]
        if self.notes:
            parts.append('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]))
        parts.append(f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {('order' if buying else 'ending')} itself takes one)")
        parts.append('\n\nTOOLS:\n' + '\n'.join(tools))
        return ''.join(parts)

    @staticmethod
    def _g9_did(said: list[str]) -> str:
        return 'Before carrying on, Cairn first ran the steps the request calls for:\n' + '\n'.join(said)

    def _g9_unproven(self, stage: str, steps: list[dict], got: dict) -> list[dict]:
        talk = stage == 'order' and any(p['messages'] > 0 for p in steps)
        proof = set(range(1, len(self.done_log) + 1))
        if talk:
            proof = {i for i in proof if i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at}
        shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]

        def backed(e: dict) -> bool:
            return e.get('done') is True and any((_count(n) or 0) in proof for n in _items(e.get('lines')))
        return [e for e in shown if not backed(e)]

    def _g9_gate_decide(self, stage: str, steps: list[dict], got: dict, said: list[str], nudged: bool, ending: set) -> tuple[str, Any]:
        told: list[str] = []
        hold = ''
        if said:
            self.take_heard({'needs': got.get('needs')}, told)
            hold = str(got.get('hold') or '').strip()
        call = got['call'] if isinstance(got.get('call'), dict) else None
        self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
        if told or hold:
            return ('leave', told + [hold] if hold else told)
        unproven = self._g9_unproven(stage, steps, got)
        ask = self.roles.get('ask')
        if not call:
            if unproven and not said and not nudged and ask and self.msg_used < self.msg_limit and self.left() >= 2:
                return ('nudge', unproven)
            self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
            return ('halt', None)
        name = str(call.get('tool') or '')
        if name not in self.schemas or name in ending:
            return ('halt', None)
        if self.left() < 2 or (name == ask and self.msg_used >= self.msg_limit):
            return ('halt', None)
        faults: list[str] = []
        raw = call.get('args')
        args = _shape(raw if isinstance(raw, dict) else {}, self.schemas[name], 'args', faults)
        if faults:
            return ('halt', None)
        return ('ask' if name == ask else 'call', (name, args))

    def _g9_gate_nudge(self, unproven: list[dict]) -> bool:
        self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
        names = '; '.join(str(e.get('step') or '') for e in unproven)[:200]
        self.done_log.append(f'(Cairn: since the cart last changed at step {self.cart_at} no line shows {names} done, while {self.msg_limit - self.msg_used} message(s) and {self.left()} steps remain: give the call that does it)')
        self.done_steps.append(-1)
        return True

    def _g9_gate_ask(self, name: str, args: dict, said: list[str]) -> bool:
        text = max(_strings(args), key=len, default='')
        batch = self.send([(1, '', text)], 1) if text else []
        if not batch:
            return False
        self.gate_msgs += 1
        said.append(f'Q: {text}\nA: {batch[0][2]}')
        return True

    def _g9_gate_call(self, name: str, args: dict, said: list[str]) -> bool:
        reply = self.act(name, args)
        if reply is None:
            return False
        seen = reply.get('observation')
        self.absorb(seen)
        shown = self.show(seen)
        self.done_log.append(f'step {self.step}: {name} {_dump(args)} -> {shown[:300]} (cart then: {self.cart_mark()})')
        self.done_steps.append(self.step)
        said.append(f'{name} {_dump(args)} -> {self.show(seen)[:400]}')
        return True

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or not self.gate_due(stage):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = []
        for t in self.tools:
            fn = t['function']
            if fn['name'] in ending:
                continue
            tools.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        said: list[str] = []
        nudged = False
        rounds = 0
        while rounds < GATE_ROUNDS:
            rounds += 1
            prompt = self._g9_gate_prompt(stage, steps, tools)
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            verb, load = self._g9_gate_decide(stage, steps, got, said, nudged, ending)
            if verb == 'leave':
                self.version += 1
                head = self._g9_did(said) + '\n' if said else ''
                return (False, head + "The shopper's reply changes the task: " + '; '.join(load) + '. Fix the cart for it, then check out again.')
            if verb == 'halt':
                break
            if verb == 'nudge':
                nudged = self._g9_gate_nudge(load)
                continue
            runner = self._g9_gate_ask if verb == 'ask' else self._g9_gate_call
            if not runner(*load, said):
                break
        return (True, self._g9_did(said) if said else '')

    def spoken(self, observation: Any) -> str:
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
                continue
            kids = node.values() if isinstance(node, dict) else node if isinstance(node, list) else ()
            stack.extend((kid, depth + 1) for kid in reversed(list(kids)))
        if not found:
            return _dump(observation)[:400]
        return ' '.join(found)[:800]

    def _g9_loose_of(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if len(value.split()) >= 5 else []
        if isinstance(value, list):
            if value and all(isinstance(v, str) for v in value):
                return [v for v in value if len(v.split()) >= 5]
            return []
        if isinstance(value, dict) and not self.ident(value, False, 1):
            return [inner for inner in value.values() if isinstance(inner, str) and len(inner.split()) >= 6]
        return []

    def loose_lines(self, observation: Any) -> list[str]:
        if not isinstance(observation, dict) or self.ident(observation):
            return []
        lines = [line for value in observation.values() for line in self._g9_loose_of(value)]
        return [line for line in lines if line not in self.heard]

    def _g9_note_budget(self, got: dict, changed: list[str]) -> bool:
        budget = got.get('budget')
        if isinstance(budget, bool) or not isinstance(budget, (int, float)):
            return False
        if budget <= 0 or budget == self.budget:
            return False
        self.budget = budget
        changed.append(f'budget now {budget}')
        return True

    def _g9_note_drop(self, entries: list[dict], changed: list[str]) -> bool:
        for entry in entries:
            item = self.items[self._g9_slot(entry.get('item'))]
            target = str(entry['need']).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not hit:
                key = _words(target.split(':')[0])
                hit = next(([m] for m in live if _words(m.split(':')[0]) == key), [])
            item.dropped.extend(hit if hit else [target])
            changed.append(f'dropped {target}')
        return bool(entries)

    def _g9_note_add(self, entries: list[dict], changed: list[str]) -> bool:
        for entry in entries:
            item = self.items[self._g9_slot(entry.get('item'))]
            need = str(entry['need']).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            changed.append(f'added {need}')
        return False

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = '\n\n'.join((NOTE, 'ITEMS AND REQUIREMENTS NOW:\n' + self.ledger(), 'THE NOTE:\n' + '\n'.join(lines)))
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        gone = [str(g) for g in _items(got.get('gone')) if isinstance(g, (str, int))]
        listed = [g for g in gone if g in self.rows]
        self.unavailable.update(listed)
        self.note_sold = bool(gone) or got.get('sold_out') is True
        self.note_named = bool(listed)
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed: list[str] = []

        def wanted(field: str) -> list[dict]:
            return [e for e in _items(got.get(field)) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        table = (('budget', lambda: self._g9_note_budget(got, changed)), ('drop', lambda: self._g9_note_drop(drops, changed)), ('add', lambda: self._g9_note_add(adds, changed)))
        drops, adds = wanted('drop'), wanted('add')
        widened = [apply() for _, apply in table]
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(got.get('summary') or '').strip()
        if self.brief.get('pick') == 'cheapest' and (any(widened) or gone or self.note_sold):
            self.wider_held = False
            self.cheaper_marks = set()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or '; '.join(changed))
        tail = ' (' + '; '.join(changed) + ')' if changed else ''
        return (summary + tail).strip()

    def needs_now(self) -> list[tuple[int, str]]:
        return [(number, need) for number, item in enumerate(self.items, 1) for need in item.musts + item.told if need not in item.dropped]

    def _g9_listed_ids(self, observation: Any) -> list[str]:
        ids: list[str] = []
        stack: list[tuple[Any, int]] = [(observation, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > 6:
                continue
            if isinstance(node, dict):
                ident = self.ident(node, False, depth)
                if ident and _fact_block(node) and ident not in ids:
                    ids.append(ident)
                    continue
                kids = list(node.values())
            elif isinstance(node, list):
                kids = node
            else:
                continue
            stack.extend((kid, depth + 1) for kid in reversed(kids))
        return ids

    def _g9_matches(self, ident: str, needs: list[tuple[int, str]], tag_of: Any) -> list[str]:
        block = _fact_block(self.full.get(ident) or {}) or {}
        parts = []
        for number, need in needs:
            prop = need.split(':', 1)[0].strip()
            stem = _stem(prop)
            hits = [f'{k}={str(v)[:80]}' for k, v in block.items() if _names(stem, _stem(k))]
            if hits:
                parts.append(tag_of(number) + f'{prop} -> ' + ', '.join(hits[:3]))
        return parts

    def evidence(self, observation: Any) -> str:
        ids = self._g9_listed_ids(observation)
        needs = self.needs_now()
        if not (ids and needs):
            return ''
        tag_of = (lambda n: f'(item {n}) ') if len(self.items) > 1 else (lambda n: '')
        found = ((ident, self._g9_matches(ident, needs[:8], tag_of)) for ident in ids[:12])
        lines = [f'{ident}: ' + '; '.join(parts) for ident, parts in found if parts]
        if not lines:
            return ''
        return "[evidence: listing keys that mention a requirement's property name. A key about some other part or aspect proves nothing either way; a key under another name can still hold that property] " + ' | '.join(lines)

    def _g9_view_fresh(self) -> bool:
        return self.cart_view is not None and self.cart_seen_at >= self.cart_at

    def cart_text(self) -> str:
        if not self._g9_view_fresh():
            return "cart as Cairn's own log of cart calls has it (listing: units): " + _dump(self.cart_lines)
        return _dump(_clip(self.cart_view, 300))[:3000]

    def _g9_view_dicts(self):
        # cart view dicts in last-in-first-out order
        stack: list[Any] = [self.cart_view]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                yield node
                stack.extend(v for v in node.values() if isinstance(v, (dict, list)))

    def held_units(self, ident: str) -> int:
        if not self._g9_view_fresh():
            return self.cart_lines.get(ident, 1)
        unit_key = re.compile('\\b(?:qty|quantity|count|units?)\\b')
        for node in self._g9_view_dicts():
            if ident not in {v.strip() for v in node.values() if isinstance(v, str)}:
                continue
            counts = [_count(v) for k, v in node.items() if not isinstance(v, str) and _count(v) is not None and unit_key.search(_words(k))]
            if len(counts) == 1:
                return counts[0]
        return self.cart_lines.get(ident, 1)

    def cart_ids(self) -> list[str]:
        if not self._g9_view_fresh():
            return list(self.cart_lines)
        text = _dump(self.cart_view)
        return list(filter(text.__contains__, self.rows))

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        if not record:
            return '(no record seen)'
        return _dump(_clip(record, 400))[:3000]

    def others(self, skip: list[str], limit: int=30) -> str:
        order = self.seen_order[::-1]
        if self.brief.get('pick') == 'cheapest':
            def rank(i: str) -> tuple[bool, float]:
                price = self.price_of(i)
                return (i not in self.full, float('inf') if price is None else price)
            order.sort(key=rank)
        picked: list[str] = []
        pending = iter(order)
        while len(picked) < limit or not picked:
            ident = next(pending, None)
            if ident is None:
                break
            if ident in skip or self.sold_out(ident):
                continue
            picked.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 160))[:900])
        return '\n'.join(picked)

    def fit_index(self) -> dict[str, Any]:
        stamp = (len(self.seen_order), len(self.full), sum(map(len, self.views.values())))
        cached = self.fit_cached
        if not (cached and cached[0] == stamp):
            self.fit_cached = (stamp, self._g10_build_index())
        return self.fit_cached[1]

    def _g10_build_index(self) -> dict[str, Any]:
        texts = {ident: self._g10_listing_text(self.full.get(ident) or self.rows.get(ident) or {}) for ident in self.seen_order}
        marks: dict[str, dict[str, Any]] = {ident: _fit_marks(text) for ident, text in texts.items()}
        notes: dict[str, set[str]] = {}
        for part, codes in (pair for mark in marks.values() for pair in mark['also']):
            notes.setdefault(part, set()).update(codes)
        series_by_code: dict[str, set[str]] = {}
        for text in texts.values():
            self._g10_collect_series(text, series_by_code)
        widened = ((mark, set().union(*(series_by_code.get(code, set()) for code in mark.get('codes') or ()))) for mark in marks.values())
        for mark, borrowed in filter(lambda mb: mb[1], widened):
            mark['named'] = sorted(set(mark.get('named') or []) | borrowed)
        return {'marks': marks, 'notes': notes}

    @staticmethod
    def _g10_listing_text(record: dict[str, Any]) -> str:
        flat = [str(v) for v in record.values() if isinstance(v, str)]
        nested = [f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)]
        return ' '.join(flat) + ' ' + '. '.join(nested)

    @staticmethod
    def _g10_collect_series(text: str, into: dict[str, set[str]]) -> None:
        for hit in re.finditer(MODEL_ALSO, text, re.I):
            series = {s.upper() for s in re.findall('[A-Za-z]{1,5}\\d{1,3}', hit.group(2))}
            for code in _codes(hit.group(1)):
                into.setdefault(code, set()).update(series)

    def fit_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        own = mark.get('own')
        shared = index['notes'].get(own, set()) if own else set()
        return set(mark.get('works') or []) | shared

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.items[number - 1]
        if not cat_key:
            return []
        label_of = lambda i: str((self.rows.get(i) or {}).get(cat_key) or '')
        return list(filter(lambda i: self.owns(item, label_of(i)), self.seen_order))

    @staticmethod
    def _g10_tag_list(marks: dict[str, Any], ident: str, key: str) -> list[str]:
        return (marks.get(ident) or {}).get(key) or []

    def _g10_family_pool(self, number: int, marks: dict[str, Any]) -> set[str] | None:
        rows = self.item_ids(number)
        tagged = [i for i in rows if self._g10_tag_list(marks, i, 'fam')]
        families = {f for i in tagged for f in marks[i]['fam']}
        thin = len(tagged) < LINK_ROWS or 2 * len(tagged) < len(rows)
        return None if thin or len(families) > LINK_FAMILIES else families

    def link_system(self, part: int, main: int, index: dict[str, Any]) -> tuple[int, int, set[str]] | None:
        marks = index['marks']
        for carrier, namer in ((part, main), (main, part)):
            families = self._g10_family_pool(carrier, marks)
            if families is None:
                continue
            named = (f for i in self.item_ids(namer) for f in self._g10_tag_list(marks, i, 'named'))
            if any(map(families.__contains__, named)):
                return (carrier, namer, families)
        return None

    def link_holds(self, carried: str, namer: str, families: set[str], index: dict[str, Any]) -> bool | None:
        marks = index['marks']
        own = families.intersection(self._g10_tag_list(marks, carried, 'fam'))
        named = families.intersection(self._g10_tag_list(marks, namer, 'named'))
        if not (own and named):
            return None
        return not own.isdisjoint(named)

    def link_proven(self, carried: str, namer: str, families: set[str], index: dict[str, Any]) -> bool | None:
        verdict = self.link_holds(carried, namer, families, index)
        if verdict is not None:
            return verdict
        marks = index['marks']
        wants = families.intersection(self._g10_tag_list(marks, namer, 'named'))
        bare_open = wants and not self._g10_tag_list(marks, carried, 'fam') and carried in self.full
        return False if bare_open else None

    def _g10_link_note(self, carried: str, namer: str, carrier: int, main: int, families: set[str], marks: dict[str, Any]) -> str:
        wanted = set(marks[namer].get('named') or []) & families
        others = [i for i in self.item_ids(carrier) if set(marks[i].get('fam') or []) & wanted and not self.sold_out(i) and i != carried][:4]
        mine = sorted(set(marks[carried].get('fam') or []) & families)
        opening = f"{carried} belongs to series {', '.join(mine)}" if mine else f"{carried} states no part series, so nothing shows it works"
        middle = f", while {namer} takes {', '.join(sorted(wanted))}"
        if others:
            return opening + middle + f"; listings seen in that series: {', '.join(others)}"
        return opening + middle + f"; try a search for that series plus words from the item, or swap to a different {self.items[main - 1].name} carrying a series the wanted lines accept"

    def link_faults(self, part: int, main: int, device: str, lines: list[str], index: dict[str, Any]) -> list[str]:
        system = self.link_system(part, main, index)
        if not system:
            return []
        carrier, _, families = system
        orient = (lambda line: (line, device)) if carrier == part else (lambda line: (device, line))
        pairs = map(orient, lines)
        return [self._g10_link_note(carried, namer, carrier, main, families, index['marks']) for carried, namer in pairs if self.link_proven(carried, namer, families, index) is False]

    def series_for(self, number: int) -> list[str]:
        index = self.fit_index()
        marks = index['marks']
        cart = self.cart_ids()
        hits: list[tuple[str, str]] = []
        for part, main in filter(lambda pair: number in pair, self.bound):
            system = self.link_system(part, main, index)
            if not system:
                continue
            hits += [(i, family) for member in (part, main) for i in self.item_ids(member) for family in sorted((set(self._g10_tag_list(marks, i, 'fam')) | set(self._g10_tag_list(marks, i, 'named'))) & system[2])]
        counts: dict[str, int] = {}
        for _, family in hits:
            counts[family] = counts.get(family, 0) + 1
        held = list(dict.fromkeys(family for i, family in hits if i in cart))
        rest = [f for f in sorted(counts, key=lambda f: (-counts[f], f)) if f not in held]
        return (held + rest)[:LINK_SEARCHES]

    @staticmethod
    def _g10_any_hit(pool: Any, codes: Any) -> bool:
        return any(_code_hit(p, c) for p in pool for c in codes)

    def _g10_covers(self, ident: str, codes: set[str], index: dict[str, Any]) -> bool:
        bars = set(index['marks'].get(ident, {}).get('bars') or [])
        return self._g10_any_hit(self.fit_set(ident, index), codes) and not self._g10_any_hit(bars, codes)

    def _g10_pair_faults(self, part: int, main: int, cart: list[str], index: dict[str, Any]) -> list[str]:
        devices = [i for i in cart if i in self.item_ids(main)]
        parts = [i for i in cart if i in self.item_ids(part)]
        if len(devices) != 1 or not parts:
            return []
        device = devices[0]
        linked = self.link_faults(part, main, device, parts, index)
        if linked:
            return linked
        codes = set(index['marks'].get(device, {}).get('codes') or [])
        if not codes:
            return []
        flagged = ((line, fits) for line in parts for fits in [self.fit_set(line, index)] if fits and not self._g10_covers(line, codes, index))
        return [self._g10_fit_note(line, fits, part, device, codes, cart, index) for line, fits in flagged]

    def _g10_fit_note(self, line: str, fits: set[str], part: int, device: str, codes: set[str], cart: list[str], index: dict[str, Any]) -> str:
        good = [i for i in self.item_ids(part) if self._g10_covers(i, codes, index) and not self.sold_out(i) and i not in cart][:4]
        head = f"{line} ({self.items[part - 1].name}) fits {', '.join(sorted(fits)[:4])}, but not {device} ({', '.join(sorted(codes))})"
        if good:
            return head + f"; seen parts of this kind that suit it: {', '.join(good)}"
        return head + '; none of the seen parts of this kind suits it, so search for one or change the device: ' + self.fit_plan(short=True)

    def fit_faults(self) -> list[str]:
        if not self.bound:
            return []
        index = self.fit_index()
        cart = self.cart_ids()
        return [fault for part, main in self.bound for fault in self._g10_pair_faults(part, main, cart, index)]

    def _g10_code_tally(self, main: int, parts: list[int], index: dict[str, Any]) -> dict[str, dict[int, int]]:
        marks = index['marks']
        steps = [(code, p) for p in parts for i in self.item_ids(p) if not self.sold_out(i) for code in sorted(self.fit_set(i, index))]
        systems = [(p, self.link_system(p, main, index)) for p in parts]
        steps += [(code, p) for p, system in systems if system and system[0] == p for i in self.item_ids(p) if not self.sold_out(i) for code in sorted(set(self._g10_tag_list(marks, i, 'fam')) & system[2])]
        tally: dict[str, dict[int, int]] = {}
        for code, p in steps:
            per = tally.setdefault(code, {})
            per[p] = per.get(p, 0) + 1
        return tally

    def _g10_plan_lines(self, main: int, index: dict[str, Any]) -> list[str]:
        marks = index['marks']
        parts = [p for p, m in self.bound if m == main]
        tally = self._g10_code_tally(main, parts, index)
        ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:6]
        named_by = lambda code, i: code in self._g10_tag_list(marks, i, 'named') or any(_code_hit(code, c) for c in marks.get(i, {}).get('codes') or [])
        seen_devices = {code: [i for i in self.item_ids(main) if named_by(code, i)][:2] for code, _ in ranked}
        pieces = []
        for code, per in ranked:
            counted = ', '.join(f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))
            where = f" (device seen: {', '.join(seen_devices[code])})" if seen_devices[code] else ' (device not seen yet)'
            pieces.append(f'{code}: {counted}{where}')
        out = [f"codes named by the parts of the {self.items[main - 1].name} candidates -> {'; '.join(pieces)}"] if pieces else []
        sets = self.cheapest_sets(main, parts, index) if self.brief.get('pick') == 'cheapest' else ''
        return out + [sets] if sets else out

    def fit_plan(self, short: bool=False) -> str:
        if not self.bound:
            return ''
        index = self.fit_index()
        lines = [line for main in sorted({m for _, m in self.bound}) for line in self._g10_plan_lines(main, index)]
        if not lines:
            return ''
        body = ' | '.join(lines)
        if short:
            return body
        return "[fit plan: device and part match when the part's listing quotes the device's model code (or an alias it is sold under), or the device's listing quotes the series that the part's code belongs to; do a single search for a code or series combined with words from the item, and when that returns nothing, browse the device's category and check the codes shown on its rows] " + body

    def _g10_first_fitting(self, shelf: list[tuple], codes: list[str], system: Any, part: int, device: str, index: dict[str, Any]) -> str | None:
        linked = bool(system) and system[0] == part
        for _, ident, fits, bars in shelf:
            coded = self._g10_any_hit(fits, codes) and not self._g10_any_hit(bars, codes)
            if coded or (linked and self.link_holds(ident, device, system[2], index) is True):
                return ident
        return None

    def _g10_device_set(self, device: str, main: int, parts: list[int], systems: dict, shelves: dict, index: dict[str, Any]) -> tuple[float, list[str]] | None:
        mark = index['marks'].get(device, {})
        codes = mark.get('codes') or []
        price = self.price_of(device)
        if not (codes or mark.get('named')) or price is None or not self.buyable(device):
            return None
        total = price * self.items[main - 1].qty
        picks = [device]
        for part in parts:
            chosen = self._g10_first_fitting(shelves[part], codes, systems[part], part, device, index)
            if not chosen:
                return None
            picks.append(chosen)
            total += (self.price_of(chosen) or 0.0) * self.items[part - 1].qty
        return (total, picks)

    @_soft('')
    def cheapest_sets(self, main: int, parts: list[int], index: dict[str, Any]) -> str:
        systems = {part: self.link_system(part, main, index) for part in parts}
        shelves: dict[int, list[tuple]] = {}
        for part in parts:
            stock = filter(lambda i: self.price_of(i) is not None and self.buyable(i), self.item_ids(part))
            shelf = [(self.price_of(i), i, self.fit_set(i, index), index['marks'].get(i, {}).get('bars') or []) for i in stock]
            shelf.sort(key=lambda o: (o[0], o[1]))
            shelves[part] = shelf
        options = []
        for device in self.item_ids(main):
            whole = self._g10_device_set(device, main, parts, systems, shelves, index)
            if whole:
                options.append(whole)
        if not options:
            return ''
        options.sort(key=lambda o: o[0])
        listed = '; '.join(f"{' + '.join(p)} = {t:g}" for t, p in options[:SET_SHOWN])
        return 'cheapest complete sets seen so far, requirements still unchecked: ' + listed

    def missing_items(self) -> list[int]:
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not cat_key or len(ids) >= len(self.items):
            return []
        label_of = lambda row: str(row.get(cat_key) or '')
        labels = [label_of(r) for r in self.rows.values()]
        held = [label_of(self.rows.get(i) or {}) for i in ids]
        owns_any = lambda item, pool: any(self.owns(item, label) for label in pool)
        return [n for n, item in enumerate(self.items, 1) if owns_any(item, labels) and not owns_any(item, held)]

    @staticmethod
    def _g10_word_set(text: str) -> set[str]:
        return set(re.sub('[,/;|]', ' ', _words(text)).split())

    def _g10_need_stated(self, need: str, block: dict[str, Any]) -> bool:
        prop, _, value = need.partition(':')
        words = self._g10_word_set(value)
        if not words or words & {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'}:
            return False
        return any(_names(_spelled(prop), _spelled(k)) and words <= self._g10_word_set(str(v)) for k, v in block.items())

    def states_needs(self, ident: str, number: int) -> bool:
        block = _fact_block(self.full.get(ident) or {}) or {}
        return all(self._g10_need_stated(need, block) for owner, need in self.needs_now() if owner == number)

    def _g10_push_lines(self, tool: str, calls: list[dict[str, Any]], settle: bool) -> bool:
        for raw in calls:
            faults: list[str] = []
            args = _shape(raw, self.schemas.get(tool), 'args', faults)
            reply = None if faults else self.act(tool, args)
            if reply is None or (settle and self.finished):
                return False
            self.note_cart(tool, args)
            if settle:
                self.observe(tool, reply.get('observation'))
        return True

    @staticmethod
    def _g10_key_number(key: Any) -> int:
        found = re.search('\\d+', str(key))
        return int(found.group()) if found else 0

    def _g10_review_pool(self, number: int, ids: list[str]) -> list[str]:
        pool = [i for i in ids if self.buyable(i) and self.price_of(i) is not None and i in self.item_ids(number)]
        code = self.car_binds().get(number)  # CF5
        if not code:
            return pool
        index = self.fit_index()
        return [i for i in pool if self.car_fit(i, code, index) == 'fits'] or pool

    def _g10_salvage_lists(self, numbers: list[int], held: dict[int, list[str]], got: Any) -> dict[int, list[str]]:
        numbered = ((self._g10_key_number(key), ids) for key, ids in self.audit_fits(got).items())
        lists: dict[int, list[str]] = {number: self._g10_review_pool(number, ids) for number, ids in numbered if number in numbers}
        for n in filter(held.get, numbers):
            lists[n] = held[n][:1] + [i for i in lists.get(n, []) if i not in held[n]]
        return lists

    def _g10_pair_fits(self, part_id: str, device_id: str, part: int, main: int, index: dict[str, Any]) -> bool:
        system = self.link_system(part, main, index)
        if system:
            ends = (part_id, device_id) if system[0] == part else (device_id, part_id)
            verdict = self.link_proven(ends[0], ends[1], system[2], index)  # FP
            if verdict is not None:
                return verdict
        codes = (index['marks'].get(device_id) or {}).get('codes') or []
        found = self.fit_set(part_id, index)
        return self._g10_any_hit(found, codes) if found and codes else True

    def _g10_salvage_choice(self, numbers: list[int], lists: dict[int, list[str]]) -> tuple[float, dict[int, str]] | None:
        index = self.fit_index()
        combos: list[list[str]] = [[]]
        for pool in (lists[n][:SALVAGE_EACH] for n in numbers):
            combos = [c + [i] for c in combos for i in pool if i not in c][:4096]
        capped = isinstance(self.budget, (int, float)) and self.brief.get('budget_scope') != 'each'

        def viable():
            for combo in combos:
                pick = dict(zip(numbers, combo))
                if not all(self._g10_pair_fits(pick[part], pick[main], part, main, index) for part, main in self.bound):
                    continue
                total = sum(((self.price_of(pick[n]) or 0.0) * self.items[n - 1].qty for n in numbers))
                if not (capped and total > self.budget + 1e-09):
                    yield (total, pick)
        if self.brief.get('pick') == 'cheapest':
            return min(viable(), key=lambda t: t[0], default=None)
        return next(viable(), None)

    @_soft(False)
    def salvage_set(self) -> bool:
        state_blocks = (lambda: self.salvaged, lambda: self.finished, lambda: self.broke, lambda: len(self.items) < 2, lambda: self.model_room() < SALVAGE_ROOM)
        if any(rule() for rule in state_blocks):
            return False
        cart_tool, buy = self.roles.get('cart'), self.roles.get('buy')
        numbers = list(range(1, len(self.items) + 1))
        room_blocks = (lambda: not (cart_tool and buy), lambda: self.left() < len(self.items) + self.tail() + 1, lambda: self.cost > SPEND_HARD + 0.02)
        if any(rule() for rule in room_blocks):
            return False
        held = {n: [i for i in self.cart_ids() if i in self.item_ids(n)] for n in numbers}
        if all(map(held.get, numbers)):
            return False
        # plan: one review call, then the best whole set
        self.salvaged = True
        self.call_cap = max(self.call_cap, self.calls + 2)  # SP3
        lists = self._g10_salvage_lists(numbers, held, self.audit_call([SPARE, LIGHT]))  # SP4
        self.trace.append({'kind': 'salvage_review', 'lists': lists})
        if not all(lists.get(n) for n in numbers):
            return False
        best = self._g10_salvage_choice(numbers, lists)
        if not best:
            self.trace.append({'kind': 'salvage_none'})
            return False
        # calls: replace the cart with the chosen lines
        lines = {best[1][n]: self.items[n - 1].qty for n in numbers}
        calls = self.cart_calls(lines, [i for i in self.cart_ids() if i not in lines])
        if not calls or len(calls) + self.tail() > self.left():
            return False
        # apply
        if not self._g10_push_lines(cart_tool, calls, True):
            return False
        self.trace.append({'kind': 'salvage_set', 'lines': lines, 'total': best[0]})
        return True

    def _g10_fill_pick(self, number: int, bound: set[int]) -> str | None:
        item = self.items[number - 1]
        ok = lambda i: i in self.full and i not in self.cart_lines and self.buyable(i) and self.price_of(i) is not None and self.states_needs(i, number)
        ranked = sorted(filter(ok, self.item_ids(number)), key=lambda i: self.price_of(i) or 0.0)
        if not ranked or number in bound:
            return None
        pick = ranked[0]
        if not isinstance(self.budget, (int, float)):
            return pick
        each = self.brief.get('budget_scope') == 'each'
        held = sum(((self.price_of(i) or 0.0) * (self.held_units(i) or 0) for i in self.cart_ids()))
        cost = (self.price_of(pick) or 0.0) * (1 if each else item.qty)
        return None if cost + (0.0 if each else held) > self.budget + 1e-09 else pick

    def fill_short_cart(self) -> bool:
        cart = self.roles.get('cart') or ''
        bound = {n for pair in self.bound for n in pair}
        for number in self.missing_items():
            # plan
            pick = self._g10_fill_pick(number, bound)
            kept = {i: self.cart_lines[i] for i in self.cart_ids() if i in self.cart_lines}
            # calls
            calls = self.cart_calls({**kept, pick: self.items[number - 1].qty}, []) if pick else None
            if not calls or len(calls) + 1 > self.left():
                return False
            # apply
            if not self._g10_push_lines(cart, calls, False):
                return False
            self.trace.append({'kind': 'filled_line', 'item': number, 'line': pick})
        return True

    def _g10_parted_shelf(self, item: Any, labels: list[str]) -> str:
        name = _roots(item.name)
        others = [o for o in self.items if o is not item]
        if not name or not item.shelf.strip() or not any(_words(o.shelf) == _words(item.shelf) for o in others):
            return ''
        if any(_roots(o.name) == name for o in others):
            return ''
        own = [label for label in labels if _roots(label) == name and _words(label) != _words(item.shelf)]
        if len(own) != 1 or any(_words(o.shelf) == _words(own[0]) for o in self.items):
            return ''
        return own[0]

    def part_shelves(self) -> None:
        cat_key = self.category_key()
        labels = sorted({str(row.get(cat_key) or '').strip() for row in self.rows.values()} - {''}) if cat_key else []
        moved = False
        for item in self.items:
            target = self._g10_parted_shelf(item, labels)
            if not target:
                continue
            self.trace.append({'kind': 'shelf_parted', 'item': item.name, 'from': item.shelf, 'to': target})
            item.shelf = target
            moved = True
        if moved:
            self.owners_seen = {}
            self.fit_cached = None

    def fuller_label(self, number: int, label: str) -> str:
        cat_key = self.category_key()
        item = self.items[number - 1]
        mine = _roots(item.name)
        blocked = (len(mine) < 2 or mine <= _roots(label) or not cat_key or (item.shelf and _roots(label) == _roots(item.shelf)))
        if blocked:
            return ''
        seen = sorted({str(r.get(cat_key) or '') for r in self.rows.values()})
        return next(filter(lambda s: s and mine <= _roots(s), seen), '')

    def named_label(self, number: int, label: str) -> str:
        cat_key = self.category_key()
        wanted = _words(self.items[number - 1].shelf or '')
        if not wanted or not cat_key or _words(label) == wanted:
            return ''
        values = (r.get(cat_key) for r in self.rows.values())
        return next((str(v) for v in values if _words(str(v or '')) == wanted), '')

    def line_faults(self) -> list[str]:
        self.adopt_labels()
        cat_key = self.category_key()
        held = list(filter(lambda i: (self.held_units(i) or 0) > 0, self.cart_ids()))
        if not (held and cat_key):
            return []
        kind_of = dict(((i, str((self.rows.get(i) or {}).get(cat_key) or '')) for i in held))
        numbered = list(enumerate(self.items, 1))
        stocked: set[int] = set()
        for n, item in numbered:
            if next((True for r in self.rows.values() if self.owns(item, str(r.get(cat_key) or ''))), False):
                stocked.add(n)
        claims = {i: [n for n, item in numbered if self.owns(item, kind_of[i])] for i in held}
        report: list[str] = []
        for n, item in numbered:
            if n in stocked:
                report.extend(self._g11_item_faults(n, item, held, kind_of, claims))
        if len(stocked) == len(self.items):
            report.extend(f"{i} ({kind_of[i] or 'unknown kind'}) is no kind the shopper wants: remove it" for i in held if not claims[i])
        report.extend(f'{i} is sold out' for i in held if self.sold_out(i))
        report.extend(self._g11_budget_faults(held))
        return report

    def _g11_item_faults(self, n: int, item: Any, held: list[str], kind_of: dict[str, str], claims: dict[str, list[int]]) -> list[str]:
        tag = f'item {n} ({item.name})'
        lines = [i for i in held if n in claims[i]]
        found: list[str] = []
        if not lines:
            found.append(f'{tag} has no line in the cart')
        elif sum(1 for i in lines if claims[i] == [n]) > 1:
            found.append(f"{tag} is spread over {len(lines)} lines ({', '.join(lines)}); keep just one")
        for i in lines:
            units = self.held_units(i)
            if claims[i] == [n] and units != item.qty:
                found.append(f'{i} is held at {units} unit(s) while {tag} asks for {item.qty}')
            better = self.fuller_label(n, kind_of[i]) or self.named_label(n, kind_of[i])
            if better:
                found.append(f'{i} comes from {kind_of[i]}; the store also has the label {better}, naming {tag} fully, so take the line from that one')
        return found

    def _g11_budget_faults(self, held: list[str]) -> list[str]:
        prices = {i: self.price_of(i) for i in held}
        cap = self.budget
        if not isinstance(cap, (int, float)) or any(p is None for p in prices.values()):
            return []
        if self.brief.get('budget_scope') == 'each':
            return [f'{i} is priced {prices[i]:g}, above the per-item limit of {cap:g}' for i in held if cap < prices[i] <= 10 * cap]
        total = sum((prices[i] * self.held_units(i) for i in held))
        return [f'the cart totals {total:g}, above the {cap:g} budget'] if cap + 1e-09 < total <= 10 * cap else []

    def count_fixes(self) -> list[tuple[str, int, int]]:
        cat_key = self.category_key()
        if not cat_key:
            return []
        kind = lambda i: str((self.rows.get(i) or {}).get(cat_key) or '')
        fixes: list[tuple[str, int, int]] = []
        for ident in self.cart_ids():
            have = self.held_units(ident) or 0
            owners = [n for n, item in enumerate(self.items, 1) if self.owns(item, kind(ident))]
            if have <= 0 or len(owners) != 1:
                continue
            item = self.items[owners[0] - 1]
            twins = sum(1 for i in self.cart_ids() if (self.cart_lines.get(i, 1) or 0) > 0 and self.owns(item, kind(i)))
            if twins == 1 and have != item.qty and not self.sold_out(ident):
                fixes.append((ident, have, item.qty))
        return fixes

    def _g11_cart_fields(self, name: str, schema: dict) -> tuple[Any, Any, Any]:
        props = schema.get('properties') or {}
        typed = [(k, v) for k, v in props.items() if isinstance(v, dict)]
        texts = [k for k, v in typed if 'string' in _kinds(v) and 'enum' not in v]
        by_id = [k for t, k in sorted(self.id_args) if t == name and k in texts]
        by_need = [k for k in schema.get('required') or [] if k in texts]
        target = (by_id[:1] or by_need[:1] or [texts[0] if len(texts) == 1 else None])[0]
        count = next((k for k, v in typed if 'integer' in _kinds(v)), None)
        mode = next((k for k, v in typed if isinstance(v.get('enum'), list) and len(v['enum']) >= 2), None)
        return target, count, mode

    def _g11_count_calls(self, name: str, props: dict, target: Any, count: Any, mode: Any, wrong: list[tuple[str, int, int]]) -> list[dict[str, Any]] | None:
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if self.cart_sets(name):
                calls.append({target: ident, count: want})
                continue
            if not mode:
                return None
            options = [str(o) for o in props[mode]['enum']]
            off = next(filter(lambda o: re.search('remov|delet|drop|clear|take', o, re.I), options), options[1])
            on = next(filter(lambda o: o != off, options), options[0])
            calls.extend(({target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}))
        return calls

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        target, count, mode = self._g11_cart_fields(name, schema)
        if not (name and target and count):
            return []
        calls = self._g11_count_calls(name, schema.get('properties') or {}, target, count, mode, wrong)
        if calls is None or len(calls) + 3 > self.left():
            return []
        done: list[str] = []
        pos = 0
        while pos < len(calls):
            faults: list[str] = []
            args = _shape(calls[pos], schema, 'args', faults)
            pos += 1
            reply = None if faults else self.act(name, args)
            if reply is None or self.finished:
                break
            seen = reply.get('observation')
            self.note_cart(name, args)
            self.cart_said = _dump(seen)
            self.observe(name, seen)
            done.append(_dump(args) + ' -> ' + self.show(seen)[:300])
        self.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    def check_key(self, ids: list[str]) -> str:
        return f"{'|'.join(sorted(ids))}#{self.version}"

    def _g11_check_prompt(self, ids: list[str], lead: bool) -> str:
        parts = [CHECK, 'REQUEST:\n' + self.request, 'REQUIREMENTS STILL IN FORCE:\n' + self.ledger()]
        if lead:
            wanted = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            parts.append('CART (only planned; judge each listing as though it sat in the cart at the asked quantity: ' + wanted + '):\n' + ', '.join(ids))
        else:
            parts.append('CART:\n' + self.cart_text())
        rows = []
        for i in ids:
            gone = 'SOLD OUT (the store reports it can no longer be bought). ' if self.sold_out(i) else ''
            rows.append(f'{i}: {gone}{self.record_of(i)}')
        parts.append("RECORDS FOR THE LISTINGS IN THE CART:\n" + '\n'.join(rows))
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            parts.append('OTHER LISTINGS SEEN SO FAR (newest first):\n' + self.others(ids, 15))
        return '\n\n'.join(parts)

    def _g11_judge(self, prompt: str, lead: bool, limit: float) -> Any:
        panel = self.judges(limit)
        got = None
        ranked = self.brief.get('pick') in ('cheapest', 'most', 'least')
        if len(panel) > 1 and not (lead or self.audited or ranked):
            got = self.ask_json(prompt, [LIGHT], 3000)
            if got:
                escalate = got.get('ok') is False and HEAVY in self.judges(limit)
                panel = [HEAVY] if escalate else []
        if panel and not self.no_credit:
            got = self.ask_json(prompt, panel, 3000) or got
        return got

    def _g11_verdict(self, ids: list[str], got: Any) -> str:
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        better = [b for b in map(str, _items(got.get('better'))) if b in self.rows and not self.sold_out(b)][:4]
        note = 'A careful review finds the cart not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            note = f"{note}. Listings that seem a better fit: {', '.join(better)}"
        return note + '. Repair those lines, or re-read the records if you believe the review erred; after that, go to checkout.'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = ids is not None
        if not lead:
            ids = self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or not self.judges(limit) or not ids:
            return ''
        self.checked.add(key)
        got = self._g11_judge(self._g11_check_prompt(ids, lead), lead, limit)
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        failed = bool(got) and got.get('ok') is False
        self.verdicts[key] = bool(got) and not failed
        return self._g11_verdict(ids, got) if failed else ''

    def _g11_pending_refs(self) -> list[str]:
        live = {need for _, need in self.needs_now()}
        return [need for _, need in self.refs if need in live and need not in self.ref_values]

    def refuse_trail(self) -> str:
        waiting = self._g11_pending_refs()
        if self.refuse_trailed or self.left() < TRAIL_ROOM or self.clock() > HUNT_BY or not waiting:
            return ''
        self.refuse_trailed = True
        text = self.trail(HUNT_BY)[0]
        if not text:
            return ''
        return text + 'Go find those values now and purchase the match; leave empty-handed only if no listing qualifies.'

    def early_trail(self) -> str:
        waiting = self._g11_pending_refs()
        if self.left() < TRAIL_ROOM or self.clock() > HUNT_BY or not waiting:
            return ''
        saved_trails, saved_blocks = self.trails, self.trail_blocks
        self.trails = 0
        self.trail_room = EARLY_TRAILS
        try:
            text = self.trail(HUNT_BY)[0]
        finally:
            self.trails = saved_trails
            self.trail_blocks = saved_blocks
            self.trail_room = None
        self.trace.append({'kind': 'early_trail', 'left': self.left(), 'values': list(self.ref_values.values())[:3]})
        return text

    def page_read(self, ident: str) -> bool:
        if ident not in self.opened:
            return False
        page = self.roles.get('offer')
        if not page or not self.single_target(page):
            return True
        return page in self.views.get(ident, ())

    def _g11_trail_records(self) -> list[str]:
        newest = self.seen_order[::-1]
        bodies = {i: self.full.get(i) or self.rows.get(i) for i in newest}
        read = set(filter(self.page_read, newest))
        shown = [i for i in newest if i in read and bodies[i]][:45]
        rest = [i for i in newest if i not in read or not bodies[i]][:60 - len(shown)]
        clip_row = lambda i: _dump(_clip(self.full[i], 120))[:700] if i in self.full else _dump(_clip(self.rows.get(i), 80))[:300]
        return ['OPENED ' + _dump(_clip(bodies[i], 300))[:1400] for i in shown] + ['ROW ' + clip_row(i) for i in rest]

    def _g11_trail_may_go(self, pending: list[str], stop: float | None) -> bool:
        if not pending:
            return False
        cap = min(6 if self.left() > 15 else 4, self.trail_room or 6)
        if self.trails >= cap or self.cost >= SPEND_HARD:
            return False
        return stop is None or self.clock() <= stop

    def _g11_trail_decide(self, pending: list[str]) -> dict:
        prompt = '\n\n'.join((TRAIL, 'REQUIREMENTS WHOSE VALUE SITS ON ANOTHER LISTING:\n' + '\n'.join(pending), 'RECORDS (newest first):\n' + '\n'.join(self._g11_trail_records())))
        models = [LIGHT, HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else [LIGHT]
        got = self.ask_json(prompt, models, 4500) or {}
        self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
        return got

    def _g11_chain_outcome(self, chain: dict, pending: list[str], lone: str | None) -> tuple[str, Any]:
        said = str(chain.get('need') or '').strip()
        need = (said if said in pending else None) or lone
        if not need:
            return ('skip', None)
        links = list(filter(lambda l: isinstance(l, dict), _items(chain.get('links'))))
        ids = [str(l.get('id') or '').strip() for l in links]
        value = str(chain.get('value') or '').strip()
        if links and all(map(self.page_read, ids)) and (chain.get('done') is True and value or not str(links[-1].get('find') or '').strip()):
            return ('value', (need, value or f'as stated on {ids[-1]}'))
        closed = [i for i in ids if i in self.rows and not self.page_read(i)]
        if closed:
            return ('reach', closed)
        gap = next((l for l, i in zip(links, ids) if i not in self.rows), {})
        about = str(gap.get('about') or '') or 'the listing it points at'
        return ('gap', (need, about, str(gap.get('find') or '').strip()))

    def _g11_trail_apply(self, got: dict, pending: list[str]) -> tuple[list[str], list[str], list[tuple[str, str]]]:
        reach: list[str] = []
        lost: list[str] = []
        gaps: list[tuple[str, str]] = []
        lone = pending[0] if len(pending) == 1 else None
        outcomes = [self._g11_chain_outcome(c, pending, lone) for c in _items(got.get('chains')) if isinstance(c, dict)]
        for verdict, data in outcomes:
            if verdict == 'value':
                self.ref_values[data[0]] = data[1]
                self.version += 1
            elif verdict == 'reach':
                reach.extend([i for i in data if i not in reach])
            elif verdict == 'gap':
                need, about, words = data
                lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
                gaps.append((about, words))
        return reach, lost, gaps

    def _g11_trail_open(self, tool: str, target: str, reach: list[str], seen: list[str]) -> None:
        for ident in reach[:max(0, min(3, 6 - len(seen)))]:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            reply = None if faults or self.rushed() else self.act(tool, args)
            if reply is None or self.finished:
                return
            page = reply.get('observation')
            self.observe(tool, page)
            seen.append(f'{ident}: ' + self.show(page)[:1500])

    def _g11_trail_relink(self, gaps: list[tuple[str, str]], lost: list[str], seen: list[str]) -> tuple[list[str], list[str]]:
        found = self.find_link(*gaps[0])
        last = self.trace[-1] if self.trace else {}
        if last.get('kind') != 'link_known':
            return found, lost
        seen.append(f"{last['id']}: this page, already open, belongs to the shop the chain names and carries the details it describes; read the needed value off it")
        return found, lost[1:]

    def trail(self, stop: float | None=None) -> tuple[str, bool]:
        pending = self._g11_pending_refs()
        tool = self.offer_tool()
        target = self.single_target(tool)
        can_open = bool(tool and target)
        seen: list[str] = []
        lost: list[str] = []
        while self._g11_trail_may_go(pending, stop):
            self.trails += 1
            reach, lost, gaps = self._g11_trail_apply(self._g11_trail_decide(pending), pending)
            pending = list(filter(lambda p: p not in self.ref_values, pending))
            relink = can_open and not reach and bool(gaps) and self.left() > 12 and self.link_searches < 3
            if relink:
                reach, lost = self._g11_trail_relink(gaps, lost, seen)
            stuck = not (reach and can_open) or self.left() < len(reach) + 4 or self.finished
            if stuck:
                break
            self._g11_trail_open(tool, target, reach, seen)
        opened = '[Cairn opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n' if seen else ''
        values = '[values read from the pages requirements point at] ' + '; '.join(map(lambda kv: f'{kv[0]} -> {kv[1]}', self.ref_values.items())) + '\n' if self.ref_values else ''
        room = self.left()
        if not (lost and self.trail_blocks < 2 + (room > 15) and room > 5):
            return (opened + values, False)
        self.trail_blocks += 1
        return (opened + values + "An order only counts once every page a requirement points at is opened in full (a search row does not do), and any page that refers on to another listing is followed too. Not found and opened yet: " + ' | '.join(lost[:3]) + ". Run a search on that shop's name together with the details its description lists (its other listings of the same kind do not count), open the result from that shop whose page carries those details, apply the value found there, and only then go to checkout.", True)

    def not_maker_keys(self) -> tuple[Any, ...]:
        return (self.id_key, self.category_key(), self.title_key())

    def _g11_mentioned(self, ident: str, low: str, skip: tuple[Any, ...]) -> bool:
        for key, val in (self.rows.get(ident) or {}).items():
            if not isinstance(val, str) or key in skip:
                continue
            name = str(val).strip()
            if 2 < len(name) <= 40 and name.lower() not in NO_NAME and re.search('(?<![a-z0-9])' + re.escape(name.lower()) + '(?![a-z0-9])', low):
                return True
        return False

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        skip = self.not_maker_keys()
        low = about.lower()
        return [i for i in ids if i not in self.opened and self._g11_mentioned(i, low, skip)]

    def opened_shop(self, about: str) -> list[str]:
        skip = self.not_maker_keys()
        low = about.lower()
        return list(filter(lambda i: self._g11_mentioned(i, low, skip), sorted(self.opened)))  # DET

    def _g11_leaves(self, node: Any) -> list[str]:
        if isinstance(node, dict):
            node = list(node.values())
        if not isinstance(node, list):
            return [str(node)]
        return [leaf for part in node for leaf in self._g11_leaves(part)]

    def _g11_link_search(self, about: str, words: str, memo: str, plan: tuple[Any, ...]) -> list[str]:
        tool, base, field = plan
        known: list[str] = []
        queries = list(dict.fromkeys((words, ' '.join(words.split()[:2]))))
        while queries:
            query = queries.pop(0)
            faults: list[str] = []
            args = _shape({**base, field: query}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            self.link_searches += 1
            reply = self.act(tool, args)
            if reply is None:
                break
            observation = reply.get('observation')
            self.observe(tool, observation)
            hits = [i for i, _ in self.records_in(observation)]
            self.link_seen[memo] = hits
            known = self.shop_rows(about, hits)
            self.trace.append({'kind': 'link_search', 'words': query[:80], 'rows': known[:4]})
            if hits:
                break
        return known

    def find_link(self, about: str, words: str) -> list[str]:
        skip = self.not_maker_keys()
        named: set[str] = set()
        for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about):
            for key, val in (self.rows.get(ident) or {}).items():
                if isinstance(val, str) and len(val.strip()) <= 40 and key not in skip and val.strip().lower() not in NO_NAME:
                    named.update(re.findall('[a-z0-9]{3,}', val.lower()))
        source = (words or about).lower()
        common = {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}
        values = {w for w in re.findall('[a-z0-9]+', source) if len(w) >= 3 or w.isdigit()} - named - common - {'of'}
        pattern = lambda v: '(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])'
        if values:
            for ident in self.opened_shop(about):
                text = ' '.join(self._g11_leaves(self.full.get(ident) or self.rows.get(ident) or {})).lower()
                if all(re.search(pattern(v), text) for v in values):
                    self.trace.append({'kind': 'link_known', 'id': ident})
                    return []
        plan = self.value_template()
        memo = ' '.join(words.lower().split())
        known: list[str] = []
        if memo and memo in self.link_seen:
            known = self.shop_rows(about, self.link_seen[memo])
        elif plan and words:
            known = self._g11_link_search(about, words, memo, plan)
        kind = set(re.findall('[a-z0-9]{3,}', source)) - named - common

        def of_kind(ids: list[str]) -> list[str]:
            if not kind:
                return list(ids)
            return [i for i in ids if kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        picked = of_kind(known)
        if not picked:
            picked = of_kind(self.shop_rows(about, list(reversed(self.seen_order))))
        return picked[:2]

    def _g11_unread_links(self) -> list[str]:
        last = next((t for t in reversed(self.trace) if isinstance(t, dict) and t.get('kind') == 'trail'), None)
        got = last.get('got') if last and isinstance(last.get('got'), dict) else {}
        unread: list[str] = []
        for chain in filter(lambda c: isinstance(c, dict), _items(got.get('chains'))):
            for link in _items(chain.get('links')):
                ident = str(link.get('id') or '').strip() if isinstance(link, dict) else ''
                if ident in self.rows and ident not in unread and not self.page_read(ident):
                    unread.append(ident)
        return unread

    def anchor_view(self) -> None:
        tool = self.offer_tool()
        target = self.single_target(tool)
        unread = self._g11_unread_links()
        if not (unread and tool and target) or self.finished or self.rushed():
            return
        self.anchor_done = True
        first = unread[0]
        faults: list[str] = []
        args = _shape({target: first}, self.schemas.get(tool), 'args', faults)
        reply = None if faults else self.act(tool, args)
        if reply is not None:
            self.absorb(reply.get('observation'))
            self.trace.append({'kind': 'anchor_view', 'id': first, 'left': self.left()})

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not (tool and target):
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if i not in self.gone and self.offer_seen.get(i, -99) < since]
        if not stale or self.left() < len(stale) + 5:
            return ''
        changed: list[str] = []
        for ident in stale:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), 'args', faults)
            if faults or self.rushed():
                break
            before = dict(self.rows.get(ident) or {})
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            observation = reply.get('observation')
            self.offer_seen[ident] = self.step
            self.observe(tool, observation)
            moved = [k for k, v in (self.rows.get(ident) or {}).items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                changed.append(f"{ident} has changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(changed)})
        if changed:
            self.version += 1
        return '\n'.join(changed)

    def sold_out(self, ident: str) -> bool:
        return any(ident in pool for pool in (self.unavailable, self.gone))

    def _g12_strings(self, root: Any) -> list[str]:
        # depth-first, pre-order, nothing below depth 4
        texts: list[str] = []
        stack: list[tuple[Any, int]] = [(root, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > 4:
                continue
            if isinstance(node, str):
                texts.append(node)
                continue
            children = list(node.values()) if isinstance(node, dict) else node if isinstance(node, list) else []
            stack.extend((child, depth + 1) for child in reversed(children))
        return texts

    def _g12_pointer_match(self, sentence: str) -> Any:
        for pattern in (OVERRIDE_FOR, OVERRIDE_TAKE):
            m = re.match(pattern, sentence, re.I)
            if m:
                return m
        return re.match(OVERRIDE_TAKEN, sentence, re.I) if re.search(OVERRIDE_MARK, sentence, re.I) else None

    def overrides(self, ident: str) -> list[tuple[str, str]]:
        sentences = (piece.strip() for text in self._g12_strings(self.rows.get(ident) or self.full.get(ident) or {}) for piece in re.split('(?<=[.!?])\\s+(?=[A-Z])', text))
        matched = filter(None, map(self._g12_pointer_match, sentences))
        found: list[tuple[str, str]] = []
        for m in matched:
            prop, pointer = m.group(1).strip().lower(), m.group(2).strip()
            if not prop or _stem(prop) in {_stem(p) for p, _ in found}:
                continue
            found.append((prop, pointer))
        return found

    def _g12_owner_number(self, line: str, cat_key: Any) -> int:
        return self.owner_of(str((self.rows.get(line) or {}).get(cat_key) or '')) if cat_key else 0

    def override_hint(self) -> str:
        if self.left() <= self.tail() + 2:
            return ''
        needs = self.needs_now()
        cat_key = self.category_key()
        notes: list[str] = []
        for line in self.cart_ids():
            number = self._g12_owner_number(line, cat_key)
            mine = [need for n, need in needs if n == number or not number]
            for prop, pointer in self.overrides(line):
                mark = f'{line}|{_words(prop)}'
                hit = next((need for need in mine if _same_prop(need.split(':', 1)[0], prop)), None)
                if hit is None or mark in self.override_held:
                    continue
                self.override_held.add(mark)
                notes.append(f"{line}'s page points its {prop} at a different listing ({pointer[:160]}); '{hit}' is therefore decided by that other listing's {prop}, not by the title or attributes of {line} itself")
        shown = notes[:3]
        if not shown:
            return ''
        self.trace.append({'kind': 'override_hold', 'lines': shown})
        return '[Cairn] ' + '; '.join(shown) + '. Locate that other listing through a search on its brand, vendor or type and read its page: if its value passes, leave the cart as it is; if not, put in a listing whose own attributes pass instead. Checkout comes after.'

    def _g12_refers(self, need: str) -> bool:
        return need in [n for _, n in self.refs]

    def shows_need(self, ident: str, need: str) -> str:
        head, _, tail = need.partition(':')
        prop, value = head.strip(), _plain_value(tail.strip())  # CO3
        stops = (
            lambda: any(_same_prop(prop, p) for p, _ in self.overrides(ident)),  # OV
            lambda: not (prop and value),
            lambda: re.search(CHEAP_LIMIT, value.lower()),
            lambda: self._g12_refers(need),
        )
        record = self.full.get(ident) or {}
        if any(stop() for stop in stops):
            return ''
        wanted = _roots(value)
        if not wanted:
            return ''
        block = _fact_block(record) or {}
        keys = list(filter(lambda k: _names(_stem(prop), _stem(k)), block))
        if not keys:
            pool = [str(v) for v in record.values() if isinstance(v, str)] + list(map(str, block.values()))
            return f'"{value}" in its text' if any(wanted <= _roots(t) for t in pool) else ''
        hit = next((k for k in keys if wanted <= _roots(str(block[k]))), None)
        return '' if hit is None else f'{hit}={str(block[hit])[:60]}'

    def car_binds(self) -> dict[int, str]:  # CF1
        if self.car_cache is not None:
            return self.car_cache
        codes = sorted({c.lower() for c in re.findall('\\b[A-Z]{2,6}\\d{3,6}\\b', self.request)})

        def named(item: Any) -> str | None:
            text = ' '.join(item.musts).lower()
            return next((c for c in codes if re.search('(?<![a-z0-9])' + re.escape(c) + '(?![a-z0-9])', text)), None)
        self.car_cache = {}
        pairs = ((number, named(item)) for number, item in enumerate(self.items, 1))
        self.car_cache.update((number, code) for number, code in pairs if code)
        return self.car_cache

    def car_fit(self, ident: str, code: str, index: dict[str, Any]) -> str:
        mark = index['marks'].get(ident) or {}
        for state, field in (('excepted', 'bars'), ('fits', 'works')):
            if any(_code_hit(entry, code) for entry in mark.get(field) or []):
                return state
        return 'other' if (mark.get('works') or mark.get('codes')) else 'none'

    def _g12_cost(self, ident: str) -> float:
        price = self.price_of(ident)
        return float('inf') if price is None else price

    def car_fitting(self, number: int, index: dict[str, Any]) -> list[str]:
        code = self.car_binds().get(number)
        if not code:
            return []
        fitting = (i for i in self.item_ids(number) if self.buyable(i) and self.car_fit(i, code, index) == 'fits')
        return sorted(fitting, key=self._g12_cost)

    def _g12_priced(self, ident: str) -> str:
        price = self.price_of(ident)
        return ident if price is None else f'{ident} ({price:g})'

    def car_lines(self, observations: list[Any]) -> str:  # CF2
        binds = self.car_binds()
        if not binds:
            return ''
        here = list(dict.fromkeys(i for seen in observations for i, _ in self.records_in(seen)))
        if not here:
            return ''
        index = self.fit_index()
        blocks: list[str] = []
        for number, code in binds.items():
            owned = self.item_ids(number)
            state = {i: self.car_fit(i, code, index) for i in here if i in owned}
            if not state:
                continue
            bucket = lambda s: [i for i in state if state[i] == s]
            table = (
                ('fits ' + code.upper() + ' by its own text', [self._g12_priced(i) for i in sorted(bucket('fits'), key=self._g12_cost)]),
                ('rules the car out by name', bucket('excepted')),
                ('lists only other codes', bucket('other')),
                ('names no code at all', bucket('none')),
            )
            text = '; '.join(f"{label}: {', '.join(ids[:8])}" for label, ids in table if ids)
            blocks.append(f'[Cairn] item {number} ({self.items[number - 1].name}), fit text of each row taken as written: {text}')
        return ('\n' + '\n'.join(blocks)) if blocks else ''

    def _g12_car_need(self, number: int, code: str, index: dict[str, Any], cart: set[str]) -> str | None:
        if any(i in cart and self.car_fit(i, code, index) == 'fits' for i in self.item_ids(number)):
            return None
        fits = self.car_fitting(number, index)
        seen = len(self.item_ids(number))
        head = f'item {number} ({self.items[number - 1].name}) must work with the car {code.upper()}: '
        if not fits:
            return head + f'no row out of {seen} seen fits it so far; search the item by noun and values (the code itself is ignored by the store) and keep paging'
        listed = ', '.join(f'{i} {self.price_of(i):g}' for i in fits[:4] if self.price_of(i) is not None)
        return head + f'{len(fits)} out of {seen} rows seen fit it by their own wording, lowest price first: ' + listed

    def car_plan(self) -> str:  # CF3
        binds = self.car_binds()
        if not binds:
            return ''
        index = self.fit_index()
        cart = set(self.cart_ids())
        lines = [text for text in (self._g12_car_need(number, code, index, cart) for number, code in binds.items()) if text]
        return '[car fit] ' + ' | '.join(lines) if lines else ''

    def car_faults(self) -> list[str]:  # CF4
        binds = self.car_binds()
        index = self.fit_index()
        reasons = {'excepted': 'lists {} as an exception', 'other': 'claims a fit for other codes only'}
        faults: list[str] = []
        pairs = ((number, code, i) for number, code in binds.items() for i in self.cart_ids() if i in self.item_ids(number))
        for number, code, i in pairs:
            why = reasons.get(self.car_fit(i, code, index))
            if why is None:
                continue
            fits = [f for f in self.car_fitting(number, index) if f != i][:4]
            tail = f"; fitting listings seen, lowest price first: {', '.join(fits)}" if fits else '; nothing seen fits it yet'
            faults.append(f'{i} (item {number}) ' + why.format(code.upper()) + ', which means it does not fit the car ' + code.upper() + tail)
        return faults

    def _g12_limit_need(self, n: int, need: str, number: int) -> bool:
        value = need.partition(':')[2]
        rules = (
            lambda: n != number,
            lambda: _conditional(need),
            lambda: self._g12_refers(need),
            lambda: not re.search(LIMIT_NUMBER, value, re.I),
        )
        return not any(rule() for rule in rules)

    def _g12_line_gaps(self, line: str, record: dict[str, Any], cat_key: str, needs: list[tuple[int, str]]) -> list[str]:
        number = self.owner_of(str((self.rows.get(line) or record).get(cat_key) or ''))
        block = _fact_block(record) or {}
        text = _spelled(' '.join(str(v) for v in record.values() if isinstance(v, str)))
        gaps: list[str] = []
        for n, need in needs:
            if not self._g12_limit_need(n, need, number):
                continue
            prop = need.partition(':')[0]
            words = _spelled(prop) - {'of', 'the', 'a', 'an', 'in', 'for'}
            if not words or words <= text or any(words & _spelled(k) for k in block):
                continue
            states = lambda i: any(words & _spelled(k) for k in (_fact_block(self.full[i]) or {}))
            others = sorted((i for i in self.item_ids(number) if i in self.full and i != line and self.buyable(i) and states(i)), key=self._g12_cost)
            tail = f"; opened listings of that item that do state it: {', '.join(others[:4])}" if others else '; search for listings of that item that state it'
            gaps.append(f"{line} gives no {prop.strip()} anywhere, so nothing shows '{need}' is met" + tail)
        return gaps

    def unstated_limits(self) -> list[str]:
        cat_key = self.category_key()
        if not cat_key:
            return []
        needs = self.needs_now()
        opened = ((line, self.full.get(line)) for line in self.cart_ids())
        return [gap for line, record in opened if record for gap in self._g12_line_gaps(line, record, cat_key, needs)]

    def _g12_holds_all(self, block: dict[str, Any], keys: list[str], wanted: Any) -> bool:
        def holds(key: str) -> bool:
            return any(wanted <= _exact_tokens(part) for part in re.split('[,;/|]', str(block[key])))
        return bool(keys) and all(map(holds, keys))

    def own_key_shows(self, ident: str, need: str, number: int=0) -> str:
        head, _, tail = need.partition(':')
        prop, value = head.strip(), _plain_value(tail.strip())  # CO3
        wanted = _exact_tokens(value)
        if not (prop and wanted) or re.search(LIMIT_WORDS, value.lower()) or self._g12_refers(need):
            return ''
        if any(_same_prop(prop, p) for p, _ in self.overrides(ident)):
            return ''
        block = _fact_block(self.full.get(ident) or {}) or {}
        base = _spelled(prop)
        whole = list(filter(lambda k: base <= _spelled(k), block))  # CW: keys naming the whole property
        item = self.items[number - 1] if 1 <= number <= len(self.items) else None
        own = frozenset() if item is None else (_spelled(item.name) | _spelled(item.shelf or ''))
        extra = lambda k: _spelled(k) - base
        named = [k for k in whole if extra(k) and (extra(k) - {'of', 'the', 'for'}) <= own]
        chosen = named or whole  # CW: a key adding the item's own name to the property decides over the bare key
        if not self._g12_holds_all(block, chosen, wanted):
            return ''
        return f'{chosen[0]}={str(block[chosen[0]])[:60]}'

    def _g12_code_ok(self, line: str, best: Any, needs: list[tuple[int, str]], cat_key: Any) -> bool:
        number = self._g12_owner_number(line, cat_key)
        mine = [need for n, need in needs if n == number]
        return bool(mine) and all(self.own_key_shows(best[0], need, number) for need in mine)  # CW

    def _g12_push(self, tool: str, calls: list[Any]) -> Any:
        for args in calls:
            faults: list[str] = []
            shaped = _shape(args, self.schemas.get(tool), 'args', faults)
            if faults:
                return
            reply = self.act(tool, shaped)
            if reply is None or self.finished:
                return
            self.note_cart(tool, shaped)
            self.observe(tool, reply.get('observation'))
            yield _dump(shaped)

    def code_swap(self, found: list[Any]) -> str:
        cart_tool = self.roles.get('cart')
        if not cart_tool or self.left() <= self.tail() + 2 * len(found) + 2 or self.rushed():
            return ''
        needs = self.needs_now()
        cat_key = self.category_key()
        if not all(self._g12_code_ok(line, best, needs, cat_key) for line, _, _, best in found):
            return ''
        drop = [line for line, _, _, _ in found]
        lines = {i: self.cart_lines.get(i, 1) for i in self.cart_ids() if i not in drop}
        for _, _, units, best in found:
            lines[best[0]] = units
        calls = self.cart_calls(lines, drop)
        if not calls or self.left() < len(calls) + self.tail() + 1:
            return ''
        done = list(self._g12_push(cart_tool, calls))
        self.trace.append({'kind': 'code_swap', 'swaps': [(line, b[0]) for line, _, _, b in found], 'done': done})
        if not done:
            return ''
        moves = '; '.join(f"{b[0]} ({b[1]:g}) replacing {line} ({price:g}); the requirement's own property in its record reads {'; '.join(b[2])}" for line, price, _, b in found)
        return '[Cairn] Cheapest is the pick rule, so Cairn made the cart change on its own: ' + moves + ' (' + '; '.join(done) + '). Go to checkout now unless some other requirement fails.'

    def _g12_over_budget(self, cost: float, line: str, cart: list[str], prices: dict[str, Any], units: dict[str, Any]) -> bool:
        if not isinstance(self.budget, (int, float)) or any(p is None for p in prices.values()):
            return False
        if self.brief.get('budget_scope') == 'each':
            total = cost
        else:
            total = sum(prices[i] * units[i] for i in cart if i != line) + cost * units[line]
        return total > self.budget + 1e-09

    def _g12_kind_of(self, ident: str, cat_key: str) -> str:
        return str((self.rows.get(ident) or self.full.get(ident) or {}).get(cat_key) or '').strip().lower()

    def _g12_cheapest_shown(self, line: str, price: float, kind: str, number: int, ctx: tuple[Any, ...]) -> Any:
        cat_key, cart, needs, prices, units = ctx
        mine = [need for n, need in needs if n == number]
        best = None
        for ident in self.full:
            if ident in cart or not self.buyable(ident) or self._g12_kind_of(ident, cat_key) != kind:
                continue
            cost = self.price_of(ident)
            if cost is None or cost >= price or (best is not None and cost >= best[1]):
                continue
            if self._g12_over_budget(cost, line, cart, prices, units):
                continue
            shown = [self.shows_need(ident, need) or self.own_key_shows(ident, need, number) for need in mine]  # CW
            if all(shown):
                best = (ident, cost, shown)
        return best

    def cheaper_fit_hint(self) -> str:
        if self.brief.get('pick') != 'cheapest' or self.cheap_holds >= CHEAP_HOLDS or self.left() <= self.tail() + 2:
            return ''
        cat_key = self.category_key()
        if not cat_key:
            return ''
        cart = self.cart_ids()
        needs = self.needs_now()
        units = {i: self.cart_lines.get(i) or 1 for i in cart}
        prices = {i: self.price_of(i) for i in cart}
        ctx = (cat_key, cart, needs, prices, units)
        found = []
        for line in cart:
            label = str((self.rows.get(line) or {}).get(cat_key) or '')
            price, kind, number = prices[line], label.strip().lower(), self.owner_of(label)
            if price is None or not (kind and number) or any(number in pair for pair in self.bound):  # CSF: a fit is never proven word for word
                continue
            best = self._g12_cheapest_shown(line, price, kind, number, ctx)
            if best:
                found.append((line, price, units[line], best))
        if not found:
            return ''
        mark = '|'.join([str(self.version)] + sorted(b[0] for _, _, _, b in found))
        if mark in self.cheap_marks:
            return ''
        self.cheap_marks.add(mark)
        self.cheap_holds += 1
        swapped = self.code_swap(found)  # CW
        if swapped:
            self.swapped_now = True  # SW
            return swapped
        self.trace.append({'kind': 'cheap_hold', 'swaps': [(line, b[0]) for line, _, _, b in found]})
        claims = '. '.join(f"{b[0]} costs {b[1]:g}, less than cart line {line} at {price:g} in the same category (opened; its record shows " + ('; '.join(b[2]) if b[2] else "the item's category") + ')' for line, price, _, b in found)
        steps = '; '.join(f'take out {line} and add {b[0]} at {n} unit(s)' for line, _, n, b in found)
        return '[Cairn] Cheapest is the pick rule. ' + claims + f'. If nothing in its record breaks a requirement in force, {steps} (using the cart tool as it expects), and then check out.'

    def swap_hint(self) -> str:
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ''
        self.gone.update(dict.fromkeys(lost, self.step))
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Cairn] No longer purchasable: {', '.join(lost)} (the store marks it unbuyable). It stays gone for good, so skip reopening it."
        cheapest = self.brief.get('pick') == 'cheapest'
        endings = (
            (lambda: picks and self.left() <= 6, lambda: f" Steps are nearly used up, so right now make two calls (put {picks[0]} in with the quantity asked for, take the dead line out) and go to checkout. Further candidates already read: {', '.join(picks[1:]) or 'none'}."),
            (lambda: picks and cheapest, lambda: f" Candidates read earlier for that item, lowest price first: {', '.join(picks)}. Since the cheapest one has to win, first run a single search whose price ceiling is what {picks[0]} costs (when the search takes a max price) unless every fitting listing is already known; then put the cheapest fitting one in where the unavailable line was."),
            (lambda: picks, lambda: f" Candidates read earlier for that item: {', '.join(picks)}. Right now, replace the unavailable cart line with {picks[0]} in the quantity asked for and go to checkout."),
        )
        tail = next((build for test, build in endings if test()), lambda: ' Drop that line from the cart and add the best other fitting listing; if no listing read so far fits, search for one.')
        return text + tail()

    def find_sold(self) -> str:
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if not ids or any(map(self.sold_out, ids)):
            return ''
        if len(ids) > 1:
            return self.reread_offers()
        self.unavailable.add(ids[0])
        return ''

    def replacements(self, lost: list[str]) -> list[str]:
        if not self.judges():
            return []
        usable = (i for i in reversed(self.seen_order) if not (i in lost or i in self.gone or self.sold_out(i)))
        records = [_dump(_clip(self.full.get(i) or self.rows.get(i), 200))[:1200] for _, i in zip(range(25), usable)]
        if not records:
            return []
        sections = (
            ('REQUEST', self.request),
            ('REQUIREMENTS IN FORCE', self.ledger()),
            ('SOLD-OUT CART LINE', '\n'.join(f'{i}: {self.record_of(i)}' for i in lost)),
            ('CART NOW', self.cart_text()),
            ('OTHER LISTING RECORDS SEEN (newest first)', '\n'.join(records)),
        )
        prompt = SWAP + ''.join(f'\n\n{head}:\n{body}' for head, body in sections)
        got = self.ask_json(prompt, self.judges()[::-1], 3500)
        fits: list[tuple[float, str]] = []
        taken: set[str] = set()
        for entry in _items((got or {}).get('fits')):
            boxed = isinstance(entry, dict)
            ident = str(entry.get('id') if boxed else entry)
            price = entry.get('price') if boxed else None
            if ident not in self.rows or ident in lost or ident in self.gone or ident in self.cart_ids() or ident in taken:
                continue
            taken.add(ident)
            numeric = isinstance(price, (int, float)) and not isinstance(price, bool)
            fits.append((float(price) if numeric else float('inf'), ident))
        order = sorted(fits, key=lambda f: f[0]) if self.brief.get('pick') == 'cheapest' else fits
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return [i for _, i in order][:4]

    def _g12_open_asked(self, got: dict[str, Any] | None) -> bool:
        wanted = [str(i) for i in _items((got or {}).get('open'))]
        rows = [i for i in wanted if i in self.rows and i not in self.opened and not self.sold_out(i)]
        return bool(rows) and bool(self.open_rows(rows))

    def audit(self) -> str:
        if self.audited or not self.judges():
            return ''
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named and self._g12_open_asked(got):
            got = self.audit_call()
            named = self.audit_fits(got)
        if not named:
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        listing = '; '.join(f"item {k}: {', '.join(v)}" for k, v in named.items())
        why = f" ({got.get('reason')})" if got.get('reason') else ''
        return 'On a closer second look these listings might fit: ' + listing + why + '. Open them and buy whatever genuinely fits; finish without buying only when every one of them fails.'

    def _g12_audit_line(self, ident: str) -> str:
        record = self.full.get(ident) if ident in self.full else None
        if record:
            return 'OPENED ' + _dump(_clip(record, 200))[:1200]
        return 'ROW ' + _dump(_clip(self.rows.get(ident), 120))[:500]

    def audit_call(self, models: list[str] | None=None) -> dict[str, Any] | None:
        ordered = sorted(reversed(self.seen_order), key=lambda i: i not in self.full)
        live = (i for i in ordered if not self.sold_out(i))
        records = [self._g12_audit_line(i) for _, i in zip(range(30), live)]
        if not records:
            return None
        prompt = '\n\n'.join([AUDIT, 'REQUEST:\n' + self.request, 'REQUIREMENTS IN FORCE:\n' + self.ledger(), 'LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)])
        judges = models if models is not None else self.judges()  # SP
        first = (models or [LIGHT]) if judges else []
        got = self.ask_json(prompt, first, 4000)  # SP2
        retry = HEAVY in judges and not self.no_credit and not self.audit_fits(got)
        if retry:
            got = self.ask_json(prompt, [HEAVY], 4000) or got
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        keep = lambda i: i in self.rows and not self.sold_out(i)
        pairs = ((str(k), [i for i in map(str, _items(v)) if keep(i)][:4]) for k, v in fits.items())
        return {k: ids for k, ids in pairs if ids}

    def _g13_open_plan(self, rows: list[str]) -> list[tuple[str, dict[str, Any]]] | None:
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and (not self.compare_bare()):
            return [(joiner, {many: rows[:self.compare_room()]})]
        if not (tool and target):
            return None
        return [(tool, {target: ident}) for ident in rows[:2]]

    def open_rows(self, rows: list[str]) -> bool:
        plan = self._g13_open_plan(rows)
        if plan is None:
            return False
        probe = {f'#{n}': 1 for n in range(len(self.items))}
        steps_for_cart = len(self.cart_calls(probe, []) or self.items)
        if len(plan) + max(5, self.tail() + steps_for_cart) > self.left():
            return False
        start = len(self.full)
        pending = list(plan)
        while pending:
            name, raw = pending.pop(0)
            faults: list[str] = []
            shaped = _shape(raw, self.schemas.get(name), 'args', faults)
            if faults:
                return False
            reply = self.act(name, shaped)
            if reply is None or self.finished:
                break
            self.observe(name, reply.get('observation'))
        gained = len(self.full) - start
        self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': gained})
        return gained > 0

    @staticmethod
    def _g13_free_text(spec: Any) -> bool:
        return isinstance(spec, dict) and 'string' in _kinds(spec) and 'enum' not in spec

    def _g13_nested_lines(self, props: dict[str, Any], lines: dict[str, int]) -> Any:
        """Cart schemas taking one array of line records; False when the schema has none."""
        def record_fields(spec: Any) -> dict[str, Any] | None:
            if not (isinstance(spec, dict) and 'array' in _kinds(spec)):
                return None
            shape = spec.get('items')
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                return shape['properties']
            return None
        found = next(((key, fields) for key, fields in ((k, record_fields(s)) for k, s in props.items()) if fields is not None), None)
        if found is None:
            return False
        key, fields = found
        id_key = next((k for k, s in fields.items() if self._g13_free_text(s)), None)
        count_key = next((k for k, s in fields.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
        if not id_key:
            return None
        records = []
        for ident, amount in lines.items():
            record = {id_key: ident}
            if count_key:
                record[count_key] = amount
            records.append(record)
        return [{key: records}]

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        nested = self._g13_nested_lines(props, lines)
        if nested is not False:
            return nested
        texts = [k for k, s in props.items() if self._g13_free_text(s)]
        pools = ((k for t, k in sorted(self.id_args) if t == name and k in texts), (k for k in schema.get('required') or [] if k in texts), iter(texts if len(texts) == 1 else []))
        target = None
        for pool in pools:
            target = next(pool, None)
            if target:
                break
        count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2)), None)
        if not target:
            return None
        sets = self.cart_sets(name)

        unset = object()

        def call(ident: str, flag: Any, amount: Any) -> dict[str, Any]:
            made = {target: ident}
            if mode and flag is not unset:
                made[mode] = flag
            if count and amount is not unset:
                made[count] = amount
            return made

        def take_out(ident: str) -> dict[str, Any] | None:
            have = self.cart_lines.get(ident, 1)
            if mode:
                return call(ident, props[mode]['enum'][1], have)
            if not count:
                return None
            if sets:
                return call(ident, unset, 0)
            floor = props[count].get('minimum')
            return call(ident, unset, -have) if isinstance(floor, int) and floor <= -have else None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            made = take_out(ident)
            if made is None:
                return None
            calls.append(made)
        held = self.cart_ids()
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in held else 0
            if have == want:
                continue
            if have and (not sets):
                return None
            calls.append(call(ident, props[mode]['enum'][0] if mode else unset, want))
        return calls

    def _g13_cc_plan(self, job: dict[str, Any]) -> bool:
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ident
        plan: dict[str, int] = {}
        taken: set[str] = set()
        count = len(self.items)
        for number, ids in self.lead_items.items():
            digits = re.search('\\d+', str(number))
            slot = int(digits.group()) if digits else (1 if count == 1 else None)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if not (pick and slot and 1 <= slot <= count) or kind(pick) in taken:
                continue
            plan[pick] = self.items[slot - 1].qty
            taken.add(kind(pick))
        if self.confirmed not in plan:
            return False
        keep: dict[str, int] = {}
        drop: list[str] = []
        for ident in self.cart_ids():
            if ident in plan:
                continue
            if self.sold_out(ident) or kind(ident) in taken:
                drop.append(ident)
            else:
                keep[ident] = self.cart_lines.get(ident, 1)
        if len(plan) + len({kind(i) for i in keep}) < count:
            return False
        job['lines'] = {**keep, **plan}
        job['drop'] = drop
        return True

    def _g13_cc_vet(self, job: dict[str, Any]) -> bool:
        lines = job['lines']
        calls = self.cart_calls(lines, job['drop'])
        if not calls or len(calls) + 1 > self.left():
            return False
        job['calls'] = calls
        for prepare in (self.fill_facts, self.show_links):
            prepare(list(lines), len(calls) + 3)
        self.check_cart(list(lines), SPEND_HARD)
        return bool(self.verdicts.get(self.check_key(list(lines))))

    def _g13_cc_fill(self, job: dict[str, Any]) -> bool:
        cart = job['cart']
        job['done'] = done = []
        for wanted in job['calls']:
            faults: list[str] = []
            args = _shape(wanted, self.schemas.get(cart), 'args', faults)
            if faults:
                return False
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return False
            self.note_cart(cart, args)
            self.observe(cart, reply.get('observation'))
            done.append(_dump(args))
        return True

    def _g13_cc_still_good(self, lines: dict[str, int]) -> bool:
        self.reread_offers()
        self.check_cart(list(lines), SPEND_HARD)
        return bool(self.verdicts.get(self.check_key(list(lines)))) and self.left() >= 1 and (not self.finished) and (not any((self.sold_out(i) for i in lines)))

    def _g13_cc_order_once(self, job: dict[str, Any], args: dict[str, Any], replies: list[str]) -> bool:
        go, said = self.gate('order')
        if said:
            replies.append(said)
        if not go:
            return False
        buy = job['buy']
        reply = self.act(buy, args)
        self.trace.append({'kind': 'completed', 'lines': job['lines'], 'drop': job['drop'], 'reply': _clip(reply, 200)})
        if reply is None:
            return False
        self.buys += 1
        ended = reply.get('finished')
        if not ended:
            self.refused_at = self.step
        self.observe(buy, reply.get('observation'))
        replies.append(self.show(reply.get('observation')))
        return not (self.finished or ended)

    def _g13_cc_buy(self, job: dict[str, Any]) -> bool:
        faults: list[str] = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(job['buy']), 'args', faults)
        if faults:
            return False
        replies: list[str] = []
        tries = 0
        while tries < 2:
            if tries and (not self._g13_cc_still_good(job['lines'])):
                break
            tries += 1
            if not self._g13_cc_order_once(job, args, replies):
                break
        if not replies:
            return False
        job['note'] = 'Skipped (Cairn handled this itself): the independent check confirmed ' + self.confirmed + ', so Cairn placed the cart (' + '; '.join(job['done']) + ') and checked out. Store answered: ' + ' | then: '.join(replies)
        return True

    def complete_cart(self) -> str:
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or (not cart) or (not buy):
            return ''
        self.completed = True
        job: dict[str, Any] = {'cart': cart, 'buy': buy}
        for step in (self._g13_cc_plan, self._g13_cc_vet, self._g13_cc_fill, self._g13_cc_buy):
            if not step(job):
                return ''
        return job['note']

    def _g13_fixed_args_match(self, role: str, brief_key: str, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get(role):
            return False
        fixed = self.brief.get(brief_key)
        if not isinstance(fixed, dict):
            return True
        for key, value in fixed.items():
            if args.get(key) != value:
                return False
        return True

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        return self._g13_fixed_args_match('buy', 'buy_args', name, args)

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        return self._g13_fixed_args_match('refuse', 'refuse_args', name, args)

    def offered(self) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get('ask')
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        rungs = ((lambda: left <= self.tail(), ()), (lambda: left <= self.tail() + 3, ('cart', 'cart_view')), (lambda: left <= len(self.items) + 5 and self.buys == 0, ('cart', 'cart_view', 'offer')))
        keep: set[Any] | None = next((set(ends) | {self.roles.get(r) for r in extra} for reached, extra in rungs if reached()), None)
        tools = []
        for tool in self.tools:
            if tool['function']['name'] == ask and self.message_args('x' * 200) is not None:
                continue
            tools.append(tool)
        if keep is None or not ends:
            return tools
        return [t for t in tools if t['function']['name'] in keep] or tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        required = schema.get('required') or []
        decline = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'
        out = dict(args)
        for key, spec in (schema.get('properties') or {}).items():
            if not self._g13_free_text(spec):
                continue
            value = out.get(key)
            if value is None and key not in required:
                continue
            text = str(value or '')
            if not text or self.brief.get('quiet_decline') is not False or any((ident in text for ident in self.rows)):
                out[key] = decline[:spec.get('maxLength', 200)]
        return out

    def _g13_mend_id(self, value: Any) -> tuple[Any, str]:
        if not isinstance(value, str):
            return (value, '')
        bare = value.strip()
        if bare in self.rows:
            return (bare, '')
        if _id_like(value, True) or len(bare) < 8:
            return (value, '')
        low = bare.lower()
        hits: list[str] = []
        for ident, row in self.rows.items():
            for field, text in row.items():
                if field != self.id_key and isinstance(text, str) and text.strip().lower().startswith(low):
                    hits.append(ident)
                    break
        if len(hits) == 1:
            return (hits[0], '')
        if hits:
            return (value, f"{value!r} matches more than one listing ({', '.join(hits[:6])}); give exactly one listing id")
        return (value, f"{value!r} isn't a listing id; use the {self.id_key} value shown in the result rows")

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_key:
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _kinds(v) or 'array' in _kinds(v)) and ('enum' not in v)]
        if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args):
            return (args, '')
        key = free[0]
        out = dict(args)
        if key not in args:
            return (out, '')
        raw = args[key]
        mended = []
        for value in (raw if isinstance(raw, list) else [raw]):
            fixed, problem = self._g13_mend_id(value)
            if problem:
                return (args, problem)
            mended.append(fixed)
        out[key] = mended if isinstance(raw, list) else mended[0]
        return (out, '')

    def cart_sets(self, name: str) -> bool:
        specs = [s for s in (self.schemas.get(name, {}).get('properties') or {}).values() if isinstance(s, dict)]
        for spec in specs:
            if isinstance(spec.get('enum'), list) and len(spec['enum']) >= 2:
                return False
        return next((True for spec in specs if 'integer' in _kinds(spec) and spec.get('minimum') == 0), False)

    def whole_removal(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        spec = self.schemas.get(name, {}).get('properties') or {}
        count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        if not count or count in args:
            return args
        if not any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items())):
            return args
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.cart_lines), None)
        if not ident:
            return args
        held = self.cart_lines[ident]
        widened = dict(args)
        widened[count] = min(held, spec[count].get('maximum', held))
        return widened

    @staticmethod
    def _g13_first_count(values: Any, fallback: Any) -> Any:
        return next((c for c in (_count(v) for v in values if not isinstance(v, str)) if c is not None), fallback)

    def _g13_cart_entries(self, spec: dict[str, Any], args: dict[str, Any], loose: Any) -> tuple[list[tuple[str, Any]], bool]:
        entries: list[tuple[str, Any]] = []
        whole = False
        for key, value in args.items():
            if not isinstance(value, list):
                if isinstance(value, str) and value in self.rows:
                    entries.append((value, loose))
                continue
            shape = (spec.get(key) or {}).get('items') or {}
            if not value:
                whole = whole or (isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape))
                continue
            for line in value:
                if isinstance(line, str):
                    if line in self.rows:
                        entries.append((line, loose))
                elif isinstance(line, dict):
                    whole = True
                    ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                    if ident:
                        entries.append((ident, self._g13_first_count(line.values(), 1)))
        return (entries, whole)

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get('properties') or {}
        removing = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            removing = removing or (isinstance(options, list) and len(options) >= 2 and value in options[1:])
        entries, whole = self._g13_cart_entries(spec, args, self._g13_first_count(args.values(), None))
        if whole:
            self.cart_lines = {i: c for i, c in entries if c and c > 0}
            return
        counted = any(('integer' in _kinds(v) for v in spec.values() if isinstance(v, dict)))
        operations = {'remove': lambda have, n: have - (n or 1) if counted else (have - n if n else 0), 'set': lambda have, n: 1 if n is None else n, 'add': lambda have, n: have + (1 if n is None else n)}
        apply = operations['remove' if removing else 'set' if self.cart_sets(name) else 'add']
        for ident, n in entries:
            after = apply(self.cart_lines.get(ident, 0), n)
            if after > 0:
                self.cart_lines[ident] = after
            else:
                self.cart_lines.pop(ident, None)

    def show(self, observation: Any) -> str:
        budget = SHOW_CHARS if self.records_in(observation) else TEXT_CHARS
        text = _dump(_clip(observation, budget))
        return text if len(text) <= OBS_LIMIT else text[:OBS_LIMIT] + '...(cut)'

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        arrays = (k for k, v in (schema.get('properties') or {}).items() if isinstance(v, dict) and 'array' in _kinds(v))
        target = next(arrays, None)
        if target and all((k == target for k in schema.get('required') or [])):
            return target
        return None

    def compare_room(self) -> int:
        props = (self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}
        cap = (props.get(self.compare_target() or '') or {}).get('maxItems')
        if isinstance(cap, int):
            return cap
        if not self.compare_bad:
            return max(4, self.compare_ok)
        return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))

    def _g13_open_picks(self, calls: list[dict[str, Any]], opener: str, target: str) -> Any:
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            raw = fn.get('arguments')
            args = _json_obj(raw) if isinstance(raw, str) else None
            if fn.get('name') != opener or not args or set(args) != {target}:
                continue
            ident = str(args.get(target) or '').strip()
            if ident in self.rows and ident not in self.gone:
                yield (index, ident)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = (self.roles.get('open'), self.roles.get('compare'), self.single_target(self.roles.get('open')))
        many = self.compare_target()
        if not (opener and joiner and target and many) or opener == joiner or self.compare_bare():
            return calls
        picks = list(self._g13_open_picks(calls, opener, target))[:self.compare_room()]
        if len(picks) < 2:
            return calls
        slots = dict(picks)
        ids = [ident for _, ident in picks]
        lead = picks[0][0]

        def rebuilt(index: int, call: dict[str, Any]) -> dict[str, Any]:
            if index not in slots:
                return call
            if index != lead:
                return {'id': call['id'], 'joined': True}
            return {'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': _dump({many: ids})}}
        out = [rebuilt(index, call) for index, call in enumerate(calls)]
        self.trace.append({'kind': 'joined', 'ids': ids})
        return out

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        batch = self.join_opens(calls)
        stages = self._g14_stages()
        size = len(batch)
        return [(call['id'], self._g14_first({'call': call, 'size': size, 'notes': []}, stages)) for call in batch]

    def _g14_first(self, job: dict[str, Any], stages: tuple) -> str | None:
        for stage in stages:
            said = stage(job)
            if said is not None:
                return said
        return None

    def _g14_stages(self) -> tuple:
        return (self._g14_joined, self._g14_unpack, self._g14_over, self._g14_talk_owned, self._g14_unknown, self._g14_out_of_reach,
                self._g14_parse, self._g14_schema, self._g14_ids, self._g14_compare_room, self._g14_gone, self._g14_absent_line,
                self._g14_lone_ending, self._g14_tail_kept, self._g14_first_page, self._g14_dead_codes, self._g14_repeat,
                self._g14_known_search, self._g14_cached_read, self._g14_seen_one, self._g14_seen_many, self._g14_review,
                self._g14_late_decline, self._g14_push_search, self._g14_salvage, self._g14_refuse_cart, self._g14_late_question,
                self._g14_gate, self._g14_execute, self._g14_compare_seen, self._g14_remember, self._g14_learn_tool,
                self._g14_cart_after, self._g14_search_after, self._g14_compose)

    def _g14_hold(self, body: str, lead: str = 'Not run yet (no step used). ') -> str:
        return lead + body + '\nCart now: ' + self.cart_text()

    def _g14_offered_names(self) -> str:
        return ', '.join(sorted({t['function']['name'] for t in self.offered()}))

    def _g14_joined(self, job: dict[str, Any]) -> str | None:
        if job['call'].get('joined'):
            return 'Opened in a single side-by-side step together with the others; see that result above.'
        return None

    def _g14_unpack(self, job: dict[str, Any]) -> None:
        spec = job['call'].get('function') or {}
        job['name'], job['raw'] = spec.get('name'), spec.get('arguments')

    def _g14_over(self, job: dict[str, Any]) -> str | None:
        return 'Not run: this episode has already ended.' if self.finished else None

    def _g14_talk_owned(self, job: dict[str, Any]) -> str | None:
        if job['name'] != self.roles.get('ask') or self.message_args('x' * 200) is None:
            return None
        return "Not run (no step used): messages to the shopper go only through Cairn's interviewer."

    def _g14_unknown(self, job: dict[str, Any]) -> str | None:
        if job['name'] in self.schemas:
            return None
        return f'Not run (no step used): no tool has that name. Available now: {self._g14_offered_names()}.'

    def _g14_out_of_reach(self, job: dict[str, Any]) -> str | None:
        if job['name'] in {t['function']['name'] for t in self.offered()}:
            return None
        return 'Not run (no step used): not enough steps remain for that tool; available now: ' + self._g14_offered_names() + '.'

    def _g14_parse(self, job: dict[str, Any]) -> str | None:
        raw = job['raw']
        if isinstance(raw, dict):
            job['args'] = raw
            return None
        job['args'] = _json_obj(raw) if (raw or '').strip() else {}
        return None if job['args'] is not None else 'Not run (no step used): could not read the arguments; send them as one JSON object with named fields.'

    def _g14_schema(self, job: dict[str, Any]) -> str | None:
        faults: list[str] = []
        job['args'] = _shape(job['args'], self.schemas[job['name']], 'args', faults)
        return 'Not run (no step used): ' + '; '.join(faults[:6]) if faults else None

    def _g14_ids(self, job: dict[str, Any]) -> str | None:
        job['args'], wrong = self.resolve_ids(job['name'], job['args'])
        if wrong:
            return 'Not run (no step used): ' + wrong + '.'
        if job['name'] == self.roles.get('cart'):
            job['args'] = self.whole_removal(job['name'], job['args'])
        return None

    def _g14_compare_room(self, job: dict[str, Any]) -> None:
        name, args = job['name'], job['args']
        many = job['many'] = self.compare_target() if name == self.roles.get('compare') else None
        if not (many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room() and self.compare_bad):
            return
        room = self.compare_room()
        rest = ', '.join(map(str, args[many][room:]))
        job['notes'].append(f'\n[Cairn] This store compares at most {room} listings at once, so only the first {room} went out. Still to view: {rest}.')
        job['args'] = {**args, many: args[many][:room]}

    def _g14_gone(self, job: dict[str, Any]) -> str | None:
        name, args = job['name'], job['args']
        target = job['target'] = self.single_target(name)
        if not target or name not in (self.roles.get('open'), self.roles.get('offer')) or str(args.get(target)) not in self.gone:
            return None
        return f'Not run (no step used): {args[target]} went out of stock at step {self.gone[str(args[target])]} for good. Put a different fitting listing on its cart line.'

    def _g14_absent_line(self, job: dict[str, Any]) -> str | None:
        args = job['args']
        if job['name'] != self.roles.get('cart') or not any(((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
            return None
        view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
        absent = next((v for v in args.values() if isinstance(v, str) and v in self.rows and v not in self.cart_lines and v not in view and v not in self.cart_said), None)
        if absent is None:
            return None
        return f'Not run (no step used): nothing to remove, {absent} is not a line in the cart. Cart now: ' + self.cart_text()

    def _g14_lone_ending(self, job: dict[str, Any]) -> str | None:
        job['ending'] = self.is_buy(job['name'], job['args']) or self.is_refuse(job['name'], job['args'])
        if job['ending'] and job['size'] > 1:
            return 'Not run (no step used): read the other results first, then send the finishing call by itself.'
        return None

    def _g14_tail_kept(self, job: dict[str, Any]) -> str | None:
        if job['ending'] or self.left() > self.tail() or not (self.roles.get('buy') or self.roles.get('refuse')):
            return None
        return "Not run (no step used): the remaining steps are reserved for the checkout (after the shopper approves the basket, if that is due) or for ending with no purchase."

    def _g14_first_page(self, job: dict[str, Any]) -> None:
        name, sent = job['name'], job['args']
        plain = {tool for role, tool in self.roles.items() if role != 'search'}
        job['fresh'] = name not in plain and self.query_is_new(name, sent)
        looked = self.first_look(name, sent, job['fresh'])
        if looked != sent:
            job['notes'].append(f'\n[Cairn] A new search starts at page one, so Cairn requested that page (you sent {_dump(sent)}).')
        job['args'] = self.row_ids(name, looked)

    def _g14_dead_codes(self, job: dict[str, Any]) -> str | None:
        if job['name'] == self.roles.get('search') and self.codes_dead and self.names_code(job['args']):
            return "Not run (no step used): searching this store by model or part code returns nothing. Use words and values or list the category, then read each row's code."
        return None

    def _g14_repeat(self, job: dict[str, Any]) -> str | None:
        name = job['name']
        job['mark'] = name + _dump(job['args'])
        if job['ending'] or name == self.roles.get('cart') or job['mark'] not in self.recent:
            return None
        return 'Not run (no step used): identical to a call made moments ago, with nothing different in between; read the earlier result above.'

    def _g14_known_search(self, job: dict[str, Any]) -> str | None:
        job['readers'] = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
        mark = job['mark']
        known = self.searches_seen.get(mark) if job['name'] == self.roles.get('search') else None  # SR
        if not known or known[0] != self.version or mark in self.read_cache:
            return None
        return f'Not run (no step used): an identical search already ran at step {known[1]}, with no change since. Result then:\n' + known[2][:4000]

    def _g14_cached_read(self, job: dict[str, Any]) -> str | None:
        cached = self.read_cache.get(job['mark'])
        if job['name'] not in job['readers'] or not cached or cached[0] != self.version:
            return None
        return f'Not run (no step used): nothing changed since this identical call at step {cached[1]}, which returned:\n' + cached[2][:3000]

    def _g14_seen_one(self, job: dict[str, Any]) -> str | None:
        name, args, target = job['name'], job['args'], job['target']
        if not target or name not in job['readers'] or name == self.roles.get('compare'):
            return None
        key = str(args.get(target))
        if not (self.shown_already(name, key) or (name, key) in self.factless):
            return None
        return f'Not run (no step used): this view of {args[target]} has nothing new since the last change. Record: ' + self.record_of(str(args[target]))

    def _g14_seen_many(self, job: dict[str, Any]) -> str | None:
        name, args, many = job['name'], job['args'], job['many']
        if not (many and isinstance(args.get(many), list) and args[many]):
            return None
        asked = args[many]
        keep = [i for i in asked if not self.shown_already(name, str(i))]
        if not keep:
            records = '\n'.join((f'{i}: {self.record_of(str(i))}' for i in asked))
            return 'Not run (no step used): this view of these listings has nothing new since the last change. Records:\n' + records[:4000]
        if len(keep) < len(asked):
            skipped = ', '.join((str(i) for i in asked if i not in keep))
            job['notes'].append(f'\n[Cairn] Skipped {skipped}: already shown in full, unchanged since.')
            job['args'] = {**args, many: keep}
            job['mark'] = name + _dump(job['args'])
        return None

    def _g14_review(self, job: dict[str, Any]) -> str | None:
        if not job['ending'] or self.left() <= 2:
            return None
        if self.is_buy(job['name'], job['args']):
            steps = (self._g14_b_hunt, self._g14_b_reread, self._g14_b_cart_look, self._g14_b_sold, self._g14_b_chain, self._g14_b_fit,
                     self._g14_b_lines, self._g14_b_fill, self._g14_b_unstated, self._g14_b_car, self._g14_b_override,
                     self._g14_b_verdict, self._g14_b_cheaper, self._g14_b_fresh_offers)
        elif self.left() > 5:
            steps = (self._g14_r_trail, self._g14_r_audit, self._g14_r_leads)
        else:
            return None
        return self._g14_first(job, steps)

    def _g14_b_hunt(self, job: dict[str, Any]) -> str | None:
        job['hunted'] = self.hunt_cheaper()
        wider = self.wider_hold()
        return wider + '\nCart now: ' + self.cart_text() if wider else None

    def _g14_b_reread(self, job: dict[str, Any]) -> None:
        job['reread'] = self.reread_offers()
        if self.note_sold:
            self.find_sold()

    def _g14_b_cart_look(self, job: dict[str, Any]) -> str | None:
        tool = self.roles.get('cart_view')
        if not (self.cart_touched and self.cart_seen_at < self.cart_at and tool and self.left() > 5 and not self.rushed()):
            return None
        faults: list[str] = []
        view_args = _shape({}, self.schemas.get(self.roles['cart_view']), 'args', faults)
        if faults:
            return None
        view = self.act(self.roles['cart_view'], view_args)
        if view is None:
            return None
        self.cart_view = view.get('observation')
        self.cart_seen_at = self.step
        change = self.observe(self.roles['cart_view'], self.cart_view)
        if not change:
            return None
        self.queued = []
        return self._g14_hold('A look at the cart turned up a change: ' + change) + '\nTake the change into account, correct the cart if needed, then check out.'

    def _g14_b_sold(self, job: dict[str, Any]) -> str | None:
        swap = self.swap_hint()
        held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
        if not (swap or held):
            return None
        if not swap:
            self.held += 1
        return self._g14_hold(swap or f"{', '.join(held)} in the cart can no longer be bought: swap that line out first.")

    def _g14_b_chain(self, job: dict[str, Any]) -> str | None:
        job['chain'], hold = self.trail()
        return self._g14_hold(job['chain']) if hold else None

    def _g14_b_fit(self, job: dict[str, Any]) -> str | None:
        if self.fit_holds >= 2 or self.left() <= self.tail() + 3:
            return None
        misfit = self.fit_faults()
        if not misfit:
            return None
        self.fit_holds += 1
        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
        return self._g14_hold("A part in the cart is not compatible with the cart's device: " + '; '.join(misfit[:3]) + '. Replace the part (or the device), then check out.')

    def _g14_b_lines(self, job: dict[str, Any]) -> str | None:
        if self.line_holds >= 2 or self.left() <= self.tail() + 2:
            return None
        faults = self.line_faults()
        if faults and all((' is held at ' in f for f in faults)):
            fixed = self.fix_counts(self.count_fixes())
            if fixed:
                faults = self.line_faults()
                if not faults:
                    self.line_holds += 1
                    return self._g14_hold('Cairn corrected the cart quantities on its own: ' + '; '.join(fixed) + '. Fix any line the store refused, then check out.')
        if not faults:
            return None
        self.line_holds += 1
        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
        return self._g14_hold('The cart still differs from the request: ' + '; '.join(faults[:6]) + '. Correct those lines (adding or removing units the way the cart tool does it), then check out.')

    def _g14_b_fill(self, job: dict[str, Any]) -> None:
        lines = job['lines'] = self.cart_ids()
        job['read'] = self.fill_facts(lines, self.tail() + (1 if len(lines) >= 2 else 3))

    def _g14_b_unstated(self, job: dict[str, Any]) -> str | None:
        if self.unstated_holds >= 1 or self.left() <= self.tail() + 3 or self.rushed():
            return None
        unstated = self.unstated_limits()  # UL
        if not unstated:
            return None
        self.unstated_holds += 1
        self.trace.append({'kind': 'unstated_hold', 'lines': unstated[:4]})
        return self._g14_hold('; '.join(unstated[:4]) + '. An unstated value does not satisfy a limit: swap that line for a listing whose attributes show the property inside the limit, then check out.')

    def _g14_b_car(self, job: dict[str, Any]) -> str | None:
        if self.car_holds >= 1 or self.left() <= self.tail() + 3 or self.rushed():
            return None
        wrong = self.car_faults()  # CF4
        if not wrong:
            return None
        self.car_holds += 1
        self.trace.append({'kind': 'car_hold', 'lines': wrong[:4]})
        return self._g14_hold('; '.join(wrong[:4]) + '. Each such line needs a listing whose own fit text covers this car: swap it in, then order.')

    def _g14_b_override(self, job: dict[str, Any]) -> str | None:
        moved = self.override_hint()  # OV
        if not moved:
            return None
        seen = "Attributes Cairn read for these listings (one step apiece):\n" + '\n'.join(job['read']) + '\n' if job['read'] else ''
        return self._g14_hold(seen + moved, 'Not run yet. ')

    def _g14_b_verdict(self, job: dict[str, Any]) -> str | None:
        self.show_links(job['lines'], self.tail() + 1)
        job['read'] += self.cheaper_rows()
        verdict = self.check_cart()
        if not verdict:
            return None
        read, hunted = job['read'], job['hunted']
        seen = "Attributes Cairn read for the check (one step apiece):\n" + '\n'.join(read) + '\n' if read else ''
        if hunted:
            seen = hunted + ' ' + self.below_cart() + '\n' + seen
        return self._g14_hold(job['chain'] + seen + verdict, 'Not run yet. ')

    def _g14_b_cheaper(self, job: dict[str, Any]) -> str | None:
        self.swapped_now = False  # SW
        cheap = self.cheaper_fit_hint()  # CS
        if not cheap:
            return None
        if not self.swapped_now:
            return self._g14_hold(cheap)
        job['notes'].insert(0, '\n' + cheap)
        return None

    def _g14_b_fresh_offers(self, job: dict[str, Any]) -> str | None:
        if not job['reread'] or self.cost <= SPEND_SOFT:
            return None
        return "Not run yet (no step used). Latest state of every listing held in the cart:\n" + job['reread'] + '\nOrder again once every line still has stock and a price within the budget; if one does not, replace it first.'

    def _g14_r_trail(self, job: dict[str, Any]) -> str | None:
        pointed = self.refuse_trail()
        return 'Not run yet (no step used). ' + pointed if pointed else None

    def _g14_r_audit(self, job: dict[str, Any]) -> str | None:
        self.hunt_fits()
        review = self.audit()
        return 'Not run yet (no step used). ' + review if review else None

    def _g14_r_leads(self, job: dict[str, Any]) -> str | None:
        if not self.leads or self.left() <= 4:
            return None
        step = {0: self._g14_r_point_leads, 1: self._g14_r_confirm_lead}.get(self.refuse_blocks)
        return step() if step else None

    def _g14_r_point_leads(self) -> str:
        self.refuse_blocks = 1
        self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
        return 'Not run yet (no step used). Possible fits came up in the review: ' + ', '.join(self.leads) + '. Look at them where needed, cart the strongest fit (the pick rule applies) and order. Declining is right only when each one plainly fails some requirement; then name it in your text and repeat the ending call.'

    def _g14_r_confirm_lead(self) -> str | None:
        self.refuse_blocks = 2
        first = self.leads[0]
        self.check_cart([first], SPEND_HARD)
        if not (self.verdicts.get(self.check_key([first])) and first in self.leads):
            return None
        self.confirmed = first
        self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
        return f'Not run yet (no step used). Checked on its own, {first} satisfies every requirement per its records. Cart it in the requested quantity and order; decline only when its record clearly breaks one.'

    def _g14_late_decline(self, job: dict[str, Any]) -> str | None:
        if not self.is_refuse(job['name'], job['args']) or self.left() > 5 or self.late_held:
            return None
        lines = list(filter(lambda i: not self.sold_out(i), self.built))
        fresh = job['fresh'] = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        carted = set(self.cart_ids())
        missing = [i for i in lines if i not in carted or (i in self.dropped and not fresh)]
        if len(lines) < len(self.items) or len(missing) + 1 > self.left():
            return None
        self.late_held = True
        self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
        redo = f"Return {', '.join(missing)} to the cart at the requested quantities, then check out." if missing else 'Check out the current cart.'
        return 'Not run yet (no step used). Declining scores nothing whenever a fitting listing exists, and being low on steps is no proof that none does. ' + redo + ' Decline only if the searches proved some requested item lacks any qualifying listing; after that, repeat the ending call.'

    def _g14_push_search(self, job: dict[str, Any]) -> str | None:
        if not self.is_refuse(job['name'], job['args']) or self.leads or self.pushed or self.left() <= self.max_steps // 3:
            return None
        self.pushed = True
        return f"Not run yet (no step used). {self.left()} steps remain, so try searches you have not run yet before ending: list each requested item's category and page through it using the token or offset the store gives back, query single value words and other names for the property, and open listings that could fit. Only then end without a purchase."

    def _g14_salvage(self, job: dict[str, Any]) -> str | None:
        if not (self.is_refuse(job['name'], job['args']) and self.salvage_set()):  # SP
            return None
        said = self.finish(again=True)
        tail = ' [episode finished]' if self.finished else '\nCart now: ' + self.cart_text()
        return 'Not run (Cairn acted instead): the opened pages already held a full set obeying the fit rules, which Cairn carted and ordered itself. ' + (said or '') + tail

    def _g14_refuse_cart(self, job: dict[str, Any]) -> str | None:
        if not self.is_refuse(job['name'], job['args']):
            return None
        done = self.complete_cart()
        if done:
            return done + (' [episode finished]' if self.finished else '')
        job['args'] = self.quiet(job['name'], job['args'])
        return None

    def _g14_late_question(self, job: dict[str, Any]) -> str | None:
        if not self.is_buy(job['name'], job['args']):
            return None
        late = self.late_question()
        return self._g14_hold(late, 'Not run yet. ') if late else None

    def _g14_gate(self, job: dict[str, Any]) -> str | None:
        if not job['ending'] or self.left() < 2:
            return None
        go, said = self.gate('order' if self.is_buy(job['name'], job['args']) else 'end')
        if not go:
            return self._g14_hold(said, 'Not run yet. ')
        if said:
            job['notes'].insert(0, '\n[Cairn] ' + said)
        return None

    def _g14_execute(self, job: dict[str, Any]) -> str | None:
        name, args = job['name'], job['args']
        if job['ending'] and self.is_buy(name, args) and self.left() == 2 and not self.anchor_done:
            self.anchor_view()  # C
        reply = job['reply'] = self.act(name, args)
        if reply is None:
            return 'The call did not go through (network or server error). Trying again is fine.'
        job['seen'] = reply.get('observation')
        return None

    def _g14_compare_seen(self, job: dict[str, Any]) -> None:
        many, args = job['many'], job['args']
        if not (many and isinstance(args.get(many), list) and len(args[many]) > 1):
            return
        sent = len(args[many])
        shown = _dump(job['seen'])
        if any((str(i) in shown for i in args[many])):
            self.compare_ok = max(self.compare_ok, sent)
            return
        self.compare_bad = min(self.compare_bad or 999, sent)
        job['notes'].append(f'\n[Cairn] Asked for {sent} listings at once, the side-by-side view came back empty: send no more than {self.compare_room()} per call.')

    def _g14_remember(self, job: dict[str, Any]) -> None:
        name, args, seen = job['name'], job['args'], job['seen']
        self.recent = [*self.recent, job['mark']][-3:]
        self.id_args |= {(name, key) for key, value in args.items() if self._g14_all_rows(value)}
        self.absorb(seen)
        rows = self.records_in(seen) if name == self.roles.get('compare') else []
        if rows:
            self.facts_from[name] = self.facts_from.get(name, False) or any((_fact_block(r) for _, r in rows))
        if name in job['readers'] - {self.roles.get('search')}:
            self.note_view(name, seen)
        plain = job['plain'] = {tool for role, tool in self.roles.items() if role != 'search'}
        browsing = name not in plain and not job['ending']
        pieces = ((browsing, self.more_pages), (job['fresh'] and not job['ending'], self.last_page_first), (browsing, self.shorter_search))
        job['paged'] = ''.join([fn(name, args, seen) for wanted, fn in pieces if wanted])

    def _g14_all_rows(self, value: Any) -> bool:
        values = value if isinstance(value, list) else [value]
        return bool(values) and all((isinstance(x, str) and x in self.rows for x in values))

    def _g14_learn_tool(self, job: dict[str, Any]) -> None:
        name, seen = job['name'], job['seen']
        if not self.single_target(name) or name in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
            return
        listing = isinstance(seen, dict) and self.ident(seen)
        if not listing:
            return
        self.note_facts(name, bool(_fact_block(seen)))
        if name == self.roles.get('open') and self.facts_from.get(name) is False:
            self.roles['open'] = next((t for t in self.schemas if self._g14_page_tool(t, name)), None) or self.roles['open']
        if self.facts_from.get(name) is False and name != self.roles.get('open') and not self.roles.get('offer'):
            self.roles['offer'] = name
        self.offer_seen.update({listing: self.step} if name == self.offer_tool() else {})

    def _g14_page_tool(self, tool: str, current: str) -> bool:
        busy = (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search'))
        return tool != current and bool(self.single_target(tool)) and self.facts_from.get(tool) is not False and tool not in busy and not self.lists_things(tool)

    def _g14_cart_after(self, job: dict[str, Any]) -> None:
        name, args, seen = job['name'], job['args'], job['seen']
        if name != self.roles.get('cart'):
            return
        self.note_cart(name, args)
        self.cart_said, self.recent = _dump(seen), []
        first = next((c for c in map(_count, [v for v in args.values() if not isinstance(v, str)]) if c is not None), None)
        if first is not None:
            named = [v for v in args.values() if isinstance(v, str) and v in self.rows]
            if first > 0:
                self.dropped.difference_update(named)
            else:
                self.dropped.update([v for v in named if not _in_rows(seen, v)])
        kept = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
        self.built = kept if len(kept) >= len(self.built) else self.built

    def _g14_search_after(self, job: dict[str, Any]) -> None:
        name, args, seen = job['name'], job['args'], job['seen']
        searching = name == self.roles.get('search')
        if searching and self.names_code(args):
            tries = self.code_tries
            tries.append(bool(self.records_in(seen)))
            self.codes_dead = not any(tries) if len(tries) >= 2 else False
        if searching and self.records_in(seen):
            self.searched = [*self.searched, dict(args)][-8:]
        if name == self.roles.get('cart_view'):
            self.cart_view, self.cart_seen_at = seen, self.step
        if self.is_buy(name, args):
            self.buys += 1
            self.refused_at = self.refused_at if job['reply'].get('finished') else self.step

    def _g14_compose(self, job: dict[str, Any]) -> str:
        name, args, seen, reply, readers = job['name'], job['args'], job['seen'], job['reply'], job['readers']
        quiet = job['ending']
        parts = [self.show(seen), job['paged'], *job['notes']]
        parts += [self.cheapest_here([seen] + self.paged_obs) + self.cap_ignored(name, args, seen)] if name not in job['plain'] and not quiet else []
        kept = ''.join(parts)
        parts.append('\n' + proof if (proof := self.evidence(seen)) else '')
        parts.append(self.car_lines([seen] + (self.paged_obs if name == self.roles.get('search') else [])) if name in readers and not quiet else '')  # CF2
        is_cart = name == self.roles.get('cart')
        zero = is_cart and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))) and any((isinstance(v, str) and v in self.rows and _in_rows(seen, v) for v in args.values()))
        parts.append('\n[Cairn] That line is still in the cart after a count of 0: this tool adds the count to the line, so take units off with a negative count.' if zero else '')
        self.note_sold = False
        noted = self.loose_lines(seen)
        meaning = noted and self.take_note(noted)
        parts.append('\n[note read] ' + meaning + '. Update your picks with this change before you check out.' if meaning else '')
        parts.append(self._g14_sold_lines(name, args, seen, reply))
        self.note_sold = False
        parts.append('\n' + swap if (swap := self.swap_hint()) else '')
        misfit = self.fit_faults() if is_cart and self.bound and not self.finished else []
        parts.append('\n[Cairn] These cart lines are not compatible yet: ' + '; '.join(misfit) if misfit else '')
        queued, self.queued = self.queued, (self.queued and [])
        parts.append("\n[notes seen on Cairn's own calls] " + ' | '.join(queued) if queued else '')
        if name in readers:
            self.read_cache[job['mark']] = (self.version, self.step, kept)
        return ''.join(parts) + (' [episode finished]' if reply.get('finished') else '')

    def _g14_sold_lines(self, name: str, args: dict[str, Any], seen: Any, reply: dict[str, Any]) -> str:
        buying = self.is_buy(name, args)
        refused = buying and not reply.get('finished')
        triggers = (refused and any((line in self.spoken(seen) for line in self.sold_lines)),
                    self.note_sold and (refused or (not self.note_named and not buying)))
        found = self.find_sold() if any(triggers) and self.cart_ids() and not self.finished else ''
        return "\n[Cairn re-read the cart's lines after the refusal]\n" + found if found else ''

    def can_refuse(self) -> bool:
        refuse = self.roles.get('refuse')
        problems: list[str] = []
        if refuse:
            _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', problems)
        return bool(refuse) and problems == []

    def _g15_closed(self) -> bool:
        return self.finished or self.left() <= 0

    @staticmethod
    def _g15_crash_note(exc: BaseException) -> dict[str, Any]:
        return {'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}

    def _g15_review_cart(self) -> bool:
        # A multi-item cart that cannot be completed is only kept when ending empty-handed is impossible.
        if not self.fill_short_cart() and self.can_refuse():
            self.trace.append({'kind': 'short_cart_refused', 'lines': self.cart_ids()})
            return False
        if len(self.cart_ids()) < 2:
            return True
        self.fill_facts(self.cart_ids(), self.tail())
        if self.left() >= 3 and self.model_room() > 5:
            self.show_links(self.cart_ids(), self.tail())
        return True

    def _g15_gated_act(self, name: str, args: dict[str, Any], gate: str, kind: str) -> tuple[bool, Any]:
        if self.left() >= 2:
            self.gate(gate)
        if self._g15_closed():
            return True, None
        reply = self.act(name, args)
        self.trace.append({'kind': kind, 'reply': _clip(reply, 200)})
        return False, reply

    def _g15_try_buy(self, plan: dict[str, Any]) -> str | None:
        buy = plan['buy']
        if not (plan['whole'] and buy):
            return None
        if not plan['holding'] and not (self.cart_touched and not plan['fresh'] and self.left() >= 2):
            return None
        problems: list[str] = []
        args = _shape(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', problems)
        if problems:
            return None
        stopped, reply = self._g15_gated_act(buy, args, 'order', 'forced_buy')
        if stopped or self._g15_closed():
            return ''
        if reply is None or not plan['again']:
            return None
        self.buys += 1
        self.refused_at = self.step
        self.observe(buy, reply.get('observation'))
        return 'the store turned the checkout down: ' + self.show(reply.get('observation'))

    def _g15_try_refuse(self, plan: dict[str, Any]) -> str | None:
        refuse = plan['refuse']
        if not refuse:
            return None
        problems: list[str] = []
        args = _shape(self.quiet(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', problems)
        if problems:
            return None
        stopped, reply = self._g15_gated_act(refuse, args, 'end', 'forced_refuse')
        if stopped or reply is None or self.finished:
            return ''
        self.observe(refuse, reply.get('observation'))
        return 'the store would not let the episode end without an order: ' + self.show(reply.get('observation'))

    def finish(self, again: bool=False) -> str:
        if self.broke or self._g15_closed():
            return ''
        plan: dict[str, Any] = {'buy': self.roles.get('buy'), 'refuse': self.roles.get('refuse'), 'again': again}
        self.salvage_set()  # SP
        if self._g15_closed():
            return ''
        plan['fresh'] = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        plan['holding'] = bool(self.cart_ids() if plan['fresh'] else self.cart_lines)
        plan['whole'] = self._g15_review_cart() if plan['holding'] and len(self.items) >= 2 else True
        if self._g15_closed():
            return ''
        for attempt in (self._g15_try_buy, self._g15_try_refuse):
            verdict = attempt(plan)
            if verdict is not None:
                return verdict
        return ''

    def _g15_shorten(self, text: str) -> str:
        found = dict.fromkeys(re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', text))
        known = [key for key in found if key in self.rows][:20]
        return text[:400] + f"...(earlier output cut short; listing ids it contained: {', '.join(known)})"

    def trim(self) -> None:
        if sum(len(_dump(entry.get('content'))) for entry in self.chat) <= HISTORY_LIMIT:
            return
        results = [entry for entry in self.chat if entry.get('role') == 'tool']
        for entry in results[:max(1, len(results) // 2)]:
            body = entry.get('content') or ''
            if len(body) > 600:
                entry['content'] = self._g15_shorten(body)

    def _g15_setup(self, ctx: dict[str, Any]) -> None:
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        if opening is not None:
            opening = _restore(opening)
            self.absorb(opening)
        ctx['opening'] = opening
        ctx['moved'] = '' if self.finished else self.first_moves()
        self.env_gap = ENV_GAP

    def _g15_open_chat(self, ctx: dict[str, Any]) -> None:
        sections = ["SHOPPER'S REQUEST:\n" + self.request]
        if ctx['opening'] is not None:
            sections.append('WHAT THE STORE SHOWS FIRST:\n' + self.show(ctx['opening']))
        if ctx['moved']:
            lead = ('Cairn already ran one search per item and opened a few of the listings it found (each used one step):\n' + ctx['moved']
                    + '\nCheck these against what the shopper requires; only search again for items that are not covered yet.')
        else:
            lead = 'Begin with a category search for every item, then open a few listings completely so the interviewer learns which properties the listings expose.'
        sections.append(lead + '\n\n' + self.status())
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': '\n\n'.join(sections)}]

    def _g15_close_out(self, ctx: dict[str, Any]) -> None:
        if self.finished:
            return
        turned = self.finish(again=True)
        retry = bool(turned) and not (self.finished or self.broke or self.no_credit) and self.left() > 1
        if retry:
            self.chat.append({'role': 'user', 'content': '[Cairn] An attempt to close the episode failed, because ' + turned + '\nAct on the store\'s reply, then wrap up.\n' + self.status()})
            self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
        if not self.finished:
            self.finish()

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        ctx: dict[str, Any] = {}
        phases = (self._g15_setup, self._g15_open_chat, lambda _ctx: self.work(LOOP_CALLS, SPEND_HARD), self._g15_close_out)
        for phase in phases:
            phase(ctx)
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls})
        return self.trace

    def _g15_safe_turns(self, loop_cap: int, spend_cap: float) -> bool:
        try:
            self.turns(loop_cap, spend_cap)
        except Exception as exc:
            self.trace.append(self._g15_crash_note(exc))
            self.answer_open_calls()
            return False
        return True

    def work(self, loop_cap: int, spend_cap: float) -> None:
        tries = CRASH_TURNS
        while tries > 0 and not self._g15_safe_turns(loop_cap, spend_cap):
            tries -= 1

    def answer_open_calls(self) -> None:
        for back, entry in enumerate(reversed(self.chat)):
            if entry.get('tool_calls'):
                break
        else:
            return
        answered = {later.get('tool_call_id') for later in self.chat[len(self.chat) - back:]}
        self.chat.extend({'role': 'tool', 'tool_call_id': call['id'], 'content': 'An error interrupted this call, so it may not have taken effect. Check the current state before trying it again.'}
                         for call in [c for c in entry['tool_calls'] if c['id'] not in answered])
        self.chat.append({'role': 'user', 'content': self.status()})

    # ---- per-turn stages: each returns None (go on), 'again' (next turn) or 'stop' (leave the loop)
    def _g15_may_turn(self, loop_cap: int, spend_cap: float) -> bool:
        if self.finished or self.broke or self.left() <= 0:
            return False
        if self.loop_calls >= loop_cap or self.cost >= spend_cap:
            return False
        return self.clock() < LAST_TURN or (not self.cart_ids() and self.model_room() > EMPTY_ROOM)

    def _g15_stage_interview(self, turn: dict[str, Any]) -> str | None:
        if not self.ready_to_ask():
            return None
        summary = self.interview()
        if summary:
            self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
        return 'again'

    def _g15_stage_trim(self, turn: dict[str, Any]) -> str | None:
        self.trim()
        return None

    def _g15_stage_model(self, turn: dict[str, Any]) -> str | None:
        self.loop_calls += 1
        order = [SPARE, LIGHT] if LIGHT in self.slow_models else [LIGHT, SPARE]
        turn['message'] = self.think(self.chat, order, 3000, self.offered(), LOOP_WAIT)  # LM
        if turn['message'] is not None:
            return None
        self.misses += 1
        if self.no_credit or self.calls >= ALL_CALLS:
            return 'stop'
        if self.misses <= 2:
            return 'again'
        if self.clock() >= OUTAGE_BY:
            return 'stop'
        time.sleep(BRIEF_GAP)
        return 'again'

    def _g15_one_call(self, number: int, call: dict[str, Any]) -> dict[str, Any]:
        fn = call.get('function')
        if not isinstance(fn, dict):
            fn = {}
        raw = fn.get('arguments')
        if isinstance(raw, dict):
            parsed = raw
        elif isinstance(raw, str) and raw.strip():
            parsed = _json_obj(raw)
        else:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        return {'id': str(call.get('id') or f'k{self.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _dump(parsed)}}

    def _g15_read_message(self, turn: dict[str, Any], message: dict[str, Any]) -> None:
        listed = message.get('tool_calls') or []
        turn['calls'] = [self._g15_one_call(number, call) for number, call in enumerate(listed) if isinstance(call, dict)]
        said = message.get('content')
        turn['content'] = said if isinstance(said, str) else (_dump(said) if said else '')

    def _g15_stage_parse(self, turn: dict[str, Any]) -> str | None:
        self._g15_read_message(turn, turn['message'])
        return None

    def _g15_stage_light_retry(self, turn: dict[str, Any]) -> str | None:
        # A light-model reply with no tool call gets one more try on the spare model.  # CL
        if turn['calls'] or self.last_model != LIGHT or LIGHT in self.slow_models or self.model_room() <= EMPTY_ROOM:
            return None
        self.light_idle += 1
        if self.light_idle >= 2:
            self.slow_models.add(LIGHT)
        self.trace.append({'kind': 'callless_light', 'text': turn['content'][:200], 'at': round(self.clock(), 1)})
        self.loop_calls += 1
        retry = self.think(self.chat, [SPARE], 3000, self.offered())
        if retry is not None and retry.get('tool_calls'):
            self._g15_read_message(turn, retry)
        return None

    def _g15_stage_idle(self, turn: dict[str, Any]) -> str | None:
        if turn['calls']:
            turn['idle'] = 0
            return None
        turn['idle'] += 1
        self.chat.append({'role': 'assistant', 'content': turn['content'] or '(no call)'})
        self.chat.append({'role': 'user', 'content': 'Continue by calling a tool.\n' + self.status()})
        return 'stop' if turn['idle'] >= 3 else 'again'

    def _g15_stage_ask_first(self, turn: dict[str, Any]) -> str | None:
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return None
        binding = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
        calls = turn['calls']
        if not any(c['function']['name'] in binding for c in calls):
            return None
        if self.left() <= 8 or self.clock() > ASK_BY:
            return None
        self.chat.append({'role': 'assistant', 'content': turn['content'] or None, 'tool_calls': calls})
        self.chat.extend({'role': 'tool', 'tool_call_id': c['id'], 'content': 'Skipped (cost no step): the shopper gets questions before this.'} for c in calls)
        summary = self.interview()
        self.chat.append({'role': 'user', 'content': (summary or 'No questions were sent: the interviewer found nothing to ask.') + '\n' + self.status()})
        return 'again'

    def _g15_stage_dispatch(self, turn: dict[str, Any]) -> str | None:
        calls = turn['calls']
        self.chat.append({'role': 'assistant', 'content': turn['content'] or None, 'tool_calls': calls})
        self.trace.append({'kind': 'turn', 'text': turn['content'][:300], 'calls': [c['function'] for c in calls]})
        replies = self.run_calls(calls)
        for cid, text in replies[:-1]:
            self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})
        for cid, text in replies[-1:]:
            self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text + '\n' + self.status()})
        return 'again'

    def turns(self, loop_cap: int, spend_cap: float) -> None:
        turn: dict[str, Any] = {'idle': 0}
        pipeline = (self._g15_stage_interview, self._g15_stage_trim, self._g15_stage_model, self._g15_stage_parse,
                    self._g15_stage_light_retry, self._g15_stage_idle, self._g15_stage_ask_first, self._g15_stage_dispatch)
        while self._g15_may_turn(loop_cap, spend_cap):
            verdict = next((outcome for outcome in (stage(turn) for stage in pipeline) if outcome), None)
            if verdict == 'stop':
                break

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Cairn(problem_data)
    except Exception as exc:
        return [Cairn._g15_crash_note(exc)]
    try:
        return list(episode.run())
    except Exception as exc:
        episode.trace.append(Cairn._g15_crash_note(exc))
    try:
        if not episode.finished:
            episode.finish()
    except Exception as again:
        episode.trace.append(Cairn._g15_crash_note(again))
    return list(episode.trace)
__all__ = ['agent_main']

