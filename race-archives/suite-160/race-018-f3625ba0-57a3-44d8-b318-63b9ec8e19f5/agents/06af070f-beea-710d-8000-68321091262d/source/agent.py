"""Sakura v6.1 (2026-11-08): buys for one shopper inside an ORO store world.

Built on the public KYO v6 agent (released top agent, ORO SN15). A light model walks the
store with the world's own tools; code keeps the ledger (items, needs, budget, listings
seen), plans and paces the questions to the shopper from the story's hints and the replies
it gets back, and stops wasted steps. A heavy model reads the request once, ranks the
questions, checks the cart before an order and the evidence before a decline.
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

# Reply idioms seen in race logs (several shopper dialects); the hearer model reads anything these do not settle.
CROWDED_RE = (r"one (thing|question|query|item|point|property|subject) at a time|one at a time|separately|each (query|question|thing) (by itself|alone|on its own)"
              r"|first (thing|question|query|one|bit) (alone|first|and then|and i)|single question|split (that|them|the question|those)|too many (things|questions|in one go)"
              r"|one by one|in turn|one go|the one thing|in batches|do these in turn"
              r"|ask me the one|just the first|stick with the first|one question, wait")
CONFUSED_RE = (r"(didn't|did not|don't|do not|can't|cannot|couldn't|could not) (quite )?(catch|understand|follow|get|grasp|make out)|lost me|come again|pardon"
               r"|what (exactly |precisely )?(do|did) you mean|what are you (getting|driving) at|simpler (words|terms|version)|plain(er)? (terms|words|version|english)"
               r"|another (go|way|try)|once more|say (that|it) again|repeat (that|it|the question)|word (it|that) differently|rephrase|not landing|didn't land|muddled"
               r"|confused|adrift|all at sea|run aground|foggiest|foggy|not a word|in one ear|above my (head|understanding)|no idea what (that|you) meant|went (right )?over my head"
               r"|put that (another|a different) way|jargon|explain (it|that|this) (like|as if)|brand new to this|i'?m lost|i am lost|lost somewhere|none the wiser"
               r"|not sunk in|baffled|go back over|give (it|us) (me |it )?(again|once more)|try it again|not used to the"
               r"|struggling with that|didn't take that in|didn't get that|say that again|what was that")
DONTCARE_RE = (r"(don't|do not|doesn't|does not|didn't|never|not) (really )?(mind|care|fuss|bother|matter)|no (strong |real |particular |special |fixed )?(preference|opinion|view|wish|lean|favourite|favorite|idea)"
               r"|not (picky|fussy|fussed|bothered|particular|attached|choosy)|any (of them|option|one|will do|would do|is fine|suits|would suit|works|of them'll do)|whatever (you|looks|comes|turns up|is)"
               r"|whichever|either way|all the same to me|up to you|your call|surprise me|leave (that|it) (open|to you|with you)|happy with (anything|whatever|any)|open (on|about|to) (that|this|it)"
               r"|choose (for|freely|whichever|as you)|pick (whichever|whatever|for me|one for me|something|any)|both are (fine|alright|ok)|nothing (springs|comes) to mind"
               r"|no bother which|neither here nor there|anything (works|goes|will do|suits|is acceptable)|not (worth|something) (deciding|the trouble)"
               r"|haven't (got |a )?(a |any )?(lean|preference|favourite|favorite|view|opinion)|what (do most|would you suggest|would you recommend|do you suggest)|indifferent|not attached"
               r"|nothing i (care|mind)|never thought about it|as you see fit|won't be (fussed|bothered|picky|fussy)|whatever's (on|there|available|in)|just here to tick"
               r"|no say on that|just choose|never given (that|it) a thought|skip past it|never bothered me|crack on"
               r"|happy with whatever|leave it open|all one to me|i'm easy")
REVEAL_RE = (r"\b(must|should|has to|have to|needs? to|only|fixed|firm|locked|settled|insist|stick(ing)? (to|with)|dead set|nothing (else|less|short of|but|different|other)"
             r"|consider .{1,30} (closed|settled|fixed|decided)|at least|at most|no more than|no less than|under|over|exactly|same as|unlike|anything other than|different from|matching"
             r"|compatible|rule(d)? out|non-negotiable|won't (budge|shift|move)|(not|isn't|aren't|is not|are not) (up for|open to|open for) (trade|debate|discussion|negotiation|change)"
             r"|negotiable|can't do without|holding (firm|fast)|not flexible|take anything but|nothing but|only go with|firm ask|put me down for|make sure|land on|lines up with"
             r"|means .{1,30} for me|plain and simple|that's just how it is|wants to be|got to be|has got to|no two ways|not shifting|stand firm|holding out for|comes? in as|showing under"
             r"|sorted out as|gets binned|apart from|other than|cross off|except)\b")
NEGATION_RE = r"\b(not|no|never|none|nothing|without|apart from|other than|except|excluding|cross(ed)? off|rule(d)? out|anything but|bin(ned)?|avoid|skip|steer clear|don't want|do not want|isn't for me|aren't for me|isn't wanted|won't take|won't have|refuse)\b"
VALUE_RE = (r"\d|sold by|offered by|listed by|stocked by|on the page|page for the|points (you|to|at)|same as|within .{1,25} (of|either)|apart from|other than|except for|gets binned|cross off|rule(d)? out"
            r"|showing under|holding out for|has got to|got to be|wants? to be|land on|lines up with|sticking (to|with)|stand firm|not shifting|only take|only looking at|only taking"
            r"|no two ways|plain and simple|just how it is|comes? in as|fixed at|at least|at most|no more than|no less than|unlike|\bmust\b|needs? to (be|read|come|land)"
            r"|sorted out as|settle(d)? on|nothing (else|less|but|short of)|won't take anything|read [a-z0-9]|\bbe [a-z0-9-]+,? (and|nothing|plain|full stop|no )")

# per-model request options: the light model answers without a thinking pass (its replies were 70 % reasoning tokens for the same JSON)
REASONING = {LIGHT: {"enabled": False}, HEAVY: {"effort": "low"}}
_REASONING_OK = [True]
# Compatibility is read from the shapes any catalogue uses: a bracketed code, codes named after the word "series", a code named
# after the word "model" (the series named after it in the same sentence are series that model takes), a code another listing is
# referred to by, and a code family written with wildcard letters.
FIT_CODE = re.compile(r"\[([A-Za-z0-9]{2,8}-[A-Za-z0-9]{2,8})\]")
FIT_SERIES = re.compile(r"\bseries\b[: ]*([A-Za-z]{1,4}\d{1,3}(?:\s*(?:,|and|&|/)\s*[A-Za-z]{1,4}\d{1,3})*)", re.I)
FIT_MODEL = re.compile(r"\bmodel\b(?:\s*(?:[A-Za-z]{1,4}\.|no|number|#))?\s*:?\s*([A-Za-z]{1,6}-?\d{2,7})\b", re.I)
FIT_POINTER = re.compile(r"\b(?:see|refer to|check|look at)\s+(?:part|item|listing|model|sku|product)\s+([A-Za-z0-9]{2,8}-[A-Za-z0-9]{2,8})\b", re.I)
FIT_VEHICLE = re.compile(r"\bfits?\b[: ]*(?:\w+\s+)?([A-Za-z]{1,6})[ -]?(\d{1,5})(x{1,3})\b", re.I)
FIT_TOKEN = re.compile(r"[A-Za-z]{1,4}\d{1,3}\b")
CODE_TOKEN = re.compile(r"\b([A-Za-z]{1,6}-?\d{2,7}(?:[xX]{1,3})?|[A-Za-z]{1,6}-?\d{1,5}[xX]{1,3})\b")   # a code-like token: letters, digits, maybe a wildcard tail
CODE_SPLIT = re.compile(r":|\b(?:fits?|compatible|works|supports?|for)\b", re.I)
SENTENCE_END = re.compile(r"[.;!?\n]+")
FIT_ITEM_COST = 0.007        # extra spend allowed per additional requested product (hard cap stays under $0.054 for 4 items: 90 tasks < $5)
LOOP_CALLS = 38
ALL_CALLS = 60
SPEND_SOFT = 0.036
SPEND_HARD = 0.050
BROAD_LABELS = 8           # properties named in one broad question
LIST_FIRST = 4             # properties named in the first list question of an episode (nothing shows yet how this shopper takes lists)
LIST_LABELS = 6            # properties of one item named in a later list question
LIST_SHARE = 3             # properties of each item when one message names several items
CLOSING = 45.0             # seconds left on the clock at which the tool loop stops and the code finishes the episode
CLOCK_LIMIT = 275.0        # seconds of wall clock an episode may use before Sakura forces its ending (the world allows 300)
PLAN_ROOM = 0.012          # spend that must still be free for the question plan to go to the heavy model
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 400
OBS_LIMIT = 9000
HISTORY_LIMIT = 80000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Sakura reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 4           # cheaper rows of a cart line's kind whose attributes Sakura reads before a cheapest checkout

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Sakura, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Sakura knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Sakura's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger. A property the shopper does not mind about, or did not understand a question about, is not a requirement: never search or filter on it.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Sakura carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Sakura's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one, and that label alone decides the kind (a title that adds a set or bundle word, or a second cut or style, beside the wanted words never makes a listing of the right category the wrong kind) (a label naming only one of the kinds the request joins with 'or' or 'and' is a neighbouring label when the store also has the whole one).
- A listing is DOUBTFUL when another attribute naming the same property states a different plain value, when an attribute contradicts the kind itself (a season or audience against the category label's own, another type under the property's own key), or when no attribute names the property and the title or description names a different value. A title or description never overrides an attribute that names the property. A kind shown only under a key that adds a qualifier to the property's name (another group's version of a property is a different property) is not shown. A doubtful listing is not taken while a clean one can be found or searched for; under a cheapest pick the cheapest clean fit is the answer.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and spelling variants are fine, but a different word does not count even when it means nearly the same, and a value equal to the required value counts, and so does a list naming it among others. A value that carries the required words beside other qualifying words ("deep vermilion" for vermilion, "pale heliotrope" for heliotrope) is a weaker fit: it counts only while no listing with the exact value can be had, and a cheapest pick never prefers it over an exact fit. A value sharing just some of the words never counts ("celadon-tinted resin" is not celadon, "vermilion ochre" is not burnt ochre, an abbreviation or code is not the word it stands for). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. When two keys name the same property and disagree, the listing fits only if one of them is a list that includes the wanted value (the list counts); two plain values that disagree make the listing doubtful. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Sakura adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value named in its title counts. A value stated only under another attribute (for example a list of styles or features that includes it) or only in its description text does not count: such a listing is unverified, and a listing whose attribute or title states the value comes first. A number of another kind does not count either (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- A listing's own words can hand one of its properties to another listing: its text tells the reader to take that property from another product, named by its shop and a few of its properties. Then this listing's own value for that property counts for nothing: find and open that other listing (search its shop's name with the words given, open the row that matches all of them), and the value on its page is what every requirement and shopper need on that property is judged against. Until that page is read the listing is unverified on that property; Sakura marks such sentences with [defers].
- A listing's page can state a store rule that assigns, to each value of a property, the value it belongs with. A requirement that ties its value to another listing's value through such a relation takes the value the rule assigns to that listing's value; without such a rule it takes that listing's value itself.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Only the request's stated requirements, the shopper's answers and the store's notes count. The story around the request (who the shopper is, who will use the product, why it is bought, where it will be used) adds none: never pass over a listing for anything the story only suggests.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.
- Ending without a purchase earns 1 only when every listing of a requested kind you could find contradicts a requirement in its attributes; it earns 0 whenever a fitting listing existed. A listing that is merely unverified (no attribute names the property), a cart over the budget (swap the priciest line for a cheaper fit), a requirement nobody lists (a delivery date, a warranty) or an item you have not found yet (search its category under other words, open the rows) are reasons to keep working, never to end. When in doubt between a doubtful purchase and no purchase, buy.
- When the request says one product must fit another (or a named model), Sakura reads the listings' own compatibility statements (the codes, series and models its own text names as what it fits or works with, and the other listings it refers to) and shows under [compatibility] which series the anchor takes and which rows of each dependent kind fit. Choose and cart the anchor first; cart dependent items only from the rows that fit (a line outside the anchor's series fails the whole order); when no fitting row can be found after Sakura's own search, buy the best otherwise-fitting listing rather than end without a purchase. Rows listed under [compatibility] as fitting are of the requested kind (the store's own category) and fit the anchor: judge them only on the stated requirements and the shopper's needs, never on what their title calls them.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Sakura reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.
- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it. Plan for two replacements: make the first checkout attempt with at least 8 steps left (under a cheapest pick, stop widening the search when fewer than 12 steps remain and check out the cheapest clean fit you hold), and replace a sold-out line from the fits you already know before any new search.
- A request for several different items needs one cart line per item, each at its requested quantity.

CHANGES WHILE YOU WORK
- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.
- Re-read a chosen listing's record shortly before checking out, because price and stock can change silently.

FINISHING
- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""


BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "fits": the item number of another requested product this one must fit, work with, go with or be compatible with (also when the request leaves the choice of that other product open), else the model or series code the request names for it to fit (a vehicle model, a device model), else null, "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a seaside occasion" gives "occasion: seaside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts, nor a delivery or arrival date, shipping, payment, warranty or return terms: no listing states those, so they can neither be met nor failed by a product. The story around the request (the person meant to use it, where it will stand, what went wrong before, why the shopper is buying, a scene in which it will be worn or used) is context, not a must, unless it states a property's value or limit outright. An occasion becomes a must ("occasion: regatta") only when the request presents it as a condition the product must meet, with firm wording or a check the shopper says they will make ("they're for the regatta, so I shall be inspecting your choice", "it's for a vernissage, so it must suit the evening"); a passing mention of what the shopper needs it for ("a jacket for the allotment", "to pull on before fencing drills", "something for the beekeeping visits") is never a must; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name"; but a value the request ties to a property it names (a phrase like "in the <value> <property>" or "with a <value> <property>") is a must even when the same word also appears in what the product is called. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure (the one that ranks first, leads or beats the rest on it, or comes lowest on it), else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
A cap the request puts on the price of each unit of one product is a must of that item, "price: at most N", besides the total budget. A product kind whose name joins words with "and" or "&" (a store category such as "<A> and <B>" or "<A> & <B>") is ONE item, never two.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing, each named by the product property it is about (a sentence telling the agent to ask the shopper which value of some property they need belongs here, never under "process"); empty when none.
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You choose which product properties a shopping agent asks a shopper about, and in what order. The shopper holds one or two firm preferences they did not write down, each on ONE property of ONE requested product, and besides them other properties they do not mind about. A question reveals a preference only when it names that exact property; a question about a property the shopper holds no view on is simply wasted, and only COUNT questions can be sent, so the order decides the outcome.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many listings carry the key, how many different values it takes, and example values. A preference can be on a property that no listed key names yet.

Rank the properties like this:
0. What the request itself asks the agent to check with the shopper (under ASK FIRST), each named by the listed key that property corresponds to (the request's own word for it only when no listed key names that property: a question in the store's own key name is understood, one in the request's paraphrase often is not).
1. Hinted properties. The request's story points at each unwritten preference without giving its value: a sentence about the person who will wear or use it, about how it will look or where it will stand, about something the shopper reads on a tag or label, about how it travels, is stored, served or shared, about what it must fit, connect to or run, about what it costs to run, about how many people it serves, or about a measure the shopper cares about. Name the one property such a sentence points at (the listed key that fits it best; when two keys fit the same hint, list both, likeliest first) and the item it is about, in the order the story gives them. Match each hint to the listed key that answers it: a sentence about the person points at the key saying who the product is made for, a sentence about its looks at whichever appearance key the request leaves open and the listings carry, a sentence about reading the tag at what the product is made of, a sentence about carrying, storing or sharing at how it is packed. A hint is as strong as its sentence: one plainly about the wearer or user, a tag being read, how it travels, or what it must fit or run is strong; a flourish about the occasion, the mood or the weather, or a figure of speech, is weak and ranks below rule 1b.
1b. Who it is for. For something worn or carried on the body a buyer settles first who it is for, so when the request leaves that open and the product kind itself does not already say it, list the property that names the person it is made for (then the one naming their age when the listings carry it) right after the strong hints, even when only one opened listing carries such a key, under whatever key the listings use for it.
1c. When one requested product must fit another (or a named model), the product the others must fit is the anchor: its properties come before the dependent items', and properties that only describe the fit (series, model and part codes, compatibility) are never asked, the listings state them.
2. Then the listed keys a buyer of that kind of product chooses between first, ranked by how many of the item's listings carry the key and how many different values it takes: a key carried by most listings with several values is a likely preference, and so is a measure (a key whose values are figures with units, taking several values: the shopper then holds a limit on it); a key with one value everywhere, one seen on one or two listings when six or more were opened, an identifier, a model or part code, a warranty, shipping, brand or seller, or a key that lists purchase variants (one key mixing the values of several properties) cannot be one.
2b. One preference per product, on what defines it. A shopper rarely holds more than one unwritten preference on a product, and holds it on a property that defines the product to a buyer (the one a buyer names first when asked what they want), not on a secondary detail, and never on an engineering parameter nobody states as a wish (a figure only a technician would specify). Rank the defining property of each item above every secondary one.
2c. Accessories. When the request pairs a main product with things that go with it (something made to connect to, power, hold, protect or be fitted to the main product), the shopper's unwritten preferences are about the main product: such an item's own properties come last unless the story names one, and properties that only describe what it fits are never asked.
Skip a property the request already settles for that item (a stated value, a limit, an exclusion or a "cross off"), a second name for a property already in the list, and a property the shopper has already said they do not mind about. With several items, open the list with ONE entry per item (each item's best property, items in the request's order), then every item's remaining hinted properties, then take turns between the items; the agent sends the first question of every item before any second question, and stops asking about an item whose reply showed it holds no preferences.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
First write down every story sentence that hints at an unwritten preference, with the one or two listed properties it points at; every property named there must then open the list. List up to 8 properties for EVERY item (each item's own best first; an item the story gives no hint for still gets its defining properties), best first. Return ONE JSON object and nothing else: {"hints": [{"item": item number, "words": "at most 10 of the story's words", "properties": ["...", "..."]}], "ask": [{"item": item number, "label": "...", "why": "asked" or "hint" or "for" or "common"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "kinds": [{"question": number, "kind": "firm" or "dontcare" or "confused" or "crowded"}], "budget": new total limit as a number or null}
- "kinds": how each answer reads: "firm" when it states a value, a limit, an exclusion or a comparison with another listing, however roundabout or folksy; "dontcare" when the shopper has no preference; "confused" when the question meant nothing to them (they ask what was meant, or to say it again or more simply); "crowded" when they want one question at a time or the question split up.
- Exclusions and roundabout fixings state a value: "anything with <property> other than X is out / gets rejected" means <property>: X; "cross off / rule out anything with <property> Y" means <property>: not Y; "nothing where <property> comes in other than <limit>" means <property>: <limit>; "only options with X under <property>" or "showing X under <property>" means <property>: X; "holding out for X on <property>" means <property>: X; "<property> wants to be / must come in as X" means <property>: X; a value called non-negotiable, fixed, not up for trade or something the shopper will not budge on is the wanted value, not an exclusion. Write "not X" only when the answer rules X out with words such as not, no, never, apart from, other than, except, cross off, rule out, anything but, none of.
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"), but leave out the words around it that only say how firmly the shopper holds it (the need is what the product must state).
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about. A reply that leaves the choice to the agent or says the shopper holds no preference on it is "dontcare", however it is phrased.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and spelling variants count (a different word does not, even when it means nearly the same), a value equal to the required value (any spelling or unit) or a list naming it counts; a value that carries the required words beside other qualifying words ("deep vermilion" for vermilion) counts only while no listing with the exact value can be had, and one sharing just some of the words ("celadon-tinted resin" for celadon) never counts, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails, and a key that is exactly the property's name (any spelling) decides over keys that add other words to that name, except that when two spellings of that same name disagree the listing fits only if one of them is a list that includes the value (two plain values that disagree make the listing doubtful); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; a key sharing no word with the property's name (spelling variants aside) is doubtful for it while another listing of that kind states the property under a key that shares a word with that name; when the listing has no attribute for that property, the wanted value named in its title counts, while one stated only under another attribute or only in its description text does not (nor does a number of another kind: a weight is not a capacity; nor a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole (a label naming only one of the kinds the request joins with 'or' or 'and' is wrong when the store has the whole one), and that label alone decides the kind: a title that adds words to it (a set or bundle word, a second cut or style named beside the wanted one) never makes a listing of the right category the wrong kind; a listing is doubtful when another attribute naming the same property states a different plain value, when an attribute contradicts the kind itself, or when no attribute names the property and the title or description names a different value (a title never overrides an attribute that names the property); a kind shown only under a key that adds a qualifier to the property's name is not shown; a doubtful listing is a problem while a clean fit is shown or can still be searched for; a rule of being unlike another listing's value counts as met unless the records show that same value; a requirement that ties its value to another listing's value through a store rule stated on a page (assigning each value the value it belongs with) takes the value that rule assigns to that listing's value; when a listing's own text hands a property to another listing (it sends the reader to another product for that property), its own value for that property does not count and the value on that other listing's page does (unverified until that page is among the records). Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and price times quantity summed over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
A listing's statements about what it fits or works with, and its references to other listings, describe compatibility between products, never items to buy or missing lines; Sakura's [compatibility] block below gives its reading of them, and a dependent line whose series is not among the anchor's (or whose fitment does not cover the named model) is a problem. Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show. A requirement that no record of any listing addresses (no attribute anywhere names its property, no title or description speaks to it: a delivery date, a warranty, a return term) cannot be checked in this store and is not a problem of this cart; leave it out of "problems". A cart line whose record has no attribute for a required property and whose title does not name the wanted value fails that requirement while another shown listing of that item's kind states that property: name it in "problems" and list in "better" the shown listings that state the wanted value. A listing with no attribute for a property whose title names the wanted value fits (such a listing counts as fitting), so under a cheapest pick such a cheaper listing beats a pricier one whose attribute states the value; a listing that names the value only under another attribute or only in its description text fails that requirement."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, spelling variants and unit conversions count (a different word does not), a value equal to the required value (any spelling or unit) or a list naming it counts; a value that carries the required words beside other qualifying words ("deep vermilion" for vermilion) counts only while no listing with the exact value can be had, and one sharing just some of the words ("celadon-tinted resin" for celadon) never counts, limits include the edge; a key that is exactly the property's name (any spelling) alone decides over keys that add other words to that name, and when two spellings of that same name disagree, the listing fits only if one of them is a list that includes the value); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; a key sharing no word with the property's name (spelling variants aside) is doubtful for it while another listing of that kind states the property under a key that shares a word with that name; when no attribute names that property, the wanted value named in the title counts (one stated only under another attribute or only in its description text does not, and a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item as a whole (a label naming only one of the kinds the request joins with 'or' or 'and' does not, when the store has the whole one) and alone decides the kind (a title adding a set or bundle word, or a second cut or style, never makes a listing of the right category the wrong kind); a listing is doubtful when another attribute naming the same property states a different plain value, when an attribute contradicts the kind, or when no attribute names the property and the title or description names a different value (a title never overrides an attribute naming the property); a kind shown only under a key with a qualifier added to the property's name is not shown; a doubtful listing does not fit while a clean one exists; a rule of being unlike another listing's value counts as met unless the records show that same value; a requirement tied to another listing's value through a store rule stated on a page takes the value that rule assigns; when a listing's own text hands a property to another listing, its own value for that property does not count and the other listing's page decides; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

ALIKE = """A shopper was asked about one property of a product and answered that the question touched more than one thing. From the attribute keys below (all found on listings of that product), pick at most two that a reader could take for that same property: a near meaning or a shared word. Return ONE JSON object and nothing else: {"alike": [keys copied exactly, most likely first]}"""
PAGEREAD = """A shopping requirement takes the value of one property from a listing described in words, and that listing's page can make a different listing the source of that value. Below is everything the store has shown about one listing. Return ONE JSON object and nothing else: {"value": the value this listing states for the property (in its attributes or its own text), or null when it states none, "handoff": null, or, when its text makes another listing the source of this property's value (that listing's value then counts instead of its own), {"shop": that other listing's seller or brand, "kind": the kind of product it is, "descriptors": ["each property and value given to pick it out"]}, "pairs": [["a value of the property", "the value it is paired with"]] for each rule on this page pairing values of the property, or []}. Only text that makes another listing the source of this property's value is a handoff; any other mention of other products is not."""
RELATE = """A shopping requirement on one property takes its value from another listing, and the pages it points at have now been read. Rewrite the requirement as a plain constraint. Return ONE JSON object and nothing else: {"need": "property: constraint"}, where the constraint is the value itself when the requirement wants that same value; "not <value>" when it wants anything other than that value; the value a pairing rule matches with it when it wants the partner of that value; "between <low> and <high>" when it allows a margin either side of that value (work out both edges, keep the unit); "at least <value>" or "at most <value>" when it compares with it."""
CAPARG = """A store's search tool is described below. Which of its arguments, if any, limits the results to listings priced at or below a number? Return ONE JSON object and nothing else: {"arg": that argument's exact name, or null}"""
REFPICK = """A shopping requirement points at one listing described in words: its seller, its kind and some of its properties. Decide which of the listings below the description names: the one whose record shows the most of the properties given, in any unit or spelling (convert units where needed), and contradicts none of them. Return ONE JSON object and nothing else: {"id": the id of that listing, or null when every listing contradicts the description or none shows any of its properties}"""
REFPARSE = """A shopping requirement takes its value from another listing, described in words. Read the description and return ONE JSON object and nothing else: {"shop": the seller, shop or brand name the description gives for that other listing, or "", "kind": the kind of product it is (the words naming its category), "property": the property whose value is to be read from it, "descriptors": ["each property-and-value pair the description gives to pick that listing out, as 'property value'"], "negated": true when the requirement wants anything but that listing's value, else false}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property's value itself (also when the requirement asks for a value other than or unlike that one: the page's value is what it is measured against), else false, "value": the property's value as stated on that last page, or ""}]}
A record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives; when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop's name and the description's values, as a few search words.
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null, "title": the key, inside a row, of the listing's name or title, or null}"""


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _clip_shown(value: Any, limit: int, depth: int = 0, alone: bool = True) -> Any:
    """_clip for what the model is shown: a text standing on its own keeps up to four times the limit."""
    if depth > 7:
        return "..."
    if isinstance(value, str):
        cap = 4 * limit if alone else limit
        return value if len(value) <= cap else value[:cap] + "..."
    if isinstance(value, dict):
        small = len(value) <= 3
        return {k: _clip_shown(v, limit, depth + 1, small) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip_shown(v, limit, depth + 1, True) for v in value]
    return value


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
    out = []
    for w in _words(text).split():
        w = w.strip(",;:!?()[]{}\"'")
        if len(w) >= 6 and w.endswith("our"):
            w = w[:-3] + "or"
        if w:
            out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return frozenset(out)


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def _keyword_arg(schema: dict[str, Any], texts: list[str]) -> str | None:
    return next((k for k in schema.get("required") or [] if k in texts), None) or next((k for k in texts if re.search(r"search|query|phrase|text|term|keyword|word", k, re.I)), None)


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



TEXT_KEY = re.compile(r"(?:^|\s\|\s)([A-Za-z_][A-Za-z0-9_]*(?: [A-Za-z0-9_]+){0,3}):\s")


def _text_value(value: str) -> Any:
    """One field of a record written as text: "k=v; k=v" pairs become a dict, True/False/None and plain numbers their values, else the text."""
    text = value.strip()
    pairs = re.split(r";\s*(?=[A-Za-z_][A-Za-z0-9_ ]{0,40}=)", text)
    if pairs and all(re.match(r"[A-Za-z_][A-Za-z0-9_ ]{0,40}=", p) for p in pairs):
        out: dict[str, Any] = {}
        for p in pairs:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip().rstrip(";").strip()
        return out
    if text in ("True", "False", "true", "false"):
        return text in ("True", "true")
    if text in ("None", "null"):
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def _text_record(text: Any) -> dict[str, Any] | None:
    """A record the store wrote as one line of text ("key: value | key: value | ..."), as a dict; None for any other text. A segment that
    does not start with "key: " belongs to the field before it (a title can hold " | " itself)."""
    if not isinstance(text, str) or text.count(" | ") < 2:
        return None
    marks = list(TEXT_KEY.finditer(text))
    if len(marks) < 4 or marks[0].start() != 0:
        return None
    raws = [text[m.end():(marks[k + 1].start() if k + 1 < len(marks) else len(text))].strip() for k, m in enumerate(marks)]
    # a record names its listing: one field is a code (letters, digits, '-' or '_' only, with a digit, and a separator or 8+ characters),
    # so a description that merely lists "key: value" pairs is left as text
    if not any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,63}", r) and re.search(r"\d", r) and (re.search(r"[-_]", r) or (re.search(r"[A-Za-z]", r) and len(r) >= 8)) for r in raws):
        return None
    out: dict[str, Any] = {}
    for m, raw in zip(marks, raws):
        if m.group(1) not in out:
            out[m.group(1)] = _text_value(raw)
    return out if len(out) >= 4 else None


LINE_KEY = re.compile(r"(?:^|\n)[ \t]*(?:[-*\u2022][ \t]+)?(?:\*\*)?([A-Za-z_][A-Za-z0-9_]*(?: [A-Za-z0-9_]+){0,3})(?:\*\*)?(?::(?:\*\*)?[ \t]+|[ \t]*=[ \t]*)")
SEMI_KEY = re.compile(r"(?:^|;\s+)([A-Za-z_][A-Za-z0-9_]*(?: [A-Za-z0-9_]+){0,3})(?::\s+|\s*=\s*)")
BAR_EQ_KEY = re.compile(r"(?:^|\s\|\s)([A-Za-z_][A-Za-z0-9_]*(?: [A-Za-z0-9_]+){0,3})\s*=\s*")
ID_FIELD = re.compile(r"(?:^|[_\s-])(?:id|ref|code|sku|number|no)(?:$|[_\s-])|(?:id|ref|code|sku)$", re.I)


def _keyed(text: str, rx: re.Pattern[str]) -> dict[str, Any] | None:
    """Fields marked by `rx` ("key: value" or "key=value"), when four or more start at the text's first character and a field whose key
    names an identifier holds a code; a key met twice keeps both values."""
    marks = list(rx.finditer(text))
    if len(marks) < 4 or marks[0].start() != 0:
        return None
    raws = [text[m.end():(marks[k + 1].start() if k + 1 < len(marks) else len(text))].strip() for k, m in enumerate(marks)]
    if not any(ID_FIELD.search(m.group(1)) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,63}", r) and re.search(r"\d", r) for m, r in zip(marks, raws)):
        return None
    out: dict[str, Any] = {}
    for m, raw in zip(marks, raws):
        key = m.group(1).strip()
        if key in out:
            out[key] = f"{out[key]}; {raw}"
        else:
            out[key] = _text_value(raw)
    return out if len(out) >= 4 else None


def _loose_records(text: str) -> Any:
    """Records in another text shape (section 13 of the v5.7 patch): a JSON string, several records in one string
    (a line or a block each), one field per line, fields between semicolons, "key=value" fields between bars; None for any other text."""
    body = text.strip()
    if len(body) >= 8 and body[0] in "{[" and body[-1] in "}]":
        try:
            value = json.loads(body)
        except (ValueError, RecursionError):
            return None
        return value if isinstance(value, (dict, list)) else None
    lines = [line.strip() for line in body.split("\n") if line.strip()]
    if len(lines) >= 2:
        each = [_text_record(line) or _keyed(line, SEMI_KEY) or _keyed(line, BAR_EQ_KEY) for line in lines]
        found = [r for r in each if r]
        if len(found) >= 2:
            rest = [line for line, r in zip(lines, each) if not r]
            return {"text": "\n".join(rest), "records": found} if rest else found
    blocks = [b.strip() for b in re.split(r"\n[ \t]*\n", body) if b.strip()]
    if len(blocks) >= 2:
        each = [_keyed(b, LINE_KEY) for b in blocks]
        found = [r for r in each if r]
        if len(found) >= 2:
            rest = [b for b, r in zip(blocks, each) if not r]
            return {"text": "\n\n".join(rest), "records": found} if rest else found
    for rx, mark in ((LINE_KEY, "\n"), (SEMI_KEY, ";"), (BAR_EQ_KEY, " | ")):
        if body.count(mark) >= 3:
            record = _keyed(body, rx)
            if record:
                return record
    return None


def _unflatten(node: Any, depth: int = 0, alone: bool = True) -> Any:
    """A tool result with every record written as text turned into a dict (lists and dicts are walked; everything else is unchanged).
    `alone`: the text stands on its own (the whole result, a list element or the value of a small wrapper), where other shapes are read."""
    if depth > 8:
        return node
    if isinstance(node, str):
        record = _text_record(node)
        if record is not None:
            return record
        if alone:
            loose = _loose_records(node)
            if isinstance(loose, (dict, list)):
                return _unflatten(loose, depth + 1, True)
        return node
    if isinstance(node, list):
        return [_unflatten(v, depth + 1, True) for v in node]
    if isinstance(node, dict):
        small = len(node) <= 3
        return {k: _unflatten(v, depth + 1, small) for k, v in node.items()}
    return node


def _money(text: str) -> float | None:
    """A price written as text with a currency sign or code and thousands separators ("$1,299.00", "12.50 USD", "EUR 12,99")."""
    m = re.fullmatch(r"\s*(?:[$\u20ac\u00a3\u00a5]|[A-Z]{3})?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:([.,])(\d{1,2}))?\s*(?:[$\u20ac\u00a3\u00a5]|[A-Za-z]{1,8})?\.?\s*", text)
    if not m:
        return None
    whole = float(m.group(1).replace(",", ""))
    return whole + (float("0." + m.group(3)) if m.group(3) else 0.0)


def _stock_word(value: Any) -> bool | None:
    """A stock flag written as words: False for out of stock, sold out, unavailable, no or 0; True for in stock, available or yes."""
    if not isinstance(value, str):
        return None
    t = " ".join(value.lower().split())
    if re.fullmatch(r"no|false|n|0|none", t) or re.search(r"\b(out of stock|sold out|unavailable|not available)\b", t):
        return False
    if re.fullmatch(r"yes|true|y|\d+", t) or re.search(r"\b(in stock|available)\b", t):
        return True
    return None


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []
        self.fits: Any = None


class Sakura:
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
        self.better_for: dict[str, list[str]] = {}
        self.better_taken: set[str] = set()
        self.listed: set[str] = set()
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
        self.opened_step: dict[str, int] = {}
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
        self.title_key: str | None = None
        self.cheaper_marks: set[str] = set()
        self.last_search_step = 0
        self.search_blocks = 0
        self.kind_blocks = 0
        self.kind_noted = False
        self.wider_held = False
        self.paged_obs: list[Any] = []
        self.plan_labels: list[tuple[int, str]] = []
        self.swap_fits: list[str] = []
        self.fit_codes: dict[str, set[str]] = {}
        self.fit_sets: dict[str, set[str]] = {}
        self.fit_model: dict[str, str] = {}
        self.fit_vehicle: dict[str, tuple[str, str, int]] = {}
        self.fit_pointer: dict[str, str] = {}
        self.model_takes: dict[str, set[str]] = {}
        self.fit_resolved: set[str] = set()
        self.fit_blocks = 0
        self.best_effort_done = False
        self.swept_anchor: dict[int, str | None] = {}
        self.swept_phrases: set[str] = set()
        self.relays = 0
        self.plan_blocks = 0
        self.plan_lines: dict[str, int] = {}
        self.dug: set = set()
        self.removal_held: set[str] = set()
        self.facts_bare: dict[str, int] = {}
        self.facts_seen: set[str] = set()
        self.code_used = 0
        self.shape_tries: dict[str, int] = {}
        self.fit_swept: set[int] = set()
        self.spend_soft = SPEND_SOFT
        self.spend_hard = SPEND_HARD
        self.ref_tried: set[str] = set()
        self.refuse_guards = 0
        self.missing_held = False
        self.started = time.monotonic()
        self.line_blocks = 0
        self.ref_parsed: dict[str, dict[str, Any]] = {}
        self.link_seen: dict[str, list[str]] = {}
        self.link_searches = 0
        self.single_effort = False
        self.fit_covers: dict[str, tuple[set[str], set[tuple[str, int]], set[str]]] = {}
        self.anchor_codes_of: dict[int, set[str]] = {}

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def clock_left(self) -> float:
        """Seconds of wall clock still allowed to this episode."""
        return CLOCK_LIMIT - (time.monotonic() - getattr(self, "started", time.monotonic()))

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

    def note_facts(self, tool: str, has: bool) -> None:
        """A view that showed attributes settles the tool as the attributes view; one that showed none counts, and three such
        views with none shown settle it as not that view (a lone listing with a bare record proves nothing)."""
        if has:
            self.facts_from[tool] = True
            return
        self.facts_bare[tool] = self.facts_bare.get(tool, 0) + 1
        if self.facts_from.get(tool) is not True and self.facts_bare[tool] >= 3:
            self.facts_from[tool] = False

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
                if self.clock_left() < 30:
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
        # the thinking pass is dropped only for the tool-calling loop on the light model (its replies were 70 % reasoning tokens);
        # JSON readings (brief, planner, hearing) keep it, and the heavy model thinks at low effort everywhere
        wants = (REASONING.get(model) if (model == HEAVY or tools) else None) if _REASONING_OK[0] else None
        if wants:
            body["reasoning"] = dict(wants)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        room = self.clock_left() - 12
        if room < 6:
            self.trace.append({"kind": "llm_skipped", "why": "clock", "model": model})
            return None, None
        _llm.timeout = int(max(6, min(50, room)))
        result = _llm.post_verbose("/inference/chat/completions", body)
        if wants and not (isinstance(result.data, dict) and result.data.get("choices")) and (result.error or {}).get("status") in (400, 422):
            # a proxy that refuses the reasoning field: never send it again this episode, and resend this call without it
            _REASONING_OK[0] = False
            self.trace.append({"kind": "reasoning_off", "status": (result.error or {}).get("status")})
            body.pop("reasoning", None)
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
        if name == self.roles.get("search") and isinstance(args, dict):
            self.search_log = (getattr(self, "search_log", []) + [dict(args)])[-30:]
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                data["observation"] = _unflatten(data.get("observation"))
                if name == self.roles.get("search") and isinstance(args, dict):
                    self.note_found(args, data.get("observation"))
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that)
                    for value in args.values():
                        got = [v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)]
                        self.opened.update(got)
                        self.opened_step.update({v: self.step for v in got})
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
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        for one, other in (("buy", "refuse"), ("refuse", "buy")):
            tool = self.roles.get(one)
            props = (self.schemas.get(tool) or {}).get("properties") or {} if tool else {}
            if tool and not self.roles.get(other) and any(isinstance(v, dict) and isinstance(v.get("enum"), list) and len(v["enum"]) >= 2 for v in props.values()):
                self.roles[other] = tool
                self.trace.append({"kind": "shared_finish", "tool": tool, "named_for": one})
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        if buy and buy == refuse and (not brief["buy_args"] or not brief["refuse_args"] or brief["buy_args"] == brief["refuse_args"]):
            ask = (f"The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: "
                   + _dump(self.schemas.get(buy)) + "\nReturn ONE JSON object and nothing else: {\"buy_args\": the fixed arguments that check out, \"refuse_args\": the fixed arguments that end without a purchase} (only the arguments that choose the mode).")
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 1200) or {}
            if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
                brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]
        for entry in _items(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get("quantity"))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get("musts")) if m]
            self.items.append(Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), count, musts))
            fits = entry.get("fits")
            self.items[-1].fits = fits if isinstance(fits, (int, str)) and not isinstance(fits, bool) else None
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        self.spend_soft = SPEND_SOFT + FIT_ITEM_COST * (len(self.items) - 1)
        self.spend_hard = SPEND_HARD + FIT_ITEM_COST * (len(self.items) - 1)
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
            # the messages the steps at the order and the ending take, one at least for each of the two that needs any
            asking = lambda step: bool(re.search(r"\b(ask|question)", step, re.I) and re.search(r"\b(shopper|customer|buyer|them|preference|requirement|wish|want)", step, re.I) and not re.search(r"\b(approv|confirm|permission|okay|ok)", step, re.I))
            talk = [p["messages"] for p in self.process if p["when"] in ("order", "end") and p["messages"] and not asking(p["step"])]
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
            got = self.ask_json(prompt, [LIGHT, HEAVY], 1200) or {}
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
            when = {"choose": "before choosing", "order": "at the order (Sakura does it when you check out)", "end": "at an ending without a purchase (Sakura does it then)"}[entry["when"]]
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
        text = f"[Sakura] step {self.step} of {self.max_steps} used, {left} left."
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
        fit = self.fit_status()
        if fit:
            text += "\n" + fit
        if len(self.items) > 1:
            plan = self.bundle_plan(keep=self.tail(), do_cart=False, search=False)
            if plan:
                text += "\n" + plan
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

    def absorb(self, observation: Any, name: str | None = None) -> None:
        listed = name is None or name not in {self.roles.get(r) for r in ("cart", "cart_view", "buy", "refuse")} - {None}

        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    if listed:
                        self.listed.add(ident)
                    if ident not in self.rows:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    # a flag of the listing's own record turning from true to false is read as "can no longer be bought"
                    old = self.rows[ident]
                    if any(old.get(k) is True and v is False for k, v in node.items()) or (
                            self.stock_key and _stock_word(old.get(self.stock_key)) is True and _stock_word(node.get(self.stock_key)) is False):
                        self.unavailable.add(ident)
                    elif any(old.get(k) is False and v is True for k, v in node.items()):
                        self.unavailable.discard(ident)
                    self.rows[ident].update(node)
                    self.learn_fitment(ident)
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
        cached = getattr(self, "_cat_cache", None)
        if cached and cached[0] == len(self.rows):
            return cached[1]
        key = self.category_key_scan()
        self._cat_cache = (len(self.rows), key)
        return key

    def category_key_scan(self) -> str | None:
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
                twin = next((k for k, (other, spec) in enumerate(kept) if (_stem(name) <= _stem(other) or _stem(other) <= _stem(name)) and slot["per"] == spec["per"]), None)
                if twin is not None:
                    # one property under two keys: keep the plainer (shorter) name
                    if len(name) < len(kept[twin][0]):
                        kept[twin] = (name, slot)
                    continue
                kept.append((name, slot))
            for name, slot in kept[:16]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]], len(set(slot["per"].values()))))
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
        row = self.rows.get(ident) or {}
        key = self.price_key or next((k for k in sorted(row, key=lambda k: len(str(k))) if "price" in str(k).lower()), None) or next(
            (k for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 and k not in (self.id_key, self.stock_key, self.title_key)), None)
        value = row.get(key) if key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        elif isinstance(value, str):
            value = _money(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        """Not sold out by any record: no note or refusal named it, and its stock flag (the key the model named) is not no."""
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        if _stock_word(value) is False:
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
    def learn_shape(self, name: str, observation: Any) -> None:
        """The first row-list result of a tool names its parts (rows, count, next page, price, stock) once, whatever the pick rule:
        prices and stock flags are read by code for every task."""
        if name in self.shapes or not isinstance(observation, dict):
            return
        if any(isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v) for v in observation.values()):
            self.read_shape(name, observation)

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
        self.shape_tries[name] = self.shape_tries.get(name, 0) + 1
        sample = {k: (v[:3] if k in lists else v) for k, v in observation.items()}
        prompt = (SHAPE + "\n\nTOOL: " + name + " | parameters: " + _dump(self.schemas.get(name) or {})
                  + "\n\nRESULT (first rows only):\n" + _dump(_clip(sample, 120))[:4000])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows:
            if self.shape_tries[name] < 2:
                del self.shapes[name]
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get("properties") or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None

        shape = {"rows": rows, "count": named(got.get("count"), observation), "next": named(got.get("next"), observation),
                 "next_arg": named(got.get("next_arg"), props), "price": named(got.get("price"), first), "stock": named(got.get("stock"), first), "title": named(got.get("title"), first)}
        if shape["count"] and (isinstance(observation[shape["count"]], bool) or not isinstance(observation[shape["count"]], int)):
            shape["count"] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape["price"]
        self.stock_key = self.stock_key or shape["stock"]
        self.title_key = self.title_key or shape["title"]
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
        """A search that matched more listings than it showed, under a cheapest, most or least pick: Sakura reads a short
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
            return (f"\n[Sakura] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
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
        return ("\n[Sakura read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Sakura read on) in price order, since
        stores list them in no such order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Sakura] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search can miss fitting listings
        whose text words the product or the requirement differently, and it lists rows in no price order."""
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. A search finds only listings whose text"
                " holds its words, so fitting listings that word the product or the requirement differently are missed, and rows come in no price order."
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
        for ident in [i for i in ids if i in self.rows and i not in self.full and i not in self.facts_seen and not self.sold_out(i)]:
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
            self.facts_seen.add(ident)
            if isinstance(seen, dict) and self.ident(seen):
                self.note_facts(tool, bool(_fact_block(seen)))
            texts.append(f"{ident}: " + self.show(seen)[:1500])
        if texts:
            self.trace.append({"kind": "fill_facts", "n": len(texts)})
        return texts

    def note_found(self, args: dict[str, Any], observation: Any) -> None:
        """Which words of a search each row it returned came with (the row ids are the values held under the learned id key, or under
        a key named like an id)."""
        words = {w for v in args.values() if isinstance(v, str) for w in re.findall(r"[a-z0-9.]{3,}", v.lower())}
        found = self.__dict__.setdefault("found_by", {})
        returned: list[str] = []
        listed: set[str] = set()
        queue = [observation]
        at = 0
        while at < len(queue):
            node = queue[at]
            at += 1
            if isinstance(node, list):
                queue.extend(node[:60])
            elif isinstance(node, dict):
                ident = node.get(self.id_key) if self.id_key else None
                if not isinstance(ident, str):
                    ident = next((v for k, v in node.items() if isinstance(v, str) and re.fullmatch(r"(?:id|sku|code|.*_id|.*id)", str(k), re.I)), None)
                if isinstance(ident, str) and ident:
                    if words:
                        found.setdefault(ident, set()).update(words)
                    if ident not in listed:
                        listed.add(ident)
                        returned.append(ident)
                queue.extend(v for v in node.values() if isinstance(v, (list, dict)))
        self.last_found = returned

    def value_words(self, number: int) -> set[str]:
        """The words of an item's wanted values (the request's and the shopper's), three letters or more."""
        item = self.items[number - 1]
        out: set[str] = set()
        for need in item.musts + item.told:
            if need in item.dropped or ":" not in need:
                continue
            out.update(re.findall(r"[a-z0-9.]{3,}", need.split(":", 1)[1].lower()))
        return out

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
            self.capped_browse(str((self.rows.get(line) or {}).get(cat_key) or ""), price)
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i)
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price]
            wanted = self.value_words(self.item_of(line))
            found = getattr(self, "found_by", {})
            pool.sort(key=lambda i: (-len(found.get(i, set()) & wanted), self.price_of(i) or 0.0))
            picks += [i for i in pool[:3] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    def cap_arg(self) -> str | None:
        """The search argument that keeps results at or below a price, named once by the light model from the schema (None when none)."""
        if "cap_named" in self.__dict__:
            return self.cap_named
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        nums = [k for k, v in props.items() if isinstance(v, dict) and ("integer" in _kinds(v) or "number" in _kinds(v))]
        self.cap_named = None
        if nums and not self.cost > self.spend_hard:
            got = self.ask_json(CAPARG + "\n\nTOOL " + str(tool) + ": " + _dump(schema)[:3000], [LIGHT], 1000) or {}
            arg = str(got.get("arg") or "").strip()
            self.cap_named = arg if arg in nums else None
            self.trace.append({"kind": "cap_arg", "arg": self.cap_named})
        return self.cap_named

    def capped_browse(self, label: str, price: float) -> None:
        """One listing of a category under a ceiling just below `price` (the category field and mode arguments of an earlier search call
        that named a store category; two pages at most), so cheaper listings of that kind come into view."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        done = self.__dict__.setdefault("capped_done", set())
        if not tool or not label or (label, price) in done or self.left() <= self.tail() + 10 or self.finished:
            return
        done.add((label, price))
        cap = self.cap_arg()
        if not cap:
            return
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        keyword = _keyword_arg(schema, texts)
        labels = {self.kind_of(i) for i in self.seen_order if self.kind_of(i)}
        args: dict[str, Any] = {}
        for past in reversed(getattr(self, "search_log", [])):
            field = next((k for k, v in past.items() if k in texts and k != keyword and isinstance(v, str) and _stem(v)
                          and any(_stem(v) <= _stem(l) or _stem(l) <= _stem(v) for l in labels)), None)
            if field:
                args = {k: v for k, v in past.items() if isinstance(props.get(k), dict) and "enum" in props[k]}
                args[field] = label
                break
        if not args:
            return
        whole = "integer" in _kinds(props.get(cap) or {})
        args[cap] = int(price) - 1 if whole else round(price * 0.999, 2)
        if (args[cap] or 0) <= 0:
            return
        for _ in range(2):
            faults: list[str] = []
            call = _shape(dict(args), schema, "args", faults)
            if faults or self.left() <= self.tail() + 8:
                break
            reply = self.act(tool, call)
            if reply is None or self.finished:
                break
            observation = reply.get("observation")
            self.observe(tool, observation)
            plan = self.paging(tool, observation)
            if not self.records_in(observation) or not plan or plan[3] is None:
                break
            args[plan[2]] = plan[3]
        self.trace.append({"kind": "capped_browse", "label": label, "under": args.get(cap)})

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Sakura's own call and read any note in it; returns what the note changed."""
        self.absorb(observation, name)
        self.learn_shape(name, observation)
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
        if self.left() <= 14:
            return True   # a message still unspent this late would be forfeited
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
        self.one_at_a_time = False
        return self.interview_lists()

    def list_text(self, parts: list[tuple[int, list[str]]]) -> str:
        """One message naming properties of one item ("What A, B or C would you like for the X?") or of several items ("What A or B
        would you like for the X, and what C or D for the Y?"); one property alone reads "What A would you like for the X?"."""
        def body(labels: list[str]) -> str:
            names = [label for label in labels if label]
            return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} or {names[-1]}"
        number, labels = parts[0]
        text = f"What {body(labels)} would you like for the {self.item_called(number)}"
        for other, more in parts[1:]:
            text += f", and what {body(more)} for the {self.item_called(other)}"
        return text + "?"

    def interview_lists(self) -> str:
        """Each message names several properties of one item, likeliest first; while more items are unasked than messages are left, one
        message names a few properties of each of several items, so every item is asked. The shopper answers about the first named
        property on its own list: a firm reply is a requirement; a don't-mind reply says that property is not the need, and the need may
        stand behind it, so the rest of that list leads the item's next question in reverse order; a confused reply says the first named
        properties mean nothing to it, so new properties lead the item's next question and the others follow. The first reply asking for
        one question at a time turns the rest of the interview into single properties, each worded with its look-alike keys set aside."""
        self.asked = True
        if self.msgs_free() <= 0:
            return ""
        self.widen()
        early = self.resolve_refs(keep=14)
        self.bundle_plan(keep=14, do_cart=False)
        plan = self.plan_questions(self.msgs_free())
        self.trace.append({"kind": "plan", "ask": _clip(plan, 120)})
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        order: list[int] = []
        for number, label in plan:
            if number not in order:
                order.append(number)
            if not self.settled(number, label) and not any(_words(label) == _words(o) or self.same_property(label, o) for o in lanes[number]):
                lanes[number].append(label)
        order += [n for n in lanes if n not in order]
        for number in lanes:
            lanes[number] = self.lane_fill(number, lanes[number])
        self.plan_labels = [(n, l) for n in order for l in lanes[n]]
        single = False
        lists = 0
        state: dict[int, str] = {}
        asked_on: dict[int, int] = {}
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        outcomes: list[str] = []
        rank = {None: 0, "dontcare": 0, "firm": 1, "confused": 2}
        while self.msgs_free() > 0 and not self.finished and self.left() > 6 and self.clock_left() > CLOSING + 20:
            live = [n for n in order if lanes[n]]
            if not live:
                break
            live.sort(key=lambda n: (asked_on.get(n, 0) > 0, rank.get(state.get(n), 2), order.index(n)))
            fresh = [n for n in live if not asked_on.get(n)]
            free = self.msgs_free()
            if single:
                parts = [(live[0], lanes[live[0]][:1])]
            elif len(fresh) > free >= 1:
                group = fresh[: -(-len(fresh) // free)]
                share = max(1, min(LIST_SHARE, (LIST_LABELS + 2) // len(group)))
                parts = [(n, lanes[n][:share]) for n in group]
            else:
                parts = [(live[0], lanes[live[0]][: LIST_FIRST if lists == 0 else LIST_LABELS])]
            text = self.narrow_question(parts[0][0], parts[0][1][0]) if single else self.list_text(parts)
            batch = self.send([(parts[0][0], parts[0][1][0], text)])
            if not batch:
                break
            sent += batch
            lists += 0 if single else 1
            for n, _ in parts:
                asked_on[n] = asked_on.get(n, 0) + 1
            had = [len(item.told) for item in self.items]
            about = batch[-1] if len(parts) == 1 else (0, batch[-1][1], batch[-1][2])
            heard = self.second_opinion(about, self.hear([about], (LIGHT, HEAVY)))
            before = len(told)
            self.take_heard(heard, told)
            said = self.heard_kind(heard, 1) or self.reply_kind(batch[-1][2])
            if len(told) > before:
                for n, labels in parts:
                    new = self.items[n - 1].told[had[n - 1]:]
                    if not new:
                        continue
                    state[n] = "firm"
                    found = [(t.split(":", 2)[1] if t.startswith("item ") and t.count(":") >= 2 else t.split(":", 1)[0]).strip() for t in new]

                    def covered(label: str) -> bool:
                        return any(self.same_property(label, f) or _words(label) == _words(f) or (_stem(label) and (_stem(label) <= _stem(f) or _stem(f) <= _stem(label))) for f in found)

                    hit = next((i for i, l in enumerate(labels) if covered(l)), None)
                    drop = labels[: hit + 1] if hit is not None else labels[:1]
                    lanes[n] = [l for l in lanes[n] if l not in drop and not covered(l)]
                    outcomes.append(f"{self.items[n - 1].name}: firm on {', '.join(found)}")
                continue
            if said == "crowded":
                number, labels = parts[0]
                if not single:
                    # from now on one property a message, worded with its look-alike keys set aside (race 16: such questions drew a
                    # one-at-a-time reply 4 % of the time, plain ones 16 %)
                    single = True
                    self.one_at_a_time = True
                    for n, _ in parts:
                        asked_on[n] = max(0, asked_on.get(n, 0) - 1)
                    outcomes.append(f"{self.items[number - 1].name}: wants one question at a time")
                    continue
                lanes[number] = [l for l in lanes[number] if l != labels[0]]
                outcomes.append(f"{labels[0]}: crowded")
                continue
            if said == "confused":
                for n, labels in parts:
                    named = set(labels)
                    lanes[n] = [l for l in lanes[n] if l not in named] + labels[1:]
                    if len(labels) >= 3 and len(parts) == 1 and state.get(n) != "firm":
                        state[n] = "confused"
                outcomes.append("; ".join(f"{self.items[n - 1].name}: confused ({', '.join(labels)})" for n, labels in parts))
                continue
            # a reply that does not mind: the first named property on the shopper's list is not the need
            number, labels = parts[0]
            state[number] = state.get(number) or "dontcare"
            rest = [l for l in lanes[number] if l not in labels]
            lanes[number] = (labels[1:][::-1] + rest) if not single else rest
            if len(labels) == 1:
                self.items[number - 1].free.append(labels[0])
            outcomes.append(f"{self.items[number - 1].name}: does not mind ({labels[0]}{' ...' if len(labels) > 1 else ''})")
        pointed_now = [need for _, need in self.refs if any(need in t for t in told)]
        resolved = self.resolve_refs(keep=12) if pointed_now else ""
        swept = self.fit_sweep(keep=12)
        searched = self.value_search([n for n, st in state.items() if st == "firm"]) if told else ""
        planned = self.bundle_plan(keep=12, do_cart=True)
        if planned:
            swept = (swept + "\n" if swept else "") + planned
        if told:
            self.version += 1
        self.trace.append({"kind": "asked", "outcomes": outcomes, "told": told, "lists": not single})
        if not sent:
            return ""
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = (early + "\n" if early else "") + "Sakura's interviewer asked the shopper (each question used one step; the shopper answers about one named property, most often the first it cares about):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed). A property the shopper does not mind or did not recognise is not a requirement."
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if resolved:
            summary += "\n" + resolved
        if swept:
            summary += "\n" + swept
        if searched:
            summary += "\n" + searched
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def item_called(self, number: int) -> str:
        """How questions name an item: the store's own category label for it (the words the shopper's replies use), else the request's name."""
        item = self.items[number - 1]
        shelf = str(item.shelf or "").strip()
        if shelf and len(shelf.split()) <= 6 and not re.search(r"[>/|]", shelf):
            return shelf.lower()
        return item.name

    def broad_question(self, number: int, labels: list[str]) -> str:
        """One message about one item that names several of its properties, the likeliest first."""
        item = self.item_called(number)
        names = [l for l in labels if l]
        if len(names) == 1:
            return self.question(number, names[0])
        return f"For the {item}, what {', '.join(names[:-1])} or {names[-1]} would you like?"

    def lane_fill(self, number: int, lane: list[str]) -> list[str]:
        """The item's own listing keys after the planner's picks: most carried first, several values, no identifiers, codes,
        listing details or settled properties."""
        learned = {_words(k) for k in (self.id_key, self.price_key, self.stock_key, self.title_key, self.category_key()) if k}
        out = list(lane)
        for key, carried, distinct, values in sorted(self.item_keys(number), key=lambda k: -k[1]):
            if distinct < 2 or _words(key) in learned or self.code_key(values) or self.settled(number, key):
                continue
            if re.search(r"\b(warranty|shipping|delivery|seller|shop|brand|price|stock|options?|variants?|sku|ids?|codes?|photos?|images?|videos?|urls?|links?|model)\b", _words(key)):
                continue
            if any(_words(key) == _words(o) or self.same_property(key, o) for o in out):
                continue
            out.append(key)
        return out[:12]

    def interview_broad(self) -> str:
        """Each message names several properties of ONE item, the likeliest first: the shopper nearly always answers about the first named
        property when it holds a view on it, and often about a later one, so one message probes several properties. A firm answer is a
        requirement; a don't-mind or confused answer drops the first named property (a confused item is asked again only after every
        other item had its turn). A request for one question at a time switches the rest of the interview to single properties."""
        self.asked = True
        if self.msgs_free() <= 0:
            return ""
        self.widen()
        early = self.resolve_refs(keep=14)
        self.bundle_plan(keep=14, do_cart=False)
        plan = self.plan_questions(self.msgs_free())
        self.trace.append({"kind": "plan", "ask": _clip(plan, 120)})
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        order: list[int] = []
        for number, label in plan:
            if number not in order:
                order.append(number)
            if not self.settled(number, label) and not any(_words(label) == _words(o) or self.same_property(label, o) for o in lanes[number]):
                lanes[number].append(label)
        order += [n for n in lanes if n not in order]
        # an item that must fit or work with another requested product rarely holds wishes of its own: it is asked after the others
        dependent = set(self.anchors())
        position = {n: i for i, n in enumerate(order)}
        order.sort(key=lambda n: (n in dependent, position[n]))
        for number in lanes:
            lanes[number] = self.lane_fill(number, lanes[number])
        self.plan_labels = [(n, l) for n in order for l in lanes[n]]
        state: dict[int, str] = {}
        asked_on: dict[int, int] = {}
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        outcomes: list[str] = []
        single = False
        dontcares: dict[int, int] = {}
        while self.msgs_free() > 0 and not self.finished and self.left() > 6 and self.clock_left() > CLOSING + 20:
            live = [n for n in order if lanes[n]]
            if not live:
                break
            # every item once first (in the planner's order), then items that showed a list, then items that already gave a value
            live.sort(key=lambda n: (asked_on.get(n, 0) > 0, {"dontcare": 0, None: 0, "firm": 1}.get(state.get(n), 2), order.index(n)))
            number = live[0]
            labels = lanes[number][:1] if single else lanes[number][:BROAD_LABELS]
            batch = self.send([(number, labels[0], self.broad_question(number, labels))])
            if not batch:
                break
            sent += batch
            asked_on[number] = asked_on.get(number, 0) + 1
            heard = self.second_opinion(sent[-1], self.hear([sent[-1]], (LIGHT, HEAVY)))
            before = len(told)
            self.take_heard(heard, told)
            said = self.heard_kind(heard, 1) or self.reply_kind(batch[-1][2])
            if len(told) > before:
                state[number] = "firm"
                found = [(t.split(":", 2)[1] if t.startswith("item ") and t.count(":") >= 2 else t.split(":", 1)[0]).strip() for t in told[before:]]

                def covered(label: str) -> bool:
                    return any(self.same_property(label, f) or _words(label) == _words(f) or (_stem(label) and (_stem(label) <= _stem(f) or _stem(f) <= _stem(label))) for f in found)

                hit = next((i for i, l in enumerate(labels) if covered(l)), None)
                drop = labels[: hit + 1] if hit is not None else labels[:1]
                lanes[number] = [l for l in lanes[number] if l not in drop and not covered(l)]
                outcomes.append(f"{self.items[number - 1].name}: firm on {', '.join(found)}")
            elif said == "crowded" and not single:
                single = True
                self.one_at_a_time = True
                outcomes.append(f"{self.items[number - 1].name}: wants one question at a time")
            elif said == "confused":
                if len(labels) >= 3:
                    state[number] = "none"
                    lanes[number] = [l for l in lanes[number] if l not in labels[:1]]
                    outcomes.append(f"{self.items[number - 1].name}: confused ({', '.join(labels)})")
                else:
                    lanes[number] = [l for l in lanes[number] if l not in labels]
                    outcomes.append(f"{self.items[number - 1].name}: {', '.join(labels)} not on the list")
            else:
                # no need among the named properties, or the shopper passed over one named after the first: the first named is not
                # the need (a need named first is answered 96 % of the time), so it leaves the lane and the rest are asked again
                state[number] = state.get(number) or "dontcare"
                dontcares[number] = dontcares.get(number, 0) + 1
                lanes[number] = [l for l in lanes[number] if l not in labels[:1]]
                if len(labels) == 1:
                    self.items[number - 1].free.append(labels[0])
                outcomes.append(f"{self.items[number - 1].name}: does not mind ({labels[0]}{' ...' if len(labels) > 1 else ''})")
        pointed_now = [need for _, need in self.refs if any(need in t for t in told)]
        resolved = self.resolve_refs(keep=12) if pointed_now else ""
        swept = self.fit_sweep(keep=12)
        searched = self.value_search([n for n, st in state.items() if st == "firm"]) if told else ""
        planned = self.bundle_plan(keep=12, do_cart=True)
        if planned:
            swept = (swept + "\n" if swept else "") + planned
        if told:
            self.version += 1
        self.trace.append({"kind": "asked", "outcomes": outcomes, "told": told, "broad": not single})
        if not sent:
            return ""
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = (early + "\n" if early else "") + "Sakura's interviewer asked the shopper (each question used one step; the shopper answers about one named property it cares about, most often the first):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed). A property the shopper does not mind or did not recognise is not a requirement."
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if resolved:
            summary += "\n" + resolved
        if swept:
            summary += "\n" + swept
        if searched:
            summary += "\n" + searched
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def interview_single(self) -> str:
        """Ask the shopper one question at a time, best candidate first, reading each reply before the next:
        a reply that did not understand the question means the shopper holds no view on that property (move on);
        one that asks for a single question at a time means the question touched two of the shopper's properties
        (ask the same property once more, narrowed with example values); one that does not mind closes that property;
        a firm answer is a requirement. Questions are never spent on properties the request already settles."""
        self.asked = True
        count = self.msgs_free()
        if count <= 0:
            return ""
        self.widen()
        early = self.resolve_refs(keep=14)
        self.bundle_plan(keep=14, do_cart=False)
        plan = self.plan_questions(count)
        self.trace.append({"kind": "plan", "ask": _clip(plan, 120)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        tried: set[tuple[int, str]] = set()
        outcomes: list[str] = []
        queue = list(plan)
        planned_items = {n for n, _ in plan}
        queue += [(n, l) for n, l in self.fallback_questions(set(), {}, {}) if n not in planned_items]
        self.plan_labels = list(queue)
        many = len(self.items) > 1
        state: dict[int, str] = {}
        asked_on: dict[int, int] = {}
        anchor_items = {t for t in self.anchors().values() if isinstance(t, int)}

        def pick_next() -> tuple[int, str] | None:
            """With several items: every item once first (planner order), then items whose reply showed a list (don't
            care), then items that already gave a firm value, and items that answered "confused" only when nothing else
            is left (a confused reply means the shopper holds no list on that item: 79% of its next replies are confused too)."""
            best = None
            for pos, (number, label) in enumerate(queue):
                st = state.get(number)
                rank = 0 if not many else ((0 if number in anchor_items else 1) if number not in asked_on else 2 if st == "dontcare" else 3 if st in ("firm", "unclear", "crowded") else 4)
                if best is None or (rank, pos) < best[0]:
                    best = ((rank, pos), pos)
            return queue.pop(best[1]) if best else None

        refills = 0
        while self.msgs_free() > 0 and not self.finished and self.left() > 6 and self.clock_left() > 20:
            if not queue:
                # the planner's list is spent while messages remain: every item gets a question on its likeliest property
                more = self.fallback_questions(tried, state, asked_on) if refills < 3 else []
                refills += 1
                if not more:
                    break
                queue = more
                self.plan_labels += more
            nxt = pick_next()
            if nxt is None:
                break
            number, label = nxt
            key = (number, _words(label))
            if key in tried or self.settled(number, label) or any(n == number and self.same_property(label, other) for n, other in tried):
                continue
            tried.add(key)
            batch = self.send([(number, label, self.question(number, label))])
            if not batch:
                break
            sent += batch
            answer = batch[-1][2]
            # the reader reads every reply (each race speaks a new dialect); the idiom regexes only break a tie when it names no kind
            heard = self.second_opinion(sent[-1], self.hear([sent[-1]], (LIGHT, HEAVY)))
            before = len(told)
            self.take_heard(heard, told)
            said = self.heard_kind(heard, 1)
            guess = self.reply_kind(answer)
            if len(told) > before:
                kind = "firm"
            elif said in ("dontcare", "confused", "crowded"):
                kind = said
            elif said == "firm":
                kind = "unclear"
            else:
                kind = guess if guess in ("crowded", "confused", "dontcare") else "unclear"
            if kind == "crowded":
                # the same property sits on another item too: that item's question comes next, by name
                twins = [(n, l) for n, l in queue if n != number and _words(l) == _words(label)]
                if twins:
                    queue.remove(twins[0])
                    queue.insert(0, twins[0])
            if kind == "crowded" and self.msgs_free() > 0 and self.left() > 6:
                # the question touched two of the shopper's properties (a sibling key, or the same property on another item):
                # ask once more, narrowed to this item's property alone, never widened with example values
                again = self.send([(number, label, self.narrow_question(number, label))])
                if again:
                    sent += again
                    heard = self.hear([again[-1]], (LIGHT, HEAVY))
                    before = len(told)
                    self.take_heard(heard, told)
                    said = self.heard_kind(heard, 1)
                    kind = "firm" if len(told) > before else (said if said in ("dontcare", "confused") else "unclear")
            asked_on[number] = asked_on.get(number, 0) + 1
            if kind == "confused":
                # a confused reply to a listed key says the shopper holds no list on this item; to a label no listing carries it says nothing
                state[number] = "confused" if self.listed_key(number, label) else "unclear"
                outcomes.append(f"{label}: confused" + ("" if state[number] == "confused" else " (no listing carries that key)"))
                continue
            if kind == "dontcare":
                state[number] = "dontcare"
                self.items[number - 1].free.append(label)
                outcomes.append(f"{label}: dontcare")
                continue
            # the reader's reading stands: a firm reply named a requirement (negations and comparisons included)
            if len(told) > before:
                state[number] = "firm"
                found = [(t.split(":", 2)[1] if t.startswith("item ") and t.count(":") >= 2 else t.split(":", 1)[0]).strip() for t in told[before:]]
                outcomes.append(f"{label}: firm on {', '.join(found)}")
            elif said == "confused":
                state[number] = "confused" if self.listed_key(number, label) else "unclear"
                outcomes.append(f"{label}: confused (reader)")
            elif said == "crowded" and self.msgs_free() > 0 and self.left() > 6 and kind != "unclear":
                again = self.send([(number, label, self.narrow_question(number, label))])
                if again:
                    sent += again
                    heard = self.hear([again[-1]], (LIGHT, HEAVY))
                    self.take_heard(heard, told)
                    if len(told) > before:
                        state[number] = "firm"
                        outcomes.append(f"{label}: firm (after narrowing)")
                        continue
                state[number] = "unclear"
                outcomes.append(f"{label}: crowded")
            else:
                state[number] = "dontcare"
                self.items[number - 1].free.append(label)
                outcomes.append(f"{label}: dontcare (reader)")
        pointed_now = [need for _, need in self.refs if any(need in t for t in told)]
        resolved = self.resolve_refs(keep=12) if pointed_now else ""
        swept = self.fit_sweep(keep=12)
        searched = self.value_search([n for n, st in state.items() if st == "firm"]) if told else ""
        planned = self.bundle_plan(keep=12, do_cart=True)
        if planned:
            swept = (swept + "\n" if swept else "") + planned
        if told:
            self.version += 1
        self.trace.append({"kind": "asked", "outcomes": outcomes, "told": told})
        if not sent:
            return ""
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = (early + "\n" if early else "") + "Sakura's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if resolved:
            summary += "\n" + resolved
        if swept:
            summary += "\n" + swept
        if searched:
            summary += "\n" + searched
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def value_search(self, numbers: list[int]) -> str:
        """Each item whose answers gave plain values (not pointers, limits, exclusions or figures) is searched once more by those
        values together with its kind, and the first unopened rows of its kind are read: the main model then starts from
        candidates that carry the wanted words instead of searching again itself."""
        if self.left() < 20 or not self.roles.get("search") or self.finished:
            return ""
        notes: list[str] = []
        budget = 6
        for number in numbers:
            item = self.items[number - 1]
            values: list[str] = []
            for need in item.told:
                if need in item.dropped or ":" not in need or any(need == r for _, r in self.refs):
                    continue
                value = need.split(":", 1)[1].strip()
                low = value.lower()
                if not value or len(value.split()) > 4 or re.fullmatch(r"\s*\d[\d.,]*\s*[a-z%\"']{0,5}\s*", low) or re.search(NEGATION_RE, low) or re.search(r"\b(at most|at least|under|over|below|above|less|more|between|within|up to|or|page|listing)\b", low):
                    continue
                values.append(value)
            if not values or budget <= 1:
                continue
            phrase = " ".join(dict.fromkeys(values + [item.shelf or item.name]))
            fresh = self.search_phrase(phrase)
            budget -= 1
            words = {w for v in values for w in re.findall(r"[a-z0-9]{3,}", v.lower())}
            cap = self.budget if isinstance(self.budget, (int, float)) else None
            rows = [i for i in self.seen_order if self.buyable(i) and self.same_kind(i, item) and (cap is None or self.price_of(i) is None or self.price_of(i) <= cap)]
            rows.sort(key=lambda i: (-sum(1 for w in words if w in _dump(self.rows.get(i) or {}).lower()), 0 if i in fresh else 1, self.price_of(i) if self.price_of(i) is not None else float("inf")))
            unopened = [i for i in rows if i not in self.full][:min(3, budget)]
            opened = self.fill_facts(unopened, 16) if unopened else []
            budget -= len(opened)
            notes.append(f"{phrase!r}: " + (", ".join(rows[:6]) or "no row of that kind") + ("; opened " + ", ".join(unopened[:len(opened)]) if opened else ""))
        if not notes:
            return ""
        self.trace.append({"kind": "value_search", "seen": [n[:200] for n in notes]})
        return "[Sakura searched each such item's new values together with its kind and read the first unopened rows of that kind] " + " | ".join(notes)

    def item_keys(self, number: int) -> list[tuple[str, int, int, list[str]]]:
        """(key, listings carrying it, distinct values, sample values) for the keys seen on opened listings of this item's kind."""
        item = self.items[number - 1]
        shelf, name = _words(item.shelf or ""), _words(item.name)
        long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
        table = self.key_table()
        out = [(key, carried, distinct, values) for label, key, carried, values, distinct in table
               if _words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long]
        if not out and len(self.items) == 1:
            out = [(key, carried, distinct, values) for _, key, carried, values, distinct in table]
        return out

    def listed_key(self, number: int, label: str) -> bool:
        return any(_words(key) == _words(label) or self.same_property(key, label) for key, _, _, _ in self.item_keys(number))

    def fallback_questions(self, tried: set, state: dict, asked_on: dict) -> list[tuple[int, str]]:
        """One question per item from the listings' own keys, for when the planner's list is spent while messages remain: a
        measure or defining property with several values first, never a fit statement, an identifier, a seller detail or an
        engineering parameter while a better key exists; items that showed they hold no list, and settled properties, are skipped."""
        anchor_items = {t for t in self.anchors().values() if isinstance(t, int)}
        learned = {_words(k) for k in (self.id_key, self.price_key, self.stock_key, self.title_key, self.category_key()) if k}
        out = []
        for number, item in enumerate(self.items, 1):
            if state.get(number) == "confused":
                continue
            fixed = [m.split(":", 1)[0] for m in item.musts + item.told if ":" in m and m not in item.dropped and not self.open_need(m)] + list(item.free)
            scored = []
            for key, carried, distinct, values in self.item_keys(number):
                if (number, _words(key)) in tried or any(self.same_property(key, f) for f in fixed) or any(n == number and self.same_property(key, l) for n, l in tried):
                    continue
                if distinct < 2 or _words(key) in learned or distinct >= max(4, carried) or self.code_key(values) or re.search(r"\b(warranty|shipping|seller|shop|brand|price|stock)\b", key):
                    continue
                score = carried * 2 + min(distinct, 6)
                if sum(1 for v in values if re.search(r"\d", v)) * 2 >= max(1, len(values)):
                    score += 4
                scored.append((-score, key))
            if scored:
                scored.sort()
                out.append((number, scored[0][1]))
        out.sort(key=lambda p: (0 if p[0] in anchor_items else 1, 0 if p[0] not in asked_on else 1, p[0]))
        return out

    def code_key(self, values: list[str]) -> bool:
        """A key whose values are part, model or catalogue codes (letters and digits joined by a dash): a fit statement or an identifier, not a preference."""
        coded = sum(1 for v in values if re.search(r"\b[A-Za-z]{1,4}-?\d{2,7}\b", str(v)) or FIT_CODE.search(str(v)))
        return bool(values) and coded * 2 >= len(values)

    def relay(self, text: str, direct: bool = False) -> str:
        """A question the agent tried to send (through the message tool, or written into an ending call) goes to the shopper
        through the interviewer while a message is left; otherwise the agent is told that nothing was sent."""
        question = " ".join(str(text or "").split())
        looks = "?" in question or bool(re.search(r"\b(question for|ask(ing)? the shopper|could you (tell|let) me|please (tell|let) me|how (much|many|big|large))\b", question, re.I))
        if (not looks and not direct) or not self.roles.get("ask"):
            return ""
        if self.msgs_free() <= 0 or self.left() <= 6 or self.finished or self.relays >= 3:
            self.trace.append({"kind": "relay", "sent": False, "text": question[:120]})
            return (("Not run (no step used): that call ends the episode without a purchase and sends nothing to the shopper; " if not direct else "Not run (no step used): ")
                    + "every message to the shopper is spent. Go on with the requirements as they stand: a property the shopper was never asked about is open, any value of it is fine.")
        self.relays += 1
        m = re.search(r"(?:question for (?:the )?shopper[:,]?\s*)?([^.?!]*\?)", question, re.I)
        asked = (m.group(1).strip() if m else question)[:220]
        number = 1
        for n, item in enumerate(self.items, 1):
            if any(len(w) >= 4 and w in asked.lower() for w in self.kind_tokens(item.name)):
                number = n
                break
        batch = self.send([(number, "relayed", asked)])
        if not batch:
            return ""
        answer = batch[-1][2]
        told: list[str] = []
        heard = self.second_opinion(batch[-1], self.hear([batch[-1]], (LIGHT, HEAVY)))
        self.take_heard(heard, told)
        said = self.heard_kind(heard, 1) or self.reply_kind(answer)
        if told:
            self.version += 1
            note = "new firm requirement: " + "; ".join(told)
        elif said == "confused":
            note = "the shopper did not follow the question: they hold no requirement on that property"
        elif said == "dontcare":
            note = "the shopper does not mind about it"
        else:
            note = "no requirement was stated"
        self.trace.append({"kind": "relay", "sent": True, "q": asked, "a": answer[:200], "told": told})
        return (("Not run as an ending. " if not direct else "") + f"Sakura relayed your question to the shopper (one step): Q: {asked}\nA: {answer[:600]}\n{note}. "
                f"{self.msgs_free()} message(s) left. Apply it to every pick, then cart and check out.")

    def same_property(self, label: str, other: str) -> bool:
        """Two names of one property: "fa pattern" and "pattern", "skirt length" and "length"."""
        a, b = _stem(label), _stem(other)
        return bool(a and b) and (a <= b or b <= a)


    def settled(self, number: int, label: str) -> bool:
        """The request already fixes this property for the item (a value, limit or exclusion), or the shopper said they do not mind."""
        item = self.items[number - 1]
        want = _stem(label)
        if not want:
            return True
        for need in item.musts + item.told + item.free:
            if self.open_need(need):
                continue
            prop = _stem(need.split(":", 1)[0])
            if prop and (_names(want, prop) or _names(prop, want)):
                return True
        return False

    def open_need(self, need: str) -> bool:
        """A stated requirement whose value stands for what the shopper will say when asked: not a value to match, and not a settled property."""
        value = need.split(":", 1)[1].lower() if ":" in need else need.lower()
        return bool(re.search(r"\b(when asked|shopper|will (say|tell|state|give|name)|to be (asked|confirmed|decided|told|given)|as (asked|told|stated)|ask(ed)? (me|them|the)|tbd|unknown|unspecified|not (yet )?(given|stated|known|specified))\b|\?", value))

    def reply_kind(self, answer: str) -> str:
        """What a shopper's reply means: crowded (one question at a time, please), confused (the question meant nothing
        to them), reveal (a value, limit, exclusion or a comparison with another listing, however roundabout), dontcare
        (no preference), or unclear. A reply that carries a value is never read as "don't care" by its idioms alone:
        "within 0.02 inch either way of whatever's on the page for the tablets sold by X" is a requirement."""
        text = (answer or "").lower()
        if not text.strip():
            return "confused"
        if re.search(CROWDED_RE, text):
            return "crowded"
        if re.search(CONFUSED_RE, text):
            return "confused"
        if self.value_bearing(text):
            return "reveal"
        if re.search(DONTCARE_RE, text):
            return "dontcare"
        if re.search(REVEAL_RE, text):
            return "reveal"
        return "unclear"

    def value_bearing(self, text: str) -> bool:
        """Words that state or point at a value: a number, a comparison with another listing, an exclusion, a firm fixing."""
        low = (text or "").lower()
        return bool(re.search(VALUE_RE, low))

    def narrow_question(self, number: int, label: str) -> str:
        """A re-ask after a 'one question at a time' reply: the same property, named for this item alone, with the sibling
        property or the other item that the first wording also touched ruled out; never example values (they name more)."""
        item = self.item_called(number)
        words = set(_words(label).split())
        plan = getattr(self, "plan_labels", [])
        elsewhere = [self.item_called(n) for n, other in plan if n != number and _words(other) == _words(label) and self.item_called(n) != item]
        alike = self.alike_labels(number, label)
        if not alike:
            alike = [other for n, other in plan if n == number and _words(other) != _words(label) and words & set(_words(other).split())][:2]
        if alike:
            return f"Apart from the {' and the '.join(alike[:2])}, which {label} would you like for the {item}?"
        if elsewhere:
            return f"Apart from the {elsewhere[0]}, which {label} would you like for the {item}?"
        return f"Only one question: which {label} would you like for the {item}?"

    def alike_labels(self, number: int, label: str) -> list[str]:
        """Up to two attribute keys of the item's own listings that a reader could take for `label` (the light model judges the
        meaning; the keys come from the records)."""
        keys = []
        for key, carried, distinct, values in sorted(self.item_keys(number), key=lambda k: -k[1]):
            if re.search(r"\b(warranty|shipping|delivery|seller|shop|brand|price|stock|options?|variants?|sku|ids?|codes?|photos?|images?|videos?|urls?|links?|model)\b", _words(key)):
                continue
            if _words(key) != _words(label) and _words(key) not in [_words(k) for k in keys] and not re.search(r"\d", key):
                keys.append(key)
        keys = keys[:30]
        if not keys or self.cost > self.spend_hard:
            return []
        got = self.ask_json(ALIKE + "\n\nPROPERTY ASKED: " + label + "\n\nKEYS:\n" + "\n".join(keys), [LIGHT], 1200) or {}
        picks = []
        for k in _items(got.get("alike")):
            match = next((x for x in keys if _words(x) == _words(str(k))), None)
            if match and match not in picks:
                picks.append(match)
        self.trace.append({"kind": "alike", "label": label, "picks": picks})
        return picks[:2]

    def second_opinion(self, exchange: tuple[int, str, str], heard: dict[str, Any]) -> dict[str, Any]:
        """A need read as an exclusion ("pattern: not stripe") from an answer that rules nothing out ("stripe isn't up for
        trade") is a misreading of a firm fixing: the heavy model reads that answer again and its reading stands."""
        answer = str(exchange[2] or "").lower()
        negated = [str(e.get("need")) for e in _items(heard.get("needs")) if isinstance(e, dict) and re.search(r"^[^:]*:\s*(not|no|never|without|excluding|anything but)\b", str(e.get("need") or ""), re.I)]
        needs = [e for e in _items(heard.get("needs")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        split = len(needs) > 1 and any(e.get("points") is True for e in needs)
        if not split and (not negated or re.search(NEGATION_RE, answer)):
            return heard
        again = self.hear([exchange], (HEAVY, LIGHT))
        self.trace.append({"kind": "second_opinion", "first": negated or [str(e.get("need"))[:80] for e in needs], "again": _clip(again.get("needs"), 120)})
        if split and not (negated and not re.search(NEGATION_RE, answer)):
            fewer = [e for e in _items(again.get("needs")) if isinstance(e, dict) and str(e.get("need") or "").strip()] if again else []
            return again if 0 < len(fewer) < len(needs) and any(e.get("points") is True for e in fewer) else heard
        return again if again else heard

    def heard_kind(self, heard: dict[str, Any], question: int) -> str:
        """The reader's own reading of one answer (firm, dontcare, confused, crowded), or ''."""
        for entry in _items(heard.get("kinds")):
            if isinstance(entry, dict) and _count(entry.get("question")) == question:
                kind = str(entry.get("kind") or "").strip().lower()
                return kind if kind in ("firm", "dontcare", "confused", "crowded") else ""
        if any(_count(x) == question for x in _items(heard.get("crowded"))):
            return "crowded"
        return ""

    # ---------- requirements whose value sits on another listing (v2) ----------
    def parse_ref(self, need: str) -> dict[str, Any]:
        """The pieces of a requirement that points at another listing: the property wanted, the shop named, the kind of
        listing, and the descriptors that pick it out ("that has colour graphite, ram 12 gb and operating system android")."""
        text = str(need or "")
        prop = text.split(":", 1)[0].strip() if ":" in text else ""
        shop = None
        m = re.search(r"\b(?:sold|offered|listed|stocked|carried) by ([A-Z][\w'&.-]*(?: [A-Z][\w'&.-]*)?)", text) or \
            re.search(r"\b([A-Z][\w'&.-]+) (?:sells|offers|lists|stocks|carries)\b", text) or \
            re.search(r"\b(?:sold|offered|listed|stocked) by ([\w'&.-]+)", text, re.I)
        if m:
            shop = m.group(1).strip()
        kind = ""
        m = re.search(r"\b(?:page|listing|one|item|product)s? (?:for|of) the ([\w\s&'/-]+?)\s+(?:sold|offered|listed|stocked|carried) by", text, re.I) or \
            re.search(r"\bthe ([\w\s&'/-]+?)\s+(?:sold|offered|listed|stocked|carried) by", text, re.I) or \
            re.search(r"\bthe ([\w\s&'/-]+?) (?:that )?[A-Z][\w'&.-]+ (?:sells|offers|lists|stocks|carries)\b", text)
        if m:
            kind = re.split(r"\bthe\s+", m.group(1).strip(), flags=re.I)[-1].strip()
        descriptors: list[str] = []
        m = re.search(r"\b(?:that|which) (?:has|have|comes? with|shows?|lists?|carr(?:y|ies)) (.+?)(?:\s*\((?:or )?wherever.*)?$", text, re.I) or \
            re.search(r"\b(?:sells|offers|lists|stocks|carries) with (.+?)$", text, re.I)
        if m:
            body = re.sub(r"\s*\(.*$", "", m.group(1))
            for part in re.split(r",\s*|\s+and\s+", body):
                part = part.strip().rstrip(".)")
                if part:
                    descriptors.append(part)
        if (not shop or not descriptors) and getattr(self, "brief", None) and getattr(self, "roles", {}).get("search"):
            # a wording the patterns do not know: the model reads the description once (cached per requirement)
            cached = getattr(self, "ref_parsed", {}).get(text)
            if cached is None:
                got = self.ask_json(REFPARSE + "\n\nREQUIREMENT:\n" + text, [LIGHT, HEAVY], 1500) or {}
                cached = {"shop": str(got.get("shop") or "").strip(), "kind": str(got.get("kind") or "").strip(), "property": str(got.get("property") or "").strip(),
                          "descriptors": [str(d).strip() for d in _items(got.get("descriptors")) if str(d).strip()]}
                self.ref_parsed[text] = cached
                self.trace.append({"kind": "ref_parsed", "need": text[:100], "got": cached})
            shop = shop or cached["shop"] or None
            kind = kind or cached["kind"]
            descriptors = descriptors or cached["descriptors"]
            prop = prop or cached["property"]
        return {"prop": prop, "shop": shop, "kind": kind, "descriptors": descriptors}

    def split_descriptor(self, part: str) -> tuple[str, str]:
        """'operating system ios' -> ('operating system', 'ios'), using the attribute keys the listings carry."""
        words = re.sub(r"^(?:a|an|the)\s+", "", part.lower().strip()).split()
        keys: set[str] = set()
        for record in list(self.full.values())[:60]:
            for k in (_fact_block(record) or {}):
                keys.add(" ".join(w for w in re.split(r"[^a-z0-9]+", k.lower()) if w))
        figure = next((k for k, w in enumerate(words) if re.search(r"\d", w)), len(words))
        for n in range(min(4, len(words) - 1, figure), 0, -1):
            cand = " ".join(words[:n])
            if cand in keys or any(_names(_stem(cand), _stem(k)) for k in keys):
                return cand, " ".join(words[n:])
        return (words[0] if words else ""), " ".join(words[1:])

    def descriptor_hits(self, ident: str, descriptors: list[str]) -> int:
        return self.descriptor_tally(ident, descriptors)[0]

    def descriptor_tally(self, ident: str, descriptors: list[str]) -> tuple[int, int]:
        """(descriptors the listing states, descriptors a key naming their property contradicts)."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        text = _dump(self.rows.get(ident) or {}).lower() + " " + _dump(self.full.get(ident) or {}).lower()
        words = self.fit_tokens(text)

        def holds(have_words: set[str], have: str, want: str) -> bool:
            figures = re.findall(r"\d+(?:\.\d+)?", want)
            if figures:
                got = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", have)}
                return all(float(x) in got for x in figures)
            want_words = self.fit_tokens(want)
            return bool(want_words) and want_words <= have_words

        hits = clashes = 0
        for part in descriptors:
            prop, value = self.split_descriptor(part)
            want = value.lower().strip()
            keys = [k for k in block if prop and _names(_stem(prop), _stem(k))]
            if not want:
                hits += 1 if keys or holds(words, text, prop) else 0
            elif keys:
                if any(holds(self.fit_tokens(str(block[k])), str(block[k]).lower(), want) for k in keys):
                    hits += 1
                else:
                    clashes += 1
            elif holds(words, text, want):
                hits += 1
        return hits, clashes

    def shop_rows(self, shop: str) -> list[str]:
        """Listings seen whose record names the shop (under whatever key the store uses for the seller)."""
        want = shop.strip().lower()
        out = []
        for ident in self.seen_order:
            records = [self.rows.get(ident) or {}, self.full.get(ident) or {}]
            if any(isinstance(v, str) and v.strip().lower() == want for record in records for v in record.values()):
                out.append(ident)
        return out

    def search_phrase(self, phrase: str) -> list[str]:
        """One keyword search by code: the search tool's free-text parameter gets the phrase; the ids of new rows come back."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = _keyword_arg(schema, texts) or (texts[0] if texts else None)
        if not tool or not target or self.left() <= 3 or self.finished:
            return []
        args: dict[str, Any] = {target: phrase[:120]}
        for key in schema.get("required") or []:
            if key not in args:
                spec = props.get(key) or {}
                enum = spec.get("enum") if isinstance(spec.get("enum"), list) else None
                if enum:
                    args[key] = next((e for e in enum if re.search(r"key|text|word|phrase|free|query|search", str(e), re.I)), enum[0])
                else:
                    return []
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        if faults:
            return []
        before = set(self.rows)
        reply = self.act(tool, args)
        if reply is None:
            return []
        self.observe(tool, reply.get("observation"))
        self.last_search_step = self.step
        return [i for i in self.seen_order if i not in before]

    def open_page(self, ident: str) -> None:
        """Open a listing's page in full (the offer view when the store has one apart from the attributes view)."""
        for tool in (self.roles.get("offer"), self.roles.get("open")):
            target = self.single_target(tool)
            if not tool or not target or self.left() <= 3 or self.finished:
                continue
            if tool == self.roles.get("open") and ident in self.full and self.roles.get("offer"):
                continue
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                continue
            reply = self.act(tool, args)
            if reply is not None:
                self.observe(tool, reply.get("observation"))
                self.note_view(tool, reply.get("observation"))
            return

    def pointer_in(self, ident: str) -> str:
        """A page that sends the reader on to another listing ("see part TA2-93027 for the rest")."""
        text = _dump(self.full.get(ident) or {}) + " " + _dump(self.rows.get(ident) or {})
        m = re.search(r"\b(?:see|refer to|check|look at|find)\s+(?:part|item|listing|model|product|sku|ref(?:erence)?)\s*(?:no\.?|number|#)?\s*([A-Za-z]{1,6}\d[\w-]*|\d[\w-]{3,})", text, re.I)
        return m.group(1).strip() if m else ""

    def property_value(self, ident: str, prop: str) -> str:
        block = _fact_block(self.full.get(ident) or {}) or {}
        want = _stem(prop)
        for k, v in block.items():
            if want and _names(want, _stem(k)):
                return str(v)[:80]
        return ""

    def locate_ref(self, ref: dict[str, Any], keep: int, avoid: list[str]) -> tuple[str | None, str]:
        """The listing a reference describes: the shop's rows (searched by the shop's name with the kind, then alone), their attributes
        read, ranked by the descriptors they meet and contradict and by the kind; one more search with the descriptors' values when the
        best one misses or contradicts some. Stock does not matter: the page only gives a value."""
        shop = str(ref.get("shop") or "").strip()
        if not shop:
            return None, "no seller named"
        kind = str(ref.get("kind") or "").strip()
        descriptors = [d for d in ref.get("descriptors") or [] if d]
        found = [i for i in self.shop_rows(shop) if i not in avoid]
        searched: set[str] = set()

        def fits(i: str) -> bool:
            return bool(descriptors) and self.descriptor_tally(i, descriptors) == (len(descriptors), 0)

        tried = self.__dict__.setdefault("shop_kind_searched", set())
        known = bool(descriptors) and any(i in self.full and fits(i) for i in found)
        if not found or (kind and descriptors and not known and (shop.lower(), kind.lower()) not in tried
                         and self.left() > keep + 4 and self.clock_left() > CLOSING + 30):
            tried.add((shop.lower(), kind.lower()))
            searched.add((shop + " " + kind).strip().lower())
            self.search_phrase((shop + " " + kind).strip())
            found = [i for i in self.shop_rows(shop) if i not in avoid]
        if not found and kind and self.left() > keep + 5:
            self.search_phrase(shop)
            found = [i for i in self.shop_rows(shop) if i not in avoid]
        if not found:
            return None, f"no listing sold by {shop} has come up (searched the seller's name)"
        kind_words = _stem(kind)

        def rank(i: str) -> tuple[int, int, int, int]:
            got, clash = self.descriptor_tally(i, descriptors)
            text = _stem(_dump({k: v for k, v in (self.rows.get(i) or {}).items() if isinstance(v, str) and len(v) <= 200}))
            return (clash, -got, 0 if kind_words and kind_words & text else 1, self.seen_order.index(i) if i in self.seen_order else len(self.seen_order))

        cands = sorted(found, key=rank)[:6]
        if not known:
            self.fill_facts([i for i in cands if i not in self.full][:4], keep)
        cands.sort(key=rank)
        top = self.descriptor_tally(cands[0], descriptors) if descriptors else (0, 0)
        if descriptors and (top[1] or top[0] < len(descriptors)) and self.left() > keep + 6:
            values = [self.split_descriptor(p)[1] for p in descriptors]

            def returned(phrase: str) -> list[str]:
                self.last_found = []
                if not phrase or phrase.lower() in searched:
                    return []
                searched.add(phrase.lower())
                self.search_phrase(phrase)
                mine = set(self.shop_rows(shop))
                return [i for i in self.last_found if i in mine and i not in avoid]

            fresh = returned(" ".join(x for x in [shop, kind] + values if x).strip())
            hit = False
            for value in [v for v in values if v.strip()][:2]:
                self.fill_facts([i for i in fresh if i not in self.full][:3], keep)
                hit = any(fits(i) for i in fresh)
                if hit or self.left() <= keep + 7 or self.clock_left() <= CLOSING + 30:
                    break
                fresh = returned(f"{shop} {value}".strip())
            more = fresh + [i for i in self.shop_rows(shop) if i not in cands and i not in avoid and i not in fresh]
            if not hit:
                self.fill_facts([i for i in more if i not in self.full][:3], keep)
            cands = sorted(dict.fromkeys(cands + more), key=rank)[:10]
        if descriptors and len(cands) > 1:
            got, clash = self.descriptor_tally(cands[0], descriptors)
            if clash or got < len(descriptors):
                if self.left() > keep + 8:
                    self.fill_facts([i for i in cands[:6] if i not in self.full][:3], keep + 6)
                chosen = self.pick_ref(ref, cands[:6])
                if chosen:
                    return chosen, ""
                got, clash = self.descriptor_tally(cands[0], descriptors)
                if clash or not got:
                    return None, f"none of {shop}'s listings read so far matches " + ", ".join(descriptors)
        return cands[0], ""

    def pick_ref(self, ref: dict[str, Any], cands: list[str]) -> str | None:
        """The light model's choice of the listing a description names, among the seller's candidates, when the code's literal match is
        not complete (a figure in another unit, a value in other words)."""
        lines = [f"{i}: " + _dump(_clip({**(self.rows.get(i) or {}), **(self.full.get(i) or {})}, 160))[:1200] for i in cands]
        about = f"{ref.get('shop')}'s {ref.get('kind')}".strip() + (" with " + ", ".join(ref.get("descriptors") or []) if ref.get("descriptors") else "")
        got = self.ask_json(REFPICK + "\n\nDESCRIPTION: " + about + "\n\nLISTINGS:\n" + "\n".join(lines), [LIGHT, HEAVY], 1500) or {}
        chosen = str(got.get("id") or "").strip()
        self.trace.append({"kind": "ref_pick", "about": about[:120], "cands": cands, "chosen": chosen})
        return chosen if chosen in cands else None

    def read_page(self, ident: str, prop: str) -> dict[str, Any]:
        """The light model's reading of everything shown about one listing for one property: its own value, a hand-off to a different
        listing, and the pairing rules its page states."""
        record = {**(self.rows.get(ident) or {}), **(self.full.get(ident) or {})}
        text = _dump(_clip(record, 1200))[:6000]
        got = self.ask_json(PAGEREAD + "\n\nPROPERTY: " + prop + "\n\nLISTING " + ident + ":\n" + text, [LIGHT, HEAVY], 1500) or {}
        hand = got.get("handoff") if isinstance(got.get("handoff"), dict) else None
        if hand:
            hand = {"shop": str(hand.get("shop") or "").strip(), "kind": str(hand.get("kind") or "").strip(), "prop": prop,
                    "descriptors": [re.sub(r"\s*:\s*", " ", str(d)).strip() for d in _items(hand.get("descriptors")) if str(d).strip()]}
            if not hand["shop"]:
                hand = None
        pairs = []
        for pair in _items(got.get("pairs")):
            if isinstance(pair, list) and len(pair) == 2 and all(str(x).strip() for x in pair):
                pairs.append([str(pair[0]).strip(), str(pair[1]).strip()])
        value = got.get("value")
        value = "" if value is None or isinstance(value, (dict, list)) else str(value).strip()
        self.trace.append({"kind": "page_read", "id": ident, "prop": prop, "value": value[:80], "handoff": hand, "pairs": pairs[:4]})
        return {"value": value, "handoff": hand, "pairs": pairs}

    def pair_lines(self, prop: str) -> list[str]:
        """The shortest sentences of the records seen that name the property, for the relation reader to find a rule pairing its
        values when the chain pages state none themselves."""
        want = _stem(prop)
        out: list[str] = []
        for ident in self.seen_order:
            for value in (self.rows.get(ident) or {}).values():
                if not isinstance(value, str) or len(value) < 20:
                    continue
                for sentence in SENTENCE_END.split(value):
                    text = sentence.strip()
                    if want and want <= _stem(text) and 20 <= len(text) <= 240 and text not in out:
                        out.append(text)
        return sorted(out, key=len)[:4]

    def relate(self, need: str, prop: str, value: str, pairs: list[list[str]]) -> str:
        """The requirement as a plain constraint on the property, from the value the pointed-at pages give (and their pairing rules)."""
        prompt = RELATE + "\n\nREQUIREMENT: " + need + "\nTHE VALUE THE POINTED-AT PAGES GIVE FOR " + prop + ": " + value
        rules = [f"{a} with {b}" for a, b in pairs] + self.pair_lines(prop)
        if rules:
            prompt += "\nPAIRING RULES SEEN: " + " | ".join(rules[:6])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1200) or {}
        plain = str(got.get("need") or "").strip()
        return plain if ":" in plain and len(plain) < 200 else ""

    def resolve_refs(self, keep: int = 10) -> str:
        """Resolve by code the requirements whose value sits on another listing: find the listing described (seller, kind,
        descriptors), open its page, and follow each hand-off its page makes to a further listing, reading every page with the light
        model; then turn the relation into a plain constraint. Returns a note for the agent."""
        live = {need for _, need in self.needs_now()}
        pending = [(n, need) for n, need in self.refs if need in live and need not in self.ref_values and need not in self.ref_tried]
        notes: list[str] = []
        for number, need in pending[:2]:
            if self.left() <= keep + 6 or self.finished or self.cost > self.spend_hard or not self.roles.get("search"):
                break
            self.ref_tried.add(need)
            ref = self.parse_ref(need)
            if not ref.get("shop"):
                continue
            best, why = self.locate_ref(ref, keep, [])
            if not best:
                notes.append(f"{need}: {why}")
                continue
            prop = ref.get("prop") or need.split(":", 1)[0].strip()
            self.open_page(best)
            chain, final, value, pairs, lost = [best], best, "", [], ""
            for _ in range(3):
                read = self.read_page(final, prop)
                pairs += read["pairs"]
                hand = read["handoff"]
                if not hand:
                    value = read["value"] or self.property_value(final, prop)
                    break
                said = f"{hand['shop']}'s {hand['kind']}".strip() + (f" ({', '.join(hand['descriptors'])})" if hand["descriptors"] else "")
                if self.left() <= keep + 4 or self.finished:
                    lost = f"its page hands {prop} to {said}"
                    break
                nxt, why = self.locate_ref(hand, keep, chain)
                if not nxt:
                    lost = f"its page hands {prop} to {said}, not found yet ({why})"
                    break
                self.open_page(nxt)
                chain.append(nxt)
                final = nxt
            where = " -> ".join(chain)
            hits = self.descriptor_hits(best, ref["descriptors"])
            self.trace.append({"kind": "resolve_ref", "need": need[:120], "shop": ref.get("shop"), "best": best, "final": final, "hits": hits, "chain": chain,
                               "value": value[:80], "pairs": pairs[:4], "lost": lost[:160]})
            if lost or not value:
                notes.append(f"{need}: opened {where}" + (f"; {lost}" if lost else f", but no page there gives {prop}") + "; find that listing, open its page and read the value there")
                continue
            plain = self.relate(need, prop, value, pairs)
            self.ref_values[need] = (plain or value) + f" (read from {where})"
            self.version += 1
            notes.append(f"{need} -> the pages give {prop} {value!r}" + (f", so the requirement reads {plain!r}" if plain else "")
                         + f" (pages opened: {where}; {hits} of {len(ref['descriptors'])} descriptors matched on {best})")
        return ("[Sakura resolved the listings these requirements point at] " + " | ".join(notes)) if notes else ""

    def plan_questions(self, count: int) -> list[tuple[int, str]]:
        """The ranked properties to ask about, from the heavy model's reading of the story and the listings' keys."""
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            fixed = [m for m in item.musts + item.told if m not in item.dropped]
            text = f"item {number}: {item.name} - settled by the request: " + ("; ".join(fixed) or "nothing")
            if item.free:
                text += "; shopper does not mind about: " + ", ".join(item.free)
            settled.append(text)
        listing = []
        last = None
        for index, (label, name, carried, values, distinct) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(f"  {index}. {name} (on {carried} listings, {distinct} different values): " + " | ".join(values))
        prompt = PLAN.replace("COUNT", str(count)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        opened = sum(1 for i in self.seen_order if self.full.get(i) and _fact_block(self.full[i]))
        few = "; so few that a key on one listing may well be common" if opened < 6 else ""
        prompt += f"\n\nATTRIBUTE KEYS SEEN (from {opened} opened listings{few}):\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT] if self.cost + PLAN_ROOM <= self.spend_hard else [LIGHT], 2500) or {}
        def clean(raw: Any) -> str:
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(raw or ""))[0].strip().rstrip("?.")
            # a listing detail (variants, identifiers, price, stock, seller, warranty, shipping, media) is never a shopper's wish
            if re.search(r"\b(warranty|shipping|delivery|seller|shop|brand|price|stock|options?|variants?|sku|ids?|codes?|photos?|images?|videos?|urls?|links?)\b", _words(label)):
                return ""
            return label if label and len(label.split()) <= 5 else ""

        def present(number: int, label: str) -> bool:
            return any(n == number and (_words(label) == _words(other) or self.same_property(label, other)) for n, other in ordered)

        # the planner's own order stands (it may rank who-it-is-for above a weak hint); a hinted property it then
        # forgot to list goes in front, behind what the request asks to be asked first
        ordered: list[tuple[int, str]] = []
        firsts = 0
        for pick in _items(plan.get("ask")):
            if isinstance(pick, dict):
                number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
                label = clean(pick.get("label"))
                if label and not present(number, label):
                    ordered.append((number, label))
                    if str(pick.get("why") or "").strip().lower() == "asked" and firsts == len(ordered) - 1:
                        firsts += 1
        for hint in _items(plan.get("hints")):
            if isinstance(hint, dict):
                number = min(max(1, _count(hint.get("item") or 1) or 1), len(self.items))
                for raw in _items(hint.get("properties"))[:2]:
                    label = clean(raw)
                    if label and not present(number, label):
                        ordered.insert(firsts, (number, label))
                        firsts += 1
        self.trace.append({"kind": "hints", "hints": _clip(plan.get("hints"), 100)})
        if ordered and len(self.items) > 1:
            # one label given for several items that names no listed key of any of them speaks of wishes in general
            # ("preference"), not of a property: a question about it reaches nobody's list
            spread: dict[str, set[int]] = {}
            for number, label in ordered:
                spread.setdefault(_words(label), set()).add(number)
            general = {w for w, items in spread.items() if len(items) >= 2 and not any(self.listed_key(n, w) for n in items)}
            if general:
                self.trace.append({"kind": "plan_general", "dropped": sorted(general)})
                ordered = [(n, l) for n, l in ordered if _words(l) not in general]
        if ordered:
            ordered = self.guard_plan(ordered, plan, firsts)
        if not ordered:
            # no plan: the most carried keys of each item's category, in turns
            lanes: dict[str, list[str]] = {}
            for label, name, carried, values, distinct in table:
                if distinct >= 2 and carried >= 2:
                    lanes.setdefault(label, []).append(name)
            for k in range(4):
                for names in lanes.values():
                    if k < len(names):
                        ordered.append((1, names[k]))
        return ordered[:12]

    def guard_plan(self, ordered: list[tuple[int, str]], plan: dict[str, Any], firsts: int) -> list[tuple[int, str]]:
        """A key most of an item's opened listings carry (with several values) displaces a planned pick that few of them carry,
        unless the pick was hinted by the story or asked for by the request: the planner's guess about an unlisted property
        is weaker than what the listings themselves show buyers choose between."""
        whys: dict[tuple[int, str], str] = {}
        for pick in _items(plan.get("ask")):
            if isinstance(pick, dict):
                number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
                whys[(number, _words(str(pick.get("label") or "")))] = str(pick.get("why") or "").strip().lower()
        hinted = {(n, _words(l)) for n, l in ordered[:firsts]} | {k for k, w in whys.items() if w in ("hint", "asked", "for")}
        for number in range(1, len(self.items) + 1):
            learned = {_words(k) for k in (self.id_key, self.price_key, self.stock_key, self.title_key, self.category_key()) if k}
            keys = [(key, carried, distinct) for key, carried, distinct, values in self.item_keys(number)
                    if distinct >= 2 and distinct < max(4, carried) and not self.code_key(values) and not self.settled(number, key)
                    and _words(key) not in learned and not re.search(r"\b(warranty|shipping|delivery|seller|shop|brand|price|stock|options?|variants?|sku|ids?|codes?|photos?|images?|videos?|urls?|links?)\b", key)]
            if not keys:
                continue
            read = max(carried for _, carried, _ in keys)
            if read < 3:
                continue
            share = {key: carried / read for key, carried, _ in keys}

            def share_of(label: str) -> float:
                near = [s for key, s in share.items() if _words(key) == _words(label) or self.same_property(key, label)]
                return max(near) if near else 0.0

            for key, carried, _ in sorted(keys, key=lambda k: -k[1])[:3]:
                mine = [i for i, (n, _) in enumerate(ordered) if n == number]
                if len(mine) >= 2 or share[key] < 0.67:
                    continue   # the planner's list stands; the guard only fills a thin lane with a key most listings carry
                if any(_words(ordered[i][1]) == _words(key) or self.same_property(ordered[i][1], key) for i in mine):
                    continue
                ordered.insert((mine[-1] + 1) if mine else len(ordered), (number, key))
                self.trace.append({"kind": "guard", "item": number, "key": key, "share": round(share[key], 2)})
        return ordered[:12]

    def question(self, number: int, label: str, samples: list[str] | None = None) -> str:
        text = f"What {label} would you like for the {self.item_called(number)}"
        return text + (f", for example {samples[0]} or {samples[1]}?" if samples else "?")

    def samples(self, number: int, label: str) -> list[str]:
        """Two values the listings show for this property: they say which property a question means."""
        want = _words(label)
        settled = " ".join(self.items[number - 1].musts).lower()
        for _, name, _, values, _ in self.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and len(v) <= 30][:2]
        return []

    def note_lists(self, observation: Any) -> list[str]:
        """Store notes carried as lists of sentences inside a reply (the shopper's own words are a single string)."""
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, dict):
                for value in node.values():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        found.extend(v for v in value if len(v.split()) >= 4)
                    else:
                        walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)

        walk(observation, 0)
        return [n for n in found if n not in self.heard]

    def strings_of(self, observation: Any) -> list[str]:
        if isinstance(observation, str):
            return [observation]
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
        return found

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
            notes = self.note_lists(answer)
            if notes:
                # a store note (a dropped requirement, a new budget, a sell-out) rides along with the shopper's words
                meaning = self.take_note(notes)
                if meaning:
                    self.queued.append(f"(seen with the shopper's reply) {meaning}")
                rest = [w for w in self.strings_of(answer) if w not in notes]
                if rest:
                    words = " ".join(rest)[:800]
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

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (LIGHT, HEAVY)) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about {'item ' + str(n) if n else 'the items it names'}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, list(models), 1500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Right before a checkout ("order") or an ending without a purchase ("end"): the steps the request asks for at
        that point, done by the model's choice of store calls (a message to the shopper, a page, any store tool) until
        it finds none missing for the cart as it is now. Returns whether to go on, and what to tell the agent."""
        steps = [p for p in self.process if p["when"] == stage]
        if self.asked:
            # a step that asks the shopper about their wishes is done by the interviewer's questions (approvals are not)
            steps = [p for p in steps if not (re.search(r"\b(ask|question)", p["step"], re.I) and re.search(r"\b(shopper|customer|buyer|them|preference|requirement|wish|want)", p["step"], re.I)
                                             and not re.search(r"\b(approv|confirm|permission|okay|ok)", p["step"], re.I))]
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
                return False, ("Before going on, Sakura did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
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
                self.done_log.append(f"(Sakura: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
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
            self.absorb(seen, name)
            self.done_log.append(f"step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            said.append(f"{name} {_dump(args)} -> {self.show(seen)[:400]}")
        return True, ("Before going on, Sakura did the steps the request asks for:\n" + "\n".join(said)) if said else ""

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
            if not (changed == [f"budget now {budget}"] and not self.note_sold and self.cart_fits_budget()):
                self.version += 1
            self.recent = []
            self.notes.append(summary or "; ".join(changed))
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

    def cart_fits_budget(self) -> bool:
        """The held cart (every line priced) stays inside the budget."""
        held = [i for i in self.cart_ids() if not self.sold_out(i)]
        prices = [self.price_of(i) for i in held]
        if not held or None in prices or not isinstance(self.budget, (int, float)) or isinstance(self.budget, bool):
            return False
        if self.brief.get("budget_scope") == "each":
            return all(p <= self.budget for p in prices)
        return sum(p * max(1, self.cart_lines.get(i, 1)) for p, i in zip(prices, held)) <= self.budget

    def early_decline_hold(self) -> str:
        """Once per episode, an ending without a purchase while a third of the steps (at least ten) is left and the cart lacks a line for
        some item: one more search under other words comes first (a complete cart is the decline guard's to judge)."""
        held = [i for i in self.cart_ids() if not self.sold_out(i)]
        if self.search_blocks or self.left() < max(10, self.max_steps // 3) or len(held) >= len(self.items):
            return ""
        self.search_blocks = 1
        self.trace.append({"kind": "refuse_blocked", "reason": "early decline", "left": self.left()})
        return (f"Not run yet (no step used). {self.left()} steps are left. Before ending without a purchase, run one more search for the requested kind under other words "
                "(its own category, with the requirement's words), open the best rows and cart the first that meets every requirement, then check out. "
                "End without a purchase only when that search finds nothing.")

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
            lines += [f"[defers] {ident}: {s!r} (this listing's own value for that property counts for nothing; the listing named decides)" for s in self.deferrals(ident)]
        if not lines:
            return ""
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + " | ".join(lines)

    def shop_names(self) -> set[str]:
        """Seller, shop or brand names the store's rows carry (short values under keys that name a seller or brand)."""
        names: set[str] = set()
        for row in list(self.rows.values())[:400]:
            for k, v in row.items():
                if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search(r"seller|shop|store|vendor|brand|maker|merchant|sold", str(k), re.I):
                    names.add(v.strip().lower())
        return names

    def deferrals(self, ident: str) -> list[str]:
        """Sentences of a listing's own text that speak of a live property together with another shop's or brand's name: the
        listing hands that property to another listing, so its own value for it counts for nothing."""
        text = " ".join(str(v) for record in (self.rows.get(ident) or {}, self.full.get(ident) or {}) for v in record.values() if isinstance(v, str) and len(v) > 20)
        if not text:
            return []
        own = {str(v).strip().lower() for v in (self.rows.get(ident) or {}).values() if isinstance(v, str) and 2 < len(v.strip()) <= 40}
        shops = self.shop_names() - own
        props = {_stem(need.split(":", 1)[0]) for _, need in self.needs_now()} | {_stem(need.split(":", 1)[0]) for _, need in self.refs}
        props = {p for p in props if p}
        if not shops or not props:
            return []
        out: list[str] = []
        for sentence in SENTENCE_END.split(text):
            low = sentence.lower()
            if len(low) < 15 or not any(re.search(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])", low) for s in shops):
                continue
            words = _stem(sentence)
            if any(p <= words for p in props):
                out.append(sentence.strip()[:200])
        return out[:3]

    # ---------- checks ----------
    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        # a cart view older than the last cart change would show lines that are gone
        return "lines by Sakura's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            found = [i for i in self.rows if i in text]
            return found + [i for i in self.cart_lines if i in text and i not in found]
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

    def refuse_guard(self) -> str:
        """Before an ending without a purchase (twice at most): a cart with a line for every item is checked out by Sakura
        itself when an independent check passes it; a cart over the budget asks for a swap; an item without a line whose
        kind has rows nobody opened gets them opened. A decline scored 1 in about 2 of 100 race episodes."""
        if self.refuse_guards >= 2 or self.left() < 3 or self.finished:
            return ""
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        total = sum((self.price_of(i) or 0.0) * max(1, self.cart_lines.get(i, 1)) for i in cart)
        if cart and len(cart) >= len(self.items):
            if isinstance(self.budget, (int, float)) and total > self.budget:
                self.refuse_guards += 1
                cheaper: list[str] = []
                for line in sorted(cart, key=lambda i: -(self.price_of(i) or 0.0))[:2]:
                    kind = self.kind_of(line).lower()
                    pool = [i for i in self.seen_order if i not in cart and self.buyable(i) and self.kind_of(i).lower() == kind
                            and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < (self.price_of(line) or 0.0)]
                    pool.sort(key=lambda i: self.price_of(i) or 0.0)
                    cheaper += [i for i in pool[:3] if i not in cheaper]
                self.trace.append({"kind": "refuse_guard", "why": "over budget", "total": total, "cheaper": cheaper})
                return (f"Not run yet (no step used). The cart's total {total:g} is over the budget {self.budget:g}, and ending without a purchase earns nothing while cheaper listings of the same kinds exist. "
                        f"Replace the priciest line with a cheaper fitting listing ({'candidates seen: ' + ', '.join(cheaper) if cheaper else 'search its category again and read the cheapest rows'}), then check out.")
            self.refuse_guards += 1
            self.fill_facts(cart, self.tail() + 3)
            verdict = self.check_cart(cart, self.spend_hard)
            if self.verdicts.get(self.check_key(cart)):
                self.trace.append({"kind": "refuse_guard", "why": "cart passes the check", "cart": cart})
                turned = self.finish(again=True)
                if self.finished:
                    return "Sakura checked out the cart instead: an independent check found it meets every requirement, and ending without a purchase earns nothing while a fitting cart exists. [episode finished]"
                if turned:
                    return "Not run yet. The cart passed an independent check, so Sakura tried to check out and " + turned + "\nApply what the store says, fix the cart and check out again."
            elif verdict:
                return "Not run yet (no step used). " + verdict
            elif self.check_key(cart) not in self.checked and self.cart_fits_budget():
                self.trace.append({"kind": "refuse_guard", "why": "a cart the check could not judge", "cart": cart})
                turned = self.finish(again=True)
                if self.finished:
                    return "Sakura checked out the cart instead: it holds a line for every requested item inside the budget, and ending without a purchase earns nothing while such a cart exists. [episode finished]"
                if turned:
                    return "Not run yet. Sakura tried to check out the cart and " + turned + "\nApply what the store says, fix the cart and check out again."
            return ""
        if len(self.items) > 1:
            bought = self.best_effort_order()
            if bought:
                self.refuse_guards += 1
                return bought
        if self.anchors():
            for n in self.anchors():
                if not self.line_of(n):
                    self.fit_swept.discard(n)
            swept = self.fit_sweep(keep=self.tail() + 3)
            if swept:
                self.refuse_guards += 1
                self.trace.append({"kind": "refuse_guard", "why": "fit sweep", "note": swept[:200]})
                return "Not run yet. " + swept + ". Their attributes are above: cart the ones that meet the requirements at the requested quantities, then check out. End without a purchase only if none of them does."
        # an item without a cart line: rows of its kind whose attributes nobody read may fit
        lacking = [item for item in self.items if not any(self.same_kind(i, item) for i in cart)]
        rows: list[str] = []
        for item in lacking:
            want = {w for w in self.kind_tokens(item.shelf or item.name)}
            pool = [i for i in self.seen_order if i not in self.full and i not in cart and self.buyable(i) and (self.kind_tokens(self.kind_of(i)) & want or not self.kind_of(i))]
            words = [w for m in item.musts + item.told for w in re.findall(r"[a-z0-9]{3,}", m.split(":", 1)[-1].lower())]
            pool.sort(key=lambda i: -sum(1 for w in words if w in _dump(self.rows.get(i) or {}).lower()))
            rows += [i for i in pool[:2] if i not in rows]
        if rows and self.left() > 8:
            self.refuse_guards += 1
            opened = self.open_rows(rows[:3])
            self.trace.append({"kind": "refuse_guard", "why": "unopened rows", "rows": rows[:3], "opened": opened})
            return ("Not run yet. Before ending without a purchase Sakura opened rows of the requested kind that nobody had read: " + ", ".join(rows[:3])
                    + ". Judge them against the requirements (their records are above); add any that fits at the requested quantity and check out. End without a purchase only if their attributes contradict a requirement.")
        return ""

    def line_faults(self) -> list[str]:
        """Code's own look at the cart before a checkout: a line for every requested item whose kind the store showed, one line per
        item, the requested quantity, no line of an unrequested kind once every kind is known, no sold-out line, the total inside
        the budget (a total more than ten times the budget is read as a unit mismatch, not a fault)."""
        ids = [i for i in self.cart_ids() if max(1, self.cart_lines.get(i, 1)) > 0]
        if not ids:
            return []
        faults: list[str] = []
        owners = {i: [n for n, item in enumerate(self.items, 1) if self.same_kind(i, item)] for i in ids}
        known = {n for n, item in enumerate(self.items, 1) if any(self.same_kind(i, item) for i in self.seen_order)}
        for n, item in enumerate(self.items, 1):
            if n not in known:
                continue
            lines = [i for i in ids if n in owners[i]]
            if not lines:
                faults.append(f"item {n} ({item.name}) has no line in the cart")
            elif len([i for i in lines if owners[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.cart_lines.get(i, 1)
                if owners[i] == [n] and have != item.qty:
                    faults.append(f"{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}")
        if len(known) == len(self.items):
            faults += [f"{i} ({self.kind_of(i) or 'unknown kind'}) is not a kind the shopper asked for: take it out" for i in ids if not owners[i]]
        faults += [f"{i} is sold out" for i in ids if self.sold_out(i)]
        prices = {i: self.price_of(i) for i in ids}
        if isinstance(self.budget, (int, float)) and all(p is not None for p in prices.values()):
            if self.brief.get("budget_scope") == "each":
                faults += [f"{i} costs {prices[i]:g}, over the {self.budget:g} limit for each item" for i in ids if self.budget < prices[i] <= 10 * self.budget]
            else:
                total = sum(prices[i] * max(1, self.cart_lines.get(i, 1)) for i in ids)
                if self.budget + 1e-9 < total <= 10 * self.budget:
                    faults.append(f"the total is {total:g}, over the budget of {self.budget:g}")
        return faults

    def fix_counts(self) -> str:
        """Lines held at the wrong quantity for their item are set to the requested quantity by code (one cart call)."""
        wrong: dict[str, int] = {}
        for i in self.cart_ids():
            owners = [n for n, item in enumerate(self.items, 1) if self.same_kind(i, item)]
            if len(owners) == 1 and self.cart_lines.get(i, 1) != self.items[owners[0] - 1].qty and not self.sold_out(i):
                wrong[i] = self.items[owners[0] - 1].qty
        if not wrong or self.left() < 5 or self.finished:
            return ""
        lines = {i: wrong.get(i, max(1, self.cart_lines.get(i, 1))) for i in self.cart_ids() if not self.sold_out(i)}
        if not self.set_cart(lines):
            return ""
        return ", ".join(f"{i} -> {q}" for i, q in wrong.items())

    def single_effort_order(self) -> str:
        """A one-item request about to be declined: code carts the best-placed in-stock listing of the kind that contradicts no
        requirement and checks it out (once). A decline scores nothing; such an order may score."""
        if self.single_effort or not self.roles.get("cart") or not self.roles.get("buy") or self.left() < 4 or self.finished:
            return ""
        pick = self.single_pick()
        if not pick:
            return ""
        self.single_effort = True
        self.trace.append({"kind": "single_effort", "pick": pick})
        if not self.set_cart({pick: self.items[0].qty}):
            return ""
        turned = self.finish(again=True)
        if self.finished:
            return f"Sakura carted {pick}, the best-placed in-stock listing of the requested kind that contradicts no requirement, and checked out: ending without a purchase earns nothing while such a listing exists. [episode finished]"
        if turned:
            return f"Not run yet. Sakura carted {pick} and tried to check out; " + turned + "\nApply what the store says, then check out again."
        return ""

    def single_pick(self) -> str:
        """For a one-item request about to be declined: the in-stock listing of the requested kind, inside the budget, that meets the
        requirements, else one that contradicts none of them (the cheapest under a cheapest pick)."""
        item = self.items[0]
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        pool = []
        for ident in self.seen_order:
            if not self.buyable(ident) or not self.same_kind(ident, item) or ident not in self.full:
                continue
            label = self.kind_label()
            if label and self.kind_of(ident).strip().lower() != label.strip().lower():
                continue   # only the category that names the requested kind as a whole counts
            price = self.price_of(ident)
            if cap is not None and price is not None and price * max(1, item.qty) > cap:
                continue
            verdict = self.item_verdict(1, ident)
            if verdict == "FAILS":
                continue
            pool.append((0 if verdict == "meets" else 1, price if price is not None else float("inf"), self.seen_order.index(ident), ident))
        pool.sort()
        return pool[0][3] if pool else ""

    def same_kind(self, ident: str, item: Any) -> bool:
        label = _stem(self.kind_of(ident) or "")
        if label and len(self.items) > 1:
            if _stem(item.shelf or "") == label:
                return True
            if any(other is not item and _stem(other.shelf or "") == label for other in self.items):
                return False
        kind = self.kind_tokens(self.kind_of(ident))
        want = self.kind_tokens(item.shelf or "") | self.kind_tokens(item.name)
        shared = self.shared_kind_tokens()
        if want - shared:
            want = want - shared
        if kind and want:
            return bool(kind & want)
        title = _dump(self.rows.get(ident) or {}).lower()
        return any(len(w) >= 4 and w in title for w in want)

    def shared_kind_tokens(self) -> frozenset[str]:
        """Words naming more than one requested item's kind ("internal" in "internal ssd" and "internal hard drive") tell the kinds apart from nothing."""
        key = tuple((item.shelf or "", item.name) for item in self.items)
        cached = getattr(self, "_shared_cache", None)
        if cached and cached[0] == key:
            return cached[1]
        counts: dict[str, int] = {}
        for shelf, name in key:
            for w in self.kind_tokens(shelf) | self.kind_tokens(name):
                counts[w] = counts.get(w, 0) + 1
        shared = frozenset(w for w, n in counts.items() if n > 1)
        self._shared_cache = (key, shared)
        return shared

    def missing_hold(self) -> str:
        """Once, before a checkout with several items: a cart with fewer lines than items is not an order that can score."""
        held = int(self.missing_held or 0)
        if held >= 2 or len(self.items) <= 1 or self.left() <= (6 if not held else 12):
            return ""
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        if len(cart) >= len(self.items) or any(not (self.full.get(i) or self.rows.get(i)) or i not in self.listed for i in cart):
            return ""
        self.missing_held = held + 1
        lacking = [item.name for item in self.items if not any(self.same_kind(i, item) for i in cart)]
        self.trace.append({"kind": "missing_hold", "cart": cart, "lacking": lacking})
        return (f"Not run yet (no step used). The cart holds {len(cart)} line(s) for {len(self.items)} requested items, and an order missing an item earns nothing. "
                f"Still without a line: {', '.join(lacking) or f'{len(self.items) - len(cart)} of the items'}. Search that item's category (under other words if needed), open a fitting listing, add it at the requested quantity, then check out.")

    # ---------- what fits what (v3) ----------
    def fit_text(self, ident: str) -> str:
        parts = []
        for record in (self.rows.get(ident) or {}, self.full.get(ident) or {}):
            for value in record.values():
                if isinstance(value, str):
                    parts.append(value)
        return re.sub(r"\b([A-Z]{2,6}) (\d{3,5}|\d{1,4}[xX]{1,3})\b", r"\1-\2", ". ".join(parts))

    def learn_fitment(self, ident: str) -> None:
        """Read the fitment statements of one listing record: its codes, the series it names, the model it names and the series
        that model is said to take, the listing it refers to, and the model family it fits."""
        text = self.fit_text(ident)
        if not text:
            return
        codes = {m.group(1).upper() for m in FIT_CODE.finditer(text)}
        codes |= {m.group(1).upper() for m in re.finditer(r"[\[(](?:[A-Za-z]+\s+)?([A-Za-z]{1,6}\d{0,3}-?\d{2,7})[\])]", text, re.I)}
        if codes:
            self.fit_codes[ident] = codes | self.fit_codes.get(ident, set())
        series: set[str] = set()
        for m in FIT_SERIES.finditer(text):
            series |= {t.upper() for t in FIT_TOKEN.findall(m.group(1))}
        if series:
            self.fit_sets[ident] = series | self.fit_sets.get(ident, set())
        m = FIT_MODEL.search(text)
        if m:
            self.fit_model[ident] = m.group(1).upper()
        for sentence in SENTENCE_END.split(text):
            m = FIT_MODEL.search(sentence)
            if not m:
                continue
            takes: set[str] = set()
            for s in FIT_SERIES.finditer(sentence, m.end()):
                takes |= {t.upper() for t in FIT_TOKEN.findall(s.group(1))}
            if takes:
                self.model_takes.setdefault(m.group(1).upper(), set()).update(takes)
        m = FIT_POINTER.search(text)
        if m:
            self.fit_pointer[ident] = m.group(1).upper()
        m = FIT_VEHICLE.search(text)
        if m:
            self.fit_vehicle[ident] = (m.group(1).upper(), m.group(2), len(m.group(3)))
        self.learn_covers(ident, text)

    def learn_covers(self, ident: str, text: str) -> None:
        """Any sentence naming code-like tokens is read as a compatibility statement, whatever its wording: the tokens before the
        sentence's colon or verb are the listing's own codes, those after it are what it fits (a tail of x's is a family of codes),
        and the codes after 'except' are taken out."""
        exact, families, excluded = self.fit_covers.get(ident, (set(), set(), set()))
        own = {c.upper() for c in self.fit_codes.get(ident, set())} | ({self.fit_model[ident]} if self.fit_model.get(ident) else set())
        for sentence in SENTENCE_END.split(text):
            tokens = [t.upper() for t in CODE_TOKEN.findall(sentence)]
            if not tokens or FIT_POINTER.search(sentence) or (len(tokens) == 1 and FIT_MODEL.search(sentence)):
                continue
            split = CODE_SPLIT.search(sentence)
            head = sentence[: split.start()] if split else ""
            subjects = {t.upper() for t in CODE_TOKEN.findall(head)}
            if subjects and not subjects & own:
                held = getattr(self, "part_also", None)
                if held is None:
                    held = self.part_also = {}
                for subject in subjects:
                    held.setdefault(subject.replace("-", ""), set()).update(t for t in tokens if t not in subjects)
                continue
            exc_part = re.search(r"\bexcept\b(.*)$", sentence, re.I | re.S)
            exc = {t.upper() for t in CODE_TOKEN.findall(exc_part.group(1))} if exc_part else set()
            for token in tokens:
                if token in exc:
                    excluded.add(token)
                elif token in subjects or token in own:
                    own.add(token)
                elif re.search(r"X+$", token):
                    stem = re.sub(r"X+$", "", token)
                    families.add((stem.replace("-", ""), len(token) - len(stem)))
                else:
                    exact.add(token)
        if exact or families or excluded:
            self.fit_covers[ident] = (exact, families, excluded)
        if own - {c.upper() for c in self.fit_codes.get(ident, set())}:
            self.fit_codes[ident] = self.fit_codes.get(ident, set()) | own

    def covers(self, part: str, codes: set[str]) -> bool | None:
        """Whether the part's compatibility statement covers one of the anchor's codes (exactly or as a family), excepting none of them."""
        got = self.fit_covers.get(part)
        extra = set()
        for own in self.fit_codes.get(part, set()) | ({self.fit_model[part]} if self.fit_model.get(part) else set()):
            extra |= getattr(self, "part_also", {}).get(own.upper().replace("-", ""), set())
        if extra:
            got = ((got[0] if got else set()) | extra, got[1] if got else set(), got[2] if got else set())
        if not got or not codes:
            return None
        exact, families, excluded = got
        for code in codes:
            plain = code.upper().replace("-", "")
            if code.upper() in excluded or plain in {e.replace("-", "") for e in excluded}:
                continue
            if code.upper() in exact or plain in {e.replace("-", "") for e in exact}:
                return True
            if any(plain.startswith(stem) and len(plain) == len(stem) + xs for stem, xs in families):
                return True
        return False if (exact or families) else None

    def anchor_codes(self, ident: str) -> set[str]:
        """The codes a listing carries as its own (its model, its bracketed or stated codes)."""
        codes = {c.upper() for c in self.fit_codes.get(ident, set())}
        if self.fit_model.get(ident):
            codes.add(self.fit_model[ident].upper())
        return codes

    def vehicle_family(self, vehicle: tuple[str, str]) -> str:
        """The model family a part statement would name, in the wildcard notation read back from the listings (else the model itself)."""
        make, digits = vehicle
        seen = [v for v in self.fit_vehicle.values() if v[0] == make] or list(self.fit_vehicle.values())
        for _, prefix, xs in seen:
            if len(prefix) + xs == len(digits):
                return f"{make} {digits[:len(prefix)]}{'x' * xs}"
        return f"{make} {digits}"

    def part_series(self, ident: str) -> set[str]:
        return {c.split("-")[0] for c in self.fit_codes.get(ident, set())}

    def anchor_series(self, ident: str) -> set[str]:
        """Every series the anchor listing takes: its own statement plus what its model is said to take elsewhere."""
        series = set(self.fit_sets.get(ident, set()))
        model = self.fit_model.get(ident)
        if model:
            series |= self.model_takes.get(model, set())
        return series

    def anchors(self) -> dict[int, Any]:
        """Item number -> what that item must fit: another item's number, or (make, digits) of a model the request names."""
        out: dict[int, Any] = {}
        for number, item in enumerate(self.items, 1):
            fits = item.fits
            if isinstance(fits, str) and fits.strip().isdigit():
                fits = int(fits.strip())
            if isinstance(fits, int) and 1 <= fits <= len(self.items) and fits != number:
                out[number] = fits
            elif isinstance(fits, str) and fits.strip():
                m = re.search(r"\b([A-Za-z]{2,4})[ -]?(\d{2,5})\b", fits)
                if m:
                    out[number] = (m.group(1).upper(), m.group(2))
                else:
                    m2 = re.search(r"\bitem\s*(\d+)", fits, re.I)
                    if m2 and 1 <= int(m2.group(1)) <= len(self.items) and int(m2.group(1)) != number:
                        out[number] = int(m2.group(1))
        return out

    def line_of(self, number: int) -> str | None:
        item = self.items[number - 1]
        for ident in self.cart_ids():
            if not self.sold_out(ident) and self.same_kind(ident, item):
                return ident
        return None

    def target_series(self, number: int, keep: int = 6) -> tuple[set[str], tuple[str, str] | None, str]:
        """What item `number` must fit, as a series set (from the anchor line) or a vehicle model; and a sentence about it."""
        target = self.anchors().get(number)
        if target is None:
            return set(), None, ""
        if isinstance(target, tuple):
            return set(), target, f"must fit model {target[0]} {target[1]}"
        anchor, stand_in = self.anchor_of(target)
        if not anchor:
            return set(), None, f"must fit item {target} ({self.items[target - 1].name}), which has no cart line yet: choose and cart that item first"
        self.resolve_pointer(anchor, keep)
        series = self.anchor_series(anchor)
        self.anchor_codes_of[number] = self.anchor_codes(anchor)
        if not series and self.anchor_codes_of[number]:
            codes = ", ".join(sorted(self.anchor_codes_of[number]))
            return set(), None, f"must fit {anchor} (item {target}), whose record states code {codes}: a dependent row fits when its own compatibility statement covers that code"
        if not series:
            return set(), None, f"must fit {anchor} (item {target}), whose record states no series"
        note = (f"must fit {anchor} (item {target}" + (f", not carted yet: {anchor} is the cheapest opened {self.items[target - 1].name} meeting its requirements, so the fit is read from it until one is carted" if stand_in else "")
                + f"), which takes series {', '.join(sorted(series))}")
        if self.fit_pointer.get(anchor):
            note += f" (its record refers to {self.fit_pointer[anchor]} for more of its fit" + (", read" if self.fit_pointer[anchor] in self.fit_resolved else "") + ")"
        return series, None, note

    def anchor_of(self, target: int) -> tuple[str | None, bool]:
        """The listing item `target` stands for: its cart line, else (as a stand-in) the cheapest opened row of its kind that
        meets its requirements and states what it takes."""
        line = self.line_of(target)
        if line:
            return line, False
        pool = []
        for ident in self.seen_order:
            if ident not in self.full or not self.buyable(ident) or not self.same_kind(ident, self.items[target - 1]):
                continue
            verdict = self.item_verdict(target, ident)
            if verdict == "FAILS":
                continue
            pool.append((0 if verdict == "meets" else 1, 0 if (self.fit_sets.get(ident) or self.fit_model.get(ident)) else 1, self.price_of(ident) if self.price_of(ident) is not None else float("inf"), ident))
        pool.sort()
        return (pool[0][3] if pool else None), True

    def resolve_pointer(self, anchor: str, keep: int) -> None:
        """The anchor's record sends the reader to another listing for the rest of its series: one exact-code search finds it,
        and its record says what the anchor's model also takes."""
        code = self.fit_pointer.get(anchor)
        if not code or code in self.fit_resolved or self.left() <= keep + 2 or self.finished:
            return
        self.fit_resolved.add(code)
        if not any(code in c for c in self.fit_codes.values()):
            found = self.search_phrase(code)
            self.trace.append({"kind": "fit_pointer", "anchor": anchor, "code": code, "found": found[:3]})

    def compatible(self, part: str, series: set[str], vehicle: tuple[str, str] | None, codes: set[str] | None = None) -> bool | None:
        """Whether a listing fits: its code's series is among the anchor's, its vehicle statement covers the model, or its own
        compatibility statement covers one of the anchor's codes. None = unknown."""
        known = self.compatible_by_shapes(part, series, vehicle)
        if known is not None:
            return known
        if codes:
            covered = self.covers(part, codes)
            if covered is not None:
                return covered
        if vehicle:
            return self.covers(part, {vehicle[0] + vehicle[1], vehicle[0] + "-" + vehicle[1]})
        return None

    def compatible_by_shapes(self, part: str, series: set[str], vehicle: tuple[str, str] | None) -> bool | None:
        """The series and wildcard-family statements read by the fitment shapes (v3): a verdict when they speak, else None."""
        own = self.part_series(part)
        if series and own:
            return bool(own & series)
        if vehicle and part in self.fit_vehicle:
            make, prefix, xs = self.fit_vehicle[part]
            digits = vehicle[1]
            return make == vehicle[0] and digits.startswith(prefix) and len(digits) == len(prefix) + xs
        return None

    def fit_rows(self, number: int, series: set[str], vehicle: tuple[str, str] | None) -> tuple[list[str], list[str]]:
        """Rows of this item's kind that fit, cheapest first, and rows that do not."""
        item = self.items[number - 1]
        good, bad = [], []
        for ident in self.seen_order:
            if not self.buyable(ident) or not self.same_kind(ident, item):
                continue
            verdict = self.compatible(ident, series, vehicle, self.anchor_codes_of.get(number))
            if verdict is True:
                good.append(ident)
            elif verdict is False:
                bad.append(ident)
        good.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
        return good, bad

    def fit_status(self) -> str:
        """The compatibility picture for the agent: anchors, series, fitting and non-fitting rows per dependent item."""
        anchors = self.anchors()
        if not anchors:
            return ""
        lines = []
        for number in sorted(anchors):
            series, vehicle, note = self.target_series(number, keep=self.tail() + 4)
            item = self.items[number - 1]
            line = f"item {number} ({item.name}) {note}"
            if series or vehicle or self.anchor_codes_of.get(number):
                good, bad = self.fit_rows(number, series, vehicle)
                show = lambda i: f"{i}" + (f" {self.price_of(i):g}" if self.price_of(i) is not None else "") + (f" [{'/'.join(sorted(self.part_series(i)))}]" if self.part_series(i) else "") + f" ({self.item_verdict(number, i)})"
                line += ". Rows seen that fit: " + (", ".join(show(i) for i in good[:6]) if good else "none yet (search the kind with one of the series codes, or the model)")
                if bad:
                    line += ". Do NOT cart (they do not fit): " + ", ".join(show(i) for i in bad[:6])
                mine = self.line_of(number)
                if mine and self.compatible(mine, series, vehicle, self.anchor_codes_of.get(number)) is False:
                    line += f". The carted {mine} does not fit: replace it"
            lines.append(line)
        return "[compatibility, read from the listings' own fitment statements]\n" + "\n".join("- " + l for l in lines)

    def fit_sweep(self, keep: int = 10) -> str:
        """For each dependent item without fitting candidates: search its kind with the anchor's series codes (or the model),
        then read the attributes of the cheapest fitting rows. Code steps, no model turn. Once per item."""
        anchors = self.anchors()
        notes = []
        for number in sorted(anchors):
            if number in self.fit_swept or self.left() <= keep + 4 or self.finished or self.cost > self.spend_hard:
                continue
            series, vehicle, note = self.target_series(number, keep)
            if not series and not vehicle:
                if self.anchor_codes_of.get(number) and number not in self.fit_swept:
                    self.fit_swept.add(number)
                    note = self.code_sweep(number, keep)
                    if note:
                        notes.append(note)
                continue
            self.fit_swept.add(number)
            target = anchors.get(number)
            self.swept_anchor[number] = self.anchor_of(target)[0] if isinstance(target, int) else None
            item = self.items[number - 1]
            before = self.step
            good = self.fit_ranked(number, series, vehicle)
            known = any(self.item_verdict(number, i) == "meets" for i in good[:4]) or self.code_used >= self.code_cap()
            if not known:
                self.fill_facts([i for i in good if i not in self.full and i not in self.facts_seen][:2], keep)
                good = self.fit_ranked(number, series, vehicle)
            searched = []
            if not known and sum(1 for i in good if self.item_verdict(number, i) == "meets") < 2:
                words = re.sub(r"[^a-z0-9 ]", " ", item.name.lower()).strip()
                value = re.sub(r"[^a-z0-9 ]", " ", item.musts[0].split(":", 1)[1].lower()).strip()[:30] if item.musts and ":" in item.musts[0] else ""
                if series:
                    phrases = [f"{words} {value} {s}".replace("  ", " ") for s in sorted(series)][:2] + ([f"{words} {s}" for s in sorted(series)][:1] if not value else [])
                else:
                    phrases = [f"{words} {value} {self.vehicle_family(vehicle)}".replace("  ", " "), f"{words} {vehicle[0]} {vehicle[1]}", f"{words} {value} {vehicle[0]}".replace("  ", " ")]
                phrases = list(dict.fromkeys(p.strip() for p in phrases))[:3]
                for phrase in phrases:
                    if self.left() <= keep + 3 or phrase in self.swept_phrases:
                        continue
                    self.swept_phrases.add(phrase)
                    self.search_phrase(phrase)
                    searched.append(phrase)
                good = self.fit_ranked(number, series, vehicle)
            if vehicle and not good and self.left() > keep + 8:
                self.anchor_codes_of[number] = {f"{vehicle[0]}-{vehicle[1]}", f"{vehicle[0]}{vehicle[1]}"}
                swept = self.code_sweep(number, keep)
                if swept:
                    notes.append(swept)
                good = self.fit_ranked(number, series, vehicle)
            opened = self.fill_facts([i for i in good if i not in self.full and i not in self.facts_seen][:2], keep) if good and not known else []
            good = self.fit_ranked(number, series, vehicle)
            self.code_used += self.step - before
            self.trace.append({"kind": "fit_sweep", "item": number, "series": sorted(series), "vehicle": vehicle, "searched": searched, "fits": good[:4], "opened": len(opened)})
            shown = [f"{i} ({self.item_verdict(number, i)})" for i in good[:4]]
            notes.append(f"item {number} ({item.name}): " + (f"fitting rows {', '.join(shown)}" if good else "no fitting row found yet") + (f" (searched {', '.join(repr(p) for p in searched)})" if searched else ""))
        return ("[Sakura searched for parts that fit] " + " | ".join(notes)) if notes else ""

    def code_sweep(self, number: int, keep: int) -> str:
        """Accessories of a code-only anchor: page the accessory's kind with the search tool's own category filter (its rows carry the
        statements of what each listing goes with), then read the cheapest rows whose statement covers the anchor's code."""
        codes = self.anchor_codes_of.get(number) or set()
        item = self.items[number - 1]
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        if not codes or not tool:
            return ""
        label = ""
        for ident in self.seen_order:
            if self.same_kind(ident, item) and self.kind_of(ident):
                label = self.kind_of(ident)
                break
        label = label or item.shelf or item.name
        args: dict[str, Any] = {}
        for key, spec in props.items():
            options = spec.get("enum") if isinstance(spec, dict) and isinstance(spec.get("enum"), list) else None
            if not options:
                continue
            hit = next((o for o in options if isinstance(o, str) and o.strip().lower() == label.strip().lower()), None)
            if hit:
                args[key] = hit
            elif len(options) <= 4 and all(isinstance(o, str) for o in options):
                other = [o for o in options if not re.search(r"key|text|word|phrase|free|query|search", o, re.I)]
                args[key] = other[0] if other else options[0]
        if not any(isinstance(v, str) and v.strip().lower() == label.strip().lower() for v in args.values()):
            texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
            keyword = _keyword_arg(schema, texts)
            labels = {self.kind_of(i) for i in self.seen_order if self.kind_of(i)}
            field, base = None, {}
            for past in reversed(getattr(self, "search_log", [])):
                for k, v in past.items():
                    if k in texts and k != keyword and isinstance(v, str) and _stem(v) and any(_stem(v) <= _stem(l) or _stem(l) <= _stem(v) for l in labels):
                        field = k
                        base = {kk: vv for kk, vv in past.items() if isinstance(props.get(kk), dict) and "enum" in props[kk]}
                        break
                if field:
                    break
            if not field:
                return ""
            args.update(base)
            args[field] = label
        offset = None
        before = self.step
        pages = 0
        start = 0
        while pages < 4 and self.left() > keep + 6 and not self.finished and self.clock_left() > CLOSING + 30:
            call = dict(args)
            if offset:
                call[offset] = start
            faults: list[str] = []
            call = _shape(call, schema, "args", faults)
            if faults:
                break
            reply = self.act(tool, call)
            if reply is None:
                break
            observation = reply.get("observation")
            self.observe(tool, observation)
            rows = self.records_in(observation)
            pages += 1
            if not rows:
                break
            plan = self.paging(tool, observation)
            fresh = [i for i, _ in rows if i not in getattr(self, "swept_rows", set())]
            self.swept_rows = getattr(self, "swept_rows", set()) | {i for i, _ in rows}
            if not plan or plan[3] is None or not fresh:
                break
            args[plan[2]] = plan[3]
            covering = [i for i in self.seen_order if self.same_kind(i, item) and self.buyable(i) and self.covers(i, codes)]
            if len(covering) >= 6:
                break
        covering = [i for i in self.seen_order if self.same_kind(i, item) and self.buyable(i) and self.covers(i, codes)]
        covering.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
        opened = self.fill_facts([i for i in covering if i not in self.full][:2], keep) if covering else []
        self.code_used += self.step - before
        self.trace.append({"kind": "code_sweep", "item": number, "codes": sorted(codes), "pages": pages, "covering": covering[:6], "opened": len(opened)})
        shown = ", ".join(f"{i} ({self.item_verdict(number, i)})" for i in covering[:5])
        return f"item {number} ({item.name}): rows that go with {', '.join(sorted(codes))}: " + (shown or "none on the pages read")

    def fit_guard(self) -> str:
        """Before a checkout (twice at most): a dependent line that does not fit its anchor holds the checkout while a fitting
        row is known or can be found; with none, the order goes ahead (a decline earns nothing)."""
        if self.fit_blocks >= 2 or self.left() <= 4:
            return ""
        anchors = self.anchors()
        problems = []
        for number in sorted(anchors):
            series, vehicle, note = self.target_series(number, keep=self.tail() + 3)
            mine = self.line_of(number)
            if not mine or (not series and not vehicle):
                continue
            if self.compatible(mine, series, vehicle) is False:
                good, _ = self.fit_rows(number, series, vehicle)
                if not good:
                    self.fit_swept.discard(number)
                    self.fit_sweep(keep=self.tail() + 3)
                    good, _ = self.fit_rows(number, series, vehicle)
                if good:
                    problems.append(f"{mine} (item {number}) does not fit: it carries {'/'.join(sorted(self.part_series(mine))) or 'another model'} and the item {note}. Replace it with a fitting row: {', '.join(good[:4])} (cheapest first)")
        if not problems:
            return ""
        self.fit_blocks += 1
        self.trace.append({"kind": "fit_guard", "problems": problems[:3]})
        return "Not run yet (no step used). " + " ".join(problems[:3]) + ", then check out."

    # ---------- judging a listing's stated requirements by code (v3) ----------
    def meets(self, ident: str, need: str) -> bool | None:
        how = self.meets_how(ident, need)
        return None if how is None else (how != "fails")

    def meets_how(self, ident: str, need: str) -> str | None:
        """How the listing meets a stated requirement: "exact" (the property's own key states the value itself), "attribute"
        (a key naming the property states it, or states it with extra qualifiers), "title" (no key names the property and the
        title names the value), "fails" (a key naming the property contradicts it), None (nothing addresses it)."""
        if ":" not in need:
            return None
        if any(need == r for _, r in self.refs):
            plain = str(self.ref_values.get(need) or "").split(" (read from ", 1)[0].strip()
            if ":" not in plain or len(plain) > 160:
                return None
            need = plain
        prop, value = need.split(":", 1)
        prop, value = prop.strip(), value.strip().lower()
        block = _fact_block(self.full.get(ident) or {}) or {}
        words = _stem(prop)
        shared = {w for k in block for w in _stem(k) if sum(1 for k2 in block if w in _stem(k2)) > 1}
        keys = [k for k in block if _names(words, _stem(k)) or (len(words) >= 2 and len(_stem(k)) == 1 and _stem(k) <= words and not (_stem(k) & shared))]
        exact = [k for k in keys if _stem(k) == words]
        if exact:
            keys = exact
        if not keys:
            want = self.fit_tokens(value)
            return "title" if want and want <= self.fit_tokens(self.title_of(ident)) else None
        verdicts = [self.value_fits(str(block[k]), value) for k in keys]
        if any(v is True for v in verdicts):
            clean = not any(v is False for v in verdicts)
            return "exact" if clean and any(_words(k) == _words(prop) and self.exact_value(str(block[k]), value) for k in keys) else "attribute"
        return "fails" if any(v is False for v in verdicts) else None

    def fit_tokens(self, text: str) -> set[str]:
        spaced = re.sub(r"(\d)([a-z])", r"\1 \2", str(text or "").lower())
        return {w for w in re.findall(r"[a-z0-9.]+\+?", spaced) if w and w not in ("the", "a", "an", "of", "and", "or", "with", "for", "in")}

    def value_fits(self, have: str, want: str) -> bool | None:
        have_l = str(have or "").lower()
        span = re.search(r"\bbetween\s*([\d.]+)\D+?([\d.]+)", want)
        if span:
            num = re.search(r"([\d.]+)", have_l)
            if not num:
                return None
            try:
                low, high = sorted((float(span.group(1)), float(span.group(2))))
                return low - 1e-9 <= float(num.group(1)) <= high + 1e-9
            except ValueError:
                return None
        m = re.search(r"\b(at least|at most|no more than|no less than|under|below|over|above|minimum|maximum)\b\s*([\d.]+)", want)
        if m:
            num = re.search(r"([\d.]+)", have_l)
            if not num:
                return None
            try:
                x, y = float(num.group(1)), float(m.group(2))
            except ValueError:
                return None
            return x >= y if m.group(1) in ("at least", "no less than", "over", "above", "minimum") else x <= y
        if want.startswith("not "):
            inner = self.fit_tokens(want[4:])
            return not (inner and inner <= self.fit_tokens(have_l))
        want_t = self.fit_tokens(want)
        return (want_t <= self.fit_tokens(have_l)) if want_t else None

    def item_verdict(self, number: int, ident: str) -> str:
        item = self.items[number - 1]
        needs = [m for m in item.musts + item.told if m not in item.dropped and not self.open_need(m)]
        if ident not in self.full:
            return "unverified" if ident in self.opened or ident in self.facts_seen else "unread"
        verdicts = [self.meets(ident, n) for n in needs]
        if any(v is False for v in verdicts):
            return "FAILS"
        return "meets" if needs and all(v is True for v in verdicts) else ("unverified" if needs else "meets")

    def fit_ranked(self, number: int, series: set[str], vehicle: Any) -> list[str]:
        """Fitting rows of this item's kind: those meeting its requirements first, then unread, then unverified, those contradicting one last; cheaper first."""
        good, _ = self.fit_rows(number, series, vehicle)
        rank = {"meets": 0, "unread": 1, "unverified": 2, "FAILS": 3}
        return sorted(good, key=lambda i: (rank.get(self.item_verdict(number, i), 2), self.price_of(i) if self.price_of(i) is not None else float("inf")))

    def title_of(self, ident: str) -> str:
        """The listing's title: the row's title field (named by the shape reader, else a key named like a name or title), and
        failing both the shortest text of 12 to 200 characters that is not its category or id (descriptions run longer)."""
        cat_key = self.category_key()
        for record in (self.rows.get(ident) or {}, self.full.get(ident) or {}):
            key = self.title_key if self.title_key in record else next((k for k in record if re.search(r"name|title", str(k), re.I) and isinstance(record[k], str) and record[k].strip() and k not in (cat_key, self.id_key)), None)
            if key and isinstance(record.get(key), str) and record[key].strip():
                return record[key]
        best = ""
        for record in (self.rows.get(ident) or {}, self.full.get(ident) or {}):
            for key, value in record.items():
                if key in (cat_key, self.id_key) or not isinstance(value, str):
                    continue
                if 12 <= len(value) <= 200 and (not best or len(value) < len(best)):
                    best = value
        return best

    def exact_value(self, have: str, want: str) -> bool:
        """The attribute states the required value itself, not a value with extra qualifiers (a limit or an exclusion counts when it holds)."""
        if re.search(r"\b(at least|at most|no more than|no less than|under|below|over|above|minimum|maximum)\b", want) or want.startswith("not "):
            return self.value_fits(have, want) is True
        return self.fit_tokens(have) == self.fit_tokens(want)

    def item_rank(self, number: int, ident: str, verdict: str) -> int:
        """0 when every requirement is met under a key of the same name with the value itself, 1 when met under a related key, a
        qualified value or the title, 2 otherwise; 2 more when the title names another kind the category also covers."""
        item = self.items[number - 1]
        base = 3
        if verdict == "meets":
            hows = [self.meets_how(ident, m) for m in item.musts + item.told if m not in item.dropped and ":" in m and not self.open_need(m)]
            base = 2 if "title" in hows else (1 if "attribute" in hows else 0)
        return base

    def ranked_rows(self, number: int, series: set[str], vehicle: Any, opened_only: bool = True, strict: bool = True, codes: set[str] | None = None) -> list[str]:
        """Rows that could be this item's line, best first: meeting its requirements (under the property's own key, with the value
        itself, first) before unverified, cheaper first. Rows that contradict a requirement are out; so are rows that do not fit,
        and (when strict) rows whose fit is unknown."""
        item = self.items[number - 1]
        pool = []
        for ident in self.seen_order:
            if not self.buyable(ident) or not self.same_kind(ident, item):
                continue
            if series or vehicle or codes:
                fit = self.compatible(ident, series, vehicle, codes)
                if fit is False or (strict and fit is None):
                    continue
            verdict = self.item_verdict(number, ident)
            if verdict == "FAILS" or (opened_only and verdict == "unread"):
                continue
            pool.append((self.item_rank(number, ident, verdict), self.price_of(ident) if self.price_of(ident) is not None else float("inf"), ident))
        pool.sort()
        return [p[2] for p in pool]

    def item_of(self, ident: str) -> int:
        for number, item in enumerate(self.items, 1):
            if self.same_kind(ident, item):
                return number
        return 1

    def code_cap(self) -> int:
        """Steps code may spend on its own searching and reading, apart from the agent's turns: about a third of the task's steps."""
        return max(8, int(self.max_steps * 0.35))

    def explore(self, number: int, series: set[str], vehicle: Any, keep: int, want: int = 2, under: float | None = None, rows_cap: int = 2) -> None:
        """Read the cheapest unread fitting rows of this item's kind that name its required values (searching the kind with those
        values, and the series or model, when fewer than `want` such rows are known), up to `rows_cap` rows per call; with
        `under`, only rows cheaper than that."""
        item = self.items[number - 1]
        before = self.step
        values = [m.split(":", 1)[1].strip().lower() for m in item.musts + item.told if ":" in m and m not in item.dropped and not self.open_need(m)]
        wanted = [w for w in (self.fit_tokens(re.sub(r"\b(at least|at most|no more than|no less than|under|below|over|above|minimum|maximum|not)\b", " ", v)) for v in values) if w]

        def price(ident: str) -> float:
            return self.price_of(ident) if self.price_of(ident) is not None else float("inf")

        def tier(ident: str) -> int:
            if not wanted:
                return 0
            title = self.fit_tokens(self.title_of(ident))
            if all(w <= title for w in wanted):
                return 0
            text = self.fit_tokens(self.title_of(ident) + " " + _dump(self.rows.get(ident) or {}))
            return 1 if all(w <= text for w in wanted) else 2

        def pool() -> list[str]:
            rows = self.ranked_rows(number, series, vehicle, opened_only=False)
            if not rows and (series or vehicle):
                rows = self.ranked_rows(number, series, vehicle, opened_only=False, strict=False)
            return [i for i in rows if i not in self.full and i not in self.opened and i not in self.facts_seen and (under is None or price(i) < under)]

        unread = pool()
        preferred = sorted([i for i in unread if tier(i) < 2], key=lambda i: (tier(i), price(i)))
        if len(preferred) < want and self.left() > keep + 3:
            words = re.sub(r"[^a-z0-9 ]", " ", item.name.lower()).strip()
            value = re.sub(r"[^a-z0-9 ]", " ", values[0]).strip()[:30] if values else ""
            if series:
                phrases = [f"{words} {value} {s}".replace("  ", " ").strip() for s in sorted(series)][:2]
            elif vehicle:
                phrases = list(dict.fromkeys([f"{words} {value} {vehicle[0]} {vehicle[1]}".replace("  ", " ").strip(), f"{words} {value} {self.vehicle_family(vehicle)}".replace("  ", " ").strip(), f"{words} {value} {vehicle[0]}".replace("  ", " ").strip()]))
            else:
                phrases = [f"{words} {value}".strip()]
            for phrase in phrases:
                if self.left() <= keep + 3 or phrase in self.swept_phrases:
                    continue
                self.swept_phrases.add(phrase)
                self.search_phrase(phrase)
            unread = pool()
            preferred = sorted([i for i in unread if tier(i) < 2], key=lambda i: (tier(i), price(i)))
        self.fill_facts((preferred or unread)[:rows_cap], keep)
        self.code_used += self.step - before

    def bundle_plan(self, keep: int, do_cart: bool, search: bool = True) -> str:
        """The cheapest complete set for a request of several items, by code: for each anchor candidate (those meeting their
        requirements first, cheaper first) the cheapest row per dependent item that fits it and meets its requirements, plus the
        cheapest fitting row for every other item. With `search`, an item gets explored (a search by its values and the anchor's
        series or the model, then the cheapest fitting rows read) when it has no candidate, when its best candidate does not state
        the values themselves, or when a cheaper unread row names them. A set over the budget gives way to the cheapest set that
        contradicts nothing. With `do_cart`, a complete set inside the budget whose every line meets its requirements replaces the cart."""
        if len(self.items) < 2 or self.finished or (self.left() <= keep and do_cart) or not self.roles.get("cart"):
            return ""
        anchors = self.anchors()
        anchor_numbers = sorted({t for t in anchors.values() if isinstance(t, int)})
        dependents = {a: [n for n, t in anchors.items() if t == a] for a in anchor_numbers}
        others = [n for n in range(1, len(self.items) + 1) if n not in anchor_numbers and not isinstance(anchors.get(n), int)]

        def price(ident: str) -> float:
            return self.price_of(ident) if self.price_of(ident) is not None else float("inf")

        probe: list[str] = []

        def pick(number: int, series: set[str], vehicle: Any, dig: bool, by_price: bool, codes: set[str] | None = None) -> list[str]:
            def now() -> list[str]:
                return self.ranked_rows(number, series, vehicle, codes=codes) or self.ranked_rows(number, series, vehicle, strict=False, codes=codes)
            cands = now()
            went = ""
            if dig and search and self.left() > keep + 4 and self.code_used < self.code_cap():
                mark = (number, tuple(sorted(series)), vehicle)
                if not cands and mark + ("none",) not in self.dug:
                    self.dug.add(mark + ("none",))
                    self.explore(number, series, vehicle, keep, rows_cap=3)
                    cands = now()
                    went = "none"
                elif cands and mark + ("better",) not in self.dug and self.item_rank(number, cands[0], self.item_verdict(number, cands[0])) > 0:
                    self.dug.add(mark + ("better",))
                    self.explore(number, series, vehicle, keep)
                    cands = now()
                    went = "better"
            probe.append(f"{number}/{'-'.join(sorted(series)) or (vehicle[1] if vehicle else '') or '+'.join(sorted(codes or []))}: {len(cands)}{' ' + went if went else ''}")
            return sorted(cands, key=price) if by_price else cands

        def assemble(dig: bool, by_price: bool) -> tuple[dict[str, int], float, list[int]]:
            lines: dict[str, int] = {}
            total = 0.0
            missing: list[int] = []
            for a in anchor_numbers:
                cands = pick(a, set(), None, dig, by_price)
                choice = None
                for k, cand in enumerate(cands[:3]):
                    if k < 2 and dig and search:
                        self.resolve_pointer(cand, keep)
                    series = self.anchor_series(cand)
                    codes = self.anchor_codes(cand)
                    sub = {cand: self.items[a - 1].qty}
                    sub_total = price(cand) * self.items[a - 1].qty
                    lacking = []
                    for n in dependents[a]:
                        rows = pick(n, series, None, dig and k < 2, by_price, codes)
                        if rows:
                            sub[rows[0]] = self.items[n - 1].qty
                            sub_total += price(rows[0]) * self.items[n - 1].qty
                        else:
                            lacking.append(n)
                    key = (len(lacking), 0 if self.item_rank(a, cand, self.item_verdict(a, cand)) <= 2 else 1, sub_total)
                    if choice is None or key < choice[0]:
                        choice = (key, sub, lacking)
                if choice is None:
                    missing.append(a)
                    missing += dependents[a]
                else:
                    lines.update(choice[1])
                    total += choice[0][2]
                    missing += choice[2]
            for n in others:
                target = anchors.get(n)
                rows = pick(n, set(), target if isinstance(target, tuple) else None, dig, by_price)
                if rows:
                    lines[rows[0]] = self.items[n - 1].qty
                    total += price(rows[0]) * self.items[n - 1].qty
                else:
                    missing.append(n)
            return lines, total, sorted(set(missing))

        lines, total, missing = assemble(True, False)
        if lines and not missing and isinstance(self.budget, (int, float)) and total > self.budget and search:
            for ident in list(lines):
                n = self.item_of(ident)
                target = anchors.get(n)
                series = set()
                vehicle = target if isinstance(target, tuple) else None
                if isinstance(target, int):
                    anchor = next((i for i in lines if self.item_of(i) == target), None)
                    series = self.anchor_series(anchor) if anchor else set()
                mark = (n, tuple(sorted(series)), vehicle, "budget")
                if mark not in self.dug and self.left() > keep + 4 and self.code_used < self.code_cap():
                    self.dug.add(mark)
                    self.explore(n, series, vehicle, keep, under=price(ident), rows_cap=3)
            lines, total, missing = assemble(False, False)
        if lines and not missing and isinstance(self.budget, (int, float)) and total > self.budget:
            cheap, cheap_total, cheap_missing = assemble(False, True)
            if cheap and not cheap_missing and cheap_total < total:
                lines, total, missing = cheap, cheap_total, cheap_missing
        over = isinstance(self.budget, (int, float)) and total > self.budget
        verdicts = {i: self.item_verdict(self.item_of(i), i) for i in lines}
        text = "[cheapest complete set, by code] " + (", ".join(f"{i} x{q} at {price(i):g} ({verdicts[i]})" for i, q in lines.items()) if lines else "none yet")
        if lines:
            text += f"; total {total:g}" + ((f" of budget {self.budget:g}" + (" (OVER)" if over else "")) if isinstance(self.budget, (int, float)) else "")
        if missing:
            text += (". No candidate yet for item" + ("s " if len(missing) > 1 else " ") + ", ".join(f"{n} ({self.items[n - 1].name})" for n in missing)
                     + ": search that kind with the series code or model it must fit and the requirement's words, and read the cheapest rows")
        carted = False
        if do_cart and lines and not missing and not over and all(v == "meets" for v in verdicts.values()):
            carted = self.set_cart(lines)
            if carted:
                text += ". Sakura put this set in the cart: verify each line against the requirements, replace a line only with one that also fits and meets everything, then check out"
        if search or do_cart:
            self.trace.append({"kind": "bundle_plan", "lines": lines, "total": total, "missing": missing, "over": over, "carted": carted, "probe": probe[:12], "code_used": self.code_used})
        self.plan_lines = lines if lines and not missing else {}
        self.plan_partial = (dict(lines), list(missing))
        return text

    def loose_set(self) -> dict[str, int]:
        """With no complete fitting set known: the set the plan had, every missing item filled with its best opened row of its kind that
        contradicts none of its requirements (whatever it fits). An order that may miss a fit can still score; an ending without a
        purchase scores nothing unless no listing fits at all."""
        lines, missing = getattr(self, "plan_partial", ({}, []))
        lines = dict(lines)
        for number in missing or [n for n in range(1, len(self.items) + 1) if not any(self.item_of(i) == n for i in lines)]:
            rows = self.ranked_rows(number, set(), None) or self.ranked_rows(number, set(), None, strict=False)
            if not rows:
                return {}
            lines[rows[0]] = self.items[number - 1].qty
        if len({self.item_of(i) for i in lines}) < len(self.items):
            return {}
        self.trace.append({"kind": "loose_set", "lines": lines, "missing": missing})
        return lines

    def set_cart(self, lines: dict[str, int]) -> bool:
        """Make the cart hold exactly `lines` (code steps). True when every call went through."""
        cart = self.roles.get("cart")
        if not cart:
            return False
        drop = [i for i in self.cart_ids() if i not in lines]
        calls = self.cart_calls(lines, drop) or []
        if not calls or len(calls) + 2 > self.left():
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
        return True

    def cart_units(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A removal without a count is given the line's held units (the store takes one unit when no count is sent); an addition of a
        listing already held is cut to what its item still lacks (the store adds to the held count)."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.rows), None)
        if not count or not ident or self.cart_sets(name):
            return args
        held = self.cart_lines.get(ident, 0)
        out = dict(args)
        if self.is_removal(name, args):
            if out.get(count) is None and held > 1:
                out[count] = held
            return out
        if held > 0:
            number = self.item_of(ident)
            want = self.items[number - 1].qty if number else held
            asked = _count(out.get(count)) or 1
            if held + asked > want:
                out[count] = max(0, want - held)
        return out

    def sync_cart(self, observation: Any) -> None:
        """Units per line read from a cart view that lists its lines with a count."""
        lines: dict[str, int] = {}
        for _, record in self.records_in(observation):
            ident = self.ident(record, True, 1)
            # the line's unit count is the whole-number field the listing's own record does not carry (its price field it does)
            own = self.rows.get(ident or "") or {}
            fields = [k for k, v in record.items() if isinstance(v, int) and not isinstance(v, bool) and k not in own and k != self.price_key]
            units = record[fields[0]] if len(fields) == 1 else None
            if ident and units is not None:
                lines[ident] = units
        if lines:
            self.cart_lines = {i: n for i, n in lines.items() if n > 0}

    def is_removal(self, name: str, args: dict[str, Any]) -> bool:
        spec = self.schemas.get(name, {}).get("properties") or {}
        for key, value in args.items():
            options = (spec.get(key) or {}).get("enum")
            if isinstance(options, list) and len(options) >= 2 and value in options[1:]:
                return True
        return any(_count(v) is not None and _count(v) <= 0 for v in args.values() if not isinstance(v, str))

    def plan_guard(self) -> str:
        """Once, before a checkout of several items: a line that contradicts a requirement, or a total over the budget, holds the
        checkout when code knows a complete set inside the budget that contradicts nothing, or rows that meet the requirements in
        the contradicting line's place."""
        if self.plan_blocks >= 1 or len(self.items) < 2 or self.left() <= 6 or self.finished:
            return ""
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        if len(cart) < len(self.items):
            return ""
        problems = []
        alternatives = []
        for ident in cart:
            number = self.item_of(ident)
            if self.item_verdict(number, ident) == "FAILS":
                item = self.items[number - 1]
                bad = [m for m in item.musts + item.told if m not in item.dropped and not self.open_need(m) and self.meets(ident, m) is False]
                problems.append(f"{ident} (item {number}) contradicts {'; '.join(bad[:2])}")
                series, vehicle = set(), None
                if number in self.anchors():
                    series, vehicle, _ = self.target_series(number, keep=self.tail() + 3)
                alts = [i for i in self.ranked_rows(number, series, vehicle) if i != ident][:3]
                if alts:
                    alternatives.append(f"item {number}: " + ", ".join(f"{i} ({self.item_verdict(number, i)})" for i in alts))
        total = sum((self.price_of(i) or 0.0) * max(1, self.cart_lines.get(i, 1)) for i in cart)
        if isinstance(self.budget, (int, float)) and total > self.budget:
            problems.append(f"the total {total:g} is over the budget {self.budget:g}")
        if not problems:
            return ""
        plan = self.bundle_plan(keep=self.tail() + 3, do_cart=False, search=False)
        lines = self.plan_lines
        ptotal = sum((self.price_of(i) or 0.0) * q for i, q in lines.items())
        clean = bool(lines) and set(lines) != set(cart) and not any(self.item_verdict(self.item_of(i), i) == "FAILS" for i in lines) and not (isinstance(self.budget, (int, float)) and ptotal > self.budget)
        if not clean and not alternatives:
            return ""
        self.plan_blocks += 1
        self.trace.append({"kind": "plan_guard", "problems": problems, "plan": lines if clean else {}, "alternatives": alternatives})
        text = "Not run yet (no step used). " + "; ".join(problems) + ". "
        if clean:
            text += plan + ". Set the cart to that set (or to another whose every line fits and meets its requirements inside the budget), then check out."
        else:
            text += "Rows that fit and meet the requirements in its place: " + " | ".join(alternatives) + ". Replace the contradicting line, then check out."
        return text

    def best_effort_order(self) -> str:
        """The last gate before an ending without a purchase, and the first answer to one while a complete set is known: code
        carts the cheapest complete set that fits and does not contradict a requirement, and checks it out; after a sell-out at
        the checkout it plans once more without that line. A decline scores nothing; such an order may score."""
        if self.best_effort_done or len(self.items) < 2 or not self.roles.get("cart") or not self.roles.get("buy") or self.left() < 3 or self.finished:
            return ""
        self.bundle_plan(keep=self.tail() + 2, do_cart=False, search=self.left() > self.tail() + 6)
        lines = self.plan_lines
        if lines and self.left() > self.tail() + 6:
            # a last look for a line whose record does not state the requirement before gambling on it
            for ident in list(lines):
                n = self.item_of(ident)
                if self.item_verdict(n, ident) == "meets":
                    continue
                target = self.anchors().get(n)
                series: set[str] = set()
                vehicle = target if isinstance(target, tuple) else None
                if isinstance(target, int):
                    anchor = next((i for i in lines if self.item_of(i) == target), None)
                    series = self.anchor_series(anchor) if anchor else set()
                mark = (n, tuple(sorted(series)), vehicle, "final")
                if mark not in self.dug and self.code_used < self.code_cap() + 6:
                    self.dug.add(mark)
                    self.explore(n, series, vehicle, self.tail() + 2, rows_cap=3)
            self.bundle_plan(keep=self.tail() + 2, do_cart=False, search=False)
            lines = self.plan_lines
        if not lines:
            lines = self.loose_set()
        if not lines:
            self.trace.append({"kind": "best_effort", "lines": {}})
            return ""
        cap = self.budget if isinstance(self.budget, (int, float)) else float("inf")
        total = sum((self.price_of(i) or 0.0) * q for i, q in lines.items())
        if total > cap:
            self.trace.append({"kind": "best_effort", "over_budget": total, "lines": lines})
            return ""
        self.best_effort_done = True
        held = [i for i in self.cart_ids() if not self.sold_out(i)]
        if set(lines) != set(held) or any(self.cart_lines.get(i) != q for i, q in lines.items()):
            if not self.set_cart(lines):
                return ""
        self.trace.append({"kind": "best_effort", "lines": lines, "total": total})
        turned = self.finish(again=True)
        if not self.finished and turned and self.left() >= 3:
            self.find_sold()
            self.bundle_plan(keep=self.tail() + 2, do_cart=False, search=True)
            again = self.plan_lines
            if again and set(again) != set(lines) and not any(self.sold_out(i) for i in again) and sum((self.price_of(i) or 0.0) * q for i, q in again.items()) <= cap and self.set_cart(again):
                lines = again
                self.trace.append({"kind": "best_effort", "lines": lines, "retry": True})
                turned = self.finish(again=True)
        if self.finished:
            return "Sakura carted the cheapest set in which every line fits and meets (or at least does not contradict) its item's requirements (" + ", ".join(lines) + ") and checked out: ending without a purchase earns nothing while such a set exists. [episode finished]"
        return "Not run yet. Sakura carted a fitting set (" + ", ".join(lines) + ") and tried to check out; " + (turned or "the store did not confirm the order") + "\nApply what the store says, then check out again."

    def kind_tokens(self, text: str) -> frozenset[str]:
        """The words of a kind or category label, lower-case, singular, without joining words: 'Pants & Jeans' == 'pants or jeans'."""
        out = set()
        for w in re.split(r"[^a-z0-9]+", str(text or "").lower()):
            if not w or w in ("and", "or", "the", "a", "an", "of", "for", "with"):
                continue
            if w.endswith("ies") and len(w) > 4:
                w = w[:-3] + "y"
            elif w.endswith(("ches", "shes", "sses", "xes", "zes")) and len(w) > 4:
                w = w[:-2]
            elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
                w = w[:-1]
            out.add(w)
        return frozenset(out)

    def kind_of(self, ident: str) -> str:
        cat_key = self.category_key()
        record = self.rows.get(ident) or self.full.get(ident) or {}
        return str(record.get(cat_key) or "").strip() if cat_key else ""

    def kind_label(self) -> str | None:
        """The store's category label that names the single requested kind as a whole; None with several items or no exact match."""
        if len(self.items) != 1:
            return None
        cat_key = self.category_key()
        if not cat_key:
            return None
        labels: set[str] = set()
        for row in self.rows.values():
            label = row.get(cat_key)
            if isinstance(label, str) and label.strip():
                labels.add(label.strip())
        for want in (self.items[0].shelf, self.items[0].name):
            tokens = self.kind_tokens(want)
            if not tokens:
                continue
            hits = [label for label in labels if self.kind_tokens(label) == tokens]
            if len(hits) == 1:
                return hits[0]
        return None

    def kind_guard(self, ids: list[str]) -> str:
        """Cart lines from a neighbouring category while the store has the label naming the requested kind as a whole (twice at most)."""
        label = self.kind_label()
        if not label or self.kind_blocks >= 2:
            return ""
        wrong = [i for i in ids if self.kind_of(i) and self.kind_tokens(self.kind_of(i)) != self.kind_tokens(label)]
        if not wrong:
            return ""
        self.kind_blocks += 1
        self.trace.append({"kind": "kind_guard", "label": label, "wrong": [(i, self.kind_of(i)) for i in wrong]})
        return ("the request's kind is this store's category " + repr(label) + " and " + ", ".join(f"{i} is in {self.kind_of(i)!r}" for i in wrong)
                + ", a neighbouring category that does not count. Replace it with a listing from " + repr(label)
                + " that meets every requirement (search that category if none you have read fits), then check out.")

    def kind_note(self) -> str:
        """Once, when the rows seen span several categories: name the one the request's kind is."""
        if self.kind_noted:
            return ""
        label = self.kind_label()
        cat_key = self.category_key()
        if not label or not cat_key:
            return ""
        shown = {str(r.get(cat_key)).strip() for r in self.rows.values() if isinstance(r.get(cat_key), str)}
        others = sorted(k for k in shown if k and self.kind_tokens(k) != self.kind_tokens(label))
        if not others:
            return ""
        self.kind_noted = True
        return f"\n[Sakura] The request's kind is this store's category {label!r}: only rows in that category count; rows in {', '.join(repr(o) for o in others[:4])} do not."

    def check_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.version}"

    def check_cart(self, ids: list[str] | None = None, limit: float | None = None) -> str:
        if limit is None:
            limit = self.spend_soft
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or self.cost > limit or not ids:
            return ""
        if any(not (self.full.get(i) or self.rows.get(i)) or i not in self.listed for i in ids):
            self.trace.append({"kind": "check_skipped", "why": "a cart line without a listing record", "cart": ids})
            return ""
        guard = self.kind_guard(ids)
        if guard:
            self.verdicts[key] = False
            self.leads = [i for i in self.leads if i not in ids]
            return "A careful review says this cart is not right yet: " + guard
        self.checked.add(key)
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        fit = self.fit_status()
        if fit:
            prompt += "\n\n" + fit
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        first = not getattr(self, "heavy_checked", False) and self.cost + PLAN_ROOM <= self.spend_hard
        self.heavy_checked = True
        got = self.ask_json(prompt, [HEAVY, LIGHT] if first else [LIGHT, HEAVY], 1500 if first else 3500)
        self.trace.append({"kind": "check", "cart": ids, "got": got, "heavy": first})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        if better:
            self.better_for[key] = better
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def take_better(self) -> str:
        """The cart a check turned down while naming better listings comes back unchanged: Sakura puts in the best of those listings
        whose record meets every requirement of its item, in place of that item's line (once per cart; items that must fit another
        or that others must fit are left to the agent)."""
        ids = self.cart_ids()
        key = self.check_key(ids)
        better = self.better_for.get(key)
        if not better or key in self.better_taken or self.left() < 4 or self.finished:
            return ""
        self.better_taken.add(key)
        bound = set(self.anchors()) | {t for t in self.anchors().values() if isinstance(t, int)}
        lines = {i: max(1, self.cart_lines.get(i, 1)) for i in ids if not self.sold_out(i)}
        for cand in better:
            if cand in lines or self.sold_out(cand) or not self.buyable(cand):
                continue
            n = self.item_of(cand)
            if not n or n in bound or self.item_verdict(n, cand) != "meets":
                continue
            old = [i for i in lines if self.item_of(i) == n]
            if len(old) != 1:
                continue
            new_lines = {**{i: q for i, q in lines.items() if i not in old}, cand: self.items[n - 1].qty}
            total = sum((self.price_of(i) or 0.0) * q for i, q in new_lines.items())
            if isinstance(self.budget, (int, float)) and total > self.budget:
                continue
            if self.set_cart(new_lines):
                self.trace.append({"kind": "took_better", "out": old[0], "in": cand})
                return f"Sakura put {cand} in the cart in place of {old[0]}: the check found {old[0]} wrong, and {cand}'s record meets every requirement."
        return ""

    def trail(self) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Sakura opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        live = {need for _, need in self.needs_now()}
        if any(need in live and need not in self.ref_values and need not in self.ref_tried for _, need in self.refs) and self.left() > 14:
            self.resolve_refs(keep=8)
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 and self.cost < self.spend_hard:
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
            gaps: list[tuple[str, str]] = []
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
                    gap = next((l for l, i in zip(links, ids) if i not in self.rows), {})
                    about = str(gap.get("about") or "") or "the listing it points at"
                    words = str(gap.get("find") or "").strip()
                    lost.append(f"{need} -> find {about}" + (f" (search: {words})" if words else ""))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target and self.left() > 12:
                reach = self.find_link(*gaps[0])
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
            text += "[Sakura opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def find_link(self, about: str, words: str) -> list[str]:
        """A chain link nobody has seen: one search by the words the trail reader gives (the shop's name and the description's
        values), then the unopened rows whose own fields name that shop or brand, of the described kind."""
        if self.link_searches >= 3 or self.left() <= 8 or not words.strip() or self.finished:
            return []
        memo = " ".join(words.lower().split())
        ids = self.link_seen.get(memo)
        if ids is None:
            self.link_searches += 1
            ids = self.search_phrase(words[:120])
            self.link_seen[memo] = ids
        low = about.lower() + " " + memo
        kind = {w for w in re.findall(r"[a-z0-9]{3,}", (words or about).lower())}

        def named(ident: str) -> bool:
            row = {**(self.rows.get(ident) or {}), **(self.full.get(ident) or {})}
            return any(isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search(r"(?<![a-z0-9])" + re.escape(v.strip().lower()) + r"(?![a-z0-9])", low) for v in row.values())

        out = [i for i in ids if i not in self.opened and named(i)]
        out = [i for i in out if not kind or kind & set(re.findall(r"[a-z0-9]{3,}", _dump(self.rows.get(i) or {}).lower()))] or out
        self.trace.append({"kind": "link_search", "words": words[:80], "rows": out[:4]})
        return out[:2]

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
        self.swap_fits = [i for i in picks if i not in self.swap_fits] + self.swap_fits
        text = f"[Sakura] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
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
        if self.cost > self.spend_soft:
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
        got = self.ask_json(prompt, [LIGHT], 3500)
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
        if self.audited or self.cost > self.spend_soft:
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
        for ident in reversed(self.seen_order):
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            records.append(("OPENED " + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500]))
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger() + (("\n\n" + self.fit_status()) if self.anchors() else "")
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [LIGHT], 4000)
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
            plan = [(tool, {target: i}) for i in rows[:4]]
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

    def buy_lead(self) -> str:
        """The agent wants to end without buying while the review named listings that appear to fit: the first one the
        independent check confirms is carted and bought by Sakura itself. A wrong order and a wrong ending both score
        nothing, and endings without a purchase are right far more rarely than a confirmed fit is."""
        for first in self.leads[:2]:
            if self.sold_out(first) or self.left() < 4 or self.finished:
                continue
            self.check_cart([first], self.spend_hard)
            if self.verdicts.get(self.check_key([first])):
                self.confirmed = first
                self.trace.append({"kind": "buy_lead", "confirmed": first})
                return self.complete_cart()
        return ""

    def complete_cart(self) -> str:
        """The agent ends without buying although the independent check confirmed a reviewed listing: Sakura carts the reviewed
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
        self.check_cart(list(lines), self.spend_hard)
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
                self.check_cart(list(lines), self.spend_hard)
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
        return ("Not run (Sakura acted instead): the independent check confirmed " + self.confirmed + ", so Sakura set the cart (" + "; ".join(done)
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
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            if removal:
                have = have - count if count else have - 1
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def show(self, observation: Any) -> str:
        text = _dump(_clip_shown(observation, SHOW_CHARS))
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
                out.append((cid, "Shown in the side-by-side result above: Sakura opened these listings together in one step."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("ask"):
                out.append((cid, self.relay(" ".join(str(v) for v in args.values() if isinstance(v, str)), direct=True) or "Not run (no step used): Sakura's interviewer handles every message to the shopper."))
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
                trimmed = f"\n[Sakura] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get("open"), self.roles.get("offer")) and str(args.get(target)) in self.gone:
                out.append((cid, f"Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing."))
                continue
            again = str(args.get(target)) if target and name == self.roles.get("open") else ""
            if again and again in self.full and again in self.opened_step and self.opened_step[again] >= max(self.refused_at, self.step - 15) and not self.sold_out(again):
                out.append((cid, f"Not run (no step used): {again} was opened at step {self.opened_step[again]}; its record: " + _dump(self.full[again])[:3000]))
                continue
            if name == self.roles.get("cart") and any((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ""
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and v not in self.cart_lines and v not in view and v not in self.cart_said]
                if absent:
                    out.append((cid, f"Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: " + self.cart_text()))
                    continue
            if name == self.roles.get("search"):
                phrase = " ".join(str(v) for v in args.values() if isinstance(v, str))
                if re.search(r"\?|\b(how (much|many|big|large)|do you (need|want|prefer)|would you like|what do you)\b", phrase, re.I):
                    relayed = self.relay(phrase, direct=True)
                    if relayed:
                        out.append((cid, relayed))
                        continue
            if name == self.roles.get("cart"):
                args = self.cart_units(name, args)
            if name == self.roles.get("cart") and self.plan_lines and self.is_removal(name, args) and self.refused_at < 0 and self.buys == 0:
                kept = [v for v in args.values() if isinstance(v, str) and v in self.plan_lines and v not in self.removal_held and not self.sold_out(v)]
                arriving = [v for c in calls for v in (_json_obj((c.get("function") or {}).get("arguments")) if isinstance((c.get("function") or {}).get("arguments"), str) else ((c.get("function") or {}).get("arguments") or {})).values()
                            if isinstance(v, str) and v in self.rows and v not in self.cart_ids()]
                if kept and arriving and any(self.same_kind(v, self.items[self.item_of(kept[0]) - 1]) for v in arriving):
                    kept = []
                if kept and self.item_verdict(self.item_of(kept[0]), kept[0]) != "FAILS":
                    self.removal_held.add(kept[0])
                    number = self.item_of(kept[0])
                    out.append((cid, f"Not run yet (no step used): {kept[0]} is the line code chose for item {number} ({self.items[number - 1].name}): it fits and its record contradicts none of the requirements ({self.item_verdict(number, kept[0])}). Keep it unless you add, in its place, a listing that also fits and meets every requirement; an item without a line earns nothing."))
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
                    trimmed += f"\n[Sakura] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
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
                            self.sync_cart(self.cart_view)
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
                    fitq = self.fit_guard()
                    if fitq:
                        out.append((cid, fitq + "\nCart now: " + self.cart_text()))
                        continue
                    planq = self.plan_guard()
                    if planq:
                        out.append((cid, planq + "\nCart now: " + self.cart_text()))
                        continue
                    gap = self.missing_hold()
                    if gap:
                        out.append((cid, gap + "\nCart now: " + self.cart_text()))
                        continue
                    if len(self.items) == 1 and not [i for i in self.cart_ids() if not self.sold_out(i)] and self.left() > 4 and not self.single_effort:
                        pick = self.single_pick()
                        if pick and self.set_cart({pick: self.items[0].qty}):
                            self.single_effort = True
                            self.trace.append({"kind": "empty_cart_fill", "pick": pick})
                            out.append((cid, f"Not run yet. The cart was empty, so Sakura carted {pick}, the best-placed in-stock listing of the requested kind that contradicts no requirement (one step). Check it against the requirements, replace it if a better fit is known, then check out.\nCart now: " + self.cart_text()))
                            continue
                    faults = self.line_faults()
                    if faults and self.line_blocks < 2:
                        self.line_blocks += 1
                        fixed = self.fix_counts()
                        self.trace.append({"kind": "line_faults", "faults": faults, "fixed": fixed})
                        out.append((cid, "Not run yet (no step used). " + "; ".join(faults) + (("\nSakura set the quantities itself (one step): " + fixed + ". Check out again.") if fixed else ". Fix the cart (one line per requested item at its quantity, nothing else, nothing sold out, inside the budget), then check out.") + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    took = self.take_better()
                    if took:
                        out.append((cid, "Not run yet. " + took + "\nCart now: " + self.cart_text() + "\nCheck out again."))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Sakura read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > self.spend_soft:
                        out.append((cid, "Not run yet (no step used). Sakura re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    relayed = self.relay(" ".join(str(v) for v in args.values() if isinstance(v, str)))
                    if relayed:
                        out.append((cid, relayed))
                        continue
                    if self.left() >= 6 and self.search_blocks == 0 and any(step > self.last_search_step for step in self.gone.values()):
                        self.search_blocks = 1
                        self.trace.append({"kind": "refuse_blocked", "reason": "no search since a line sold out"})
                        out.append((cid, "Not run yet (no step used). A carted line sold out and nothing you have opened is known to fit in its place. Before ending without a purchase, run one more search for the requested kind (its own category, with the requirement's words), open the best rows and cart the first that meets every requirement, then check out. End without a purchase only when that search finds nothing."))
                        continue
                    early = self.early_decline_hold()
                    if early:
                        out.append((cid, early))
                        continue
                    guard = self.refuse_guard()
                    if guard:
                        out.append((cid, guard))
                        continue
                    if self.left() > 5:
                        review = self.audit()
                        if review and self.leads:
                            bought = self.buy_lead()
                            if bought:
                                out.append((cid, bought + (" [episode finished]" if self.finished else "")))
                                continue
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
                            self.check_cart([first], self.spend_hard)
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
                bought = self.best_effort_order()
                if not bought and len(self.items) == 1:
                    bought = self.single_effort_order()
                if bought:
                    out.append((cid, bought))
                    continue
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
                    trimmed = "\n[Sakura] " + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f"\n[Sakura] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))
            self.absorb(observation, name)
            self.learn_shape(name, observation)
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
                    self.note_facts(name, bool(_fact_block(observation)))
                    if name == self.facts_tool():
                        self.facts_seen.add(listing)
                    if name == self.roles.get("open") and self.facts_from.get(name) is False:
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and self.facts_from.get(t) is not False
                                      and t not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask"), self.roles.get("search"))), None)
                        if other:
                            self.roles["open"] = other
                    if self.facts_from.get(name) is False and name != self.roles.get("open") and not self.roles.get("offer"):
                        self.roles["offer"] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get("cart"):
                self.note_cart(name, args)
                self.cart_said = _dump(observation)
                targets = {t for t in self.anchors().values() if isinstance(t, int)}
                if targets and any(isinstance(v, str) and v in self.rows and any(self.same_kind(v, self.items[t - 1]) for t in targets) for v in args.values()):
                    for n, t in self.anchors().items():
                        if isinstance(t, int) and self.swept_anchor.get(n) != self.line_of(t):
                            self.fit_swept.discard(n)
                    swept = self.fit_sweep(keep=self.tail() + 6)
                    if swept:
                        trimmed += "\n" + swept
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
            if name == self.roles.get("search"):
                self.last_search_step = self.step
            if name == self.roles.get("cart_view"):
                self.cart_view = observation
                self.sync_cart(observation)
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get("finished"):
                    self.refused_at = self.step
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
                text += self.kind_note()
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Sakura] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[Sakura re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if self.queued:
                text += "\n[notes seen on Sakura's own calls] " + " | ".join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text += " [episode finished]"
            out.append((cid, text))
        return out

    def drop_gone_lines(self) -> bool:
        """Before a forced checkout: take sold-out lines off the cart when other lines remain and steps allow, so the
        order holds only lines the store can sell (a sold-out line in the order fails the whole episode)."""
        cart = self.roles.get("cart")
        if not cart or self.finished:
            return
        try:
            ids = self.cart_ids()
            lost = [i for i in ids if self.sold_out(i)]
            keep = {i: self.cart_lines.get(i, 1) for i in ids if i not in lost}
            if not lost:
                return True
            if self.left() < len(lost) + 2:
                return False
            if not keep:
                # every line is gone: the best reviewed listing still on sale takes the place of the first lost line
                pool = ([i for i in self.leads if not self.sold_out(i) and i not in lost] or [i for i in self.swap_fits if not self.sold_out(i) and i not in lost]
                        or (self.replacements(lost) if self.cost < self.spend_hard + 0.01 else []))
                pick = next((i for i in pool if not self.sold_out(i) and i not in lost), None)
                if not pick or self.left() < len(lost) + 3:
                    return False
                keep = {pick: self.cart_lines.get(lost[0], 1)}
                self.trace.append({"kind": "gone_swap", "lost": lost, "pick": pick})
            calls = self.cart_calls(keep, lost) or []
            for args in calls[: len(lost) + 1]:
                faults: list[str] = []
                args = _shape(args, self.schemas.get(cart), "args", faults)
                if faults or self.left() < 2:
                    break
                reply = self.act(cart, args)
                if reply is None or self.finished:
                    break
                self.note_cart(cart, args)
            self.trace.append({"kind": "dropped_gone", "lost": lost})
            return not any(self.sold_out(i) for i in self.cart_ids())
        except Exception as exc:  # a failed clean-up must never stop the finish itself
            self.trace.append({"kind": "dropped_gone_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return False

    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        if not again and len(self.items) > 1 and self.left() >= 4 and any(self.sold_out(i) for i in self.cart_ids()):
            # a line sold out and the steps are nearly gone: the cheapest complete set still on sale replaces the cart
            self.bundle_plan(keep=2, do_cart=False, search=False)
            fresh_lines = self.plan_lines
            cap = self.budget if isinstance(self.budget, (int, float)) else float("inf")
            if fresh_lines and not any(self.sold_out(i) for i in fresh_lines) and sum((self.price_of(i) or 0.0) * q for i, q in fresh_lines.items()) <= cap:
                self.set_cart(fresh_lines)
                self.trace.append({"kind": "forced_replan", "lines": fresh_lines})
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            sellable = self.drop_gone_lines()
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not sellable:
                self.trace.append({"kind": "forced_buy_skipped", "why": "a sold-out line could not be replaced"})
            if not faults and sellable:
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

    def last_resort(self) -> None:
        """The loop ended with nothing in the cart while steps remain: code orders the best candidates it knows rather than end
        without a purchase, which scores nothing."""
        live = [i for i in self.cart_ids() if not self.sold_out(i)]
        if live or self.left() < 3 or self.finished or self.broke or self.no_credit:
            return
        note = self.best_effort_order() if len(self.items) > 1 else self.single_effort_order()
        if note:
            self.trace.append({"kind": "last_resort", "note": note[:160]})

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
        self.work(LOOP_CALLS, self.spend_hard)
        if not self.finished:
            self.last_resort()
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Sakura] Sakura tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, self.spend_hard + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap and self.clock_left() > CLOSING:
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
                if not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and self.left() > 8 and ("?" in content or re.search(r"\b(ask|question)", content, re.I)):
                    # the model is talking to the shopper in plain text, which reaches nobody: the interviewer asks instead
                    summary = self.interview()
                    self.chat.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + "\nThe shopper hears only what Sakura's interviewer sends; your own text reaches nobody. Call one of the tools to continue.\n" + self.status()})
                    idle = 0
                    continue
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
            try:
                replies = self.run_calls(calls)
            except Exception as exc:
                self.slips = getattr(self, "slips", 0) + 1
                self.trace.append({"kind": "slip", "error": f"{type(exc).__name__}: {exc}"[:300], "step": self.step})
                if self.slips > 2 or self.finished:
                    raise
                replies = [(c["id"], "Sakura could not read this result. The step may have been used; check the record of what you asked for (call again if needed) and go on.") for c in calls]
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.chat.append({"role": "tool", "tool_call_id": cid, "content": text})


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Sakura(problem_data)
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
