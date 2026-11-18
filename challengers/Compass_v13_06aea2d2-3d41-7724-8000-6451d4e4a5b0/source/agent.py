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
SPEND_SOFT = 0.040         # (y1) was 0.055: a race run's key holds $5 for its 90 tasks ($0.056 a task) and KYO's
SPEND_HARD = 0.055         # race runs spent $4.63-4.68 (93-94%); at the cap the tasks still running score 0 (was 0.078)
                           # (r12c) were 0.050 / 0.070: on the race 11 bed KYO and r12 averaged $0.061 a task ($5.48 for 90,
                           # 10 of 30 tasks past 0.07) and race 12's official KYO runs spent $5.03-5.10; the targets now aim
                           # the run's mean at about $0.047 (85% of the cap)
CLOCK_LOOP = 235.0         # (y1) seconds after which the working loop stops and the episode finishes (KYO had no clock:
                           # 6 race 10 episodes ended did_not_finish)
STRUCT_GATE = True         # (y2a) before a checkout the code counts the cart against the request's items: one line per
                           # item, each on a shelf of a requested item, at the item's quantity (race 10 official: 8 of KYO's
                           # request_not_met orders had a missing line, a wrong shelf or an extra line; the model check
                           # that could catch them is skipped past SPEND_SOFT)
CHEAPER_SCAN = True        # (y2a) under a cheapest rule the code lists each cart line's own shelf below its price with the
                           # search tool's filter mode and reads the cheapest rows (race 10: 7 not_the_best_option orders; the
                           # cheaper fit never showed in KYO's keyword searches, which ignore the price filter)
NUDGE_ORDER = True         # (y2a) once, when steps run short with an empty cart: cart the best verified picks and check out
                           # (race 10: first checkouts at step 42-43 of 46, one sent with an empty cart)
REASK_SAMPLES = False      # (y1) a question answered "one at a time" is not sent again with example values: race 10
                           # official, 73% of those re-asks drew the same one-at-a-time line; the message asks the next pick
REPLAN_EACH = True         # (r12) the interview asks in waves of REPLAN_WAVE questions; after a wave whose answers state a
                           # new requirement (or ask for one question at a time) the rest are planned again with what was
                           # asked and answered (research/race12/top2_study.md s3.3: CHIDORI re-plans after each answer;
                           # hidden-key study 11-02: re-planning after answers +0.5~1.2 a race, no loss). Waves, not single
                           # questions, keep the reads of the answers near KYO's count (cost); off past SPEND_SOFT
REPLAN_WAVE = 2
REJUDGE_ALL = True         # (r12) after every change of the requirements in force (version), the next tool reply carries
                           # the code's recheck of every opened listing against all stated word requirements: which state
                           # them all (cheapest first under a cheapest pick), which state another value; facts, no order
                           # (top2_study.md s3.2/s8 rank 1: CHIDORI/LV re-choose from all candidates on each new fact,
                           # all-revealed pass 48%/45% against KYO 23%); no model call, no step
JUDGED_FRESH = True        # (r11) judged() ignores a listing's own "does not fit" verdict from an older requirement
                           # version (research/race12/conversion_mechanisms.md: withdrawn needs kept listings out)
DECLINE_SEARCH = True      # (r11) before a decline the review found nothing for: a stated requirement no opened listing
                           # of its item states is searched once by its value and two rows of it read, then the review
                           # runs again (at most two searches an episode; wrong declines were 10 of 12 single-item
                           # all-revealed failures, research/review-20261101/24_hidden_multi_solutions.md)
AUDIT_ALWAYS = True        # (y2b) the review before a decline runs once whatever the spend, up to SPEND_HARD + 0.01 (KYO
                           # skipped it past SPEND_SOFT; race 9: 22 declines after every hidden requirement was revealed)
UNOPENED_HOLD = True       # (y2b) a decline is held once per item set while some requested item has buyable rows but no
                           # listing opened in full: Kyo reads its two cheapest such rows first (race 9: 11 of 21 wrong
                           # declines on tasks others solved never opened the winning listing)
LOOP_END_ORDER = True      # (y2b) when the loop stops on its call, spend or clock limit with a cart that misses items, Kyo
                           # sets the cart from checked picks and checks out instead of ending without a purchase (race 9:
                           # 17 unexplained declines from finish() with an empty cart, 16 with 27-66 steps left)
REFUSAL_DECLINE_HOLD = True  # (y2b) a decline right after a refused checkout or a removed cart line is held once when an
                           # opened, buyable, in-budget listing of that line's item passes the check (race 10 bed
                           # t-3de6661eab: y2a dropped its line after a budget change and declined with a cheaper fit opened)
REFUSAL_RECENT = 6         # (y2b) steps after a refusal or a removal within which a decline counts as "right after"
NOTE_HOLD = True           # (y2b) a checkout is held when a new note seen on Kyo's own pre-checkout reads changed the
                           # request: the actor re-plans with it (race 10 bed t-85a9d0fe8b: a dropped requirement arrived
                           # on the code's reread of the carted listing and the costlier pick was checked out unseen)
TRAIL_SCOPE = True         # (r5) a requirement whose wording names another requested item of the same order is judged on
                           # that item's cart line, not chased as a chain of pages (race 10 bed: 13 tasks, no listing found)
TRAIL_RELEASE = True       # (r5) a chain whose next page is not found in two tries stops holding the order: the checks judge
                           # the cart on the other requirements, and a decline over it alone is held once
SWAP_SURE = True           # (r5) a sold-out line's replacement search runs up to SPEND_HARD (was SPEND_SOFT), is retried once
                           # when the model call fails, and opened listings already judged fitting fill in and come first
BACKUP_OPEN = True         # (r5) before the first checkout, a cart line with no other opened candidate gets one more read
EMPTY_FINISH = True        # (r5) finish() sends no checkout for a cart that Kyo's record and the last cart reply show empty
EMPTY_BUY = True           # (r6) the actor's checkout of a cart that Kyo's record shows empty is not sent, whatever the steps
                           # left (the pre-checkout checks stop at 2 steps left; race 12: v12 3 and KYO 2 empty-cart
                           # checkouts, t-b32ba6a806 dropped its fitting line and checked out an empty cart twice)
NOTE_REBUY = True          # (r6) a checkout refused with a note that changes the request is not sent again for the same,
                           # unchanged cart: the actor re-plans under the change first (race 12: after a budget-change note
                           # v12 6 and KYO 5 episodes checked out the same cart again, t-1876d9860e)
STEP_CAP = 45              # (r6) the working loop sends no new model turn once 45 store steps are used and the episode goes
                           # to the usual finish (loop_end_order, then finish); 0 turns it off. Spend tracks steps and
                           # official passes fall with them (45-55 steps 14%, 55-65 6%, 65-75 2%): for the same saving a
                           # step cap cuts 3 passing bed tasks, a $0.055 cap 13 (research/race12/spend_cap_study.md;
                           # race 12: 162 of 470 runs exhausted the $5 key, v12 one of two, its last 23 tasks scored 0)
CAP_LEAKS = True           # (r6) spend past SPEND_HARD is bounded: the review, the loop-end and chain checks stop at
                           # SPEND_HARD (were +0.01), the short second round opens only when the loop did not stop on its
                           # spend or step cap and itself stops at SPEND_HARD, and past SPEND_CEIL no model call is sent
                           # but the request's own pre-order and pre-ending steps (gate)
SPEND_CEIL = SPEND_HARD + 0.012


def _late_limit() -> float:
    """(r6) The spend up to which the review and the loop-end and chain checks run: SPEND_HARD with CAP_LEAKS, else r5's
    SPEND_HARD + 0.01."""
    return SPEND_HARD if CAP_LEAKS else SPEND_HARD + 0.01


QUESTION_SPREAD = False    # (r6) within each item's slots of the question plan, properties the opened listings of that item
                           # split on are asked first and properties they all share last (the plan asks one more than the
                           # messages allow, so a shared one drops out); the plan's item order, its picks, the request's
                           # ask-first properties and the PLAN prompt stay (race 11+12: where DNA's and KYO's asked
                           # properties differed DNA +3.5 tasks, p 0.08; DNA's own-only properties single-valued 0.38 vs
                           # KYO's 0.52). Kept off: research/race12/question_spread_study.md found split properties reveal
                           # no more (position-adjusted OR 0.87, p 0.30), hidden properties split less (OR 0.55), and the
                           # reorder replayed on official question lists costs -0.77 to +0.04 tasks a race
RESELECT = True            # (r7) after a reveal or a sold-out line, pick again from every candidate seen, word for word:
                           # R1 a revealed word requirement no opened listing states while two or more state another value
                           # makes Kyo read the listed rows that state it, else search for it once; R2 a cart line whose
                           # attribute is not the requirement's words in a row is held once when an opened fit states them;
                           # R3 sold-out replacements are ranked by the words they state, then the pick rule. No model call
                           # (race 12 official: after such a reveal KYO passed 13/55 when it searched the value, 2/18 when
                           # not, LV 14/63 and 0/11; leaders converted 48%/45% of full reveals vs KYO 23%;
                           # research/race12/reselect_design.md)
RESELECT_OPENS = 2         # (r7) rows R1 reads a trigger
RESELECT_ROOM = 8          # (r7) steps that must stay free beyond tail() (and below STEP_CAP) for R1 to use the store
RESELECT_R1 = False        # (r9) R1 (reveal_research) alone off: race 10 bed paired on r6's cache, r7 -4 and r8 -1 net, both
                           # losses on tasks where R1 fired (t-8671b42781, t-a1d8d0a25a); R2 and R3 stay on (R3 fired 8 times,
                           # no loss). The bed cannot show R1's upside (research/race12/bed_fidelity_study.md), so it waits
                           # for official evidence
EXACT_ANY = True           # (r9) R2 also holds a cart line whose attribute states another value than a word requirement (or
                           # its words out of order) when no opened fit can be named: race 12 official, such lines pass 0 of
                           # 807; lines stating the value in a row pass 64% (research/race12/mechanism_natural_experiments.md:
                           # the one conversion mechanism high-confidence in both races 11 and 12)
STEP_CAP_PER_ITEM = True   # (r9d) the step cap grows with the order: max(STEP_CAP, 35 + 10 x items), never past the task's
                           # own limit; a fixed 45 (r6, set on race 12) would have cost v12 about 2.5 tasks a run in race 11,
                           # whose limits ran 25-143 and whose 4-5 item orders opened their last items after step 45
                           # (research/race12/counterfactual_race11.md)
NOTE_REBUY_FITS = True     # (r9d) NOTE_REBUY holds no checkout whose cart Kyo's record shows inside the (changed) budget:
                           # race 11, v12 re-checked out the same cart after a budget note and passed 23 of 28
REVEAL_WIDEN = True        # (r9e, from r10) before the question plan Kyo's own reads fill each item up to three opened
                           # listings (was at most two more): official counterfactual +0.3 tasks a run in race 12 and in
                           # race 11 (research/race12/counterfactual_cost_reveal.md, counterfactual_race11.md); r10's
                           # QUESTION_NAME_CLEAN is left out (race 11 risk on a shared word, race 12 gain 0)
STEP_CAP_EXTEND = True     # (r9f) the step cap is reached with a cart short of the order's items and steps to spare: it
                           # moves ten steps on, once (official race 12: final orders short of lines v12 0.31, KYO 0.20,
                           # leaders 0.06 of multi-item episodes; research/race12/mechanism_natural_experiments.md)
REF_LINK_SEARCH = True     # (r9f) a chain link nobody found is searched by Kyo once with the trail model's description of
                           # it (seller, kind, attributes) and the best matching unread rows are opened with the page view
                           # (race 12: 26 reference tasks scored ~0 for everyone; KYO stopped 35 of 58 chains)
REF_PAGE_OPEN = True       # (r9) a chain link counts as opened only once its page view was read when the store has a page
                           # view apart from the attributes view: the page carries the remarks that point further (race 12:
                           # KYO stopped 35 of 58 chains, 24 with the link seen but its page never read, 13 of them read by
                           # the attributes view only; CHIDORI 0; research/race12/search_secrets.md s4)
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
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


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property, and each question is answered only about the first property it names, so every question costs one property.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.
Name a hinted property by its plain name: a hint about who will wear it points at gender, a hint about its looks at colour or finish (whichever the request leaves open).
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its size, capacity or dimensions, its type, its material or its colour, whichever that kind of product is bought by) and keys carried by most of the item's listings; avoid keys seen on only one or two of them.
Skip a property the request already settles for that item, a second name for a property already chosen, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics. With several items, put every item's hinted properties first, then take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

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


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def _tokens(text: str) -> frozenset[str]:
    """(r5) A text's words without punctuation, each singular as _stem makes it."""
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", _words(text)))


def _seq(text: str) -> list[str]:
    """(r7) A text's words in order, each singular as _tokens makes it."""
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", _words(text))]


def _phrase_in(value: str, text: str) -> bool:
    """(r7) The value's words stand one after another, in order, inside one comma-, semicolon-, slash- or bar-separated
    part of the text: "10k gold" is in "10k gold, white" but not in "10k white gold"; "sata" is in "sata 3gb/s"."""
    want = _seq(value)
    if not want:
        return False
    for part in re.split(r"[,;/|]", str(text)):
        words = _seq(part)
        if any(words[i:i + len(want)] == want for i in range(len(words) - len(want) + 1)):
            return True
    return False


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
        self.rejudged_at = 0        # (r12) REJUDGE_ALL: the version last rechecked
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
        self.empty_buys = 0                          # (r6) EMPTY_BUY holds sent
        self.note_refusal: tuple[int, str] | None = None   # (r6) (cart_at, the note's meaning) of a refused checkout
        self.rebuy_holds = 0                         # (r6) NOTE_REBUY holds sent
        self.gating = False                          # (r6) inside gate(): the spend ceiling does not apply
        self.capped = False                          # (r6) the working loop stopped on its spend or step cap
        self.researched: set[tuple[int, str]] = set()   # (r7) R1 triggers spent, (item index, requirement)
        self.exact_holds = 0                         # (r7) R2 holds sent
        self.exact_marks: set[str] = set()           # (r7) R2 cart states held
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
        self.cross: set[str] = set()            # (r5) requirements judged on another requested item's cart line
        self.chain_lost: dict[str, int] = {}    # (r5) tries in which a chain's next page was not found
        self.released: set[str] = set()         # (r5) chains that no longer hold the order
        self.compare_ok = 0
        self.compare_bad = 0
        self.t0 = time.monotonic()      # (y1) the episode's own clock
        self.struct_holds = 0           # (y2a)
        self.scan_marks: set[str] = set()
        self.search_shapes: list[dict[str, Any]] = []  # (y2b r4) the actor's own search arguments, in order
        self.nudged = False
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
        if CAP_LEAKS and self.cost > SPEND_CEIL and not self.gating:
            # (r6) past the ceiling no call is sent: every caller already treats None as a failed call
            self.trace.append({"kind": "spend_ceiling", "cost": round(self.cost, 5)})
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
            if text and self.other_item(number, text) is not None:
                # (r5) TRAIL_SCOPE: it points at another item of this order, so the checks judge it on that cart line
                self.cross.add(text)
            elif text and (number, text) not in self.refs:
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
            # (r5) a requirement judged on another item's line is never moved into a chain's description
            rest = [m for m in item.musts if m not in pointers and m not in self.cross]
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

    def other_item(self, number: int, need: str) -> int | None:
        """(r5) TRAIL_SCOPE: the other requested item (1-based) whose name's words the requirement's value carries in full
        ("fits the <item> you end up buying"): its value comes from a line of this same order, not from a page to find."""
        if not TRAIL_SCOPE:
            return None
        value = _tokens(str(need).partition(":")[2] or str(need))
        for other, item in enumerate(self.items, 1):
            name = {w for w in _tokens(item.name) if len(w) >= 3}
            if other != number and name and name <= value:
                return other
        return None

    def need_note(self, number: int, need: str) -> str:
        """What the ledger adds after a requirement: the value read from the page it points at; (r5) the order's line it
        refers to; or that its page was not found and it does not decide the order."""
        if need in self.ref_values:
            return f" (the page it points at gives: {self.ref_values[need]})"
        other = self.other_item(number, need) if need in getattr(self, "cross", ()) else None
        if other is not None:
            return (f" (it refers to item {other} of this same order, {self.items[other - 1].name}: judge it against the cart's"
                    f" line for item {other}, not against a page to find)")
        if need in getattr(self, "released", ()):
            return " (the page it points at was not found, so it does not decide the order: judge the cart on the other requirements)"
        return ""

    def ledger(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            needs = [m + self.need_note(number, m) for m in item.musts + item.told if m not in item.dropped]
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

    def line_items(self, ident: str) -> list[int]:
        """(y2a) The requested items whose shelf or name the listing's category label matches (as item_groups matches)."""
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
        if not label:
            return []
        words = {w for w in _stem(label) if len(w) >= 4}
        out = []
        for number, item in enumerate(self.items):
            long = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            if _words(label) in (_words(item.shelf or ""), _words(item.name)) or words & long:
                out.append(number)
        return out

    def struct_gate(self) -> str:
        """(y2a) Before a checkout: the cart against the request's items, by the code alone (no model): an empty cart, a
        line count that differs from the item count, a line on a shelf of no requested item, an item with no line, a line
        holding another quantity than its item's. Held at most twice; silent when no shelf label matches any item."""
        if not STRUCT_GATE or not self.items or self.struct_holds >= 2 or self.left() <= self.tail() + 2:
            return ""
        lines: dict[str, int | None] = dict(self.cart_lines) or {i: None for i in self.cart_ids()}
        if not lines:
            self.struct_holds += 1
            return "The cart is empty: put the chosen listing of every requested item in the cart at its quantity first."
        matched = {ident: self.line_items(ident) for ident in lines}
        if not any(matched.values()):
            return ""
        problems: list[str] = []
        if len(lines) != len(self.items):
            problems.append(f"the request asks for {len(self.items)} different item(s) and the cart holds {len(lines)} line(s)")
        for ident, hits in matched.items():
            if not hits:
                problems.append(f"{ident} sits on a shelf that matches none of the requested items")
        for number, item in enumerate(self.items):
            if not any(number in hits for hits in matched.values()):
                problems.append(f"no line holds item {number + 1} ({item.name})")
            for ident in [i for i, hits in matched.items() if hits == [number]]:
                units = lines.get(ident)
                if units is not None and item.qty and units != item.qty:
                    problems.append(f"{ident} holds {units} unit(s) and item {number + 1} ({item.name}) asks for {item.qty}")
        if not problems:
            return ""
        self.struct_holds += 1
        self.trace.append({"kind": "struct_gate", "problems": problems[:6]})
        return ("The cart does not match the request's items: " + "; ".join(problems[:6])
                + ". Fix the cart (one line per requested item, at its quantity), then check out.")

    def buy_hold(self) -> str:
        """(r6) A checkout the code's own record shows cannot pass is not sent (no step used), whatever the steps left:
        EMPTY_BUY, an empty cart (at most twice); NOTE_REBUY, the cart a checkout was refused for with a note that changed
        the request, unchanged since (once). No model call; "" when the checkout may go."""
        empty = not (self.cart_lines or self.cart_ids()) and (not self.cart_touched or self.cart_reply_empty())
        if EMPTY_BUY and self.empty_buys < 2 and empty:
            self.empty_buys += 1
            lines = [i for i in self.built if not self.sold_out(i)]
            self.trace.append({"kind": "empty_buy_held", "built": lines, "left": self.left()})
            back = (f"The fullest cart so far held {', '.join(lines)}: put back the lines that still fit, at their quantities, then check out. "
                    if lines else "Put the chosen listing of every requested item in the cart at its quantity first, then check out. ")
            return ("Not run yet (no step used). The cart is empty, so this checkout would fail and use a step. " + back
                    + "End without a purchase only if no opened listing fits.")
        if NOTE_REBUY and self.rebuy_holds < 1 and self.note_refusal and self.note_refusal[0] == self.cart_at \
                and not (NOTE_REBUY_FITS and self.within(dict(self.cart_lines)) is True):
            self.rebuy_holds += 1
            meaning = self.note_refusal[1]
            self.trace.append({"kind": "note_rebuy_held", "note": meaning[:200]})
            return ("Not run yet (no step used). The last checkout of this same cart was refused with a note that changed the request: "
                    + meaning + ". The cart has not changed since: check every line, its quantity and the total against the changed "
                    "request, change the cart to fit it, or end without a purchase if no opened listing fits it.")
        return ""

    def criteria_args(self, label: str, below: float) -> dict[str, Any] | None:
        """(y2b r4) Arguments that make the search tool list one shelf under a price limit, copied from the shape of the
        actor's own earlier searches (no word lists): the argument that held a shelf label seen on listings, the one number
        argument that held a value inside the seen price range, and the enum choices that went with them. Free text and
        other numbers (paging) are left out. None when the actor never searched that way."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        if not getattr(self, "price_key", None) or not getattr(self, "search_shapes", None):
            return None
        cat_key = self.category_key()
        labels = {str(r.get(cat_key)).strip().lower() for r in self.rows.values() if cat_key and r.get(cat_key)}
        prices = [p for p in (self.price_of(i) for i in self.rows) if p is not None]
        if not props or not labels or not prices:
            return None
        low, high = min(prices), max(prices)
        for shape in reversed(self.search_shapes):
            shelf = next((k for k, v in shape.items() if k in props and isinstance(v, str) and v.strip().lower() in labels), None)
            numbers = [k for k, v in shape.items() if k in props and isinstance(v, (int, float)) and not isinstance(v, bool)
                       and low <= float(v) <= 2 * high]
            # one number in the price range only: with two (a limit and an offset) the shape does not say which is which
            if not shelf or len(numbers) != 1:
                continue
            price = numbers[0]
            args: dict[str, Any] = {k: v for k, v in shape.items()
                                    if k not in (shelf, price) and isinstance((props.get(k) or {}).get("enum"), list)}
            args[shelf] = next((v for v in props[shelf].get("enum") or [] if str(v).lower() == label.lower()), label)
            args[price] = int(below) if "integer" in _kinds(props[price]) else round(below, 2)
            faults: list[str] = []
            args = _shape(args, schema, "args", faults)
            if not faults:
                return args
        return None

    def cheaper_scan(self) -> list[str]:
        """(y2a) Before a cheapest checkout, once per cart state: each cart line's own shelf listed below its price with the
        search tool's filter mode (keyword searches read titles and may ignore the price filter); the cheapest unread
        buyable rows get their attributes read so the last check can weigh them."""
        if not CHEAPER_SCAN or self.brief.get("pick") != "cheapest" or not self.price_key:
            return []
        mark = self.cart_mark() + f"#{self.version}"
        cat_key, tool = self.category_key(), self.roles.get("search")
        if mark in self.scan_marks or not cat_key or not tool:
            return []
        self.scan_marks.add(mark)
        found: list[str] = []
        for line in self.cart_ids():
            price = self.price_of(line)
            label = str((self.rows.get(line) or {}).get(cat_key) or "").strip()
            if price is None or not label or self.left() <= self.tail() + 6 or self.finished:
                continue
            args = self.criteria_args(label, price - 1 if float(price).is_integer() else price * 0.999)
            if not args:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            rows = [i for i, _ in self.records_in(seen) if i != line and self.buyable(i) and (self.price_of(i) or float("inf")) < price]
            rows.sort(key=lambda i: self.price_of(i) or 0.0)
            found += [i for i in rows if i not in self.full and i not in found][:2]
        self.trace.append({"kind": "cheaper_scan", "found": found})
        return self.fill_facts(found[:CHEAPER_OPENS], self.tail() + 4)

    # ---------- (y2b) the decline guard ----------
    def item_of(self, ident: str, loose: bool = False) -> list[int]:
        """(y2b) The requested items a listing can fill, by its shelf label (line_items); with `loose`, a one-item request
        whose seen listings match no item by label counts every listing as that item."""
        hits = self.line_items(ident)
        if hits or not loose or len(self.items) != 1:
            return hits
        # whether any seen listing matches an item by label, remembered per count of listings seen (one scan, not one per call)
        seen = getattr(self, "label_seen", None)
        if not seen or seen[0] != len(self.seen_order):
            seen = self.label_seen = (len(self.seen_order), any(self.line_items(i) for i in self.seen_order))
        return [] if seen[1] else [0]

    def within(self, lines: dict[str, int]) -> bool | None:
        """(y2b) Whether these lines keep to the budget by the listings' recorded prices; None when a price or the budget
        cannot be read (the model checks decide then)."""
        if self.budget is None:
            return True
        if not isinstance(self.budget, (int, float)) or isinstance(self.budget, bool):
            return None
        prices = {i: self.price_of(i) for i in lines}
        if any(p is None for p in prices.values()):
            return None
        if self.brief.get("budget_scope") == "each":
            return all(p <= self.budget + 1e-9 for p in prices.values())
        return sum(prices[i] * max(1, n or 1) for i, n in lines.items()) <= self.budget + 1e-9

    def clash(self, ident: str, number: int) -> bool:
        """(y2b) The listing's opened record contradicts a short word-valued requirement of item `number`: every attribute
        whose key names the requirement's property holds none of the value's words. Figures and long values (limits,
        values read from other listings) are left to the model checks. A wrong call here only makes a hold stay silent
        (the decline goes on as before); a cart Kyo plans from it still needs the model check to pass."""
        block = _fact_block(self.full.get(ident)) or {}
        if not block or not 0 <= number < len(self.items):
            return False
        item = self.items[number]
        for need in [m for m in item.musts + item.told if m not in item.dropped]:
            prop, _, value = str(need).partition(":")
            words = re.findall(r"[a-z]+", value.lower())
            if not prop.strip() or not words or len(words) > 3 or re.search(r"\d", value):
                continue
            keys = [k for k in block if _names(_stem(prop), _stem(k))]
            want = _stem(" ".join(words))
            if keys and all(not (want & _stem(str(block[k]).lower())) for k in keys):
                return True
        return False

    def judged(self, ident: str) -> bool | None:
        """(y2b) The latest model judgment of a listing: its own newest check, else a passed check of a cart holding it,
        else the review's or a check's confirmation; None when nothing judged it."""
        own = []
        for key, ok in self.verdicts.items():
            ids, _, version = key.rpartition("#")
            if ids == ident and version.isdigit():
                own.append((int(version), ok))
        if own:
            version, ok = max(own)
            # (r11) JUDGED_FRESH: a "no" given before the requirements changed (a withdrawal, a new budget) no longer
            # rules the listing out; it counts as not judged until a check under the current requirements
            if ok or not JUDGED_FRESH or version == self.version:
                return ok
        if any(ok and ident in key.rpartition("#")[0].split("|") for key, ok in self.verdicts.items()):
            return True
        if ident == self.confirmed or ident in self.leads or any(ident in ids for ids in self.lead_items.values()):
            return True
        return None

    def fits_for(self, number: int, skip: set[str], label: str = "", loose: bool = False) -> list[str]:
        """(y2b) Opened listings that can fill item `number`: a full record, buyable, on that item's shelf (the given label
        when one is known), no plain requirement contradicted, not judged unfit; judged fits first, then cheapest first."""
        cat_key = self.category_key()
        pool = []
        for ident in reversed(self.seen_order):
            if ident in skip or ident not in self.full or not self.buyable(ident):
                continue
            if label and cat_key:
                if str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() != label:
                    continue
            elif number not in self.item_of(ident, loose):
                continue
            if self.clash(ident, number) or self.judged(ident) is False:
                continue
            pool.append(ident)
        return sorted(pool, key=lambda i: (self.judged(i) is not True, self.price_of(i) if self.price_of(i) is not None else float("inf")))

    def word_need(self, number: int, need: str) -> tuple[str, str] | None:
        """(r7) A requirement of item `number` (0-based) the code can check word for word: "property: value" with a value
        of one to four words, none a bare figure (limits and amounts stay with the model checks), not a reference to
        another listing, not dropped. Returns (property, value) or None."""
        item = self.items[number]
        if need in item.dropped or need in self.cross or need in self.released or any(need == n for _, n in self.refs):
            return None
        prop, sep, value = str(need).partition(":")
        words = _seq(value)
        if not sep or not prop.strip() or not 1 <= len(words) <= 4 or any(w.isdigit() for w in words):
            return None
        return prop.strip(), value.strip()

    def live_needs(self, number: int) -> list[str]:
        item = self.items[number]
        return [m for m in item.musts + item.told if m not in item.dropped]

    def states(self, ident: str, prop: str, value: str) -> str:
        """(r7) What an opened listing says on a word requirement: "yes" when an attribute naming the property holds the
        value's words in a row (or, with no such attribute, a text field of the record does), "scattered" when it holds
        every word but not in a row, "no" when it holds something else, "silent" when nothing in the record names it."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        block = _fact_block(record) or {}
        texts = [str(block[k]) for k in block if _names(_stem(prop), _stem(k))]
        if texts:
            if any(_phrase_in(value, t) for t in texts):
                return "yes"
            want = set(_seq(value))
            return "scattered" if any(want <= set(_seq(t)) for t in texts) else "no"
        return "yes" if self.row_says(ident, value, record) else "silent"

    def prop_value(self, ident: str, prop: str) -> str:
        block = _fact_block(self.full.get(ident)) or {}
        return next((str(block[k]) for k in block if _names(_stem(prop), _stem(k))), "")

    def row_says(self, ident: str, value: str, record: dict[str, Any] | None = None) -> bool:
        """(r7) A text field of the listing's record (title, description...) holds the value's words in a row."""
        node = record if record is not None else (self.rows.get(ident) or {})
        return any(isinstance(v, str) and _phrase_in(value, v) for v in node.values())

    def value_rows(self, number: int, value: str, limit: int = RESELECT_OPENS) -> list[str]:
        """(r7) Listed rows of item `number` never opened, buyable, inside the budget, whose text holds the value; cheapest
        first under a cheapest pick, else in the order seen."""
        qty = self.items[number].qty or 1
        rows = [i for i in self.seen_order if i not in self.full and self.buyable(i) and number in self.item_of(i, True)
                and self.within({i: qty}) is not False and self.row_says(i, value)]
        if self.brief.get("pick") == "cheapest":
            rows.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
        return rows[:limit]

    def value_search(self, query: str, keep: int = 1) -> str:
        """(r7) One search for `query` with the search tool: the free-text argument the actor's last search filled (else
        the schema's first required free-text one) takes the query, the enum choices of that search stay, nothing else
        (no paging). (r8) A query longer than that argument's maxLength loses whole words from its end, never its first
        `keep` words (the value), instead of being cut mid-word by _shape (race 10 bed: "pullover blazers & vests for b").
        Returns the query sent, "" when no such call can be shaped or the call failed."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        if not tool or not free:
            return ""
        last = dict(self.search_shapes[-1]) if self.search_shapes else {}
        filled = [k for k in free if isinstance(last.get(k), str) and last[k].strip()]
        target = max(filled, key=lambda k: len(last[k])) if filled else next((k for k in schema.get("required") or [] if k in free), free[0])
        limit = props[target].get("maxLength")
        if isinstance(limit, int) and limit > 0 and len(query) > limit:
            words = query.split()
            while len(words) > max(1, keep) and len(" ".join(words)) > limit:
                words.pop()
            query = " ".join(words)
            if len(query) > limit:
                query = query[:limit].rsplit(" ", 1)[0] if " " in query[:limit] else query[:limit]
        args: dict[str, Any] = {k: v for k, v in last.items() if isinstance(props.get(k), dict) and "enum" in props[k]}
        args[target] = query
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        if faults:
            return ""
        reply = self.act(tool, args)
        if reply is None:
            return ""
        self.observe(tool, reply.get("observation"))
        return query

    def reveal_research(self, pairs: list[tuple[int, str]]) -> str:
        """(r7) RESELECT R1: a revealed word requirement that no opened listing of its item states while two or more state
        something else rules them all out (race 12: 140 of 162 reveals on a property every opened listing shared asked for
        another value). Kyo reads the listed, unopened rows whose text holds the value; when none of them states it, it
        searches once for "<value> <item name>" and reads the rows that hold it (else the cheapest on the item's shelf).
        At most RESELECT_OPENS reads and one search a trigger, two triggers an episode, the store only with RESELECT_ROOM
        steps free; no model call. Returns what to tell the actor ("" when nothing applies)."""
        if not RESELECT or not RESELECT_R1:
            return ""
        notes: list[str] = []
        for number, need in pairs:
            spec = self.word_need(number, need)
            if not spec or (number, need) in self.researched or len(self.researched) >= 2:
                continue
            prop, value = spec
            opened = [i for i in self.full if number in self.item_of(i, True) and self.buyable(i)]
            said = {i: self.states(i, prop, value) for i in opened}
            if "yes" in said.values():
                continue
            against = [i for i, s in said.items() if s in ("no", "scattered")]
            if len(against) < 2:
                continue
            self.researched.add((number, need))
            room = (self.left() > self.tail() + RESELECT_ROOM and not (self.step_cap() and self.step > self.step_cap() - RESELECT_ROOM)
                    and not self.finished and not self.broke and self.cost <= SPEND_HARD)
            query = ""
            if room:
                self.fill_facts(self.value_rows(number, value), self.tail() + RESELECT_ROOM - 2)
                if not any(self.states(i, prop, value) == "yes" for i in self.full if i not in opened and number in self.item_of(i, True)):
                    # (r8) the value with the store's own word for the item, not the request's category words
                    noun = self.shelf_word(number, set(_seq(value)))
                    query = self.value_search(f"{value} {noun or self.items[number].name}", len(value.split()))
                    if query:
                        before = set(self.full)
                        picks = self.value_rows(number, value) or [i for i in self.value_rows_any(number) if i not in before][:1]
                        self.fill_facts(picks, self.tail() + RESELECT_ROOM - 3)
            fresh = [i for i in self.full if i not in opened and number in self.item_of(i, True)]
            found = [i for i in fresh if self.states(i, prop, value) == "yes"]
            self.trace.append({"kind": "reveal_research", "need": need[:120], "against": against[:6], "query": query,
                               "read": fresh[:6], "found": found[:4], "room": room})
            shown = " | ".join(dict.fromkeys(self.prop_value(i, prop) for i in against if self.prop_value(i, prop)))[:200]
            # (r8) facts, not orders: race 10 bed r7 lost t-a1d8d0a25a and t-65b5a13915 when "pick only among listings
            # stating it" made the actor drop a fitting listing whose record left the property empty
            text = (f"[Kyo] The answer \"{need}\" does not match the {prop} of the listings opened so far for item {number + 1}"
                    + (f" (theirs: {shown})" if shown else "") + ".")
            if fresh:
                text += (f" Kyo searched \"{query}\" and" if query else " Kyo") + " read " + ", ".join(
                    f"{i} ({prop}: {self.prop_value(i, prop) or 'not stated'}, price {self.price_of(i)})" for i in fresh[:4]) + "."
            if found:
                text += f" {', '.join(found[:3])} state{'s' if len(found) == 1 else ''} \"{value}\"."
            text += (f" A listing whose {prop} states another value does not fit; one whose record does not state {prop}"
                     " may still fit.")
            notes.append(text)
        return "\n".join(notes)

    def shelf_word(self, number: int, skip: set[str]) -> str:
        """(r8) The store's own word for item `number`: the word that most of its listed rows carry in their text fields
        (the category field and ids left out), at least half of them, not a figure, not one of `skip`. Race 12's search
        indexes titles, attribute values and seller names but not category labels: adding the request's category
        words pushed a page-one gold off in 14% of query pairs, a title word brought one in 23%
        (research/race12/search_secrets.md s2). "" when no word is that common."""
        cat_key = self.category_key()
        rows = [self.rows.get(i) or {} for i in self.seen_order if number in self.item_of(i, True)]
        counts: dict[str, int] = {}
        for row in rows:
            words = set()
            for key, value in row.items():
                if key in (cat_key, self.id_key) or not isinstance(value, str):
                    continue
                words |= {w for w in _seq(value) if len(w) >= 3 and not any(ch.isdigit() for ch in w) and w not in skip}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
        if not rows or not counts:
            return ""
        word, n = max(counts.items(), key=lambda kv: (kv[1], -len(kv[0]), kv[0]))
        return word if n * 2 >= len(rows) else ""

    def value_rows_any(self, number: int) -> list[str]:
        """(r7) Listed rows of item `number` never opened, buyable, cheapest first (R1's read when no row holds the value)."""
        rows = [i for i in self.seen_order if i not in self.full and self.buyable(i) and number in self.item_of(i, True)]
        return sorted(rows, key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))

    def exact_hold(self) -> str:
        """(r7) RESELECT R2, before a checkout: a cart line of one item whose attribute holds a word requirement's words
        but not in a row, or holds another value, while an opened fit of that item (fits_for) states the value in a row,
        breaks none of the item's other word requirements and keeps the budget: the checkout is held once per cart state
        (twice an episode) and the actor shown the swap. No model call; silent with few steps left."""
        if not RESELECT or self.exact_holds >= 2 or self.left() <= self.tail() + 2:
            return ""
        mark = self.cart_mark() + f"#{self.version}"
        if mark in self.exact_marks:
            return ""
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        for line in cart:
            hits = self.item_of(line, True)
            if len(hits) != 1:
                continue
            number = hits[0]
            specs = [s for s in (self.word_need(number, m) for m in self.live_needs(number)) if s]
            for prop, value in specs:
                said = self.states(line, prop, value)
                if said not in ("no", "scattered"):
                    continue
                # (r9c) another value that shares a word with the requirement may be the same value differently written
                # (race 11 t-271d10ec22: four passing lines whose listing spelled "polyster"); both R2 paths skip it
                if said == "no" and set(_seq(self.prop_value(line, prop))) & set(_seq(value)):
                    continue
                rest = {i: self.cart_lines.get(i, 1) for i in cart if i != line}
                alts = [c for c in self.fits_for(number, set(cart))
                        if self.states(c, prop, value) == "yes"
                        and not any(self.states(c, p, v) in ("no", "scattered") for p, v in specs)
                        and self.within({**rest, c: self.items[number].qty or 1}) is not False]
                if not alts:
                    stated = self.prop_value(line, prop)
                    if not EXACT_ANY or not stated or any(ch.isdigit() for ch in value):
                        continue
                    # another wording of the same value shares a word with it ("1 pack" for "single pack": race 10 bed
                    # t-2bd1cfc387 lost on such a hold); only a value with no word in common, or the words out of order, is held
                    if self.states(line, prop, value) == "no" and set(_seq(stated)) & set(_seq(value)):
                        continue
                    # (r9) R2b: no opened fit to name, but a line stating another value (or the words out of order)
                    # passes almost never (race 12 official: 0 of 299 scattered and 0 of 508 other-value lines)
                    self.exact_marks.add(mark)
                    self.exact_holds += 1
                    self.trace.append({"kind": "exact_hold_any", "line": line, "need": f"{prop}: {value}"[:120]})
                    return (f"Not run yet (no step used). {line} states {prop} \"{self.prop_value(line, prop)}\", which is not"
                            f" \"{value}\" as the requirement words it, and the order is judged on the value as written."
                            f" Replace {line} with a listing whose {prop} states \"{value}\" (search the value with the"
                            " item's name if none you opened does), then check out.")
                self.exact_marks.add(mark)
                self.exact_holds += 1
                alt = alts[0]
                self.trace.append({"kind": "exact_hold", "line": line, "need": f"{prop}: {value}"[:120], "alt": alt})
                return (f"Not run yet (no step used). {line} states {prop} \"{self.prop_value(line, prop)}\", which is not"
                        f" \"{value}\" as the requirement words it; {alt} states \"{self.prop_value(alt, prop) or value}\""
                        f" (price {self.price_of(alt)}). Swap {line} for {alt} unless {alt}'s record breaks another"
                        " requirement, then check out.")
        return ""

    def place(self, lines: dict[str, int], calls: list[dict[str, Any]], why: str) -> str:
        """(y2b) Kyo sets the cart with `calls` and checks out (the request's steps at the order run first); returns what
        happened in the store's words, or "" when nothing was sent."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if not cart or not buy or self.finished:
            return ""
        shaped = []
        for args in calls:
            faults: list[str] = []
            shaped.append(_shape(args, self.schemas.get(cart), "args", faults))
            if faults:
                return ""
        faults = []
        buy_args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        done = []
        for args in shaped:
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ("Kyo set part of the cart (" + "; ".join(done) + ") and the store stopped answering.") if done else ""
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        head = f"Not run (Kyo acted instead): {why} Kyo set the cart ({'; '.join(done) or 'it already held ' + ', '.join(lines)})"
        if self.left() < 1:
            return head + "."
        go, said = self.gate("order") if self.left() >= 2 else (True, "")
        if not go:
            return head + "; " + said
        reply = self.act(buy, buy_args)
        self.trace.append({"kind": "y2b_placed", "lines": lines, "reply": _clip(reply, 200)})
        if reply is None:
            return head + " and the checkout call failed (network or server)."
        self.buys += 1
        if not reply.get("finished"):
            self.refused_at = self.step
        self.observe(buy, reply.get("observation"))
        refused = "" if self.finished or reply.get("finished") else "the checkout was refused: "
        return head + " and checked out. Store reply: " + refused + self.show(reply.get("observation"))

    def unopened_hold(self) -> str:
        """(y2b) UNOPENED_HOLD: a decline while some requested item has buyable rows on its shelf but none opened in full
        is held once per item set; Kyo reads that item's two cheapest buyable (in-budget) unopened rows first. Silent when
        some item's opened listings all contradict a plain requirement or break the budget (the decline may be right)."""
        if not UNOPENED_HOLD or not self.items or self.left() <= self.tail() + 3:
            return ""
        groups: dict[int, list[str]] = {n: [] for n in range(len(self.items))}
        for ident in self.seen_order:
            if self.buyable(ident):
                for number in self.line_items(ident):
                    groups[number].append(ident)
        bare = [n for n, ids in groups.items() if ids and not any(i in self.full for i in ids)]
        if not bare:
            return ""
        for number, ids in groups.items():
            opened = [i for i in ids if i in self.full]
            if opened and all(self.clash(i, number) or self.within({i: self.items[number].qty}) is False for i in opened):
                return ""
        marks = getattr(self, "unopened_marks", None)
        if marks is None:
            marks = self.unopened_marks = set()
        mark = "|".join(map(str, bare))
        if mark in marks:
            return ""
        marks.add(mark)
        picks: list[str] = []
        for number in bare:
            rows = [i for i in groups[number] if i not in self.full and i not in picks and self.within({i: self.items[number].qty}) is not False]
            rows.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            picks += rows[:2]
        if not picks:
            return ""
        before = set(self.full)
        texts = self.fill_facts(picks, self.tail() + 3)
        if not texts and self.open_rows(picks):
            texts = [f"{i}: {self.record_of(i)}" for i in picks if i in self.full and i not in before]
        self.trace.append({"kind": "unopened_held", "items": [n + 1 for n in bare], "rows": picks, "read": len(texts)})
        if not texts:
            return ""
        if self.audited and not getattr(self, "reaudited", False):
            # the review may look once more, now that these records exist
            self.audited, self.reaudited = False, True
        names = ", ".join(f"item {n + 1} ({self.items[n].name})" for n in bare)
        return ("Not run yet. No listing of " + names + " had been opened in full, so ending now would judge it unseen. Kyo read the"
                " cheapest buyable rows of its shelf (one step each):\n" + "\n".join(texts)
                + "\nIf one of them meets every requirement, put it in the cart with your picks for the other items and check out. End"
                " without a purchase only if each requested item has no listing that meets its requirements; then say which"
                " requirement fails and send the ending call again.")

    def refusal_hold(self) -> str:
        """(y2b) REFUSAL_DECLINE_HOLD: a decline right after a refused checkout or a removed cart line is held once per cart
        state when, for every item the cart now misses, an opened listing of that line's shelf is buyable, keeps to the
        budget, contradicts no plain requirement, is not judged unfit, and the planned cart passes the check. With steps
        only just enough, Kyo sets that cart and checks out itself."""
        if not REFUSAL_DECLINE_HOLD or not self.items or not self.roles.get("cart") or not self.roles.get("buy"):
            return ""
        removed_at = getattr(self, "removed_at", -1)
        recent = (self.refused_at >= 0 and self.step - self.refused_at <= REFUSAL_RECENT) or (removed_at >= 0 and self.step - removed_at <= REFUSAL_RECENT)
        if not recent:
            return ""
        marks = getattr(self, "refusal_marks", None)
        if marks is None:
            marks = self.refusal_marks = set()
        mark = self.cart_mark() + f"#{self.version}"
        if mark in marks or len(marks) >= 2:
            return ""
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        hits = {i: self.item_of(i, True) for i in cart}
        lost = [n for n in range(len(self.items)) if not any(n in h for h in hits.values())]
        if not lost:
            return ""
        keep = {i: (self.items[h[0]].qty if len(h) == 1 else self.cart_lines.get(i, 1)) for i, h in hits.items() if h}
        cat_key = self.category_key()
        taken = [i for i in list(getattr(self, "removed_lines", [])) + sorted(self.dropped) if i not in cart]
        plan: dict[str, int] = {}
        for number in lost:
            label = next((str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() for i in taken
                          if cat_key and number in self.item_of(i, True)), "")
            picks = [c for c in self.fits_for(number, set(cart) | set(plan) | set(taken), label, True)
                     if self.within({**keep, **plan, c: self.items[number].qty}) is not False]
            if not picks:
                return ""
            plan[picks[0]] = self.items[number].qty
        lines = {**keep, **plan}
        drop = [i for i in self.cart_ids() if i not in lines]
        calls = self.cart_calls(lines, drop)
        need = (len(calls) if calls is not None else len(plan) + len(drop)) + self.tail()
        if self.left() < need:
            return ""
        marks.add(mark)
        key = self.check_key(list(lines))
        if not self.verdicts.get(key):
            self.check_cart(list(lines), _late_limit())
        self.trace.append({"kind": "refusal_decline_held", "lines": lines, "drop": drop, "ok": bool(self.verdicts.get(key))})
        if not self.verdicts.get(key):
            return ""
        named = "; ".join(f"{c} for item {n + 1} ({self.items[n].name})" for c, n in zip(plan, lost))
        if calls and self.left() <= need + 1:
            done = self.place(lines, calls, f"too few steps were left for another round, and an opened listing fills the lost line: {named}.")
            if done:
                return done
        return ("Not run yet (no step used). The checkout was just refused or a cart line was taken off, and a listing already"
                f" opened fills that line: {named}. An independent check of the planned cart ({', '.join(lines)}) found it meets"
                f" every requirement inside the budget. Put {', '.join(plan)} in the cart at the requested quantity"
                + (f" and take {', '.join(drop)} off" if drop else "") + ", then check out. End without a purchase only if its"
                " record clearly breaks a requirement; then say which one and send the ending call again.")

    def loop_end_order(self) -> str:
        """(y2b) LOOP_END_ORDER: the working loop stopped (calls, spend or clock) while the cart misses items: when a checked
        pick exists for every item, Kyo sets that cart and checks out, if a check passes it, instead of finish() ending
        without a purchase. Returns what happened, or "" (then the episode ends as before)."""
        if not LOOP_END_ORDER or self.finished or self.broke or not self.items or self.left() < 2:
            return ""
        if not self.roles.get("cart") or not self.roles.get("buy"):
            return ""
        marks = getattr(self, "end_marks", None)
        if marks is None:
            marks = self.end_marks = set()
        mark = self.cart_mark() + f"#{self.version}"
        if mark in marks or len(marks) >= 2:
            return ""
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        hits = {i: self.item_of(i, True) for i in cart}
        if cart and not any(hits.values()):
            return ""
        if cart and len(cart) == len(self.items) and all(len(h) == 1 for h in hits.values()) \
                and len({h[0] for h in hits.values()}) == len(self.items):
            return ""
        marks.add(mark)
        plan: dict[str, int] = {}
        for number, item in enumerate(self.items):
            mine = [i for i in cart if hits[i] == [number] and self.judged(i) is True]
            pick = mine[0] if mine else next((c for c in self.fits_for(number, set(plan), "", True) if self.judged(c) is True), None)
            if not pick or pick in plan:
                return ""
            plan[pick] = item.qty
        if self.within(plan) is False:
            return ""
        drop = [i for i in self.cart_ids() if i not in plan]
        calls = self.cart_calls(plan, drop)
        if calls is None or len(calls) + self.tail() > self.left():
            return ""
        key = self.check_key(list(plan))
        if not self.verdicts.get(key):
            self.fill_facts(list(plan), len(calls) + self.tail() + 1)
            self.check_cart(list(plan), _late_limit())
        self.trace.append({"kind": "loop_end_order", "lines": plan, "drop": drop, "ok": bool(self.verdicts.get(key))})
        if not self.verdicts.get(key):
            return ""
        return self.place(plan, calls, "the working loop stopped with a cart that misses items, and a checked pick exists for every item.")

    def backup_rows(self) -> list[str]:
        """(r5) BACKUP_OPEN: once, at the first checkout, a cart line whose item has no other opened candidate (fits_for:
        buyable, same shelf, uncontradicted) gets one more read, so a stock refusal finds a replacement already open: the
        unopened buyable row of that shelf, inside the budget with the other lines, whose words share most with the item's
        requirements, then the cheapest. Only with steps for a later swap (remove, add, checkout) and spend under
        SPEND_SOFT; no model call."""
        if not BACKUP_OPEN or self.buys or getattr(self, "backup_done", False) or self.cost >= SPEND_SOFT:
            return []
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        if not cart or self.left() < self.tail() + 6 + 2 * len(cart):
            return []
        self.backup_done = True
        cat_key = self.category_key()
        keep = {i: self.cart_lines.get(i, 1) for i in cart}
        picks: list[str] = []
        for line in cart:
            hits = self.item_of(line, True)
            if len(hits) != 1:
                continue
            number = hits[0]
            label = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower() if cat_key else ""
            if self.fits_for(number, set(cart) | set(picks), label, True):
                continue
            item = self.items[number]
            want = set()
            for need in item.musts + item.told:
                if need not in item.dropped:
                    want |= _tokens(str(need).partition(":")[2])
            others = {i: n for i, n in keep.items() if i != line}
            rows = [i for i in self.seen_order if i not in self.full and i not in keep and i not in picks and self.buyable(i)
                    and (str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == label if label else number in self.item_of(i, True))
                    and self.within({**others, i: item.qty}) is not False]
            if not rows:
                continue

            def rank(ident: str) -> tuple[int, float]:
                text = " ".join(str(v) for v in (self.rows.get(ident) or {}).values() if isinstance(v, str))
                price = self.price_of(ident)
                return -len(want & _tokens(text)), price if price is not None else float("inf")

            picks.append(min(rows, key=rank))
            if len(picks) >= 2:
                break
        if not picks:
            return []
        texts = self.fill_facts(picks, self.tail() + 4 + 2 * len(cart))
        self.trace.append({"kind": "backup_open", "rows": picks, "read": len(texts)})
        return texts

    def chain_hold(self) -> str:
        """(r5) TRAIL_RELEASE: a decline while a requirement's chain of pages is unresolved, when the cart holds a line for
        every item and the check passes it on the other requirements (a released chain decides nothing there), is held
        once per cart state: an unresolved chain alone is no reason to end without a purchase. With steps only just
        enough, Kyo checks the cart out itself. Silent when no chain is open, the cart misses an item or the check fails."""
        if not TRAIL_RELEASE or not self.items or not self.roles.get("buy") or self.left() < self.tail() + 1:
            return ""
        live = {need for _, need in self.needs_now()}
        chains = [n for n in list(self.released) + [need for _, need in self.refs] if n in live and n not in self.ref_values]
        if not chains:
            return ""
        cart = [i for i in self.cart_ids() if not self.sold_out(i)]
        hits = {i: self.item_of(i, True) for i in cart}
        if not cart or any(not any(n in h for h in hits.values()) for n in range(len(self.items))):
            return ""
        marks = getattr(self, "chain_marks", None)
        if marks is None:
            marks = self.chain_marks = set()
        mark = self.cart_mark() + f"#{self.version}"
        if mark in marks or len(marks) >= 2:
            return ""
        marks.add(mark)
        key = self.check_key(self.cart_ids())
        if key not in self.verdicts:
            self.check_cart(limit=_late_limit())
        self.trace.append({"kind": "chain_decline_held", "chains": chains[:3], "cart": cart, "ok": bool(self.verdicts.get(key))})
        if not self.verdicts.get(key):
            return ""
        if self.left() <= self.tail() + 1:
            done = self.place({i: self.cart_lines.get(i, 1) for i in cart}, [], "an unresolved chain of pages alone is no reason to end without a purchase, and the check passed the cart.")
            if done:
                return done
        return ("Not run yet (no step used). The cart holds a line for every item, and an independent check found it meets the"
                " requirements except the value a chain of pages did not give (" + " | ".join(chains[:2]) + "). An unresolved chain"
                " alone is no reason to end without a purchase: check out this cart. End without a purchase only if a record"
                " clearly breaks a requirement; then say which one and send the ending call again.")

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
        cap = self.compare_room() if self.compare_bad else (props[target].get("maxItems") if isinstance(props[target].get("maxItems"), int) else 4)
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
            # (r10) REVEAL_WIDEN: up to three opened listings an item before the plan (was two more at most)
            want = max(min(2, 8 - opened), 3 - opened) if REVEAL_WIDEN else min(2, 8 - opened)
            lanes.append(pool[: max(0, want)])
        queue = [lane[k] for k in range(3) for lane in lanes if k < len(lane)]
        # while steps are plentiful: 22 stay for the rest of the episode
        self.fill_facts(queue[: min(9, 3 * len(lanes))] if REVEAL_WIDEN else queue[:5], 22)

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
        prompt = PLAN.replace("COUNT", str(count + 1)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
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
        if QUESTION_SPREAD:
            ordered = self.spread_order(ordered, first)
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        known = [list(item.told) for item in self.items]      # (r7) to tell the requirements these answers add
        # all messages but the last first; the last re-asks a question whose answer wanted a single question
        # (named with example values this time), else it goes to the next pick
        if REPLAN_EACH and self.cost <= SPEND_SOFT and count >= 2:
            sent += self.ask_waves(ordered, count, first, told)
            if not sent:
                return ""
        head = [] if sent else (ordered[: count - 1] if count >= 2 else ordered[:count])
        batch = self.send([(number, label, self.question(number, label)) for number, label in head]) if head else []
        if not batch and not sent:
            return ""
        sent += batch
        heard = self.hear(batch) if batch else {}
        self.take_heard(heard, told)
        if batch and count >= 2 and self.msgs_free() > 0:
            crowded =[batch[i - 1] for i in (_count(x) for x in _items(heard.get("crowded"))) if i and 1 <= i <= len(batch)]
            again = None
            for number, text, _ in (crowded if REASK_SAMPLES else []):
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
                    self.take_heard(self.hear(batch, (LIGHT, HEAVY)), told)
        if told:
            self.version += 1
        added = [(n, need) for n, item in enumerate(self.items) for need in item.told if need not in known[n]]
        research = self.reveal_research(added) if RESELECT and added else ""
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        if research:
            summary += "\n" + research
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def plan_picks(self, plan: dict[str, Any]) -> list[tuple[int, str]]:
        """(r12) The (item number, label) picks of a PLAN reply, cleaned as interview() cleans them."""
        ordered: list[tuple[int, str]] = []
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all(_words(label) != _words(other) for n, other in ordered if n == number):
                ordered.append((number, label))
        return ordered

    def replan(self, sent: list[tuple[int, str, str]], count: int, first: list[str]) -> list[tuple[int, str]]:
        """(r12) REPLAN_EACH: the remaining picks planned again with the requirements now known and the questions already
        asked with their answers; the light model first. [] when the plan call fails."""
        settled = [f"item {n}: {it.name} - settled: " + ("; ".join(it.musts + [t for t in it.told if t not in it.dropped]) or "nothing")
                   for n, it in enumerate(self.items, 1)]
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(self.key_table(), 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(f"  {index}. {name} (on {carried}): " + " | ".join(values))
        prompt = PLAN.replace("COUNT", str(count + 1)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += ("\n\nALREADY ASKED (never ask these again; an answer that gives no value means that property holds no "
                   "preference for that item):\n" + "\n".join(f"(item {n}) Q: {q} A: {a[:160]}" for n, q, a in sent))
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        picks = [(n, lab) for n, lab in self.plan_picks(plan)
                 if not any(n == m and set(_seq(lab)) <= set(_seq(q)) for m, q, _ in sent)]
        self.trace.append({"kind": "replan", "after": len(sent), "ask": picks[:8]})
        return picks

    def ask_waves(self, ordered: list[tuple[int, str]], count: int, first: list[str], told: list[str]) -> list[tuple[int, str, str]]:
        """(r12) REPLAN_EACH: questions in waves of REPLAN_WAVE; each wave's answers are read at once, and a wave that
        states a new requirement or asks for one question at a time has the rest planned again (two re-plans at most).
        A question an answer wanted asked singly is asked again with two sample values when REASK_SAMPLES allows."""
        sent: list[tuple[int, str, str]] = []
        queue = list(ordered)
        replans = 0
        while queue and self.msgs_free() > 0 and len(sent) < count and not self.finished:
            room = min(REPLAN_WAVE, self.msgs_free(), count - len(sent))
            wave = []
            while queue and len(wave) < room:
                number, label = queue.pop(0)
                if any(number == m and self.question(number, label) == q for m, q, _ in sent):
                    continue
                wave.append((number, label, self.question(number, label)))
            if not wave:
                break
            batch = self.send(wave)
            if not batch:
                break
            sent += batch
            before = len(told)
            heard = self.hear(batch)
            self.take_heard(heard, told)
            crowded = [batch[i - 1] for i in (_count(x) for x in _items(heard.get("crowded"))) if i and 1 <= i <= len(batch)]
            if crowded and REASK_SAMPLES and self.msgs_free() > 0 and len(sent) < count:
                number, text, _ = crowded[0]
                label = next((lab for n, lab, q in wave if n == number and q == text), None)
                samples = self.samples(number, label) if label else []
                if label and len(samples) >= 2:
                    again = self.send([(number, label, self.question(number, label, samples))])
                    if again:
                        sent += again
                        self.take_heard(self.hear(again, (LIGHT, HEAVY)), told)
            if (len(told) > before or crowded) and queue and self.msgs_free() > 0 and replans < 2 and self.cost <= SPEND_SOFT:
                replans += 1
                fresh = self.replan(sent, min(self.msgs_free(), count - len(sent)), first)
                if fresh:
                    queue = fresh
        return sent

    def spread(self, number: int, label: str) -> int:
        """(r6) How the opened listings of item `number` (1-based) split on the property `label`, read off their attribute
        records by key name: 2 for two or more values, 0 for one value on two or more listings, 1 when it cannot be told
        (no opened listing of the item carries the key, or only one does)."""
        want = _stem(label)
        values: list[str] = []
        for ident, record in self.full.items():
            if number - 1 not in self.item_of(ident, True):
                continue
            block = _fact_block(record) or {}
            keys = [k for k in block if _names(want, _stem(k))]
            if keys:
                values.append(str(block[keys[0]]).strip().lower())
        if len(values) < 2:
            return 1
        return 2 if len(set(values)) >= 2 else 0

    def spread_order(self, ordered: list[tuple[int, str]], first: list[str]) -> list[tuple[int, str]]:
        """(r6) QUESTION_SPREAD: the plan's picks reordered inside each item's own slots, so the item order and the count
        per item stay: the request's ask-first properties first, then the properties the item's opened listings split on,
        the ones that cannot be told, and last the ones every opened listing shares. Ties keep the plan's order."""
        pinned = {_words(x) for x in first}
        slots: dict[int, list[str]] = {}
        for number, label in ordered:
            slots.setdefault(number, []).append(label)
        for number, labels in slots.items():
            labels.sort(key=lambda lab: -3 if _words(lab) in pinned else -self.spread(number, lab))
        queue = {number: iter(labels) for number, labels in slots.items()}
        out = [(number, next(queue[number])) for number, _ in ordered]
        if out != ordered:
            self.trace.append({"kind": "spread_order", "before": ordered[:8], "after": out[:8]})
        return out

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
            if entry.get("points") is True and self.other_item(number, need) is not None:
                self.cross.add(need)            # (r5) TRAIL_SCOPE
            elif entry.get("points") is True and (number, need) not in self.refs and need not in self.released:
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
        """(r6) gate_steps() with the spend ceiling lifted: the request's own steps before an order or an ending are done
        whatever the spend (race 6: an order without the asked-for approval scored nothing)."""
        self.gating = True
        try:
            return self.gate_steps(stage)
        finally:
            self.gating = False

    def gate_steps(self, stage: str) -> tuple[bool, str]:
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

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
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
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
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

    def step_cap(self) -> int:
        """(r6/r9d) The store step after which the working loop sends no new model turn; 0 for none. With
        STEP_CAP_PER_ITEM it grows with the order (35 + 10 a requested item, at least STEP_CAP) and a cap at or past the
        task's own limit is none."""
        if not STEP_CAP:
            return 0
        cap = max(STEP_CAP, 35 + 10 * len(self.items)) if STEP_CAP_PER_ITEM else STEP_CAP
        if STEP_CAP_EXTEND and self.items and self.step >= cap:
            # (r9f) reached with a cart short of the order's items and steps of its own to spare: ten more steps, once
            short = len(set(self.cart_ids())) < len(self.items)
            if getattr(self, "cap_extended", False) or (short and self.left() > self.tail() + 10):
                if not getattr(self, "cap_extended", False):
                    self.cap_extended = True
                    self.trace.append({"kind": "cap_extended", "steps": self.step, "lines": len(set(self.cart_ids())), "items": len(self.items)})
                cap += 10
        return 0 if STEP_CAP_PER_ITEM and self.max_steps and cap >= self.max_steps else cap

    def page_read(self, ident: str) -> bool:
        """(r9) REF_PAGE_OPEN: whether a listing's page was read. With a page view apart from the attributes view only a
        read of the page view counts (its remarks carry a chain's pointers); otherwise any full read does, as before."""
        page = self.roles.get("offer")
        if not REF_PAGE_OPEN or not page or page == self.facts_tool():
            return ident in self.opened
        return ident in self.offer_seen

    def link_search(self, lost: list[str]) -> list[str]:
        """(r9f) REF_LINK_SEARCH: for a chain link nobody has found yet, Kyo searches once with the words of the trail
        model's own description of the listing it points at (its seller, kind and attributes; first eight words), then
        names the two unread rows sharing the most words with that description (two at least) for trail() to open with
        the page view. Once a requirement, twice an episode, only with steps to spare; no model call. Race 12: a query
        with the seller's name and two words of the target put the link first 77% of the time, top two 92%; chain
        completion CHIDORI 56% vs KYO 32% (research/race12/search_secrets.md s4)."""
        searched = getattr(self, "link_searched", None)
        if searched is None:
            searched = self.link_searched = set()
        cap = self.step_cap()
        if self.left() <= self.tail() + 6 or (cap and self.step > cap - 6) or self.cost > SPEND_HARD:
            return []
        for entry in lost:
            need, _, about = entry.partition(" -> find ")
            words = about.split()
            if need in searched or len(searched) >= 2 or len(words) < 2 or about == "the listing it points at":
                continue
            searched.add(need)
            query = self.value_search(" ".join(words[:8]), 2)
            want = _tokens(about)
            scored = []
            for ident in self.seen_order:
                if self.page_read(ident):
                    continue
                text = " ".join(str(v) for v in (self.rows.get(ident) or {}).values() if isinstance(v, str))
                shared = len(want & _tokens(text))
                if shared >= 2:
                    scored.append((-shared, self.seen_order.index(ident), ident))
            picks = [i for _, _, i in sorted(scored)[:2]]
            self.trace.append({"kind": "link_search", "need": need[:120], "query": query, "picks": picks})
            if picks:
                return picks
        return []

    def trail(self) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Kyo opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        (r5) TRAIL_RELEASE: a chain whose next page was not found in two tries, or that Kyo can follow no further (its
        four tries spent), is released: it holds nothing, leaves the status line, and the ledger tells the checks to judge
        the cart on the other requirements. Returns what it saw and whether to hold the checkout."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        spent = self.trails >= 4
        while pending and self.trails < 4 and self.cost < SPEND_HARD:
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record and not self.page_read(ident):
                    # (r9) read by the attributes view only: its page (and the remarks on it) is still unread
                    records.append("ATTRIBUTES ONLY, PAGE NOT READ " + _dump(_clip(record, 300))[:1400])
                elif record:
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
            for chain in _items(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                if chain.get("done") is True and links and all(self.page_read(i) for i in ids) and str(chain.get("value") or "").strip():
                    self.ref_values[need] = str(chain["value"]).strip()
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and not self.page_read(i)]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(l.get("about") or "") for l, i in zip(links, ids) if i not in self.rows), "") or "the listing it points at"
                    if not any(l.startswith(need + " -> ") for l in lost):
                        self.chain_lost[need] = self.chain_lost.get(need, 0) + 1     # (r5) one try per call
                    lost.append(f"{need} -> find {about}")
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and lost and REF_LINK_SEARCH and tool and target and not self.finished:
                reach = self.link_search(lost)
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
                if tool == self.roles.get("offer"):
                    self.offer_seen[ident] = self.step       # (r9) the page view was read
                seen.append(f"{ident}: " + self.show(reply.get("observation"))[:1500])
        text = ""
        if seen:
            text += "[Kyo opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        freed = [p for p in pending if p not in self.ref_values and (spent or self.chain_lost.get(p, 0) >= 2)] if TRAIL_RELEASE else []
        if freed:
            # (r5) TRAIL_RELEASE: these chains hold the order no longer
            self.released.update(freed)
            self.refs = [(n, need) for n, need in self.refs if need not in freed]
            self.version += 1
            lost = [l for l in lost if not any(l.startswith(f + " -> ") for f in freed)]
            self.trace.append({"kind": "trail_released", "needs": freed})
            text += ("[Kyo] Kyo's tries did not find the page that " + " | ".join(freed[:3]) + " points at, so it no longer holds"
                     " the order: the cart is judged on the other requirements.\n")
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
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
        """Listings that can replace sold-out cart lines (swap_hint comes here only for a line the store now shows sold out).
        (r5) SWAP_SURE: the model's search runs up to SPEND_HARD (was SPEND_SOFT) and, when its call fails, once more with
        the light model, more room and only the opened candidates; a pick the records rule out is dropped; opened
        candidates a check passed, or whose attributes show every plain requirement, fill in and come first (in price
        order under a cheapest pick)."""
        if not SWAP_SURE and self.cost > SPEND_SOFT:
            return []
        pool = self.swap_pool(lost) if SWAP_SURE else []
        known = [c for c, number in pool if self.judged(c) is True or self.shows_all(c, number)]
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        got = None
        if self.cost <= (SPEND_HARD if SWAP_SURE else SPEND_SOFT):
            got = self.ask_json(self.swap_prompt(lost, records), [HEAVY, LIGHT], 1200)
            if got is None and SWAP_SURE and self.cost <= SPEND_HARD and not (self.no_credit or self.broke) and self.calls < ALL_CALLS:
                narrow = [_dump(_clip(self.full.get(c) or self.rows.get(c), 200))[:1200] for c, _ in pool] or records
                got = self.ask_json(self.swap_prompt(lost, narrow), [LIGHT], 2400)
                self.trace.append({"kind": "swap_retry", "records": len(narrow), "ok": got is not None})
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        if SWAP_SURE:
            fits = [(p, i) for p, i in fits if self.swap_fit(i, lost)]
        # sorted here: the model names fits well but orders them by price unreliably
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        if known:
            named = {i: p for p, i in fits}
            if self.brief.get("pick") == "cheapest":
                ranked = sorted(dict.fromkeys(ranked + known), key=lambda i: named.get(i, self.price_of(i) if self.price_of(i) is not None else float("inf")))
            else:
                ranked = list(dict.fromkeys(known + ranked))
        if RESELECT:
            ranked = self.rerank_swaps(ranked, pool)
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits, **({"known": known} if SWAP_SURE else {})})
        return ranked[:4]

    def rerank_swaps(self, ranked: list[str], pool: list[tuple[str, int]]) -> list[str]:
        """(r7) RESELECT R3: the replacements the model named and the opened fits of the sold-out line's item (swap_pool),
        ranked by the code: fewest word requirements of the item its record contradicts (another value, or the words out
        of order; a record that says nothing on one is not counted against it), then the price under a cheapest pick,
        then a passed judgment first; ties keep the model's order (race 12
        t-2729d4908b: KYO and v12, 4 runs of 4, swapped to a 1174 listing past an opened 688 one whose title stated the
        colour)."""
        numbers = dict(pool)
        cands = list(dict.fromkeys(ranked + [c for c, _ in pool]))

        def key(ident: str) -> tuple[int, float, bool]:
            number = numbers.get(ident)
            if number is None:
                hits = self.item_of(ident, True)
                number = hits[0] if hits else None
            specs = [s for s in (self.word_need(number, m) for m in self.live_needs(number)) if s] if number is not None else []
            miss = sum(1 for p, v in specs if self.states(ident, p, v) in ("no", "scattered"))
            price = self.price_of(ident) if self.price_of(ident) is not None else float("inf")
            return miss, price if self.brief.get("pick") == "cheapest" else 0.0, self.judged(ident) is not True

        out = sorted(cands, key=key)
        if out[:1] != ranked[:1]:
            self.trace.append({"kind": "swap_rerank", "before": ranked[:4], "after": out[:4]})
        return out

    def swap_prompt(self, lost: list[str], records: list[str]) -> str:
        prompt = SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        return prompt + "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)

    def swap_pool(self, lost: list[str]) -> list[tuple[str, int]]:
        """(r5) Opened listings that can fill a sold-out line, with the item each fills: on that line's shelf, buyable, no
        plain requirement contradicted, not judged unfit (fits_for), inside the budget with the cart's other lines."""
        cart = [i for i in self.cart_ids() if i not in lost and not self.sold_out(i)]
        keep = {i: self.cart_lines.get(i, 1) for i in cart}
        cat_key = self.category_key()
        skip = set(lost) | set(cart) | set(self.gone)
        pool: list[tuple[str, int]] = []
        for line in lost:
            label = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower() if cat_key else ""
            for number in self.item_of(line, True):
                for c in self.fits_for(number, skip | {p for p, _ in pool}, label, True):
                    if self.within({**keep, c: self.items[number].qty}) is not False:
                        pool.append((c, number))
        return pool[:8]

    def shows_all(self, ident: str, number: int) -> bool:
        """(r5) The listing's opened attributes state every requirement of item `number`: each is a short word value (no
        figure) that an attribute naming its property holds in full. A figure or a long value (a limit, a value read from
        another listing) is left to the model checks, so such an item never passes here."""
        block = _fact_block(self.full.get(ident)) or {}
        if not block or not 0 <= number < len(self.items):
            return False
        item = self.items[number]
        needs = [m for m in item.musts + item.told if m not in item.dropped]
        if not needs:
            return False
        for need in needs:
            prop, _, value = str(need).partition(":")
            words = re.findall(r"[a-z]+", value.lower())
            if not prop.strip() or not words or len(words) > 3 or re.search(r"\d", value):
                return False
            keys = [k for k in block if _names(_stem(prop), _stem(k))]
            want = _tokens(" ".join(words))
            if not any(want <= _tokens(str(block[k])) for k in keys):
                return False
        return True

    def swap_fit(self, ident: str, lost: list[str]) -> bool:
        """(r5) A replacement the model named that the records do not rule out: not sold out, not judged unfit, no plain
        requirement of the lost line's item contradicted, inside the budget with the cart's other lines."""
        if self.sold_out(ident) or self.judged(ident) is False:
            return False
        cart = [i for i in self.cart_ids() if i not in lost and not self.sold_out(i)]
        keep = {i: self.cart_lines.get(i, 1) for i in cart}
        for number in sorted({n for line in lost for n in self.item_of(line, True)}):
            if self.clash(ident, number) or self.within({**keep, ident: self.items[number].qty}) is False:
                return False
        return True

    def audit(self) -> str:
        # (y2b) the review before a decline is guaranteed once: only a spend past the hard cap (plus one call) skips it
        if self.audited or self.cost > (_late_limit() if AUDIT_ALWAYS else SPEND_SOFT):
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
        if not named and DECLINE_SEARCH and self.decline_search():
            got = self.audit_call()
            named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def rejudge_table(self) -> str:
        """(r12) REJUDGE_ALL: every opened, buyable listing of each item checked by the code against all the item's stated
        word requirements now in force (word_need: no figures, no references): those that state every value (cheapest
        first under a cheapest pick, cart lines marked) and those that state another value. Items with fewer than two such
        listings or no word requirement are left out. "" when nothing is left; facts only, no instruction."""
        self.rejudged_at = self.version
        parts = []
        cart = set(self.cart_ids())
        for number, item in enumerate(self.items):
            specs = [x for x in (self.word_need(number, n) for n in self.live_needs(number)) if x]
            if not specs:
                continue
            pool = [i for i in self.seen_order if i in self.full and self.buyable(i) and not self.sold_out(i) and number in self.item_of(i, True)]
            if len(pool) < 2:
                continue
            fit, other = [], []
            for i in pool:
                said = [(prop, value, self.states(i, prop, value)) for prop, value in specs]
                bad = [(prop, value) for prop, value, st in said if st == "no"]
                if bad:
                    other.append(f"{i} ({bad[0][0]}: {self.prop_value(i, bad[0][0]) or 'another value'})")
                elif all(st == "yes" for _, _, st in said):
                    fit.append(i)
            if not fit and not other:
                continue
            if self.brief.get("pick") == "cheapest":
                fit.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            line = f"item {number + 1} ({item.name}; " + ", ".join(f"{p}: {v}" for p, v in specs)[:160] + "): "
            line += ("state them all: " + ", ".join(f"{i} (price {self.price_of(i)}{', in the cart' if i in cart else ''})" for i in fit[:4])
                     if fit else "no opened listing states them all")
            if other:
                line += "; state another value: " + ", ".join(other[:4])
            parts.append(line)
        if not parts:
            return ""
        self.trace.append({"kind": "rejudge", "version": self.version, "lines": [x[:160] for x in parts]})
        return "[Kyo rechecked every opened listing against the requirements now in force] " + " | ".join(parts)[:900]

    def decline_search(self) -> bool:
        """(r11) DECLINE_SEARCH: a stated requirement ("property: value", one to four words, figures allowed) of an item
        that no opened, buyable listing of the item states is searched once as "<value> <store word for the item>" and the
        rows that hold the value (else the cheapest new rows of the item) are read, two at most. At most two searches an
        episode, only with steps free beyond tail() and below the step cap and spend under SPEND_HARD; no model call.
        Returns True when it read a listing."""
        before = set(self.full)
        done = 0
        for number in range(len(self.items)):
            for need in self.live_needs(number):
                if done >= 2:
                    break
                if need in self.cross or need in self.released or any(need == n for _, n in self.refs):
                    continue
                prop, sep, value = str(need).partition(":")
                prop, value = prop.strip(), value.strip()
                words = _seq(value)
                if not sep or not prop or not 1 <= len(words) <= 4:
                    continue
                opened = [i for i in self.full if number in self.item_of(i, True) and self.buyable(i)]
                if any(self.states(i, prop, value) == "yes" for i in opened):
                    continue
                if not (self.left() > self.tail() + 6 and not (self.step_cap() and self.step > self.step_cap() - 6)
                        and not self.finished and not self.broke and self.cost <= SPEND_HARD):
                    return len(self.full) > len(before)
                done += 1
                noun = self.shelf_word(number, set(words))
                query = self.value_search(f"{value} {noun or self.items[number].name}", len(value.split()))
                if not query:
                    continue
                picks = self.value_rows(number, value) or [i for i in self.value_rows_any(number) if i not in before][:2]
                self.fill_facts(picks[:2], self.tail() + 3)
                self.trace.append({"kind": "decline_search", "need": need[:120], "query": query, "read": picks[:2]})
        return len(self.full) > len(before)

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
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1500)
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
        if self.compare_bad:
            # (y1) a failed list outranks the schema's limit: races 5, 7, 8 and 9 all stated a larger limit than worked
            # (10 for 6, 10 for 5, 10 for 5, 4 for 2); race 9 official: 636 of KYO's 943 side-by-side calls failed
            learned = max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
            return min(learned, spec["maxItems"]) if isinstance(spec.get("maxItems"), int) else learned
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
            if name == self.roles.get("search"):
                self.search_shapes.append(dict(args))
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
            if self.is_refuse(name, args) and not self.finished:
                # (y2b) the decline guard: a line lost to a refusal or a removal that an opened listing fills, then an
                # item whose shelf has no listing opened in full; each fires once per state and stays silent when the
                # records show the decline may be right
                guard = self.refusal_hold() or self.unopened_hold() or self.chain_hold()
                if guard:
                    out.append((cid, guard + "\nCart now: " + self.cart_text() + (" [episode finished]" if self.finished else "")))
                    continue
            if self.is_buy(name, args) and not self.finished:
                stop = self.buy_hold()
                if stop:
                    out.append((cid, stop + "\nCart now: " + self.cart_text()))
                    continue
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    before = self.version
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
                    gate = self.struct_gate()
                    if gate:
                        out.append((cid, "Not run yet (no step used). " + gate + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows() + self.cheaper_scan() + self.backup_rows()
                    exact = self.exact_hold() if RESELECT else ""
                    if exact:
                        # (r7) R2: held before the model check, which the swap would make run again
                        seen = ("Kyo read these listings (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, exact.replace("Not run yet (no step used). ", "Not run yet. " + seen, 1) + "\nCart now: " + self.cart_text()))
                        continue
                    if NOTE_HOLD and self.version != before and self.queued and self.left() > self.tail() + 2:
                        # (y2b) a note seen on Kyo's own reads changed the request: the actor re-plans the cart with it
                        change = " | ".join(self.queued)
                        self.queued = []
                        self.trace.append({"kind": "note_hold", "change": change[:200]})
                        seen = ("Kyo read these listings (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + seen + "A note seen on those reads changed the request: " + change + "\nCart now: " + self.cart_text() + "\nApply the change (a relaxed requirement can make a cheaper opened listing the right pick), fix the cart if needed, then check out."))
                        continue
                    # (y2a) the checkout's own check runs up to SPEND_HARD (y1's lower SPEND_SOFT would skip it more often)
                    verdict = self.check_cart(limit=SPEND_HARD)
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
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
                before = list(self.cart_lines)      # (y2b) lines this call takes off
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
                taken = [i for i in before if i not in self.cart_lines] + [v for v in args.values() if isinstance(v, str) and v in self.dropped and v not in self.cart_lines]
                if taken:
                    self.removed_at, self.removed_lines = self.step, list(dict.fromkeys(taken))
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
                    if self.is_buy(name, args) and not reply.get("finished"):
                        self.note_refusal = (self.cart_at, meaning)
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if REJUDGE_ALL and self.version != self.rejudged_at:
                table = self.rejudge_table()
                if table:
                    text += "\n" + table
            if self.queued:
                text += "\n[notes seen on Kyo's own calls] " + " | ".join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text += " [episode finished]"
            out.append((cid, text))
        return out

    def cart_reply_empty(self) -> bool:
        """(r5) The last cart call's reply holds an empty list (its lines) and names no listing seen."""
        try:
            reply = json.loads(self.cart_said) if self.cart_said else None
        except ValueError:
            return False
        empty = False
        stack: list[tuple[Any, int]] = [(reply, 0)]
        while stack:
            node, depth = stack.pop()
            if isinstance(node, list):
                empty = empty or not node
                stack += [(v, depth + 1) for v in node if depth < 5]
            elif isinstance(node, dict):
                stack += [(v, depth + 1) for v in node.values() if depth < 5]
        return empty and not any(ident in self.cart_said for ident in self.rows)

    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if EMPTY_FINISH and buy and not holding and self.cart_touched and not fresh and self.left() >= 2 and self.cart_reply_empty():
            # (r5) Kyo's record and the last cart reply both show the cart empty: a checkout would only be refused
            self.trace.append({"kind": "empty_finish", "again": again})
            if again:
                self.buys += 1
                self.refused_at = self.step
                return "found the cart empty (the last cart reply listed no line), so no checkout was sent"
            buy = None
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
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
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        # (r6) stopped on the spend or the step cap (not on a finish, the call cap, the clock or a broken model)
        self.capped = not self.finished and not self.broke and (self.cost >= SPEND_HARD or bool(self.step_cap() and self.step >= self.step_cap()))
        if self.capped:
            self.trace.append({"kind": "capped", "steps": self.step, "cost": round(self.cost, 5)})
        if not self.finished:
            # (y2b) the loop stopped on a limit with a cart that misses items: checked picks for every item are ordered first
            placed = self.loop_end_order()
            turned = "" if self.finished or placed else self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if (turned or placed) and not self.finished and not self.broke and not self.no_credit and self.left() > 1 \
                    and not (CAP_LEAKS and self.capped):
                said = ("[Kyo] " + placed) if placed else ("[Kyo] Kyo tried to end the episode and " + turned)
                self.chat.append({"role": "user", "content": said + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD if CAP_LEAKS else SPEND_HARD + 0.01)
            if not self.finished:
                self.loop_end_order()
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap \
                and time.monotonic() - self.t0 < CLOCK_LOOP and not (self.step_cap() and self.step >= self.step_cap()):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            if NUDGE_ORDER and not self.nudged and self.items and not self.cart_lines and self.buys == 0 \
                    and self.left() <= max(8, self.max_steps // 5) + self.tail():
                # (y2a) once: few steps left and nothing carted yet
                self.nudged = True
                self.chat.append({"role": "user", "content": "[Kyo] Steps are running low and the cart is still empty: put your best "
                                  "verified pick for every requested item in the cart now, at its quantity, and check out.\n" + self.status()})
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
