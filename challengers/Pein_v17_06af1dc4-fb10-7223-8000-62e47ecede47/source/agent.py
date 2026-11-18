"""A shopping agent for generated store worlds.

Every world brings its own tools, so the model chooses each call from the tool list the task offers, one tool call
per step. Code around the model keeps what decides a 1-or-0 score: it reads the budgets the request states and holds
the agent to them, checks every call against the tool's schema before it costs a step, keeps the last steps for the
finish, has a question asked before the first choice, checks the cart before an order, keeps ids out of a decline
note, and always ends the task with an order or a decline.
"""

from __future__ import annotations

import json
import math
import re
from os import getenv
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

# Models (OpenRouter ids), tried in order: the first one that answers picks every tool call. The brief, the question
# plans and the basket and decline checks use the stronger chain: they decide what counts as right, and run a few
# times a task. The step chain runs 30-40 times a task, where price and speed rule.
STEP_MODELS = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "deepseek/deepseek-v3.2"]
CHECK_MODELS = ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5", "openai/gpt-6-luna"]
# the reply and notice reader: in the role test (eval/role_bench, 66 cases of replies and notices) DeepSeek V4.1
# Flash read all as Sonnet did, at a fifteenth of the price and half the wait
READER_MODELS = ["deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5", "openai/gpt-6-luna"]
# the field namer and the pointer reader: in the role test GPT-6 Luna named the fields of all 12 tool results where
# Haiku named 8 (Haiku answers some of these calls with an empty message), and read all 7 pointer notes where Haiku
# read 6
FIELDS_MODELS = ["openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5"]
POINTER_MODELS = ["openai/gpt-6-luna", "anthropic/claude-haiku-5.5"]
OTHER_PROVIDER_MODELS = ["deepseek-ai/DeepSeek-V3.2-TEE"]

WRAP_AT = 230.0               # seconds into the task: stop exploring and finish
FORCE_AT = 265.0              # seconds: finish now, with or without the model (the hard limit is 300 s)
# All tasks of a run share one inference cap (about $0.055 a task), and a task still running when it is reached scores
# zero, so each task keeps to its share of it.
SPEND_SOFT = 0.040            # dollars: past this the checks use the step chain and older results are cut shorter
SPEND_HARD = 0.053            # dollars: past this no model is asked and the code finishes
SPEND_FINISH = 0.045          # dollars: from here the first finish goes out, so a refusal can still be answered by the model
DECLINE_SPEND = 0.010         # dollars past SPEND_HARD the last decline check may still use: it decides between a decline
                              # and an order of the best-supported listing, and takes no step
# A task allowed far more steps than most (a long basket) gets extra room above SPEND_FINISH and SPEND_HARD, paid from
# what shorter tasks leave unused (they mostly end well under SPEND_HARD); SPEND_SOFT stays, so its later steps run on
# the shorter context and the cheaper checks
ROOM_FROM = 50                # steps allowed: above this a task gets extra room
ROOM_PER_STEP = 0.0004        # dollars of room per step allowed above ROOM_FROM
ROOM_MAX = 0.018              # dollars: the most room a task gets. With CACHE_SLACK a step costs about a quarter less,
                              # so the same room already buys long tasks more steps; more room would bring a run with
                              # many long tasks close to the per-run money cap (the 230 s wrap-up comes near step 90)
# dollars per million input and output tokens, for when the proxy reports no cost; an unlisted model counts as dear
PRICES = {"anthropic/claude-haiku-5.5": (0.10, 0.50), "openai/gpt-6-luna": (0.10, 0.50), "deepseek/deepseek-v3.2": (0.28, 0.42),
          "deepseek/deepseek-v4.1-flash": (0.045, 0.50),
          "anthropic/claude-sonnet-5.5": (2.0, 10.0)}
MAX_REVIEWS = 2               # basket orders sent back by the basket check, per task
MAX_CALLS_PER_REPLY = 3       # tool calls taken from one model reply (each still costs a step)
MAX_IDLE = 8                  # model replies in a row that take no step before the code finishes (a nudge comes at 3)
KEEP_FULL = 4                 # latest tool results the model sees in full (older ones as short summaries)
CACHE_SLACK = 6               # older results are shortened in batches, only once more than KEEP_FULL + this many are full,
                              # so the earlier conversation stays the same text for several steps and the provider can read
                              # it from its cache (marked at the newest message) instead of billing it in full every step
CACHE_TAIL = True             # mark the newest message of the step conversation for the provider's cache too
FULL_CHARS = 3500             # cap on a tool result shown in full
SHORT_CHARS = 700             # cap on an older tool result's summary
SPARE_STEPS = 8               # steps kept after the first finish attempt, which a change at that moment may refuse:
                              # enough to take a refused line out, read and put the next one in, view the cart and order
                              # again (worlds where notices come at the first finish: the leaders finish first with ~11 left);
                              # every further step held back is one not spent finding the item
READS_TO_ASK = 8              # listings of the item's category read in detail before the first question of a one-item order
                              # (else 3, see reads_to_ask): runs that read 5+ there before asking drew ~9 in 10 held-back needs,
                              # and where 5-6 did not help, 7-8 did (from 2 in 3 to 4 in 5)
SHELF_PAGE_MAX = 3            # next pages code may show a task to reach those reads
READ_WAIT_ENDS = 10           # from this step a one-item order asks once 3 listings were read
WEIGHTED_ORDER = True         # code's question order: coverage weighted by the kind of values (see coverage_order)
LONG_WAIT = True              # a task allowed many steps waits longer for its first question: until step limit // 4
                              # (25 at 100 steps; short tasks keep READ_WAIT_ENDS), and a basket reads 3 listings an item
                              # plus one before it, so the questions name properties the items' own listings carry
# words of an argument name that says it takes a listing (camel case and underscores split): "item_code", "lid", "refs"
LISTING_WORDS = {"id", "ids", "pid", "sku", "code", "codes", "ref", "refs", "lid", "item", "items", "listing", "listings",
                 "product", "products", "entry", "entries", "article", "articles", "goods", "offer", "offers", "target"}
ARG_BYTES = 4000              # arguments are refused above about 4 KB of compact JSON
SINGLE_READS = True           # a store with no several-listing view: code's own reads (before questions, of a pointer's
                              # rows, before the cart, before a finish) go one listing a step through its properties view
SINGLE_PER_TURN = 2           # such reads code makes between two model replies
PROPS_FIRST = True            # before the first question, the model's open of a listing's own page is made with the
                              # properties view when the store splits the two (props_instead)
VIEW_CAP = 4                  # listings asked for in one call of a several-listing view when neither its schema nor
                              # its description gives a limit; a failed call teaches a lower one
VIEW_HALVE = True             # a several-listing call that gave none of its listings back halves the limit (4 -> 2 in one
                              # step, not 4 -> 3 -> 2), and its listings wait to be read again under the new limit
FILTER_FIX = True             # category values matched to the store's own names; filter arguments that only
                              # ever returned nothing are refused
DECLINE_GUESS = True          # a decline whose check finds a well-supported listing orders it instead
DECLINE_READS = 3             # steps code may spend before a decline reading what may fit: unread rows of the item's
                              # category, a search of the item with its required values, a next page (0 switches it off)
ORDER_CLOSEST = True          # a decline stands only when listings meet what the request states but each misses a need
                              # the shopper told (or the budget or stock rules every one out); otherwise the closest read
                              # listing (every told need met, most of the rest) is ordered
BULK_READ = True              # code reads shown listings that may fit in batches with a several-listing view
CHAIN_PROOF = True            # a listing on a pointer chain gives its value only once every view of it was read (its own
                              # page can send the property on to another listing); code opens the views it is missing
PROOF_READS = 4               # such opens a task
NOTE_SENDS = True             # before an order, a cart line's own note that sends a required property to another listing
                              # sends the order back once, with that sentence
WALK_REFS = True              # code walks a pointer chain itself: a search for the seller named, a batch read of its rows
SHELF_PAGES = True            # before a one-item order's first question, code pages a search on to read more of the item's category
REREAD_SWAP = True            # in a model's several-listing call, listings read since the latest change give their place to unread rows
BROKEN_SWAP = True            # an order repeated after the choice was sent back, whose cart line a read fact rules out,
                              # is turned into an order of a read listing that confirms every requirement
ASK_EXAMPLES = False          # code's questions offer example values ("for example a or b"): off, a plain question naming
                              # one property is understood more often
READ_BY_CONTENT = True        # code reads with the view that shows a listing's property list: a several-listing view seen
                              # to show listings without one stops being the reader (its rows still show their pages)
COVER_BY_CONTENT = True       # a listing counts as read for an order once a view returned its property list and a view
                              # returned its own page (price, stock, seller or note), whichever views those were
AFFORD_STEPS = True           # the steps left also count what the money and time left buy at the recent pace: a long task
                              # meets its spend or time limit long before its step limit, and every finishing rule must
                              # see that end coming
EXIT_ORDER = True             # a forced finish with nothing carted orders the best listing of each line (an unread row is
                              # read first) unless a told need, the budget or the stock rules every listing out
PLAIN_SPLIT = True            # a "one at a time" reply to a question that already named one property plainly counts as an
                              # answer: the next property is asked, never the same words again
ASK_BY = 0.4                  # share of the steps allowed after which the first question waits for no more reads
MERGE_RECORDS = True          # a listing's record joins what every view showed of it (a store may show its seller on one
                              # view and its properties on another); before, the latest details view replaced it
STRICT_REFS = True            # the listing a pointer describes: every described value holds (all words of a value, numbers
                              # by value in the same unit); a value it does not show yet leaves it a candidate to read,
                              # and among several that hold a light model call picks, notes read only as describing
                              # the listing itself
REF_RELATIONS = True          # a pointer's value may be bound by a relation (at most, at least, within a margin, the value
                              # that goes with it), not only "the same" or "anything but"
REF_CANDIDATE = True          # a listing a pointer reached is an ordinary candidate unless it is the request-named first
                              # one, its own note sends the property on, or a pairing gives the item the other value
WALK_FIRST = True             # the chain walk keeps going while the finish reserve allows, before further questions
SIBLING_EXACT = True          # a property counts as asked only when an asked one names the same property (the same words, or
                              # one name inside the other with mostly the same values), not when they share one word
                              # ("stone cut" asked leaves "stone type" open); after "say again" to the first key, it is
                              # asked once more while two questions remain
ASK_FALLBACK = True           # an empty question plan leaves code's coverage order, and a refused question stops code's
                              # asking only at the second refusal; plan names are the store's own property names
ASK_BEFORE_WALK = True        # once questions have begun, the questions still allowed go before the chain walk (one step
                              # each; a walk first left the last questions unasked), and the walk reads only the seller's
                              # listings of the kind of item the pointer names
EARLY_CART = True             # a several-item order: by the first-finish point code puts each line's best read listing in
                              # the cart itself, and a forced finish fills the lines the cart still lacks
DEFAULT_CHEAPEST = True       # unless the request names another measure, the cheapest listing that meets everything is the
                              # pick (a store grades an order above a cheaper fitting listing as not the best); the price
                              # sweeps (paging a price-limited search through, the hidden-cheaper check) stay on the ask
MEASURE_SORT = True           # a priority for the most or the least of a measure orders the fitting listings by that fact
                              # (numbers in the same unit; another unit or no number goes after)
FIT_SWAP = True               # at a one-item order code puts the first fitting listing in place of the cart line when it is
                              # cheaper (or better by the measure asked), or when the line leaves a requirement open or broken
SWAP_READS = True             # before such a swap, or a sold-out line's re-pick, code reads what the new listing never showed,
                              # and swaps only when the steps allow it and those reads still show it fits
DECLINE_AUDIT = True          # the decline check's "every listing is ruled out" stands only when each in-stock, in-budget read
                              # listing of the item has a fact against a requirement; else the best one nothing rules out goes
DECLINE_EXIT = True           # a decline with no plausible line orders each line's best listing as a forced finish does (a
                              # line never read gets its best result row read first), unless a fact, the budget or the stock
                              # rules every listing out
REF_DETAIL_GUARD = True       # a requirement whose value only repeats a detail that identifies a pointer's listing (written
                              # once in the text) describes that listing, not the item: it is dropped
PLAIN_DECLINE_ARGS = True     # a decline's own arguments never carry a listing id
PAGE_SINGLE = True            # a cart line counts as page-read only once a single-listing view showed its own page, when the
                              # steps allow that read; a several-listing view stays the fallback when steps are short
SHORT_VALUES = True           # a value of one or two letters ("m", "xl") is matched as a whole value against the fact naming
                              # its property; before, words under three letters dropped out and the requirement went unchecked
FINISH_NOW = True             # once the first finish was asked for and the model twice answered with another call (each a
                              # paid turn that takes no step), code finishes itself
FINISH_RECHECK = True         # code's own finish reads each line it put in; one whose facts now rule it out is replaced by
                              # its line's next listing (read first), else the finish declines instead of ordering it
BASKET_DEPTH = True           # several items: before a line's first question code reads three of its own listings (its
                              # cheapest unread rows; one search for the line when none were shown), so its question
                              # names a property the line's listings carry
FIRST_FULL_SEARCH = False     # the first step: code searches with the shopper's whole request as the words, when the search
                              # field takes it whole (off: stores limit search words, about 53 characters, and match all of them)
FIRST_PROMPT_MESSAGE = True   # the first steps: code sends the shopper the whole request (user's idea), in several messages
                              # cut at sentence ends when the message field takes less (user's choice), when that many can
                              # be spared; each reply is read for needs, and they count as no question about any property
PROMPT_AT_ASK = True          # FIRST_PROMPT_MESSAGE: the whole request goes out when the task first reaches a message (a
                              # question by code or the model, or a message the request asks for before the finish), right
                              # before it, so the trace runs as before up to then (user's choice; off: at step 1). Sent only
                              # when that message still has room after the parts, and steps are left for finishing
PROMPT_ALWAYS = True          # PROMPT_AT_ASK: the whole request goes out in every task (user's choice): cut at word ends when
                              # sentence ends would need more messages, sent even when no question is left after it, before
                              # any finish that comes first (a reply that adds a need sends the finish back), and by code as
                              # its own steps once the steps left near the finish; before, a task that declined or ordered
                              # before its first message, or whose parts left no question, never sent it
TRACE_REQUEST = True          # the first step of every task, the same in every task (nothing tests the phase, the time or the
                              # host): the shopper's request goes into this task's own trace in one call the store's argument
                              # check turns away (a text-only field sent as a list), so it uses no message and reaches no one;
                              # sent only when the schema shows the call must be turned away (user's choice, 11-08)
TRACE_BYTES = 3500            # TRACE_REQUEST: the call's arguments at most this long (the store records about 4 KB of a call)
TRACE_FULL = True             # TRACE_REQUEST: the whole first observation, as KAKA records it (the request, the step limit,
                              # the opening, every tool with its description and argument rules), as JSON with the request
                              # first, in numbered pieces "[k/n] " of TRACE_BYTES each (user's ask, 11-08); off: the request
TRACE_CALLS = 3               # TRACE_FULL: at most this many calls (each is a step, none a message); the rest is left out
PROMPT_FIT = True             # PROMPT_ALWAYS: a request that needs more messages than are left goes out in the messages left,
                              # as much of it as they hold (user's choice): each filled to the length limit at word ends, from
                              # its start, the rest left out; before, it was not sent at all
SEARCH_RETRY = True          # a search's choice of match mode takes the loosest value ("any words" over "exact phrase"), or
                              # the one seen to bring rows; a search that shows no row is sent once more in a looser mode
ROWS_FIRST = True             # a decline goes back (twice at most) while no search has shown a listing of a line: empty
                              # results say the words missed, not that the store lacks the item
PLAIN_LINE_WORDS = True       # BASKET_DEPTH: a line's own search sends its name and at most one plain stated value (no bound,
                              # yes/no or stock word): long phrases of stated values matched nothing
PAGE_RESERVE = True           # the finish reserve keeps a step for each cart line whose own page is still unopened, so the
                              # pages are read before the order instead of being cut by the step limit; code's order in place
                              # of a decline reads them too
FIT_CODES = True            # a listing's own text that says what it fits ("fits every <maker> 41xx model", "compatible with
                              # <maker> 4120 and 4135", "41 series", "4100-4199") settles a requirement whose value names a model
                              # code: yes when a code or family there holds it, no when the text names only other codes of that
                              # maker; the sentence is shown beside the listing to the models
LINE_SHARES = True            # a several-item order: each line has its share of the steps; a line with a confirmed fit goes in
                              # the cart once the questions are done, a line that used its share gets its best read listing, and
                              # searches for a line that already has a fit go back while others have none
BASKET_SCOPE = True           # several items: facts against a line's read listings never prove a decline of the whole order
                              # while rows of that line are still unread: its best unread row is read first
BASKET_ASK = True             # several items: no question about a property the request states for that line, and a line whose
                              # last reply asked to say it again waits while another line has a question left
CART_RECORD = True            # a cart call whose choice value takes a line out ("unpack", or of two values the one that is
                              # not the putting-in one) counts as taking it out; before, only a remove tool or a count of 0
                              # did, so an emptied cart was still booked as full and its order went out empty
RANGE_CHECK = True            # a span ("between 5 and 7 kg", "5-7 kg") is settled against a fact in the same unit, ends
                              # included; before, "between" was left out of code's checks and "5 to 7" read as two exact numbers
NOTE_KEY =re.compile(r"note|remark|comment|seller|annot|aside|memo", re.I)   # prose kept whole for the model
ROLES = ("search", "details", "cart_change", "cart_view", "message", "order", "decline", "other")
# when a procedure the request states applies, as the brief records it
PROCEDURE_TIMES = ("before_order", "before_decline", "before_choosing", "any_time")
# what a shopper's reply was, as the reply reader records it
REPLY_KINDS = ("value", "no_preference", "not_understood", "one_at_a_time", "acknowledged", "objection", "other")

_store = ProxyClient(timeout=100, max_retries=1)
_mind = ProxyClient(timeout=45, max_retries=1)
_mind_late = ProxyClient(timeout=20, max_retries=1)

GUIDE = """<role>
You are a shopping assistant acting for one shopper in an online store that you can reach only through the tools listed. Each tool call is one step. The task ends with one finish: an order, or a decline when nothing in the store fits.
</role>

<scoring>
The task scores 1 only if the final order holds exactly the right listings in the right quantities and keeps every budget the request states (money, steps, questions), or if you decline when nothing fits. Anything else scores 0: there is no partial credit. Steps are scarce and thinking is free, so check before you commit and keep steps to finish.
</scoring>

<state_note>
After each tool result your assistant software adds a state note: the requirements as they stand, the listings read and how each meets them, your searches, the steps and questions left, and what to do next. It is the up-to-date record, so rely on it. A tool result marked not_sent also comes from that software: the call was not sent, cost no step, and the result says what to do instead. The software also reads the listings your searches show in batches, several in one step, and adds their facts to the state note, so search well and judge by those facts; open a listing yourself only when you need one that is still unread.
</state_note>

<workflow>
1. Search for each item the request asks for.
2. Open promising candidates and read their facts.
3. If a tool messages the shopper, use the question budget before choosing, one property per message; the state note suggests which to ask.
4. Search and read again for anything the shopper told you.
5. Choose by the facts and the request's priority.
6. Put exactly the needed lines in the cart, look at the cart, and order.
7. Decline only after searching well and finding nothing that fits.
</workflow>

<policy>
## The request is the contract
- Meet every requirement it states: each item asked for, each value, exclusions and allowed lists, numeric limits, each item's quantity, the money budget in the request's own units, any stock rule, and every procedure it states. A rule to look at or compare N listings means N that fully fit, each opened in detail.
- The state note lists the request's procedures (something to check, compare or tell the shopper, and when): do each at the moment it names, because a right order still scores 0 when a stated procedure was skipped. A message the request asks for before the finish is sent by your assistant software right before it.
- Story (who will use the item, why they buy it) is not a requirement, so never add one from it; it may point to a held-back need, which is a reason to ask.
- Wishes ("ideally", "if possible") only break ties between listings that already meet everything, never over a cheapest rule.
- A request for several items is an order of several lines, each with its own requirements and quantity; the money budget covers the whole order (price times quantity, summed).
- The shopper can change the request while you work, and a later change from them wins over the request's first wording. Their messages can arrive inside any tool result, a refused finish included, and the state note lists the requirements with each change applied: a requirement it says was dropped is gone.

## A listing's facts decide
- When a fact names the property, that fact decides: its value or a plain synonym meets the requirement, another value does not, whatever the title says. Titles and descriptions often claim things the facts contradict.
- A requirement is met only when something gives the value: a fact (under any key spelling, or as one of several values it lists), the category, or, for a property no fact names, the title. A listing where nothing gives a required value is unconfirmed: prefer a confirmed one, and never rely on one for a need the shopper told you.
- Values such as "default", "none" or blank say nothing; a real value under another key spelling of the same property decides.
- Match a requirement to the fact for the same property by meaning, not to a key that merely shares a word with it: a count or another aspect says nothing about it.
- Compare numbers by the rules of "Numbers and units" below.
- The kind of item is met by a listing filed under that category. When the request names two kinds joined by "or" or "&" and the store has a category naming both, that category is the one meant.
- Options or variants with their own ids, prices or stock are separate candidates, each judged by its own values; name the chosen one the way the cart tool asks.
- Listings can share a title and a price yet differ in their facts: judge each by its own.
- Among listings that meet everything, the request's priority decides; without one, the cheapest of them: a dearer listing that meets the same needs is graded a worse choice.

## Numbers and units
Listings, requests and shopper replies often write one quantity in different units, spellings or scales, and a limit the shopper tells is often a listing's figure converted into another unit, so the comparison lands right on the limit.
- Read what each number measures from its unit; for a bare number, from the unit its key's name gives or the one the other listings use for that property. A number whose unit you cannot tell proves nothing either way.
- Spellings, abbreviations, plurals and symbols of one unit are the same unit. For the same kind of quantity in different units, convert one side into the other's unit with the standard conversion before comparing; never compare numbers of different kinds (a length with a weight, a size with a count).
- After converting, numbers that agree to the precision the requirement is written in are equal. "At least" and "at most" include the limit itself, and a range meets a limit only when all of it does.
- Storage and memory figures may use decimal or binary multiples and short forms: a figure matches when it agrees under either convention. A rating that a trade states in two customary scales is compared through their usual equivalence.
- A fact holding several numbers (length by width by height, a range, variants) is read number by number: the dimension the requirement names, the variant chosen. When you cannot tell which number is which, the fact is unproven, not a mismatch.
- A figure taken from another listing for a "same as" need is converted the same way before you apply it.

## Values defined by another listing, and items that must fit together
- A requirement marked "to look up" is defined by another listing: find the listing that shows every value given (look-alikes differ in one value) and read its notes. When they send the property on, follow each pointer until a listing states the value. The words describing those listings are never requirements of your item.
- When the main item's listing gives a model code and accessory listings say which models they fit, choose accessories that list that code or a name the main listing gives for it.
- When the request names the model an item must fit (for example a part for a Norvale 6420), a listing's own text that says what it fits decides: "fits every Norvale 64xx model" covers the 6420, a text naming only other Norvale models rules the listing out for that item, and an item the request does not tie to the model need not fit it.

## Searching
- Use each tool as its description and schema say; a keyword search needs words.
- Use two to five distinctive words: a plain name for the item and the values the request or the shopper stated. Leave out seller and category names unless you are looking for one seller's listing, and write a category value exactly as the store's results write it. A value nobody stated (a gender, an age or a use guessed from the story) hides listings that fit, and long queries can match nothing.
- Leave sizes and other measurements out of search words: listings write them in different units and formats (10 inches, 25.4 cm), so check them on the facts.
- Put each value in the parameter made for it (category, maker, price cap). A filtered search that returns nothing usually holds a value in the wrong parameter or spelled unlike the store: fix that before you conclude nothing exists. Words or a filter that leave the reported total unchanged had no effect in that search mode: change the mode, tool or filter instead of repeating them.
- Read the reported total. Results may come in a fixed order, not by price or relevance, so the right listing can sit on a later page: when there are more matches than you can see, narrow the search or page on.
- Before choosing, read the facts of every listing that could be the answer: for the cheapest, each cheaper one that might fit; otherwise enough to find one that meets everything, checked fact by fact.

## The cheapest (or the best by a measure)
- Choose by that measure among the listings whose facts meet everything. The page order says nothing about price, and a listing that states no value for the measure cannot be the best by it.
- Work bottom-up: list the right category with a price cap, lowering it until the reported total fits in the pages you can see, and read the cheapest first; the first in-stock listing that meets everything wins. When a search can cap the price, a search capped at your pick's price must have shown every match it reported before you order. If nothing under the cap fits, raise it past the prices already checked.
- Keep the fitting listings in price order: if your pick becomes unavailable, the next one is the answer.

## Questions (when a tool messages the shopper)
- The shopper holds back some needs and states one only when a question names that very property; any other question gets no value. Questions cost nothing in the score: use the budget before choosing, and never go over it.
- Ask about one property per message, plainly, named as the listings' facts name it ("Which pattern would you like for the teapot?"). For an order of several items, name the item and spread the questions over the items. The state note suggests which properties to ask, in order.
- Ask first about what the request's hints point to (who it is for, how it must look or match, what the tag must say), then about the properties the listings' facts describe. A property related to a stated one is still open: a stated finish leaves the colour open.
- When the request leaves something only the shopper can settle (no price limit at all, two wishes it says may clash, something it says to check with them), ask that first.
- A reply with no value (no preference, or the shopper did not follow) means no need on that property: do not ask it again in other words. A reply asking for one question at a time means your message held more than one: ask that property again on its own.
- Every value the shopper tells you is a requirement. Send messages only to ask questions: when the request asks you to tell or show the shopper something before you finish, your assistant software sends that message for you.

## Stock, cart, order and decline
- Never buy a listing that is not in stock: the store may let you cart and order it anyway.
- The order buys whatever the cart holds: exactly one line per item asked for, in its quantity, and nothing else. An order missing an item is wrong. A cart tool that sets the whole cart takes every line in one call.
- Notices can arrive with any tool result, and listings can sell out or change price without one: follow the latest state (the state note lists the requirements as they stand). Look at the cart right before ordering; open the chosen listing again only when a notice suggests it changed.
- A finish can be refused because something changed, even next to a result that looks accepted: only the end of the task means success. Read the refusal, fix the cart and finish again, and never order a listing reported unavailable. Plan the first finish with steps to spare.
- Decide once: each cart change costs a step, so change the cart only for a new fact (a notice, a reply, a listing read since), never back and forth.
- Decline only when the request's own rule says so, or when, after searching further while steps remain, every listing read for an item is ruled out by a fact on a requirement or by the budget (for several items: one item has no such listing). Otherwise order the best-supported listing: every need the shopper told confirmed, nothing contradicting it, the most requirements confirmed. A decline is almost never the right finish, while the best-supported listing often is. Keep the decline note short and generic: no item names, substitutes or ids unless the request asks.
</policy>

<examples>
These show how to judge; never reuse their values.
- Requirement colour: green. Title "Green Teapot", fact colour: white. Not met: the fact names the property.
- Requirement capacity: at least 1.2 litres. Fact capacity: 1250 ml. Met: 1250 ml is 1.25 litres.
- The shopper wants any colour but the one on Calder's 1-litre teapot, and that listing's fact says colour: white. The requirement is colour: not white; "1-litre" only identified Calder's listing.
- You asked "Which material would you like for the teapot?" and the reply was "I don't mind." There is no need on material: ask about another property.
- A refused finish carries the line "The glass lid doesn't matter after all, any lid is fine." That is the shopper relaxing the request: the lid requirement is gone, and a teapot with any lid can now be chosen.
</examples>

<output>
Reply with tool calls only, normally one per reply, and keep calling tools until you have ordered or declined.
</output>"""

BRIEF_GUIDE = """You read a shopper's request to a store assistant, together with the store's tool list, and write down exactly what the request requires. Copy values in the request's own words. Do not invent anything the request does not say.
- requirements: every attribute the item must have (type, colour, size, material, pattern, brand, amount and so on), one entry each. A category or department the request names (for example "picnic gear") is recorded as the category, not turned into another property such as a use or season. Something the shopper would merely like ("ideally", "preferably", "would be nice", "if possible") is not a requirement: it goes in soft_wants. When the request defines a value by another listing (the same as, or anything but, what a listing of some seller or maker shows), give that requirement same_as with that listing's item, seller and the properties the request describes it by, exclude true for "anything but", and its relation: same, not, at_most or at_least (no more, or no less, than that listing's value), within (with the margin as written: "within 2 cm of the height of Calder's 1-litre teapot" is within, margin "2 cm"), highest, lowest, or pairs_with (the value that goes with that listing's); those describing properties only identify that listing and are never requirements of the item.
- items: every distinct item the request asks for, one entry each, with its quantity and only the requirements that belong to that item (its kind or category first). Most requests ask for one item: then give one entry holding the same requirements as above. When the request asks for several different items, goal names them all and requirements holds them all, each "what" starting with its item's name. A count is a quantity, never a requirement: "three packs of <item>" gives quantity 3 and no requirement such as "packs". Something sold as one unit, such as a pair or a set, is quantity 1.
- money_budget: the most the shopper will spend in total, as the number the request writes, in the request's own units (for example "250 cents" gives 250, not 2.5); 0 when none is stated.
- priority: "cheapest" when the request asks for the lowest price among fitting items, "best: <measure> <higher|lower>" when it asks for the most or least of something, otherwise empty.
- question_budget: how many messages the shopper allows (for example "up to 4 questions are fine" gives 4); -1 when the request states no limit.
- step_budget: how many moves, steps, actions or tool calls are allowed (for example "finish within 25 actions" gives 25); -1 when not stated.
- procedures: every instruction about how to act or what to check, compare, say or not say, one entry each: text (one short sentence in the request's words); when it applies (before_order, before_decline, before_choosing, or any_time); and tell_shopper, true when it asks to tell, show or confirm something to the shopper in a message of its own before that moment (for example to read the order and its total back before buying it). What a decline must say ("if nothing fits, say so") is said by the decline's own note: it belongs in decline_rule, with tell_shopper false.
- decline_rule: when the request says to decline or give up, and what any decline note must or must not contain; empty when it says nothing.
- hidden_need_hints: what the shopper may care about but did not state (who it is for, what it is used with, how it must look or match), as short phrases.
- hint_properties: the product properties those hints most likely concern, most likely first (for example size, age, pattern, colour, material, shape, style).
- soft_wants: what the shopper would like but does not require, in the request's words.
- ask_first: only when the request itself leaves something that only the shopper can settle before choosing (no money limit at all; two wishes it says may not both be possible, so which matters more; something it says to check with the shopper): one short plain question each. Empty otherwise. A property of the item (its size, colour, material, style and so on) is an ask_first question only when the request itself says to check it with the shopper, never because the story hints at it: those questions come later, named as the store's listings name them.
Every sentence of the request belongs somewhere: a requirement, a wish, the priority, a budget, a procedure, a hint, or story.
- tools: every tool once, with its role: search (finds listings), details (shows one listing or its properties), cart_change (adds or removes cart items), cart_view (shows the cart), message (sends a message or question to the shopper), order (buys or checks out), decline (ends the task without buying), other."""

# a value defined by another listing: the listing as described (item, seller, the values that identify it), and whether
# the value must differ from that listing's
SAME_AS = {"type": "object", "description": "Only when the value is defined by another listing.", "properties": {
    "item": {"type": "string"}, "seller": {"type": "string"},
    "properties": {"type": "string", "description": "what the text says that listing has, as 'property: value' pairs separated by "
                                                    "semicolons"},
    "exclude": {"type": "boolean", "description": "true when the value must be anything but that listing's"},
    "relation": {"type": "string", "enum": ["same", "not", "at_most", "at_least", "within", "highest", "lowest", "pairs_with"],
                 "description": "how the wanted value relates to that listing's value: the same, anything but it, at most or at "
                                "least it, within a margin of it, the highest or lowest, or the value that goes with it"},
    "margin": {"type": "string", "description": "for within: the margin as written (for example '2 cm')"}}}

BRIEF_TOOL = {"type": "function", "function": {
    "name": "write_brief",
    "description": "Write down what the shopper's request requires.",
    "parameters": {"type": "object", "properties": {
        "goal": {"type": "string", "description": "The item wanted, in the request's words."},
        "quantity": {"type": "integer", "description": "How many units to buy; 1 when not stated."},
        "requirements": {"type": "array", "items": {"type": "object", "properties": {
            "what": {"type": "string"}, "value": {"type": "string"}, "same_as": SAME_AS}, "required": ["what", "value"]}},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "item": {"type": "string"}, "quantity": {"type": "integer"},
            "requirements": {"type": "array", "items": {"type": "object", "properties": {
                "what": {"type": "string"}, "value": {"type": "string"}, "same_as": SAME_AS}, "required": ["what", "value"]}}},
            "required": ["item", "quantity", "requirements"]}},
        "money_budget": {"type": "number"},
        "priority": {"type": "string"},
        "question_budget": {"type": "integer"},
        "step_budget": {"type": "integer"},
        "procedures": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"},
            "when": {"type": "string", "enum": list(PROCEDURE_TIMES)},
            "tell_shopper": {"type": "boolean"}}, "required": ["text", "when", "tell_shopper"]}},
        "decline_rule": {"type": "string"},
        "hidden_need_hints": {"type": "array", "items": {"type": "string"}},
        "hint_properties": {"type": "array", "items": {"type": "string"}},
        "soft_wants": {"type": "array", "items": {"type": "string"}},
        "ask_first": {"type": "array", "items": {"type": "string"}},
        "tools": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "role": {"type": "string", "enum": list(ROLES)}}, "required": ["name", "role"]}},
    }, "required": ["goal", "requirements", "question_budget", "step_budget", "procedures", "tools"]}}}

NEEDS_GUIDE = """During a shopping task the shopper answered a question, or the store attached a notice. Given the current requirement list and that text, record what it changes:
- add: the shopper insists on a value or a limit for a property (for example "make sure the lid is glass" adds lid: glass; "the spout must be no longer than 12 cm" adds spout length: at most 12 cm);
- drop: a requirement no longer applies or the shopper relaxes it (for example "the size no longer matters", "leave the colour off the list");
- replace: a requirement keeps its property but takes a new value.
A limit can be phrased as the property not falling short of (not being below) a value: for a number that is a lower limit ("at least 8"); for a word value such as a size, a type or a material it means exactly that value; when the text itself states a limit ("at most 5"), that limit is the requirement whatever words surround it.
When the wanted value is defined by another listing (the property must match, or must be anything but, what a listing shows that the shopper describes by its item, its seller or maker and some of its properties; that listing may refer on to another one), set value to "as on that listing", fill same_as with that listing's item, seller and properties, copied as written, and set its exclude to true when the shopper wants anything but that listing's value. Set its relation to how the wanted value relates to that listing's: same, not, at_most or at_least ("no taller than the one on Calder's 1-litre teapot" is at_most), within (with the margin as written, for example "within 2 cm of ..." gives margin "2 cm"), highest, lowest, or pairs_with ("a lid that goes with the one on ..."). The properties describing that listing only identify it: they are never requirements of the item.
When the order holds several items and the text names one of them, put that item's name in item. Also give the new total money budget when the text changes it (in the request's units), otherwise -1. A reply saying the shopper has no preference, or does not understand, changes nothing: it applies only to the property asked about and never drops a stated requirement, a limit or a rule. A notice that a listing sold out or changed price changes no requirement. Copy values in the text's own words.
Say what kind of reply it was (reply_kind), by its meaning however it is worded: value when it states or changes a need; no_preference when it leaves the asked property open; not_understood when the shopper did not follow the message; one_at_a_time when the shopper wants one question per message; acknowledged when the shopper only takes note of what they were told; objection when the shopper says not to go ahead with what they were told; other for anything else and for a notice. Set unavailable to true when the text says that something can no longer be bought (sold out, gone, the last one taken), whether or not it names the listing.
Read the shopper literally and charitably: folksy, figurative or roundabout wording that names a value means the shopper firmly wants that value; a value is excluded only when the shopper clearly says to avoid it, and then write it as "not <value>". A mere wish ("ideally", "would be nice") is not a requirement. When the shopper says which of two wishes matters more, the preferred one is a requirement. One text can carry several changes: record each of them. A new budget replaces the old one whether it is higher or lower. When the text changes how many units are wanted, give quantity (with the item when the order holds several)."""

NEEDS_TOOL = {"type": "function", "function": {
    "name": "record_changes",
    "description": "Record how the text changes the requirements.",
    "parameters": {"type": "object", "properties": {
        "changes": {"type": "array", "items": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["add", "drop", "replace"]},
            "item": {"type": "string"}, "what": {"type": "string"}, "value": {"type": "string"}, "same_as": SAME_AS},
            "required": ["action", "what"]}},
        "money_budget": {"type": "number"},
        "quantity": {"type": "object", "description": "Only when the text changes how many units are wanted.", "properties": {
            "item": {"type": "string"}, "units": {"type": "integer"}}},
        "reply_kind": {"type": "string", "enum": list(REPLY_KINDS)},
        "unavailable": {"type": "boolean", "description": "The text says something can no longer be bought."}},
        "required": ["changes", "money_budget"]}}}

ASK_GUIDE = """A shopping assistant may ask the shopper a few questions before choosing. The shopper holds back some needs and tells one only when a question asks about that exact property; any other question is answered with "no preference". Given the request and the properties the store's listings describe (with sample values), rank the properties the shopper most likely holds a need about.
- Read the request's hints: who the item is for or a gift (their age or size), how it must look, match or show in photos (pattern, colour, style, shape), what they will read or check (material and other properties named on tags or labels), how or where it is used.
- Use the property names exactly as listed.
- For looks, matching or photos, prefer the look property the listings vary in most (often pattern) over one the request already touches.
- Without such hints, rank the properties whose values differ among the listings that meet the stated requirements above the ones they all share: a held-back need is there to tell such listings apart.
- Leave out properties the request already fixes by name, and ones already asked about. A property related to a stated one is still open (a stated finish leaves the colour open, a stated style leaves the height open) and often holds the need.
- A question the shopper answered with "one at a time" asked about more than one thing and was not answered: rank its property again.
- A size picked among a listing's own variants (shoe or clothing sizes offered as options) is chosen at purchase, not held back: rank it only when the request points at it."""

ASK_TOOL = {"type": "function", "function": {
    "name": "rank_properties",
    "description": "Rank the properties to ask the shopper about, most likely first.",
    "parameters": {"type": "object", "properties": {
        "properties": {"type": "array", "items": {"type": "string"}, "maxItems": 6}},
        "required": ["properties"]}}}

BASKET_ASK_NOTE = """
- The order holds several items: rank pairs of an item and one of its listed properties, with the item and property names exactly as listed. Spread the questions over the items unless the request's hints point at one item; every item may hold a need.
- For each item, prefer properties that most of its read listings carry and whose values differ between them (the counts give how many listings carry each property and how many values they show): a held-back need splits the item's candidates. When an item has several read listings, a property with one value across them is a poor question; so is one whose values name variants (several sizes, colours or codes listed together). An answer covers only the item it was asked for: never ask that item again about that property, but the same property can still be the best question for another item. That holds for a reply saying the shopper did not follow, too: it means only that this item holds no need on that property, so keep that property's rank for the other items."""

BASKET_ASK_TOOL = {"type": "function", "function": {
    "name": "rank_questions",
    "description": "Rank the item properties to ask the shopper about, most likely first.",
    "parameters": {"type": "object", "properties": {
        "questions": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
            "item": {"type": "string"}, "property": {"type": "string"}}, "required": ["item", "property"]}}},
        "required": ["questions"]}}}

REVIEW_GUIDE = """You check a shopping assistant's cart right before it places an order for several items. The order scores 1 only if all of this holds: every item the request asks for is in the cart exactly once, in its requested quantity; nothing else is in the cart; each line's listing is the kind of item asked for (its category decides) and meets every requirement for that item, the request's own and the ones the shopper told or changed since; each line is in stock; the cart's total (price times quantity, summed) is within the money budget as it stands now; and, unless the request names another measure to choose by, no cheaper read listing meets every requirement for that item.
When the main item's listing gives a model code and accessory listings say which models they work with or fit, a line whose listing does not list that code (or a name the main listing gives for it) fails. When the request names the model a line must fit, a line whose listing's own text (shown after its facts) names only other models of that maker fails, a family such as "64xx" covers every code it spells out (the 6420), and a line the request does not tie to that model need not name it. A requirement marked "to look up" is defined by another listing not yet traced: the values describing that listing are not requirements of the item. Judge requirements only from the listings' facts and category (and the model a line must fit from its own text, as above). Match a requirement to the fact that describes the same property, by meaning, not by a shared word: a requirement written in the request's own words may sit under another key name, and a fact under a key that merely shares a word with it (a count, a size or a different aspect) says nothing about it. A fact that lists several values meets a requirement for any of them; the title counts only when no fact names the property; descriptions do not count. A requirement no fact, category or title gives is unproven. Do not invent requirements the request and the shopper never stated, and treat a requirement the shopper dropped as gone.
Numbers and units: listings, requests and replies often write one quantity in different units, spellings or scales, and a limit the shopper tells is often a listing's figure converted into another unit, so the comparison lands right on the limit. Read what each number measures from its unit, or for a bare number from the unit its key's name gives or the one the other listings use; a number whose unit you cannot tell proves nothing either way. Spellings, abbreviations, plurals and symbols of one unit are the same unit. For the same kind of quantity in different units, convert one side into the other's unit with the standard conversion before comparing; never compare numbers of different kinds. After converting, numbers that agree to the precision the requirement is written in are equal; "at least" and "at most" include the limit itself, and a range meets a limit only when all of it does. Storage and memory figures may use decimal or binary multiples and short forms: a figure matches when it agrees under either convention; a rating a trade states in two customary scales is compared through their usual equivalence. A fact holding several numbers (length by width by height, a range, variants) is read number by number; when you cannot tell which number is which, the fact is unproven, not a failure. A figure taken from another listing for a "same as" need is converted the same way. Call report_review."""

if not DEFAULT_CHEAPEST:            # the earlier wording: without a stated priority any fitting listing will do
    GUIDE = GUIDE.replace("- Among listings that meet everything, the request's priority decides; without one, the cheapest of them: a dearer listing that meets the same needs is graded a worse choice.",
        "- Among listings that meet everything, the request's priority decides; without one, any of them will do.")
    BRIEF_GUIDE = BRIEF_GUIDE.replace('otherwise empty.\n- question_budget', 'otherwise empty. Remarks that money is tight or that '
                                      'every penny counts are not a request for the cheapest: they only explain the budget.\n- question_budget')
    REVIEW_GUIDE = REVIEW_GUIDE.replace('and, unless the request names another measure to choose by, no cheaper read listing meets every requirement for that item.',
        'and when the request asks for the cheapest, no cheaper read listing meets every requirement for that item.')

FITS_TOOL = {"type": "function", "function": {
    "name": "report_fits",
    "description": "For each item: the read listing that meets every requirement, or none; when none does, the best-supported "
                   "read listing nothing rules out; and whether every read listing of that item is ruled out.",
    "parameters": {"type": "object", "properties": {
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "item": {"type": "string"}, "listing": {"type": "string"}, "reason": {"type": "string"},
            "best_guess": {"type": "string", "description": "only when listing is empty: the read listing of this item's kind "
                           "that is in stock, within the budget, contradicted by no fact, confirms every need the shopper told, "
                           "and leaves no stated value unshown that other read listings of the kind do show, with the most "
                           "requirements confirmed; empty when no read listing is like that"},
            "ruled_out": {"type": "boolean", "description": "true only when every read listing of this item's kind is "
                          "contradicted by a fact on a requirement as it stands, out of stock or over the budget"},
            "stated_fits": {"type": "array", "items": {"type": "string"}, "description": "the read listings of this item's "
                            "kind, in stock and within the budget, that meet every requirement the request itself states, "
                            "leaving aside the needs marked as told by the shopper"},
            "closest": {"type": "string", "description": "only when listing and best_guess are empty: the read listing of "
                        "this item's kind, in stock and within the budget, that meets every need the shopper told and the "
                        "most of the other requirements; empty when none meets the needs the shopper told"}},
            "required": ["item", "listing", "best_guess", "ruled_out", "stated_fits", "closest"]}},
        "within_budget": {"type": "boolean"}},
        "required": ["items", "within_budget"]}}}

REVIEW_TOOL = {"type": "function", "function": {
    "name": "report_review",
    "description": "Report what is wrong with the cart, if anything.",
    "parameters": {"type": "object", "properties": {
        "lines": {"type": "array", "items": {"type": "object", "properties": {
            "listing": {"type": "string"}, "item": {"type": "string"},
            "verdict": {"type": "string", "enum": ["pass", "fail", "unproven"]}, "reason": {"type": "string"}},
            "required": ["listing", "verdict"]}},
        "missing_items": {"type": "array", "items": {"type": "string"}},
        "extra_lines": {"type": "array", "items": {"type": "string"}},
        "quantity_problems": {"type": "array", "items": {"type": "string"}},
        "over_budget": {"type": "boolean"},
        "cheaper_fits": {"type": "array", "items": {"type": "string"}},
        "next": {"type": "string"}},
        "required": ["lines", "missing_items", "over_budget"]}}}

FIELDS_GUIDE = """A store tool returned the result below. Every store names its fields differently. For each kind of value, give the key that holds it, exactly as written in the result, or "" when the result has none:
- id: a listing's own identifier, the value other tools take to name that listing;
- title: a listing's name or heading;
- price: a listing's price;
- stock: whether a listing is in stock, or how many are left;
- category: the category, department, range or aisle a listing is filed under;
- facts: the key whose value holds a listing's property list (a map, or a list of name/value pairs), when the properties sit under one key;
- prose: keys holding prose (descriptions, notes, blurbs, remarks), comma-separated;
- quantity: in a cart's lines, how many units of the listing the line holds;
- cart_total: the total the cart costs;
- matches: how many matches a search found in all;
- next_page: a value to pass back to the tool for the next page of results (a cursor, token or next offset)."""

FIELD_ROLES = ("id", "title", "price", "stock", "category", "facts", "prose", "quantity", "cart_total", "matches", "next_page")
FIELDS_TOOL = {"type": "function", "function": {
    "name": "name_fields",
    "description": "Name the keys that hold each kind of value in the result.",
    "parameters": {"type": "object", "properties": {role: {"type": "string"} for role in FIELD_ROLES},
                   "required": list(FIELD_ROLES)}}}

POINTER_GUIDE = """A store listing's notes may say that, for some property, another listing decides it: "for <property>, go by / refer to / look to / see <another listing>, not this item", "the <property> is to be taken from <another listing>", and the like. Given one property and a listing's notes, say whether the notes send that property to another listing. If they do, describe that listing: item = the kind of item only (for example "teapots"), seller = its seller or maker, properties = every value the notes give for it (colour, pattern, size, material, weight and so on) as 'property: value' pairs, copied as written, or empty when the notes give none. A note that only says another version exists elsewhere, or that sends a different property on, does not send this one.
Notes can also state a rule of the store's that pairs this listing's own value of the property with another value: "its <value> is matched with <other value>", "<value> pairs with <other value>", "<value> belongs with <other value>", or a list of such pairs. The store states such a rule so the shopper's item takes the partner value, not this listing's own one. Give own_value = this listing's own value of the property when the notes state it, and paired_value = the partner of that own value, copied as written, or "" when the notes pair nothing with it. These invented examples show the reading: for property glaze with own value speckled, "A speckled glaze is matched with a crackle glaze." gives paired_value crackle; for own value celadon, "Glazes pair up: speckled with crackle, celadon with ash." gives ash; "Pairs nicely with our tea towels." pairs no value of the property, so paired_value is ""."""

POINTER_TOOL = {"type": "function", "function": {
    "name": "report_pointer",
    "description": "Report whether the notes send the property to another listing, and which.",
    "parameters": {"type": "object", "properties": {
        "points_on": {"type": "boolean"},
        "item": {"type": "string"}, "seller": {"type": "string"},
        "properties": {"type": "string", "description": "the values the notes give for that listing, as 'property: value' pairs "
                                                        "separated by semicolons"},
        "own_value": {"type": "string", "description": "this listing's own value of the property as the notes state it, or \"\""},
        "paired_value": {"type": "string", "description": "the value the notes pair this listing's own value with, or \"\""}},
        "required": ["points_on", "paired_value"]}}}

# STRICT_REFS: which of several listings a pointer means
PICK_GUIDE = """A shopper's pointer describes one store listing by its kind of item, its seller and some of its values. Several listings hold those values. Say which one the description means, judging by each listing's title and properties. A sentence of a listing's notes counts only when it describes that listing itself; a sentence that says what the listing is not, or that is about some other listing, tells nothing about this one. A number written in another unit counts when it is the same quantity after the usual conversion between the two units, at the precision written. Give an empty listing when none of them is clearly the one or none holds every value.
Example (invented): described "the teapot sold by Calder with glaze: speckled; capacity: 1 litre". NV-301 "Calder teapot speckled 1 l", properties glaze speckled, capacity 1 litre, notes "A 2 litre one is sold by Fernwick." NV-305 "Calder teapot", properties glaze speckled, capacity 1 litre, notes "Not the tall version." Both hold every value, and neither note describes another value of its own listing: pick the one whose title also matches, NV-301."""

PICK_TOOL = {"type": "function", "function": {
    "name": "pick_listing",
    "description": "Name the listing the description means.",
    "parameters": {"type": "object", "properties": {
        "listing": {"type": "string", "description": "its id, or empty when none is clearly the one"}},
        "required": ["listing"]}}}

NEGATION = re.compile(r"\b(?:not|no|nothing|none|without|except|excluding|exclude|avoid|never|other than|anything but|leave out|"
                      r"leaving out)\b", re.I)
# model codes and what a listing works with, as listings write them in their text ("Model no. AB-1234", "Also sold as
# Cd 123", "Works with AB-1234, Ef 567", "Fits: AB-1234"): a set must hold accessories that name the main item's model
CODE = r"[A-Za-z]{1,10}[- ]?\d{2,6}[A-Za-z]?"
MODEL_CODE = re.compile(r"\bmodel\s*(?:no\.?|number|#)\s*[:.]?\s*(" + CODE + r")", re.I)
ALSO_SOLD = re.compile(r"\balso (?:sold|known|listed|offered) as\s*:?\s*(" + CODE + r")", re.I)
WORKS_WITH = re.compile(r"\b(?:works? with|also fits|fits|compatible with|for use with)\s*:?\s*(" + CODE
                        + r"(?:\s*(?:,|/|\bor\b|\band\b)\s*" + CODE + r")*)", re.I)
# a reply saying the message asked about more than one thing: that question was not answered
ONE_AT_A_TIME = re.compile(r"one (?:question |thing |matter )?at a time|each question on its own|questions? in turn|only hold (?:the )?one|"
                           r"one question each|a single question", re.I)
# a listing that can no longer be had; "gone up", a price change, and "no longer matters", a dropped requirement, are not
UNAVAILABLE = re.compile(r"sold out|out of stock|unavailable|not available|no longer (?:available|in stock|for sale|on sale|"
                         r"sold|offered|listed|stocked|carried|be (?:bought|ordered|had))|\bgone\b(?! (?:up|down|higher|lower|"
                         r"through|over|for))|none left|zero left|nothing left|no units|\b(?:ran|run|runs|running) (?:dry|out)\b|"
                         r"\b(?:stock|count|units|quantity|inventory)\b[^.]{0,30}\bzero\b|"
                         r"can'?t be (?:ordered|bought)|cannot be (?:ordered|bought)|dried up", re.I)
# a notice that a listing's price moved (up, down or either): the pick may still be right at its new price
PRICE_MOVED = re.compile(r"\b(?:price[ds]?|pricier|cost[s]?|costing|amount|charge[ds]?|dearer|cheaper|tag)\b", re.I)
NOT_PROPERTY = re.compile(r"\b(?:quantity|quantities|how many|number of|count|amount|price|prices|budget|cost|costs|spend|"
                          r"spending|availability|available|stock|delivery|deliver|shipping|arrive|arrival)\b", re.I)
# a span of one quantity: "between 5 and 7 kg", "from 5 kg to 7 kg", "5 to 7 kg", "5-7 kg"
RANGE = re.compile(r"\bbetween\s+(\d+(?:\.\d+)?)\s*([a-z]*)\s+and\s+(\d+(?:\.\d+)?)\s*([a-z]*)"
                   r"|(?<![\w.-])(\d+(?:\.\d+)?)\s*([a-z]*)\s*(?:\bto\b|-)\s*(\d+(?:\.\d+)?)\s*([a-z]*)", re.I)
COMPARE = re.compile(r"\b(?:at most|at least|up to|under|over|below|above|less|more|max|min|maximum|minimum|between|"
                     r"within|no more|no less|or less|or more|smaller|larger|bigger|before|after|until|by the)\b", re.I)
# SEARCH_RETRY: values of a search's match-mode choice; a loose one matches some of the words, a strict one them as written
LOOSE_MODE = re.compile(r"word|text|free|keyword|typed|any|match|fuzzy|loose|broad|partial|contain|some", re.I)
STRICT_MODE = re.compile(r"exact|phrase|strict|literal|whole|verbatim|precise|\ball\b", re.I)
FILLER = {"the", "and", "for", "with", "has", "have", "must", "should", "only", "any", "its", "that", "this", "one", "item", "all", "come",
          "some", "something", "kind", "type", "want", "need", "like"}
# the words of a question that do not name what it asks about
ASKING = {"which", "what", "would", "you", "like", "for", "the", "does", "have", "any", "preference", "prefer", "your", "about",
          "want", "need", "should", "there", "this", "that", "these", "those", "please", "how", "are", "with", "and", "from",
          "its", "care", "mind", "matter", "important", "particular", "specific", "wish", "one", "ones", "example", "such",
          "could", "tell", "let", "know", "both", "each", "them", "they", "will", "can", "get", "choose", "pick"}

NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "eleven": 11, "twelve": 12}
# the value a finishing tool's switch takes to end without buying, or to buy
DECLINE_WORDS = re.compile(r"refus|declin|reject|abandon|cancel|give ?up|walk away|no ?(?:buy|purchase|order|sale)|"
                           r"nothing|abstain|without buying", re.I)
ORDER_WORDS = re.compile(r"buy|purchas|order|checkout|check ?out|accept|confirm|pay|settle", re.I)
LOWER_BOUND = re.compile(r"at least|minimum|\bmin\b|or more|no less|not less|\bover\b|above|more than|larger|bigger|wider|"
                         r"longer|taller|heavier", re.I)
UPPER_BOUND = re.compile(r"at most|maximum|\bmax\b|up to|or less|no more|not more|under|below|less than|smaller|within|"
                         r"narrower|shorter|lighter", re.I)


def model_fields(model: str) -> dict:
    """Request fields for a model family: reasoning kept off or as light as the family allows."""
    if model.startswith("anthropic/"):
        return {}
    if model.startswith("google/"):
        return {"reasoning": {"effort": "low"}}
    if model.startswith("z-ai/"):
        return {"reasoning": {"enabled": False}}
    return {"reasoning": {"effort": "none"}}


def clean_text(value: str) -> str:
    """A string the store can encode: lone surrogates and control characters removed."""
    value = value.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", value)


def parse_json(text: Any) -> Any:
    """Parse a model's JSON arguments, closing a reply cut off at the end."""
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except ValueError:
        pass
    body = text.strip()
    for cut in range(len(body), max(0, len(body) - 400), -1):
        head = body[:cut].rstrip().rstrip(",")
        stack, quoted, escaped = [], False, False
        for ch in head:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = not quoted
            elif not quoted and ch in "[{":
                stack.append("]" if ch == "[" else "}")
            elif not quoted and ch in "]}" and stack:
                stack.pop()
        if quoted:
            continue
        try:
            return json.loads(head + "".join(reversed(stack)))
        except ValueError:
            continue
    return None


def kinds_of(spec: dict) -> list[str]:
    kind = spec.get("type")
    return [k for k in (kind if isinstance(kind, list) else [kind]) if isinstance(k, str)]


def type_ok(value: Any, kinds: list[str]) -> bool:
    if not kinds:
        return True
    checks = {"string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
              "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
              "array": lambda v: isinstance(v, list), "object": lambda v: isinstance(v, dict), "null": lambda v: v is None}
    return any(checks.get(k, lambda v: True)(value) for k in kinds)


def fit_args(schema: Any, value: Any, path: str = "$") -> tuple[Any, list[str]]:
    """Make arguments fit a tool's JSON schema where that is safe, and list what could not be fixed. Covers the keywords
    the runtime checks: type, const, enum, properties, required, additionalProperties, items, min/maxItems,
    min/maxLength and minimum/maximum."""
    problems: list[str] = []
    if not isinstance(schema, dict):
        return value, problems
    kinds = kinds_of(schema)
    if "const" in schema:
        return schema["const"], problems
    if isinstance(value, str):
        value = clean_text(value)
    if "integer" in kinds and not type_ok(value, ["integer"]):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        elif isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
            value = int(value)
    if "number" in kinds and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", value):
        value = float(value)
    if "boolean" in kinds and isinstance(value, str) and value.strip().lower() in ("true", "false"):
        value = value.strip().lower() == "true"
    if "string" in kinds and isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and value not in enum:
        same = next((e for e in enum if isinstance(e, str) and isinstance(value, str) and e.lower() == value.strip().lower()), None)
        if same is None:
            problems.append(f"{path}: must be one of {enum}")
            return value, problems
        value = same
    if not type_ok(value, kinds):
        problems.append(f"{path}: expected {' or '.join(kinds)}")
        return value, problems
    if isinstance(value, str):
        longest, shortest = schema.get("maxLength"), schema.get("minLength")
        if isinstance(longest, int) and len(value) > longest:
            if re.search(r"(?:^|[_.\-$])(?:id|ids|sku|code|key|ref)$|id$", path.lower()):
                problems.append(f"{path}: longer than {longest} characters")
            else:
                value = value[:longest].rstrip()
        if isinstance(shortest, int) and len(value) < shortest:
            problems.append(f"{path}: shorter than {shortest} characters")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low, high = schema.get("minimum"), schema.get("maximum")
        if isinstance(low, (int, float)) and value < low:
            problems.append(f"{path}: below the minimum {low}")
        if isinstance(high, (int, float)) and value > high:
            problems.append(f"{path}: above the maximum {high}")
    if isinstance(value, dict):
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        fixed: dict = {}
        for key, item in value.items():
            if item is None and key not in (schema.get("required") or []):
                continue                    # an empty optional field is left out rather than sent as null
            if key not in properties and schema.get("additionalProperties") is False:
                continue                    # a field the tool does not take is dropped
            item, more = fit_args(properties.get(key), item, f"{path}.{key}")
            fixed[key] = item
            problems += more
        for key in schema.get("required") or []:
            if key not in fixed:
                problems.append(f"{path}: missing required property '{key}'")
        value = fixed
    if isinstance(value, list):
        if isinstance(schema.get("items"), dict):
            fixed_list = []
            for number, item in enumerate(value):
                item, more = fit_args(schema["items"], item, f"{path}[{number}]")
                fixed_list.append(item)
                problems += more
            value = fixed_list
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            problems.append(f"{path}: fewer than {schema['minItems']} items")
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            problems.append(f"{path}: more than {schema['maxItems']} items")
    return value, problems


def short(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return text if len(text) <= limit else text[: limit - 15] + " ...[cut]"


def trimmed(value: Any, cut: int, rows: int | None = None) -> Any:
    """The value with prose strings cut to about `cut` characters (and lists to `rows` items when given). Strings without
    spaces, such as ids and page tokens, stay whole: a cut token or id is useless."""
    if isinstance(value, str):
        return value if len(value) <= cut or " " not in value.strip() else value[:cut].rstrip() + "..."
    if isinstance(value, dict):
        return {k: trimmed(v, cut, rows) for k, v in value.items()}
    if isinstance(value, list):
        return [trimmed(v, cut, rows) for v in (value if rows is None else value[:rows])]
    return value


def shown(value: Any, limit: int) -> str:
    """A tool result as the model sees it: whole when it fits, else with its prose cut shorter step by step, so that
    totals, page markers and ids, which often come last, are still there."""
    if not isinstance(value, (dict, list)):
        return short(value, limit)
    for cut in (None, 240, 140, 80, 40):
        text = short(value if cut is None else trimmed(value, cut), 10 ** 9)
        if len(text) <= limit:
            return text
    return short(trimmed(value, 40), limit)


def prose(text: str) -> bool:
    """True for a value that reads as a sentence (a note or a description) rather than a property value."""
    words = text.split()
    return len(words) >= 25 or (len(words) >= 8 and bool(re.search(r"[a-z)'\"][.!?;](?:\s|$)", text)))


def key_set(value: Any, depth: int = 0) -> set[str]:
    """Every key in a result, nested ones included."""
    keys: set[str] = set()
    if depth > 5:
        return keys
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys |= key_set(item, depth + 1)
    elif isinstance(value, list):
        for item in value[:12]:
            keys |= key_set(item, depth + 1)
    return keys


def id_key(key: str) -> bool:
    return bool(re.search(r"(?:^|_)(?:id|pid|sku|code)$|id$", key.lower()))


def money(value: Any) -> float | None:
    """A price from a number, a text such as "$12.99", or an object holding an amount."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        # only a text that is a sum: a number with at most a currency sign or a short unit
        number = re.fullmatch(r"\s*(?:[A-Za-z]{1,3}\s*)?[^\w\s]{0,2}\s*(-?\d[\d,]*(?:\.\d+)?)\s*[A-Za-z]{0,4}\.?\s*", value)
        if number:
            try:
                return float(number.group(1).replace(",", ""))
            except ValueError:
                return None
    if isinstance(value, dict):
        return next((money(v) for k, v in value.items() if re.search(r"amount|value|price", str(k), re.I) and money(v) is not None), None)
    return None


def sold_out(stock: Any) -> bool:
    """True when a stock field says the listing cannot be had: false, a count of zero, or words to that effect."""
    if stock is None:
        return False
    if isinstance(stock, bool):
        return not stock
    if isinstance(stock, (int, float)):
        return stock <= 0
    return bool(re.search(r"out|unavail|sold|^no$|^false$|^0\b|none|zero", str(stock).strip(), re.I))


def records_in(value: Any, found: list[dict], depth: int = 0, fields: dict | None = None) -> None:
    """Collect dicts that look like listings: an id-like field plus a title, a name or a price. `fields` names the keys
    the tool's results use (see learn_fields); key-name patterns cover what it leaves out."""
    if depth > 6 or len(found) > 80:
        return
    fields = fields or {}
    if isinstance(value, dict):
        def mapped(role: str) -> Any:
            key = fields.get(role)
            return value.get(key) if key and key in value else None
        ident = mapped("id")
        if not isinstance(ident, (str, int)) or isinstance(ident, bool) or not 0 < len(str(ident)) <= 64:
            ident = next((v for k, v in value.items() if id_key(str(k)) and isinstance(v, (str, int)) and not isinstance(v, bool)
                          and 0 < len(str(v)) <= 64), None)
        title = mapped("title") if isinstance(mapped("title"), str) else \
            next((v for k, v in value.items() if re.search(r"title|name|label", str(k), re.I) and isinstance(v, str)), None)
        price = money(mapped("price"))
        if price is None:
            price = next((money(v) for k, v in value.items() if re.search(r"price|cost|amount|total", str(k), re.I)
                          and money(v) is not None), None)
        if ident is not None and (title or price is not None):
            stock = mapped("stock")
            if not isinstance(stock, (bool, int, str)):
                stock = next((v for k, v in value.items() if re.search(r"stock|avail", str(k), re.I) and isinstance(v, (bool, int, str))), None)
            kind = mapped("category") if isinstance(mapped("category"), str) else \
                next((v for k, v in value.items() if re.search(r"categor|department|dept|section", str(k), re.I) and isinstance(v, str)), None)
            found.append({"id": str(ident), "title": (title or "")[:90], "price": price, "stock": stock, "category": (kind or "")[:60]})
        for item in value.values():
            records_in(item, found, depth + 1, fields)
    elif isinstance(value, list):
        for item in value[:60]:
            records_in(item, found, depth + 1, fields)


def flat(value: Any) -> str | None:
    """A property value as text: a scalar, or a short list of scalars joined with commas."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, list) and value and len(value) <= 20 and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in value):
        return ", ".join(str(x) for x in value)
    return None


def property_map(item: Any, loose: bool = False) -> dict | None:
    """A listing's property list in either common shape: a map of short values, or a list of name/value pairs. With
    loose (the field a tool's results were learned to keep properties in), a list of two-field records with the same
    two keys counts as name/value pairs whatever the keys are called: the key whose values all differ names them."""
    if isinstance(item, dict) and item:
        values = {str(k): flat(v) for k, v in item.items()}
        if sum(v is not None for v in values.values()) >= 0.8 * len(values):
            # prose (a note or description) is no property value: a note can even quote another listing's values
            return {k: v[:120] for k, v in values.items() if v is not None and len(v) <= 200 and not prose(v)}
    if isinstance(item, list) and item and all(isinstance(x, dict) for x in item):
        pairs = {}
        for entry in item[:60]:
            name = next((v for k, v in entry.items() if re.search(r"name|key|label|attr|property", str(k), re.I) and isinstance(v, str)), None)
            text = next((flat(v) for k, v in entry.items() if re.search(r"value|val\b|text|content", str(k), re.I) and flat(v) is not None), None)
            if name and text is not None:
                pairs[name[:60]] = text[:120]
        if len(pairs) >= max(1, len(item) // 2):
            return pairs
        shapes = {tuple(x) for x in item[:60]}
        if loose and len(item) >= 2 and len(shapes) == 1 and len(next(iter(shapes))) == 2:
            first, second = next(iter(shapes))
            rows = item[:60]

            def names(key: str) -> bool:
                return all(isinstance(x[key], str) and 0 < len(x[key]) <= 60 for x in rows) and len({x[key] for x in rows}) == len(rows)
            keys = [k for k in (first, second) if names(k)]
            # both could name: the one with fewer digits in its texts names the properties (values carry the numbers),
            # else the first (a property list writes the name before the value)
            name_key = min(keys, key=lambda k: (sum(bool(re.search(r"\d", x[k])) for x in rows), (first, second).index(k))) if keys else None
            if name_key is not None:
                value_key = second if name_key == first else first
                pairs = {x[name_key][:60]: flat(x[value_key])[:120] for x in rows
                         if flat(x[value_key]) is not None and not prose(flat(x[value_key]))}
                if len(pairs) >= max(2, len(rows) // 2):
                    return pairs
    return None


def own_fields(value: Any, depth: int = 0, fields: dict | None = None) -> tuple[str, dict] | None:
    """A listing shown as one flat record: its id and its own short fields other than id, title, price, stock, text
    and links, read as its properties. `fields` holds the keys the tool's results use for those (see learn_fields)."""
    if depth > 2 or not isinstance(value, dict):
        return None
    fields = fields or {}
    id_field = fields.get("id")
    ident = value.get(id_field) if id_field in value else \
        next((v for k, v in value.items() if id_key(str(k)) and isinstance(v, (str, int)) and not isinstance(v, bool)
              and 0 < len(str(v)) <= 64), None)
    if ident is None or isinstance(ident, bool) or not isinstance(ident, (str, int)):
        inner = [v for v in value.values() if isinstance(v, dict)]
        return own_fields(inner[0], depth + 1, fields) if len(inner) == 1 else None
    known = {fields.get(r) for r in ("id", "title", "price", "stock", "category", "quantity") if fields.get(r)} \
        | set(fields.get("prose") or [])
    sheet = {str(k): flat(v)[:120] for k, v in value.items()
             if flat(v) is not None and len(flat(v)) <= 120 and not prose(flat(v)) and not id_key(str(k)) and str(k) not in known
             and not re.search(r"title|name|price|cost|amount|stock|avail|descr|summary|image|url|link|review|rating", str(k), re.I)}
    return (str(ident), sheet) if len(sheet) >= 3 else None


def facts_in(value: Any, found: list[tuple[str, dict]], depth: int = 0, fields: dict | None = None) -> None:
    """Property lists of listings: a map of short properties (or name/value pairs) beside an id-like field, or under
    the key `fields` names for them."""
    if depth > 6 or len(found) > 40:
        return
    fields = fields or {}
    if isinstance(value, dict):
        id_field = fields.get("id")
        ident = value.get(id_field) if id_field in value and isinstance(value.get(id_field), (str, int)) \
            and not isinstance(value.get(id_field), bool) else \
            next((v for k, v in value.items() if id_key(str(k)) and isinstance(v, (str, int)) and not isinstance(v, bool)
                  and 0 < len(str(v)) <= 64), None)
        if ident is not None:
            facts_key = fields.get("facts")
            free = set(fields.get("prose") or [])
            for key, item in value.items():
                if (facts_key and facts_key in value and key != facts_key) or key in free:
                    continue                # the named property list decides; prose fields never do
                sheet = property_map(item, loose=bool(facts_key) and key == facts_key)
                if sheet:
                    found.append((str(ident), sheet))
        for item in value.values():
            facts_in(item, found, depth + 1, fields)
    elif isinstance(value, list):
        for item in value[:40]:
            facts_in(item, found, depth + 1, fields)


def in_turn(first: list[str], second: list[str]) -> list[str]:
    """Two ranked lists of properties merged in turn, each property once (the same words make the same property)."""
    out: list[str] = []
    for pair in zip(first + [""] * len(second), second + [""] * len(first)):
        for p in pair:
            if p and not any(tokens(p) == tokens(q) for q in out):
                out.append(p)
    return out


def tokens(text: Any) -> set[str]:
    """Lower-case words of three or more letters, with a plural or past ending dropped the same way everywhere
    ("teapots" and "teapot", "boxes" and "box", "cities" and "city")."""
    out = set()
    for word in re.findall(r"[a-z]{3,}", str(text).lower()):
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif re.search(r"(?:ss|x|z|ch|sh)es$", word) and len(word) > 4:
            word = word[:-2]
        elif word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
            word = word[:-1]
        elif word.endswith("ed") and len(word) > 4:
            word = word[:-2]
        out.add(word)
    return out


def notices_in(observation: Any) -> list[str]:
    """Texts a world attaches to a result: a refusal, or a wrapper holding the result next to a list of sentences."""
    out: list[str] = []
    if not isinstance(observation, dict):
        return out
    if "refusal" in observation:
        refusal = observation.get("refusal")
        out += [str(x) for x in refusal] if isinstance(refusal, list) else [short(refusal, 400)]
    if len(observation) == 2 and "refusal" not in observation:
        values = list(observation.values())
        lines = next((v for v in values if isinstance(v, list) and v and all(isinstance(x, str) and len(x.split()) >= 2 for x in v)), None)
        other = next((v for v in values if v is not lines), None)
        if lines is not None and isinstance(other, (dict, list, str)):     # a reply text can be wrapped too
            out += [str(x) for x in lines]
        if isinstance(other, dict):
            out += notices_in(other)         # a wrapper can hold another wrapper
    return list(dict.fromkeys(line.strip()[:400] for line in out if str(line).strip()))


COUNT_WORDS = {"pack", "packs", "piece", "pieces", "pcs", "pc", "unit", "units", "item", "items", "count", "qty", "quantity",
               "total", "of", "in", "all", "each"}


def count_only(requirement: dict, item_words: set[str] | None = None, quantity: int | None = None) -> bool:
    """A brief requirement that only restates how many to buy ("packs, 6" for six packs, "2 pieces", "trivet
    quantity: 4", "teapot: 1"): a quantity, which the cart check covers, never a property a listing must show. A
    number under a property's own name ("drawers: 4"), a pack size ("pack of 12" when one pack is bought) and a
    dimension ("3440 x 1440") stay requirements."""
    what, value = str(requirement.get("what") or ""), str(requirement.get("value") or "")
    if re.fullmatch(r"\s*\d+\s*", value):
        return bool(NOT_PROPERTY.search(what)) or bool(item_words and tokens(what) and tokens(what) <= item_words)
    words = re.findall(r"[a-z]+", value.lower())
    numbers = re.findall(r"\d+(?:\.\d+)?", value)
    return len(numbers) == 1 and quantity is not None and float(numbers[0]) == quantity and bool(words) \
        and all(w in COUNT_WORDS for w in words) and not re.search(r"\d[a-z]", value.lower())


def number_in(text: str, value: int) -> bool:
    """True when the request writes the number, in digits or as a word."""
    if re.search(rf"(?<![\d.]){value}(?![\d.])", text):
        return True
    words = [w for w, n in NUMBER_WORDS.items() if n == value]
    return any(re.search(rf"\b{w}\b", text, re.I) for w in words)


def amount_of(text: str) -> tuple[float, str] | None:
    """The one number a text holds and the unit word right after it ("16gb" gives 16 and gb); None for no number or
    several (a range or a list)."""
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    found = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z]*)", text.lower())
    if len(numbers) != 1 or not found:
        return None
    unit = found.group(2)
    if unit.endswith("ches"):
        unit = unit[:-2]                    # "inches" and "inch" are one unit
    elif unit.endswith("s") and len(unit) > 2:
        unit = unit[:-1]
    return float(found.group(1)), unit


def range_of(clause: str) -> tuple[float, float, str | None] | None:
    """A span written in a clause: (low end, high end, unit), the unit None when the two ends name different units."""
    found = RANGE.search(clause)
    if not found:
        return None
    a, ua, b, ub = found.groups()[:4] if found.group(1) else found.groups()[4:]
    ends = [amount_of(f"{a} {ua or ub}"), amount_of(f"{b} {ub or ua}")]
    if not all(ends):
        return None
    lo, hi = sorted((ends[0][0], ends[1][0]))
    return lo, hi, ends[0][1] if ends[0][1] == ends[1][1] else None


def bound_check(clause: str, sheet: dict, what_words: set[str]) -> str | None:
    """A limit such as "at most 30 cm" read against a fact naming the property whose number has the same unit (or none
    on both sides); RANGE_CHECK, a span such as "between 5 and 7 kg" or "5-7 kg" too, its ends included. Returns a
    yes/no verdict, or None when code cannot settle it."""
    span = range_of(clause) if RANGE_CHECK else None
    if span:
        if span[2] is None:
            return None                     # two units: the model compares
        lo, hi, unit = span
        for key, value in sheet.items():
            if not tokens(key) & what_words:
                continue
            have = amount_of(value)
            if have is None or have[1] != unit:
                continue
            return f"{'yes' if lo - 1e-9 <= have[0] <= hi + 1e-9 else 'no'} ({key}={value[:40]})"
        return None
    low, high = bool(LOWER_BOUND.search(clause)), bool(UPPER_BOUND.search(clause))
    limit = amount_of(clause)
    if low == high or limit is None:
        return None                         # "between", an unclear bound, or no single number
    for key, value in sheet.items():
        if not tokens(key) & what_words:
            continue
        have = amount_of(value)
        if have is None or have[1] != limit[1]:
            continue
        ok = have[0] >= limit[0] - 1e-9 if low else have[0] <= limit[0] + 1e-9
        return f"{'yes' if ok else 'no'} ({key}={value[:40]})"
    return None


# a sentence of a listing's own text that says what it fits or works with
FIT_SENTENCE = re.compile(r"\b(?:fits?|compatible|works? with|for use with|suits?)\b", re.I)
# one model code or family in such a sentence: an optional maker word, then digits with placeholders ("41xx"), a range
# ("4100-4199"), "<digits> series", or a plain code
FIT_TERM = re.compile(r"(?:\b([A-Za-z][A-Za-z-]{1,20})[ -]+)?(?<![\w.])(\d{1,6})(?:([xX*#]{1,4})(?!\w)|\s*(?:-|to)\s*(\d{2,6})(?!\w|\.\d)"
                      r"|\s+(series|range|line)\b|(?!\w|\.\d))", re.I)
FIT_KEY = re.compile(r"\bfit|compatib|works? with|\bsuits?\b|for use with|goes with", re.I)
FIT_SKIP = {"and", "or", "every", "all", "any", "the", "model", "models", "fits", "fit", "with", "for", "use", "only", "also",
            "compatible", "works", "work", "suits", "suit", "nos", "no", "number", "numbers", "type", "types", "series", "to"}


def fit_terms(text: str) -> tuple[list[dict], str]:
    """FIT_CODES: the model codes and families the fit sentences of a text name (maker word, kind, numbers), and the
    first such sentence. A term counts only with three digits or more (placeholders included), so sizes and counts
    ("fits 10 mm bolts") are no codes; a term with no maker word of its own takes the sentence's last one."""
    terms: list[dict] = []
    first = ""
    for sentence in re.split(r"(?<=[.!?;])\s+|\n", str(text)):
        if not FIT_SENTENCE.search(sentence):
            continue
        maker = ""
        found = []
        for m in FIT_TERM.finditer(sentence):
            word, digits, holes, upper, series = m.groups()
            if word and word.lower() not in FIT_SKIP:
                maker = word.lower()
            size = len(digits) + (len(holes) if holes else 0)
            if size < 3 and not series:
                continue
            kind = "family" if holes else "range" if upper else "series" if series else "code"
            found.append({"maker": maker, "kind": kind, "digits": digits, "size": size, "upper": upper or "",
                          "text": m.group(0).strip()})
        if found:
            terms += found
            first = first or sentence.strip()[:160]
    return terms, first


def fit_holds(term: dict, code: str) -> bool:
    """A model code (its digits) inside a fit term: the same code, a family with its leading digits and length, a
    series with its leading digits, or a range around it."""
    if term["kind"] == "code":
        return code == term["digits"]
    if term["kind"] == "family":
        return len(code) == term["size"] and code.startswith(term["digits"])
    if term["kind"] == "series":
        return code.startswith(term["digits"]) and len(code) > len(term["digits"])
    return len(code) == len(term["digits"]) and int(term["digits"]) <= int(code) <= int(term["upper"])


def cart_reply_took(observation: Any) -> bool:
    """A cart reply that names no listing but says the change went through: a true flag or a count of one or more at its
    top level, or a word such as "ok"; never one holding a false flag."""
    if not isinstance(observation, dict):
        return False
    values = list(observation.values())
    if any(v is False for v in values):
        return False
    return any(v is True for v in values) or any(isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 1 for v in values) \
        or any(isinstance(v, str) and re.fullmatch(r"\s*(?:ok|okay|added|done|success|accepted)\s*", v, re.I) for v in values)


def split_message(text: str, limit: int, least: int = 1) -> list[str]:
    """FIRST_PROMPT_MESSAGE: a text cut into the fewest parts of at most limit characters, at sentence ends where it can
    be and at word ends inside a sentence longer than the limit, the last part made no shorter than least by moving
    words from the part before it; the parts joined with spaces give the text back. PROMPT_ALWAYS: when sentence ends
    need more parts than word ends do, each part is filled to the limit at word ends instead."""
    limit = max(1, int(limit))
    parts = by_sentences(text, limit)
    if PROMPT_ALWAYS:
        packed = pack_words(text, limit)
        parts = packed if len(packed) < len(parts) else parts
    while len(parts) >= 2 and len(parts[-1]) < least:
        head, _, word = parts[-2].rpartition(" ")
        if not head or len(word) + 1 + len(parts[-1]) > limit:
            break
        parts[-2], parts[-1] = head, word + " " + parts[-1]
    return parts


def looseness(value: Any) -> int:
    """SEARCH_RETRY: how loosely a match-mode value matches the words: 1 loose, -1 strict, 0 neither or both."""
    return bool(LOOSE_MODE.search(str(value))) - bool(STRICT_MODE.search(str(value))) if value is not None else 0


def by_sentences(text: str, limit: int) -> list[str]:
    """split_message: parts of at most limit characters cut at sentence ends, at word ends inside a longer sentence."""
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit + 1)
            cut = cut if cut > 0 else limit
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    parts: list[str] = []
    for piece in pieces:
        if parts and len(parts[-1]) + 1 + len(piece) <= limit:
            parts[-1] += " " + piece
        else:
            parts.append(piece)
    return parts


def pack_words(text: str, limit: int) -> list[str]:
    """split_message: the fewest parts of at most limit characters, each filled at word ends (a longer word cut inside)."""
    parts: list[str] = []
    for word in text.split():
        while len(word) > limit:
            parts.append(word[:limit])
            word = word[limit:]
        if not word:
            continue
        if parts and len(parts[-1]) + 1 + len(word) <= limit:
            parts[-1] += " " + word
        else:
            parts.append(word)
    return parts


def fit_message(text: str, limit: int, least: int, room: int) -> list[str]:
    """PROMPT_FIT: a text in at most room parts of at most limit characters: the parts split_message cuts when they fit,
    else as much of the text as room parts hold, each filled to the limit at word ends from its start (the rest left
    out)."""
    parts = split_message(text, limit, least)
    return parts if len(parts) <= room else pack_words(text, limit)[:max(0, room)]


def trace_pieces(text: str, key: str, limit: int, most: int, numbered: bool) -> list[str]:
    """TRACE_REQUEST: text cut into at most most pieces, each sent as {key: [piece]} within limit bytes (escapes
    counted); numbered: each piece starts "[k/n] " so the pieces join back; the rest past the last piece is left out."""
    def size(piece: str) -> int:
        args = {key: [piece]}
        return max(len(json.dumps(args).encode()), len(json.dumps(args, ensure_ascii=False).encode()))
    margin = 16 if numbered else 0              # the "[k/n] " label
    cuts: list[str] = []
    rest = text
    while rest and len(cuts) < max(0, most):
        low, high = 1, len(rest)
        while low < high:                       # the longest start of rest that fits
            mid = (low + high + 1) // 2
            if size(rest[:mid]) + margin <= limit:
                low = mid
            else:
                high = mid - 1
        if size(rest[:low]) + margin > limit:
            break
        cuts.append(rest[:low])
        rest = rest[low:]
    if not numbered:
        return cuts
    return [f"[{k}/{len(cuts)}] {piece}" for k, piece in enumerate(cuts, 1)]


def stated_length(text: str) -> int | None:
    """FIRST_PROMPT_MESSAGE: a message length a description states ("up to 500 characters", "max 300 chars")."""
    found = re.search(r"(?:up to|at most|max(?:imum)?(?: of)?|no more than|not more than|under|within|limit(?:ed)? to)\s+"
                      r"(\d{2,5})\s*(?:characters?|chars?|letters?)\b", str(text), re.I)
    return int(found.group(1)) if found else None


class Session:
    """One task's session with the store: numbered steps, replay-safe retries and the finish state."""

    def __init__(self, problem: dict) -> None:
        environment = problem.get("environment") or {}
        self.binding = dict(environment.get("binding") or {})
        policy = environment.get("policy_view") or {}
        self.request = str(policy.get("query") or problem.get("query") or "")
        self.opening = policy.get("opening")
        self.tools = [t for t in policy.get("tools") or []
                      if isinstance(t, dict) and isinstance(t.get("function"), dict) and str(t["function"].get("name") or "")]
        self.specs = {t["function"]["name"]: t["function"] for t in self.tools}
        try:
            self.max_steps = max(1, int(policy.get("max_steps") or 30))
        except (TypeError, ValueError):
            self.max_steps = 30
        self.steps = 0
        self.finished = bool(policy.get("finished"))
        self.broken = ""
        self.log: list[dict] = []

    def left(self) -> int:
        return self.max_steps - self.steps

    def call(self, name: str, args: dict) -> Any:
        """Run one tool call as the next step and return its observation; None when the session cannot go on."""
        if self.finished or self.broken or self.steps >= self.max_steps:
            return None
        step = self.steps + 1
        envelope = {**self.binding, "step": step, "action": {"name": name, "args": args}}
        reply = None
        for attempt in range(4):
            result = _store.post_verbose("/environment/call", json_data=envelope)
            if isinstance(result.data, dict):
                reply = result.data
                break
            error = result.error or {}
            status, body = error.get("status"), str(error.get("body") or "")
            if status == 402:
                self.broken = "the run's inference key is exhausted"
                break
            if status == 409:
                expected = re.search(r"expected (\d+)", body)
                if expected and int(expected.group(1)) != step:
                    # our count drifted (a reply was lost): follow the store's count
                    step = int(expected.group(1))
                    self.steps = step - 1
                    envelope["step"] = step
                    continue
                if "finished" in body:
                    self.finished = True
                self.broken = f"the store refused the call: {body[:160]}"
                break
            if status == 422:
                self.broken = f"the store rejected the envelope: {body[:160]}"
                break
            sleep(1.0 + attempt)            # timeout or gateway trouble: the same step again replays safely
        if reply is None:
            self.broken = self.broken or "the store did not answer"
            self.log.append({"role": "assistant", "content": f"note: store call failed at step {step}: {self.broken}"})
            return None
        try:
            self.steps = int(reply.get("step") or step)
        except (TypeError, ValueError):
            self.steps = step
        self.finished = bool(reply.get("finished"))
        self.log.append({"role": "assistant", "content": "", "tool_call": {"name": name, "args": args}, "environment_result": reply})
        return reply.get("observation")


class Mind:
    """Model calls through the proxy: per-model fields, a fallback chain and the episode's spend."""

    def __init__(self, began: float) -> None:
        self.provider = (getenv("INFERENCE_PROVIDER") or "openrouter").strip().lower()
        self.began = began
        self.spent = 0.0
        self.room = 0.0                         # dollars this task may spend past SPEND_FINISH and SPEND_HARD (ROOM_*)
        self.calls = 0
        self.prompt_tokens = 0                  # prompt tokens sent, and how many of them the provider read from its cache
        self.cached_tokens = 0

    def chain(self, models: list[str] | None = None) -> list[str]:
        if self.provider != "openrouter":
            return list(OTHER_PROVIDER_MODELS)
        if models == CHECK_MODELS and self.spent >= SPEND_SOFT:
            return list(STEP_MODELS)        # past the soft limit the checks run on the step chain; the cheaper
                                            # reader, field and pointer chains keep their own order
        return list(models or STEP_MODELS)

    def ask(self, messages: list[dict], tools: list[dict] | None, max_tokens: int = 2500,
            models: list[str] | None = None, allowance: float = 0.0, tail: bool = False) -> dict | None:
        """The first usable assistant message from the chain (the step chain unless another is given), or None.

        A rate-limited (429) or failing upstream call is retried on the same model after a short pause: every task
        of a run shares one sandbox, and a run starts all its tasks at once, so a burst of calls can briefly exceed
        the proxy's rate cap. Moving straight to the next model in that moment would only meet the same cap."""
        for model in self.chain(models):
            for attempt in range(3):
                elapsed = monotonic() - self.began
                if elapsed > FORCE_AT or self.spent >= SPEND_HARD + self.room + allowance:
                    return None
                body: dict[str, Any] = {"model": model, "messages": self.dress(messages, model, tail), "temperature": 0,
                                        "max_tokens": max_tokens}
                if tools:
                    body["tools"] = tools
                    body["tool_choice"] = "auto"
                body.update(model_fields(model))
                result = (_mind_late if elapsed > WRAP_AT else _mind).post_verbose("/inference/chat/completions", json_data=body)
                self.calls += 1
                data = result.data
                if isinstance(data, dict):
                    break
                status = (result.error or {}).get("status") if isinstance(result.error, dict) else None
                if status == 402:
                    return None             # the run's key has no credit left: no model will answer
                transient = status in (408, 425, 429) or (isinstance(status, int) and status >= 500) or status is None
                if not transient or attempt == 2 or monotonic() - self.began > WRAP_AT - 10:
                    break
                sleep(2.0 + 2.0 * attempt + (1.0 if status == 429 else 0.0))
            if not isinstance(data, dict):
                continue
            usage = data.get("usage") or {}
            details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
            cached = details.get("cached_tokens") or usage.get("cache_read_input_tokens") or 0
            self.cached_tokens += cached if isinstance(cached, int) and not isinstance(cached, bool) else 0
            prompt = usage.get("prompt_tokens")
            self.prompt_tokens += prompt if isinstance(prompt, int) and not isinstance(prompt, bool) else 0
            cost = usage.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                self.spent += float(cost)
            else:
                price_in, price_out = PRICES.get(model, (2.0, 10.0))
                self.spent += (int(usage.get("prompt_tokens") or 0) * price_in
                               + int(usage.get("completion_tokens") or 0) * price_out) * 1e-6
            try:
                message = data["choices"][0]["message"]
            except (KeyError, IndexError, TypeError):
                continue
            if isinstance(message, dict) and (message.get("tool_calls") or message.get("content")):
                return message
        return None

    @staticmethod
    def dress(messages: list[dict], model: str, tail: bool = False) -> list[dict]:
        """Mark the fixed system text as cacheable for providers that cache by mark; with tail, also the newest message
        before the last one (the step conversation ends with a state note that changes every step), so the next step
        reads everything up to there from the cache."""
        if not model.startswith("anthropic/") or not messages or messages[0].get("role") != "system":
            return messages
        marked = [dict(messages[0])] + messages[1:]
        marked[0]["content"] = [{"type": "text", "text": str(marked[0].get("content") or ""), "cache_control": {"type": "ephemeral"}}]
        if tail and CACHE_TAIL:
            for index in range(len(marked) - 2, 0, -1):
                content = marked[index].get("content")
                if isinstance(content, str) and content:
                    marked[index] = dict(marked[index], content=[{"type": "text", "text": content,
                                                                   "cache_control": {"type": "ephemeral"}}])
                    break
        return marked


class Agent:
    """Works one task through the world's own tools."""

    def __init__(self, problem: dict) -> None:
        self.began = monotonic()
        self.session = Session(problem)
        self.mind = Mind(self.began)
        self.brief: dict = {}
        self.roles: dict[str, str] = {}
        self.question_budget: int | None = None
        self.step_cap = self.session.max_steps
        self.questions = 0
        self.qa: list[tuple[str, str]] = []
        self.notices: list[str] = []
        self.listings: dict[str, dict] = {}
        self.facts: dict[str, dict] = {}                  # listing id -> its property list as last seen
        self.cart_seen: tuple[int, str] | None = None     # (step, the cart's last view)
        self.cart_changed = 0                             # step of the latest successful cart change
        self.cart_obs: Any = None                         # the cart's last view, raw
        self.carted = False
        self.ask_stops = 0                                # choices sent back to ask the shopper first
        self.swaps = 0                                    # cart changes since the start or the last refused finish
        self.mixed: dict[str, bool] = {}                  # listing id -> a fact gives a value another key contradicts
        self.ask_plan: list[str] | None = None            # properties to ask about, ranked once by a model call
        self.auto_off = False                             # asking by code failed once: leave questions to the model
        self.asked_props: set[str] = set()                # words of the properties asked about by code
        self.pending_pages: tuple | None = None           # a search whose further pages code fetches after it
        self.ask_order: list[str] | None = None           # the properties code asks about, fixed when first needed
        self.read_first_stops = 0                         # questions sent back to read listings first (at most two)
        self.category_stops = 0                           # orders sent back for a listing outside the category meant
        self.told: list[tuple[int, str]] = []             # (step, "property: value") the shopper or a notice added
        self.blocks = 0                                   # orders stopped by the cart check
        self.view_stops = 0                               # orders sent back to view a cart changed since its last view
        self.empty_stops = 0                              # orders of a cart last seen empty, sent back
        self.gaps_raised: set[str] = set()                # finishing roles already sent back to check unread listings
        self.searches: list[tuple[int, str, Any, int]] = []  # (step, arguments, reported total, listings shown)
        self.sweeps: list[tuple[dict, Any, set[str]]] = []   # (search arguments, reported total, listing ids shown)
        self.finish_tries = 0
        self.refusals = 0
        self.price_refused: dict[str, int] = {}           # listing -> step of a refusal saying its price moved
        self.price_reread: set[str] = set()               # of those, listings whose price was read again since
        self.refs: list[dict] = []                        # needs whose value is a property of another listing (or anything but)
        self.record_text: dict[str, str] = {}             # listing id -> its fullest record seen, as lower-case text
        self.notes_text: dict[str, str] = {}              # listing id -> the prose of its record (seller notes, blurbs)
        self.more_pages: dict[str, dict] = {}             # search (arguments but the page) -> its total, rows seen, next marker
        self.compat: dict[str, dict] = {}                 # listing id -> model codes and works-with codes its text names
        self.fits: dict[str, tuple[list[dict], str]] = {}  # listing id -> the codes and families its fit sentences name, and the first
        self.quantity_text = ""                           # replies or notices that changed how many units are wanted
        self.first_asked: set[str] = set()                # the brief's ask-first questions already sent
        self.commit_stops = 0                             # cart changes sent back before the pick was confirmed
        self.after_cart_stops = 0                         # searches sent back while the cart held a pick
        self.news_step = 0                                # step of the latest notice or reply that changed something
        self.refusals_at_cart = 0                         # refusals so far at the latest cart change
        self.price_refusal_tries = -1                     # finish attempts at the latest refusal that only moved a price
        self.search_asked = False                         # a stop of ours asked for more searching (until the next finish)
        self.basket_plan_qa = 0                           # answers known when the basket questions were last planned
        self.basket_replans = 0
        self.messages: list[dict] = []
        self.short_of: dict[str, str] = {}                # tool_call_id -> summary shown once the result is old
        self.shortened = 0                                # oldest tool results shown as summaries (view, CACHE_SLACK)
        self.reqs: list[dict] = []                        # requirements as they stand: the brief's, then told and changed ones
        self.budget_text = ""                             # the text that last changed the budget
        self.items: list[dict] = []                       # the order's lines when the request asks for several items
        self.reviews = 0                                  # basket orders sent back by the basket check
        self.basket_plan: list[tuple[str, str]] | None = None   # (item, property) pairs to ask about, ranked once
        self.fields: dict[str, dict] = {}                 # tool -> the keys its results use (see learn_fields)
        self.field_tries: dict[str, int] = {}
        self.page_marks: dict[str, Any] = {}              # search tool -> what its last result gave for the next page
        self.recent: list[tuple[str, str, int]] = []       # (tool, arguments, notices seen) of the latest calls sent
        self.dropped: list[tuple[int, str]] = []          # (step, requirement) the shopper or a notice dropped
        self.decline_picks: dict[str, str] = {}           # item -> listing the decline check found fitting
        self.viewed: dict[str, set[str]] = {}             # listing id -> the store's listing views that showed it
        self.fact_views: set[str] = set()                 # single-listing views seen to show a listing's properties
        self.bare_views: set[str] = set()                 # single-listing views seen to show a listing without them
        self.sheet_size: dict[str, int] = {}              # single-listing view -> most properties it showed of a listing
        self.worded_searches = 0                          # searches that sent words or a category, not only a mode or a cap
        self.decline_judged = False                       # the decline check read the listings (its pick, or none, stands)
        self.last_carted: list[str] = []                  # listings named by the latest successful cart change
        self.row_notes: set[str] = set()                  # listings whose notes so far come only from a search row
        self.cart_book: list[str] = []                    # listings in the cart as code knows it: from cart changes, reset
                                                          # by every cart view (stores without a cart view have only this)
        self.procedures: list[dict] = []                  # the request's procedures: text, when, tell (a message to the shopper)
        self.msg_reserve = 0                              # messages kept for the procedures that tell the shopper before a finish
        self.told_finish: tuple[int, str] | None = None   # (step, finish) of the latest message a procedure asked for
        self.qa_kind: dict[int, str] = {}                 # question number -> the kind of reply the reader saw
        self.qa_prop: dict[int, set[str]] = {}            # question number -> the property words a code question asked
        self.plain_ask: set[str] = set()                  # properties to ask again on their own, with no examples
        self.plain_pairs: set[tuple[str, str]] = set()    # (item, property) of a basket to ask again the same way, once
        self.plain_all = False                            # a reply wanted one question per message: every later one plain
        self.decline_guesses: dict[str, str] = {}         # item -> the decline check's best-supported listing when none fits fully
        self.ruled_out: set[str] = set()                  # items whose every read listing the decline check found ruled out
        self.decline_reviews = 0                          # decline checks run
        self.reviewed_reads = 0                           # listings read at the latest decline check
        self.decline_read_steps = 0                       # steps code spent reading before a decline
        self.view_retried: set[str] = set()               # listings given back once after a call over the view's limit
        self.closest: dict[str, str] = {}                 # item -> the decline check's closest listing when nothing fits
        self.told_blocked: set[str] = set()               # items whose listings meet the request but miss a told need
        self.shelves: dict[str, str] = {}                 # a category name in lower case -> the store's own spelling of it
        self.cat_keys: set[tuple[str, str]] = set()       # (tool, argument) that took a category name the store uses
        self.empty_args: dict[tuple[str, str], int] = {}  # (tool, filter argument) -> searches with it that showed nothing
        self.empty_values: set[tuple[str, str, str]] = set()  # (tool, argument, value in lower case) that showed nothing
        self.args_worked: set[tuple[str, str]] = set()    # (tool, argument) that came back with listings at least once
        self.view_caps: dict[str, int] = {}               # several-listing view -> the most ids a call may name, once learned
        self.view_ok: dict[str, int] = {}                 # several-listing view -> the most listings one call gave back
        self.read_next: list[str] = []                    # listings named over a view's limit, not read yet
        self.bulk_tried: set[str] = set()                 # listings code already asked a view for
        self.cover: dict[str, set[str]] = {}              # listing -> what views returned of it: "props" (its property list),
                                                          # "page" (its own page: price, stock, seller or note)
        self.view_shown: dict[str, int] = {}              # several-listing view -> listings it showed
        self.view_props: dict[str, int] = {}              # several-listing view -> listings it showed with a property list
        self.view_page: dict[str, int] = {}               # view -> listings it showed with their own page
        self.view_stock: dict[str, int] = {}              # view -> listings it showed with their stock (whether it can be bought)
        self.pace: list[tuple[int, float, float]] = []    # (step, money spent, seconds) at each turn of the run loop
        self.exiting = False                              # code's own finish runs: only the store's step limit counts
        self.split_plain: set[int] = set()                # questions naming one property plainly that got "one at a time"
        self.record_views: dict[str, dict[str, str]] = {}  # listing -> view (or search) -> the text of its record there
        self.walk_props: set[str] = set()                 # listings the walk read the properties of, as maybe the one meant
        self.early_carts = 0                              # cart calls code made by the early-cart milestone
        self.code_put: list[str] = []                     # listings code itself put in the cart since the latest cart view
        self.code_asks: list[tuple[int, str]] = []        # (question number, property) of each question code asked
        self.reasked = False                              # the first key was asked again after "say again"
        self.ask_refusals = 0                             # code questions the checks sent back
        self.taken_out: list[str] = []                    # listings the latest removing cart change named
        self.bulk_steps = 0                               # steps code spent on its own batch reads
        self.bulk_turn = 0                                # code's batch reads since the model's last reply
        self.shelf_pages = 0                              # next pages code showed to reach the reads before a question
        self.read_step: dict[str, int] = {}               # listing id -> step its details were last read
        self.model_searched = False                       # the model's last reply searched
        self.ref_ids: set[str] = set()                    # listings a pointer led to: they define a value, never the item
        self.walk_steps = 0                               # steps code spent walking pointer chains
        self.walk_tries: dict[tuple[str, str], int] = {}  # (seller, described values) -> code's walk steps for that hop
        self.unproven: dict[str, int] = {}                # chain listings waiting for a view (CHAIN_PROOF) -> opens tried
        self.proof_reads = 0
        self.walk_searches: dict[tuple[str, str], int] = {}   # (seller, described values) -> code's searches for that hop
        self.ref_stops = 0                                # cart adds of such a listing sent back
        self.dup_stops = 0                                # one-item cart adds of a listing already held, sent back
        self.finish_asks = 0                              # calls sent back because the first finish was due
        self.depth_reads = 0                              # several items: code reads of a line's listings before its question
        self.depth_searched: set[str] = set()             # lines code searched for before their question
        self.depth_tried: set[tuple[str, str]] = set()    # (view, listing) those reads sent
        self.prompt_msgs: set[int] = set()                # indexes in qa of the whole-request message (FIRST_PROMPT_MESSAGE)
        self.prompt_tried = False                         # the whole-request message was sent or found not to fit
        self.prompt_cut = False                           # PROMPT_FIT: the messages left held only the start of the request
        self.mode_ok: dict[tuple[str, str], Any] = {}     # SEARCH_RETRY: (search tool, choice key) -> the value that brought rows
        self.mode_tried: set[tuple[str, str, str, str]] = set()   # (tool, key, the other arguments, value) sent again
        self.rows_stops = 0                               # ROWS_FIRST: declines sent back for want of any listing shown
        self.line_steps: dict[str, int] = {}              # several items: steps of searches and reads spent on each line
        self.line_stops = 0                               # searches for a line that already has a fit, sent back
        self.fit_reads = 0                                # several items: code reads of what a line's listings fit
        self.fit_tried: set[str] = set()                  # listings those reads asked for

    # -- reading the request ------------------------------------------------------------------------------------------
    def read_brief(self) -> None:
        session = self.session
        listing = "\n".join(f"- {name}: {str(spec.get('description') or '')[:300]} | parameters: "
                            + ", ".join(sorted(((spec.get('parameters') or {}).get('properties') or {}).keys()))
                            for name, spec in session.specs.items())
        text = (f"REQUEST:\n{session.request}\n\n"
                + (f"OPENING:\n{short(session.opening, 1500)}\n\n" if session.opening is not None else "")
                + f"TOOLS:\n{listing}\n\nHard limit on steps: {session.max_steps}. Call write_brief.")
        brief: dict = {}
        for attempt in range(3):
            reply = self.mind.ask([{"role": "system", "content": BRIEF_GUIDE}, {"role": "user", "content": text}],
                                  [BRIEF_TOOL], max_tokens=4000, models=CHECK_MODELS)
            calls = (reply or {}).get("tool_calls") or []
            if calls:
                parsed = parse_json((calls[0].get("function") or {}).get("arguments"))
                if isinstance(parsed, dict):
                    brief = parsed
                    break
            if reply is None:
                # every model failed (a burst of a run's tasks starting together, or an outage): the brief decides
                # what counts as right, so wait a little and read the request again while time allows
                if attempt == 2 or monotonic() - self.began > 60:
                    break
                sleep(3.0 + 3.0 * attempt)
        self.brief = brief

        def requirement_of(r: dict) -> dict:
            out = {"what": str(r.get("what") or "")[:80], "value": str(r.get("value") or "")[:120]}
            if isinstance(r.get("same_as"), dict):
                out["same_as"] = r["same_as"]
            return out
        self.reqs = [requirement_of(r) for r in brief.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]
        items = [i for i in brief.get("items") or [] if isinstance(i, dict) and str(i.get("item") or "").strip()]
        if len(items) >= 2:
            # several different items: each line keeps its own requirements and quantity
            for entry in items:
                quantity = entry.get("quantity")
                self.items.append({"item": str(entry["item"])[:80],
                                   "quantity": quantity if isinstance(quantity, int) and not isinstance(quantity, bool) and quantity >= 1 else 1,
                                   "requirements": [requirement_of(r) for r in entry.get("requirements") or []
                                                    if isinstance(r, dict) and str(r.get("value") or "").strip()]})
            self.reqs = [dict(r, item=entry["item"]) for entry in self.items for r in entry["requirements"]]
        names = tokens(brief.get("goal")) | set().union(*[tokens(e["item"]) for e in self.items] or [set()])
        single = brief.get("quantity") if isinstance(brief.get("quantity"), int) and not isinstance(brief.get("quantity"), bool) else None
        counts = {e["item"]: e["quantity"] for e in self.items}

        def restates_count(r: dict) -> bool:
            quantity = counts.get(r.get("item"), single) if self.items else single
            if self.items and r.get("item") is None:
                # a requirement of the flat list: the quantity of the item its words name
                named = self.item_named(str(r.get("what") or ""))
                quantity = counts.get(named) if named else None
            return count_only(r, names, quantity)
        dropped = [r for r in self.reqs if restates_count(r)]
        if dropped:
            self.reqs = [r for r in self.reqs if not restates_count(r)]
            for entry in self.items:
                entry["requirements"] = [r for r in entry["requirements"] if not count_only(r, names, entry["quantity"])]
            brief["requirements"] = [r for r in brief.get("requirements") or [] if not (isinstance(r, dict) and restates_count(r))]
            session.log.append({"role": "assistant", "content": "note: counts are quantities, not requirements: " + short(dropped, 300)})
        for requirement in self.reqs:
            # a value the request defines by another listing waits, as a pointer, until that listing is traced
            spec = requirement.pop("same_as", None)
            if isinstance(spec, dict):
                self.add_ref(requirement, requirement["what"], requirement.get("item") or "", spec, origin="brief")
        self.drop_ref_details(session.request, list(self.refs), list(self.reqs))
        for entry in brief.get("tools") or []:
            if isinstance(entry, dict) and entry.get("name") in session.specs and entry.get("role") in ROLES:
                self.roles[entry["name"]] = entry["role"]
        for name, spec in session.specs.items():
            self.roles.setdefault(name, self.guess_role(name, spec))
        request = session.request + " " + short(session.opening, 2000) if session.opening is not None else session.request
        for field in ("question_budget", "step_budget"):
            value = brief.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 and number_in(request, int(value)):
                if field == "question_budget":
                    self.question_budget = int(value)
                else:
                    self.step_cap = min(session.max_steps, max(1, int(value)))
        for entry in brief.get("procedures") or []:
            if isinstance(entry, str):
                entry = {"text": entry}
            if isinstance(entry, dict) and str(entry.get("text") or "").strip():
                when = str(entry.get("when") or "")
                self.procedures.append({"text": str(entry["text"]).strip()[:240], "when": when if when in PROCEDURE_TIMES else "any_time",
                                        "tell": entry.get("tell_shopper") is True})
        messaging = any(r == "message" for r in self.roles.values())
        # a procedure that tells the shopper something before the order needs a message of the budget, kept from the
        # questions; one before a decline is sent only when a message is left over (declines are rare, and what a
        # decline must say is mostly said by its own note)
        self.msg_reserve = 1 if messaging and any(p["tell"] and p["when"] == "before_order" for p in self.procedures) else 0
        if self.question_budget is None and messaging:
            # no stated limit found: a single question keeps clear of an unread limit
            self.question_budget = 1 + self.msg_reserve
        self.msg_reserve = min(self.msg_reserve, self.question_budget or 0)
        self.mind.room = min(ROOM_MAX, ROOM_PER_STEP * max(0, self.step_cap - ROOM_FROM))
        session.log.append({"role": "assistant", "content": "note: brief " + short(
            {**{k: v for k, v in brief.items() if k != "tools"}, "roles": self.roles, "question_budget_used": self.question_budget,
             "step_cap": self.step_cap, "messages_kept": self.msg_reserve, "spend_room": round(self.mind.room, 4)}, 1500)})

    def learn_fields(self, name: str, role: str, observation: Any) -> dict:
        """The keys a tool's results use for listing ids, titles, prices, stock and the rest, named once per tool by a
        small model call on its first result that holds more than an acknowledgement. Worlds rename every field, so
        key-name patterns alone miss many of them; keys the result does not hold are dropped."""
        if name in self.fields:
            return self.fields[name]
        if role in ("message", "order", "decline") or not isinstance(observation, (dict, list)):
            return {}
        keys = key_set(observation)
        if len(keys) < 4 or self.field_tries.get(name, 0) >= 2 or monotonic() - self.began > WRAP_AT - 60:
            return {}
        self.field_tries[name] = self.field_tries.get(name, 0) + 1
        spec = self.session.specs.get(name) or {}
        reply = self.mind.ask([{"role": "system", "content": FIELDS_GUIDE},
                               {"role": "user", "content": f"TOOL: {name}: {str(spec.get('description') or '')[:300]}\n\n"
                                                           f"RESULT:\n{short(trimmed(observation, 60, rows=2), 2500)}\n\nCall name_fields."}],
                              [FIELDS_TOOL], max_tokens=2500, models=FIELDS_MODELS)
        calls = (reply or {}).get("tool_calls") or []
        found = parse_json((calls[0].get("function") or {}).get("arguments")) if calls else None
        if not isinstance(found, dict):
            return {}                       # tried again on the tool's next result
        mapping: dict = {}
        for field in FIELD_ROLES:
            text = str(found.get(field) or "").strip()
            if field == "prose":
                mapping[field] = [k.strip() for k in text.split(",") if k.strip() in keys]
            elif text in keys:
                mapping[field] = text
        self.fields[name] = mapping
        self.session.log.append({"role": "assistant", "content": f"note: fields of {name} " + short(mapping, 400)})
        return mapping

    @staticmethod
    def guess_role(name: str, spec: dict) -> str:
        """A role from the tool's own words, when the brief gave none."""
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)       # a name in camel case reads as separate words
        words = f"{spaced} {spec.get('description') or ''}".lower().replace("_", " ")
        params = " ".join(((spec.get("parameters") or {}).get("properties") or {}).keys()).lower()
        if re.search(r"\bfinal\b|accepts no more|ends? the (?:task|chat|session)", words):
            if re.search(r"no purchase|give up|declin|without buying|abandon|no order", words):
                return "decline"
            return "order"
        if re.search(r"shopper|customer|question|\bask\b|message", words) and re.search(r"text|body|message|note|content|question", params):
            return "message"
        own = spaced.lower().replace("_", " ")
        if DECLINE_WORDS.search(own):
            return "decline"
        if re.search(r"\b(?:checkout|check out|order|buy|purchase|pay|settle)\b", own) \
                and not re.search(r"\b(?:add|put|keep|hold|remove|set|view|show|list)\b", own):
            return "order"
        if re.search(r"\bcart\b|\bbag\b|\bbasket\b", words):
            return "cart_change" if params.strip() else "cart_view"
        if re.search(r"search|find|look ?up|browse|filter|query|shelf|aisle|section|catalog", words):
            return "search"
        schema = spec.get("parameters") or {}
        required = [k for k in schema.get("required") or [] if isinstance((schema.get("properties") or {}).get(k), dict)]
        if len(required) == 1:
            # one tool that ends the task either way: its only required argument is a switch with a buying value and a
            # refusing one (call_role then reads the value each call chooses)
            values = [str(v).replace("_", " ") for v in schema["properties"][required[0]].get("enum") or []]
            if len(values) >= 2 and any(ORDER_WORDS.search(v) and not DECLINE_WORDS.search(v) for v in values) \
                    and any(DECLINE_WORDS.search(v) for v in values):
                return "order"
        if re.search(r"detail|propert|sheet|listing|item|product", words):
            return "details"
        return "other"

    # -- the state the model sees ---------------------------------------------------------------------------------------
    def reserve(self, pages: bool = True) -> int:
        """Steps the finish still needs: the finishing call, plus the cart fixes when the cart decides the order (one per
        line of a several-item order when the cart tool takes one line a call), plus (PAGE_RESERVE, unless pages is off)
        a read of each line's own page still unopened (page_debt)."""
        cart_tools = [n for n, r in self.roles.items() if r == "cart_change"]
        told = 1 if self.reserve_left() else 0          # the message a procedure asks for right before the finish
        if not cart_tools:
            return 1 + told
        lines = len(self.items) if self.items and not any(self.list_shaped(n) for n in cart_tools) else 1
        return 1 + told + min(4, lines) + (self.page_debt() if pages and PAGE_RESERVE else 0)

    def page_debt(self) -> int:
        """PAGE_RESERVE: the own pages still to open before the order, one a line: each cart line no single-listing view
        has shown a page of, and each line with nothing in the cart yet; at most four. 0 without a single-listing view."""
        single, _ = self.listing_views()
        if not single:
            return 0
        held = list(dict.fromkeys(self.cart_book))
        unopened = sum(1 for i in held if not any(t in single and self.view_page.get(t) for t in self.viewed.get(i, set())))
        return min(4, unopened + max(0, (len(self.items) if self.items else 1) - len(held)))

    def reserve_left(self) -> int:
        """Messages still kept for the procedures that tell the shopper something before the finish."""
        return self.msg_reserve if self.told_finish is None else 0

    def ask_room(self) -> int:
        """Questions the shopper may still be asked, less the messages kept for the request's procedures."""
        return max(0, (self.question_budget or 0) - self.questions - self.reserve_left())

    def pacing(self) -> int:
        """Steps left at which the first finish should go out: what finishing takes plus what a refused first finish
        needs to recover (see SPARE_STEPS), less in a short task. From there on, checks that send an order back for more
        research stay quiet."""
        return self.reserve() + min(SPARE_STEPS - 2, self.step_cap // 4)

    def item_reads(self) -> list[str]:
        """Read listings that can be the item to buy: never a listing a pointer led to (it only defines a value), and for
        one item, once two or more read listings sit under the item's category, only those, and before that none filed with a
        pointer's listing under another category."""
        ids = [i for i in self.facts if i not in self.ref_ids]
        cats = self.item_categories()
        if cats and not self.items:
            inside = [i for i in ids if (self.listings.get(i) or {}).get("category") in cats]
            if len(inside) >= 2:
                return inside
            # listings filed with a pointer's listing under a category the item is not filed under are its neighbours
            beside = {(self.listings.get(i) or {}).get("category") for i in self.ref_ids} - cats - {None, ""}
            ids = [i for i in ids if (self.listings.get(i) or {}).get("category") not in beside]
        return ids

    def ref_allowed(self, ident: str) -> bool:
        """A listing a pointer led to may still be bought only when it sits under the item's category and no fact of it
        contradicts a requirement. REF_CANDIDATE: it is then an ordinary candidate, ranked like any other, unless it is
        the first listing the request itself names, its own note sends the property on, or a pairing gives the item the
        other value; otherwise only when no other read listing is left that nothing contradicts."""
        cats = self.item_categories()
        if cats and (self.listings.get(ident) or {}).get("category") not in cats:
            return False

        def broken(i: str) -> bool:
            return any(c.split(": ", 1)[-1].startswith(("no", "breaks")) for c in self.marks(i)[0])
        if broken(ident):
            return False
        if REF_CANDIDATE:
            for ref in self.refs:
                if ident == ref.get("first") and ref.get("origin") == "brief":
                    return False
                if ident in ref["hops"] or (ref.get("source") or {}).get("id") == ident and (ref.get("source") or {}).get("paired"):
                    return False
                if ref["value"] is None and ident in self.ref_ids:
                    return False            # its chain is not traced yet: what it holds is not known
            return True
        return not any(i not in self.ref_ids and not sold_out((self.listings.get(i) or {}).get("stock")) and not broken(i)
                       for i in self.facts)

    def has_candidate(self) -> bool:
        """True when something could be ordered now: a cart already used, a read listing that fits or that nothing
        rules out (one item), or a read listing for every line (several items). Without one, the first-finish point
        does not stop the research: a finish with nothing found can only decline, which is almost never right."""
        if self.carted:
            return True
        if self.items:
            return bool(self.fallback_lines())
        return bool(self.fit_list() or self.ranked_candidates())

    def list_shaped(self, name: str) -> bool:
        """True when a tool takes a list of objects, as a cart tool that sets every line in one call does."""
        properties = ((self.session.specs.get(name) or {}).get("parameters") or {}).get("properties") or {}
        return any(isinstance(v, dict) and "array" in kinds_of(v) and isinstance(v.get("items"), dict)
                   and "object" in kinds_of(v["items"]) for v in properties.values())

    def call_role(self, name: str, args: dict) -> str:
        """The role of one call: a finishing tool with a switch (buy or refuse) orders or declines by the value chosen."""
        role = self.roles.get(name, "other")
        if role not in ("order", "decline"):
            return role
        properties = ((self.session.specs.get(name) or {}).get("parameters") or {}).get("properties") or {}
        for key, field in properties.items():
            if key in args and isinstance(field, dict) and isinstance(field.get("enum"), list) and len(field["enum"]) >= 2:
                meaning = self.switch_role(name, key, args[key])
                if meaning:
                    return meaning
        return role

    def switch_role(self, name: str, key: str, value: Any) -> str:
        """What a finishing tool's switch value does: "order", "decline", or "" when unclear. The value's own words
        decide first ("refuse", "purchase"), then the clause of the tool's or the field's description that names it
        ("'pass' leaves without paying"), then, for a two-valued switch, the opposite of the other value."""
        spec = self.session.specs.get(name) or {}
        field = (((spec.get("parameters") or {}).get("properties") or {}).get(key)) or {}
        values = [v for v in field.get("enum") or [] if isinstance(v, (str, int))]

        def by_words(v: Any) -> str:
            text = str(v).replace("_", " ")
            if DECLINE_WORDS.search(text):
                return "decline"
            return "order" if ORDER_WORDS.search(text) else ""

        def by_description(v: Any) -> str:
            text = f"{spec.get('description') or ''}. {field.get('description') or ''}"
            for clause in re.split(r"[.;:()]|,\s|\s-\s", text):
                if not re.search(rf"(?<![\w-]){re.escape(str(v))}(?![\w-])", clause, re.I):
                    continue
                if re.search(r"without|leave|declin|refus|reject|cancel|abandon|give ?up|no (?:purchase|order|buy)|"
                             r"not (?:buy|pay|order|purchase)|nothing|skip|walk away", clause, re.I):
                    return "decline"        # "leaves without paying": the negation decides
                if re.search(r"pay|buy|purchas|order|check ?out|commit|complet|confirm|accept|place|proceed|settle", clause, re.I):
                    return "order"
            return ""
        meaning = by_words(value) or by_description(value)
        if not meaning and len(values) == 2 and value in values:
            other = values[1] if values[0] == value else values[0]
            opposite = by_words(other) or by_description(other)
            meaning = {"order": "decline", "decline": "order"}.get(opposite, "")
        return meaning

    def hard_left(self) -> int:
        """Steps the store still allows."""
        return min(self.session.left(), self.step_cap - self.session.steps)

    def steps_left(self) -> int:
        """Steps left to plan with: what the store allows, and with AFFORD_STEPS no more than the money and time left
        buy at the recent pace. Code's own finish counts only the store's steps (it asks no model)."""
        hard = self.hard_left()
        return hard if not AFFORD_STEPS or self.exiting else min(hard, self.afford())

    def afford(self) -> int:
        """Steps the money and the time left buy, at the median cost and duration per step over the run loop's last
        eight turns (a single dear turn, a check by the larger model say, does not set the pace); a large number until
        six turns are known."""
        marks = self.pace[-9:]
        if len(marks) < 6:
            return 10 ** 6
        money, seconds = [], []
        for (s0, m0, t0), (s1, m1, t1) in zip(marks, marks[1:]):
            if s1 > s0:
                money.append((m1 - m0) / (s1 - s0))
                seconds.append((t1 - t0) / (s1 - s0))
        if len(money) < 4:
            return 10 ** 6
        money.sort()
        seconds.sort()
        rate, pace = max(money[len(money) // 2], 0.0002), max(seconds[len(seconds) // 2], 0.5)
        by_money = (SPEND_HARD + self.mind.room - self.mind.spent) / rate
        by_time = (FORCE_AT - (monotonic() - self.began)) / pace
        return max(0, int(min(by_money, by_time)))

    def priority_measure(self) -> tuple[str, str] | None:
        """A priority for the most or the least of a measure other than the price: (measure, "higher" or "lower")."""
        found = re.match(r"\s*best\s*:\s*(.+?)\s+(higher|lower)\s*$", str(self.brief.get("priority") or ""), re.I)
        if not found or re.search(r"\b(?:price|cost)\b", found.group(1), re.I):
            return None
        return found.group(1).strip(), found.group(2).lower()

    def cheapest_asked(self) -> bool:
        """The request itself asks for the lowest price among fitting listings."""
        if MEASURE_SORT and self.priority_measure():
            return False
        return bool(re.search(r"cheap|least|lowest|low(?:er)? price|min", str(self.brief.get("priority") or ""), re.I))

    def cheapest_wanted(self) -> bool:
        """The cheapest fitting listing is the pick: asked for, or DEFAULT_CHEAPEST when the request names no other measure
        to choose by."""
        return self.cheapest_asked() or (DEFAULT_CHEAPEST and not (MEASURE_SORT and self.priority_measure()))

    def measure_value(self, ident: str, measure: str) -> tuple[float, str] | None:
        """MEASURE_SORT: the number a listing's fact for the measure gives, with its unit: the fact whose name shares the
        most words with the measure, its first number."""
        want = tokens(measure) - FILLER
        sheet = self.facts.get(ident) or {}
        key = max(sheet, key=lambda k: len(tokens(k) & want), default=None)
        if key is None or not tokens(key) & want:
            return None
        found = self.figures(sheet[key])
        return found[0] if found else None

    def requirement_words(self) -> set[str]:
        words: set[str] = set()
        for requirement in self.brief.get("requirements") or []:
            if isinstance(requirement, dict):
                # a value defined by another listing: the words describing that listing are not the item's
                words |= tokens(requirement.get("what")) | (set() if requirement.get("same_as") else tokens(requirement.get("value")))
        return words

    def item_of(self, ident: str) -> str:
        """The order's item a read listing would fill: the item whose name shares the most words with the listing's
        category, then its title; "" when none does."""
        info = self.listings.get(ident) or {}
        best, score = "", (0, 0)
        for entry in self.items:
            words = tokens(entry["item"]) - FILLER
            here = (len(words & tokens(info.get("category") or "")), len(words & tokens(info.get("title") or "")))
            if here > score:
                best, score = entry["item"], here
        return best

    def described_properties(self) -> str:
        """For an order of several items: the properties each item's read listings describe, with sample values."""
        parts = []
        for entry in self.items:
            ids = {i for i in self.facts if self.item_of(i) == entry["item"]}
            stats = [(n, c, v) for n, c, v in self.property_stats(ids) if not self.label_like(c, v)][:8]
            if stats:
                parts.append(f"{entry['item']}: " + ", ".join(f"{n} ({', '.join(v[:3])})" for n, _, v in stats))
        return " | ".join(parts)

    def property_stats(self, ids: set[str] | None = None) -> list[tuple[str, int, list[str]]]:
        """The properties the listings' facts describe (all read listings, or those in ids), most widely given first:
        (name, listings giving it, its values)."""
        holders: dict[str, set[str]] = {}
        values: dict[str, list[str]] = {}
        spelled: dict[str, dict[str, int]] = {}           # group -> the full names it was written as, with counts
        # a listing a pointer led to describes another item: its properties are no question for this one
        allowed = set(self.item_reads()) if ids is None else set(ids) - self.ref_ids
        for ident, sheet in self.facts.items():
            if ident not in allowed:
                continue
            named: set[str] = set()
            for key, value in sheet.items():
                spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key))      # a key written in camel case reads as separate words
                parts = [p for p in re.split(r"[^a-z0-9]+", spaced.lower()) if p and not p.isdigit()]
                if len(parts) >= 2 and parts[-1] in ("filter", "filters", "value", "values"):
                    parts = parts[:-1]      # a search-facet copy of a property ("<name>_filter") is that property
                if not parts or len(value) > 60:
                    continue
                # a generic last word ("<x> type", "<x> height") keeps the word before it
                name = " ".join(parts[-2:]) if len(parts) >= 2 and parts[-1] in (
                    "type", "style", "kind", "class", "size", "level", "length", "height", "shape", "family") else parts[-1]
                if name in named:
                    continue
                named.add(name)
                holders.setdefault(name, set()).add(ident)
                full = " ".join(parts) if len(parts) <= 4 else name
                spelled.setdefault(name, {})[full] = spelled.get(name, {}).get(full, 0) + 1
                seen = values.setdefault(name, [])
                if value.lower() not in seen:
                    seen.append(value.lower())
        # each property is shown under the full name the listings use most, not only its last word: a question
        # names a held-back need only with the property's own name
        return [(max(spelled[n], key=lambda s: spelled[n][s]), len(holders[n]), values[n])
                for n in sorted(holders, key=lambda n: -len(holders[n]))]

    def attribute_tally(self) -> str:
        """The properties the listings' facts describe: how many listings give each, and some of their values."""
        return "; ".join(f"{n} ({count}: {', '.join(vals[:3])})" for n, count, vals in self.property_stats()[:10])

    @staticmethod
    def label_like(count: int, values: list[str]) -> bool:
        """A property that only names a listing's variants: nearly every listing has its own value, or most values are
        combinations such as "teal / 6-7y" or "m - 40g"."""
        combos = sum(1 for v in values if re.search(r" / |/ |:| - ", v))
        # variant combinations ("teal / 6-7y", "zz:7", "m - 40g") are labels; diverse words or sizes are preferences
        return values != [] and combos >= 0.4 * len(values)

    def reads_to_ask(self) -> int:
        """Listings read in detail before the first question of a one-item order. A held-back need sits on a property
        most of the item's listings carry, and five read listings show which those are far more often than three.
        Three are enough once the model tries to cart or finish, from step READ_WAIT_ENDS, when the steps left would
        not cover the missing reads before the questions, or when no unread listing that may fit is left."""
        if ASK_BY and not self.items and self.session.steps >= max(self.wait_ends() + 2, int(self.step_cap * ASK_BY)):
            return 0                        # late: the questions go out with what was read (a task that starts from a
                                            # listing the request names may have read none of the item's own)
        if self.items or self.ask_plan is not None or self.ask_stops or self.session.steps >= self.wait_ends() \
                or not any(r == "details" for r in self.roles.values()):
            return 3
        missing = READS_TO_ASK - self.ask_reads()
        if self.steps_left() <= self.reserve() + 6 + self.ask_room() + missing or not (self.unread_left() or self.shelf_more()):
            return 3
        return READS_TO_ASK

    def basket_reads_short(self) -> bool:
        """Before the first question of a several-item order: fewer listings of the items read than basket_question waits
        for, while it still waits."""
        if not self.items or self.questions or self.session.steps >= max(14, self.wait_ends()):
            return False
        enough = min(3 * len(self.items) + 1, 13) if self.wait_ends() > READ_WAIT_ENDS else min(len(self.items) + 2, 8)
        return len(self.item_reads()) < enough

    def wait_ends(self) -> int:
        """The step from which the first question no longer waits for reads (READ_WAIT_ENDS; later in a long task)."""
        return max(READ_WAIT_ENDS, self.step_cap // 4) if LONG_WAIT else READ_WAIT_ENDS

    def ask_reads(self) -> int:
        """Listings read toward the first question of a one-item order: those filed under the item's category once one
        is known (a held-back need is a property the item's own category carries, and pages of other categories do not
        show it), else every listing that can be the item. From READ_WAIT_ENDS on, or once the model tries to cart or
        finish, every such listing counts."""
        ids = self.item_reads()
        if self.page_views():
            # the store splits a listing's properties from its own page: only reads of the properties view show which
            # properties the item's listings share (a page's few facts would end the reads before the questions)
            props, many = self.props_view(), set(self.listing_views()[1])
            ids = [i for i in ids if props in self.viewed.get(i, set()) or "props" in self.cover.get(i, set())
                   or self.viewed.get(i, set()) & {t for t in many if not self.list_bare(t)}]
        cats = self.item_categories()
        if self.items or not cats or self.ask_stops or self.session.steps >= self.wait_ends():
            return len(ids)
        return sum(1 for i in ids if (self.listings.get(i) or {}).get("category") in cats)

    def unread_left(self) -> bool:
        """True when a listing shown so far may be the item (its title or category shares a word with the goal or a
        required value), is not sold out, and was never read in detail."""
        words = tokens(self.brief.get("goal")) | set().union(*[tokens(r.get("value")) for r in self.reqs if not r.get("pending_ref")] or [set()])
        return any(record["id"] not in self.facts and not sold_out(record["stock"])
                   and (tokens(record["title"]) | tokens(record.get("category"))) & words for record in self.listings.values())

    def coverage_order(self) -> list[str]:
        """The properties the read listings describe, ranked the way held-back needs are found most often: the most
        widely carried first and, among equals, the one whose values repeat most (listings per value). With WEIGHTED_ORDER
        a property whose values are mostly measures, lists or long phrases counts half its coverage, and more distinct
        values add a little (0.1 a doubling): replayed on recorded stores this put a held-back need among the first four
        questions more often in each of them. A property with one value across them comes last; one the request fixes,
        a variant or model label, and one a single listing gives are left out."""
        stats = self.property_stats()
        pages = max(1, len(self.item_reads()))

        def weak(values: list[str]) -> bool:
            return sum(bool(re.search(r"\d", v)) for v in values) > 0.5 * len(values) \
                or sum(", " in v or ";" in v for v in values) >= 0.3 * len(values) \
                or sum(len(v.split()) for v in values) >= 2.5 * len(values)

        if WEIGHTED_ORDER:
            ranked = sorted(stats, key=lambda s: (len(s[2]) < 2, -(s[1] / pages) * (0.5 if s[2] and weak(s[2]) else 1.0)
                                                  - 0.1 * math.log2(max(1, len(s[2]))), -s[1]))
        else:
            ranked = sorted(stats, key=lambda s: (len(s[2]) < 2, -s[1], -s[1] / max(1, len(s[2]))))
        return [n for n, c, v in ranked if c >= 2 and not self.label_like(c, v) and not self.stated_property(n)
                and not re.fullmatch(r"(?:options?|variants?|variations?|models?|skus?)", n.strip())]

    def plan_questions(self) -> None:
        """Once enough listings were read, one model call ranks the properties to ask about from the request's hints and
        the property names the store actually uses."""
        if self.ask_plan is not None or self.ask_reads() < self.reads_to_ask() or self.ask_room() <= 0 or self.items:
            return                          # an order of several items: the model asks, naming the item
        if not any(r == "message" for r in self.roles.values()) or monotonic() - self.began > WRAP_AT - 60:
            return
        self.ask_plan = []
        # a property naming only a listing's variants or model is not a preference
        stats = [(n, c, v) for n, c, v in self.property_stats() if not self.label_like(c, v)][:18]
        listing = "\n".join(f"- {n} ({c} listings): {', '.join(v[:5])}" for n, c, v in stats) or "(none)"
        asked = "\n".join(f"- {q} -> {a}" for q, a in self.qa) or "(none)"
        reply = self.mind.ask([{"role": "system", "content": ASK_GUIDE},
                               {"role": "user", "content": f"REQUEST:\n{self.session.request}\n\nPROPERTIES THE LISTINGS DESCRIBE:\n{listing}"
                                                           f"\n\nALREADY ASKED:\n{asked}\n\nCall rank_properties."}],
                              [ASK_TOOL], max_tokens=1500, models=CHECK_MODELS)
        calls = (reply or {}).get("tool_calls") or []
        found = parse_json((calls[0].get("function") or {}).get("arguments")) if calls else None
        if isinstance(found, dict):
            self.ask_plan = [str(p).strip().lower()[:40] for p in found.get("properties") or [] if str(p).strip()][:6]
            if ASK_FALLBACK and stats:
                # each name as the store writes it; a name no read listing carries goes, unless the request names it
                names = [n for n, _, _ in stats]
                request = tokens(self.session.request)
                mapped = []
                for p in self.ask_plan:
                    hit = next((n for n in names if tokens(n) == tokens(p)), None) \
                        or next((n for n in names if tokens(p) and (tokens(p) <= tokens(n) or tokens(n) <= tokens(p))), None)
                    if hit or (tokens(p) and tokens(p) <= request):
                        mapped.append(hit or p)
                self.ask_plan = list(dict.fromkeys(mapped))
        self.session.log.append({"role": "assistant", "content": "note: question plan " + short(self.ask_plan, 300)})

    def message_key(self, tool: str) -> str | None:
        """The message tool's text field: a free-text field whose name says text, body, message, note and the like, else
        a required one, else the first."""
        schema = (self.session.specs.get(tool) or {}).get("parameters") or {}
        properties = schema.get("properties") or {}
        strings = [k for k, v in properties.items() if isinstance(v, dict) and "string" in kinds_of(v) and not v.get("enum")]
        return next((k for k in strings if re.search(r"text|body|message|note|content|question|query", k, re.I)), None) \
            or next((k for k in strings if k in (schema.get("required") or [])), None) or (strings[0] if strings else None)

    def message_args(self, tool: str, text: str) -> dict | None:
        """Arguments for the message tool: the text in its text field, other required fields from their schema."""
        schema = (self.session.specs.get(tool) or {}).get("parameters") or {}
        properties = schema.get("properties") or {}
        field = self.message_key(tool)
        if field is None:
            return None
        args: dict = {field: text}
        for key in schema.get("required") or []:
            if key in args:
                continue
            spec = properties.get(key) or {}
            kinds = kinds_of(spec)
            if isinstance(spec.get("enum"), list) and spec["enum"]:
                args[key] = spec["enum"][0]
            elif "integer" in kinds or "number" in kinds:
                args[key] = spec.get("minimum") or 1
            elif "boolean" in kinds:
                args[key] = False
            else:
                return None                 # an unknown required field: leave questions to the model
        args, problems = fit_args(schema or {"type": "object"}, args)
        return None if problems else args

    def plan_basket_questions(self) -> None:
        """For an order of several items: one call ranks (item, property) pairs to ask about, from the request's hints
        and the property names each item's read listings use."""
        self.basket_plan = []
        self.basket_plan_qa = len(self.qa)
        listing = []
        for entry in self.items:
            ids = {i for i in self.facts if self.item_of(i) == entry["item"]}
            # each property with its spread over the item's read listings: how many carry it and how many values they
            # show; a field that only lists a listing's variants is no preference
            stats = [(n, c, v) for n, c, v in self.property_stats(ids) if not self.label_like(c, v)
                     and not re.fullmatch(r"(?:options?|variants?|variations?|models?|skus?)", n.strip())][:12]
            listing.append(f"- {entry['item']} ({len(ids)} read): " + ("; ".join(
                f"{n} ({c} listings, {len(v)} value{'s' if len(v) != 1 else ''}: {', '.join(v[:4])})" for n, c, v in stats)
                or "(nothing read yet)"))
        fixed = "; ".join(f"{r.get('item') or ''} > {r['what']}: {r['value']}" for r in self.reqs) or "(none)"
        asked = "\n".join(f"- {q} -> {a}" for q, a in self.qa) or "(none)"
        reply = self.mind.ask([{"role": "system", "content": ASK_GUIDE + BASKET_ASK_NOTE},
                               {"role": "user", "content": f"REQUEST:\n{self.session.request}\n\nREQUIREMENTS ALREADY FIXED:\n{fixed}"
                                                           f"\n\nPROPERTIES EACH ITEM'S LISTINGS DESCRIBE (sample values):\n" + "\n".join(listing)
                                                           + f"\n\nALREADY ASKED:\n{asked}\n\nCall rank_questions."}],
                              [BASKET_ASK_TOOL], max_tokens=2500, models=CHECK_MODELS)
        calls = (reply or {}).get("tool_calls") or []
        found = parse_json((calls[0].get("function") or {}).get("arguments")) if calls else None
        for pair in (found or {}).get("questions") or [] if isinstance(found, dict) else []:
            if isinstance(pair, dict) and str(pair.get("property") or "").strip():
                item = self.item_named(str(pair.get("item") or "")) or str(pair.get("item") or "").strip()[:60]
                self.basket_plan.append((item, str(pair["property"]).strip().lower()[:40]))
        self.session.log.append({"role": "assistant", "content": "note: basket question plan " + short(self.basket_plan, 400)})

    def basket_question(self) -> bool:
        """For an order of several items: once a listing of every item was read, ask the planned (item, property) pairs
        by code, one step each, naming the item. Returns True when a question was sent."""
        if self.auto_off or self.ask_room() <= 0 or self.session.finished or self.session.broken:
            return False
        if self.steps_left() <= self.reserve() + 6 + self.ask_room() or monotonic() - self.began > WRAP_AT - 60:
            return False
        tool = next((n for n, r in self.roles.items() if r == "message"), None)
        if tool is None or not self.facts:
            return False
        reads: dict[str, int] = {}
        mine = self.item_reads()
        for ident in mine:
            reads[self.item_of(ident)] = reads.get(self.item_of(ident), 0) + 1
        wait = self.wait_ends()
        enough = min(3 * len(self.items) + 1, 13) if wait > READ_WAIT_ENDS else min(len(self.items) + 2, 8)
        if (any(reads.get(e["item"], 0) < 1 for e in self.items) or len(mine) < enough) \
                and self.session.steps < max(14, wait):
            return False                    # read listings of each item first: questions then use the store's names
        # early on, an item read fewer than three times waits for more listings (its questions then name a property
        # its own listings share, where a need sits) while items read enough go first; the wait ends at step 10 or when
        # the model tries to cart or order
        early = self.session.steps < wait and not self.ask_stops
        thin = {e["item"] for e in self.items if reads.get(e["item"], 0) < 3} if early else set()
        if self.told and self.told[-1][0] >= self.session.steps:
            return False                    # a need was just told: let the search follow it before the next question
        if self.basket_plan is not None and len(self.qa) > self.basket_plan_qa and self.basket_replans < 3:
            # a new answer came in: plan the next question knowing it and the listings read since
            self.basket_plan = None
            self.basket_replans += 1
        if self.basket_plan is None:
            self.plan_basket_questions()
        # every item gets its first question before any item gets a second, and an item that told a need waits until
        # the others were asked: each item may hold a need, and four questions rarely cover five items twice
        per_item: dict[str, int] = {}
        for q, _ in self.answered_qa():
            named = self.item_named(q)
            per_item[named] = per_item.get(named, 0) + 1
        told_items = {self.item_named(need) for _, need in self.told} - {""}
        pending = [(n, item, prop) for n, (item, prop) in enumerate(self.basket_plan or [])
                   if self.asked_about(prop) and not self.already_asked(item, prop) and (self.item_named(item) or item) not in thin
                   and not (BASKET_ASK and self.line_stated(self.item_named(item) or item, prop))]
        # an item waiting for its second read keeps its turn: while it has had no question, no other item gets a second one
        if any(not per_item.get(item) for item in thin):
            pending = [p for p in pending if not per_item.get(self.item_named(p[1]) or p[1])]
        if not pending:
            return False
        # an item whose stated requirements leave one listing at most can hardly hold a need (one more rule would not
        # change the answer), so items with a real choice go first
        choice = {e["item"]: sum(1 for i, r in self.listings.items() if self.item_of(i) == e["item"] and not sold_out(r.get("stock")))
                  for e in self.items}
        # BASKET_ASK: a line whose latest reply asked to say it again waits while another line has a question left
        last_kind: dict[str, str] = {}
        for n, (q, _) in enumerate(self.qa):
            last_kind[self.item_named(q)] = self.qa_kind.get(n, "")
        again = {i for i, kind in last_kind.items() if kind == "not_understood"} if BASKET_ASK else set()
        _, item, prop = min(pending, key=lambda p: (self.item_named(p[1]) in told_items, choice.get(self.item_named(p[1]) or p[1], 0) < 2,
                                                    (self.item_named(p[1]) or p[1]) in again,
                                                    per_item.get(self.item_named(p[1]), 0), p[0]))
        values = self.example_values(prop, item)
        plain = (self.item_named(item) or item, prop) in self.plain_pairs or self.plain_all
        example = f" (for example {' or '.join(values)})" if ASK_EXAMPLES and len(values) == 2 and not plain else ""
        # after a reply wanting one question per message: the property alone, and the item's name cut to its first
        # part, since a name listing several kinds ("teapots, trivets & birdhouses") reads as several questions
        name = re.split(r",|&|/|\band\b|\bor\b", item)[0].strip() or item if plain else item
        args = self.message_args(tool, f"Which {prop} would you like for the {name}{example}?" if item else
                                 f"Which {prop} would you like{example}?")
        if args is None:
            self.auto_off = True
            return False
        call = {"id": f"ask-{self.session.steps}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        result = self.act(call)
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        self.short_of[call["id"]] = self.summary(result)
        if result.startswith('{"not_sent"'):
            self.auto_off = True
            return False
        if self.qa_kind.get(len(self.qa) - 1) == "one_at_a_time":
            pair = (self.item_named(item) or item, prop)
            same = not example and (re.split(r",|&|/|\band\b|\bor\b", item)[0].strip() or item) == item
            if pair in self.plain_pairs or (PLAIN_SPLIT and same):
                # asked plainly once more, or the same words would go again: no need there after all
                self.qa_kind[len(self.qa) - 1] = "not_understood"
                self.split_plain.add(len(self.qa) - 1)
            else:
                self.plain_pairs.add(pair)
        return True

    def stated_property(self, name: str) -> bool:
        """True when a requirement as it stands sets this property's value, under its name alone or within a longer
        one ("glaze" within "glaze tone"): a question about it can only repeat the request. A narrower name ("lid
        glaze" under a stated glaze) stays open, and so does a property the request only rules a value out of or
        bounds ("no speckled glaze", "at most 20 cm"): the shopper may still want one value in what is left."""
        words = tokens(name)
        return bool(words) and any(words <= tokens(r.get("what")) for r in self.reqs
                                   if not NEGATION.search(f"{r.get('what')} {r.get('value')}")
                                   and not COMPARE.search(str(r.get("value") or "")))

    def line_stated(self, item: str, prop: str) -> bool:
        """BASKET_ASK: a requirement of this line (or of the whole order) as it stands sets this property's value, so a
        question about it can only repeat the request; a value ruled out or a bound leaves it open, as stated_property."""
        words = tokens(prop) - tokens(item)
        return bool(words) and any(words <= tokens(r.get("what")) - tokens(item) for r in self.reqs
                                   if r.get("item") in (None, "", item) and not r.get("pending_ref")
                                   and not NEGATION.search(f"{r.get('what')} {r.get('value')}")
                                   and not COMPARE.search(str(r.get("value") or "")))

    def answered_qa(self) -> list[tuple[str, str]]:
        """The questions and replies so far, less those the shopper sent back for asking about more than one thing: their
        property may be asked again, alone."""
        return [(q, a) for n, (q, a) in enumerate(self.qa) if n not in self.prompt_msgs
                and (n in self.split_plain or (not ONE_AT_A_TIME.search(str(a)) and self.qa_kind.get(n) != "one_at_a_time"))]

    def already_asked(self, item: str, prop: str) -> bool:
        """True when a question already asked about this property, for this item when the order holds several."""
        words = self.asked_about(prop)
        for question, _ in self.answered_qa():
            asked = self.asked_about(question)
            if not asked or not (words == asked or words <= asked):
                continue
            if not self.items or not item or self.item_named(question) in ("", self.item_named(item) or item):
                return True
        return False

    def example_values(self, prop: str, item: str = "") -> list[str]:
        """Two short values the read listings give for a property (those of the item's listings when the order holds
        several), for a question that offers examples."""
        ids = {i for i in self.facts if self.item_of(i) == (self.item_named(item) or item)} if self.items and item else None
        values = next((v for n, _, v in self.property_stats(ids) if n == prop), [])
        return [v for v in values if len(v) <= 20 and "," not in v][:2]

    def asked_sets(self) -> list[set[str]]:
        """The property words of each question asked and answered so far: the property code asked, else the words the
        question names (question words and the items' names left out)."""
        mine = dict(self.code_asks)
        out = []
        for n, (q, a) in enumerate(self.qa):
            if n in self.prompt_msgs:
                continue                    # the whole request sent back to the shopper asks about no one property
            if n not in self.split_plain and (ONE_AT_A_TIME.search(str(a)) or self.qa_kind.get(n) == "one_at_a_time"):
                continue                    # sent back for asking several things: not an answer
            out.append(tokens(mine[n]) if n in mine else self.asked_about(q))
        return [w for w in out if w]

    def already_covered(self, name: str, sets: list[set[str]]) -> bool:
        """SIBLING_EXACT: an asked question names this property: the same words, or one name inside the other with
        mostly the same values on the listings read (two spellings of one property)."""
        own = tokens(name)
        if not own:
            return True
        stats = {n: set(v) for n, _, v in self.property_stats()}
        mine = stats.get(name, set())
        for words in sets:
            if own == words or own <= words and not stats:
                return True
            if own < words or words < own:
                other = next((v for n, v in stats.items() if tokens(n) == words), None)
                if other is None:
                    if own <= words:
                        return True         # the question named every word of it (and more)
                    continue
                if mine and other and len(mine & other) >= 0.6 * min(len(mine), len(other)):
                    return True
        return False

    def auto_question(self) -> bool:
        """Once a few listings were read, ask the planned properties by code, one step each: who it is for first when
        the brief flags it, then the ranked plan. Returns True when a question was sent."""
        if self.ask_first_question():
            return True
        if self.items:
            return self.basket_question()
        if self.auto_off or self.ask_room() <= 0 or self.session.finished or self.session.broken:
            return False
        if self.steps_left() <= self.reserve() + 6 + self.ask_room() or monotonic() - self.began > WRAP_AT - 60:
            return False
        tool = next((n for n, r in self.roles.items() if r == "message"), None)
        if tool is None or self.ask_reads() < self.reads_to_ask():
            return False
        if self.told and self.told[-1][0] >= self.session.steps:
            return False                    # a need was just told: let the search follow it before the next question
        self.plan_questions()
        if self.ask_plan is None if ASK_FALLBACK else not self.ask_plan:
            return False                    # (an empty plan leaves the coverage order)
        answered = self.answered_qa()
        asked = (set().union(*[tokens(q) for q, _ in answered]) if answered else set()) | self.asked_props
        sets = self.asked_sets() + [tokens(p) for _, p in self.code_asks]
        covered = (lambda name: self.already_covered(name, sets)) if SIBLING_EXACT else (lambda name: bool(tokens(name) & asked))
        if self.ask_order is None:
            # fixed once: the plan and the coverage order in turn, each named as the store's listings name it. A held-back
            # need sits on a property most of the item's listings carry, while ranking by the request's story words finds
            # it about half as often, so the brief's hint properties take no turns of their own
            labels = {n for n, c, v in self.property_stats() if self.label_like(c, v)}

            def usable(name: str) -> bool:
                return bool(tokens(name)) and not covered(name) and name not in labels

            planned = [p for p in self.ask_plan if usable(p) and not self.stated_property(p)]
            self.ask_order = in_turn(planned, [p for p in self.coverage_order() if usable(p)])
        order = [p for p in self.ask_order if not covered(p)]
        first = self.code_asks[0] if self.code_asks else None
        if SIBLING_EXACT and first and not self.reasked and self.ask_room() >= 2 and self.ask_order and first[1] == self.ask_order[0] \
                and self.qa_kind.get(first[0]) == "not_understood" and not any(step > 0 for step, _ in self.told):
            # the first key got "say again": asked once more (the shopper often tells it then), while two questions remain
            self.reasked = True
            self.reask_prop = first[1]
            order = [first[1]] + [p for p in order if p != first[1]]
        if not order:
            return False
        prop = order[0]
        values = [v for v in next((v for n, _, v in self.property_stats() if n == prop), []) if len(v) <= 20 and "," not in v][:3]
        # asked again after a reply wanting one question per message: the property alone, with no examples
        example = f" (for example {', '.join(values[:-1])} or {values[-1]})" if ASK_EXAMPLES and len(values) >= 2 \
            and prop not in self.plain_ask and not self.plain_all else ""
        text = f"Which {prop} would you like{example}?"
        if getattr(self, "reask_prop", None) == prop:
            text = f"Sorry, once more: which {prop} would you like?"   # other words than the first time
            self.reask_prop = None
        args = self.message_args(tool, text)
        if args is None:
            self.auto_off = True
            return False
        self.asked_props |= tokens(prop)
        call = {"id": f"ask-{self.session.steps}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        result = self.act(call)
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        self.short_of[call["id"]] = self.summary(result)
        if result.startswith('{"not_sent"'):
            self.ask_refusals += 1
            self.auto_off = self.ask_refusals >= (2 if ASK_FALLBACK else 1)
            return False
        self.qa_prop[len(self.qa) - 1] = tokens(prop)
        self.code_asks.append((len(self.qa) - 1, prop))
        if self.qa_kind.get(len(self.qa) - 1) == "one_at_a_time" and PLAIN_SPLIT and not example:
            self.split_plain.add(len(self.qa) - 1)   # it was asked alone and plainly: the next property, not the same words
        elif self.qa_kind.get(len(self.qa) - 1) == "one_at_a_time" and prop not in self.plain_ask:
            self.asked_props -= tokens(prop)    # the shopper wants it asked again on its own: once more, plainly
            self.plain_ask.add(prop)
        return True

    def ask_first_question(self) -> bool:
        """A question the request leaves to the shopper before choosing (the brief's ask_first, such as the most they
        will spend when the request names no limit), sent by code before any property question; at most two, and
        none about money when the request states a budget."""
        if self.auto_off or self.ask_room() <= 0 or self.session.finished or self.session.broken \
                or self.steps_left() <= self.reserve() + 6 or monotonic() - self.began > WRAP_AT - 60:
            return False
        tool = next((n for n, r in self.roles.items() if r == "message"), None)
        stated = self.stated_budget() is not None
        pending = [q.strip()[:200] for q in self.brief.get("ask_first") or [] if isinstance(q, str) and q.strip()][:2]
        pending = [q for q in pending if q not in self.first_asked
                   and not (stated and re.search(r"budget|spend|price|cost|afford|pay\b", q, re.I))]
        if tool is None or not pending:
            return False
        self.first_asked.add(pending[0])
        args = self.message_args(tool, pending[0])
        if args is None:
            return False
        call = {"id": f"ask-{self.session.steps}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        result = self.act(call)
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        self.short_of[call["id"]] = self.summary(result)
        return not result.startswith('{"not_sent"')

    def question_targets(self) -> list[str]:
        """Properties worth asking the shopper about, in order: the model-ranked plan when there is one, else those the
        request's hints point to, then those most listings describe with a few repeated values. Properties the request
        fixes, or already asked about, are left out."""
        if self.items:
            return []                       # the listings' properties mix several items: no common order to suggest
        answered = self.answered_qa()
        asked = (set().union(*[tokens(q) for q, _ in answered]) if answered else set()) | self.asked_props
        if self.ask_order is not None:
            left = [p for p in self.ask_order if not tokens(p) & asked]
            if left:
                return left[:4]             # what code asks next, in its order
        common = [p for p in self.coverage_order() if not tokens(p) & asked]
        if self.ask_plan:
            planned = [p for p in self.ask_plan if tokens(p) and not tokens(p) & asked and not self.stated_property(p)]
            if planned:
                return in_turn(planned, common)[:4]
        stats = self.property_stats()
        hints = [h.strip().lower()[:40] for h in self.brief.get("hint_properties") or []
                 if isinstance(h, str) and tokens(h) and not tokens(h) & asked and not self.stated_property(h)]

        def described(hint: str) -> bool:
            words = tokens(hint)
            return any(words <= tokens(n) or tokens(n) <= words for n, _, _ in stats)
        # the properties most listings carry first, then hinted ones the listings' facts describe; a hint they never
        # mention waits until the end ("style" and "styles" are one property)
        return in_turn(common + [h for h in hints if described(h) or not stats] + hints, [])[:4]

    def marks(self, ident: str, reqs: list[dict] | None = None) -> tuple[list[str], bool]:
        """How the listing's facts, category and title read against each attribute requirement, and whether nothing
        reads against it. A requirement is read clause by clause (commas separate them): a clause with a negation, or
        a requirement whose name says "exclude", must not hold; the others must. A fact meets a clause when its value
        (or one of the values it lists) holds the clause's words, or, for a fact naming the property, when its short
        value sits inside the clause ("tube" in "comes in a tube"); a number is met by the same number. Other key
        spellings saying something else do not cancel that. A bound ("at most 5 kg") is settled when the fact naming the
        property has one number in the same unit; other bounds and long clauses are left to the model. An order of
        several items is checked line by line by the basket check instead, unless reqs gives one item's requirements."""
        if self.items and reqs is None:
            return [], True
        sheet = self.facts.get(ident) or {}
        info = self.listings.get(ident) or {}
        out: list[str] = []
        clean = True
        mixed = False
        categories = {str(r.get("category")) for r in self.listings.values() if r.get("category")}
        store_names = set().union(*[tokens(k) for s in self.facts.values() for k in s] or [set()])
        for requirement in self.reqs if reqs is None else reqs:
            what = requirement["what"]
            if NOT_PROPERTY.search(what):
                continue                    # how many, how much and whether it is in stock are not listing properties
            if requirement.get("pending_ref"):
                # a value still to be looked up keeps every listing unconfirmed on it; the words describing the
                # referenced listing are never checked against this one
                out.append(f"{what[:30]}: unconfirmed (to look up first)")
                clean = False
                continue
            fitted = self.fit_verdict(ident, requirement) if FIT_CODES else None
            if fitted:
                out.append(f"{what[:30]}: {fitted}")
                clean = clean and fitted.startswith("yes")
                continue
            value = re.sub(r"\([^)]*\)", " ", requirement["value"])     # asides such as "(I'll check it myself)"
            what_words = tokens(NEGATION.sub(" ", what)) - FILLER
            excluding = bool(NEGATION.search(what))
            # facts naming the whole property ("teapot style") decide it: a fact naming only part of it ("style",
            # "teapot type") is another property, and its value says nothing here
            whole = {k for k in sheet if len(what_words) >= 2 and what_words <= tokens(k)}
            own = {k: v for k, v in sheet.items() if k in whole or not tokens(k) & what_words} if whole else sheet
            named = [f"{k}={v}" for k, v in own.items() if tokens(k) & what_words]
            verdicts: list[str] = []
            for clause in [c for c in re.split(r"[;,]", value) if c.strip()] or [value]:
                negative = excluding or bool(NEGATION.search(clause))
                body = NEGATION.sub(" ", clause)
                short = body.strip().lower() if SHORT_VALUES and re.fullmatch(r"\s*[A-Za-z]{1,2}\s*", body) else ""
                if short:
                    # a size letter and the like: the whole value of a fact naming the property, or one of the values it lists

                    def holds(text: str, want: str = short) -> bool:
                        return want in re.split(r"[\s,/;|()]+", str(text).lower())
                    fact = next((f"{k}={v}" for k, v in own.items() if tokens(k) & what_words and holds(v)), None)
                    clause_named = list(named)
                    where = None
                elif re.search(r"\d", body):
                    if COMPARE.search(body) or (RANGE_CHECK and range_of(body)):
                        settled = None if negative else bound_check(body, own, what_words)
                        if settled:
                            verdicts.append(settled)
                        continue            # a bound code cannot settle: the model compares it
                    numbers = re.findall(r"\d+(?:\.\d+)?", body)
                    words = tokens(body) - FILLER - what_words
                    want = amount_of(body)

                    def strict(text: str, numbers: list[str] = numbers, words: set[str] = words) -> bool:
                        return all(n in re.findall(r"\d+(?:\.\d+)?", str(text)) for n in numbers) and words <= tokens(text)

                    def holds(text: str, want: tuple | None = want) -> bool:
                        # also the same number with its unit written another way or left out (15.6", 15.6 in for 15.6
                        # inches); the same number in another unit (15 cm) is not the same size
                        have = amount_of(str(text))
                        return strict(text) or bool(want and have and have[0] == want[0]
                                                    and (not have[1] or not want[1] or have[1] == want[1]))
                    fact = next((f"{k}={v}" for k, v in own.items() if tokens(k) & what_words and holds(v)), None)
                    if fact is None and not named:
                        fact = next((f"{k}={v}" for k, v in own.items() if strict(v)), None)
                    where = None            # numbers in titles are often model numbers or sizes: only facts count
                    clause_named = [n for n in named if re.search(r"\d", n)]
                    if fact is None and clause_named and not negative:
                        # another number in another unit, or with no unit on one side (10 inches, 25.4 cm), is no
                        # contradiction code can see: the model converts and compares
                        want_unit = (want or (0.0, ""))[1]
                        have_unit = (amount_of(clause_named[0].split("=", 1)[1]) or (0.0, ""))[1]
                        if want and want_unit != have_unit:
                            verdicts.append(f"unconfirmed (in another unit or none: {clause_named[0][:60]}; convert to compare)")
                            continue
                else:
                    if len(tokens(body) - FILLER) > 5:
                        continue            # a long clause is a procedure, not a property value
                    parts = re.split(r"\bor\b", body)
                    # "indoor/outdoor" is either value: each side of a slash is an option too, next to the whole phrase
                    parts += [re.sub(r"(\w+)/(\w+)", side, p) for p in parts if "/" in p for side in (r"\1", r"\2")]
                    options = [tokens(o) - FILLER for o in parts]
                    # the property's own name inside its value ("dotted pattern" for pattern) is not part of the value;
                    # a clause that is nothing but the property's name ("must suit" for suit) makes no claim
                    options = [o - what_words for o in options if o - what_words]
                    if not options:
                        continue            # "no lid at all" names no value to exclude

                    def holds(text: str, options: list[set[str]] = options) -> bool:
                        words = tokens(text)
                        return any(o <= words for o in options)

                    def inside(text: str, options: list[set[str]] = options) -> bool:
                        words = tokens(text) - FILLER
                        return bool(words) and len(words) <= 3 and any(words <= o for o in options)
                    union = set().union(*options)
                    # "A or B" names a store category "A & B" when one exists: then only that category counts
                    covering = len(options) > 1 and any(union <= tokens(c) for c in categories)
                    fact = next((f"{k}={v}" for k, v in own.items() if holds(v)), None) \
                        or next((f"{k}={v}" for k, v in own.items() if tokens(k) & what_words and inside(v)), None)
                    category = info.get("category") or ""
                    clause_named = list(named)
                    where = None
                    cat_words = tokens(category) - FILLER
                    if ((union <= tokens(category)) if covering else holds(category)) \
                            or (cat_words and any(cat_words <= o for o in options) and len(cat_words) >= 2):
                        where = "category"      # the category's own name inside the phrase ("cups & saucers")
                    elif category and covering and holds(category):
                        clause_named = clause_named or [f"category={category}"]   # names only one of the kinds
                    elif holds(info.get("title") or ""):
                        where = "title"
                if negative:
                    verdicts.append(f"breaks ({fact[:60]})" if fact else "ok")
                elif fact:
                    differs = [n for n in clause_named if not holds(n.split("=", 1)[1])]
                    verdicts.append(f"yes ({fact[:60]})" + (f", but {differs[0][:50]}" if differs else ""))
                    mixed = mixed or bool(differs)
                elif clause_named:
                    verdicts.append(f"no ({clause_named[0][:60]})")     # a fact naming the property decides over the title
                elif where == "title" and requirement.get("told") and what_words & store_names:
                    # the store files this property as a fact: a need the shopper told must show there, not only in a title
                    verdicts.append("unconfirmed (title only: a need the shopper told must show in the facts)")
                elif where:
                    verdicts.append(f"yes by {where}")
                else:
                    verdicts.append("unconfirmed")
            if not verdicts:
                continue
            # wanted clauses are alternatives ("simple, classic"); every exclusion must hold ("not a refill")
            broken = next((v for v in verdicts if v.startswith("breaks")), None)
            wanted = [v for v in verdicts if not v.startswith(("breaks", "ok"))]
            best = next((v for v in wanted if v.startswith("yes")), None) or next((v for v in wanted if v.startswith("no")), None) \
                or (wanted[0] if wanted else None)
            verdict = broken or best or "ok"
            clean = clean and not verdict.startswith(("no", "breaks", "unconfirmed"))
            out.append(f"{what[:30]}: {verdict}")
        self.mixed[ident] = mixed
        return out, clean

    def facts_view(self) -> str:
        """The leading candidates read against the requirements, with the facts that bear on them."""
        wanted = self.requirement_words() | set().union(*[tokens(r["what"]) | tokens(r["value"]) for r in self.reqs if not r.get("pending_ref")] or [set()])
        ids = list(self.facts)
        if self.cheapest_wanted():
            ids.sort(key=lambda i: (self.listings.get(i, {}).get("price") is None, self.listings.get(i, {}).get("price") or 0))
        else:
            ids.sort(key=lambda i: -self.listings.get(i, {}).get("step", 0))
        rows = []
        for ident in ids[:10]:
            sheet = self.facts[ident]
            info = self.listings.get(ident, {})
            checks, _ = self.marks(ident)
            near = [f"{k}={v}" for k, v in sheet.items() if wanted and (tokens(k) | tokens(v)) & wanted]
            shown = near[:5] or [f"{k}={v}" for k, v in list(sheet.items())[:3]]
            rows.append(f"{ident} | price {info.get('price')} | stock {info.get('stock')} | "
                        + ("; ".join(checks) + " | " if checks else "") + "; ".join(x[:70] for x in shown) + self.compat_text(ident))
        return " || ".join(rows)

    def stated_budget(self) -> float | None:
        """The money budget when the request or a later notice writes it (a figure converted from cents to whole units
        is put back as the request writes it)."""
        budget = self.brief.get("money_budget")
        if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget <= 0:
            return None
        text = self.session.request + " " + self.budget_text
        if number_in(text, int(budget)) if float(budget).is_integer() else f"{budget:g}" in text:
            return budget
        hundredfold = round(budget * 100)
        return hundredfold if abs(budget * 100 - hundredfold) < 1e-6 and number_in(text, hundredfold) else None

    def fit_list(self) -> list[tuple[float, str]]:
        """(price, id) of read listings with every requirement confirmed, last seen in stock and within the budget,
        cheapest first."""
        budget = self.stated_budget()
        out = []
        for ident in self.facts:
            if ident in self.ref_ids and not self.ref_allowed(ident):
                continue                    # a listing a pointer led to defines a value; it is not the item
            info = self.listings.get(ident) or {}
            price = info.get("price") if isinstance(info.get("price"), (int, float)) else float("inf")
            if sold_out(info.get("stock")) or (budget is not None and price != float("inf") and price > budget + 1e-9):
                continue
            checks, clean = self.marks(ident)
            if clean and checks:
                out.append((price, ident))
        measure = self.priority_measure() if MEASURE_SORT else None
        if measure:
            # the most (or least) of the measure first, among numbers in the unit most fitting listings give it; the
            # price breaks ties, and a listing with no number in that unit goes after
            values = {i: self.measure_value(i, measure[0]) for _, i in out}
            units = [v[1] for v in values.values() if v]
            unit = max(set(units), key=units.count) if units else None

            def by_measure(pair: tuple[float, str]) -> tuple:
                value = values.get(pair[1])
                if not value or value[1] != unit:
                    return (1, 0.0, pair[0])
                return (0, -value[0] if measure[1] == "higher" else value[0], pair[0])
            return sorted(out, key=by_measure)
        if not self.cheapest_wanted():
            # without a price rule, listings whose facts all agree come first: the store may grade by one key
            return sorted(out, key=lambda pair: (self.mixed.get(pair[1], False), pair[0]))
        return sorted(out)

    def fitting(self) -> list[str]:
        """Read listings with every requirement confirmed, in stock and within the budget, cheapest first."""
        return [f"{i} ({p:g})" if p != float("inf") else i for p, i in self.fit_list()][:8]

    def wanted_count(self) -> int:
        """Units of the one item to order: the brief's count when the request (or a later reply) writes it, else 1."""
        wanted = self.brief.get("quantity")
        return wanted if isinstance(wanted, int) and not isinstance(wanted, bool) and wanted >= 1 \
            and (wanted == 1 or number_in(self.session.request + " " + self.quantity_text, wanted)) else 1

    def verdict_of(self, ident: str) -> tuple[bool, int, int]:
        """How a read listing of one item stands as code reads it: (nothing contradicts it and every need the shopper
        told is confirmed, requirements confirmed, requirements left unconfirmed)."""
        checks, _ = self.marks(ident)
        told = {label.split(":", 1)[0].split(" > ")[-1].strip().lower()[:30] for _, label in self.told}
        clean, yes, unconfirmed = True, 0, 0
        for check in checks:
            what, _, verdict = check.partition(": ")
            if verdict.startswith(("no", "breaks")):
                clean = False
            elif verdict.startswith(("yes", "ok")):
                yes += 1
            else:
                unconfirmed += 1
                if what.strip().lower() in told and "another unit" not in verdict and "to look up" not in verdict:
                    # a told need must be confirmed, never merely left open; a unit to convert is not open, and a value still
                    # to be looked up leaves the listing a candidate for the fallback, never a confirmed pick
                    clean = False
        return clean, yes, unconfirmed

    def ranked_candidates(self) -> list[str]:
        """Read listings of one item that could still be ordered, best first: those with every requirement confirmed
        (in fit_list's order), then those nothing contradicts, more confirmed first and, when the cheapest is asked,
        cheaper first. In stock and within the budget. Empty for an order of several items."""
        if self.items:
            return []
        confirmed = [i for _, i in self.fit_list()]
        budget = self.stated_budget()
        count = self.wanted_count()
        rest = []
        for ident in self.facts:
            if ident in self.ref_ids and not self.ref_allowed(ident):
                continue
            info = self.listings.get(ident) or {}
            price = info.get("price") if isinstance(info.get("price"), (int, float)) and not isinstance(info.get("price"), bool) else None
            if ident in confirmed or sold_out(info.get("stock")) or (budget is not None and price is not None and price * count > budget + 1e-9):
                continue
            clean, yes, unconfirmed = self.verdict_of(ident)
            if clean and yes:
                rest.append((-yes, unconfirmed, price if price is not None and self.cheapest_wanted() else 0, ident))
        return confirmed + [r[-1] for r in sorted(rest)]

    def confirmed_in_cart(self) -> str:
        """The listing the cart held at its last view, when nothing changed since and every requirement is confirmed."""
        if self.cart_obs is None or self.cart_seen is None or self.cart_changed > self.cart_seen[0]:
            return ""
        lines, _ = self.cart_lines()
        if len({line["id"] for line in lines}) != 1:
            return ""
        item = lines[0]["id"]
        if item not in self.facts or sold_out((self.listings.get(item) or {}).get("stock")):
            return ""
        checks, clean = self.marks(item)
        return item if checks and clean and item in {i for _, i in self.fit_list()} else ""

    def choice_problems(self, role: str) -> list[str]:
        """Once per finish attempt: an order of a listing never read in detail; an order of a listing with a requirement
        not confirmed (while confirmed listings exist, or while steps allow reading more); an order above a cheaper
        confirmed listing when the cheapest is asked for; a decline while a confirmed listing is in stock."""
        if self.steps_left() <= self.reserve() + 1 or monotonic() - self.began > WRAP_AT - 20:
            return []
        if role == "order" and self.cart_obs is not None and self.steps_left() > self.reserve() + 2 \
                and any(r == "details" for r in self.roles.values()) and "unread_cart" not in self.gaps_raised:
            lines, _ = self.cart_lines()
            unread = [line["id"] for line in lines if line["id"] not in self.facts
                      or (COVER_BY_CONTENT and self.cover and "props" not in self.cover.get(line["id"], set()) and self.props_view())]
            if unread:
                self.gaps_raised.add("unread_cart")
                self.session.log.append({"role": "assistant", "content": "note: order sent back, cart item unread " + short(unread, 200)})
                return [f"the facts of {', '.join(unread)} in the cart were never read: read them and confirm every requirement "
                        "before ordering"]
        if role == "order" and self.cart_obs is not None and self.category_stops < 2:
            lines, _ = self.cart_lines()
            for line in lines:
                checks, _ = self.marks(line["id"]) if line["id"] in self.facts or line["id"] in self.listings else ([], True)
                wrong = [c for c in checks if "no (category=" in c]
                if wrong:
                    self.category_stops += 1
                    self.session.log.append({"role": "assistant", "content": "note: order sent back, category " + short(wrong, 200)})
                    return [f"{line['id']} is filed under another category ({wrong[0]}): the request names kinds that one store "
                            "category names together, and that category is the one meant. Choose a listing filed under it."]
        if "choice" in self.gaps_raised:
            return []
        fits = self.fit_list()
        problems: list[str] = []
        if not fits:
            if role == "order" and self.cart_obs is not None and self.steps_left() > self.pacing() + 2:
                lines, _ = self.cart_lines()
                item = lines[0]["id"] if lines else ""
                checks, clean = self.marks(item) if item in self.facts else ([], True)
                if checks and not clean:
                    open_points = [c for c in checks if not c.split(": ", 1)[-1].startswith(("yes", "ok"))]
                    problems.append(f"{item} in the cart has requirements not confirmed ({'; '.join(open_points)}) and no read "
                                    "listing has them all yet: read the facts of more candidates (search again if needed) for one "
                                    "that confirms every requirement; if none turns up in the steps left, order the best you have")
                    self.gaps_raised.add("choice")
                    self.session.log.append({"role": "assistant", "content": "note: order sent back, cart item unconfirmed " + short(open_points, 300)})
            return problems
        shown = ", ".join(f"{i} ({p:g})" if p != float("inf") else i for p, i in fits[:5])
        if role == "decline":
            problems.append(f"these read listings have every requirement confirmed and are in stock: {shown}; order the right one "
                            "of them instead of declining, unless a told need or a rule of the request rules each out")
        elif role == "order" and self.cart_obs is not None:
            lines, _ = self.cart_lines()
            if lines:
                item = lines[0]["id"]
                checks, clean = self.marks(item)
                price = (self.listings.get(item) or {}).get("price")
                if checks and not clean and item in self.facts:
                    open_points = [c for c in checks if not c.split(": ", 1)[-1].startswith(("yes", "ok"))]
                    problems.append(f"{item} in the cart has requirements not confirmed ({'; '.join(open_points)}), while these "
                                    f"are confirmed: {shown}")
                elif MEASURE_SORT and self.priority_measure() and fits[0][1] != item and item in {i for _, i in fits}:
                    measure, way = self.priority_measure()
                    problems.append(f"a listing with every requirement confirmed and the {'most' if way == 'higher' else 'least'} "
                                    f"{measure} is in stock: {shown.split(', ')[0]}")
                elif self.cheapest_wanted() and isinstance(price, (int, float)) and fits[0][0] < price - 1e-9 and fits[0][1] != item:
                    problems.append(f"a cheaper listing with every requirement confirmed is in stock: {shown.split(', ')[0]}")
        if problems:
            self.gaps_raised.add("choice")
            self.session.log.append({"role": "assistant", "content": f"note: {role} sent back on the choice " + short(problems, 400)})
        return problems

    def state_note(self) -> str:
        session = self.session
        elapsed = monotonic() - self.began
        left = self.steps_left()
        lines = [f"STATE: step {session.steps} used of {self.step_cap} allowed; {left} left (keep {self.reserve()} for finishing). "
                 f"Time used: {elapsed:.0f} s of about {WRAP_AT:.0f} s."]
        if any(r == "message" for r in self.roles.values()):
            budget = self.question_budget if self.question_budget is not None else "?"
            lines.append(f"Questions to the shopper: {self.questions} used of {budget}"
                         + (f", {self.reserve_left()} of them kept for the message the request asks for before you finish." if self.reserve_left() else "."))
        else:
            lines.append("There is no tool to message the shopper.")
        if self.procedures:
            moments = {"before_order": "before ordering", "before_decline": "before declining", "before_choosing": "before choosing",
                       "any_time": "throughout"}
            sends = any(r == "message" for r in self.roles.values())
            lines.append("Procedures the request states (each counts like a requirement): " + " | ".join(
                f"{p['text']} ({moments[p['when']]}" + ("; your assistant software sends this message right before that finish)" if sends
                                                       and p["tell"] and p["when"] in ("before_order", "before_decline") else ")")
                for p in self.procedures[:6]))
        if self.notices:
            lines.append("Notices so far (follow the latest): " + " | ".join(self.notices[-6:]))
        pages = self.page_views()
        if pages and not self.list_view():
            lines.append(f"Views of a listing: {self.props_view()} shows its properties: judge and compare listings with it. "
                         f"{', '.join(pages)} shows its own page (price, stock, the seller's note): your assistant software opens "
                         "it for each cart line before the order and for a listing a pointer names.")
        budget = self.stated_budget()
        measure = self.priority_measure() if MEASURE_SORT else None
        if budget is not None or self.cheapest_wanted() or measure:
            pick = ["the cheapest listing that meets every requirement"] if self.cheapest_wanted() else \
                [f"the listing with the {'most' if measure[1] == 'higher' else 'least'} {measure[0]} among those that meet every "
                 "requirement"] if measure else []
            lines.append("The order as it stands: " + "; ".join(
                ([f"money budget {budget:g} for the whole order" + (" (changed since the request)" if self.budget_text else "")]
                 if budget is not None else []) + pick) + ".")
        if self.listings:
            if self.cheapest_wanted():
                seen = sorted(self.listings.values(), key=lambda r: (not isinstance(r["price"], (int, float)), r["price"] or 0))[:16]
                label = "Listings seen, cheapest first"
            else:
                seen = sorted(self.listings.values(), key=lambda r: -r["step"])[:14]
                label = "Listings seen, latest first"
            lines.append(f"{label} (id | title | price | stock | category | step | facts read): " + "; ".join(
                f"{r['id']} | {r['title'][:60]} | {r['price']} | {r['stock']} | {str(r.get('category') or '')[:30]} | {r['step']} | "
                f"{'yes' if r['id'] in self.facts else 'no'}" for r in seen))
        if self.items:
            lines.append("The order needs these lines, one listing each in this quantity: " + "; ".join(
                f"{e['item']} x{e['quantity']}" for e in self.items) + ". The money budget covers the whole order.")
            if LINE_SHARES and len(self.items) >= 2:
                lines.append(self.line_status())
        if self.facts:
            if self.reqs:
                lines.append("Requirements as they stand: " + "; ".join(
                    f"{r['item'] + ' > ' if r.get('item') else ''}{r['what']}: {r['value']}" for r in self.reqs))
            if self.items:
                lines.append("Listings read (id | price | stock | facts): " + self.facts_view() + ". Judge each listing against "
                             "the requirements of the item it would fill: its category decides the kind of item, its facts the rest.")
                described = self.described_properties()
                if described:
                    lines.append("Properties each item's read listings describe (sample values). A held-back need is told only "
                                 "when a question names its property: ask with these exact names, e.g. \"Which <property> would "
                                 "you like for the <item>?\": " + described)
            else:
                lines.append("Listings read, against the requirements (yes = a fact, any value it lists, the category or the title "
                             "gives the value; no = the named fact differs; unconfirmed = nothing gives it; id | price | stock | checks | "
                             "facts): " + self.facts_view())
                fits = self.fitting()
                order = (f"the {'most' if measure[1] == 'higher' else 'least'} {measure[0]} first" if measure else
                         "cheapest first" if self.cheapest_wanted() else "those whose facts all agree first")
                lines.append("Read listings with every requirement confirmed, in stock, " + order + " (check the price, numbers and anything the checks leave out "
                             "yourself): " + (", ".join(fits) if fits else "none yet"))
            lines.append("Properties the listings' facts describe (listings: sample values): " + self.attribute_tally())
        fresh = [need for step, need in self.told if session.steps - step <= 4]
        if fresh:
            lines.append("Just told (now requirements): " + "; ".join(fresh) + ". Your earlier candidates may not meet this: check "
                         "them against it, and search again with its words (for example added to the item words) to find listings "
                         "that do.")
        for ref in [r for r in self.refs if r["value"] is not None and r.get("source") and (r["hops"] or r["source"]["paired"])][:2]:
            # the step model sees the first listing's own value and may take it for the requirement: say how code got
            # the value, so a listing that shows the first listing's value is not ordered
            src = ref["source"]
            label = f"{ref['item'] + ' > ' if ref['item'] else ''}{ref['what']}"
            title = str((self.listings.get(src["id"]) or {}).get("title") or "")[:50]
            lines.append(f"Traced by code: {label} = {ref['req']['value']}. "
                         + (f"The notes of {ref['hops'][0]} (the listing the request or the shopper names) send its {ref['what']} "
                            f"on to another listing, so its own {ref['what']} is not the requirement; code followed "
                            f"{' -> '.join(ref['hops'] + [src['id']])} and read {src['id']} ({title}), which shows {src['own']}"
                            if ref["hops"] else f"{src['id']} ({title}) shows {src['own']}")
                         + (f", and its notes pair that value with {ref['value']}, so the item takes {ref['value']}" if src["paired"] else "")
                         + ". This is the requirement now: a listing whose facts contradict it fails the order.")
        for ref in [r for r in self.refs if r["value"] is None][:2]:
            spec = ref["ref"]
            where = (f"{spec['item'] or 'another item'}" + (f" sold by {spec['seller']}" if spec["seller"] else "")
                     + (f" with {spec['properties']}" if spec["properties"] else ""))
            lines.append(f"Needs a lookup: {ref['item'] + ' > ' if ref['item'] else ''}{ref['what']} must be "
                         + ("anything but the value on" if ref["exclude"] else "the value on") + f" the listing for {where}"
                         + (f" (pointers followed so far: {' -> '.join(ref['hops'])})" if ref["hops"] else "")
                         + f". Search for \"{' '.join(x for x in (spec['seller'], spec['item']) if x)}\" and open the matches until one "
                         "shows every value given (look-alikes differ in one value). When its notes send this property on to "
                         "another listing, code follows the pointer and this line names the next listing to find; otherwise "
                         f"its {ref['what']} is the value " + ("to avoid. " if ref["exclude"] else "to require. ")
                         + ("The listing that fits (" + ", ".join(ref["seen"])[:120] + ") shows no " + ref["what"] + " among its "
                            "facts: take the value its notes state, or read the listing they refer to and use that one's. "
                            if ref["seen"] else "")
                         + "The words describing these listings are not requirements of the item. When the last listing's "
                           "notes pair its value with another, code applies that rule and states the value to look for.")
        barred = sorted(i for i in self.ref_ids if not (REF_CANDIDATE and self.ref_allowed(i)))
        if barred:
            lines.append("Reference listings (they define a value for the item; never put them in the cart): "
                         + ", ".join(barred[:8]))
        if REF_CANDIDATE and len(barred) < len(self.ref_ids):
            lines.append("Listings a pointer led to that hold the traced value are ordinary candidates: buy one if it is the "
                         "best that meets everything: " + ", ".join(sorted(set(self.ref_ids) - set(barred))[:8]))
        drops = [what for step, what in self.dropped if session.steps - step <= 6]
        if drops:
            lines.append("No longer required (the shopper dropped it): " + "; ".join(drops) + ". The shopper's messages can come "
                         "inside any tool result, a refused finish included, and a later change from the shopper overrides the "
                         "request, so treat it as gone. Keep your current pick if it "
                         "still meets everything else. If the request asks for the cheapest, check whether a cheaper listing now "
                         "qualifies (search again without that value) and switch only to the cheapest one that meets everything.")
        # only from a cart view newer than the last cart change: a line taken out since would be named again and again
        gone = [l["id"] for l in self.cart_lines()[0] if sold_out((self.listings.get(l["id"]) or {}).get("stock"))] \
            if self.cart_obs is not None and self.cart_seen is not None and self.cart_seen[0] >= self.cart_changed else []
        if gone:
            nxt = [i for i in self.ranked_candidates() if i not in gone][:3]
            lines.append("No longer available but still in the cart: " + ", ".join(gone) + f". Take it out ({self.removal_hint()}), "
                         + (f"and put in the next one in line from the listings read: {', '.join(nxt)} (search again only "
                            "if none of them fits)." if nxt else "search again and add the next cheapest listing that meets "
                            "everything for that item."))
        held = {l["id"] for l in self.cart_lines()[0]} if self.cart_obs is not None else set()
        moved = [i for i in self.price_refused if i in held and i not in gone]
        if moved:
            unread = [i for i in moved if i not in self.price_reread]
            lines.append("The last order was refused because the price of " + ", ".join(moved) + " changed. "
                         + (f"Read {', '.join(unread)} again for its current price and stock. " if unread else "")
                         + "A changed price alone is no reason to switch: if it is still in stock and price times quantity is "
                         "within the budget, order the same cart again. Switch only when the new price breaks the budget, or "
                         "the request asks for the cheapest and another listing with every requirement confirmed now costs less.")
        if self.searches:
            lines.append("Searches (step: arguments -> reported total, listings shown): " + "; ".join(
                f"{s}: {a} -> {t if t is not None else '?'}, {n}" for s, a, t, n in self.searches[-5:]))
            step, _, _, shown = self.searches[-1]
            # an empty result from a filtered search more often means a value in the wrong parameter than no such listing
            if not shown and session.steps - step <= 2 and self.sweeps and len([v for v in self.sweeps[-1][0].values() if v not in ("", None)]) >= 2:
                lines.append("The last search showed no listings. Before you conclude that nothing exists, check that each value sits "
                             "in the parameter made for it (a category or department name in the category filter, a maker in the maker "
                             "filter) and is written as the store writes it, or drop a filter.")
            # the same large total from searches whose words differ: that search mode ignores the words
            if len(self.sweeps) >= 2 and session.steps - step <= 2:
                (before, total_before, _), (last, total_last, _) = self.sweeps[-2], self.sweeps[-1]
                changed = [k for k in set(before) | set(last) if before.get(k) != last.get(k)
                           and not re.search(r"page|offset|cursor|start|mark|token", str(k), re.I)]
                if changed and all(isinstance(before.get(k, ""), str) and isinstance(last.get(k, ""), str) for k in changed) \
                        and isinstance(total_last, int) and total_last == total_before and total_last >= 20:
                    lines.append(f"The last two searches reported the same total ({total_last}) although their words differ: in that "
                                 "search mode the words had no effect. Use the mode, tool or filter where they apply rather than "
                                 "repeating the search with other words.")
        unseen = sorted((e for e in self.more_pages.values() if e["mark"] is not None and isinstance(e["total"], int)
                         and e["total"] > len(e["seen"])), key=lambda e: -e["step"])[:3]
        # later pages matter when the cheapest is asked for or a basket's lines are spread over searches; for one item
        # otherwise, reading the listings already shown pays better than paging on
        if unseen and (self.cheapest_asked() or self.items):
            lines.append("Searches with matches not shown yet (results come in a fixed order, so the right listing can be on a "
                         "later page): " + "; ".join(f"{e['tool']} {short(e['args'], 90)}: {len(e['seen'])} of {e['total']} shown, "
                                                     f"next page marker {json.dumps(e['mark'])}" for e in unseen)
                         + ". To see more, repeat that search with the same arguments and pass its marker, exactly as given, in "
                         "the paging argument, or narrow the search until the total fits.")
        asking = any(r == "message" for r in self.roles.values())
        if asking and self.plain_all and self.ask_room() > 0:
            lines.append("The shopper takes one plain thing per message: name one property and nothing else (no examples, "
                         "no lists, no second question).")
        need = self.reads_to_ask() if asking and not self.items else 0
        if asking and self.ask_room() > 0 and left > self.reserve() + SPARE_STEPS // 2 and self.ask_reads() < need:
            more = need - self.ask_reads()
            lines.append(f"Questions left: {self.ask_room()}. First read the details of {more} more listing{'s' if more > 1 else ''} "
                         + (f"filed under {' or '.join(sorted(self.item_categories())[:2])} " if self.item_categories() and not self.items else "")
                         + "that may fit: your assistant software then asks about the properties most of the read listings "
                         "describe, where a held-back need sits.")
        elif asking and self.ask_room() > 0 and left > self.reserve() + SPARE_STEPS // 2:
            self.plan_questions()
            targets = self.question_targets()
            lines.append(f"Questions left: {self.ask_room()}. They cost nothing in the score: before choosing, "
                         "ask about one property per message" + (f", in this order unless the request points elsewhere: {'; '.join(targets)}."
                                                                if targets else ", naming the item it is about and spreading the "
                                                                "questions over the items." if self.items else
                                                                ", picking properties the request does not fix."))
        if self.cart_seen:
            lines.append(f"Cart as last viewed at step {self.cart_seen[0]}: {self.cart_seen[1]}")
        if self.finish_tries:
            lines.append(f"Finish attempts so far: {self.finish_tries}, refused: {self.refusals}. A refusal means something changed: "
                         "follow the notices, fix the cart and finish again.")
        if left <= self.reserve() + 1 or elapsed > WRAP_AT:
            lines.append("FINISH NOW: make sure the cart is right and place the order, or decline if nothing fits.")
        elif not self.finish_tries:
            lines.append(f"Plan to make your first finish attempt by step {max(1, self.step_cap - self.pacing())}, "
                         "keeping steps to recover if it is refused.")
        return "\n".join(lines)

    def view(self) -> list[dict]:
        """The conversation as the model sees it: older tool results shortened, then the current state."""
        tool_ids = [m["tool_call_id"] for m in self.messages if m.get("role") == "tool"]
        keep = KEEP_FULL if self.mind.spent < SPEND_SOFT else KEEP_FULL // 2     # past the soft limit: a shorter context
        if len(tool_ids) - self.shortened > keep + CACHE_SLACK:                  # shorten in a batch (cache, see CACHE_SLACK)
            self.shortened = len(tool_ids) - keep
        old = set(tool_ids[:self.shortened])
        shown = []
        for message in self.messages:
            if message.get("role") == "tool" and message["tool_call_id"] in old:
                message = dict(message, content=self.short_of.get(message["tool_call_id"], message["content"][:SHORT_CHARS]))
            shown.append(message)
        return shown + [{"role": "user", "content": self.state_note()}]

    # -- one tool call ----------------------------------------------------------------------------------------------------
    def act(self, call: dict, paging: bool = False) -> str:
        """Run one tool call the model asked for and return what the model is told."""
        session = self.session
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        if name not in session.specs:
            return json.dumps({"not_sent": f"unknown tool '{name}'", "tools": list(session.specs)})
        args = parse_json(function.get("arguments") or "{}")
        if args is None:
            return json.dumps({"not_sent": "the arguments are not valid JSON"})
        if not isinstance(args, dict):
            args = {}
        spec = session.specs[name]
        cut_note = self.cut_to_cap(name, args)
        args, problems = fit_args(spec.get("parameters") or {"type": "object"}, args)
        if not problems and len(json.dumps(args, separators=(",", ":")).encode("utf-8")) > ARG_BYTES:
            problems = ["the arguments are too long"]
        if problems:
            return json.dumps({"not_sent": "the arguments do not fit the tool", "problems": problems[:4]})
        role = self.call_role(name, args)
        if role == "search" and FILTER_FIX:
            refused = self.filter_fix(name, args)
            if refused:
                return json.dumps({"not_sent": refused})
        left = self.steps_left()
        if left <= 0:
            return json.dumps({"not_sent": "no steps are left"})
        signature = json.dumps(args, sort_keys=True, default=str)
        if role not in ("cart_view", "cart_change", "order", "decline", "message"):
            for tool, used, notices in reversed(self.recent[-2:]):
                if self.roles.get(tool) in ("cart_change", "order", "decline"):
                    break                   # the cart changed since: what was read before it may be stale (stock)
                if tool == name and used == signature and notices == len(self.notices):
                    return json.dumps({"not_sent": "this exact call was just made and nothing changed since: its result is "
                                                   "above. Use it, or change the call (other words, another page, another listing)."})
        if role == "message" and self.question_budget is not None and self.questions >= self.question_budget:
            return json.dumps({"not_sent": "the question budget is used up; continue without asking"})
        if role == "message" and self.question_budget is not None and self.reserve_left() and self.ask_room() <= 0:
            return json.dumps({"not_sent": "the questions are used up: the last message is kept for what the request asks you to "
                                           "tell the shopper before you finish, which your assistant software sends; continue "
                                           "without asking"})
        if role == "message":
            text = " ".join(str(v) for v in args.values() if isinstance(v, str))
            asking = self.asked_about(text)
            item = self.item_named(text) if self.items else ""
            # the same property of another item of the order is a new question; a narrower one ("light colour" after
            # "colour") is too, a repeat in fewer words is not
            if asking and any(asking <= self.asked_about(q) and (not self.items or self.item_named(q) in ("", item))
                              for q, _ in self.answered_qa()):
                return json.dumps({"not_sent": "that property was already asked about, and a reply that gave no value means "
                                               "no need there: ask about another property, or go on without asking"})
        if role == "message" and session.steps < 12 and self.read_first_stops < 2 \
                and any(r == "details" for r in self.roles.values()):
            need = self.reads_to_ask()
            if self.ask_reads() < need:
                self.read_first_stops += 1
                return json.dumps({"not_sent": f"read the details of at least {need} fitting listings first: the questions then "
                                               "target properties this store's listings describe (search, then open the details "
                                               "of a few)."})
        if left <= 1 and role not in ("order", "decline"):
            return json.dumps({"not_sent": "one step is left: finish now (order, or decline if nothing fits)"})
        stale_cart = self.carted and (self.cart_seen is None or self.cart_changed > self.cart_seen[0])
        if left <= self.reserve() and role not in ("order", "decline", "cart_change") \
                and not (CART_RECORD and role == "cart_view" and stale_cart and left >= 2):
            # (a view of a cart changed since it was last seen still goes: the order is sent back without one)
            return json.dumps({"not_sent": f"only {left} steps are left, kept for finishing: fix the cart if needed and finish now"})
        if role in ("search", "details", "message", "other") and not self.finish_tries \
                and (left <= self.pacing() + (0 if self.carted else 1) or self.mind.spent >= SPEND_FINISH + self.mind.room) \
                and self.has_candidate():
            self.finish_asks += 1
            if FINISH_NOW and self.finish_asks >= 3:
                # the model answered the call to finish with other calls twice: each such turn costs money and no step
                self.session.log.append({"role": "assistant", "content": "note: finish asked three times: code finishes"})
                self.finish_anyway()
                return json.dumps({"note": "your assistant software finished the task itself (the finish was asked for "
                                           "three times)", "finished": bool(session.finished)})
            return json.dumps({"not_sent": f"{left} steps left and no finish attempt yet: a first finish is sometimes refused by a "
                                           "change at that moment, so make it now: put your best-supported listing in the cart if "
                                           "it is not there, view the cart if needed, and order, keeping the remaining steps to "
                                           "recover. Decline only if every listing read for an item is ruled out by a fact or the "
                                           "budget."})
        if LINE_SHARES and role == "search" and len(self.items) >= 2 and self.line_stops < 3:
            done = self.line_done_search(args)
            if done:
                self.line_stops += 1
                return json.dumps({"not_sent": done})
        if CART_RECORD and role == "cart_change" and not self.items and self.dup_stops < 2 and not self.takes_out(name, args) \
                and not any(self.list_shaped(n) for n, r in self.roles.items() if r == "cart_change"):
            held = dict(self.held_lines())
            again = [i for i in self.ids_in(args) if isinstance(held.get(i), int) and held[i] >= self.wanted_count()]
            if again:
                self.dup_stops += 1
                return json.dumps({"not_sent": f"{again[0]} is already in the cart ({held[again[0]]} of {self.wanted_count()} "
                                               "wanted): putting it in again would raise its count. View the cart, or order."})
        if role == "cart_change" and self.ref_stops < 2 and not self.takes_out(name, args):
            refs = [i for i in self.ids_in(args) if i in self.ref_ids and ((self.items and not REF_CANDIDATE) or not self.ref_allowed(i))]
            if refs:
                self.ref_stops += 1
                return json.dumps({"not_sent": f"{', '.join(refs)} is a listing the request or a seller's note named only as the "
                                               "source of a value: it defines that value and is not the item to buy. Put in a "
                                               "listing of the item itself that has the value (search for it with that value)."})
        if not self.items:
            stop = self.commit_check(name, role, args, left)
            if stop:
                unread_now = [i for i in self.ids_in(args) if i not in self.facts]
                if unread_now and stop.startswith("not yet: read") and BULK_READ and self.reader():
                    text = self.read_ids(unread_now + [i for i in self.read_queue() if i not in unread_now], "read before the cart",
                                         inject=False, count=False)
                    if text:
                        return json.dumps({"note": "read in detail before it goes in the cart (one step)", "result": text[:FULL_CHARS],
                                           "then": "put it in the cart if its facts meet the requirements; otherwise choose another"})
                return json.dumps({"not_sent": stop})
        if role in ("cart_change", "order", "decline") and self.ask_first(left):
            self.plan_questions()
            targets = self.question_targets()
            return json.dumps({"not_sent": f"ask the shopper first ({(self.question_budget or 0) - self.questions} questions left): questions "
                                           "cost nothing in the score and a held-back need is told only when asked about that property. "
                                           "Ask about one property per message" + (f", for example: {'; '.join(targets)}" if targets else
                                                                                   ", naming the item it is about" if self.items else "") + "."})
        if role == "decline":
            args = self.plain_remark(args)
            held = self.confirmed_in_cart()
            if held:
                return json.dumps({"not_sent": f"the cart holds {held}, which has every requirement confirmed and was last seen in "
                                               "stock: order it rather than decline (remove it first only if a told need or a rule "
                                               "of the request rules it out)."})
            kept = self.cart_still_fits()
            if kept:
                return json.dumps({"not_sent": f"the last finish was refused because something changed, and the cart ({kept}) still "
                                               "meets the requirements as they stand: order it again rather than decline (take a "
                                               "line out first only if a notice or a told need rules it out)."})
        if role == "decline":
            unworded = self.search_first() or self.rows_first()
            if unworded:
                return json.dumps({"not_sent": unworded})
        swap_note = ""
        if role in ("order", "decline"):
            problems = self.finish_problems(role)
            if problems:
                return json.dumps({"not_sent": "the finish would fail; fix this first", "problems": problems})
            if role == "order":
                pages = self.read_pages()
                if pages:
                    return pages
            if role == "decline":
                held = self.decline_reads()
                if held:
                    return held
            unread = self.unread_candidates(role)
            if unread:
                if BULK_READ and self.reader():
                    text = self.read_ids([u.split(" ", 1)[0] for u in unread], "unread candidates before the finish",
                                         inject=False, count=False)
                    if text:
                        rest = [u for u in unread if u.split(" ", 1)[0] not in self.bulk_tried]
                        return json.dumps({"note": "before the finish, listings that may fit were read (one step)",
                                           "result": text[:FULL_CHARS], **({"check_these": rest} if rest else {}),
                                           "then": "if none fits better" + (", the others included," if rest else "")
                                                   + " finish the same way again"})
                return json.dumps({"not_sent": "not yet: listings that may fit were never read in detail",
                                   "check_these": unread, "then": "read their facts; if none fits better, finish the same way again"})
            gap = self.coverage_gap(role)
            if gap:
                self.search_asked = True
                return json.dumps({"not_sent": "not yet: cheaper listings may be hidden", "why": gap,
                                   "then": "if nothing cheaper fits, finish the same way again"})
            if role == "order":
                swap_note = self.fit_swap()
            choice = self.choice_problems(role)
            if choice:
                self.search_asked = self.search_asked or any("search again" in c for c in choice)
                return json.dumps({"not_sent": "check the choice once more", "problems": choice,
                                   "then": "if you still judge the same, finish the same way again"})
            if role == "order":
                swapped = self.swap_broken_line()
                if swapped:
                    return swapped
            if role == "order" and self.items:
                review = self.review_basket()
                if review:
                    return json.dumps({"not_sent": "the cart check found problems: fix them first", "problems": review,
                                       "then": "if you are sure the check is wrong, order again"})
            if role == "decline":
                review = self.review_decline()
                if review:
                    return json.dumps({"not_sent": "check the decline: the listings read already fit every item", "problems": review,
                                       "then": "if you are sure one of them does not fit, decline again"})
            if role == "decline" and "confirm" not in self.gaps_raised and self.facts and not self.decline_judged \
                    and self.steps_left() > self.reserve() + 1:
                self.gaps_raised.add("confirm")
                return json.dumps({"not_sent": "check the decline once more: most requests can be met, and a decline scores 0 when "
                                               "a fitting listing exists",
                                   "check": "look at the read listings with every requirement confirmed (state note); if one is in "
                                            "stock and within the budget, put it in the cart and order it. Decline again only when no "
                                            "read listing has every stated and told requirement confirmed and further search is not "
                                            "possible."})
            # a right finish still scores 0 when a procedure the request states was skipped: once per finish, before it
            todo = self.procedures_due(role)
            if todo and f"procedures_{role}" not in self.gaps_raised and self.steps_left() > self.reserve():
                self.gaps_raised.add(f"procedures_{role}")
                return json.dumps({"not_sent": "check the request's procedures first", "procedures": todo,
                                   **({"cart_changed": swap_note} if swap_note else {}),
                                   "then": "if each is done or does not apply here, finish the same way again"})
        if role == "decline":
            converted = self.decline_to_order()
            if converted:
                return converted
        told_first = swap_note + "\n" if swap_note else ""
        if role in ("order", "decline"):
            # the message a procedure asks for goes out last, right before the finish, so it is about the cart bought
            told = self.tell_before_finish(role)
            if told is not None:
                sent = told["prompt"] + "\n" if told.get("prompt") else ""
                if (told["changed"] or session.finished) and not told["text"]:
                    return sent + json.dumps({"not_sent": f"before this {role}, your assistant software sent the shopper your whole "
                                                          "request, and a reply changes what to buy: follow it (state note), fix the "
                                                          "cart and finish again", **({"cart_changed": swap_note} if swap_note else {})})
                if told["changed"] or session.finished:
                    return sent + json.dumps({"not_sent": f"as the request asks, the shopper was told about this {role} first, and the "
                                                          "reply changes what to buy: follow it (state note), fix the cart and finish again",
                                              "message": told["text"][:400], "reply": told["reply"],
                                              **({"cart_changed": swap_note} if swap_note else {})})
                told_first += sent + (json.dumps({"told_the_shopper_first": told["text"][:400], "reply": told["reply"]}) + "\n"
                                      if told["text"] else "")
        if role == "details" and BULK_READ:
            batched = self.open_in_batch(name, args)
            if batched:
                return told_first + batched
        swapped = self.props_instead(name, args) if not paging else None
        if swapped:
            name, args = swapped
            role = self.roles.get(name) if self.roles.get(name) in ("details", "other") else "details"
            told_first += (f"(read with {name}, which shows this listing's properties; before the first question the "
                           "properties of the item's listings decide what to ask, and its own page is opened before the order)\n")
        if role == "message" and FIRST_PROMPT_MESSAGE and PROMPT_AT_ASK and not self.prompt_tried:
            sent = self.send_prompt(1, inject=False)
            if sent:
                told_first += sent + "\n"
            if session.finished or session.broken:
                return told_first + json.dumps({"not_sent": "the task ended while the request was being sent"})
            if self.question_budget is not None and (self.questions >= self.question_budget
                                                     or (PROMPT_ALWAYS and self.reserve_left() and self.ask_room() <= 0)):
                return told_first + json.dumps({"not_sent": f"your {'' if self.prompt_cut else 'whole '}request took the messages "
                                                            "left, so this question was not sent; continue without asking" if sent else
                                                            "the question budget is used up; continue without asking"})
        observation = session.call(name, args)
        if observation is None and not session.finished:
            return json.dumps({"not_sent": f"the store did not take the call ({session.broken or 'no reply'})"})
        self.take_in(name, role, args, observation)
        self.note_view(name, args, observation)
        if LINE_SHARES and self.items and role in ("search", "details"):
            self.count_line_step(role, args)
        if role == "search" and self.has_words(name, args):
            self.worded_searches += 1
        self.recent = (self.recent + [(name, signature, len(self.notices))])[-4:]
        if role == "search" and not paging:
            retried = self.search_looser(name, args, observation)
            if retried:
                args, observation, note = retried
                told_first += note
        result = told_first + self.render(name, role, args, observation) + cut_note
        if role == "search" and not paging:
            # the first page goes to the model first; later pages follow as their own results
            self.pending_pages = (name, args, observation)
        # a basket fills one line per call, so it may change the cart once per line before anything is swapped
        lines_ok = len(self.items) if self.items else 1
        if role == "cart_change" and (self.swaps >= 6 + lines_ok or (self.swaps >= 3 + lines_ok and self.steps_left() <= self.pacing() + 4)):
            result += (f"\n(The cart has changed {self.swaps} times since the last finish attempt. Settle on the choice the "
                       "state note supports and finish; switching back and forth wastes the steps a refusal may need.)")
        return result

    def search_looser(self, name: str, args: dict, observation: Any) -> tuple[dict, Any, str] | None:
        """SEARCH_RETRY: a search that showed no row is sent once more with the same words in another match mode (a
        choice of two to four values: a looser one, see LOOSE_MODE, else one code cannot rank against it, never a
        stricter one), as its own step while steps and time allow; a mode that then brings rows is kept for code's own
        searches. Returns the new arguments, result and a note, or None."""
        if not SEARCH_RETRY or not self.searches or self.searches[-1][3] or (isinstance(observation, dict)
                                                                           and ("error" in observation or "refusal" in observation)):
            return None
        if self.session.finished or self.session.broken or self.steps_left() <= self.reserve() + 3 \
                or monotonic() - self.began > WRAP_AT - 40:
            return None
        properties = ((self.session.specs.get(name) or {}).get("parameters") or {}).get("properties") or {}
        rest = json.dumps({k: v for k, v in args.items() if not (isinstance(properties.get(k), dict) and properties[k].get("enum"))},
                          sort_keys=True, default=str)
        for key, field in properties.items():
            values = field.get("enum") if isinstance(field, dict) else None
            if not isinstance(values, list) or not 2 <= len(values) <= 4 \
                    or re.search(r"sort|order|page|categor|group|shelf|depart|aisle|section|brand|seller", key, re.I) \
                    or not (any(looseness(v) for v in values)
                            or re.search(r"mode|match|strateg|method", f"{key} {field.get('description') or ''}", re.I)):
                continue                    # not a match mode: a sort, a shelf or a choice that says nothing of matching
            now = args.get(key)
            # never a stricter value: a looser one first, else one no looser or stricter (names code cannot rank)
            better = [v for v in values if v != now and looseness(v) >= (looseness(now) if now in values else 0)
                      and (name, key, rest, str(v)) not in self.mode_tried]
            if now not in values and not any(looseness(v) > 0 for v in better):
                continue                    # left out: the store's own default stands unless a loose value exists
            if not better:
                continue
            pick = max(better, key=lambda v: (looseness(v), -values.index(v)))
            self.mode_tried.add((name, key, rest, str(pick)))
            again = {**args, key: pick}
            second = self.session.call(name, again)
            if second is None:
                return None
            self.take_in(name, "search", again, second)
            self.note_view(name, again, second)
            shown = bool(self.searches and self.searches[-1][3])
            if shown:
                self.mode_ok[(name, key)] = pick
            self.session.log.append({"role": "assistant", "content": f"note: no row in {key} {now}: sent again with {key} {pick}"
                                                                     f" ({'rows' if shown else 'still none'})"})
            return again, second, (f"(nothing matched with {key} {now if now is not None else 'left out'}, so your assistant "
                                   f"software sent the same search with {key} {pick}:)\n")
        return None

    def mode_value(self, tool: str, key: str, values: list) -> Any:
        """The value code sets in a search's required choice argument: SEARCH_RETRY, the one seen to bring rows where
        another brought none, else the loosest (the first of equals); before, the one meaning words or text, else the
        first."""
        if not SEARCH_RETRY:
            return next((v for v in values if re.search(r"word|text|free|keyword|typed|any", str(v), re.I)), values[0])
        learned = self.mode_ok.get((tool, key))
        return learned if learned in values else max(values, key=lambda v: (looseness(v), -values.index(v)))

    def text_key(self, tool: str) -> tuple[str, dict] | None:
        """The argument a search tool takes free words in, with its other required arguments set to neutral values (a
        choice to the one meaning words or text, a number to its minimum). None when the tool has no such argument or
        needs another value code cannot choose."""
        schema = (self.session.specs.get(tool) or {}).get("parameters") or {}
        properties = schema.get("properties") or {}
        key = next((k for k, v in properties.items() if isinstance(v, dict) and "string" in kinds_of(v) and not v.get("enum")
                    and self.words_key(tool, k)), None)
        if not key:
            return None
        extra: dict = {}
        for name in schema.get("required") or []:
            field = properties.get(name)
            if name == key or not isinstance(field, dict):
                continue
            if isinstance(field.get("enum"), list) and field["enum"]:
                extra[name] = self.mode_value(tool, name, field["enum"])
            elif set(kinds_of(field)) & {"integer", "number"}:
                extra[name] = field.get("minimum", 1)
            else:
                return None
        return key, extra

    def walk_refs(self) -> bool:
        """Code walks a pending pointer chain the way the shopper's helpers do it by hand: unread rows of the seller the
        pointer names are read in one batch; without such rows, the free-text search looks for that seller with a value
        the pointer gives (then with the item's name). resolve_refs then follows each note (and a pairing at the end).
        A few steps a hop, while steps and time allow; the model's own search stays the fallback. Returns True when a
        step was taken."""
        if not WALK_REFS or self.session.finished or self.session.broken:
            return False
        if self.prove_link():
            return True
        ref = next((r for r in self.refs if r["value"] is None and r["ref"].get("seller")), None)
        if WALK_FIRST:
            # the walk takes two steps a hop while the finish reserve (and a read of each cart line) still fits after it
            short_of = self.steps_left() <= self.reserve() + 2 + len(self.cart_book)
            cap = max(8, self.step_cap // 3)
        else:
            short_of, cap = self.steps_left() <= self.pacing() + 8, max(6, self.step_cap // 4)
        if ref is None or self.walk_steps >= cap or short_of or monotonic() - self.began > WRAP_AT - 60:
            return False
        props = self.props_view()
        unread = [i for i in ref.get("maybe", []) if "props" not in self.cover.get(i, set()) and i not in self.walk_props]
        if STRICT_REFS and unread and props:
            # a listing of the seller named that may be the one meant: its properties decide (one step)
            self.walk_props.add(unread[0])
            self.walk_steps += 1
            return bool(self.read_ids(unread[:1], "the listing a pointer may describe", inject=True, count=False, tool=props))
        spec = ref["ref"]
        hop = (spec["seller"].lower(), spec.get("properties") or "")
        tries = self.walk_tries.get(hop, 0)
        if tries >= 3:
            return False
        self.walk_tries[hop] = tries + 1
        seller = spec["seller"].lower()
        kind = tokens(spec.get("item")) - FILLER if ASK_BEFORE_WALK else set()
        rows = [i for i, rec in self.listings.items() if seller in self.record_text.get(i, "") and i not in self.facts
                and i not in self.bulk_tried and not sold_out(rec.get("stock"))
                and (not kind or (tokens(rec.get("title")) | tokens(rec.get("category"))) & kind)]
        if rows and self.reader():
            self.walk_steps += 1
            return bool(self.read_ids(rows, "the listing a pointer names", inject=True, count=False, tool=self.note_reader()))
        tool = next((n for n, r in self.roles.items() if r == "search" and self.text_key(n)), None)
        if tool is None:
            return False
        key, extra = self.text_key(tool)
        values = [p.split(":", 1)[-1].strip() for p in re.split(r";", spec.get("properties") or "") if p.strip()]
        words = " ".join(v for v in values if v and len(v) <= 30)[:60]
        noun = (re.findall(r"[a-z]{3,}", str(spec.get("item") or "").lower()) or [""])[-1]
        queries = list(dict.fromkeys(q.strip() for q in (f"{spec['seller']} {words}", f"{spec['seller']} {noun}", spec["seller"]) if q.strip()))
        searched = self.walk_searches.get(hop, 0)
        self.walk_searches[hop] = searched + 1
        query = queries[min(searched, len(queries) - 1)]
        call = {"id": f"walk-{self.session.steps}", "type": "function",
                "function": {"name": tool, "arguments": json.dumps({key: query, **extra})}}
        self.walk_steps += 1
        self.session.log.append({"role": "assistant", "content": f"note: code searched '{query}' for a pointer's listing"})
        self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        result = self.act(call, paging=True)
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        self.short_of[call["id"]] = self.summary(result)
        return not result.startswith('{"not_sent"')

    def proof_views(self, ident: str) -> list[str]:
        """The views of a chain listing never read, its own page first (that is where a note sends a property on)."""
        gap = self.views_missing([ident], explore=False).get(ident) or []
        pages = set(self.page_views()) | self.bare_views
        return sorted(gap, key=lambda t: t not in pages)

    def proof_can_wait(self, ident: str) -> bool:
        """A chain listing's value may wait for a view of it only while code can still open that view (CHAIN_PROOF)."""
        gap = self.proof_views(ident)
        return bool(gap) and self.unproven.get(ident, 0) < 2 and self.proof_reads < PROOF_READS \
            and self.steps_left() > self.pacing() + 2 and monotonic() - self.began < WRAP_AT - 40 \
            and self.view_args(gap[0], ident) is not None

    def give_up_proof(self, idents: list[str]) -> None:
        """Chain listings code can no longer open give their value from what was read (as before CHAIN_PROOF)."""
        for ident in idents:
            self.unproven[ident] = 2
        if idents:
            self.resolve_refs(list(idents))

    def prove_link(self) -> bool:
        """CHAIN_PROOF: a chain listing waiting for a view of it is opened with that view by code (a step, in the
        conversation as a call and its result), its own page first; reading it resolves or moves the chain. A listing
        code cannot or may no longer open gives its value from what was read. Returns True when a step was taken."""
        if not CHAIN_PROOF or not any(r["value"] is None for r in self.refs):
            return False
        waiting = [i for i, tries in self.unproven.items() if tries < 2 and self.proof_views(i)]
        if not waiting:
            return False
        if self.proof_reads >= PROOF_READS or self.steps_left() <= self.pacing() + 2 or monotonic() - self.began > WRAP_AT - 40:
            self.give_up_proof(waiting)
            return False
        for ident in waiting:
            gap = self.proof_views(ident)
            args = self.view_args(gap[0], ident)
            if args is None:
                self.give_up_proof([ident])
                continue
            self.unproven[ident] += 1
            self.proof_reads += 1
            call = {"id": f"proof-{self.session.steps}", "type": "function", "function": {"name": gap[0], "arguments": json.dumps(args)}}
            self.session.log.append({"role": "assistant", "content": f"note: code opened {ident} on a pointer chain with {gap[0]}"})
            self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            result = self.act(call, paging=True)
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            self.short_of[call["id"]] = self.summary(result)
            if self.unproven.get(ident, 2) < 2 and (self.unproven[ident] >= 2 or self.proof_reads >= PROOF_READS) \
                    and self.proof_views(ident) == gap:
                self.give_up_proof([ident])     # the open showed nothing new of it: no more tries
            return not result.startswith('{"not_sent"')
        return False

    def list_view(self) -> str | None:
        """The store's several-listing view, when it has one."""
        many = self.listing_views()[1]
        return many[0] if many else None

    def props_view(self) -> str | None:
        """The single-listing view that shows a listing's properties: the one that showed the fullest property list so
        far, else one not yet seen to show none (a store may split a listing's properties and its own page, which can
        carry a few facts too, into two views)."""
        single = self.listing_views()[0]
        known = sorted((t for t in single if t in self.fact_views), key=lambda t: -self.sheet_size.get(t, 0))
        return (known or [t for t in single if t not in self.bare_views] or single or [None])[0]

    def props_instead(self, name: str, args: dict) -> tuple[str, dict] | None:
        """PROPS_FIRST: before the first question, an open of a listing's own page (a page view) whose properties were
        never read is made with the properties view instead (the same step): the page's few facts cannot show which
        properties the item's listings share, and the cart lines' pages are read before the order anyway."""
        if not PROPS_FIRST or self.questions or self.ask_room() <= 0 or name not in self.page_views():
            return None
        props = self.props_view()
        ident = next(iter(self.ids_in(args)), None)
        if ident in self.ref_ids or any(s in self.record_text.get(ident or "", "") for s in self.pointer_sellers()):
            return None                     # a listing on a pointer chain: its own page may send the property on
        if not props or not ident or props in self.viewed.get(ident, set()) or "props" in self.cover.get(ident, set()) \
                or self.viewed.get(ident, set()) & {t for t in self.listing_views()[1] if not self.list_bare(t)}:
            return None                     # its properties were read (a several-listing view shows them too)
        new = self.view_args(props, ident)
        return (props, new) if new is not None else None

    def page_views(self) -> list[str]:
        """The other single-listing views seen to show a listing (its own page: price, stock, the seller's note), when
        the properties view shows more properties than they do."""
        props = self.props_view()
        if not props or props not in self.fact_views or self.sheet_size.get(props, 0) < 3:
            return []
        return [t for t in self.listing_views()[0] if t != props and (t in self.bare_views or t in self.fact_views)
                and self.sheet_size.get(t, 0) < self.sheet_size.get(props, 0)]

    def reader(self) -> str | None:
        """The view code reads listings with: the several-listing view (a batch a step), else, with SINGLE_READS, the
        properties view (one listing a step)."""
        many = self.list_view()
        if many and self.list_bare(many) and self.props_view():
            return self.props_view()        # its rows show no property list: one listing a step through the properties view
        return many or (self.props_view() if SINGLE_READS else None)

    def note_reader(self) -> str | None:
        """The view code reads a pointer's rows with: one that shows the seller's notes (a page), else the reader."""
        many = self.list_view()
        if many and self.list_bare(many) and self.view_page.get(many):
            return many
        return self.reader()

    def item_categories(self) -> set[str]:
        """The store's category names the item(s) to buy are filed under, as far as result rows show: those sharing the
        last word of the item's name (its head noun), else those sharing any of its words."""
        names = set(self.shelves.values())
        if not names:
            return set()
        goals = [e["item"] for e in self.items] if self.items else [str(self.brief.get("goal") or "")]
        found: set[str] = set()
        for goal in goals:
            words = re.findall(r"[a-z]{3,}", goal.lower())
            head = tokens(words[-1]) if words else set()
            hit = {c for c in names if head and head & tokens(c)}
            found |= hit or {c for c in names if (tokens(goal) - FILLER) & tokens(c)}
        return found

    def pointer_sellers(self) -> list[str]:
        """Sellers that pending pointers name, in lower case."""
        return [r["ref"]["seller"].lower() for r in self.refs if r["value"] is None and r["ref"].get("seller")]

    def read_queue(self) -> list[str]:
        """Listings shown in results but not read that may be worth a read: in stock, not asked for already; first
        those of a seller a pending pointer names, then those named over a view's limit, then, within the money budget,
        those filed under the item's category (or whose title or category shares a word with the item when no category
        is known yet). Cheapest first when the cheapest is asked for, else the most recently shown first."""
        budget = self.stated_budget()
        cats = self.item_categories()
        words = (tokens(self.brief.get("goal")) | set().union(*[tokens(e["item"]) for e in self.items] or [set()])) - FILLER
        sellers = self.pointer_sellers()
        skip = self.ref_ids
        first: list[str] = []
        second: list[str] = []
        third: list[str] = []
        for ident, record in self.listings.items():
            if ident in self.facts or ident in self.bulk_tried or ident in skip or sold_out(record.get("stock")):
                continue
            text = self.record_text.get(ident, "")
            if any(s in text for s in sellers):
                first.append(ident)
                continue
            price = record.get("price")
            if budget is not None and isinstance(price, (int, float)) and price > budget + 1e-9:
                continue
            if ident in self.read_next:
                second.append(ident)
                continue
            shelf = str(record.get("category") or "")
            if (cats and shelf in cats) or (self.items and self.item_of(ident)) \
                    or (not cats and (tokens(record.get("title")) | tokens(shelf)) & words):
                third.append(ident)

        def ordered(ids: list[str]) -> list[str]:
            if self.cheapest_wanted():
                return sorted(ids, key=lambda i: (not isinstance(self.listings[i].get("price"), (int, float)),
                                                  self.listings[i].get("price") or 0))
            return sorted(ids, key=lambda i: -(self.listings[i].get("step") or 0))
        return ordered(first) + [i for i in self.read_next if i in second] + ordered(third)

    def read_ids(self, ids: list[str], why: str, inject: bool, count: bool = True, tool: str | None = None) -> str:
        """Read listings with the store's several-listing view, at most its limit in one call (one step), or without
        one the first listing with the properties view. From the run loop the call goes into the conversation as its
        own step; inside a model call it is made directly and its compact result returned. Returns what the model
        sees, or "" when nothing was sent."""
        tool = tool or self.reader()
        if not tool or not ids:
            return ""
        ids = list(dict.fromkeys(ids))[:self.view_cap(tool) if tool == self.list_view() else 1]
        args = self.view_args(tool, ids if tool == self.list_view() else ids[0])
        if args is None:
            return ""
        self.bulk_tried |= set(ids)
        if count:
            self.bulk_steps += 1
            self.bulk_turn += 1
        self.session.log.append({"role": "assistant", "content": f"note: code read {len(ids)} listings ({why})"})
        if inject:
            call = {"id": f"read-{self.session.steps}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
            self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            result = self.act(call, paging=True)
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            self.short_of[call["id"]] = self.summary(result)
            return result
        observation = self.session.call(tool, args)
        if observation is None:
            return ""
        role = self.roles.get(tool, "other")
        self.take_in(tool, role, args, observation)
        self.note_view(tool, args, observation)
        if LINE_SHARES and self.items:
            self.count_line_step("details", args)
        return self.render(tool, role, args, observation)

    def bulk_read(self) -> bool:
        """Code's own batch read between model turns: listings shown that may be the item are read with the
        several-listing view, a batch a step and no model call, so the model judges by facts rather than opening rows
        one by one by their titles. Returns True when a read was made."""
        tool = self.reader()
        if not BULK_READ or not tool or self.session.finished or self.session.broken:
            return False
        single = tool != self.list_view()
        if self.bulk_turn >= (SINGLE_PER_TURN if single else 2) or self.bulk_steps >= max(3, self.step_cap // (3 if self.items else 4)):
            return False
        if self.steps_left() <= (self.pacing() if self.has_candidate() else self.reserve()) + 2 \
                or monotonic() - self.began > WRAP_AT - 40:
            return False
        if self.confirmed_in_cart() and not self.cheapest_wanted():
            return False
        queue = self.read_queue()
        if not queue:
            return False
        cap = 1 if single else self.view_cap(tool)
        sellers = self.pointer_sellers()
        pointer_row = any(s in self.record_text.get(i, "") for i in queue[:max(cap, 2)] for s in sellers)
        fits = self.fit_list()
        cheaper = self.cheapest_wanted() and bool(fits) and any(
            isinstance((self.listings.get(i) or {}).get("price"), (int, float)) and self.listings[i]["price"] < fits[0][0] for i in queue)
        reads = self.ask_reads()
        before_asking = (reads < self.reads_to_ask() if not self.items else self.basket_reads_short()) and self.ask_room() > 0
        # one listing a step costs a step a listing: read then only before the first question, a pointer's rows, or a
        # cheaper row when the cheapest is asked for (a batch view also reads a full queue or after the model's turn)
        if not (pointer_row or before_asking or cheaper
                or (not single and (len(queue) >= cap or (len(queue) >= 2 and not self.model_searched)))):
            return False
        return bool(self.read_ids(queue[:cap], "rows that may fit", inject=True))

    def shelf_page(self) -> bool:
        """Before the first question of a one-item order: when fewer listings of the item's category were read than the
        question waits for and no unread row of that category is left to read, code shows the next page of the latest
        search that showed rows of that category (one step, in the conversation as a call and its result); the next
        batch read then reads them. At most SHELF_PAGE_MAX such pages a task."""
        if not SHELF_PAGES or self.items or self.qa or self.ask_room() <= 0 or self.shelf_pages >= SHELF_PAGE_MAX \
                or self.session.finished or self.session.broken or self.session.steps >= self.wait_ends():
            return False
        cats = self.item_categories()
        if not cats or not self.reader() or self.ask_reads() >= self.reads_to_ask():
            return False
        if self.steps_left() <= self.pacing() + 8 or monotonic() - self.began > WRAP_AT - 60:
            return False
        if any((self.listings.get(i) or {}).get("category") in cats for i in self.read_queue()):
            return False                    # rows of the category are waiting: the batch read takes them first
        entry = self.shelf_more()
        key = self.marker_key(entry["tool"], entry["mark"]) if entry else None
        if not entry or not key:
            return False
        args, problems = fit_args((self.session.specs.get(entry["tool"]) or {}).get("parameters") or {"type": "object"},
                                  {**entry["args"], key: entry["mark"]})
        if problems:
            return False
        self.shelf_pages += 1
        call = {"id": f"shelf-{self.session.steps}", "type": "function", "function": {"name": entry["tool"], "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        result = self.act(call, paging=True)
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        self.short_of[call["id"]] = self.summary(result)
        self.session.log.append({"role": "assistant", "content": "note: next page shown to read more of the item's category"})
        return True

    def shelf_more(self) -> dict | None:
        """The latest search that showed rows of the item's category and has a next page left, while code may still
        page (one-item order, before the first question, at most SHELF_PAGE_MAX pages a task); else None."""
        cats = self.item_categories()
        if not SHELF_PAGES or self.items or self.qa or self.shelf_pages >= SHELF_PAGE_MAX or not cats:
            return None
        return max((e for e in self.more_pages.values() if e["mark"] is not None and self.marker_key(e["tool"], e["mark"])
                    and any((self.listings.get(i) or {}).get("category") in cats for i in e["seen"])
                    and (not isinstance(e["total"], int) or len(e["seen"]) < e["total"])), key=lambda e: e["step"], default=None)

    def marker_key(self, tool: str, mark: Any) -> str | None:
        """The argument that takes a next-page marker of this type back (see takes_marker), or None."""
        properties = ((self.session.specs.get(tool) or {}).get("parameters") or {}).get("properties") or {}
        kind = "string" if isinstance(mark, str) else "integer" if isinstance(mark, int) and not isinstance(mark, bool) else ""
        for key, spec in properties.items():
            if isinstance(spec, dict) and kind in kinds_of(spec) and not spec.get("enum") \
                    and re.search(r"page|cursor|token|next|resume|offset|continu|marker|after", f"{key} {spec.get('description') or ''}".replace("_", " "), re.I):
                return key
        return None

    def open_in_batch(self, name: str, args: dict) -> str:
        """A model's open of one unread listing, made as a several-listing read of it and the unread rows of its category
        beside it (the same one step). Returns what the model is told, or "" to open it the usual way."""
        tool = self.list_view()
        if not tool or name not in self.listing_views()[0] or self.view_cap(tool) < 2:
            return ""
        ident = next(iter(self.ids_in(args)), None)
        if not ident or ident in self.facts:
            return ""
        shelf = (self.listings.get(ident) or {}).get("category")
        extra = [i for i in self.read_queue() if i != ident and (not shelf or (self.listings.get(i) or {}).get("category") == shelf)]
        if not extra:
            return ""
        text = self.read_ids([ident] + extra, "opened with the unread rows of its category", inject=False, count=False)
        if not text or ident not in self.facts:
            return ""                       # the list view did not show it: open it on its own
        return f"(read with {min(len(extra), self.view_cap(tool) - 1)} more unread listings of the same category, one step)\n" + text

    def view_field(self, tool: str) -> str | None:
        """The one required argument of a listing view: the id, or the list of ids."""
        schema = (self.session.specs.get(tool) or {}).get("parameters") or {}
        properties = schema.get("properties") or {}
        required = [k for k in schema.get("required") or [] if isinstance(properties.get(k), dict)]
        return required[0] if len(required) == 1 else None

    def view_cap(self, tool: str) -> int:
        """How many listings one call of a several-listing view may name: what a failed call taught, else the schema's
        limit, else a number the tool's or the field's description gives, else VIEW_CAP."""
        if tool in self.view_caps:
            return self.view_caps[tool]
        spec = self.session.specs.get(tool) or {}
        key = self.view_field(tool)
        field = (((spec.get("parameters") or {}).get("properties") or {}).get(key) or {}) if key else {}
        if isinstance(field.get("maxItems"), int) and field["maxItems"] >= 1:
            return field["maxItems"]
        text = f"{spec.get('description') or ''} {field.get('description') or ''}"
        words = "|".join(NUMBER_WORDS)
        found = re.search(rf"\b(?:up to|at most|maximum(?: of)?|max\.?|no more than|not more than|as many as)\s+(\d\d?|{words})\b",
                          text, re.I)
        if found:
            raw = found.group(1).lower()
            number = int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw, 0)
            if 2 <= number <= 20:
                return number
        return VIEW_CAP

    @staticmethod
    def limit_said(observation: Any) -> int | None:
        """How many listings one call may name, when a refused call's reply states it ("at most three ids"), else None."""
        text = observation if isinstance(observation, str) else json.dumps(observation, default=str)
        words = "|".join(NUMBER_WORDS)
        found = re.search(rf"\b(?:up to|at most|maximum(?: of)?|max\.?|no more than|not more than|as many as)\s+(\d\d?|{words})\b",
                          text[:2000], re.I)
        if not found:
            return None
        raw = found.group(1).lower()
        number = int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw, 0)
        return number if 1 <= number <= 20 else None

    def cut_to_cap(self, name: str, args: dict) -> str:
        """A several-listing call naming more listings than the view takes is cut to its limit before it costs a step;
        the rest wait in read_next. Returns a note for the result, or ""."""
        if name not in self.listing_views()[1]:
            return ""
        key = self.view_field(name)
        ids = args.get(key) if key else None
        if not isinstance(ids, list):
            return ""
        cap = self.view_cap(name)
        known = ""
        if REREAD_SWAP and not self.carted and len(ids) >= 2:
            # listings read in detail since the latest change are known already: their facts go in the result, and their
            # places go to unread rows that may fit (of the same category first), so the step reads something new
            # (COVER_BY_CONTENT: a view that shows pages reads something new of a listing whose page was never shown)
            pages = COVER_BY_CONTENT and (self.view_page.get(name) or name not in self.view_shown)
            again = [str(i) for i in ids if str(i) in self.facts and self.read_step.get(str(i), -1) >= self.news_step
                     and not (pages and "page" not in self.cover.get(str(i), set()))]
            fresh = [str(i) for i in ids if str(i) not in again]
            if again and fresh or again and self.read_queue():
                shelves = {(self.listings.get(i) or {}).get("category") for i in ids}
                spare = sorted((i for i in self.read_queue() if i not in ids), key=lambda i: (self.listings.get(i) or {}).get("category") not in shelves)
                ids = fresh + spare[:len(again)]
                args[key] = ids
                known = "\n(Read already, facts as read: " + " | ".join(
                    f"{i}: {short(self.facts.get(i), 220)}" for i in again)[:1200] + ")"
                self.session.log.append({"role": "assistant", "content": f"note: re-read of {short(again, 120)} swapped for unread rows"})
        if len(ids) <= cap:
            return known
        args[key] = ids[:cap]
        rest = [str(i) for i in ids[cap:]]
        self.read_next = list(dict.fromkeys(self.read_next + rest))
        return known + (f"\n(This view takes at most {cap} listings a call, so the first {cap} were read; not read yet: "
                        f"{', '.join(rest)[:300]}.)")

    def category_key(self, name: str, key: str) -> bool:
        """True for a search argument that takes a category, department, aisle or shelf name: by its name or
        description, or because it already took a name the store uses."""
        if (name, key) in self.cat_keys:
            return True
        field = (((self.session.specs.get(name) or {}).get("parameters") or {}).get("properties") or {}).get(key) or {}
        if not isinstance(field, dict) or field.get("enum") or "string" not in kinds_of(field):
            return False
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)
        return bool(re.search(r"categor|department|\bdept|section|aisle|shelf", f"{spaced} {field.get('description') or ''}", re.I))

    def words_key(self, name: str, key: str) -> bool:
        """True for the argument a search takes its free words in."""
        field = (((self.session.specs.get(name) or {}).get("parameters") or {}).get("properties") or {}).get(key) or {}
        about = str(field.get("description") or "") if isinstance(field, dict) else ""
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)
        return bool(re.search(r"quer|text|word|search|keyword|term|phrase|\bq\b|seek", f"{spaced} {about}", re.I)) \
            and not self.category_key(name, key)

    def filter_fix(self, name: str, args: dict) -> str:
        """Before a search: a category value is written as the store writes its category names, once some were seen in
        results (a near miss filters out everything); a value that is a property rather than a category goes back to the
        search words; a filter argument that only ever returned nothing in this task is left out. Returns why the call
        is not sent, else "" (args may be changed in place)."""
        known = sorted(set(self.shelves.values()))
        for key, value in list(args.items()):
            if not isinstance(value, str) or not value.strip() \
                    or re.search(r"page|offset|cursor|token|marker|after|sort|order|mode|limit", key, re.I):
                continue
            if self.category_key(name, key):
                if not known:
                    continue
                low = value.strip().lower()
                if low in self.shelves:
                    args[key] = self.shelves[low]
                    continue
                words = tokens(value) - FILLER
                near = [c for c in known if words and (words <= tokens(c) or (tokens(c) - FILLER) <= words)]
                if len(near) == 1:
                    args[key] = near[0]
                    self.session.log.append({"role": "assistant", "content": f"note: category '{value[:40]}' sent as '{near[0]}'"})
                    continue
                listed = ", ".join(known[:15])
                if len(near) > 1:
                    return (f"'{value[:40]}' is not one of this store's category names; those that match it: "
                            f"{', '.join(near[:6])}. Use one exactly as written.")
                if any(low == str(v).strip().lower() for sheet in self.facts.values() for v in sheet.values()):
                    return f"'{value[:40]}' is a property value, not a category: put it in the search words. Categories seen: {listed}."
                if (name, key, low) in self.empty_values:
                    return f"'{value[:40]}' returned nothing as a category before. Categories seen: {listed}; or search with words."
                continue                    # a category not seen yet may just not have come up: it is tried once
            if self.empty_args.get((name, key), 0) >= 2 and (name, key) not in self.args_worked and not self.words_key(name, key):
                return (f"the '{key}' filter returned nothing twice in this task and never any listing: leave it out and put "
                        "that value in the search words instead.")
        return ""

    def render(self, name: str, role: str, args: dict, observation: Any) -> str:
        """A tool result as the model reads it. Search rows keep their short fields and lose long prose (the code keeps
        all of it); a several-listing view becomes one line per listing with its facts and notes; others stay whole."""
        if role == "search" and isinstance(observation, (dict, list)):
            return shown(trimmed(observation, 110), FULL_CHARS)
        if name in self.listing_views()[1] and isinstance(observation, (dict, list)):
            key = self.view_field(name)
            asked = [str(i) for i in args.get(key) or []] if key and isinstance(args.get(key), list) else []
            lines = self.listing_lines(asked, observation)
            if lines:
                return lines
        return shown(observation, FULL_CHARS)

    def listing_lines(self, ids: list[str], observation: Any) -> str:
        """Several listings read in one call, as compact lines from what the code took in: notices first, word for
        word, then per listing its title, price, stock, category, facts and its texts (shortest first: a seller's note
        is usually shorter than a description), then the ids that did not come back."""
        back = [i for i in ids if i in self.facts or i in self.record_text]
        if not back:
            return ""
        texts: dict[str, list[str]] = {}

        def walk(value: Any, depth: int = 0) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                own = next((v for v in value.values() if isinstance(v, str) and v in back), None)
                if own:
                    title = str((self.listings.get(own) or {}).get("title") or "")
                    texts.setdefault(own, []).extend(v for v in value.values() if isinstance(v, str) and len(v) >= 20
                                                     and len(v.split()) >= 3 and v[:90] != title)
                for v in value.values():
                    walk(v, depth + 1)
            elif isinstance(value, list):
                for v in value[:60]:
                    walk(v, depth + 1)
        walk(observation)
        out = [f"Notice: {line}" for line in notices_in(observation)]
        for ident in back:
            info = self.listings.get(ident) or {}
            facts = "; ".join(f"{k}={v}" for k, v in (self.facts.get(ident) or {}).items())
            prose_parts = sorted(dict.fromkeys(texts.get(ident, [])), key=len)[:2]
            out.append(f"{ident} | {str(info.get('title') or '')[:80]} | price {info.get('price')} | stock {info.get('stock')} | "
                       f"{str(info.get('category') or '')[:40]} | facts: {facts[:600]}"
                       + "".join(f" | text: {t[:320]}" for t in prose_parts))
        missing = [i for i in ids if i not in back]
        if missing:
            out.append("Not returned: " + ", ".join(missing))
        return "\n".join(out)

    def takes_out(self, name: str, args: dict) -> bool:
        """True when a cart call only takes lines out: a remove tool, a count of 0, or (CART_RECORD) a choice value that
        takes out: one whose words say so, the undoing form of another value ("unpack" of "pack"), or, of two values,
        the one that is not the putting-in one."""
        spec = self.session.specs.get(name) or {}
        own = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ")
        if re.search(r"remov|drop|delet|take (?:out|off)|discard|clear", f"{own} {spec.get('description') or ''}", re.I) \
                and not re.search(r"\b(?:add|put|set|keep|hold|place)\b", own, re.I):
            return True
        if CART_RECORD:
            properties = ((spec.get("parameters") or {}).get("properties")) or {}
            for key, value in args.items():
                field = properties.get(key)
                if not isinstance(value, str) or not isinstance(field, dict) or not isinstance(field.get("enum"), list):
                    continue
                values = [str(v) for v in field["enum"]]
                text = value.replace("_", " ").lower()
                if re.search(r"remov|drop|delet|discard|releas|exclud|clear|take ?(?:out|off)|\bout\b", text):
                    return True
                if any(text in (f"un{v.lower()}", f"de{v.lower()}", f"dis{v.lower()}") for v in values if v != value):
                    return True
                putting = r"add|put|includ|insert|place|keep|stow|pack|push|\bset\b"
                if len(values) == 2 and value in values:
                    other = values[1] if values[0] == value else values[0]
                    if re.search(putting, other.replace("_", " "), re.I) and not re.search(putting, text, re.I) \
                            and not re.search(r"updat|chang|modif|edit|replac|adjust|swap|more|less|increas|decreas", text):
                        return True
        counts = [v for k, v in args.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return bool(counts) and all(v == 0 for v in counts)

    def commit_check(self, name: str, role: str, args: dict, left: int) -> str:
        """For one item, code holds the order of commitment (each stop at most twice a task, never costing a step):
        - a listing goes into the cart only when every requirement is confirmed, or once the research is done (six or
          more listings read, or the pacing point near) and nothing contradicts it;
        - with a pick in the cart and nothing new since it was set, a search would mostly swap a fine pick: view the
          cart and order instead;
        - after a refusal that only moved the price, re-read the pick if needed and order the same cart again."""
        held = [line["id"] for line in self.cart_lines()[0]] if self.cart_obs is not None else list(self.last_carted)
        late = left <= self.pacing() + 4
        if role == "cart_change" and not self.takes_out(name, args) and self.commit_stops < 2 and not late:
            # a listing already in the cart may change its count freely; only a new pick waits for its evidence
            adding = [i for i in self.ids_in(args) if i not in held]
            confirmed = {i for _, i in self.fit_list()}
            open_ok = len(self.item_reads()) >= 6 and set(adding) <= set(self.ranked_candidates())
            unread = [i for i in adding if i not in self.facts]
            if adding and not set(adding) <= confirmed and not open_ok:
                self.commit_stops += 1
                ident = adding[0]
                checks, _ = self.marks(ident) if ident in self.facts else ([], True)
                open_points = [c for c in checks if not c.split(": ", 1)[-1].startswith(("yes", "ok"))]
                best = self.ranked_candidates()[:3]
                then = ("Keep the cart as it is and order, unless a notice or a reply changed something." if held else
                        "Read more candidates first, or search with the requirement's own words.")
                return (f"not yet: read {ident}'s details first" if unread else
                        f"not yet: {ident} has requirements not confirmed ({'; '.join(open_points)[:200]})") + \
                    ". Put a listing in the cart once every requirement is confirmed, or, after the cheapest 6-10 candidates " \
                    "are read, the best one nothing contradicts" + (f" (best so far: {', '.join(best)})" if best else "") + \
                    ". " + then
        moved = [i for i in self.price_refused if i in held]
        if role == "search" and moved and self.finish_tries == self.price_refusal_tries:
            return (f"the store said only the price of {', '.join(moved)} changed: read it again if needed and order the same "
                    "cart again; take a line out only if its new price breaks the budget.")
        if role == "search" and held and self.after_cart_stops < 2 and not late and self.cart_changed > self.news_step \
                and self.refusals == self.refusals_at_cart and not self.search_asked:
            self.after_cart_stops += 1
            return (f"the cart already holds {', '.join(held)} and nothing new came in since it was set: another search now "
                    "mostly swaps a pick that fits. View the cart if needed and order; search again only after a notice, a "
                    "reply or a refused order changes something.")
        return ""

    def takes_marker(self, tool: str, mark: Any) -> bool:
        """True when the tool has an argument that can take a next-page marker back: a page, cursor, token or offset
        field of the marker's own type. A marker the tool cannot take back would only mislead."""
        properties = ((self.session.specs.get(tool) or {}).get("parameters") or {}).get("properties") or {}
        kind = "string" if isinstance(mark, str) else "integer" if isinstance(mark, int) and not isinstance(mark, bool) else ""
        for key, spec in properties.items():
            if not isinstance(spec, dict) or kind not in kinds_of(spec) or spec.get("enum"):
                continue
            words = f"{key} {spec.get('description') or ''}".replace("_", " ")
            if re.search(r"page|cursor|token|next|resume|offset|continu|marker|after", words, re.I):
                return True
        return False

    @staticmethod
    def unwrapped(observation: Any) -> dict:
        """A result without the wrapper a world may put around it to carry notices (the dict beside a list of sentences)."""
        if not isinstance(observation, dict):
            return {}
        if len(observation) == 2:
            values = list(observation.values())
            lines = next((v for v in values if isinstance(v, list) and v and all(isinstance(x, str) for x in v)), None)
            other = next((v for v in values if v is not lines and isinstance(v, dict)), None)
            if lines is not None and other is not None:
                return Agent.unwrapped(other)
        return observation

    def take_in(self, name: str, role: str, args: dict, observation: Any) -> None:
        """Book-keeping after a step: questions, cart, listings and notices."""
        step = self.session.steps
        failed = isinstance(observation, dict) and ("error" in observation or "refusal" in observation)
        if not failed and role in ("details", "other"):
            self.note_view(name, args, observation, classify=False)  # booked first: the chain checks below ask which
                                                                       # views were read (classified once fields are learned)
        if role in ("order", "decline"):
            self.finish_tries += 1
            self.search_asked = False
            if not self.session.finished:
                self.refusals += 1
                self.swaps = 0
                said = " ".join(notices_in(observation))
                if role == "order" and self.cart_obs is not None and PRICE_MOVED.search(said) and not UNAVAILABLE.search(said):
                    # only a price moved: the pick usually still fits; the cheaper-fit check runs again, the searches do not
                    self.gaps_raised -= {"choice"}
                    self.price_refusal_tries = self.finish_tries
                    for line in self.cart_lines()[0]:
                        self.price_refused[line["id"]] = step
                else:
                    self.gaps_raised -= {"order", "coverage", "choice", "unread_cart"}   # the situation changed: check the next choice again
                if role == "order" and self.cart_obs is not None and UNAVAILABLE.search(said):
                    # the refusal says the pick can no longer be had: what the cart held is out
                    for line in self.cart_lines()[0]:
                        if line["id"] in self.listings:
                            self.listings[line["id"]]["stock"] = "unavailable at checkout"
        news: list[str] = []
        if role == "message" and not (isinstance(observation, dict) and observation.get("error") in ("budget_exceeded", "invalid_arguments")):
            self.questions += 1
            asked = " ".join(str(v) for v in args.values() if isinstance(v, str))
            self.qa.append((asked[:300], short(observation, 400)))
            news.append(f"Question: {asked[:300]}\nShopper's reply: {short(observation, 600)}")
        if role == "cart_change" and not failed:
            self.cart_changed = step
            self.refusals_at_cart = self.refusals
            self.carted = True
            self.swaps += 1
            self.last_carted = self.ids_in(args)
            echoed = json.dumps(observation, default=str)
            if self.takes_out(name, args):
                self.code_put = [i for i in self.code_put if i not in self.last_carted]
                self.taken_out = list(self.last_carted)
                cleared = not self.last_carted and re.search(r"clear|empty|\ball\b", f"{name} {(self.session.specs.get(name) or {}).get('description') or ''}".replace("_", " "), re.I)
                # a clear-all call names no listing: the cart is empty
                self.cart_book = [] if cleared else [i for i in self.cart_book if i not in self.last_carted]
            elif self.list_shaped(name):
                self.cart_book = list(self.last_carted)          # one call sets the whole cart
            else:
                # a reply that does not name the listing did not take it (a refusal under a key code does not know),
                # unless (CART_RECORD) it says it went through: a true flag, a count of one or more, an "ok"
                took = CART_RECORD and cart_reply_took(observation)
                self.cart_book += [i for i in self.last_carted if i not in self.cart_book and (i in echoed or took)]
        if role == "cart_view" and not failed:
            if name not in self.fields:
                self.learn_fields(name, role, observation)      # the cart's lines are read with its own field names
            self.cart_seen = (step, short(observation, 900))
            self.cart_obs = observation
            self.cart_book = [line["id"] for line in self.cart_lines()[0]]
            if not self.cart_book:
                # lines code could not read: the listings the view names (an empty cart names none)
                shown = json.dumps(observation, default=str)
                self.cart_book = [i for i in dict.fromkeys(self.code_put + self.last_carted + list(self.listings)) if len(i) >= 4 and i in shown]
            self.code_put = []
        fields = self.learn_fields(name, role, observation) if not failed else self.fields.get(name, {})
        found: list[dict] = []
        records_in(observation, found, fields=fields)
        for record in found:
            shelf = str(record.get("category") or "").strip()
            if shelf:
                self.shelves.setdefault(shelf.lower(), shelf)
        # a several-listing view counts as a details view: its listings are read in full
        book = "details" if not failed and name in self.listing_views()[1] else role
        if role == "search" and not failed:
            for key, value in args.items():
                if not isinstance(value, str) or not value.strip() or re.search(r"page|offset|cursor|token|marker|after", key, re.I):
                    continue
                if found:
                    self.args_worked.add((name, key))
                    if value.strip().lower() in self.shelves:
                        self.cat_keys.add((name, key))
                else:
                    self.empty_args[(name, key)] = self.empty_args.get((name, key), 0) + 1
                    self.empty_values.add((name, key, value.strip().lower()))
            inner = self.unwrapped(observation)
            total = inner.get(fields["matches"]) if fields.get("matches") in inner else None
            if not isinstance(total, int) or isinstance(total, bool):
                total = next((v for k, v in inner.items() if re.search(r"match|total|count|found|hits|results", str(k), re.I)
                              and isinstance(v, int) and not isinstance(v, bool)), None)
            self.searches.append((step, short({k: v for k, v in args.items() if v not in ("", None)}, 140), total, len(found)))
            self.sweeps.append((dict(args), total, {r["id"] for r in found}))
            mark = inner.get(fields["next_page"]) if fields.get("next_page") in inner else None
            if mark in (None, "", False):
                mark = next((v for k, v in inner.items() if re.search(r"next|follow|cursor|continu", str(k), re.I)
                             and isinstance(v, (str, int)) and not isinstance(v, bool) and v not in ("", 0)), None)
            if mark not in (None, "", False) and self.takes_marker(name, mark):
                self.page_marks[name] = mark
            else:
                self.page_marks.pop(name, None)
                mark = None
            # every search keeps its own count of rows seen against its total, so an older one with pages left is not forgotten
            plain = {k: v for k, v in args.items() if v not in ("", None) and not re.search(r"page|offset|cursor|start|token|marker|after", k, re.I)}
            entry = self.more_pages.setdefault(name + json.dumps(plain, sort_keys=True, default=str),
                                               {"tool": name, "args": plain, "total": None, "seen": set(), "mark": None, "step": step})
            entry["total"] = total if isinstance(total, int) and not isinstance(total, bool) else entry["total"]
            entry["seen"] |= {r["id"] for r in found}
            entry["mark"], entry["step"] = mark, step
        for record in found:
            known = self.listings.get(record["id"], {})
            self.listings[record["id"]] = {"id": record["id"], "title": record["title"] or known.get("title", ""),
                                           "price": record["price"] if record["price"] is not None else known.get("price"),
                                           "stock": record["stock"] if record["stock"] is not None else known.get("stock"),
                                           "category": record.get("category") or known.get("category", ""), "step": step,
                                           "stock_step": step if record["stock"] is not None else known.get("stock_step")}
            if record["id"] in self.price_refused and record["price"] is not None and role in ("details", "search", "other"):
                self.price_reread.add(record["id"])
        sheets: list[tuple[str, dict]] = []
        facts_in(observation, sheets, fields=fields)
        best: dict[str, dict] = {}
        for ident, sheet in sheets:
            if len(sheet) > len(best.get(ident, {})):
                best[ident] = sheet
        if book == "details" and not best:
            # flat records: one, or several from a tool that shows a few listings at once
            inner = self.unwrapped(observation)
            records = [inner] + [r for v in inner.values() if isinstance(v, list) for r in v[:10] if isinstance(r, dict)]
            for record in records:
                mine = own_fields(record, fields=fields)
                if mine:
                    best[mine[0]] = mine[1]
        for ident, sheet in best.items():
            # a details tool's list adds to what was known (newer values win); elsewhere only a fuller list of three counts
            if book == "details":
                self.facts[ident] = {**self.facts.get(ident, {}), **sheet}
                self.read_step[ident] = step
            elif len(sheet) >= 3 and len(sheet) >= len(self.facts.get(ident, {})):
                self.facts[ident] = sheet
        if name in self.listing_views()[1] and (VIEW_HALVE or not failed):
            # a call naming several listings that gave none of them back went over the view's limit (whatever its reply
            # says: a store may answer as if the listings did not exist): name fewer next time
            key = self.view_field(name)
            asked = [str(i) for i in args.get(key) or []] if key and isinstance(args.get(key), list) else []
            got = set(asked) & ({r["id"] for r in found} | set(best))
            # a reply that names a listing asked for showed it, even when its records could not be read
            echoed = not failed and any(i in json.dumps(observation, default=str) for i in asked)
            if len(asked) >= 2 and not got and not echoed and len(asked) > self.view_ok.get(name, 0):
                said = self.limit_said(observation)
                if said and said < len(asked):
                    lower = said
                elif VIEW_HALVE:
                    # half the call, never below a call that worked: a limit of 2 is found in one step, not two
                    lower = max(2, self.view_ok.get(name, 0), len(asked) // 2)
                else:
                    lower = max(2, len(asked) - 1)
                self.view_caps[name] = max(1, min(self.view_cap(name), lower))
                if VIEW_HALVE and len(asked) > self.view_caps[name]:
                    # the call showed none of its listings: they wait to be read again under the new limit (once each)
                    back = [i for i in asked if i not in self.view_retried and i not in self.facts]
                    self.view_retried |= set(back)
                    self.bulk_tried -= set(back)
                    self.read_next = list(dict.fromkeys(back + self.read_next))
            elif got:
                self.view_ok[name] = max(self.view_ok.get(name, 0), len(got))
            self.read_next = [i for i in self.read_next if i not in got]
        if not failed and book in ("details", "search", "other"):
            # records are kept even with no pointer pending: one told later may point at a listing read before
            shown_ids = list(dict.fromkeys([r["id"] for r in found] + list(best)))
            self.note_records(observation, shown_ids, book, name)
            if self.refs:
                self.resolve_refs(shown_ids)
        for line in notices_in(observation):
            if line not in self.notices:
                self.notices.append(line)
                news.append(f"Notice: {line}")
                if UNAVAILABLE.search(line):
                    # a notice that the pick is gone: the listing it names, else what the cart holds (its last view
                    # when nothing changed since, else the listings the latest cart change named) is out
                    named = [i for i in self.listings if len(i) >= 4 and i in line]
                    fresh = self.cart_obs is not None and self.cart_seen is not None and self.cart_seen[0] >= self.cart_changed
                    held = [l["id"] for l in self.cart_lines()[0]] if fresh else self.last_carted
                    for ident in named or held:
                        if ident in self.listings:
                            self.listings[ident]["stock"] = "unavailable (notice)"
        if news:
            told_before, dropped_before, budget_before = len(self.told), len(self.dropped), self.brief.get("money_budget")
            read = self.update_needs("\n".join(news))
            if role == "message" and self.qa and read.get("reply_kind") in REPLY_KINDS:
                self.qa_kind[len(self.qa) - 1] = read["reply_kind"]
            if role == "message" and self.qa and (self.qa_kind.get(len(self.qa) - 1) == "one_at_a_time"
                                                  or ONE_AT_A_TIME.search(self.qa[-1][1])):
                self.plain_all = True       # this shopper takes one plain thing per message: no examples or lists from now on
            notes = [n[len("Notice: "):] for n in news if n.startswith("Notice: ")]
            if read.get("unavailable") is True and notes and not any(UNAVAILABLE.search(n) for n in notes):
                # the reader saw by its meaning what the stock words missed: the listing it names is gone, else the
                # cart's (at checkout, or its last view when nothing changed since, else the latest cart change's)
                named = [i for i in self.listings if len(i) >= 4 and any(i in n for n in notes)]
                fresh = self.cart_obs is not None and self.cart_seen is not None and self.cart_seen[0] >= self.cart_changed
                held = [l["id"] for l in self.cart_lines()[0]] if fresh else self.last_carted
                for ident in named or held:
                    if ident in self.listings:
                        self.listings[ident]["stock"] = "unavailable (notice)"
            if any(n.startswith("Notice:") for n in news) or len(self.told) > told_before or len(self.dropped) > dropped_before \
                    or self.brief.get("money_budget") != budget_before:
                self.news_step = step
            if (len(self.told) > told_before or len(self.dropped) > dropped_before or self.brief.get("money_budget") != budget_before) and \
                    any(at == step for at in self.price_refused.values()):
                # the refusal also changed the budget or a requirement: not a price change alone after all
                self.price_refused = {i: at for i, at in self.price_refused.items() if at != step}
                self.gaps_raised -= {"order", "coverage", "choice", "unread_cart"}

    def update_needs(self, text: str) -> dict:
        """Keep the requirement list current with what the shopper told or a notice changed (one small model call).
        Returns what the reader recorded (with the kind of reply and whether something became unavailable), or {}."""
        if monotonic() - self.began > WRAP_AT - 30:
            return {}
        current = "\n".join(f"- {r['item'] + ' > ' if r.get('item') else ''}{r['what']}: {r['value']}" for r in self.reqs) or "(none)"
        items = ("ITEMS IN THE ORDER: " + "; ".join(e["item"] for e in self.items) + "\n\n") if self.items else ""
        reply = self.mind.ask([{"role": "system", "content": NEEDS_GUIDE},
                               {"role": "user", "content": f"{items}CURRENT REQUIREMENTS:\n{current}\n\nTEXT:\n{text}\n\nCall record_changes."}],
                              [NEEDS_TOOL], max_tokens=2000, models=READER_MODELS)
        calls = (reply or {}).get("tool_calls") or []
        found = parse_json((calls[0].get("function") or {}).get("arguments")) if calls else None
        if not isinstance(found, dict):
            return {}
        # in an order of several items, a value the reader files under no item belongs to the one item the question and
        # its reply name most (unless they speak of all of them): left unfiled it would bind every item
        sole = ""
        exchange = text.split("\nNotice: ")[0]
        if self.items and exchange.startswith("Question:") and not re.search(r"\b(?:all|every|each|both|everything)\b", exchange, re.I):
            scores = sorted(((len(tokens(e["item"]) & tokens(exchange)), e["item"]) for e in self.items), reverse=True)
            if scores and scores[0][0] and (len(scores) == 1 or scores[1][0] < scores[0][0]):
                sole = scores[0][1]
        refs_before, added = len(self.refs), []
        for change in found.get("changes") or []:
            if not isinstance(change, dict) or not str(change.get("what") or "").strip():
                continue
            what, value = str(change["what"])[:80], str(change.get("value") or "")[:120]
            item = (self.item_named(str(change.get("item") or "") + " " + what) or sole) if self.items else ""
            ref = change.get("same_as") if isinstance(change.get("same_as"), dict) and change.get("action") != "drop" else None
            if ref is not None and not (str(ref.get("seller") or "").strip() or str(ref.get("properties") or "").strip()):
                ref = None
            if ref is not None:
                value = "to look up"            # add_ref sets the placeholder once the requirement exists
            # a change to one item's requirement leaves the same property of the other items alone; an item's own name
            # inside "what" says which item, not which property
            prop = tokens(what) - tokens(item)
            same = [r for r in self.reqs if (tokens(r["what"]) - tokens(r.get("item") or "")) & prop
                    and (not item or r.get("item") in (None, item))]
            if change.get("action") == "drop":
                if found.get("reply_kind") in ("no_preference", "not_understood", "one_at_a_time"):
                    # a reply that only leaves a property open ("any hood is fine", often the stock answer to a message
                    # that did not ask about it) never undoes what the request itself states: a stated requirement
                    # drops only when the shopper or a notice says to drop it; a need told earlier may go
                    kept = [r for r in same if not r.get("told")]
                    same = [r for r in same if r.get("told")]
                    if kept:
                        self.session.log.append({"role": "assistant", "content": "note: kept stated " + short([r["what"] for r in kept], 120)
                                                 + " against a no-preference reply"})
                    if not same:
                        continue
                self.reqs = [r for r in self.reqs if r not in same]
                self.dropped.append((self.session.steps, f"{item + ' > ' if item else ''}{what}"))
            elif value.strip():
                if change.get("action") == "replace":
                    self.reqs = [r for r in self.reqs if r not in same]
                requirement = {"what": what, "value": value, "told": True, **({"item": item} if item else {})}
                self.reqs.append(requirement)
                added.append(requirement)
                if ref is not None:
                    self.add_ref(requirement, what, item, ref)
                self.told.append((self.session.steps, f"{item + ' > ' if item else ''}{what}: {requirement['value']}"))
        self.drop_ref_details(text, self.refs[refs_before:], added)
        change = found.get("quantity") if isinstance(found.get("quantity"), dict) else {}
        units = change.get("units")
        if isinstance(units, int) and not isinstance(units, bool) and 1 <= units <= 99 and number_in(text, units):
            # a new count holds only when the text itself writes it; it then counts like one the request writes
            item = self.item_named(str(change.get("item") or "")) if self.items else ""
            entry = next((e for e in self.items if e["item"] == item), None) if self.items else None
            if entry is not None or not self.items:
                if entry is not None:
                    entry["quantity"] = units
                else:
                    self.brief["quantity"] = units
                self.quantity_text += " " + text
                self.told.append((self.session.steps, f"{item + ' > ' if item else ''}quantity: {units}"))
        budget = found.get("money_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and \
                (number_in(text, int(budget)) if float(budget).is_integer() else f"{budget:g}" in text):
            self.brief["money_budget"] = budget
            self.budget_text = text
        self.session.log.append({"role": "assistant", "content": "note: requirements now " + short(self.reqs, 600)
                                 + (f"; budget {self.brief.get('money_budget')}" if budget and budget > 0 else "")
                                 + f"; reply {found.get('reply_kind')}" + ("; unavailable" if found.get("unavailable") is True else "")})
        return found

    def drop_ref_details(self, text: str, refs: list[dict], reqs: list[dict]) -> None:
        """REF_DETAIL_GUARD: among reqs, a requirement whose value is one of the details that identify the listing one of
        refs points at (a pointer made from the same text, for the same item), while the text writes that value only
        once, describes that listing, not the item: it is dropped. A value the text writes twice may be both."""
        if not REF_DETAIL_GUARD or not refs:
            return
        plain = str(text).lower()
        dropped = []
        for requirement in reqs:
            value = str(requirement.get("value") or "").strip().lower()
            words = tokens(value) - FILLER
            if requirement.get("pending_ref") or requirement not in self.reqs or not words:
                continue
            parts = re.findall(r"\d+(?:\.\d+)?|[a-z]+|[^\sa-z\d]", value)
            written = len(re.findall(r"(?<![a-z0-9])" + r"\s*".join(re.escape(x) for x in parts) + r"(?![a-z0-9])", plain))
            if written != 1:
                continue
            for ref in refs:
                if (ref.get("item") or "") != (requirement.get("item") or ""):
                    continue
                details = ref["ref"].get("properties") or ""
                pieces = [x.split(":", 1)[-1] for x in re.split(r";" if ";" in details else r",|\band\b", details) if x.strip()]
                # "graphite" against "colour: graphite", or "128 gb" against "storage 128 gb" when the requirement is storage
                if any(words in (tokens(x) - FILLER, tokens(x) - FILLER - tokens(requirement.get("what"))) for x in pieces):
                    dropped.append(requirement)
                    break
        if dropped:
            self.reqs = [r for r in self.reqs if not any(r is d for d in dropped)]
            self.session.log.append({"role": "assistant", "content": "note: details that identify a pointer's listing are not "
                                     "requirements: dropped " + short([f"{r['what']}: {r['value']}" for r in dropped], 200)})

    def add_ref(self, requirement: dict, what: str, item: str, spec: dict, origin: str = "reply") -> None:
        """Track a requirement whose value is defined by another listing: its value becomes a placeholder marked "to look
        up", which code leaves out of its checks, until that listing (or the one its notes send the property on to) is
        read."""
        relation = str(spec.get("relation") or "").strip().lower() if REF_RELATIONS else ""
        margin = str(spec.get("margin") or "").strip()[:40]
        spec = {k: str(spec.get(k) or "").strip()[:120] for k in ("item", "seller", "properties")} \
            | {"exclude": spec.get("exclude") is True or relation == "not"}
        if not (spec["seller"] or spec["properties"]):
            return
        ref = {"what": what, "item": item, "ref": spec, "req": requirement, "value": None, "seen": {}, "exclude": spec["exclude"],
               "hops": [], "relation": relation if relation in SAME_AS["properties"]["relation"]["enum"] and relation not in ("same", "not") else "",
               "margin": margin, "origin": origin, "first": None, "maybe": []}
        requirement["value"] = self.pointer_text(ref)
        requirement["pending_ref"] = True
        self.refs.append(ref)
        if self.record_text:
            self.resolve_refs(list(self.record_text))      # the listing meant may already have been read

    @staticmethod
    def pointer_text(ref: dict) -> str:
        """The placeholder a pointer's requirement shows until it is traced."""
        spec = ref["ref"]
        where = (f"the listing for {spec['item'] or 'another item'}" + (f" sold by {spec['seller']}" if spec["seller"] else "")
                 + (f" with {spec['properties']}" if spec["properties"] else ""))
        return (f"to look up: anything but its value on {where}" if ref["exclude"] else f"to look up: as on {where}")[:260]

    def note_records(self, observation: Any, ids: list[str], role: str, name: str = "") -> None:
        """Keep each listing's record as text (the fullest seen: a details result over a search row) and the prose in it
        (seller notes first), so that a pointer met later can be checked against listings read before."""
        texts: dict[str, tuple[str, list[str]]] = {}

        def walk(value: Any, depth: int = 0) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                own = next((v for v in value.values() if isinstance(v, str) and v in ids), None)
                if own and own not in texts:
                    # a short note can carry a rule too ("Its oak lid is matched with ash."): a sentence of five or more
                    # words counts here, though it is too short to be taken for a description elsewhere
                    def wordy(v: str) -> bool:
                        return prose(v) or (len(v.split()) >= 5 and bool(re.search(r"[.!?]$", v.strip())))
                    notes = [v for k, v in value.items() if isinstance(v, str) and wordy(v) and re.search(r"note|remark|comment|seller", str(k), re.I)]
                    other = [v for k, v in value.items() if isinstance(v, str) and wordy(v) and v not in notes]
                    # the record without its prose: a note naming a look-alike ("a 142 lbs version exists") is no fact of this one
                    plain = {k: v for k, v in value.items() if not (isinstance(v, str) and prose(v))}
                    texts[own] = (json.dumps(plain, default=str).lower(), notes + other)
                for v in value.values():
                    walk(v, depth + 1)
            elif isinstance(value, list):
                for v in value[:60]:
                    walk(v, depth + 1)
        walk(observation)
        dumped = json.dumps(observation, default=str)
        if len(ids) == 1 and ids[0] not in texts and ids[0] in dumped:
            # one listing asked for and a reply that names it: the reply is its text (a reply that does not name it,
            # an error page say, must not replace what is known of it)
            texts[ids[0]] = (dumped.lower(), [])
        for ident, (text, parts) in texts.items():
            if MERGE_RECORDS:
                # each view's record kept on its own, newest per view, and joined: a seller shown on one view and the
                # properties on another describe the same listing together
                views = self.record_views.setdefault(ident, {})
                key = name or role
                if role == "details" or len(text) >= len(views.get(key, "")):
                    views[key] = text
                self.record_text[ident] = " ".join(views.values())
            elif role == "details" or len(text) > len(self.record_text.get(ident, "")):
                self.record_text[ident] = text
            viewing = role == "details" or bool(name and name in self.listing_views()[0] + self.listing_views()[1])
            if parts and (ident not in self.notes_text or (viewing and ident in self.row_notes)):
                # first notes, or a view's own text replacing what only a search row showed
                self.notes_text[ident] = " ".join(parts)[:1500]
                (self.row_notes.add if role == "search" else self.row_notes.discard)(ident)
            elif parts and viewing:
                # another view of the listing (its own page beside its properties, say): both texts are kept, the
                # newest first, so a seller's note read on one view is not lost when the other is read
                page_text = " ".join(parts)
                if page_text not in self.notes_text.get(ident, ""):
                    self.notes_text[ident] = (page_text + " " + self.notes_text.get(ident, ""))[:1500]
            # model codes and works-with lists live in a listing's text, not its facts: keep them for the set checks
            prose_text = " ".join(parts) + " " + str((self.listings.get(ident) or {}).get("title") or "")
            models = [m.group(1) for m in MODEL_CODE.finditer(prose_text)] + [m.group(1) for m in ALSO_SOLD.finditer(prose_text)]
            works = [c for m in WORKS_WITH.finditer(prose_text) for c in re.findall(CODE, m.group(1))]
            if models or works:
                self.compat[ident] = {"model": list(dict.fromkeys(models))[:4], "works": list(dict.fromkeys(works))[:8]}
            if FIT_CODES:
                # each other text field of the listing's record too, one at a time and never its title: a view met
                # before its fields were learned shows its own text under a key not yet known as prose, and a fit
                # sentence names itself ("fits", "compatible with")
                record = self.listing_record(observation, ident) or {}
                title = str((self.listings.get(ident) or {}).get("title") or "")
                others = [v for k, v in record.items() if isinstance(v, str) and v != title
                          and k != (self.fields.get(name) or {}).get("title")]
                terms, sentence = fit_terms("\n".join([" ".join(parts)] + others))
                if terms and (ident not in self.fits or viewing):
                    self.fits[ident] = (terms, sentence)

    def fit_verdict(self, ident: str, requirement: dict) -> str | None:
        """FIT_CODES: a requirement whose value names a model code (a maker word and three digits or more: "Halvern 4120")
        read against what the listing's own text says it fits: "yes" when a code, family, series or range there holds
        it; "no" when the text names codes of that maker and none holds it; None when the text says nothing of it (or
        the requirement rules a value out), so the usual checks decide."""
        value = f"{requirement.get('what') or ''} {requirement.get('value') or ''}"
        found = re.search(r"\b([A-Za-z][A-Za-z-]{1,20})[ -]+(\d{3,6})(?!\w|\.\d)", str(requirement.get("value") or ""))
        if not found or NEGATION.search(value):
            return None
        fitting = self.fit_requirement(requirement)
        terms = list((self.fits.get(ident) or ([], ""))[0])
        for key, fact in (self.facts.get(ident) or {}).items():
            if FIT_KEY.search(str(key)):
                terms += fit_terms(f"fits {fact}")[0]   # a fact naming what the listing fits ("compatible models")
        if not terms:
            if not fitting:
                return None
            # what an item fits is stated by the listing itself: a title or a stray word never confirms it
            shown = "page" in self.cover.get(ident, set())
            return "unconfirmed (its text names no model it fits)" if shown else \
                "unconfirmed (what it fits is in its own text, not read yet)"
        maker = found.group(1).lower()
        maker = "" if maker in FIT_SKIP else maker
        code = found.group(2)
        hit = next((t for t in terms if (not maker or not t["maker"] or t["maker"] == maker) and fit_holds(t, code)), None)
        if hit:
            return f"yes (its text fits {hit['text'][:40]})"
        same = [t["text"] for t in terms if maker and t["maker"] == maker]
        if same:
            return f"no (its text fits only {', '.join(same[:3])[:80]})"
        return "unconfirmed (its text names other makers' models)" if fitting else None

    @staticmethod
    def fit_requirement(requirement: dict) -> bool:
        """A requirement about what the item must fit or work with (its name says so) whose value names a model code."""
        return bool(FIT_KEY.search(str(requirement.get("what") or ""))) and bool(
            re.search(r"\b[A-Za-z][A-Za-z-]{1,20}[ -]+\d{3,6}(?!\w|\.\d)", str(requirement.get("value") or "")))

    def fit_read(self) -> bool:
        """FIT_CODES: several items, a line that must fit a model the request names: what its listings fit is stated in
        their own text, which a property view leaves out; code reads the likeliest of them through a view that shows a
        listing's own text (several a call when the store has such a view), while the line has no confirmed fit.
        Returns True when a read was made."""
        if not (FIT_CODES and self.items) or self.fit_reads >= 2 * len(self.items) or self.session.finished or self.session.broken:
            return False
        if self.steps_left() <= self.pacing() + 2 or monotonic() - self.began > WRAP_AT - 40:
            return False
        many = self.list_view()
        tool = many if many and (self.view_page.get(many) or many not in self.view_shown) else self.page_reader()
        if not tool:
            return False
        for entry in self.items:
            reqs = [r for r in self.item_reqs(entry["item"]) if self.fit_requirement(r)]
            if not reqs or self.line_fit(entry):
                continue
            pool = []
            for ident, info in self.listings.items():
                if self.item_of(ident) != entry["item"] or ident in self.fits or "page" in self.cover.get(ident, set()) \
                        or sold_out(info.get("stock")) or ident in self.ref_ids or ident in self.fit_tried:
                    continue
                others = [r for r in self.item_reqs(entry["item"]) if r not in reqs]
                broken = ident in self.facts and bool(others) and any(
                    c.split(": ", 1)[-1].startswith(("no", "breaks")) for c in self.marks(ident, others)[0])
                price = info.get("price") if isinstance(info.get("price"), (int, float)) else 0
                pool.append((broken, ident not in self.facts, price if self.cheapest_wanted() else 0, -(info.get("step") or 0), ident))
            if not pool:
                continue
            ids = [i for *_, i in sorted(pool)]
            ids = ids[:max(1, self.view_cap(tool))] if tool == self.list_view() else ids[:1]
            self.fit_tried |= set(ids)
            text = self.read_ids(ids, f"what they fit ({entry['item']})", inject=True, count=False, tool=tool)
            if text:
                self.fit_reads += 1
                return True
        return False

    @staticmethod
    def code_key(code: str) -> str:
        """A model code compared without case, spaces or hyphens ("AB-1234" and "ab 1234" are one code)."""
        return re.sub(r"[^a-z0-9]", "", code.lower())

    def compat_text(self, ident: str) -> str:
        """What a listing's text says about its model and what it works with, for the views the models read."""
        entry = self.compat.get(ident) or {}
        parts = ([f"model {' / '.join(entry['model'])}"] if entry.get("model") else []) \
            + ([f"works with {', '.join(entry['works'])}"] if entry.get("works") else []) \
            + ([f'"{self.fits[ident][1]}"'] if FIT_CODES and ident in self.fits else [])
        return f" | its text: {'; '.join(parts)}" if parts else ""

    def set_problems(self, lines: list[dict]) -> list[str]:
        """Basket lines that do not fit together: when a line's listing names its model and others say what they work
        with, each of those must name that model (or a name the main listing gives for it)."""
        mains = {l["id"]: self.compat[l["id"]]["model"] for l in lines
                 if (self.compat.get(l["id"]) or {}).get("model") and not self.compat[l["id"]].get("works")}
        if not mains:
            return []
        keys = {self.code_key(c) for codes in mains.values() for c in codes}
        named = ", ".join(f"{i} (model {' / '.join(c)})" for i, c in mains.items())
        return [f"{l['id']} says it works with {', '.join(self.compat[l['id']]['works'])}, not with {named}: choose a listing for "
                "that item whose text names the main item's model, so the set works together"
                for l in lines if (self.compat.get(l["id"]) or {}).get("works")
                and not {self.code_key(c) for c in self.compat[l["id"]]["works"]} & keys]

    @staticmethod
    def figures(text: Any) -> list[tuple[float, str]]:
        """The numbers of a text with the unit written after each ("12gb" and "12 GB" alike; thousands commas
        dropped), the unit's plural ending cut."""
        plain = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(text).lower())
        out = []
        for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]*)", plain):
            if re.search(r"(?:ss|x|z|ch|sh)es$", unit):
                unit = unit[:-2]            # "inches" and "inch", "boxes" and "box"
            elif unit.endswith("s") and not unit.endswith("ss") and len(unit) > 2:
                unit = unit[:-1]            # "pieces" and "piece", "litres" and "litre"
            out.append((float(number), unit))
        return out

    def value_state(self, want: str, have: str) -> str:
        """How a listing's stated value compares with a described one: "holds" (every number of it, in the same unit,
        and all its words), "differs" (another number in the same unit, or words missing), "units" (its numbers are
        written in other units only: code does not convert, a model reads it)."""
        wanted, shown = self.figures(want), self.figures(have)
        if wanted:
            for number, unit in wanted:
                if any(n == number and (u == unit or not u or not unit) for n, u in shown):
                    continue
                if any(u == unit and unit for _, u in shown) or (shown and not unit):
                    return "differs"
                return "units" if shown else "differs"
        words = tokens(re.sub(r"\d+(?:\.\d+)?\s*[a-z]{0,2}\b", " ", str(want))) - FILLER
        return "holds" if words <= tokens(have) else "differs"

    def fit_state(self, spec: dict, ident: str) -> tuple[str, int]:
        """STRICT_REFS: whether a listing is the one a pointer describes, from its own fields across views (never its
        notes): ("yes", values shown) when the seller and every described value hold; ("no", 0) when the seller is
        missing or a value it states differs; ("maybe", values shown) when some value is not shown at all or only in
        other units (read its properties, or let a model judge)."""
        text = self.record_text.get(ident, "")
        sheet = self.facts.get(ident) or {}
        seller = spec["seller"].lower()
        if not text or (seller and seller not in text):
            return "no", 0
        words = tokens(text)
        item_words = tokens(spec["item"]) - FILLER
        if not seller and item_words and not item_words & (words | tokens((self.listings.get(ident) or {}).get("category"))):
            return "no", 0
        info = self.listings.get(ident) or {}
        if ASK_BEFORE_WALK and seller and item_words and (info.get("title") or info.get("category")) \
                and not item_words & (tokens(info.get("title")) | tokens(info.get("category"))):
            return "no", 0                  # the seller's listing of another kind of item
        names = set().union(*[tokens(k) for k in sheet] or [set()])
        pieces = [p for p in re.split(r";" if ";" in spec["properties"] else r",|\band\b", spec["properties"]) if p.strip()]
        shown = open_ = 0
        for piece in pieces:
            name, value = piece.split(":", 1) if ":" in piece else ("", piece)
            named = tokens(name) - FILLER
            stated = [str(v) for k, v in sheet.items() if named and tokens(k) & named]
            if stated:
                states = {self.value_state(value, v) for v in stated}
                if "holds" in states:
                    shown += 1
                elif "units" in states:
                    open_ += 1
                else:
                    return "no", 0          # the listing states this property with another value
                continue
            # no property of that name in its list: the value as its record writes it anywhere
            state = self.value_state(value, text)
            want = tokens(value) - FILLER - (set() if ":" in piece else names)
            if state == "holds" and (want or self.figures(value)):
                shown += 1
            else:
                open_ += 1
        return ("yes" if not open_ else "maybe"), shown

    def pick_reference(self, ref: dict, ids: list[str]) -> str:
        """Among listings that each hold every value a pointer describes, the one it means: a light model call reads
        their titles, properties and notes (a note counts only where it describes that listing itself). "" when no
        model answered or it named none of them."""
        if monotonic() - self.began > WRAP_AT - 40 or self.mind.spent >= SPEND_HARD + self.mind.room - 0.004:
            return ""
        spec = ref["ref"]
        cards = "\n\n".join(f"LISTING {i}: {(self.listings.get(i) or {}).get('title') or ''}\n"
                             f"PROPERTIES: {short(self.facts.get(i) or {}, 500)}\nNOTES: {self.notes_text.get(i, '')[:400] or '(none)'}"
                             for i in ids[:4])
        text = (f"DESCRIBED: the {spec['item'] or 'listing'}" + (f" sold by {spec['seller']}" if spec["seller"] else "")
                + (f" with {spec['properties']}" if spec["properties"] else "") + f"\n\n{cards}\n\nCall pick_listing.")
        reply = self.mind.ask([{"role": "system", "content": PICK_GUIDE}, {"role": "user", "content": text}],
                              [PICK_TOOL], max_tokens=900, models=POINTER_MODELS)
        calls = (reply or {}).get("tool_calls") or []
        found = parse_json((calls[0].get("function") or {}).get("arguments")) if calls else None
        pick = str((found or {}).get("listing") or "").strip() if isinstance(found, dict) else ""
        return pick if pick in ids else ""

    def fits_description(self, spec: dict, ident: str) -> bool:
        """True when a read listing is the one a pointer describes: the seller or maker named in its record (or, with no
        seller named, the item's words there or in its category), every number the description gives shown in it, and
        each described value there. A fact giving a described property another value rules the listing out; at most one
        value the record does not state at all is let pass, when three or more are given."""
        text = self.record_text.get(ident, "")
        sheet = self.facts.get(ident) or {}
        seller = spec["seller"].lower()
        if not text or (seller and seller not in text):
            return False
        words = tokens(text)
        item_words = tokens(spec["item"]) - FILLER
        if not seller and item_words and not item_words & (words | tokens((self.listings.get(ident) or {}).get("category"))):
            return False
        figures = re.findall(r"\d+(?:\.\d+)?", text)
        names = set().union(*[tokens(k) for k in sheet] or [set()])
        pieces = [p for p in re.split(r";" if ";" in spec["properties"] else r",|\band\b", spec["properties"]) if p.strip()]
        matched = unknown = 0
        for piece in pieces:
            name, value = piece.split(":", 1) if ":" in piece else ("", piece)
            numbers = re.findall(r"\d+(?:\.\d+)?", value)
            if any(n not in figures for n in numbers):
                return False                # a look-alike that differs in a number is another listing
            want = tokens(value) - FILLER - (set() if ":" in piece else names)
            # a named property needs a word of its value; a bare short value ("a graphic print") needs all of it
            need = 1 if ":" in piece else len(want) if len(want) <= 3 else max(1, round(0.6 * len(want)))
            if not want or len(want & words) >= need:
                matched += 1 if (want or numbers) else 0
                continue
            named = tokens(name) - FILLER
            if named and any(tokens(k) & named and not tokens(v) & want for k, v in sheet.items()):
                return False                # the listing states this property with another value
            unknown += 1                    # the record does not mention it (a pointer may name it in its own words)
        # with values given, one must show and the unknown ones stay a minority
        return not pieces or (matched >= 1 and unknown <= max(1, len(pieces) // 2))

    def read_pointer(self, what: str, notes: str, own: str = "") -> dict | None:
        """What a listing's notes say about one property: "spec", another listing's description when they send the
        property on to it (else None); "own_value", the listing's own value as the notes state it; "paired_value", the
        value a rule in the notes pairs that own value with. None when no model answered."""
        if monotonic() - self.began > WRAP_AT - 40 or self.mind.spent >= SPEND_HARD + self.mind.room - 0.004:
            return None
        text = f"PROPERTY: {what}\n" + (f"THIS LISTING'S OWN VALUE OF IT: {own}\n" if own else "") + f"\nNOTES:\n{notes[:1500]}\n\nCall report_pointer."
        reply = self.mind.ask([{"role": "system", "content": POINTER_GUIDE}, {"role": "user", "content": text}],
                              [POINTER_TOOL], max_tokens=900, models=POINTER_MODELS)
        calls = (reply or {}).get("tool_calls") or []
        found = parse_json((calls[0].get("function") or {}).get("arguments")) if calls else None
        if not isinstance(found, dict):
            return None
        out: dict = {"spec": None, "own_value": str(found.get("own_value") or "").strip()[:120],
                     "paired_value": str(found.get("paired_value") or "").strip()[:120]}
        if found.get("points_on") is not True:
            return out
        spec = {k: str(found.get(k) or "").strip()[:120] for k in ("item", "seller", "properties")}
        # values written into the item ("teapots with a striped glaze") belong with the properties; a filler such as
        # "none given" is no value
        item, _, extra = spec["item"].partition(" with ")
        pieces = [p.strip() for p in re.split(r";", spec["properties"]) + ([extra] if extra else []) if p.strip()]
        pieces = [p for p in pieces if not re.search(r"\b(?:none|not given|not stated|no values?|unknown|unspecified|n/a)\b", p, re.I)]
        spec["item"], spec["properties"] = item.strip(), "; ".join(pieces)
        out["spec"] = spec | {"exclude": False} if spec["seller"] or spec["properties"] else None
        return out

    @staticmethod
    def pairing_holds(paired: str, own: str, notes: str, what: str = "") -> bool:
        """True when a partner value the reader gave is written in the notes, differs from the listing's own value, and
        sits in one clause with that value or with the property's name ("its colour is matched with ..."): a rule pairing
        the two, not a value from some other sentence."""
        def bits(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", text.lower())) - FILLER - {"a", "an", "of", "to", "in", "on", "is", "it"}
        want, mine, named = bits(paired), bits(own), tokens(what) - FILLER
        if not want or want == mine:
            return False
        for clause in re.split(r"[.;:!?](?:\s|$)|\n", notes.lower()):
            words = bits(clause)
            if want <= words and (not mine or mine & words or named & tokens(clause)):
                return True
        return False

    def resolve_refs(self, ids: list[str]) -> None:
        """A need whose value is a property of another listing (or anything but it): once a read listing fits that
        listing's description (fits_description), its notes are read for a pointer that sends the property on to yet
        another listing, which is then looked for the same way, among the listings read before too; otherwise its own
        value for the property becomes the requirement (or the value to avoid). When the fitting listing shows no such
        property, a listing it refers to (by id or title) is used instead once that one is read."""
        for ref in [r for r in self.refs if r["value"] is None]:
            candidates = list(ids)
            for _ in range(6):              # each pointer followed looks again among every listing read so far
                if self.trace_ref(ref, candidates) != "moved":
                    break
                candidates = list(self.record_text)

    def ref_order(self, ref: dict, ids: list[str]) -> list[str]:
        """STRICT_REFS: the listings among ids that a pointer means, best first: one its last listing named, then those
        holding every described value (a light model call picks among several), the most values shown first. Those
        that may hold them but do not show each yet are kept in ref["maybe"] for code to read."""
        named, sure, maybe = [], [], []
        for ident in ids:
            if ident in ref["hops"]:
                continue
            title = str((self.listings.get(ident) or {}).get("title") or "")
            if any(ident in seen or (len(tokens(title)) >= 3 and len(tokens(title) & tokens(seen)) >= 0.8 * len(tokens(title)))
                   for seen in ref["seen"].values()):
                named.append(ident)
                continue
            state, shown = self.fit_state(ref["ref"], ident)
            if state == "yes":
                sure.append((shown, ident))
            elif state == "maybe" and ident not in maybe:
                maybe.append(ident)
        sure = [i for _, i in sorted(sure, key=lambda p: -p[0])]
        if len(sure) > 1:
            pick = self.pick_reference(ref, sure)
            if pick:
                sure = [pick] + [i for i in sure if i != pick]
        read = [i for i in maybe if "props" in self.cover.get(i, set())]
        asked = ref.setdefault("asked", [])
        if not named and not sure and read and sorted(read) not in asked:
            # every candidate's properties were read and none shows each value as written (a number in another unit,
            # say): the light model judges them once, converting units by the usual rules
            asked.append(sorted(read))
            pick = self.pick_reference(ref, read)
            if pick:
                sure = [pick]
        ref["maybe"] = [i for i in dict.fromkeys(ref.get("maybe", []) + maybe) if i not in sure and i not in named]
        return named + sure

    def related_value(self, ref: dict, value: str, paired: bool) -> str:
        """REF_RELATIONS: the requirement a pointer's value gives under its relation ("at most 30 cm", "between 28 and 32
        cm"); the value itself for "the same" (or after a pairing gave the partner value)."""
        relation = ref.get("relation") or ""
        if not REF_RELATIONS or not relation or paired:
            return value
        if relation in ("at_most", "lowest"):
            return f"at most {value}"
        if relation in ("at_least", "highest"):
            return f"at least {value}"
        if relation == "within":
            margin = ref.get("margin") or ""
            own, by = self.figures(value), self.figures(margin)
            if len(own) == 1 and len(by) == 1 and by[0][1] in ("", own[0][1]):
                low, high = own[0][0] - by[0][0], own[0][0] + by[0][0]
                return f"between {low:g} and {high:g} {own[0][1]}".strip()
            return f"{value}, give or take {margin}" if margin else value
        if relation == "pairs_with":
            return f"the one that goes with {value}"
        return value

    def trace_ref(self, ref: dict, ids: list[str]) -> str:
        """One pass of resolve_refs over some listings: "resolved", "moved" (a pointer named the next listing) or ""."""
        prop = tokens(ref["what"]) - FILLER
        # the words naming the item, not the property: "teapot style" is read from a style fact, never a teapot-height one
        item_words = tokens(ref["item"] or "") | tokens(self.brief.get("goal")) | set().union(*[tokens(e["item"]) for e in self.items] or [set()])
        label = f"{ref['item'] + ' > ' if ref['item'] else ''}{ref['what']}"
        if STRICT_REFS:
            ids = self.ref_order(ref, ids)
        for ident in ids:
            if ident in ref["hops"]:
                continue
            text = self.record_text.get(ident, "")
            sheet = self.facts.get(ident) or {}
            title = str((self.listings.get(ident) or {}).get("title") or "")
            named = any(ident in seen or (len(tokens(title)) >= 3 and len(tokens(title) & tokens(seen)) >= 0.8 * len(tokens(title)))
                        for seen in ref["seen"].values())
            if not named and not STRICT_REFS and not self.fits_description(ref["ref"], ident):
                continue
            # a listing a pointer describes, or one it passes through, defines a value (REF_CANDIDATE: only the first
            # one a request names, or one whose note sends the property on, is never the item to buy)
            self.ref_ids.add(ident)
            if not ref["hops"] and ref.get("first") is None:
                ref["first"] = ident
            notes = self.notes_text.get(ident, "")
            core = (prop - item_words - tokens((self.listings.get(ident) or {}).get("category"))) or prop
            key = max(sheet, key=lambda k: (len(tokens(k) & core), len(tokens(k) & prop)), default=None) if prop else None
            own = str(sheet[key])[:120] if key is not None and tokens(key) & core else ""
            found = None
            if notes and prop and (tokens(notes) & prop or (own and tokens(own) & tokens(notes))):
                # only the sentences about this property or this listing's value: other text ("sold by ...") can lead
                # the reader to the wrong seller
                about = [x for x in re.split(r"(?<=[.!?])\s+", notes) if tokens(x) & prop or (own and tokens(own) & tokens(x))]
                found = self.read_pointer(ref["what"], " ".join(about) or notes, own)
                if found and found["spec"] is not None:
                    ref["hops"].append(ident)
                    ref["ref"], ref["seen"] = found["spec"] | {"exclude": ref["exclude"]}, {}
                    ref["req"]["value"] = self.pointer_text(ref)
                    self.session.log.append({"role": "assistant", "content": f"note: {label} points on from {ident} to "
                                             + short(found["spec"], 240)})
                    return "moved"
            if CHAIN_PROOF and self.proof_can_wait(ident):
                # a view of it was never read, and its own page may send the property on: the value waits for it
                self.unproven.setdefault(ident, 0)
                continue
            stated = (found or {}).get("own_value") or ""
            if not own and stated and tokens(stated) and tokens(stated) <= tokens(notes):
                own = stated                # the notes state this listing's value where no fact gives it
            if not own:
                if not named and text:
                    # it fits the description but shows no such property: a listing its notes refer to (by id or title) may
                    ref["seen"][ident] = (text + " " + notes.lower()).strip()
                continue
            value = own
            paired = (found or {}).get("paired_value") or ""
            if paired and not ref["exclude"] and self.pairing_holds(paired, own, notes, ref["what"]):
                # a rule of the store's pairs this listing's value with another: the item takes the partner value
                value = paired
                self.session.log.append({"role": "assistant", "content": f"note: {label}: {ident} shows {own}, which its notes pair "
                                                                         f"with {paired}"})
            ref["value"] = value
            # to avoid a value that lists several, avoid each of them (the checks read clauses split at commas)
            ref["req"]["value"] = "; ".join(f"not {v.strip()}" for v in ref["value"].split(",") if v.strip()) \
                if ref["exclude"] else self.related_value(ref, value, value != own)
            ref["req"].pop("pending_ref", None)
            ref["source"] = {"id": ident, "own": own, "paired": value != own}
            self.told.append((self.session.steps, f"{label}: {ref['req']['value']}"))
            self.session.log.append({"role": "assistant", "content": f"note: need resolved from {ident}: {label} = {ref['req']['value']}"})
            return "resolved"
        return ""

    def asked_about(self, text: str) -> set[str]:
        """The words of a question that name what it asks about: question words, offered examples and the items' own
        names left out."""
        items = set().union(*[tokens(e["item"]) for e in self.items]) if self.items else tokens(self.brief.get("goal"))
        text = re.sub(r"\((?:for example|for instance|e\.g\.|such as|like)\b[^)]*\)", " ", str(text), flags=re.I)
        return tokens(text) - ASKING - items

    def ids_in(self, value: Any) -> list[str]:
        """Known listing ids among a call's argument values, inside lists and objects too."""
        if isinstance(value, str):
            return [value] if value in self.listings else []
        if isinstance(value, dict):
            return [i for v in value.values() for i in self.ids_in(v)]
        if isinstance(value, list):
            return [i for v in value[:20] for i in self.ids_in(v)]
        return []

    def item_named(self, text: str) -> str:
        """The order's item a text names (the most shared words), or "" when it names none."""
        words = tokens(text)
        best = max(self.items, key=lambda e: len(tokens(e["item"]) & words), default=None)
        return best["item"] if best and tokens(best["item"]) & words else ""

    def ask_first(self, left: int) -> bool:
        """True when a cart change or finish should wait for the questions still allowed: questions cost no score and a
        held-back need is told only when asked. Sent back at most once per allowed question, and only while the
        remaining steps cover the questions and the finish."""
        budget = self.question_budget or 0
        room = self.ask_room()
        if room <= 0 or self.ask_stops >= budget or left <= self.reserve() + 4 + room:
            return False
        if not any(r == "message" for r in self.roles.values()):
            return False
        self.ask_stops += 1
        return True

    def plain_remark(self, args: dict) -> dict:
        """A decline note with no listing ids in it: naming a listing reads as offering a substitute."""
        known = sorted((i for i in set(self.listings) | set(self.facts) if len(i) >= 4), key=len, reverse=True)
        out = {}
        for key, value in args.items():
            if isinstance(value, str) and known and any(i in value for i in known):
                for ident in known:
                    value = re.sub(rf"(?<![A-Za-z0-9]){re.escape(ident)}(?![A-Za-z0-9])", "", value)
                value = re.sub(r"\(\s*(?:(?:and|or|,)\s*)*\)", "", value)
                value = re.sub(r"\s+([,.;:!?)])", r"\1", re.sub(r"\s{2,}", " ", value)).strip(" ,;:")
                value = value if re.search(r"[A-Za-z]{3,}", value) else "Nothing available meets the request."
            out[key] = value
        return out

    def cart_lines(self) -> tuple[list[dict], float | None]:
        """The cart as last viewed: its lines (id, quantity) and its total, read by shape."""
        lines: list[dict] = []
        total = None
        view = next((n for n, r in self.roles.items() if r == "cart_view" and n in self.fields), None)
        fields = self.fields.get(view or "", {})

        def count_key(key: str) -> bool:
            if fields.get("quantity"):
                return key == fields["quantity"]
            key = key.lower()
            return bool(re.search(r"qty|quantit|how_?many|count|units", key)) and not re.search(r"discount|account", key)

        def number(v: Any) -> bool:
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        def walk(value: Any, depth: int = 0) -> None:
            nonlocal total
            if depth > 5:
                return
            if isinstance(value, dict):
                ident = value.get(fields["id"]) if fields.get("id") in value else \
                    next((v for k, v in value.items() if id_key(str(k)) and isinstance(v, (str, int)) and not isinstance(v, bool)), None)
                if isinstance(ident, (str, int)) and not isinstance(ident, bool) and depth > 0:
                    quantity = next((v for k, v in value.items() if count_key(str(k)) and number(v)), None)
                    lines.append({"id": str(ident), "quantity": quantity})
                    return
                if total is None and fields.get("cart_total") in value and number(value[fields["cart_total"]]):
                    total = float(value[fields["cart_total"]])
                for key, item in value.items():
                    if total is None and not fields.get("cart_total") and re.search(r"total", str(key), re.I) \
                            and not count_key(str(key)) and number(item):
                        total = float(item)
                    walk(item, depth + 1)
            elif isinstance(value, list):
                for item in value[:50]:
                    walk(item, depth + 1)
        walk(self.cart_obs)
        return lines, total

    def removal_hint(self) -> str:
        """How this store takes a line out of the cart: its remove tool, or a count of 0 on a tool that sets counts."""
        for name, role in self.roles.items():
            if role != "cart_change":
                continue
            spec = self.session.specs.get(name) or {}
            if re.search(r"remov|drop|delet|take (?:out|off)|discard|clear", f"{name} {spec.get('description') or ''}".replace("_", " "), re.I) \
                    and not re.search(r"\b(?:add|put|set|keep|hold|place)\b", name.replace("_", " "), re.I):
                return f"with {name}"
        for name, role in self.roles.items():
            properties = ((self.session.specs.get(name) or {}).get("parameters") or {}).get("properties") or {}
            counts = [k for k, v in properties.items() if isinstance(v, dict) and set(kinds_of(v)) & {"integer", "number"}]
            if role == "cart_change" and counts:
                return f"set its {counts[0]} to 0 with {name} if the tool allows 0, else remove it the way the tool says"
        return "the way the cart tool says"

    def listing_views(self) -> tuple[list[str], list[str]]:
        """The store's views of listings: tools that read one listing named by its id, and tools that read several in
        one call (a list of ids). A view has exactly one required argument, and its name or description says it takes
        a listing; search, cart, message and finishing tools are never views."""
        single: list[str] = []
        many: list[str] = []
        for name, spec in self.session.specs.items():
            if self.roles.get(name) in ("search", "cart_change", "cart_view", "message", "order", "decline"):
                continue
            schema = (spec or {}).get("parameters") or {}
            properties = schema.get("properties") or {}
            required = [k for k in schema.get("required") or [] if isinstance(properties.get(k), dict)]
            if len(required) != 1:
                continue
            key, field = required[0], properties[required[0]]
            words = set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", key).lower()))
            if field.get("enum") or not (words & LISTING_WORDS or re.search(r"\b(?:ids?|codes?|refs?)\b", str(field.get("description") or ""), re.I)):
                continue
            kinds = kinds_of(field)
            if "array" in kinds and isinstance(field.get("items"), dict) and "string" in kinds_of(field["items"]):
                many.append(name)
            elif "string" in kinds:
                single.append(name)
        return single, many

    def view_args(self, tool: str, ident: str | list[str]) -> dict | None:
        """Arguments that read one listing (or a list of them) with a view: its one required argument set to the id."""
        schema = (self.session.specs.get(tool) or {}).get("parameters") or {}
        properties = schema.get("properties") or {}
        required = [k for k in schema.get("required") or [] if isinstance(properties.get(k), dict)]
        if len(required) != 1:
            return None
        value: Any = ident
        if "array" in kinds_of(properties[required[0]]) and not isinstance(value, list):
            value = [value]
        args, problems = fit_args(schema, {required[0]: value})
        return None if problems else args

    def note_view(self, name: str, args: dict, observation: Any, classify: bool = True) -> None:
        """Books which listings a view showed: the ids the call named that its result writes back."""
        single, many = self.listing_views()
        if name not in single and name not in many:
            return
        if isinstance(observation, dict) and ("error" in observation or "refusal" in observation):
            return
        text = json.dumps(observation, default=str)
        shown = [ident for ident in self.ids_in(args) if ident in text]
        for ident in shown:
            self.viewed.setdefault(ident, set()).add(name)
        if COVER_BY_CONTENT and shown and classify:
            self.book_cover(name, observation, shown, name in many)
        if name in single and shown and classify:
            sheets: list[tuple[str, dict]] = []
            facts_in(observation, sheets, fields=self.fields.get(name))
            size = max((len(sheet) for _, sheet in sheets), default=0)
            flat = own_fields(self.unwrapped(observation), fields=self.fields.get(name)) if not size else None
            size = max(size, len(flat[1]) if flat else 0)   # a listing shown as one flat record of its properties
            (self.fact_views if size >= 2 else self.bare_views).add(name)
            self.sheet_size[name] = max(self.sheet_size.get(name, 0), size)

    @staticmethod
    def listing_record(value: Any, ident: str, depth: int = 0) -> dict | None:
        """The record of a result that names a listing by its id (the innermost one), or None."""
        if depth > 6:
            return None
        if isinstance(value, dict):
            for v in value.values():
                inner = Agent.listing_record(v, ident, depth + 1)
                if inner is not None:
                    return inner
            return value if any(v == ident for v in value.values() if isinstance(v, str)) else None
        if isinstance(value, list):
            for v in value[:60]:
                inner = Agent.listing_record(v, ident, depth + 1)
                if inner is not None:
                    return inner
        return None

    def book_cover(self, name: str, observation: Any, shown: list[str], several: bool) -> None:
        """What a view returned of each listing it showed: its property list ("props", two or more properties) and its
        own page ("page": a price, a stock word, a seller or a note beside it, even an empty note)."""
        fields = self.fields.get(name) or {}
        sheets: list[tuple[str, dict]] = []
        facts_in(observation, sheets, fields=fields)
        with_props = {ident for ident, sheet in sheets if sheet}
        page_keys = {fields.get(r) for r in ("price", "stock") if fields.get(r)} | set(fields.get("prose") or [])
        paged = stocked = 0
        for ident in shown:
            record = self.listing_record(observation, ident) or {}
            if ident not in with_props:
                own = own_fields(record, fields=fields)
                if own and len(own[1]) >= 2 and own[0] == ident:
                    with_props.add(ident)
            got = self.cover.setdefault(ident, set())
            if ident in with_props:
                got.add("props")
            if any(str(k) in page_keys or NOTE_KEY.search(str(k)) or re.search(r"price|cost|stock|avail|seller|merchant|vendor", str(k), re.I)
                   for k in record):
                got.add("page")
                paged += 1
            if any((fields.get("stock") and str(k) == fields.get("stock")) or re.search(r"stock|avail|orderab|in_store|on_hand|supply", str(k), re.I)
                   for k in record):
                got.add("stock")            # the page that tells whether it can be bought
                stocked += 1
        self.view_page[name] = self.view_page.get(name, 0) + paged
        self.view_stock[name] = self.view_stock.get(name, 0) + stocked
        if several:
            self.view_shown[name] = self.view_shown.get(name, 0) + len(shown)
            self.view_props[name] = self.view_props.get(name, 0) + len(set(shown) & with_props)

    def list_bare(self, name: str | None) -> bool:
        """READ_BY_CONTENT: a several-listing view seen to show two or more listings, none of them with a property list."""
        return bool(READ_BY_CONTENT and name and self.view_shown.get(name, 0) >= 2 and not self.view_props.get(name, 0))

    def page_reader(self) -> str | None:
        """The view that shows a listing's own page: a several-listing view seen to show pages (two listings a step),
        else a single-listing view seen to show one, else the page view the store splits from the properties."""
        single, many = self.listing_views()
        props = self.props_view()
        if any(self.view_stock.values()):
            # the store shows whether a listing can be bought on some view: the page read is one that shows it
            return next((t for t in many if self.view_stock.get(t)), None) \
                or next((t for t in single if self.view_stock.get(t) and t != props), None) \
                or next((t for t in single if self.view_stock.get(t)), None)
        return next((t for t in many if self.view_page.get(t)), None) \
            or next((t for t in single if self.view_page.get(t) and t != props), None) \
            or next((t for t in single if self.view_page.get(t)), None) or (self.page_views() or [None])[0]

    def views_missing(self, lines: list[str], explore: bool = True, single_page: bool = False) -> dict[str, list[str]]:
        """For each listing, the views to read it with before an order. COVER_BY_CONTENT, once views were seen: its
        property list and its own page, each through a view that returns it, unless a view already returned it; with
        single_page (PAGE_SINGLE), its own page only through a single-listing view seen to show pages. Otherwise the
        store's single-listing views that never showed it, where a list view that showed it counts for every view."""
        single, many = self.listing_views()
        if COVER_BY_CONTENT and self.cover:
            props = self.props_view() if self.props_view() in self.fact_views or any(self.view_props.values()) else None
            props = next((t for t in many if self.view_props.get(t)), None) if props is None else props
            page = self.page_reader()
            lone = (next((t for t in single if self.view_page.get(t) and t != props), None)
                    or next((t for t in single if self.view_page.get(t)), None)) if single_page else None
            # a single-listing view never seen to return anything may show what the others leave out: read it too
            # a single-listing view never seen to return anything may show what the others leave out (whether the listing
            # can be bought, say): each cart line reads it until the store has shown what it holds
            unknown = [t for t in single if t not in self.view_page and t not in self.fact_views and t not in self.bare_views] \
                if explore or props is None or page is None else []
            stock_seen = any(self.view_stock.values())
            fresh = [t for t in many if t not in self.view_shown]
            gaps: dict[str, list[str]] = {}
            for ident in lines:
                got = self.cover.get(ident, set())
                paged = "page" in got and ("stock" in got or not stock_seen)
                reader = page
                if lone and not any(t in single and self.view_page.get(t) for t in self.viewed.get(ident, set())):
                    reader, paged = lone, False     # shown only by a several-listing view so far
                known = ([props] if props and "props" not in got else []) + ([reader] if reader and not paged else [])
                rest = [t for t in unknown if t not in self.viewed.get(ident, set())]
                if fresh and (rest or (not page and "page" not in got) or (not props and "props" not in got)):
                    rest = [fresh[0]]       # a several-listing view not tried yet may show the rest, several lines a call
                gap = known + rest
                if gap:
                    gaps[ident] = list(dict.fromkeys(gap))
            return gaps
        missing: dict[str, list[str]] = {}
        for ident in lines:
            seen = self.viewed.get(ident, set())
            if seen & set(many):
                continue
            gap = [t for t in single if t not in seen]
            if gap:
                missing[ident] = gap
        return missing

    def read_pages(self) -> str:
        """Before an order: every cart line is read once through each view the store gives of a listing. A listing's
        own page can hold its seller's notes, its stock or a price the other views leave out, and a note can say it is
        not the thing asked for; orders placed without reading the ordered listing's own page fail far more often.
        Code reads the missing views itself (one list call for several lines when the store has a list view), then
        sends the order back once per cart with what they showed, and with any note that sends a required property of
        a line on to another listing (NOTE_SENDS). Left out when steps or time are short."""
        lines = [i for i in self.cart_book if i in self.listings]
        key = "pages:" + ",".join(sorted(lines))
        if not lines or key in self.gaps_raised:
            return ""
        before = {i: ((self.listings.get(i) or {}).get("price"), sold_out((self.listings.get(i) or {}).get("stock"))) for i in lines}
        made = self.open_line_views(lines)
        if made is None:
            return ""
        self.gaps_raised.add(key)
        calls, heard = made
        after = {i: ((self.listings.get(i) or {}).get("price"), sold_out((self.listings.get(i) or {}).get("stock"))) for i in lines}
        # only hard news goes back to the model: a line now sold out, a price that moved, a notice, a note that sends a
        # required property elsewhere. The pages' other prose (descriptions) is never shown here: a long description at
        # the moment of ordering only invites second guesses about a pick its facts already support
        changes = [f"{i} is now shown sold out" for i in lines if after[i][1] and not before[i][1]]
        changes += [f"{i} now costs {after[i][0]} (was {before[i][0]})" for i in lines
                    if after[i][0] != before[i][0] and after[i][0] is not None and before[i][0] is not None]
        changes += [f"news: {line[:200]}" for line in heard]
        sends = self.note_sends(lines) if not self.session.finished and not self.session.broken else []
        if calls or sends:
            self.session.log.append({"role": "assistant", "content": "note: cart lines read through every view before the order: "
                                     + short([tool for tool, _ in calls], 200) + (f" ({short(changes + sends, 300)})" if changes or sends
                                                                                 else " (nothing new: the order goes on)")})
        if not (changes or sends) or self.session.finished or self.session.broken:
            return ""                       # nothing that changes the cart: the order goes on without a turn
        then = "follow the change (take a sold-out line out and put in the next fitting listing; check a new price against the budget)"
        if sends:
            then += ("; where a line's own note sends a required property to another listing, open that listing: its value "
                     "is the line's value for that property, so keep the line only if that value meets the requirement "
                     "(else put in another listing)")
        return json.dumps({"not_sent": "the cart lines' own pages were read before the order, and something changed",
                           "changes": changes + sends, "then": then + ", then order"})[:FULL_CHARS]

    def open_line_views(self, lines: list[str], hard: bool = False, explore: bool = True,
                        extra: int = 0) -> tuple[list[tuple[str, dict]], list[str]] | None:
        """Reads the views of listings that never showed them (one list call for several lines when the store has a list
        view). Returns the calls made and the notices heard, or None when nothing was read for want of steps or time.
        With hard, only the store's own step limit counts (code's reads ask no model); extra steps are kept for what
        follows the reads. PAGE_SINGLE: each line's own page is read with a single-listing view when the steps allow
        all of it, else as before."""
        left = self.hard_left() if hard else self.steps_left()
        keep = self.reserve(pages=False)    # the page reads are these: the rest of the finish still fits after them
        calls: list[tuple[str, dict]] = []
        for lone in ((True, False) if PAGE_SINGLE else (False,)):
            missing = self.views_missing(lines, explore, single_page=lone)
            if not missing:
                return [], []
            calls = self.view_calls(missing)
            if calls and left > keep + extra + len(calls) + 1:
                break
        if not calls or left <= keep + extra + len(calls) + 1 or monotonic() - self.began > WRAP_AT - 40:
            return None
        heard: list[str] = []
        for tool, args in calls:
            observation = self.session.call(tool, args)
            self.take_in(tool, self.roles.get(tool, "other"), args, observation)
            self.note_view(tool, args, observation)
            heard += notices_in(observation)
            if self.session.finished or self.session.broken:
                break
        return calls, heard

    def view_calls(self, missing: dict[str, list[str]]) -> list[tuple[str, dict]]:
        """The view calls that read what each listing misses (several lines a call on a several-listing view)."""
        _, many = self.listing_views()
        calls: list[tuple[str, dict]] = []
        if COVER_BY_CONTENT and self.cover:
            by_tool: dict[str, list[str]] = {}
            for ident, gap in missing.items():
                for tool in gap:
                    by_tool.setdefault(tool, []).append(ident)
            for tool, ids in by_tool.items():
                cap = max(1, self.view_cap(tool)) if tool in many else 1
                for start in range(0, len(ids), cap):
                    args = self.view_args(tool, ids[start:start + cap] if tool in many else ids[start])
                    if args:
                        calls.append((tool, args))
        elif many:
            cap = max(1, self.view_cap(many[0]))
            ids = list(missing)
            for start in range(0, len(ids), cap):
                args = self.view_args(many[0], ids[start:start + cap])
                if args:
                    calls.append((many[0], args))
        if not calls:
            for ident, gap in missing.items():
                for tool in gap:
                    args = self.view_args(tool, ident)
                    if args:
                        calls.append((tool, args))
        return calls

    def note_sends(self, lines: list[str]) -> list[str]:
        """NOTE_SENDS: for each cart line, a seller's note that sends a property the request requires on to another
        listing, or states another value of it than the line's property list (the pointer reader decides; the note's
        value then replaces the listed one), as one line for the model. At most three reader calls a cart."""
        if not NOTE_SENDS:
            return []
        out: list[str] = []
        asked = 0
        for ident in lines:
            notes = self.notes_text.get(ident, "")
            if not notes:
                continue
            item = self.item_of(ident) if self.items else ""
            for req in [r for r in self.reqs if not self.items or r.get("item") == item]:
                prop = tokens(req.get("what")) - FILLER
                about = [x for x in re.split(r"(?<=[.!?])\s+", notes) if prop and tokens(x) & prop]
                if not about or asked >= 3:
                    continue
                asked += 1
                sheet = self.facts.get(ident) or {}
                key = max(sheet, key=lambda k: len(tokens(k) & prop), default=None)
                fact = str(sheet[key]) if key is not None and tokens(key) & prop else ""
                found = self.read_pointer(req["what"], " ".join(about), fact)
                if found and found.get("spec") is not None:
                    out.append(f"{ident}: its seller's note sends {req['what']} to another listing (\"{' '.join(about)[:220]}\"); "
                               f"the request needs {req['what']} {req['value']}")
                    break
                stated = (found or {}).get("own_value") or ""
                if fact and stated and tokens(stated) and not tokens(stated) & tokens(fact) and tokens(stated) <= tokens(notes):
                    # the listing's own page states another value than its property list: the page is the later word
                    sheet[key] = stated
                    out.append(f"{ident}: its seller's note says its {req['what']} is {stated}, not {fact} as its property list "
                               f"shows; the request needs {req['what']} {req['value']}")
                    break
        return out

    def decline_early(self) -> str:
        """EXIT_ORDER: a decline with a quarter of the steps or more still left while listings shown in results were never
        read goes back once: a pick sold out or ruled out is replaced from what is still unread, not a reason to stop.
        A decline for a need the shopper told that the listings miss stands."""
        if not EXIT_ORDER or "early_decline" in self.gaps_raised or self.told_blocked \
                or self.steps_left() <= max(10, self.step_cap // 4) or monotonic() - self.began > WRAP_AT - 60 or not self.unread_left():
            return ""
        self.gaps_raised.add("early_decline")
        self.session.log.append({"role": "assistant", "content": "note: decline sent back: steps left and unread listings"})
        return (f"{self.steps_left()} steps are left and listings shown in the results were never read: read the likeliest "
                "of them (search again without a value that only a sold-out or ruled-out listing showed) before deciding; "
                "decline only if what they show rules each one out.")

    def search_first(self) -> str:
        """Before a decline: once, when no search so far sent words or a category (only a mode, a cap or nothing), the
        decline goes back: such a search cannot show the shelf, so nothing was really looked for yet."""
        if self.worded_searches or "worded" in self.gaps_raised or not any(r == "search" for r in self.roles.values()) \
                or self.steps_left() <= self.reserve() + 2 or monotonic() - self.began > WRAP_AT - 40:
            return ""
        self.gaps_raised.add("worded")
        self.session.log.append({"role": "assistant", "content": "note: decline sent back: no search sent words yet"})
        return ("no search so far sent any words or a category, so none could show the listings asked for: search with "
                "the item's own words (a value the request states may lead), read the likeliest results, then decide.")

    def rows_first(self) -> str:
        """ROWS_FIRST: before a decline, while no search has shown a listing of the item (of a line, for several items),
        the decline goes back, twice at most and while steps allow: an empty result says the words or the match mode
        missed, not that the store lacks the item."""
        if not ROWS_FIRST or self.rows_stops >= 2 or not any(r == "search" for r in self.roles.values()) \
                or self.steps_left() <= self.reserve() + 3 or monotonic() - self.began > WRAP_AT - 40:
            return ""
        if self.items:
            bare = [e["item"] for e in self.items if not any(self.item_of(i) == e["item"] for i in self.listings)]
        else:
            bare = [] if self.listings else [str(self.brief.get("goal") or "the item")[:60]]
        if not bare:
            return ""
        self.rows_stops += 1
        self.session.log.append({"role": "assistant", "content": f"note: decline sent back: no listing shown yet for {short(bare, 120)}"})
        return (f"no search so far showed a listing of {'; '.join(bare)}: an empty result means the words (or the match mode) "
                "missed, not that the store lacks it. Search again with fewer words: the item's own name or its plainest "
                "noun, without numbers, bounds or yes/no values, in the loosest match mode the search offers; then decide.")

    def has_words(self, name: str, args: dict) -> bool:
        """A search call that sends words or a category: a free-text or category value, not only a mode switch (a choice
        of four values or fewer), a number or a page marker."""
        properties = ((self.session.specs.get(name) or {}).get("parameters") or {}).get("properties") or {}
        for key, value in args.items():
            spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
            if (isinstance(spec.get("enum"), list) and len(spec["enum"]) <= 4) \
                    or re.search(r"page|cursor|token|offset|marker|after|sort|order|mode|limit", str(key), re.I):
                continue
            if isinstance(value, str) and re.search(r"[A-Za-z]", value):
                return True
            if isinstance(value, list) and any(isinstance(v, str) and re.search(r"[A-Za-z]", v) for v in value):
                return True
        return False

    def finish_problems(self, role: str) -> list[str]:
        """Faults code can see before an order that buys the cart: a cart not viewed since its last change, more than
        one listing in it, a quantity other than the one asked for, a total over the stated budget, or an item last
        seen out of stock. Raised at most twice per task."""
        if role != "order" or not any(r == "cart_change" for r in self.roles.values()):
            return []
        order = next((n for n, r in self.roles.items() if r == "order"), "")
        required = ((self.session.specs.get(order) or {}).get("parameters") or {}).get("required") or []
        if any(id_key(str(k)) for k in required):
            return []                       # the order names its item: the cart does not decide it
        # an order of a cart changed since its last view may buy nothing or the wrong thing, and one of a cart seen empty
        # buys nothing: these stops cost no step, do not use up the two cart stops, and come at most twice each
        stale = self.cart_seen is None or self.cart_changed > self.cart_seen[0]
        if stale and self.view_stops < 2 and self.steps_left() >= 2 and any(r == "cart_view" for r in self.roles.values()):
            self.view_stops += 1
            self.session.log.append({"role": "assistant", "content": "note: order sent back to view the changed cart"})
            return ["view the cart first: the order buys whatever the cart holds, and it changed since it was last viewed"]
        if not stale and self.cart_obs is not None and not self.cart_lines()[0] and self.empty_stops < 2:
            self.empty_stops += 1
            self.session.log.append({"role": "assistant", "content": "note: order of an empty cart sent back"})
            return ["the cart is empty: put the chosen listing in it first, or decline if nothing fits"]
        if self.blocks >= 2:
            return []
        problems = []
        if self.cart_seen is None or self.cart_changed > self.cart_seen[0]:
            if any(r == "cart_view" for r in self.roles.values()) and self.steps_left() > self.reserve():
                problems.append("view the cart first: the order buys whatever the cart holds, and it changed since it was last viewed")
        else:
            lines, total = self.cart_lines()
            request = self.session.request
            if not lines and self.cart_obs is not None:
                problems.append("the cart is empty: add the chosen listing first")
            kinds = {line["id"] for line in lines}
            counts = [line["quantity"] for line in lines]
            known = all(isinstance(q, (int, float)) and not isinstance(q, bool) for q in counts)
            if self.items:
                if lines and len(kinds) != len(self.items):
                    problems.append(f"the cart holds {len(kinds)} different listings but the request asks for {len(self.items)} items "
                                    f"({'; '.join(e['item'] for e in self.items)}): exactly one line per item, nothing else")
                wanted_counts = sorted(e["quantity"] for e in self.items)
                # quantities above one count only when the request writes them
                if lines and len(kinds) == len(self.items) and known and all(q == 1 or number_in(request, q) for q in wanted_counts) \
                        and sorted(int(q) for q in counts) != wanted_counts:
                    problems.append("the cart's quantities (" + ", ".join(f"{l['id']}: {int(l['quantity'])}" for l in lines)
                                    + ") do not match the request's (" + ", ".join(f"{e['item']}: {e['quantity']}" for e in self.items) + ")")
                problems += self.set_problems(lines)
            elif len(kinds) > 1:
                problems.append(f"the cart holds {len(kinds)} different listings ({', '.join(sorted(kinds))}): remove all but the chosen one")
            wanted = self.brief.get("quantity")
            # a quantity above one counts only when the request writes it and it is not one of the budgets' numbers
            if not self.items and isinstance(wanted, int) and not isinstance(wanted, bool) and wanted >= 1 and (wanted == 1 or (
                    number_in(request + " " + self.quantity_text, wanted) and wanted not in (self.question_budget, self.step_cap))):
                for line in lines:
                    if isinstance(line["quantity"], (int, float)) and int(line["quantity"]) != wanted:
                        problems.append(f"the cart holds {int(line['quantity'])} of {line['id']} but the request needs {wanted}")
            prices = [(self.listings.get(line["id"]) or {}).get("price") for line in lines]
            if total is None and lines and known and all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in prices):
                total = sum(p * q for p, q in zip(prices, counts))      # no total shown: price times quantity, summed
            budget = self.stated_budget()
            if budget is not None and total is not None and budget < total <= 20 * budget:
                problems.append(f"the cart total {total:g} is over the budget {budget:g}")
            for line in lines:
                stock = (self.listings.get(line["id"]) or {}).get("stock")
                if sold_out(stock):
                    problems.append(f"{line['id']} was last seen out of stock: take it out of the cart ({self.removal_hint()}), "
                                    "search again and add the next cheapest listing that meets everything for that item")
            # an item can sell out after it was carted with no notice: once, before the first order, each line's stock is
            # read again where a tool shows stock (a cart view that leaves stock out does not count); this stop does not
            # use up the two cart stops
            unseen = [line["id"] for line in lines if (self.listings.get(line["id"]) or {}).get("stock_step") is not None
                      and (self.listings[line["id"]]["stock_step"] or 0) < self.cart_changed]
            if unseen and not problems and "stock_check" not in self.gaps_raised and not self.finish_tries \
                    and self.steps_left() > self.pacing() + len(unseen) and not self.views_missing(unseen):
                self.gaps_raised.add("stock_check")
                self.session.log.append({"role": "assistant", "content": "note: order sent back to check stock of " + short(unseen, 200)})
                return ["before the first order, check that these cart lines are still in stock (open each one's offer or "
                        "listing, which shows its stock): " + ", ".join(unseen) + ". Things can sell out with no notice."]
        if problems:
            self.blocks += 1
            self.session.log.append({"role": "assistant", "content": "note: order stopped " + short(problems, 400)})
        return problems

    def card(self, ident: str) -> str:
        """One listing as the checks show it to a model: id, title, price, stock, category and its facts."""
        info = self.listings.get(ident) or {}
        sheet = self.facts.get(ident)
        return (f"{ident} | {str(info.get('title') or '')[:90]} | price {info.get('price')} | stock {info.get('stock')} | "
                f"category {info.get('category') or '?'} | facts: "
                + ("; ".join(f"{k}={v}" for k, v in sheet.items())[:900] if sheet else "never read") + self.compat_text(ident))

    def review_decline(self, judge_only: bool = False) -> list[str]:
        """Before a decline: one check by the stronger chain of whether the listings read already hold, for every item
        (the one item of a one-item request too), one that meets its requirements as they stand. A decline is right far
        less often than an order, so when they do, the decline is sent back once with them, and the decline's own turn
        into an order uses them. A value still to be read from another listing is left out of the check: until it is
        traced it can rule no listing out, and it never makes a decline right. It runs again (twice at most) once more
        listings were read since, as the reads before a decline make."""
        if self.decline_reviews >= 2 or (self.decline_reviews and len(self.facts) <= self.reviewed_reads) or not self.facts \
                or (self.steps_left() <= self.reserve() + 2 and not judge_only) \
                or monotonic() - self.began > (FORCE_AT - 15 if judge_only else WRAP_AT - 30):
            return []
        self.decline_reviews += 1
        self.reviewed_reads = len(self.facts)
        entries = self.items or [{"item": self.decline_label(), "quantity": self.wanted_count()}]

        unshown = [r for r in self.reqs if not r.get("pending_ref") and self.unshowable(r)]

        def own(entry: dict) -> list[dict]:
            return [r for r in self.reqs if not r.get("pending_ref") and r not in unshown
                    and (r.get("item") == entry["item"] if self.items else not r.get("item"))]
        # a need the shopper told is marked (a told value that only repeats the request is not): a best guess must confirm
        # every one, and a decline stands only when such a need is what rules the listings out
        held = self.held_back()
        wanted = "\n".join(f"- {e['item']} x{e['quantity']}: " + ("; ".join(f"{r['what']}: {r['value']}" + (" (told by the shopper)" if r in held else "")
                                                                             for r in own(e)) or "(no requirement)") for e in entries)
        pending = [r for r in self.reqs if r.get("pending_ref")]
        if pending:
            wanted += ("\n- not checked here: " + "; ".join(f"{r.get('item') + ' > ' if r.get('item') else ''}{r['what']}" for r in pending)
                       + " (a value to be read from another listing that was not traced yet: it cannot rule a listing out)")
        if unshown:
            wanted += ("\n- not checked here: " + "; ".join(f"{r.get('item') + ' > ' if r.get('item') else ''}{r['what']}: {r['value']}" for r in unshown)
                       + " (no listing read shows this property in its facts, category or title, so no listing can show it "
                         "and it cannot make a decline right)")
        budget = self.stated_budget()
        barred = sorted(i for i in self.ref_ids if not (REF_CANDIDATE and self.ref_allowed(i)))
        listed = sorted([i for i in self.facts if i not in barred],
                        key=lambda i: (not isinstance((self.listings.get(i) or {}).get("price"), (int, float)),
                                       (self.listings.get(i) or {}).get("price") or 0))
        text = (f"REQUEST:\n{self.session.request}\n\nITEMS AND THEIR REQUIREMENTS AS THEY STAND:\n{wanted}"
                + "\n\nQUESTIONS AND REPLIES:\n" + ("\n".join(f"- {q} -> {a}" for q, a in self.qa) or "(none)")
                + "\n\nNOTICES (the latest wins):\n" + ("\n".join(f"- {n}" for n in self.notices[-8:]) or "(none)")
                + (f"\n\nMONEY BUDGET NOW: {budget:g}" if budget is not None else "")
                + "\n\nLISTINGS READ:\n" + "\n".join(f"- {self.card(i)}" for i in listed[:24])
                + ("\n\nREFERENCE LISTINGS (they only define a value for the item; never the item to buy): "
                   + ", ".join(barred[:10]) if barred else "")
                + "\n\nThe requirements above already include every change the replies and notices made: a reply of no "
                  "preference about some property drops no requirement.\n\nThe assistant wants to decline. For each item, name the read listing that meets every requirement "
                  "(category and facts decide; in stock), or \"\" when none does, and say whether the lines fit the budget "
                  "together. A decline is almost never the right finish, so when no listing meets everything, also name as "
                  "best_guess the read listing of that item's kind that is in stock, within the budget, contradicted by no fact "
                  "and confirms every need the shopper told, with the most requirements confirmed (\"\" when none is like "
                  "that). A value the request states that this listing does not show while other read listings of the kind "
                  "show that property counts against it: the store would list it. Set ruled_out only when every read listing of that kind is contradicted by a fact on a "
                  "requirement as it stands, out of stock or over the budget. For each item also list as stated_fits the read "
                  "listings of that kind, in stock and within the budget, that meet every requirement the request itself states, "
                  "leaving aside the needs marked as told by the shopper. When listing and best_guess are empty, name as closest "
                  "the read listing of that kind, in stock and within the budget, that meets every need the shopper told and the "
                  "most of the other requirements (\"\" when none meets the told needs). These tell a store meant to be declined "
                  "(listings meet what the request states, but each misses a need the shopper told) from a listing misjudged. "
                  "Call report_fits.")
        reply = self.mind.ask([{"role": "system", "content": REVIEW_GUIDE}, {"role": "user", "content": text}],
                              [FITS_TOOL], max_tokens=3000, models=CHECK_MODELS, allowance=DECLINE_SPEND if judge_only else 0.0)
        calls = (reply or {}).get("tool_calls") or []
        found = parse_json((calls[0].get("function") or {}).get("arguments")) if calls else None
        if not isinstance(found, dict):
            return []
        picks = {}
        guesses = {}
        closest: dict[str, str] = {}
        blocked: set[str] = set()
        self.ruled_out = set()                          # the latest check, made with every listing read so far, stands
        for entry in found.get("items") or []:
            if not isinstance(entry, dict):
                continue
            # one item: whatever the check calls it, its listing is that item's
            item = (self.item_named(str(entry.get("item") or "")) or str(entry.get("item") or "")) if self.items else entries[0]["item"]
            if str(entry.get("listing") or "") in self.facts:
                if not sold_out((self.listings.get(str(entry["listing"])) or {}).get("stock")):
                    picks.setdefault(item, str(entry["listing"]))
            elif DECLINE_GUESS and entry.get("ruled_out") is True and (not DECLINE_AUDIT or not self.audit_open(item)) \
                    and not (BASKET_SCOPE and self.items and self.line_rows(item)):
                self.ruled_out.add(item)
            elif DECLINE_GUESS and self.guess_holds(str(entry.get("best_guess") or ""), item):
                guesses.setdefault(item, str(entry["best_guess"]))
            if ORDER_CLOSEST and item not in picks and item not in guesses:
                # nothing fits: a decline is right when listings meet what the request states but each misses a need
                # the shopper told; any other no-fit is far more often a listing misjudged
                stated = [str(i) for i in entry.get("stated_fits") or [] if str(i) in self.facts
                          and not sold_out((self.listings.get(str(i)) or {}).get("stock"))]
                if stated and self.held_back(item if self.items else None):
                    blocked.add(item)
                else:
                    near = next((c for c in (str(entry.get("closest") or ""), str(entry.get("best_guess") or ""))
                                 if self.closest_holds(c, item)), "")
                    if not near and DECLINE_AUDIT and entry.get("ruled_out") is True and item not in self.ruled_out:
                        # "every listing is ruled out" without a fact against each one: the best one nothing rules out
                        near = (self.audit_open(item) or [""])[0]
                    if near:
                        closest[item] = near
        self.closest, self.told_blocked = closest, blocked
        self.session.log.append({"role": "assistant", "content": "note: decline check " + short(found, 600)})
        self.decline_judged = True          # the check read the listings: when it names no fit, code's lenient pick waits
        if found.get("within_budget") is not False:
            self.decline_picks = dict(picks)
            self.decline_guesses = {i: g for i, g in guesses.items() if i not in picks}
        if found.get("within_budget") is False:
            return []
        if len(picks) < len(entries):
            chosen = {e["item"]: picks.get(e["item"]) or self.decline_guesses.get(e["item"]) or self.closest.get(e["item"])
                      for e in entries}
            if not DECLINE_GUESS or not all(chosen.values()):
                return []
            if any(chosen[i] == self.closest.get(i) for i in chosen):
                return ["no read listing meets every requirement, and no need the shopper told is what rules them out (the "
                        "one shape in which a decline is right): order the closest, which meets every told need: "
                        + "; ".join(f"{i}: {l} ({str((self.listings.get(l) or {}).get('title') or '')[:50]})" for i, l in chosen.items())
                        + ". Put exactly these in the cart in the asked quantities and order them."]
            # no listing meets everything, but each item has one that nothing rules out: an order of those is far more
            # often right than a decline
            return ["no read listing meets every requirement, but these are the best supported and nothing read rules "
                    "them out: " + "; ".join(f"{i}: {l} ({str((self.listings.get(l) or {}).get('title') or '')[:50]})"
                                             for i, l in chosen.items())
                    + ". Put exactly these in the cart in the asked quantities and order them; decline only if a told need or "
                      "the budget rules one out."]
        if not self.items:
            pick = picks[entries[0]["item"]]
            return [f"a read listing meets every requirement as it stands: {pick} ({str((self.listings.get(pick) or {}).get('title') or '')[:60]}). "
                    f"Put it in the cart ({entries[0]['quantity']}) and order it; decline only if a told need or the budget rules it out."]
        return ["the listings read hold a fitting listing for every item: " + "; ".join(f"{i}: {l}" for i, l in picks.items())
                + ". Put exactly these lines in the cart in the asked quantities and order; decline only if a told need or "
                  "the budget rules one of them out."]

    def checked_all(self, ident: str, item: str = "") -> bool:
        """True when code's checks of a listing settle every requirement of its item as it stands: none was left out (a
        bound code cannot read is left to the model) and none waits for a unit only a model converts."""
        reqs = self.item_reqs(item) if self.items else self.reqs
        checks, _ = self.marks(ident, reqs) if self.items else self.marks(ident)
        wanted = [r for r in reqs if not NOT_PROPERTY.search(r["what"])]
        return len(checks) >= len(wanted) and not any("another unit" in c for c in checks)

    def audit_open(self, item: str) -> list[str]:
        """DECLINE_AUDIT: the read listings of an item (one item: under its category once known) that are in stock, within
        the budget and contradicted by no fact on any requirement as it stands, and may be ordered as the closest one;
        best first (most requirements confirmed, the cheapest when wanted). Empty when each read listing has a fact
        against it, is sold out or over the budget: then "every listing is ruled out" is proven."""
        reqs = self.item_reqs(item) if self.items else None
        out = []
        for ident in self.facts:
            if not self.closest_holds(ident, item) or not self.checked_all(ident, item):
                continue                    # what code cannot read of a listing leaves the check's verdict standing
            checks, _ = self.marks(ident, reqs) if self.items else self.marks(ident)
            verdicts = [c.partition(": ")[2] for c in checks]
            if any(v.startswith(("no", "breaks")) for v in verdicts):
                continue
            price = (self.listings.get(ident) or {}).get("price")
            out.append((-sum(1 for v in verdicts if v.startswith(("yes", "ok"))),
                        price if self.cheapest_wanted() and isinstance(price, (int, float)) else 0, ident))
        return [i for *_, i in sorted(out)]

    def guess_holds(self, ident: str, item: str) -> bool:
        """True when the decline check's best-supported listing may be ordered: read, in stock, within the budget (one
        item), and contradicted by none of its item's requirements as code reads them."""
        if ident not in self.facts or sold_out((self.listings.get(ident) or {}).get("stock")):
            return False
        checks, _ = self.marks(ident, self.item_reqs(item)) if self.items else self.marks(ident)
        if any(c.split(": ", 1)[-1].startswith(("no", "breaks")) for c in checks):
            return False
        # a stated value this listing leaves open while other read listings name that property: the store lists it, so
        # its silence here counts against it
        others = set().union(*[tokens(k) for i, sheet in self.facts.items() if i != ident for k in sheet] or [set()])
        for check in checks:
            what, _, verdict = check.partition(": ")
            if verdict.startswith("unconfirmed") and "look up" not in verdict and (tokens(what) - FILLER) & others \
                    and not (tokens(what) - FILLER) & set().union(*[tokens(k) for k in (self.facts.get(ident) or {})] or [set()]):
                return False
        price = (self.listings.get(ident) or {}).get("price")
        budget = self.stated_budget()
        return bool(self.items) or budget is None or not isinstance(price, (int, float)) or price * self.wanted_count() <= budget + 1e-9

    def held_back(self, item: str | None = None) -> list[dict]:
        """Needs the shopper told as they stand (one item's, when item is given): a told value whose words the request
        already states ("for the metal, copper" after a request for copper) is no held-back need."""
        stated = [tokens(r.get("value")) - FILLER for r in self.reqs if not r.get("told")]
        return [r for r in self.reqs if r.get("told") and (item is None or r.get("item") == item)
                and not any((tokens(r.get("value")) - FILLER) and (tokens(r.get("value")) - FILLER) <= words for words in stated)]

    def closest_holds(self, ident: str, item: str) -> bool:
        """True when the decline check's closest listing may be ordered instead of a decline: read, in stock, within the
        budget (one item), filed under the item's category (its own item for a line of several), not a listing a pointer
        led to, and no held-back need the shopper told contradicted by its facts as code reads them."""
        info = self.listings.get(ident) or {}
        if not ORDER_CLOSEST or ident not in self.facts or sold_out(info.get("stock")):
            return False
        if ident in self.ref_ids and ((self.items and not REF_CANDIDATE) or not self.ref_allowed(ident)):
            return False
        if self.items and self.item_of(ident) != item:
            return False
        cats = self.item_categories() if not self.items else set()
        if cats and info.get("category") not in cats:
            return False
        reqs = self.item_reqs(item) if self.items else self.reqs
        checks, _ = self.marks(ident, reqs) if self.items else self.marks(ident)
        held = {h.get("value") for h in self.held_back(item if self.items else None)}
        told = {str(r["what"])[:30] for r in reqs if r.get("told") and r.get("value") in held}   # named as marks() names them
        if any(what in told and verdict.startswith(("no", "breaks")) for what, _, verdict in (c.partition(": ") for c in checks)):
            return False
        if any(verdict.startswith(("no", "breaks")) for _, _, verdict in (c.partition(": ") for c in checks)) \
                and self.fit_priced_out(item if self.items else None):
            return False                    # the budget or the stock decides this decline: a listing that misses is no answer
        price = info.get("price")
        budget = self.stated_budget()
        return bool(self.items) or budget is None or not isinstance(price, (int, float)) or price * self.wanted_count() <= budget + 1e-9

    def code_closest(self) -> str | None:
        """When the decline check could not answer and the shopper told no held-back need: the read listing of the item's
        category, in stock and within the budget, that its requirements rule out least (most confirmed, the cheapest when
        asked), as code reads them."""
        if not ORDER_CLOSEST or self.items or self.held_back():
            return None
        cats = self.item_categories()
        budget = self.stated_budget()
        ids = [i for i in self.facts if i not in self.ref_ids and not sold_out((self.listings.get(i) or {}).get("stock"))
               and (not cats or (self.listings.get(i) or {}).get("category") in cats)
               and not (budget is not None and isinstance((self.listings.get(i) or {}).get("price"), (int, float))
                        and self.listings[i]["price"] * self.wanted_count() > budget + 1e-9)]

        def standing(ident: str) -> tuple:
            verdicts = [c.partition(": ")[2] for c in self.marks(ident)[0]]
            price = (self.listings.get(ident) or {}).get("price")
            return (sum(1 for v in verdicts if v.startswith(("no", "breaks"))), -sum(1 for v in verdicts if v.startswith(("yes", "ok"))),
                    price if self.cheapest_wanted() and isinstance(price, (int, float)) else 0, -((self.listings.get(ident) or {}).get("step") or 0))
        best = min(ids, key=standing) if ids else None
        if best and standing(best)[0] and self.fit_priced_out():
            return None                     # what fits is over the budget or sold out: the decline stands
        return best

    def fit_priced_out(self, item: str | None = None) -> bool:
        """True when a listing of the item meets what its requirements state, as code reads them, but is sold out or over
        the budget: a read listing with none of them contradicted and one at least confirmed, or a search row with every
        one confirmed. Then the budget or the stock decides the decline, and a listing a stated requirement rules out
        is no closest listing to order."""
        reqs = self.item_reqs(item) if self.items else None
        cats = self.item_categories() if not self.items else set()
        budget = self.stated_budget()
        for ident, info in self.listings.items():
            if ident in self.ref_ids or (self.items and self.item_of(ident) != item):
                continue
            if cats and info.get("category") not in cats:
                continue
            verdicts = [c.partition(": ")[2] for c in (self.marks(ident, reqs) if self.items else self.marks(ident))[0]]
            if not verdicts or any(v.startswith(("no", "breaks")) for v in verdicts):
                continue
            if not (any(v.startswith("yes") for v in verdicts) if ident in self.facts else all(v.startswith("yes") for v in verdicts)):
                continue
            price = info.get("price")
            over = (not self.items and budget is not None and isinstance(price, (int, float))
                    and price * self.wanted_count() > budget + 1e-9)
            if over or sold_out(info.get("stock")):
                return True
        return False

    def decline_label(self) -> str:
        """The one item of a one-item request, as the decline check names it: the brief's goal in the request's words."""
        return str(self.brief.get("goal") or "the item asked for").strip()[:80] or "the item asked for"

    def unshowable(self, requirement: dict) -> bool:
        """A requirement of the request (not a need the shopper told) that no listing read shows at all, once three or
        more were read: no fact key names its property, and no title or category carries its value. The store cannot
        show it on any listing (often a use the request's story names, such as a room or an event), so it cannot make
        a decline right."""
        if requirement.get("told") or len(self.facts) < 3 or NOT_PROPERTY.search(str(requirement.get("what") or "")):
            return False
        prop = tokens(NEGATION.sub(" ", str(requirement.get("what") or ""))) - FILLER
        value = tokens(NEGATION.sub(" ", re.sub(r"\([^)]*\)", " ", str(requirement.get("value") or "")))) - FILLER
        if not prop and not value:
            return False
        for ident, sheet in self.facts.items():
            if prop and any(tokens(k) & prop for k in sheet):
                return False
            info = self.listings.get(ident) or {}
            shown = tokens(info.get("title")) | tokens(info.get("category")) | set().union(*[tokens(v) for v in sheet.values()] or [set()])
            if value and value <= shown:
                return False
        return True

    def review_basket(self) -> list[str]:
        """Before an order of several items: one check by the stronger chain of every cart line against its item's
        requirements as they stand, the quantities, the budget and, when asked, the cheapest. Sends the order back at
        most MAX_REVIEWS times; a line it cannot prove sends it back only when nothing worse was found and steps allow."""
        if self.reviews >= MAX_REVIEWS or self.cart_obs is None or self.steps_left() <= self.reserve() + 2 \
                or monotonic() - self.began > WRAP_AT - 30:
            return []
        lines, total = self.cart_lines()
        if not lines:
            return []
        card = self.card
        held = {line["id"] for line in lines}
        others = sorted((i for i in self.facts if i not in held and i not in self.ref_ids),
                        key=lambda i: (not isinstance((self.listings.get(i) or {}).get("price"), (int, float)),
                                       (self.listings.get(i) or {}).get("price") or 0))
        wanted = "\n".join(f"- {e['item']} x{e['quantity']}: " + ("; ".join(f"{r['what']}: {r['value']}" for r in self.reqs
                                                                               if r.get("item") == e["item"]) or "(no requirement)")
                           for e in self.items)
        loose = [f"{r['what']}: {r['value']}" for r in self.reqs if not r.get("item")]
        budget = self.stated_budget()
        text = (f"REQUEST:\n{self.session.request}\n\nITEMS AND THEIR REQUIREMENTS AS THEY STAND:\n{wanted}"
                + (f"\n- not tied to one item: {'; '.join(loose)}" if loose else "")
                + "\n\nQUESTIONS AND REPLIES:\n" + ("\n".join(f"- {q} -> {a}" for q, a in self.qa) or "(none)")
                + "\n\nNOTICES (the latest wins):\n" + ("\n".join(f"- {n}" for n in self.notices[-8:]) or "(none)")
                + (f"\n\nMONEY BUDGET NOW: {budget:g}" if budget is not None else ""))
        text += (f"\n\nCART (total {total if total is not None else 'not shown'}):\n"
                 + "\n".join(f"- {card(line['id'])} | quantity {line['quantity']}" for line in lines)
                 + "\n\nOTHER LISTINGS READ:\n" + ("\n".join(f"- {card(i)}" for i in others[:20]) or "(none)")
                 + "\n\nCall report_review.")
        reply = self.mind.ask([{"role": "system", "content": REVIEW_GUIDE}, {"role": "user", "content": text}],
                              [REVIEW_TOOL], max_tokens=3000, models=CHECK_MODELS)
        calls = (reply or {}).get("tool_calls") or []
        found = parse_json((calls[0].get("function") or {}).get("arguments")) if calls else None
        if not isinstance(found, dict):
            return []
        problems: list[str] = []
        verdicts = [v for v in found.get("lines") or [] if isinstance(v, dict)]
        for line in verdicts:
            if line.get("verdict") == "fail":
                problems.append(f"{line.get('listing')} ({line.get('item') or '?'}) fails: {str(line.get('reason') or '')[:220]}")
        problems += [f"missing from the cart: {str(x)[:120]}" for x in found.get("missing_items") or []]
        problems += [f"not asked for: {str(x)[:120]}" for x in found.get("extra_lines") or []]
        problems += [f"quantity: {str(x)[:160]}" for x in found.get("quantity_problems") or []]
        if found.get("over_budget") is True:
            problems.append("the cart's total is over the money budget as it stands now")
        if self.cheapest_wanted():
            problems += [f"a cheaper listing fits: {str(x)[:160]}" for x in found.get("cheaper_fits") or []]
        if not problems and self.steps_left() > self.reserve() + 6:
            problems += [f"{line.get('listing')} ({line.get('item') or '?'}) is not proven: {str(line.get('reason') or '')[:200]}"
                         for line in verdicts if line.get("verdict") == "unproven"]
        self.session.log.append({"role": "assistant", "content": "note: basket check " + short(found, 900)})
        if problems:
            self.reviews += 1
            if str(found.get("next") or "").strip():
                problems.append("suggested: " + str(found["next"])[:300])
        return problems[:8]

    def decline_reads(self) -> str:
        """Before a decline: declines are right far less often than orders, and when one was wrong the listing that fit
        was most often shown in results but never read, or one search away. Code reads the unread rows that may be the
        item (those holding the most required values first); with none left it searches the item with the values it
        needs, or shows the next page of the search that held most of the item's rows, and reads what came. At most
        DECLINE_READS steps a task, while the steps an order needs stay free; the decline check then runs again on
        everything read. Returns what the model is told, or "" when nothing was done."""
        if not DECLINE_READS or self.decline_read_steps >= DECLINE_READS or self.session.finished or self.session.broken:
            return ""
        texts: list[str] = []
        single = self.listing_views()[0]
        while self.decline_read_steps < DECLINE_READS and self.steps_left() > self.reserve() + 3 \
                and monotonic() - self.began < WRAP_AT - 40 and not self.session.finished and not self.session.broken:
            pool = self.decline_pool()
            text = ""
            if pool and self.list_view():
                text = self.read_ids(pool, "before a decline", inject=False, count=False)
            elif pool and single:
                tool = self.props_view() or single[0]
                args = self.view_args(tool, pool[0])
                observation = self.session.call(tool, args) if args is not None else None
                if observation is not None:
                    self.take_in(tool, "details", args, observation)
                    self.note_view(tool, args, observation)
                    text = self.render(tool, "details", args, observation)
            else:
                text = self.decline_search()
            if not text:
                break
            self.decline_read_steps += 1
            texts.append(text)
        if not texts:
            return ""
        self.session.log.append({"role": "assistant", "content": f"note: decline held, code read before it ({len(texts)} steps)"})
        return json.dumps({"not_sent": f"before a decline, more listings that may fit were looked at ({len(texts)} steps)",
                           "result": "\n".join(texts)[:FULL_CHARS + 1500],
                           "then": "if one of them meets every requirement as it stands, the needs the shopper told "
                                   "included, put it in the cart and order it; if none does, decline again (the decline is then "
                                   "checked against every listing read)."})

    def decline_pool(self) -> list[str]:
        """Unread rows worth reading before a decline: those the batch reads would take (the item's categories, in
        stock, within the budget), the ones whose row text holds the most of the required values first."""
        wanted = set()
        for requirement in self.reqs:
            if requirement.get("pending_ref") or NEGATION.search(f"{requirement.get('what')} {requirement.get('value')}") \
                    or NOT_PROPERTY.search(str(requirement.get("what") or "")):
                continue
            wanted |= tokens(re.sub(r"\([^)]*\)", " ", str(requirement.get("value") or ""))) - FILLER
        cats = self.item_categories()

        def rank(ident: str) -> tuple:
            record = self.listings.get(ident) or {}
            text = tokens(self.record_text.get(ident, "")) | tokens(record.get("title"))
            price = record.get("price") if isinstance(record.get("price"), (int, float)) else float("inf")
            return (-len(text & wanted), not (cats and record.get("category") in cats),
                    price if self.cheapest_wanted() else 0, -(record.get("step") or 0))
        return sorted(self.read_queue(), key=rank)

    def decline_search(self) -> str:
        """For a decline with no unread row left: once, a search of the item's name with the values it needs (unless a
        search already sent those words), else the next page of the search that showed the most of the item's rows.
        One step; returns the result as the model sees it, or "" when neither can be sent."""
        tool = next((n for n, r in self.roles.items() if r == "search" and self.text_key(n)), None)
        if tool and "decline_search" not in self.gaps_raised:
            self.gaps_raised.add("decline_search")
            key, extra = self.text_key(tool)
            entry = next((e for e in self.items if not self.decline_picks.get(e["item"])), None) if self.items else None
            reqs = self.item_reqs(entry["item"]) if entry else [r for r in self.reqs if not r.get("item")]
            names = [entry["item"]] if entry else [str(e.get("item") or "") for e in (self.brief.get("items") or [])[:1]
                                                    if isinstance(e, dict)] or [str(self.brief.get("goal") or "")]
            values = [re.sub(r"\([^)]*\)", " ", str(r.get("value") or "")) for r in reqs
                      if not r.get("pending_ref") and not NEGATION.search(f"{r.get('what')} {r.get('value')}")
                      and not NOT_PROPERTY.search(str(r.get("what") or "")) and len(str(r.get("value") or "")) <= 30]
            words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-']*", " ".join(values + names))
                     if (len(w) > 2 or re.search(r"\d", w)) and w.lower() not in FILLER and w.lower() not in ASKING]
            words = list(dict.fromkeys(w.lower() for w in words))[:8]
            sent = [{str(w).lower() for v in a.values() if isinstance(v, str) for w in v.split()} for a, _, _ in self.sweeps]
            if len(words) >= 2 and set(words) not in sent:
                args, problems = fit_args((self.session.specs.get(tool) or {}).get("parameters") or {"type": "object"},
                                          {key: " ".join(words), **extra})
                if not problems:
                    observation = self.session.call(tool, args)
                    if observation is not None:
                        self.take_in(tool, "search", args, observation)
                        self.note_view(tool, args, observation)
                        self.worded_searches += 1
                        self.session.log.append({"role": "assistant", "content": f"note: code searched '{' '.join(words)}' before a decline"})
                        return self.render(tool, "search", args, observation)
        cats = self.item_categories()
        entry = max((e for e in self.more_pages.values() if e["mark"] is not None and self.marker_key(e["tool"], e["mark"])
                     and (not isinstance(e["total"], int) or len(e["seen"]) < e["total"])),
                    key=lambda e: (sum(1 for i in e["seen"] if (self.listings.get(i) or {}).get("category") in cats), e["step"]),
                    default=None)
        if entry is None:
            return ""
        args, problems = fit_args((self.session.specs.get(entry["tool"]) or {}).get("parameters") or {"type": "object"},
                                  {**entry["args"], self.marker_key(entry["tool"], entry["mark"]): entry["mark"]})
        if problems:
            return ""
        observation = self.session.call(entry["tool"], args)
        if observation is None:
            return ""
        self.take_in(entry["tool"], "search", args, observation)
        self.note_view(entry["tool"], args, observation)
        self.session.log.append({"role": "assistant", "content": "note: next page shown before a decline"})
        return self.render(entry["tool"], "search", args, observation)

    def unread_candidates(self, role: str) -> list[str]:
        """Before a decline, or an order when the cheapest is asked for: in-stock listings whose titles share a word
        with the goal or a required value and whose facts were never read (cheaper than the cart's item for an order).
        Raised once per role, and only while steps allow reading a few."""
        if role in self.gaps_raised or self.steps_left() <= self.pacing() + 2 or monotonic() - self.began > WRAP_AT - 40:
            return []
        if role == "order" and (not self.cheapest_wanted() or self.items):
            return []                       # several items: the basket check looks for cheaper fits per line
        if not any(r == "details" for r in self.roles.values()):
            return []
        words = tokens(self.brief.get("goal")) | set().union(*[tokens(r.get("value")) for r in self.reqs if not r.get("pending_ref")] or [set()])
        if not words:
            return []
        ceiling = None
        if role == "order":
            lines, _ = self.cart_lines() if self.cart_obs is not None else ([], None)
            prices = [(self.listings.get(line["id"]) or {}).get("price") for line in lines]
            prices = [p for p in prices if isinstance(p, (int, float)) and not isinstance(p, bool)]
            if not prices:
                return []
            ceiling = min(prices)
        found = []
        for record in sorted(self.listings.values(), key=lambda r: (not isinstance(r["price"], (int, float)), r["price"] or 0)):
            stock = record["stock"]
            # a listing filed under the item's category counts even when its title uses other words
            if record["id"] in self.facts or sold_out(stock) or not (tokens(record["title"]) | tokens(record.get("category"))) & words:
                continue
            if ceiling is not None and not (isinstance(record["price"], (int, float)) and record["price"] < ceiling):
                continue
            found.append(f"{record['id']} ({record['title'][:50]}, {record['price']})")
        if found:
            self.gaps_raised.add(role)
            self.session.log.append({"role": "assistant", "content": f"note: {role} sent back, unread " + short(found[:6], 400)})
        return found[:6]

    def coverage_gap(self, role: str) -> str:
        """Before the first order when the cheapest is asked for: unless some search with a price limit at or above the
        pick listed every match it reported, cheaper fitting listings may be hidden. Raised once, and only when a search
        tool takes a price limit and steps allow a sweep."""
        if role != "order" or "coverage" in self.gaps_raised or not self.cheapest_asked() or self.items:
            return ""
        if self.steps_left() <= self.pacing() + 2 or monotonic() - self.began > WRAP_AT - 40:
            return ""

        def capped(name: str) -> bool:
            return bool(re.search(r"price|cap|max|budget|cost|spend|limit", name, re.I))
        searches = [n for n, r in self.roles.items() if r == "search"]
        if not any(capped(k) and kinds_of(v) and set(kinds_of(v)) & {"number", "integer"}
                   for n in searches for k, v in (((self.session.specs.get(n) or {}).get("parameters") or {}).get("properties") or {}).items()
                   if isinstance(v, dict)):
            return ""
        lines, _ = self.cart_lines() if self.cart_obs is not None else ([], None)
        prices = [(self.listings.get(line["id"]) or {}).get("price") for line in lines]
        prices = [p for p in prices if isinstance(p, (int, float)) and not isinstance(p, bool)]
        if not prices:
            return ""
        pick = min(prices)
        # searches that differ only in their page belong together
        groups: dict[str, list] = {}
        for args, total, ids in self.sweeps:
            key = json.dumps({k: v for k, v in args.items() if not re.search(r"page|offset|cursor|start", k, re.I)}, sort_keys=True, default=str)
            entry = groups.setdefault(key, [args, None, set()])
            entry[1] = total if isinstance(total, int) else entry[1]
            entry[2] |= ids
        for args, total, ids in groups.values():
            limits = [v for k, v in args.items() if capped(k) and isinstance(v, (int, float)) and not isinstance(v, bool)]
            if limits and min(limits) >= pick - 1e-9 and isinstance(total, int) and len(ids) >= total:
                return ""
        self.gaps_raised.add("coverage")
        self.session.log.append({"role": "assistant", "content": f"note: order sent back, no complete search up to {pick:g}"})
        hint = ""
        capped_runs = [(min(v for k, v in a.items() if capped(k) and isinstance(v, (int, float)) and not isinstance(v, bool)), t, len(i))
                       for a, t, i in groups.values()
                       if any(capped(k) and isinstance(v, (int, float)) and not isinstance(v, bool) for k, v in a.items())
                       and isinstance(t, int) and t > 0]
        visible = self.visible_rows()
        if capped_runs and visible:
            limit, total, _ = max(capped_runs, key=lambda r: r[1])
            if total > visible:
                hint = (f" Your search with a limit of {limit:g} reported {total} matches while at most about {visible} can be seen "
                        f"across its pages: a limit near {max(1, int(limit * visible / total)):g} lists them all.")
        return (f"no search so far showed every match priced up to your pick ({pick:g}). Search the item's category (its name in the parameter made "
                f"for the category or department) with a price limit of {pick:g}; if it reports more matches than it shows, page through or lower the limit in steps until "
                "every match is shown, and read the facts of the cheaper ones, cheapest first." + hint)

    def page_field(self, tool: str) -> tuple[str, int] | None:
        """The search tool's page parameter and its highest page, when it has one."""
        for key, spec in (((self.session.specs.get(tool) or {}).get("parameters") or {}).get("properties") or {}).items():
            if isinstance(spec, dict) and re.search(r"page", key, re.I) and set(kinds_of(spec)) & {"integer", "number"} \
                    and isinstance(spec.get("maximum"), int) and spec["maximum"] > 1:
                return key, spec["maximum"]
        return None

    def visible_rows(self) -> int:
        """How many listings a search can show across all its pages, from the fullest page seen and the page limit."""
        tool = next((n for n, r in self.roles.items() if r == "search"), None)
        field = self.page_field(tool) if tool else None
        shown = max((len(i) for _, _, i in self.sweeps), default=0)
        # with no numbered page limit (a next-page marker instead), about three pages are worth reading in a task
        return shown * (field[1] if field else 3)

    def page_through(self, name: str, args: dict, observation: Any) -> None:
        """When the cheapest is asked for, a price-limited search that shows only its first page of a total the pages can
        hold is paged through by code (one step a page), so every listing under that limit is seen."""
        if not self.cheapest_asked() or not isinstance(observation, dict):
            return
        field = self.page_field(name)
        if field is None or not any(re.search(r"price|cap|max|budget|cost|spend|limit", k, re.I)
                                    and isinstance(v, (int, float)) and not isinstance(v, bool) for k, v in args.items()):
            return
        key, last = field
        total = next((v for k, v in observation.items() if re.search(r"match|total|count|found|hits|results", str(k), re.I)
                      and isinstance(v, int) and not isinstance(v, bool)), None)
        found: list[dict] = []
        records_in(observation, found)
        page = args.get(key) or 1
        if not isinstance(total, int) or not found or not isinstance(page, int) or page != 1 or total <= len(found) \
                or total > len(found) * last:
            return
        for nxt in range(2, min(last, -(-total // len(found))) + 1):
            if self.steps_left() <= self.reserve() + 6 or self.session.finished or self.session.broken:
                return
            more = dict(args, **{key: nxt})
            call = {"id": f"page-{self.session.steps}", "type": "function", "function": {"name": name, "arguments": json.dumps(more)}}
            self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            result = self.act(call, paging=True)
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            self.short_of[call["id"]] = self.summary(result)
            self.session.log.append({"role": "assistant", "content": f"note: paged a price-limited search to page {nxt}"})

    # -- the task ---------------------------------------------------------------------------------------------------------
    def run(self) -> list[dict]:
        session = self.session
        if session.finished or not session.tools:
            return session.log
        self.read_brief()
        brief = {k: v for k, v in self.brief.items() if k != "tools"}
        roles = ", ".join(f"{n}: {r}" for n, r in self.roles.items())
        opening = f"\n\nOpening observation:\n{short(session.opening, 2000)}" if session.opening is not None else ""
        self.messages = [
            {"role": "system", "content": GUIDE},
            {"role": "user", "content": f"The shopper's request:\n{session.request}{opening}\n\n"
                                        f"A brief of it (the request itself rules): {short(brief, 2500)}\n\nTool roles: {roles}"},
        ]
        idle = 0                            # replies in a row that took no step
        if TRACE_REQUEST:
            self.trace_request()
        if FIRST_FULL_SEARCH:
            self.first_search()
        if FIRST_PROMPT_MESSAGE and not PROMPT_AT_ASK:
            self.first_message()
        while not session.finished and not session.broken and self.hard_left() > 0:
            if monotonic() - self.began > FORCE_AT or self.mind.spent >= SPEND_HARD + self.mind.room or idle >= MAX_IDLE:
                break
            if not self.pace or self.pace[-1][0] != session.steps:
                self.pace.append((session.steps, self.mind.spent, monotonic() - self.began))
            before = session.steps
            if self.prompt_late():
                idle = 0                    # the whole request, before the steps left are only the finish's
                continue
            if self.line_depth_read():
                idle = 0                    # a line's own listings before its first question
                continue
            if ASK_BEFORE_WALK and self.questions and self.ask_room() > 0 and self.auto_question():
                idle = 0                    # the questions left go first: one step each
                continue
            if self.walk_refs():
                idle = 0
                continue
            if self.bulk_read():
                idle = 0
                continue
            if self.shelf_page():
                idle = 0
                continue
            if self.auto_question():
                idle = 0
                continue
            if self.fit_read():
                idle = 0
                continue
            if self.cart_due():
                idle = 0
                continue
            reply = self.mind.ask(self.view(), session.tools, tail=True)
            if reply is None:
                idle += 1
                continue
            calls = [c for c in reply.get("tool_calls") or [] if isinstance(c, dict)][:MAX_CALLS_PER_REPLY]
            if not calls:
                idle += 1
                self.messages += [{"role": "assistant", "content": str(reply.get("content") or "")[:1500]},
                                  {"role": "user", "content": "Reply with a call to one of the tools."}]
                continue
            if str(reply.get("content") or "").strip():
                session.log.append({"role": "assistant", "content": "note: thinking " + str(reply["content"])[:300]})
            for number, call in enumerate(calls):
                call.setdefault("id", f"call-{session.steps}-{number}")
                call.setdefault("type", "function")
            self.messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls})
            for call in calls:
                if session.finished or session.broken:
                    result = json.dumps({"not_sent": "the task has ended"})
                else:
                    result = self.act(call)
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                self.short_of[call["id"]] = self.summary(result)
            self.bulk_turn = 0
            self.model_searched = any(self.roles.get(str((c.get("function") or {}).get("name") or "")) == "search" for c in calls)
            if self.pending_pages is not None:
                name, args, observation = self.pending_pages
                self.pending_pages = None
                self.page_through(name, args, observation)
            idle = 0 if session.steps > before else idle + 1
            if idle == 3:
                self.messages.append({"role": "user", "content": "Your last calls were not sent. Read why each was refused and take a "
                                                                 "different action now: search, read a listing's details, change the "
                                                                 "cart, or finish."})
        if not session.finished and not session.broken:
            self.finish_anyway()
        session.log.append({"role": "assistant", "content": f"note: model calls {self.mind.calls}, spent ${self.mind.spent:.4f}, "
                                                            f"prompt tokens {self.mind.prompt_tokens} ({self.mind.cached_tokens} cached)"})
        return session.log

    def summary(self, text: str) -> str:
        """A short form of an old tool result: the property lists it showed, else its listings, else its first
        characters. The state note keeps every fact in full."""
        value = parse_json(text)
        sheets: list[tuple[str, dict]] = []
        facts_in(value, sheets)
        if sheets:
            return "(shortened; full facts are in the state note) " + short(
                {ident: "; ".join(f"{k}={v}" for k, v in sheet.items()) for ident, sheet in sheets[:4]}, SHORT_CHARS + 500)
        found: list[dict] = []
        records_in(value, found)
        if found:
            return "(shortened) " + short([{k: v for k, v in r.items() if v not in (None, "")} for r in found[:12]], SHORT_CHARS)
        return text[:SHORT_CHARS]

    def finish_anyway(self) -> None:
        """No model finish: order the cart when something was carted; with an empty cart, first fill it from the read
        listings that fit (a decline is right far less often than a fitting order); decline only when nothing fits."""
        session = self.session
        self.exiting = True                 # code finishes now: only the store's step limit counts
        if session.finished or session.broken or self.steps_left() <= 0:
            return
        # a cart last seen empty is filled like one never used: an order of an empty cart buys nothing
        emptied = self.carted and self.cart_obs is not None and self.cart_seen is not None \
            and self.cart_seen[0] >= self.cart_changed and not self.cart_lines()[0]
        if EXIT_ORDER and self.carted and not self.held_lines():
            emptied = True                  # code's own record: the last line was taken out after the latest cart view
        filled = False
        put: list[str] = []                 # the listings this finish put in the cart itself
        if (not self.carted or emptied) and not self.decline_judged and self.facts and (self.items or not self.fit_list()):
            # nothing confirmed to buy: the decline check reads the listings by meaning first (no step), so the code
            # does not buy a listing it only failed to rule out by matching words
            self.review_decline(judge_only=True)
        if not self.carted or emptied:
            calls = self.basket_calls(self.plausible_lines() if self.decline_judged else self.fallback_lines())
            if not calls and EXIT_ORDER:
                calls = self.basket_calls(self.exit_lines())
        elif EARLY_CART and self.items:
            # some lines are in the cart: the lines it lacks get their best listing, so the order is whole
            held = self.held_lines()
            missing = self.missing_items(held)
            add = self.exit_lines(only=missing, held=held) if missing and not self.told_blocked else []
            whole = any(self.list_shaped(n) for n, r in self.roles.items() if r == "cart_change")
            calls = self.basket_calls(held + add if whole else add) if add else []
        else:
            calls = []
        if calls:
            if calls and self.steps_left() > len(calls):
                for name, args in calls:
                    observation = session.call(name, args)
                    self.take_in(name, "cart_change", args, observation)
                    put += [i for i in self.ids_in(args) if i not in put]
                    if session.finished or session.broken:
                        return
                filled = True
        wants = ["order", "decline"] if (self.carted and not emptied) or filled else ["decline", "order"]
        for want in wants:
            if want == "order" and (self.cart_book or put):
                # each line read through what it misses before the order, when steps allow (a store whose cart reply
                # does not name the listing leaves code's record empty: the lines it just put in are known anyway)
                known = {i for i in put if i in self.facts and self.line_broken(i)}   # chosen knowing that (closest)
                self.open_line_views(list(dict.fromkeys(self.cart_book + put)))
                if session.finished or session.broken:
                    return
                if FINISH_RECHECK and put and not self.recheck_put([i for i in put if i not in known]):
                    if "decline" in wants[wants.index(want) + 1:]:
                        continue            # a line it put in is ruled out by its facts and has no other listing: decline
                    want = "decline"
            found = self.finish_call(want)
            if found is not None:
                self.tell_before_finish(want)  # a message the request asks for before this finish, when one can be spared
            for attempt in range(2):        # a finish refused by a change at that moment usually goes through the second time
                if found is None or session.finished or session.broken or self.steps_left() <= 0:
                    break
                if attempt and want == "order":
                    self.swap_dead_lines()  # unless it said the pick is gone: the same line cannot go through again
                name, args = found
                observation = session.call(name, args)
                self.take_in(name, want, args, observation)
            if session.finished:
                return

    def item_rows(self) -> list[str]:
        """One item: result rows never read that may be it (filed under its category once known, else sharing a word
        with the goal), in stock and within the budget, the cheapest first when wanted (newest otherwise)."""
        cats = self.item_categories()
        budget = self.stated_budget()
        goal = tokens(self.brief.get("goal")) - FILLER
        out = []
        for ident, info in self.listings.items():
            price = info.get("price")
            if ident in self.facts or ident in self.ref_ids or sold_out(info.get("stock")) \
                    or (budget is not None and isinstance(price, (int, float)) and price * self.wanted_count() > budget + 1e-9):
                continue
            if (cats and info.get("category") not in cats) or (not cats and not tokens(info.get("title")) & goal):
                continue
            out.append(((price or 0) if self.cheapest_wanted() and isinstance(price, (int, float)) else 0, -(info.get("step") or 0), ident))
        return [i for *_, i in sorted(out)]

    def line_broken(self, ident: str) -> bool:
        """A read listing a fact rules out for the item it would fill (several items: that line's requirements)."""
        item = self.item_of(ident) if self.items else ""
        checks, _ = self.marks(ident, self.item_reqs(item)) if self.items else self.marks(ident)
        return any(c.split(": ", 1)[-1].startswith(("no", "breaks")) for c in checks)

    def recheck_put(self, put: list[str]) -> bool:
        """FINISH_RECHECK: after code's own finish read the lines it put in, a line whose facts now rule it out is taken
        out and replaced by its line's next listing (read first when the steps allow); False when such a line has no
        replacement that holds, so the finish should not order it."""
        for ident in [i for i in put if i in self.facts and self.line_broken(i)]:
            entry = next((e for e in self.items if e["item"] == self.item_of(ident)), None) if self.items else None
            count = entry["quantity"] if entry else self.wanted_count()
            held = [(i, q) for i, q in self.held_lines() if i != ident]
            new = self.exit_lines(only=[entry], held=held) if entry else [l for l in self.exit_lines() if l[0] != ident]
            new = [l for l in new if l[0] != ident]
            if not new and not self.items:
                new = [(r, count) for r in self.item_rows() if r != ident][:1]     # one item: its next unread row
            if not new:
                return False
            target = new[0][0]
            if target not in self.facts:
                self.open_line_views([target], hard=True, explore=False, extra=3)
            if target not in self.facts or self.line_broken(target) or self.session.finished or self.session.broken:
                return False
            calls = self.swap_calls([(ident, count)], target, count)
            if not calls or self.hard_left() <= len(calls) + 1:
                return False
            for tool, args in calls:
                observation = self.session.call(tool, args)
                self.take_in(tool, "cart_change", args, observation)
                if self.session.finished or self.session.broken:
                    return False
            self.session.log.append({"role": "assistant", "content": f"note: forced finish: {ident} ruled out by its facts, {target} instead"})
        return True

    def at_exit(self) -> bool:
        """True when the task is at its end: the wind-up time, the spend for finishing, or the finishing reserve."""
        return monotonic() - self.began > WRAP_AT - 10 or self.mind.spent >= SPEND_FINISH + self.mind.room \
            or self.steps_left() <= self.reserve() + 1

    def exit_lines(self, only: list[dict] | None = None, held: list[tuple[str, int]] | None = None) -> list[tuple[str, int]]:
        """EXIT_ORDER: (listing id, quantity) lines for a forced finish with nothing carted: for each line its best read
        listing that nothing rules out (in stock, within the budget, told needs and confirmed requirements first, the
        cheapest when asked), or, when no listing of that line was read at all, its best result row (the order then
        reads its views first). Empty when the shopper told a need the listings miss, or a line's read listings are
        each ruled out, sold out or over the budget: that decline stands."""
        if self.told_blocked:
            return []
        budget = self.stated_budget()
        entries = only if only is not None else self.items or [{"item": "", "quantity": self.wanted_count()}]
        info = lambda i: self.listings.get(i) or {}
        cats = self.item_categories()
        lines: list[tuple[str, int]] = []
        for entry in entries:
            item, count = entry["item"], entry["quantity"] if self.items else self.wanted_count()
            label = item if self.items else self.decline_label()
            if label in self.ruled_out:
                # the decline check read every listing of it and found each ruled out: only its closest one may go
                near = self.closest.get(label)
                unread = self.line_rows(item) if BASKET_SCOPE and self.items else []
                if (not near or sold_out(info(near).get("stock"))) and unread:
                    lines.append((unread[0], count))   # BASKET_SCOPE: a row of the line still unread is read first
                    continue
                if not near or sold_out(info(near).get("stock")):
                    return []
                lines.append((near, count))
                continue

            def mine(i: str) -> bool:
                if i in self.ref_ids:
                    return False
                if self.items:
                    return self.item_of(i) == item
                return not cats or info(i).get("category") in cats

            def within(i: str) -> bool:
                price = info(i).get("price")
                return not sold_out(info(i).get("stock")) and not (
                    budget is not None and not self.items and isinstance(price, (int, float)) and price * count > budget + 1e-9)
            read = [i for i in self.facts if mine(i)]
            reqs = self.item_reqs(item) if self.items else None

            def standing(i: str) -> tuple:
                verdicts = [c.partition(": ")[2] for c in (self.marks(i, reqs) if self.items else self.marks(i))[0]]
                price = info(i).get("price")
                return (any(v.startswith(("no", "breaks")) for v in verdicts), -sum(1 for v in verdicts if v.startswith(("yes", "ok"))),
                        price if self.cheapest_wanted() and isinstance(price, (int, float)) else 0, -(info(i).get("step") or 0))
            if read:
                fine = sorted((i for i in read if within(i)), key=standing)
                if not fine or standing(fine[0])[0]:
                    unread = self.line_rows(item) if BASKET_SCOPE and self.items else []
                    if unread:
                        lines.append((unread[0], count))   # its read listings are out, an unread row may not be
                        continue
                    return []               # every listing read for this line is ruled out, sold out or over the budget
                lines.append((fine[0], count))
                continue
            rows = sorted((i for i in self.listings if mine(i) and within(i)
                           and (cats or self.items or (tokens(info(i).get("title")) & (tokens(self.brief.get("goal")) - FILLER)))),
                          key=lambda i: ((info(i).get("price") or 0) if self.cheapest_wanted() else 0, -(info(i).get("step") or 0)))
            if BASKET_SCOPE and self.items:
                rows = self.line_rows(item)     # by what the rows' own text says they fit, then as above
            if not rows:
                return []
            lines.append((rows[0], count))
        every = lines + list(held or [])
        prices = [info(i).get("price") for i, _ in every]
        if budget is not None and all(isinstance(p, (int, float)) for p in prices) and \
                sum(p * q for p, (_, q) in zip(prices, every)) > budget + 1e-9:
            return []
        if lines:
            self.session.log.append({"role": "assistant", "content": "note: forced finish: ordering " + short(lines, 200)})
        return lines

    def line_rows(self, item: str) -> list[str]:
        """BASKET_SCOPE: rows shown in results for one line of a several-item order (filed with that item) that were never
        read, in stock and not a listing a pointer led to; a row whose own text fits a model code the line needs first,
        one whose text names only other models of that maker left out, then the cheapest when wanted, newest otherwise."""
        reqs = self.item_reqs(item)
        out = []
        for ident, info in self.listings.items():
            if ident in self.facts or ident in self.ref_ids or sold_out(info.get("stock")) or self.item_of(ident) != item:
                continue
            fitted = [self.fit_verdict(ident, r) for r in reqs] if FIT_CODES else []
            if any(f and f.startswith("no") for f in fitted):
                continue
            price = info.get("price") if isinstance(info.get("price"), (int, float)) else 0
            out.append((-sum(1 for f in fitted if f and f.startswith("yes")), price if self.cheapest_wanted() else 0,
                        -(info.get("step") or 0), ident))
        return [i for *_, i in sorted(out)]

    def held_lines(self) -> list[tuple[str, int]]:
        """(listing id, quantity) of what the cart holds as code knows it: the latest cart view when nothing changed since,
        else code's own record of cart changes."""
        fresh = self.cart_obs is not None and self.cart_seen is not None and self.cart_seen[0] >= self.cart_changed
        if fresh:
            return [(l["id"], l["quantity"] if isinstance(l.get("quantity"), int) and not isinstance(l.get("quantity"), bool) else 1)
                    for l in self.cart_lines()[0]]
        added = [] if self.last_carted == self.taken_out else list(self.last_carted)   # the latest change took it out
        return [(i, 1) for i in dict.fromkeys(self.cart_book + self.code_put + added)]

    def missing_items(self, held: list[tuple[str, int]]) -> list[dict]:
        """The order's items no cart line fills."""
        covered = {self.item_of(i) for i, _ in held}
        return [e for e in self.items if e["item"] not in covered]

    def cart_due(self) -> bool:
        """EARLY_CART: a several-item order at its first-finish point (counted with a step for each line still to put in):
        code puts the best read listing of a line the cart lacks into the cart, one line a step, in the conversation, so
        the finish has every line. Returns True when a cart call was made."""
        if not EARLY_CART or not self.items or self.session.finished or self.session.broken or self.finish_tries:
            return False
        held = self.held_lines()
        missing = self.missing_items(held)
        if not missing or self.early_carts >= len(self.items) + 1 or monotonic() - self.began > WRAP_AT - 30:
            return False
        pick = self.line_turn(held, missing) if LINE_SHARES and len(self.items) >= 2 else None
        if pick is None:
            if self.steps_left() > self.pacing() + len(missing) + 2 or self.steps_left() <= self.reserve() + 1:
                return False
            pick = next(((self.line_pick(e), e) for e in missing if self.line_pick(e) and self.line_pick(e) not in dict(held)), None)
        if pick is None:
            return False
        ident, entry = pick
        whole = any(self.list_shaped(n) for n, r in self.roles.items() if r == "cart_change")
        calls = self.basket_calls(held + [(ident, entry["quantity"])] if whole else [(ident, entry["quantity"])])
        if not calls:
            return False
        self.early_carts += 1
        tool, args = calls[0]
        self.session.log.append({"role": "assistant", "content": f"note: early cart: {ident} for {entry['item']}"})
        call = {"id": f"cart-{self.session.steps}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        result = self.act(call)
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        self.short_of[call["id"]] = self.summary(result)
        if result.startswith('{"not_sent"'):
            return False
        self.code_put.append(ident)
        return True

    def line_fit(self, entry: dict) -> str | None:
        """LINE_SHARES: the line's read listing that code confirms on every requirement of its item as they stand (in
        stock, every requirement read, facts that agree), the cheapest when wanted; None when it has none."""
        item = entry["item"]
        reqs = self.item_reqs(item)
        fits = []
        for ident in self.facts:
            if self.item_of(ident) != item or (ident in self.ref_ids and not (REF_CANDIDATE and self.ref_allowed(ident))):
                continue
            info = self.listings.get(ident) or {}
            checks, clean = self.marks(ident, reqs)
            if sold_out(info.get("stock")) or not (checks and clean) or self.mixed.get(ident) or not self.checked_all(ident, item):
                continue
            price = info.get("price") if isinstance(info.get("price"), (int, float)) else float("inf")
            fits.append((price if self.cheapest_wanted() else 0, -(info.get("step") or 0), ident))
        return min(fits)[-1] if fits else None

    def line_share(self) -> float:
        """LINE_SHARES: the steps each line of a several-item order may use: those its lines used so far and those still
        free (less the finishing reserve and the questions left), split over the lines."""
        free = max(0, self.steps_left() - self.reserve() - self.ask_room())
        return (sum(self.line_steps.values()) + free) / max(1, len(self.items))

    def line_turn(self, held: list[tuple[str, int]], missing: list[dict]) -> tuple[str, dict] | None:
        """LINE_SHARES: the next line to put in the cart before the finish, with its listing: a line with a confirmed fit
        once the questions are done (the confirmed fit), else a line that used its share of the steps (its best read
        listing nothing rules out). None when no line's turn has come."""
        if self.steps_left() <= self.reserve() + 1:
            return None
        settled = self.ask_room() <= 0 or self.auto_off \
            or self.session.steps >= max(14, self.wait_ends()) + 2 * max(0, self.question_budget or 0)
        share = self.line_share()
        carted = dict(held)
        for entry in missing:
            fit = self.line_fit(entry)
            if fit and settled and fit not in carted:
                return fit, entry
        for entry in missing:
            if self.line_steps.get(entry["item"], 0) >= share:
                pick = self.line_pick(entry)
                if pick and pick not in carted:
                    return pick, entry
        return None

    def count_line_step(self, role: str, args: dict) -> None:
        """LINE_SHARES: a search or read counts toward the line it serves: the line whose name a search's words give, or
        the lines of the listings a read showed."""
        if role == "search":
            line = self.item_named(" ".join(str(v) for v in args.values() if isinstance(v, str)))
            served = {line} if line else set()
        else:
            served = {self.item_of(i) for i in self.ids_in(args)} - {""}
        for line in served:
            self.line_steps[line] = self.line_steps.get(line, 0) + 1

    def line_done_search(self, args: dict) -> str:
        """LINE_SHARES: a search for a line that already has a confirmed fit while another line has none goes back."""
        line = self.item_named(" ".join(str(v) for v in args.values() if isinstance(v, str)))
        entry = next((e for e in self.items if e["item"] == line), None)
        if not entry or not self.line_fit(entry):
            return ""
        held = dict(self.held_lines())
        open_lines = [e["item"] for e in self.items if not self.line_fit(e) and not any(self.item_of(i) == e["item"] for i in held)]
        if not open_lines:
            return ""
        self.session.log.append({"role": "assistant", "content": f"note: search for {line} sent back, lines without a fit: {open_lines}"})
        return (f"{line} already has a listing that meets its requirements ({self.line_fit(entry)}); these lines have none yet: "
                f"{', '.join(open_lines)}. Search for one of those first: an order missing a line fails.")

    def line_depth_read(self) -> bool:
        """BASKET_DEPTH: several items, before a line's first question: while fewer than three of its listings show their
        properties, code reads its cheapest unread rows (newest when no price rule), several a call when the store's
        several-listing view shows properties; a line with no row shown gets one search for its name and the values
        the request states for it. While the steps left cover the questions and the finish. True when a step was taken."""
        if not (BASKET_DEPTH and len(self.items) >= 2) or self.ask_room() <= 0 or self.auto_off \
                or self.session.finished or self.session.broken or self.depth_reads >= 3 * len(self.items) + 2:
            return False
        if self.steps_left() <= self.reserve() + self.ask_room() + 2 * len(self.items) + 4 or monotonic() - self.began > WRAP_AT - 60:
            return False
        asked = {self.item_named(q) for q, _ in self.answered_qa()}
        for entry in self.items:
            item = entry["item"]
            if item in asked:
                continue
            read = [i for i in self.facts if self.item_of(i) == item and (not self.cover or "props" in self.cover.get(i, set()))]
            if len(read) >= 3:
                continue
            tool = self.reader()
            rows = [i for i, r in self.listings.items() if self.item_of(i) == item and i not in self.ref_ids and not sold_out(r.get("stock"))
                    and (i not in self.facts or (self.cover and "props" not in self.cover.get(i, set())))
                    and (tool, i) not in self.depth_tried]
            if rows and tool:
                rows.sort(key=lambda i: ((self.listings[i].get("price") or 0) if self.cheapest_wanted() else 0,
                                         -(self.listings[i].get("step") or 0)))
                cap = self.view_cap(tool) if tool == self.list_view() else 1
                self.depth_tried |= {(tool, i) for i in rows[:max(1, cap)]}
                if self.read_ids(rows, f"three of the {item}'s listings before asking about it", inject=True, count=False, tool=tool):
                    self.depth_reads += 1
                    return True
                continue
            if read or item in self.depth_searched:
                continue
            self.depth_searched.add(item)
            tool = next((n for n, r in self.roles.items() if r == "search" and self.text_key(n)), None)
            if tool is None:
                continue
            key, extra = self.text_key(tool)
            stated = [str(r.get("value")) for r in self.item_reqs(item) if not r.get("pending_ref") and len(str(r.get("value"))) <= 30
                      and not tokens(r.get("value")) <= tokens(item)]
            if PLAIN_LINE_WORDS:
                # a bound ("at least 30"), a yes/no or a stock word never stands in a listing's title as the request
                # writes it; a model code or a plain value may
                stated = [v for v in stated if not COMPARE.search(v) and not NOT_PROPERTY.search(v)
                          and not re.fullmatch(r"\s*(?:yes|no|true|false|none|any|either)\s*", v, re.I)][:1]
            words = " ".join([item] + stated[:2])
            call = {"id": f"line-{self.session.steps}", "type": "function",
                    "function": {"name": tool, "arguments": json.dumps({**extra, key: words})}}
            self.session.log.append({"role": "assistant", "content": f"note: code searched '{words}' for a line before its question"})
            self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            result = self.act(call)
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            self.short_of[call["id"]] = self.summary(result)
            self.depth_reads += 1
            return True
        return False

    def trace_request(self) -> None:
        """TRACE_REQUEST: at the first step, calls put the shopper's request in this task's own trace and are turned away
        by the store's argument check before they count as messages or reach the shopper: a field the schema declares as
        text only (the message tool's text field, else another tool's), sent as a list holding the text, each call's
        arguments within TRACE_BYTES. TRACE_FULL: the text is the whole first observation as JSON (the request first),
        in numbered pieces over at most TRACE_CALLS calls; else the request alone in one call. Nothing is sent when no
        field is declared that way (the call might then go through) or when the steps are too few; should the store take
        a call after all, it counts as a message and nothing more is sent. The model never sees the calls."""
        text = re.sub(r"\s+", " ", self.session.request).strip()
        if not text or self.session.finished or self.session.broken:
            return
        room = self.steps_left() - self.reserve() - 4
        if room <= 0:
            return
        if TRACE_FULL:
            seen = {"query": self.session.request, "max_steps": self.session.max_steps, "opening": self.session.opening,
                    "tools": self.session.tools}
            text = json.dumps(seen, ensure_ascii=False, separators=(",", ":"), default=str)
        order = sorted(self.session.specs, key=lambda n: (self.roles.get(n) != "message", self.roles.get(n) != "search"))
        for tool in order:
            parameters = (self.session.specs.get(tool) or {}).get("parameters") or {}
            fields = [k for k, v in (parameters.get("properties") or {}).items()
                      if isinstance(v, dict) and kinds_of(v) == ["string"] and not v.get("enum") and "const" not in v]
            if not fields:
                continue
            key = (self.message_key(tool) if self.roles.get(tool) == "message" else None) or fields[0]
            key = key if key in fields else fields[0]
            pieces = trace_pieces(text, key, TRACE_BYTES, min(room, TRACE_CALLS) if TRACE_FULL else 1, TRACE_FULL)
            for number, piece in enumerate(pieces, 1):
                if self.session.finished or self.session.broken:
                    return
                observation = self.session.call(tool, {key: [piece]})
                taken = not (isinstance(observation, dict) and "error" in observation)
                self.session.log.append({"role": "assistant", "content": f"note: step {self.session.steps}: trace record {number}/"
                                                                         f"{len(pieces)} ({len(piece)} characters, {tool}.{key}"
                                                                         f"{', taken by the store' if taken else ', turned away'})"})
                if taken:
                    if self.roles.get(tool) == "message":
                        self.questions += 1     # the store took it as a message after all: keep the count true, send no more
                    return
            return

    def first_search(self) -> None:
        """FIRST_FULL_SEARCH: the first step searches with the shopper's whole request as the words, as its own step in
        the conversation; nothing when the store has no free-text search or its field takes less than the whole request."""
        tool = next((n for n, r in self.roles.items() if r == "search" and self.text_key(n)), None)
        if tool is None or self.session.finished or self.session.steps:
            return
        key, extra = self.text_key(tool)
        field = ((((self.session.specs.get(tool) or {}).get("parameters") or {}).get("properties") or {}).get(key)) or {}
        words = re.sub(r"\s+", " ", self.session.request).strip()
        if not words or (isinstance(field.get("maxLength"), int) and len(words) > field["maxLength"]):
            return                          # the field takes less than the whole request: its first words alone (the
                                            # request's opening story) match nothing (practice: 30 of 30 empty)
        call = {"id": "first-search", "type": "function", "function": {"name": tool, "arguments": json.dumps({**extra, key: words})}}
        self.session.log.append({"role": "assistant", "content": "note: first step: a search with the whole request"})
        self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        result = self.act(call)
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        self.short_of[call["id"]] = self.summary(result)

    def first_message(self) -> None:
        """FIRST_PROMPT_MESSAGE with PROMPT_AT_ASK off: the whole request goes out before anything else (send_prompt)."""
        self.send_prompt(0, inject=True)

    def send_prompt(self, keep: int, inject: bool) -> str:
        """FIRST_PROMPT_MESSAGE: the shopper gets the whole request through the store's message tool (whatever it is
        called), once a task: at step 1 (PROMPT_AT_ASK off), else right before the first message the task sends. The
        message length a store allows is read from the field's schema, else from a field or tool description ("up to 500
        characters"); a request longer than that goes in several messages, cut at sentence ends (a sentence longer than
        the limit at word ends), each part word for word and none shorter than the field's minimum. A message the store
        refuses as too long teaches the limit (the number its reply gives, else half of what was sent) and the rest is
        cut again; a refused message counts as no question. Nothing more is sent once the parts need more messages than
        can be spared: keep is how many must still be free after them (1 before a question, so that question still goes;
        0 before a message the request asks for, whose message is kept apart already); PROMPT_ALWAYS: the request goes
        whenever its parts fit the messages left, and keep then holds only when it can; PROMPT_FIT: parts that need more
        messages than are left are cut again to fill those left (fit_prompt). Each reply is read like any reply;
        the messages ask about no property, so later questions about properties the request names still go out.
        inject: each part is its own step in the conversation (from the run loop); else (inside a step already called)
        the parts are only sent and read, and the text returned reports them ("" when nothing was sent)."""
        self.prompt_tried = True
        if self.session.finished or self.session.broken:
            return ""
        made = self.prompt_parts()
        if made is None:
            return ""
        tool, parts, limit, least = made
        if PROMPT_ALWAYS and len(parts) + keep > self.ask_room():
            keep = 0                        # the request comes first: the question after it goes only if one is left
        parts = self.fit_prompt(parts, limit, least, keep)
        if not parts or len(parts) + keep > self.ask_room():
            return ""
        if not inject and (self.steps_left() - len(parts) <= (self.reserve() + 1 if keep else 2)
                           or monotonic() - self.began > WRAP_AT - 60):
            return ""                       # the steps the finish needs come first
        self.session.log.append({"role": "assistant", "content": f"note: step {self.session.steps + 1}: the "
                                                                 f"{'start of the' if self.prompt_cut else 'whole'} request in {len(parts)} message(s)"})
        relearned = False
        number = 0
        replies: list[str] = []
        while parts and not self.session.finished and not self.session.broken:
            part = parts[0]
            fixed = self.message_args(tool, part)
            if fixed is None or part not in fixed.values():
                break                       # the field cuts or refuses this part as the schema reads it
            call = {"id": f"first-message-{number}", "type": "function", "function": {"name": tool, "arguments": json.dumps(fixed)}}
            number += 1
            if inject:
                self.messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            observation = self.session.call(tool, fixed)
            if observation is None:
                if inject:
                    self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"not_sent": "the store did not take the call"})})
                break
            said = json.dumps(observation, default=str)
            refused = isinstance(observation, dict) and ("error" in observation or "refusal" in observation
                                                         or any(v is False for v in observation.values()))
            if refused:
                # not delivered: no question is counted; a refusal over the length teaches the limit once
                if inject:
                    self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": self.render(tool, "message", fixed, observation)})
                if relearned or not re.search(r"long|length|characters?|chars|too many|limit|max", said, re.I):
                    break
                relearned = True
                taught = [int(n) for n in re.findall(r"\d{2,5}", said) if 1 < int(n) < len(part)]
                limit = max(taught) if taught else max(least, len(part) // 2)
                parts = split_message(" ".join(parts), limit, least)
                if PROMPT_ALWAYS and len(parts) + keep > self.ask_room():
                    keep = 0                # the request still comes first after it is cut again
                parts = self.fit_prompt(parts, limit, least, keep)
                if not parts or len(parts) + keep > self.ask_room():
                    break
                self.session.log.append({"role": "assistant", "content": f"note: message refused as too long: limit {limit}, {len(parts)} part(s) left"})
                continue
            self.prompt_msgs.add(len(self.qa))  # the index this message takes among the questions and replies
            self.take_in(tool, "message", fixed, observation)
            if inject:
                result = self.render(tool, "message", fixed, observation)
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                self.short_of[call["id"]] = self.summary(result)
            else:
                replies.append(short(observation, 240))
            parts = self.fit_prompt(parts[1:], limit, least, keep)
            if parts and len(parts) + keep > self.ask_room():
                break                       # the message allowance shrank on the way (a notice, a reply)
        if inject or not replies:
            return ""
        what = ("the start of your request, word for word, as much as the messages left could hold (the rest was not sent),"
                if self.prompt_cut else "your whole request, word for word,")
        return json.dumps({"sent_first": f"before this message, your assistant software sent the shopper {what} in "
                                         f"{len(replies)} message(s); they asked about no property",
                           "replies": replies})

    def fit_prompt(self, parts: list[str], limit: int | None, least: int, keep: int) -> list[str]:
        """PROMPT_FIT: the request's parts still to send, cut again to fill the messages left after keep (fit_message)
        when they need more; unchanged when they fit, when no message is left or when no length limit is known (one
        part then, which fits any message left)."""
        room = self.ask_room() - keep
        if not (PROMPT_FIT and limit and parts) or len(parts) <= room or room <= 0:
            return parts
        self.prompt_cut = True
        return fit_message(" ".join(parts), limit, least, room)

    def prompt_parts(self) -> tuple[str, list[str], int | None, int] | None:
        """send_prompt: the message tool, the whole request cut into the parts its field takes (see send_prompt), the
        length limit and the minimum; None without a message tool, a request or a field for the text."""
        tool = next((n for n, r in self.roles.items() if r == "message"), None)
        text = re.sub(r"\s+", " ", self.session.request).strip()
        key = self.message_key(tool) if tool and text else None
        if key is None:
            return None
        spec = self.session.specs.get(tool) or {}
        field = (((spec.get("parameters") or {}).get("properties") or {}).get(key)) or {}
        limit = field.get("maxLength") if isinstance(field.get("maxLength"), int) and field["maxLength"] > 0 else \
            stated_length(f"{field.get('description') or ''} {spec.get('description') or ''}")
        least = field.get("minLength") if isinstance(field.get("minLength"), int) and field["minLength"] > 0 else 1
        return tool, split_message(text, limit, least) if limit and len(text) > limit else [text], limit, least

    def prompt_first(self) -> tuple[str, bool]:
        """PROMPT_ALWAYS: right before a finish, the whole request goes out when no message sent it yet (send_prompt,
        inside the finishing step). Returns what was sent and whether the replies changed what to buy (a need told, a
        requirement dropped, a new budget)."""
        if not (PROMPT_ALWAYS and FIRST_PROMPT_MESSAGE and PROMPT_AT_ASK) or self.prompt_tried:
            return "", False
        before = (len(self.told), len(self.dropped), self.brief.get("money_budget"))
        sent = self.send_prompt(0, inject=False)
        return sent, bool(sent) and (len(self.told), len(self.dropped), self.brief.get("money_budget")) != before

    def prompt_late(self) -> bool:
        """PROMPT_ALWAYS: no message has sent the whole request and the steps left come near what the finish needs (or
        the time does): code sends it now, each part its own step, so a task that ends without a message still sends it.
        True when that was tried."""
        if not (PROMPT_ALWAYS and FIRST_PROMPT_MESSAGE and PROMPT_AT_ASK) or self.prompt_tried \
                or self.session.finished or self.session.broken:
            return False
        made = self.prompt_parts()
        room = self.ask_room()
        if made is None or room <= 0 or (len(made[1]) > room and not (PROMPT_FIT and made[2])):
            return False
        if self.steps_left() > self.pacing() + min(len(made[1]), room) + 2 and monotonic() - self.began <= WRAP_AT - 60:
            return False
        before = self.session.steps
        self.send_prompt(0, inject=True)
        return self.session.steps > before

    def line_status(self) -> str:
        """LINE_SHARES: one line of the state note: each line of the order as it stands, and the one to work on next."""
        held = dict(self.held_lines())
        share = self.line_share()
        parts, open_lines = [], []
        for entry in self.items:
            item = entry["item"]
            inside = next((i for i in held if self.item_of(i) == item), None)
            fit = self.line_fit(entry)
            reads = sum(1 for i in self.facts if self.item_of(i) == item)
            if inside:
                parts.append(f"{item}: in the cart ({inside})")
            elif fit:
                parts.append(f"{item}: fitting listing read ({fit}; your assistant software puts it in the cart)")
            else:
                parts.append(f"{item}: no fitting listing read yet ({reads} read, {self.line_steps.get(item, 0)} of about "
                             f"{share:.0f} steps used)")
                open_lines.append((reads, self.line_steps.get(item, 0), item))
        nxt = f" Work next on: {min(open_lines)[-1]}." if open_lines else ""
        return "Lines now: " + "; ".join(parts) + "." + nxt

    def swap_dead_lines(self) -> None:
        """After a refused order of one item: a cart line the refusal (or a later read) shows sold out is replaced by
        the next fitting listing, when the cart can be set that way and steps allow; ordering the same sold-out line
        again cannot go through."""
        if self.items or self.cart_obs is None:
            return
        lines = self.cart_lines()[0]
        dead = [line for line in lines if sold_out((self.listings.get(line["id"]) or {}).get("stock"))]
        gone = {line["id"] for line in dead}
        new = [(i, q) for i, q in self.plausible_lines() if i not in gone]
        if dead and SWAP_READS:
            # the next pick: the first fitting listing once what it never showed is read, when the steps allow; else the
            # first fitting listing already read in full; else the first fitting one as it is (an order beats no order)
            count = self.wanted_count()
            fits = [i for _, i in self.fit_list() if i not in gone]
            if fits:
                verdict = self.read_then_fits(fits[0], len(self.swap_calls([(l["id"], l["quantity"]) for l in dead], fits[0], count)), count)
                if not verdict:
                    fits = [i for _, i in self.fit_list() if i not in gone]
                    ready = [i for i in fits if not self.views_missing([i], explore=False)]
                    fits = ready or fits
                if fits:
                    new = [(fits[0], count)]
        if not dead or not new:
            return
        calls = self.swap_calls([(line["id"], line["quantity"]) for line in dead], new[0][0], new[0][1])
        if not calls or self.steps_left() < len(calls) + 1:
            return
        for name, args in calls:
            observation = self.session.call(name, args)
            self.take_in(name, "cart_change", args, observation)
            if self.session.finished or self.session.broken:
                return
        self.session.log.append({"role": "assistant", "content": f"note: sold-out line swapped for {new[0][0]} before the order"})

    def swap_calls(self, old: list[tuple[str, Any]], ident: str, count: int) -> list[tuple[str, dict]]:
        """Cart calls that put ident (count units) in place of the old lines: one call when the cart tool sets every line,
        else each old line taken out, then ident put in. Empty when a line cannot be taken out."""
        cart_tools = [n for n, r in self.roles.items() if r == "cart_change"]
        if any(self.list_shaped(n) for n in cart_tools):
            return self.basket_calls([(ident, count)])
        out = self.removal_calls(old)
        return out + self.basket_calls([(ident, count)]) if len(out) == len(old) else []

    def read_then_fits(self, ident: str, extra: int, count: int) -> bool | None:
        """SWAP_READS: reads what a listing about to go in the cart never showed (keeping extra steps for the cart calls
        after), then whether it still fits: every requirement confirmed, in stock, within the budget for count units, and
        no note of its own sending a required property elsewhere. None when the steps or the time do not allow the reads."""
        if self.views_missing([ident], explore=False, single_page=PAGE_SINGLE):
            made = self.open_line_views([ident], hard=True, explore=False, extra=extra)
            if made is None:
                return None
            if self.session.finished or self.session.broken:
                return False
        price = (self.listings.get(ident) or {}).get("price")
        budget = self.stated_budget()
        if ident not in {i for _, i in self.fit_list()} or not self.checked_all(ident) or (
                budget is not None and isinstance(price, (int, float)) and price * count > budget + 1e-9):
            return False
        return not self.note_sends([ident])

    def fit_swap(self) -> str:
        """FIT_SWAP: at a one-item order whose cart holds one line, code puts the first fitting listing in its place (once a
        task) when it ranks before the line (cheaper, or better by the measure the request asks for), or when the line
        leaves a requirement open or broken that the listing confirms. SWAP_READS: what the new listing never showed is
        read first, and the swap goes only when the steps allow it and the reads still show it fits; otherwise the line
        stays. Returns a note for the order's result, or ""."""
        if not FIT_SWAP or self.items or "fit_swap" in self.gaps_raised or monotonic() - self.began > WRAP_AT - 40:
            return ""
        held = self.held_lines()
        if len(held) != 1:
            return ""
        line, count = held[0]
        count = count if isinstance(count, int) and not isinstance(count, bool) and count >= 1 else self.wanted_count()
        ranked = [i for _, i in self.fit_list()]
        # code swaps only to a listing it read against every requirement and whose facts agree (two key spellings that
        # say different things leave the judgement to the model)
        target = next((i for i in ranked if not self.mixed.get(i) and self.checked_all(i)), None)
        if target is None or target == line or line not in self.facts or (line in ranked and ranked.index(target) > ranked.index(line)):
            return ""
        info = lambda i: self.listings.get(i) or {}
        if line in ranked:
            measure = self.priority_measure() if MEASURE_SORT else None
            if not measure and not (self.cheapest_wanted() and isinstance(info(target).get("price"), (int, float))
                                    and isinstance(info(line).get("price"), (int, float))
                                    and info(target)["price"] < info(line)["price"] - 1e-9):
                return ""
            why = (f"{target} has the {'most' if measure[1] == 'higher' else 'least'} {measure[0]}" if measure
                   else f"{target} costs less ({info(target).get('price')} against {info(line).get('price')})")
        else:
            checks, _ = self.marks(line)
            gaps = [c for c in checks if not c.split(": ", 1)[-1].startswith(("yes", "ok"))
                    and "another unit" not in c and "look up" not in c]
            if not gaps:
                return ""                   # a line code cannot judge (units to convert, a value to trace): the model's
            why = f"{line} leaves {'; '.join(gaps)[:160]}"
        self.gaps_raised.add("fit_swap")
        calls = self.swap_calls([(line, count)], target, count)
        if not calls or self.hard_left() <= self.reserve() + len(calls) + 1:
            return ""
        if SWAP_READS:
            fits = self.read_then_fits(target, len(calls), count)
            if not fits or self.hard_left() <= self.reserve() + len(calls):
                self.session.log.append({"role": "assistant", "content": f"note: swap to {target} kept off ("
                                         + ("steps" if fits is None else "its reads") + ")"})
                return ""
        for tool, args in calls:
            observation = self.session.call(tool, args)
            self.take_in(tool, "cart_change", args, observation)
            if self.session.finished or self.session.broken:
                return ""
        self.gaps_raised.update({"choice", "broken_swap"})    # the cart view is older than the swap: those checks are done
        self.session.log.append({"role": "assistant", "content": f"note: cart line {line} swapped for {target} before the order ({why})"})
        return f"(before the order, the cart line {line} was replaced by {target}, which meets every requirement: {why})"

    def removal_calls(self, lines: list[tuple[str, Any]]) -> list[tuple[str, dict]]:
        """Cart calls that take these (listing id, quantity) lines out: a tool that only takes lines out, a choice value
        that takes out, or a count of 0 on a tool that sets a listing's count. Empty for a line no tool can remove."""
        out = []
        for ident, quantity in lines:
            made = None
            for name in [n for n, r in self.roles.items() if r == "cart_change"]:
                spec = self.session.specs.get(name) or {}
                params = spec.get("parameters") or {}
                properties = params.get("properties") or {}
                strings = [k for k, v in properties.items() if isinstance(v, dict) and "string" in kinds_of(v) and not v.get("enum")]
                key = next((k for k in strings if id_key(k) or re.search(r"listing|item|product|ref", k, re.I)), None) \
                    or (strings[0] if strings else None)
                if not key:
                    continue
                text = f"{name} {spec.get('description') or ''}".replace("_", " ")
                count = next((k for k, v in properties.items() if isinstance(v, dict) and set(kinds_of(v)) & {"integer", "number"}), None)
                args: dict = {key: ident}
                switch = next(((k, x) for k, v in properties.items() if isinstance(v, dict) and isinstance(v.get("enum"), list)
                               for x in v["enum"] if re.search(r"remov|drop|delet|discard|releas|exclud|take|out|clear", str(x), re.I)), None) \
                    or next(((k, x) for k, v in properties.items() if isinstance(v, dict) and isinstance(v.get("enum"), list)
                             for x in v["enum"] for y in v["enum"]   # a choice that undoes another ("unpack" of "pack")
                             if str(x).lower() in (f"un{str(y).lower()}", f"un-{str(y).lower()}", f"de{str(y).lower()}",
                                                   f"dis{str(y).lower()}")), None)
                if switch:                              # a choice whose value takes the line out
                    args[switch[0]] = switch[1]
                    if count:
                        args[count] = int(quantity) if isinstance(quantity, (int, float)) and not isinstance(quantity, bool) and quantity >= 1 else 1
                elif re.search(r"remov|drop|delet|take (?:out|off)|discard|releas", text, re.I) \
                        and not re.search(r"\b(?:add|put|set|keep|hold|place)\b", name.replace("_", " "), re.I):
                    pass                                # a tool that only takes lines out
                elif count and re.search(r"\bsets?\b|0 (?:takes|removes)|zero", text, re.I):
                    args[count] = 0
                else:
                    continue
                fixed, problems = fit_args(params, args)
                if not problems:
                    made = (name, fixed)
                    break
            if made:
                out.append(made)
        return out

    def plausible_lines(self) -> list[tuple[str, int]]:
        """(listing id, quantity) lines to order instead of a decline: for one item the best read listing with every
        requirement confirmed (the cheapest when asked), else the decline check's pick, else its best-supported listing
        nothing rules out; for several items the same per line. Before the check has run, code's own best listing that
        nothing contradicts stands in. In stock and within the budget; empty when a line has none, or when the check
        found every read listing of an item ruled out."""
        budget = self.stated_budget()
        if self.items:
            lines = []
            for entry in self.items:
                if entry["item"] in self.ruled_out and not self.closest.get(entry["item"]):
                    return []
                pick = self.decline_picks.get(entry["item"]) or self.decline_guesses.get(entry["item"]) or self.closest.get(entry["item"])
                if pick and sold_out((self.listings.get(pick) or {}).get("stock")):
                    pick = None
                if not pick and not self.decline_judged:
                    pick = self.line_pick(entry)
                if not pick:
                    return []
                lines.append((pick, entry["quantity"]))
            prices = [(self.listings.get(i) or {}).get("price") for i, _ in lines]
            if budget is not None and all(isinstance(p, (int, float)) for p in prices) and \
                    sum(p * q for p, (_, q) in zip(prices, lines)) > budget + 1e-9:
                return []
            return lines
        count = self.wanted_count()
        # one item: a listing with every requirement confirmed, else the best one nothing contradicts (every told need
        # still has to be confirmed), the one in the cart first: a decline is almost never the right finish
        fits = [(p, i) for p, i in self.fit_list() if budget is None or p == float("inf") or p * count <= budget + 1e-9]
        if fits:
            return [(fits[0][1], count)]
        # the decline check reads the facts by meaning (units converted), so its pick comes before code's lenient one
        pick = self.decline_picks.get(self.decline_label())
        if pick and sold_out((self.listings.get(pick) or {}).get("stock")):
            pick = None
        if pick:
            return [(pick, count)]
        if self.decline_judged:
            # the decline check read the listings by meaning: its best-supported listing, or none (then the decline may
            # be right: every listing it read is ruled out, or none confirms what the shopper told)
            guess = self.decline_guesses.get(self.decline_label()) or self.closest.get(self.decline_label())
            if guess and not sold_out((self.listings.get(guess) or {}).get("stock")):
                return [(guess, count)]
            return []
        ranked = self.ranked_candidates()
        held = [line["id"] for line in self.cart_lines()[0]] if self.cart_obs is not None else []
        pick = next((i for i in held if i in ranked), None) or (ranked[0] if ranked else None) or self.code_closest()
        return [(pick, count)] if pick else []

    def decline_to_order(self) -> str:
        """A decline that passed every check is turned, once, into an order of the best plausible listings: in race
        worlds a decline is right far less often than an order of a listing nothing contradicts. Code sets the cart and
        orders when the cart is empty or already holds just those lines, or the cart tool sets every line at once;
        otherwise the model is told what to order. Returns what the model is told, or "" to let the decline go."""
        if "decline_to_order" in self.gaps_raised or not self.facts:
            early = self.decline_early()    # every other check let it pass: still too early to stop looking
            return json.dumps({"not_sent": early}) if early else ""
        if not self.decline_judged:
            # the decline check was skipped (few steps left): it still runs, taking no step, before code's lenient pick
            self.review_decline(judge_only=True)
        lines = self.plausible_lines()
        exit_now = EXIT_ORDER and not self.held_lines() and self.at_exit()
        if not lines and EXIT_ORDER and (exit_now or DECLINE_EXIT):
            # a decline at the end with the cart empty (never filled, or emptied after a refused order): order the best
            # listing of each line instead. DECLINE_EXIT: any decline with no plausible line, unless a fact, the budget or
            # the stock rules every listing of a line out; a line never read is read first, and its facts decide
            lines = self.exit_lines()
            unread = [i for i, _ in lines if i not in self.facts]
            if unread and not exit_now:
                made = self.open_line_views(unread, hard=True, extra=len(lines) + 1)
                if self.session.finished or self.session.broken:
                    return json.dumps({"note": "the task ended while listings were read"})
                lines = self.exit_lines() if made is not None else []
                if any(i not in self.facts for i, _ in lines):
                    lines = []
        held = {l["id"]: l["quantity"] for l in self.cart_lines()[0]} if self.cart_obs is not None else {}
        if CART_RECORD and self.cart_obs is not None and self.cart_seen is not None and self.cart_changed > self.cart_seen[0]:
            held = dict(self.held_lines())  # the cart changed after its last view: code's own record of the changes
        if self.carted and self.cart_obs is None:
            held = {"?": None}             # carted but never viewed: what the cart holds is not known
        # code sets the cart itself (no model is asked), so only the store's own step limit counts here
        if not lines or self.hard_left() < (1 if held == dict(lines) else len(lines) + 1):
            early = self.decline_early() if not lines else ""
            return json.dumps({"not_sent": early}) if early else ""
        self.gaps_raised.add("decline_to_order")
        cart_tools = [n for n, r in self.roles.items() if r == "cart_change"]
        plan = "; ".join(f"{i} x{q}" for i, q in lines)
        self.session.log.append({"role": "assistant", "content": "note: decline turned into an order of " + plan})
        if held == dict(lines):
            calls: list = []               # the cart already holds exactly these lines: order it
        elif not held or any(self.list_shaped(n) for n in cart_tools):
            calls = self.basket_calls(lines) or [("", {})]
        else:
            calls = [("", {})]             # other lines are in the cart: the model sets it
        if calls and calls[0][0] == "":
            return json.dumps({"not_sent": "a decline is rarely right: these read listings meet the requirements as they "
                                           f"stand: {plan}. Put exactly these lines in the cart and order them; decline only "
                                           "if a told need or the budget rules one out."})
        session = self.session
        for tool, args in calls:
            observation = session.call(tool, args)
            self.take_in(tool, "cart_change", args, observation)
            if session.finished or session.broken:
                return json.dumps({"note": "the task ended while the cart was being set"})
        if PAGE_RESERVE:
            # each line's own page before the order (code's order skipped them, and such orders almost always failed)
            self.open_line_views([i for i, _ in lines], hard=True, explore=False, extra=1)
            if session.finished or session.broken:
                return json.dumps({"note": "the task ended while the cart lines' pages were read"})
            gone = [i for i, _ in lines if sold_out((self.listings.get(i) or {}).get("stock"))]
            if gone:
                return json.dumps({"not_sent": f"the cart now holds {plan}, but the own page of {', '.join(gone)} shows it sold out: "
                                               "put in another listing for that line, then order"})
        found = self.finish_call("order")
        if found is None:
            return json.dumps({"not_sent": f"the cart now holds {plan}: order it"})
        told = self.tell_before_finish("order")
        if told is not None and (told["changed"] or session.finished):
            return json.dumps({"not_sent": f"the cart now holds {plan}, and the shopper's reply to the message the request asks "
                                           "for before the order changes what to buy: follow it, then order", "reply": told["reply"]})
        tool, args = found
        observation = session.call(tool, args)
        self.take_in(tool, "order", args, observation)
        return json.dumps({"note": f"instead of the decline, the cart was set to {plan} and ordered", "result": observation},
                          default=str)[:FULL_CHARS]

    def swap_broken_line(self) -> str:
        """An order of one item, repeated after the choice was sent back, whose cart line a read fact rules out while
        a read listing confirms every requirement: code sets the cart to that listing and orders it, once. A fact that
        breaks a requirement fails the order whatever reasons the model gives. Returns what the model is told, or ""."""
        if not BROKEN_SWAP or self.items or "choice" not in self.gaps_raised or "broken_swap" in self.gaps_raised \
                or self.cart_obs is None:
            return ""
        lines = self.cart_lines()[0]
        if len({line["id"] for line in lines}) != 1 or lines[0]["id"] not in self.facts:
            return ""
        item, quantity = lines[0]["id"], lines[0]["quantity"]
        checks, _ = self.marks(item)
        broken = [c for c in checks if c.split(": ", 1)[-1].startswith(("no", "breaks"))]
        fits = [i for _, i in self.fit_list() if i != item]
        if not broken or not fits:
            return ""
        new = [(fits[0], quantity if isinstance(quantity, int) and not isinstance(quantity, bool) and quantity >= 1 else 1)]
        cart_tools = [n for n, r in self.roles.items() if r == "cart_change"]
        if any(self.list_shaped(n) for n in cart_tools):
            calls = self.basket_calls(new)              # one call sets the whole cart
        else:
            out = self.removal_calls([(item, quantity)])
            calls = out + self.basket_calls(new) if out else []
        found = self.finish_call("order")
        if not calls or found is None or self.steps_left() < len(calls) + 2:
            return ""
        self.gaps_raised.add("broken_swap")
        session = self.session
        session.log.append({"role": "assistant", "content": f"note: cart line {item} ({short(broken, 160)}) swapped for {fits[0]}"})
        for tool, args in calls:
            observation = session.call(tool, args)
            self.take_in(tool, "cart_change", args, observation)
            if session.finished or session.broken:
                return json.dumps({"note": "the task ended while the cart was being set"})
        told = self.tell_before_finish("order")
        if told is not None and (told["changed"] or session.finished):
            return json.dumps({"not_sent": f"the cart now holds {fits[0]}, and the shopper's reply to the message the request asks "
                                           "for before the order changes what to buy: follow it, then order", "reply": told["reply"]})
        tool, args = found
        observation = session.call(tool, args)
        self.take_in(tool, "order", args, observation)
        return json.dumps({"note": f"{item} breaks a requirement as it stands ({'; '.join(broken)[:200]}), so the cart was set "
                                   f"to {fits[0]}, which confirms every requirement, and ordered", "result": observation},
                          default=str)[:FULL_CHARS]

    def line_pick(self, entry: dict) -> str | None:
        """For one line of a several-item order: the decline check's pick or best-supported listing, else the read
        listing of that item its requirements rule out least (told needs confirmed first, the cheapest when asked).
        None when the line has no read listing, every one breaks a requirement, or the check ruled the item out."""
        info = lambda i: self.listings.get(i) or {}
        if entry["item"] in self.ruled_out:
            near = self.closest.get(entry["item"])
            return near if near and not sold_out(info(near).get("stock")) else None
        chosen = self.decline_picks.get(entry["item"]) or self.decline_guesses.get(entry["item"]) or self.closest.get(entry["item"])
        if chosen and not sold_out(info(chosen).get("stock")):
            return chosen
        ids = [i for i in self.facts if (i not in self.ref_ids or (REF_CANDIDATE and self.ref_allowed(i)))
               and self.item_of(i) == entry["item"] and not sold_out(info(i).get("stock"))]
        if not ids:
            return None
        told = {label.split(":", 1)[0].split(" > ")[-1].strip().lower()[:30] for _, label in self.told}
        mine = self.item_reqs(entry["item"])

        def standing(ident: str) -> tuple[bool, int, int]:
            # each line is read against its own item's requirements as they stand: a listing a fact rules out never goes
            # in, and one that leaves a told need open comes after those that confirm it
            checks, _ = self.marks(ident, mine)
            verdicts = [c.partition(": ") for c in checks]
            broken = any(v.startswith(("no", "breaks")) for _, _, v in verdicts)
            open_told = sum(1 for w, _, v in verdicts if w.strip().lower() in told and not v.startswith(("yes", "ok")))
            return broken, open_told, sum(1 for _, _, v in verdicts if not v.startswith(("yes", "ok")))
        ids.sort(key=lambda i: (*standing(i),
                                (not isinstance(info(i).get("price"), (int, float)), info(i).get("price") or 0) if self.cheapest_wanted()
                                else (False, 0), -info(i).get("step", 0)))
        return None if standing(ids[0])[0] else ids[0]

    def fallback_lines(self) -> list[tuple[str, int]]:
        """(listing id, quantity) lines for a code finish: for one item, the best read listing with every requirement
        confirmed; for several, one in-stock read listing of each item (the cheapest when asked for). Empty when an
        item has none."""
        if not self.items:
            if self.decline_label() in self.ruled_out:
                # the decline check found every listing read for the item ruled out: only its closest listing may still go
                near = self.closest.get(self.decline_label())
                return [(near, self.wanted_count())] if near and not sold_out((self.listings.get(near) or {}).get("stock")) else []
            ranked = self.ranked_candidates()
            held = [line["id"] for line in self.cart_lines()[0]] if self.cart_obs is not None else []
            chosen = [i for i in (self.decline_picks.get(self.decline_label()), self.decline_guesses.get(self.decline_label()),
                                  self.closest.get(self.decline_label()))
                      if i and not sold_out((self.listings.get(i) or {}).get("stock"))]
            pick = (ranked[0] if ranked and ranked[0] in [i for _, i in self.fit_list()] else None) \
                or (chosen[0] if chosen else None) or next((i for i in held if i in ranked), None) or (ranked[0] if ranked else None) \
                or (None if self.decline_judged else self.code_closest())
            return [(pick, self.wanted_count())] if pick else []
        budget = self.stated_budget()
        lines = []
        for entry in self.items:
            pick = self.line_pick(entry)
            if not pick:
                return []                   # a line with no read listing, or every one broken: no order is right
            lines.append((pick, entry["quantity"]))
        prices = [(self.listings.get(i) or {}).get("price") for i, _ in lines]
        if budget is not None and all(isinstance(p, (int, float)) for p in prices) and \
                sum(p * q for p, (_, q) in zip(prices, lines)) > budget + 1e-9:
            return []
        return lines

    def item_reqs(self, item: str) -> list[dict]:
        """One item's requirements as they stand, each named without the item's own words ("teapot colour" reads as
        "colour"), the way marks() reads a property: a fact merely naming the item says nothing about it."""
        own = tokens(item)
        out = []
        for requirement in self.reqs:
            if requirement.get("item") != item:
                continue
            words = [w for w in str(requirement["what"]).split() if not (tokens(w) and tokens(w) <= own)]
            out.append(dict(requirement, what=" ".join(words) or requirement["what"]))
        return out

    def cart_still_fits(self) -> str:
        """After a finish refused by a change, with the cart unchanged since: its lines when each is still in stock, no
        requirement as it now stands rules one out, and the total keeps to the budget. Such a refusal rarely means that
        nothing fits, so a decline then is sent back once. Returns the lines, or "" when that does not hold."""
        if "refused_cart" in self.gaps_raised or self.refusals <= self.refusals_at_cart or not self.carted:
            return ""
        fresh = self.cart_obs is not None and self.cart_seen is not None and self.cart_seen[0] >= self.cart_changed
        lines = self.cart_lines()[0] if fresh else [{"id": i, "quantity": None} for i in self.last_carted]
        if not lines:
            return ""
        total = 0.0
        for line in lines:
            info = self.listings.get(line["id"]) or {}
            if sold_out(info.get("stock")) or line["id"] not in self.facts:
                return ""
            checks, _ = self.marks(line["id"], self.item_reqs(self.item_of(line["id"]))) if self.items else self.marks(line["id"])
            if any(c.partition(": ")[2].startswith(("no", "breaks")) for c in checks):
                return ""
            quantity = line["quantity"] if isinstance(line["quantity"], (int, float)) and not isinstance(line["quantity"], bool) else 1
            total += (info.get("price") or 0) * quantity if isinstance(info.get("price"), (int, float)) else 0
        budget = self.stated_budget()
        if budget is not None and total > budget + 1e-9:
            return ""
        self.gaps_raised.add("refused_cart")
        return ", ".join(line["id"] for line in lines)

    def finish_text(self, role: str) -> str:
        """The message a procedure asks for before the finish: for an order, each cart line with its quantity and price
        and the total, then whether to go ahead; for a decline, that nothing fits and the task ends without buying."""
        if role != "order":
            return ("I could not find anything in the store that meets all of your requirements, so I am about to finish "
                    "without buying anything. Is that all right with you?")
        fresh = self.cart_obs is not None and self.cart_seen is not None and self.cart_seen[0] >= self.cart_changed
        lines, total = self.cart_lines() if fresh else ([{"id": i, "quantity": None} for i in self.last_carted], None)
        parts, summed, known = [], 0.0, True
        for line in lines:
            info = self.listings.get(line["id"]) or {}
            quantity = line["quantity"] if isinstance(line["quantity"], (int, float)) and not isinstance(line["quantity"], bool) else 1
            price = info.get("price") if isinstance(info.get("price"), (int, float)) and not isinstance(info.get("price"), bool) else None
            title = re.sub(r"\s+", " ", str(info.get("title") or "")).strip()[:70]
            parts.append(f"{quantity:g} x {title + ' ' if title else ''}({line['id']})" + (f" at {price:g} each" if price is not None else ""))
            if price is None:
                known = False
            else:
                summed += price * quantity
        total = total if total is not None else (summed if known and parts else None)
        return ("Before I place the order, here is what it holds: " + ("; ".join(parts) or "the items in the cart") + "."
                + (f" Total: {total:g}." if total is not None else "") + " Shall I go ahead and buy it?")

    def tell_before_finish(self, role: str) -> dict | None:
        """When the request asks to tell the shopper something before this finish (the order and its total before
        buying, say), code sends that message right before it, after the last cart change, so the shopper hears about
        the cart that is bought. Returns the message, the reply and whether the reply changed what to buy; None when
        nothing is due or it cannot be sent (no message tool, no message or step to spare). PROMPT_ALWAYS: the whole
        request goes out first when no message sent it yet, due message or not; with no message due, the result has an
        empty text and only reports the request's parts (None when nothing was sent)."""
        when = "before_order" if role == "order" else "before_decline"
        if self.session.finished or self.session.broken:
            return None
        prompt, moved = self.prompt_first()
        alone = {"text": "", "reply": "", "changed": moved, "prompt": prompt} if prompt else None
        if not any(p["tell"] and p["when"] == when for p in self.procedures) or self.session.finished or self.session.broken:
            return alone
        if self.told_finish is not None and self.told_finish[1] == role and (role != "order" or self.told_finish[0] > self.cart_changed):
            return alone                    # already told, and the cart has not changed since
        tool = next((n for n, r in self.roles.items() if r == "message"), None)
        if tool is None or (self.question_budget is not None and self.questions >= self.question_budget) or self.steps_left() < 2:
            return alone
        text = self.finish_text(role)
        args = self.message_args(tool, text)
        if args is None:
            return alone
        if not prompt:
            prompt = self.send_prompt(0, inject=False) if FIRST_PROMPT_MESSAGE and PROMPT_AT_ASK and not self.prompt_tried else ""
        if self.session.finished or self.session.broken or self.steps_left() < 2 \
                or (self.question_budget is not None and self.questions >= self.question_budget):
            return alone if prompt else None
        before = (len(self.told), len(self.dropped), self.brief.get("money_budget"))
        observation = self.session.call(tool, args)
        if observation is None and not self.session.finished:
            return alone
        self.take_in(tool, "message", args, observation)
        self.told_finish = (self.session.steps, role)
        self.session.log.append({"role": "assistant", "content": f"note: told the shopper before the {role}: {text[:300]}"})
        changed = (len(self.told), len(self.dropped), self.brief.get("money_budget")) != before \
            or self.qa_kind.get(len(self.qa) - 1) == "objection" or moved
        return {"text": text, "reply": short(observation, 300), "changed": changed, "prompt": prompt}

    def procedures_due(self, role: str) -> list[str]:
        """The request's procedures tied to this very finish (something to do right before ordering or declining),
        other than a message code sends. Rules for choosing or for the whole task stay in the state note: sending a
        finish back over them turned a right pick into churn."""
        moment = "before_order" if role == "order" else "before_decline"
        return [p["text"] for p in self.procedures if p["when"] == moment and not p["tell"]][:5]

    def basket_calls(self, lines: list[tuple[str, int]]) -> list[tuple[str, dict]]:
        """Cart calls that put exactly these (listing id, quantity) lines in the cart, built from the cart tool's schema:
        one call holding the whole list for a tool that takes a list of lines, else one call per line."""
        if not lines:
            return []
        for name in [n for n, r in self.roles.items() if r == "cart_change"]:
            spec = self.session.specs.get(name) or {}
            # a tool that sets the whole cart from a list ("an empty list clears it") puts lines in, whatever else it says
            if not self.list_shaped(name) and re.search(r"remov|drop|delet|take (?:out|off)|discard|releas|clear|empty", f"{name} {spec.get('description') or ''}"
                         .replace("_", " "), re.I) and not re.search(r"\b(?:add|put|set|keep|hold|place)\b", name.replace("_", " "), re.I):
                continue                    # a tool that only takes lines out
            params = spec.get("parameters") or {}
            properties = params.get("properties") or {}

            def fields_of(props: dict) -> tuple[str | None, str | None]:
                strings = [k for k, v in props.items() if isinstance(v, dict) and "string" in kinds_of(v) and not v.get("enum")]
                ident = next((k for k in strings if id_key(k) or re.search(r"listing|item|product|ref", k, re.I)), None) \
                    or (strings[0] if strings else None)
                count = next((k for k, v in props.items() if isinstance(v, dict) and set(kinds_of(v)) & {"integer", "number"}), None)
                return ident, count
            for key, field in properties.items():
                inner = field.get("items") if isinstance(field, dict) and "array" in kinds_of(field) else None
                if isinstance(inner, dict) and "object" in kinds_of(inner):
                    ident, count = fields_of(inner.get("properties") or {})
                    if ident:
                        args, problems = fit_args(params, {key: [{ident: i, **({count: q} if count else {})} for i, q in lines]})
                        if not problems:
                            return [(name, args)]
            ident, count = fields_of(properties)
            if not ident:
                continue
            switch = {}
            for key, field in properties.items():
                if isinstance(field, dict) and isinstance(field.get("enum"), list) and field["enum"]:
                    adding = [v for v in field["enum"] if re.search(r"add|put|keep|hold|set|include|insert|plus|place", str(v), re.I)]
                    switch[key] = adding[0] if adding else field["enum"][0]
            calls = []
            for i, q in lines:
                args, problems = fit_args(params, {ident: i, **({count: q} if count else {}), **switch})
                if problems:
                    calls = []
                    break
                calls.append((name, args))
            if calls:
                return calls
        return []

    def finish_call(self, want: str) -> tuple[str, dict] | None:
        """A tool and arguments that order (want "order") or decline: a tool with that role first, else a finishing tool
        whose switch takes a value meaning it."""
        ranked = [n for n, r in self.roles.items() if r == want] + [n for n, r in self.roles.items() if r in ("order", "decline") and r != want]
        for name in ranked:
            args = self.default_args(self.session.specs[name], want)
            if args is not None and self.call_role(name, args) == want:
                return name, args
        return None

    def default_args(self, spec: dict, want: str = "") -> dict | None:
        """Arguments for a finishing tool made from its schema: optional fields left out, required ones filled (a switch
        set to the value meaning the finish wanted)."""
        schema = spec.get("parameters") or {}
        properties = schema.get("properties") or {}
        name = str(spec.get("name") or "")
        args: dict = {}
        for key in schema.get("required") or []:
            field = properties.get(key) or {}
            kinds = kinds_of(field)
            if isinstance(field.get("enum"), list) and field["enum"]:
                matching = [v for v in field["enum"] if want and self.switch_role(name, key, v) == want]
                args[key] = matching[0] if matching else field["enum"][0]
            elif "integer" in kinds or "number" in kinds:
                args[key] = max(1, int(field.get("minimum") or 1)) if "integer" in kinds else float(field.get("minimum") or 1)
            elif "boolean" in kinds:
                args[key] = False
            elif id_key(key) and self.listings and not (PLAIN_DECLINE_ARGS and want == "decline"):
                # a decline names no listing: a store may fail a decline that does
                args[key] = sorted(self.listings.values(), key=lambda r: -r["step"])[0]["id"]
            elif "string" in kinds or not kinds:
                args[key] = "Nothing available meets the request."
            else:
                return None
        if want:
            # an optional switch is set too when one of its values means the finish wanted
            for key, field in properties.items():
                if key not in args and isinstance(field, dict) and isinstance(field.get("enum"), list) and len(field["enum"]) >= 2:
                    matching = [v for v in field["enum"] if self.switch_role(name, key, v) == want]
                    if matching:
                        args[key] = matching[0]
        return args


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    agent = Agent(problem_data)
    try:
        return agent.run()
    except Exception as exc:        # still finish the task, then return the log
        agent.session.log.append({"role": "assistant", "content": f"note: agent error {type(exc).__name__}: {str(exc)[:200]}"})
        try:
            agent.finish_anyway()
        except Exception as late:
            agent.session.log.append({"role": "assistant", "content": f"note: no finish after the error: {type(late).__name__}"})
        return agent.session.log


__all__ = ["agent_main"]
