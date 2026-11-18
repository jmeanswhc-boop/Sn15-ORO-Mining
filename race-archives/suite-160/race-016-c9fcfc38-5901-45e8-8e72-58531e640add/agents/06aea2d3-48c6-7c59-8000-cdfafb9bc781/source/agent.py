from __future__ import annotations

import concurrent.futures
import itertools
import json
import re
import time
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient

MAIN_MODEL = "anthropic/claude-haiku-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"
SETUP_MODEL = "anthropic/claude-sonnet-5.5"
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
READ_MODEL = "anthropic/claude-haiku-5.5"
VALUE_MODEL = "anthropic/claude-sonnet-5.5"
# pointer resolution, the first question while no property menu exists, and the later questions: on the recorded prompts it took
# the strong model's action on 358 of 363 pointer prompts (all 107 page redirects) at about half the price
REFERENCE_MODEL = "google/gemini-3.8-flash"

# Both models may reason before they answer, and reasoning tokens count against max_tokens, so every cap leaves room for a
# few thousand of them on top of the visible answer (caps sized for the answer alone cut 17 of 90 setups and 23% of the
# question plans on the race-2 replica). Only generated tokens are billed.
SETUP_TOKENS = 6000
JUDGE_TOKENS = 4000
NAV_TOKENS = 6000

# Request settings a model needs beyond the defaults. The reference model always reasons (it cannot be switched off), at about
# 80 tokens a second; its token cap keeps one call within a judgement's time (ONE_CALL_SECONDS) instead of the minute a
# 4000-token run of reasoning took (on 363 recorded pointer prompts the cap cut 3 runs of reasoning short, and each still gave
# its answer), and a call cut off without an answer goes on to the next model of its route.
MODEL_SETTINGS = {REFERENCE_MODEL: {"reasoning": {"effort": "low"}}}
# Request settings of the setup reading only (merged over MODEL_SETTINGS for that one call). The setup model reasons by default
# (on 180 recorded setups: 169 reasoned, 948 tokens on average, 56% of the call's output and about $0.010 of its $0.022); an
# effort setting is the only control the endpoint takes ("reasoning is mandatory ... cannot be disabled" rejects the off
# switch, and a token budget cannot go below the 1024 it averages), and on 20 recorded setups replayed with it the reading
# reasoned 0 tokens and cost $0.0087 against $0.0218, with the same budgets in 20/20 and products in 18/20.
SETUP_SETTINGS = {SETUP_MODEL: {"reasoning": {"effort": "low"}}}
MODEL_TOKENS = {REFERENCE_MODEL: 1500}

# Dollars per token, as OpenRouter lists them (2026-10-26): prompt, cache read, cache write, completion, then the prompt size
# from which a long-prompt rate applies and its factor on all four (0: none). A model not listed is priced at the dearest rates.
PRICES = {
    "anthropic/claude-haiku-5.5": (1e-7, 1e-8, 1.25e-7, 5e-7, 100000, 5.0),
    "anthropic/claude-sonnet-5.5": (2e-6, 1e-7, 2.5e-6, 1e-5, 0, 1.0),
    "openai/gpt-6-luna": (1e-7, 1e-8, 1.25e-7, 5e-7, 272000, 2.0),
    "google/gemini-3.8-flash": (8.25e-7, 8.25e-8, 4.17e-8, 4.125e-6, 0, 1.0),
}
# A task's spend, priced from the token counts of its answered calls, is booked by phase inside one cap (its budget, set_cap): the setup
# reading (its own phase, spent first), the planner's questions and the reading of their answers, the reserve (instruction
# discharge and the reading of what the finish brings back), and exploration (the navigator, the checks, pointer judgements:
# everything else). Exploration is owed EXPLORE_FLOOR of its own spend whatever the setup cost: until it has had it, and
# while a product has no opened listing, the planned questions leave it that much (late_room), and the search goes on.
# Exploration stops at the cap less the core reserve (the finish, the instructions that need a message or a tool, and the
# standing checks the gate decides), or before a navigator turn whose expected price (forecast) could cross that edge, or, once
# it has had its floor and every product has an opened listing, at the cap less the core reserve and the questions still
# planned. Any other exploration call (a check, a pointer judgement) that could cross the edge is skipped by itself, and the
# search goes on. Past the stop the planned questions still go out while they fit beside the core reserve, the listing checks
# the finish needs are paid beside the whole core reserve, and the reserve up to the cap and the overrun; a call started with
# the core reserve inside the cap that cost more than its forecast (a prompt the provider did not read from its cache) moves
# that edge to its spend and the core reserve then owed (reserve_cap). No call starts once that is reached.
# Every constant below is measured on the recorded e_w20 runs of the race-6 and race-7 replicas (180 tasks, 5,642 calls), the
# reserve and the forecast on run local-c7dd8d14 of the race-7 replica (89 tasks, 2,121 calls).
TASK_CAP = 0.050           # the least budget of a task (a one-product task of 43 steps needs about $0.044: it keeps this)
EXPLORE_FLOOR = 0.016      # exploration a task that was never stopped needed: p90 $0.016 (median $0.010); a listing of every product opened: p90 $0.018 for three products
QUESTION_COST = 0.0039     # one planned question with the reading of its answer: mean of 177 tasks (median $0.0032, p90 $0.0070)
QUESTION_SPEND = 0.016     # the most the planner's share keeps for the questions still planned: p90 of a task's planner spend
# the core reserve: what a task's executor calls, finish check and the readings of what they bring back cost, by the number of
# instructions and standing checks it carries: median $0.0012 with one, $0.0025 with two, $0.0032 with three (p90 $0.0014,
# $0.0035, $0.0040); a sum of per-call p90s kept twice what was used ($0.0055 against $0.0025 at the median stop)
INSTRUCTION_SPEND = 0.0013 # kept per instruction that needs a message or a tool (8 of 89 tasks used more than the reserve, by $0.0007 at most)
FINISH_SPEND = 0.0006      # kept for the finish check and the reading of what it brings back
CHECK_SPEND = 0.0013       # kept per standing check (no message, no tool: an availability check) the gate's executor decides
RESERVE_OVERRUN = 0.002    # a call the reserve pays for may run this far past the cap: a finish the reserve fell short of is never cut (at most $0.18 a run)
EXPLORE_TOKENS = 1375      # answer tokens a navigator turn (and any exploration call of no listed size) is priced with before it starts: p90 of 1,148 navigator answers, race-7 replica run local-677607f0 (median 550)
SHORT_ANSWER_TOKENS = 240  # the same for a listing check with a short answer (which product, stated values, variant fields, the model check): p90 179, 103, 191, 235
PROMPT_CHARS = 2.6         # characters of a request (messages and tools, as JSON) per prompt token: 2.64 on 797 navigator prompts (2.6-3.4 on the other calls)
# A task's budget follows its own step limit and products (set_cap, once the setup has read them): BUDGET_SHARE of its need
# (what the setup spent, its planned questions, its core reserve, and SEARCH_STEP_SPEND for each step left to the search: the
# step limit less the messages, the instructions carried out with a message or a tool, and ENDGAME_STEPS), never less than
# TASK_CAP. Measured on race-7 replica run local-64d84e08 (90 tasks); the share keeps every known pack (races 5-8, 14-30
# tasks of more than 60 steps each) at $4.41 a run or less expected and $4.86 or less if every task spent its whole budget.
SEARCH_STEP_SPEND = 0.00061  # one search step (a navigator turn with its checks and readings): median of 90 tasks (p75 $0.00087)
FIRST_QUESTION_SPEND = 0.0061  # the first planned question with the reading of its answer (median)
NEXT_QUESTION_SPEND = 0.0023   # each later one
ENDGAME_STEPS = 3          # the cart, the order and its notice: steps the search does not use
BUDGET_SHARE = 0.65        # of the need: on the race-7 replica a five-product task of 141-153 steps gets $0.070-0.083 and can search about 64 of its steps (about 31 at $0.050)
# A request that states no step limit may still meet a store that stops earlier than the policy allows (race 5: 21 of 90 tasks,
# hidden caps of 40 or 82; every pass on them used at most 40 steps): with a verified choice ready, the code orders it by then.
SOFT_STEPS = 40
TASK_SECONDS = 265.0
DRY_SEARCHES = 3           # searches in a row that added no row before the next is held once (at two, 42% of 261 holds on recorded runs would have added rows; the note after the first one already steers)
REVIEW_READS = 12          # store reads the review of a close takes at most (12 rows reach 21 of 41 tasks lost with the right row shown)
SWEEP_STEPS = 8            # steps before the commit that go to the store's own rows while a product has no proven fit in the budget (a rule's choice also): 8 cheapest rows reach 14 of 41 tasks lost with the right row shown, 12 reach 21
LINK_ROUNDS = 3            # the code's own looks for a listing that goes with a chosen hub (a store's search or opening costs no model spend)
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
OBS_CHARS = 16000
MAX_TURNS = 80

# The phases of a task, in steps left, derived from its step record (Episode.th): the finish takes one step (FINISH_ONLY 1,
# LAST 2 are fixed); every other phase begins this many steps above the steps the endgame needs, which are the cart calls one
# fresh basket takes (one for a basket tool, one per product otherwise) plus one per instruction still to carry out with a
# message or a tool. For a one-product task with nothing to carry out these are the numbers named below them.
LADDER = {
    "cart_only": 2,       # only cart and finishing tools; a refused order is placed again by the code
    "repair": 3,          # about what holding a call back costs: a refused order's repair, a reply, a look at cheaper listings
    "cheaper_open": 3,    # past reading_ends the details tool stays open while cheaper unchecked rows remain, with more steps left than
                          # this plus 2 per further product: measured on the r6 and r7 replica runs (commit record to the finish, p90 by
                          # products) 4 steps for 1 product, 7 for 2, 9 for 3, 13 for 5; with a basket tool and a notice the endgame is 2
                          # steps, so 3 + 2 = 5 left after the opening step keeps the one-product p90 plus one spare, 2 per product more
    "finish_now": 4,      # the status says FINISH NOW
    "commit": 5,          # with no order attempted, the code orders its best candidates; a decline is no longer questioned
    "reading_ends": 6,    # search, details and messages end (unless a product still has no listing)
    "questions_end": 9,   # planned questions stop with this many steps left (or a quarter of the steps, at least 4)
    "question_hold": 10,  # an order waits for one more planned question only with more steps than this
    "ready_commit": 11,   # with a verified choice ready and no question left, the code orders it, leaving room to repair a refused order
    "low": 13,            # the status warns that steps run low; planned questions stop pacing themselves
}
LADDER_FULL = 30          # a step budget this large keeps the whole ladder; a smaller one compresses its upper phases
QUESTIONS_START_MAX = 10  # by this step (a quarter of the step budget when that is less, at least 4) planned questions go out even before two listings are opened; a multi-product question waits as long again for its product's listings
# the ladder of a one-product task with nothing to carry out (what the notes and tests quote)
FINISH_ONLY_STEPS = 1     # only the order or the decline can be sent
LAST_STEPS = 2            # the status says LAST STEPS, and no check holds an order back any more
CART_ONLY_STEPS, REPAIR_STEPS, FINISH_NOW_STEPS, COMMIT_STEPS, READING_ENDS_STEPS, QUESTIONS_END_STEPS, QUESTION_HOLD_STEPS, READY_COMMIT_STEPS, LOW_STEPS = (
    LADDER[k] + 1 for k in ("cart_only", "repair", "finish_now", "commit", "reading_ends", "questions_end", "question_hold", "ready_commit", "low"))
QUESTIONS_START_STEP = PRODUCT_WAIT_STEPS = QUESTIONS_START_MAX
DEFAULT_MESSAGES = 3       # messages allowed when the request states no number
MENU_FITS = 2              # a product's question menu counts only its listings that meet the requirements once this many do
MENU_VALUE_CHARS = 70      # a fact value longer than this enters the menu only as a list of shorter values
MENU_PART_CHARS = 40       # ... and only when every one of them is a label's length
MENU_LIST_VALUES = 12      # the most values one listed fact adds to the menu

# A requirement whose value another listing gives (a chain of pages, each naming the next listing by its shop and details): the
# code finds each listing itself, by a search of the shop's name and the described values, tells it apart from the shop's other
# listings by comparing them (the pointer judgement reads them), and opens it with the details tool (a store may count only
# those as opened). On the race-5 pack a hop's shop has 11 listings (median), more than a page for 60 of 102 hops.
HOP_TERMS = 5              # words in a hop search at most (the race-5 search reads the first 5, every one required): the name first
HOP_VALUES = 3             # described values a hop search adds to the name (more of them found no listing more often)
HOP_COMPARE = 5            # listings compared at once (one pointer judgement reads five); halved when a compare fails
HOP_STEPS = 8              # steps spent looking for one described listing (searches, pages, compares) before the navigator takes over
HOP_PAGES = 4              # steps of those a shop's result pages may take before its rows are ranked and compared: a shop's rows come in id order, so on the 82 recorded race-7 hops walking them took 2.8 steps with 4 hops over HOP_STEPS, and reading the pages first 2.8 with 1
HOP_FAILURES = 2           # the code's hop calls the store answered with an error, in a row, after which it sends no more of them

# Seconds a step needs beyond the wrap-up margin before it is started.
ONE_CALL_SECONDS = 20      # one model judgement
CHECK_SECONDS = 30         # a batch of value checks, or an audit of a decline
QUESTION_SECONDS = 40      # a planned question: planned, sent, and its answer read
VOTE_SECONDS = 60          # several planner samples
LAST_CALL_SECONDS = 2      # the least time a store call needs
# A call of these models that takes longer than this goes on to the next model of its route (under load 1 in 100 reference calls
# took 20-37 s, some of them with few tokens, against 8 s at most for the strong model).
MODEL_SECONDS = {REFERENCE_MODEL: ONE_CALL_SECONDS}

SYSTEM_PROMPT = f"""You are a meticulous shopping assistant. You serve one shopper inside a simulated store, and you can act only through the supplied tools. The task is scored 1 or 0. It scores 1 only if your finish is fully right: every ordered item meets every requirement the shopper stated for its product, every hidden need the shopper reveals when asked, and every change the shopper announces while you work; the quantity and the budget are right; the cheapest (or best) choice is made when the request asks for it; and every rule the request states is followed. If nothing in the store can meet everything, the right finish is to decline, exactly the way the request says.

THE WORLD
- Every tool call uses one step, including calls that fail or return errors, and the number of steps is limited. Model thinking is free; steps are not.
- You may put several tool calls in one reply when none depends on another's result (for example, opening the details of several listings at once). They run in order. Never put an order or a decline together with other calls.
- Tool names, arguments and the shape of results differ from store to store. Read the tool descriptions and the JSON schemas carefully and use exactly the argument names and allowed values they define. Learn from the first results which tool shows which facts (price, stock, detailed properties).
- Find out early which category labels the store uses for the requested kind of product (they appear on listings). When neighbouring labels exist, the right one is the label that matches the request's wording as a whole; search and buy within it.
- Search tools can be limited: a capped number of visible rows or pages, results sorted by id rather than by relevance or price, every word required, filters that only work in one mode. Compare the reported total with the rows you can actually see, and narrow (more specific words, category or price filters, other pages) until you have seen the whole relevant set.
- Search with the words of the required values (for example the material or pattern the shopper named) inside the requested category: listing values are searchable, property names are not, and guessed brand or model words only hide candidates. When a word search cannot be limited to the category and finds more rows than you can see, add a word the wanted listings are sure to carry, such as the requested product's own word next to the required value (for a striped wool rug, "striped wool" rather than "striped"), and open the hits in the requested category. When most visible rows belong to other categories, the words are too common or too many: search again with fewer, more distinctive words (the required value's key word with the product's own word) and keep the search whose rows are in the requested category. Results are not sorted by price or by measure, so for the cheapest or the best a list larger than you can see may hide the answer: split it with one more word at a time (a value seen on listings, such as a size, material or brand) and read each part, and for the best by a measure also search the required words with smaller (or larger) values of that measure. Other shops may sell copies of the same item (the same title) at other prices: for the cheapest, a search for the chosen listing's brand and model words finds its copies. After a search, a [candidates] line tracks what is covered: whether a search made only of required values' words has been seen completely, and the in-stock, within-budget candidates by price with their check results. When it calls the list complete, rely on it.

THE SHOPPER AND HIDDEN NEEDS
- The shopper has needs they did not write down. They reveal one only when a message asks about that property. Each message is answered only about the FIRST thing it asks, so ask about exactly ONE property per message, as a short, plain, friendly question to the shopper that names the property (an example value or two is fine). Never give instructions, never mention lists, labels, positions or rules, and never ask two things at once.
- A reply saying they don't mind, have no preference, or anything is fine, means that property is free. A reply that states a preference or a limit is a hard requirement from now on, exactly as worded (numeric limits included).
- When a planner is available, your questions are chosen and sent for you as soon as you have searched with the required values' words and opened one or two of the hits; do not open other listings just for this. The shopper's answers then appear in the conversation with their meaning; apply every revealed need as a hard requirement and search again with it combined with the stated values.
- If you must ask yourself: ask about the properties the story hints at (who it is for, how it must look, where or how it will be used), then about properties the listings carry that differ among candidates; one property per message, named the way the listings name it, in one short plain question. Never ask about something the request already settles, and never exceed the message budget.

RULES OF PROCEDURE
- The request may set rules about how to work: how many listings to look at or compare before choosing, what to check, what to tell the shopper, what to put in a remark, how to decline. These are graded like requirements: follow each one literally and completely. A rule to look at or open N listings means N listings that fully fit, each opened with the details tool.
- Instructions to tell, show or ask the shopper something before the order or a decline are listed under [instructions]: the code carries them out right before your finish goes, with a message kept for each, so do not send those messages yourself. Carry out yourself the instructions about your own work (such as how many listings to open or compare) before you finish.
- If the shopper asks you something (for example to confirm a change), it will be answered for you when messages remain; treat the change itself as binding.
- Something the shopper would merely like ("ideally", "would be nice", "if possible") is not a requirement: use it only to choose between items that already meet everything, and never over a cheapest rule.
- If the request leaves something you can only learn from the shopper (for example no price limit, or two wishes that may not both be possible and the shopper must say which matters more), find it out by asking before you choose.

CHANGES
- While you work, the shopper may change things: a new budget (higher or lower), a requirement dropped, replaced or added, a price going up, an item sold out. One shopper line can carry several changes. Shopper lines can appear inside any tool result (an extra field holding sentences) or attached to an order attempt. Each change is binding from that moment: re-check your pick and your backups against it, and when the request asks for the cheapest or best, choose again over the new acceptable set.
- An order attempt can be REFUSED. The result then has a "refusal" field, even if the inner result looks successful. The episode continues and the cart is unchanged. Read the refusal, apply every change it states, remove anything that is no longer acceptable from the cart (for example a sold-out item), add the next verified fit, re-check stock and price, and order again.
- Items can also sell out or change price silently. That is why stock and price must be re-checked right before every order attempt; an item whose current price times quantity is over the current budget no longer fits.

HOW TO JUDGE A LISTING (this decides the score)
- After you open a listing's details, a [check] line gives the exact result for each requirement under the scoring rule: a listing meets a requirement when one of its own property fields whose name contains the requirement's property word states the value (singular or plural, one element of a list is enough); its title or description counts only when it has no such field at all. MET and NOT MET are final. NO EVIDENCE means the listing does not show the value, so it does not fit. UNCLEAR (for example different units) is for you to decide from the details.
- A listing whose check says it fits every requirement is a valid choice: do not reject it for anything the request and the shopper did not ask. Among valid listings, prefer one whose check shows no contradiction, unless a cheapest or best rule decides.
- Requirements marked "not blocking" (things no listing can show, such as delivery time) never rule a listing out. Wishes only decide between valid listings.
- The requested kind of product is met by the listing's category; never demand that some property repeat the category's name. The item must be the requested kind of product: when the store labels listings with categories, it must sit in the category whose label matches the request's wording as a whole (a request for "A or B" matches a category called "A & B", not a narrower neighbour called "B"). It must be in stock when the request asks for things on hand, and price times quantity must be within the budget (equal is allowed).

CHEAPEST OR BEST
- When the request asks for the cheapest, the order must be the lowest-priced item among ALL items that meet every requirement. Seeing one fitting item is not enough. Use price filters when a tool offers them, lower the price limit until every remaining row is visible, sort the visible candidates by price yourself, and verify them in ascending price; the first one that fully fits, with every cheaper one checked and rejected, is the answer. Keep the next fitting ones as backups.
- When the request asks for the best by some measure (most, largest, longest, highest, and so on), compare ALL fitting items on that measure the same way and pick the extreme. An item whose listing states no value for that measure cannot win.

FINISHING
- Keep a ranked list of verified fits, with at least one backup, because your first pick may sell out when you check out.
- Before every order attempt: the cart holds exactly one line per requested product (most requests ask for one product), the chosen listing at that product's quantity, and nothing else (remove any other line); you have re-read the chosen item's stock and price after your last cart change, and it is in stock and within budget.
- Decline only after a thorough search finds nothing that meets every requirement; while a listing fits, a decline is sent back. A decline almost never scores: when nothing fits exactly, ordering the closest listing is the better finish. Follow the request's rule for declining exactly, as listed under [instructions], including what a decline's remark may or may not say.
- Order as soon as your choice is final (the questions are asked, its [check] says it fits, and for a cheapest or best request the [candidates] line proves it): every pending change shows up at your first order attempt, and a refused order costs one step and can be fixed. When <READING_ENDS> steps remain, only cart, stock-check and finishing tools stay available, and if no order was attempted when <COMMIT> steps remain, your best candidate is put in the cart and ordered for you. Decide once: do not swap the cart back and forth, since every cart change costs a step.
- Every reply must contain at least one tool call. Keep the reply text short: your current requirements, the verified fits with price and stock, and your next move."""

SETUP_PROMPT = """Read the shopper's request and the store's tool list below. Return ONE compact JSON object (no indentation) and nothing else, with these keys in this order:

"roles": an object naming the tool for each role, using the exact tool names from the list, or null when no tool fits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "message" is the tool that sends a message to the shopper. "cart" is the tool that adds to or removes from the cart. "order" is the tool that buys or checks out. "decline" is the tool that ends the task without buying. "view" is a tool that opens several listings at once, if there is one. One tool may fill both "order" and "decline" when an argument chooses between buying and not buying.
"other_tools": a list with one object per tool that fits none of these roles, each {"tool": its exact name, "does": what it does, in a few words}; [] when every tool has a role.
"order_args": an object with the argument values that make the "order" tool buy (such as the value of a mode or choice argument), using the exact argument names and allowed values from that tool's schema; {} when no such argument exists.
"decline_args": the same for the "decline" tool: the argument values that make it end without buying; {} when no such argument exists.
"cart_args": {"add": the argument values (other than the item and the amount) that make the "cart" tool add an item, "remove": the same for taking an item out}, using exact argument names and allowed values from its schema; {} for either when no such argument exists.
"price_arg": the exact name of the "search" tool's argument that sets the highest price a result may have; null when it has none.
"items": a list with one object per different product the request asks you to buy, in the request's order (most requests ask for one), each {"product": ..., "quantity": ..., "quantity_quote": ..., "quantity_word": ..., "category": ...}:
  "product": the kind of product in the request's words, as a short noun phrase without the count (for example "women's boots").
  "quantity": how many units of ONE listing of this product the shopper explicitly asks you to buy (an integer). It is 1 unless the request literally asks for several units of it. Numbers that count people, occasions, days, rooms, uses, earlier purchases, money, messages or steps are not a quantity. Something sold as one listing, such as a pair of shoes, a set or a pack, counts once per pair, set or pack ("a pair of boots" is 1, "two pairs of boots" is 2).
  "quantity_quote": the exact words of the request that ask for that number of units, copied verbatim; "" when the quantity is 1.
  "quantity_word": the number exactly as it is written inside quantity_quote (for example "3" or "three"); "" when the quantity is 1.
  "category": when a tool argument lists the store's shelves or departments as allowed values, the one value that holds this product, copied exactly; "" otherwise.
"requirements": a list of objects, one per firm stated requirement on a product, each {"item": ..., "property": ..., "op": ..., "values": [...], "unit": ..., "blocking": ..., "reference": ...}:
  "item": the position in "items" of the product it is about (0 for the first), or null when it is about every product.
  "property": the product property in the request's own word for it, for example "material", "fit", "collar type". When the request names it only through a phrase (such as "made of"), use the plain property noun ("material"). When the request gives a material for a named part of the product ("the body has to be spruce"), the property is that part's material ("body material"). Otherwise never rename it to a narrower or different property, and never add words the request does not use.
  "op": "is" (it must be one of the values, also when worded as allowing nothing else, such as "nothing but X" or "only X"; when the request wants several values of a property all at once, write one requirement per value), "not" (it must be none of the values), "at_most", "at_least" or "between" (number limits that include the limit: "at most", "no more than", "up to", "or less", "at least", "or more"), "below" or "above" (strict limits that exclude it: "under", "less than", "below", "more than", "over", "above"), "within" or "paired" (see "reference"), or, only when none of these can express the condition, "text" with "values" holding the condition in the request's own words.
  "values": the required values in the request's own words, each the plain term a listing would state, keeping a qualifier that belongs to the value ("GMO free" stays one value); when the request stacks describing words on one property, such as "high, deep and plush", the value is the plain term "high"; numbers as numbers; for "between", [low, high]. A range written as one label (such as "a 15-30 minute assembly") is a value as written: op "is" with "15-30 minutes"; use "between" only when the request states two limits in words ("between 15 and 30 minutes").
  "unit": the unit the request writes next to a number, as written; "" otherwise.
  "blocking": false for a statement that no product listing could show (a delivery time, the shop's service, the shopper's own plans or checks); true otherwise.
  "field": "price" only for a limit on what one listing of a product may cost, not on the order's total ("I won't pay over 315 cents for the bulbs", "each must stay under 40", with its "item", or null when it holds for every product), with a limit "op", the number in "values" as the request writes it, and "property" "price". Leave "field" out otherwise.
  "when": left out normally. When a requirement holds or lapses depending on another property of a listing ("the bulb's base must be e26, except when the fixture's material is aluminum", "if the sofa you pick is a sofa bed, the cover has to be green"): {"item": the position in "items" of the product whose listing decides it (its own position when the same listing decides it), "property": ..., "op": ..., "values": [...], "unit": ..., "negate": true when the requirement holds except when that is so ("except when", "unless"), false when it holds only when that is so ("if", "when")}. Such a condition is never a "reference" and never a relation.
  "reference": "" normally. When the request gives the value only by pointing at another listing ("the colour has to be the colour of the Dorrilenmo jumpsuit with short sleeves", "the sleeves have to be the same as on the Tadorri romper in chiffon", "a pin count within 3 of Lulenmomo's black RAM"), the words that describe that other listing ("the Dorrilenmo jumpsuit with short sleeves"), copied in full: its maker or shop, its kind and every detail given with it (such as "with weight 1050 g") belong to this description and are never separate requirements of the product; then "values" is [] for the same value, op "not" with values [] when the value must differ from that listing's ("anything but the colour of X", "it can be anything bar the value on the page for X", "set apart from the value on the page for X"), or op "within" with values [3] for "within 3 of". A value named as going with that listing's value is neither the same value nor a different one: when the wanted value is the one the pages pair with that listing's value ("the counterpart of the one on the page of X", "the closure's counterpart on X's page", "that pattern's opposite number", "whatever goes with the value on the page for X"), use op "paired" with values [], and never write the word for the relation as a value. "The same as" and "the very same" name the same value, even when the page points on to another listing: never "paired". Words around "anything bar" that only insist on it keep op "not": "nothing but anything bar the value on the page for X", "nothing else will do but anything bar ...", "I couldn't accept anything besides anything bar ..." and "only anything bar ... will do" all mean any value except that listing's.
  When a sentence names a wanted value and also rules one out for the same property ("a single bar, not a multi-pack"), record the ruled-out value as its own requirement with op "not" as well.
  A statement of what the product is for, such as an occasion or a use, is a requirement on that property ("occasion": "party") only when the request insists on it ("something I'd wear for a party and nothing else", "it has to be for"), and not when it is part of the category's name; a mere mention of where or when it will be used ("to wear on the shop floor") is story, not a requirement.
  A requirement names a value a product listing could state. A sentence that only describes a benefit, a worry or a use without naming such a value (for example "it has to wash easily" or "it must keep going with no fuss") is a hint about something to ask, not a requirement.
  Do not include the order's total price, stock, quantity, budgets or mere wishes here, and do not restate the product kind or category as a requirement: words that are part of the category's name (a season, an occasion or an audience in the category name) are covered by the category and are not a separate property requirement.
"price_cap": the maximum total price of the whole order as a number written exactly as the request writes it (in the request's own unit, never converted), or null when none is stated.
"currency": the price unit the request uses, or null.
"in_stock_only": true when the request wants only items actually on hand or in stock.
"priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option, or says the lower the price or total the better; "best" only when it explicitly asks for the best by some measure; otherwise "none". Remarks that money is tight or that every penny counts are not a request for the cheapest; they only explain the price cap.
"priority_detail": for "best", what "best" is measured by; otherwise "".
"priority_property": for "best", the listing property it is measured by, in the request's own word (for example "width"); "" otherwise.
"priority_direction": for "best", "least" when the smallest value wins (narrowest, lightest, shortest, lowest) or "most" when the largest wins; "" otherwise.
"priority_scope": for "cheapest" or "best", "order" when it is asked of the whole order's total, "item" when of each product; "" otherwise.
"budgets": {"messages": the number of messages the shopper allows you to send, as an integer, or null when not stated, "steps": the number of moves, steps or tool calls the request allows, as an integer, or null when not stated}.
"soft_wants": a list of things the shopper would like but does not strictly require (phrased as ideally, preferably, would be nice, if possible). Keep these out of "requirements".
"relations": a list with one object per condition that one product must meet relative to whichever listing is chosen for ANOTHER product of the order (it must work with, fit, suit or go with that one), each {"dependent": the position in "items" of the product that must meet it, "hub": the position of the product it depends on, "quote": the request's words for it}. Such a condition is a relation only: not a requirement and not an instruction. Empty when none.
"if_nothing_fits": the request's own rule for when no listing meets everything: {"closest": "wanted" when it asks to be offered or sent the listing that comes closest, "forbidden" when it says not to offer a substitute, null when it says nothing, "margin": how far over the cheapest of the closest it allows, as a fraction (11% is 0.11), or null, "counts": which of "stock" and "budget" it counts among the requirements a listing can miss, or []}.
"instructions": a list with one object per procedural instruction the request gives the assistant: every procedural rule it states for the assistant (how many listings to look at, what to check, tell, show or ask, how to decline, what not to do, what to include in a remark), quoted word for word, and every fact the assistant can only learn from the shopper and should ask about before choosing (for example which of two wishes matters more when the request says both may not be possible; then also put both wishes in soft_wants, not in requirements): such a fact is "at_start" with "message" true and "content" naming what to learn. Not a property of the product, not a budget, not a price rule, not a number of messages or steps; remarks about what the shopper will do themselves are story. Each is <INSTRUCTION_FIELDS>. A rule about what a decline must or must not say is "before_decline" with "message" false. Empty when the request gives none.
Every sentence of the request must land somewhere: a requirement, the priority, a soft want, an instruction, a relation, a budget, a hint, or pure story. When unsure whether a sentence is an instruction, make it one. A question the shopper asks the assistant or themselves is story unless it names a fact only the shopper knows that the choice depends on.
"""

INTERPRET_PROMPT = """You help a shopping assistant understand what a shopper just said. Return ONE JSON object and nothing else, with these keys:
"meaning": one short sentence saying what the shopper means.
"needs": a list of new or changed firm requirements on a product, each an object {"item": ..., "property": ..., "op": ..., "values": [...], "unit": ..., "quote": ...} (op "text" with the condition in the shopper's words when none of the ops below can express it): "item" is the position under PRODUCTS IN THE ORDER of the product it is about (0 for the first), or null when it is about every product; "property" is the product property in the shopper's own word for it (the plain property noun when they name it only through a phrase), never renamed or narrowed, and without the product it belongs to ("capacity of the ram" is property "capacity" with the item of the RAM); "op" is "is", "not", "at_most", "at_least", "between" (limits that include the limit), "below" or "above" (strict: "under"/"less than" and "more than"/"over" exclude the limit itself); "values" are the shopper's values (numbers as numbers; [low, high] for "between"); "unit" is the unit written next to a number, or ""; "quote" is the shopper's exact words; "reference" is "" normally, but when the shopper gives the value only by pointing at another store listing ("the same as on the page for X's pepper", "set apart from the value on the page for X's rings, with ..."), it is the words that describe that other listing, copied in full (its maker or shop, its kind and every detail given with it, such as "with color family clear", belong to the description and are never separate needs), with "values" [] and op "is" for the same value ("the very same as on the page for X") or "not" when the value must differ from that listing's ("anything bar the value on the page for X"), or "paired" when the wanted value is the one the pages pair with that listing's value ("the counterpart of the one on the page of X", "that pattern's opposite number", "whatever goes with the value on the page for X"), which is neither the same value nor a different one ("the same as" and "the very same" are the same value, never "paired"). Words around "anything bar" that only insist on it keep op "not": "nothing but anything bar the value on the page for X", "nothing else will do but anything bar ...", "I couldn't accept anything besides anything bar ..." and "only anything bar ... will do" all mean any value except that listing's. A requirement that replaces an earlier one on the same property is listed here with its new value; "change": "add" when a value comes on top of the earlier one, so both must hold. A price limit on one product only, not the order's total, is a need with "field": "price", "property": "price" and its "item".
"understood": false when the shopper says they did not follow or could not make sense of the assistant's question, or asks for it again in other words; true otherwise.
"one_at_a_time": true when the shopper asks to be asked about one thing at a time (only the first matter, a single ask, one by one): the question touched more than one of their points at once; false otherwise. Such a reply is not a failure to follow, so "understood" stays true.
"no_preference": a list of properties the shopper said they do not care about.
"dropped": a list of the properties (as named under REQUIREMENTS SO FAR) whose requirement the shopper no longer needs, without a replacement; {"property": ..., "item": ...} when that holds for one product only.
"new_budget": the new maximum total price as a number, or null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), else false.
"price_changed": true when the shopper says the price of an item has changed (up or down), else false.
"new_quantity": the number of units of one product the shopper now explicitly asks you to buy, as an integer, or null when they say nothing about how many units to buy. A number that counts people, days, occasions or money is not a quantity.
"quantity_item": the position under PRODUCTS IN THE ORDER of the product whose quantity changes, or null when new_quantity is null.
"quantity_quote": the shopper's exact words that ask for that number of units, copied verbatim; "" when new_quantity is null.
"quantity_word": the number exactly as it is written inside quantity_quote; "" when new_quantity is null.
"reply": when the shopper asks the assistant something or asks for a confirmation, a short, plain reply to send back that answers it and names the change in substance (for example confirming the new value); otherwise "".
"speaker": "shopper" when these lines are the shopper's own words, "store" when they are the store's or the system's notice (a sell-out, a price change, a refusal written by the store).
"ack": "accepted" when the shopper accepts, approves or simply acknowledges what the assistant told or proposed (yes, okay, go ahead, noted, carry on) without asking for anything else; "refused" when they reject it or say no; "none" otherwise.
"instructions": procedural instructions these lines give the assistant from now on (what to tell, show, ask or check before ordering, what not to do), each <INSTRUCTION_FIELDS>; [] when none. A need, a budget, a dropped requirement or a limit on messages or steps is not an instruction. A line that only tells the assistant it need not wait for the shopper, or says how or whether the shopper will answer, asks nothing to be done and is not an instruction.
"allowance": when the lines set a new limit on the assistant's messages, steps (moves, calls) or time: {"messages_total": the new total number of messages allowed or null, "messages_left": how many may still be sent from now or null, "steps_total": the new total number of steps or null, "steps_left": how many steps are left from now or null, "seconds_left": the time left in seconds or null}; null when no such limit is stated.
One line can carry several changes: report every one. A requirement replaced by another on the same property is only a new need (with the new value); a replacement on a different property is a new need plus the old property in dropped. When the shopper says which of two wishes matters more, the preferred wish is a need. When the shopper states a budget, it is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and charitably. Folksy, figurative or roundabout phrasing that names a value means the shopper firmly wants that value; treat a value as excluded only when the shopper clearly says to avoid it. A number with a comparison is a limit, kept with its unit: "at most", "no more than", "up to", "at least" and "or more" include the limit (at_most, at_least); "under", "less than", "more than" and "over" exclude it (below, above). When the shopper refuses anything that "falls short of" (or otherwise misses) a value, that value is the requirement, and a comparison inside it is the limit: "anything where height falls short of at most 34 inches" is at_most 34 inches, "falls short of at least 500 gb" is at_least 500 gb, and "falls short of single" is op "is" with "single". A new budget replaces the old one whether it is higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference, or anything works, put the property asked about in no_preference; such an answer never drops or changes any other requirement."""

VERIFY_PROMPT = """You are the final checker before a shopping assistant puts an item in the cart or orders it. The property requirements were already checked by exact rules; their results are listed under EXACT CHECKS and are final, so do not judge MET, NOT MET or NO EVIDENCE again. Judge ONLY these, from the records shown:
- The item is the requested kind of product: when listings carry a category label, it must be the category whose label matches the request's wording as a whole, not a narrower or neighbouring one.
- Price, budget and stock are checked separately in code: do not judge them.
- Each requirement marked UNCLEAR: decide from the records whether the item meets it (convert units when they differ; limits include the edge; a range counts only when the whole range meets the limit). A value that only a variant or option label states meets it only when it is unambiguously a value of that property (a colour option is not a dial glass). A value that contains the required word but names another kind of thing (a plating or coating for a colour, a period or school for a style, a blend for a material) does not state it; one that adds a qualifier of the same kind ("sapphire glass" for sapphire, "dark red" for red) does. A requirement that excludes a value is confirmed only by a field that names the property with another value; a listing lacking the field is doubtful, so decide it from its records.
Return ONE JSON object and nothing else: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a check above that fails]}. Use fits=false only for a check above that the records clearly fail."""

PLAN_PROMPT = """You plan the questions a shopping assistant sends to a shopper. The shopper has hidden needs: each is the value of one property of one product in the order, a property that the listings' details record, revealed only when a message asks about that property of that product; a question about anything else gets a polite "don't mind" (or a reply that the shopper did not follow it) and is spent for nothing. A question the shopper did not follow (marked "not understood" below) was about a property that is not a hidden need of that product: never ask about that property of that product again, in any words. A question the shopper wanted taken one matter at a time (marked "one at a time" below) touched two of their points at once and got no answer, so it asked nothing yet: ask now about a narrower property it covers, by that property's own name as the listings record it. Choose exactly QUESTION_COUNT different questions, most promising first.
Some requests hint at a hidden need without naming it: a story sentence that states no requirement can point at one property through the part of the product it involves, the place or setting the product must suit, what it is used for or with, or how it will be seen. Many story sentences are only colour (the shopper's job, travels, mood or figures of speech) and point at nothing. Work in three steps:
1. list every story sentence of the request that states no requirement;
2. for each, write what the shopper would check on a listing to be satisfied, then name the listed property that records exactly that, or "none". Prefer the property of the specific part or setting the sentence involves over a general property of the whole product: the part that hands, feet or body touch points to that part's material or shape; a place the product must fit, stand or be used in points to the size, height or width that place limits; what it will be made into or used with points to the property naming that use or compatibility (such as type, style or form factor); photographs or being seen points to colour, pattern or the visible part. A sentence never points back at a property the request already settles;
3. choose: first anything under MUST BE LEARNED FROM THE SHOPPER that is not asked yet; then each property a sentence clearly points at (the sentence describes what that property records, not merely who will use the product or where it goes); then, to fill the remaining questions, the listed properties that most listings of that product record and whose values differ, preferring those marked as differing among the listings that meet the stated requirements (a hidden need always splits those listings): the look, material, pattern, shape, type or key measure a shopper of that product most often insists on. Guesses about the wearer or user (gender, age, size class, occasion) come after those unless a sentence clearly points at them.
Use property names as the listings below show them (the plainest one when several name the same thing); another noun is allowed only for a clear hint that no listed property covers. Never choose a property the request already settles (its stated requirements, the price, stock, quantity or category), a property already asked for that product, two names for one property, or a field that bundles several properties (such as variant labels mixing colour and size). When the order has several products, every question is about one product: give its position under PRODUCTS IN THE ORDER as "item", choose properties from that product's own listings, and spread the questions over the products (each product can have its own hidden need).
Return ONE compact JSON object and nothing else: {"hints": [{"because": "...", "item": 0, "property": "..."}], "questions": [{"item": 0, "property": "...", "question": ""}]}. In "hints", write one entry per story sentence that points at a property, with "because" in two to six words naming the clue; do not copy sentences. Leave "question" empty for a product property, since that question is written for you; write "question" (one short plain sentence) only for an item under MUST BE LEARNED FROM THE SHOPPER that is not a product property, such as the budget."""

INSTRUCTION_FIELDS = """{"quote": the instruction's own words copied exactly from the text, "when": "before_order" (it must be done before the order is placed), "before_decline" (only when declining, or only when nothing fits), "before_finish" (before either), "before_cart_change" (before changing, replacing or removing something already in the cart), "before_message" (before a message is sent to the shopper), "after_change" (each time the shopper or the store changes something, such as a price, the stock or a requirement), "at_start" (before choosing anything) or "any" (it holds throughout), "message": true when carrying it out means sending the shopper a message, else false, "wait_reply": true when the assistant must have the shopper's answer (such as their yes) before going on, else false, "count": a number the instruction sets, or null, "counts": what that number counts: "open" (listings opened with the details tool), "compare" (listings compared side by side), "search" (searches), "cart" (cart changes), "message" (messages to the shopper), "step" (tool calls of any kind), or null, "limit": "at_least" when at least that many are required, "at_most" when no more than that many are allowed, or null, "tool": the exact name of a tool from the list that the instruction requires using, or null, "names": "forbidden" when it forbids naming, offering, suggesting or pointing to any listing or substitute in what is said, "required" when it asks for one to be named, else null, "content": what must be done or said, in a few words}"""
SETUP_PROMPT = SETUP_PROMPT.replace("<INSTRUCTION_FIELDS>", INSTRUCTION_FIELDS)
INTERPRET_PROMPT = INTERPRET_PROMPT.replace("<INSTRUCTION_FIELDS>", INSTRUCTION_FIELDS)

PAIR_PROMPT = """Two store listings are shown: the HUB, chosen for one product of an order, and a CANDIDATE for another product that must work with (fit, suit, go with) whichever hub is chosen. Decide from the CANDIDATE's own page whether it says it works with this hub listing (by naming it, its model or its line). A page that names a different model, or a different part's compatibility, does not.
Return ONE JSON object and nothing else: {"works": true or false}."""

EXECUTOR_PROMPT = """You carry out one instruction the shopper's request (or the shopper later) gave a shopping assistant, at the moment it falls due. The code calls you right before the finish named below (or the cart change), and the instruction has not been carried out for the cart as it is now. Decide the ONE action that carries it out now:
- "say": send the shopper one message. Write it in full in "text", as one or two short plain sentences that state exactly what the instruction asks for, using only the facts below (listing ids, counts, prices, totals and pages as the instruction needs them). Ask for nothing the instruction does not ask for, and never ask about product properties.
- "call": call one of the TOOLS listed, with "tool" its exact name and "args" its arguments exactly as its schema asks.
- "remark": when the instruction shapes what the finish itself says (its remark or note), the finish's text in full in "text" (only when THE FINISH TAKES A TEXT says yes).
- "hold": the assistant itself must do something first that only it can do (such as opening or comparing more listings); put what is missing in "why".
- "none": nothing needs doing for this finish (the instruction is already met by what was done, it does not concern this finish, or no listed tool can carry it out); say why in "why".
Read the messages already sent: a message sent after the last cart change that already did what the instruction asks means "none". When an earlier attempt for this instruction got a reply that did not accept it, write the message again more plainly.
Return ONE JSON object and nothing else: {"action": "say" | "call" | "remark" | "hold" | "none", "text": "...", "matters": how many separate things the message asks or tells (a basket with its total is one), "tool": "...", "args": {}, "why": "..."}."""

CATCH_ALL_PROMPT = """A shopping assistant is about to finish an errand (the finish is named below). The request and the shopper may have given it instructions to carry out before finishing; they are listed below with what the code has done about each. Decide which of the LISTED instructions are still undone for THIS finish, with the cart as it is now: one that must be done before this finish and was not done after the last cart change, or whose earlier doing no longer fits the cart. Never invent an instruction: answer only with numbers from the list.
Return ONE JSON object and nothing else: {"undone": [numbers of the listed instructions still undone]}."""

AUDIT_PROMPT = """A shopping assistant is about to tell the shopper that nothing in the store fits. The listings below passed the exact property checks except for the requirements marked UNCLEAR. For each listing, decide from its record whether it meets its UNCLEAR requirements (convert units when they differ; limits include the edge), is not shown out of stock, and has a price within the budget.
Return ONE JSON object and nothing else: {"possible_fits": [ids of listings that appear to meet every requirement, best first, at most 5], "reason": "one sentence"}."""

VALUES_PROMPT = """You check store listings against a shopper's requirements the way a strict store grades orders: by the terms a listing uses, not by loose similarity of meaning. Each numbered value below was found in a listing field named like the requirement's property, and it shares a word or word stem with the required value.
Answer true only when the value names the required thing itself in another form: another spelling or word form, an abbreviation, or the same thing with words added that keep it that thing (such as a blend that contains it). Answer false when the value names a different thing that merely shares a word or stem, such as an effect, a part, a related feature, a narrower or broader kind, or the opposite, and when the value joins the required thing with another one by "+" (such a combination is a different thing).
Return ONE JSON object and nothing else, with one entry per requirement label, for example {"R1": {"1": true, "2": false}}."""

# added to the value check when it is asked about a measure written in another unit (the model converts; the code holds no unit table)
VALUES_UNITS_NOTE = """Some values below state a measure in another unit than the required one instead of sharing a word with it: such a value is the required value only when it converts to it, allowing for the rounding a conversion brings (50.8 mm is 2 inches; 3/4 in is 19 mm); answer false when it converts to another value (55 mm is not 2 inches) or cannot be converted to the required unit."""

VARIANTS_PROMPT = """Each field below comes from the property details of store listings, shown with values seen under it. Which fields hold variant labels, that is the option a listing is sold in, naming its colour, size, pack or version as a store shows a selected choice (such as "Black / XS" or "white")? A field named after one property of the product (such as a style, material, type or features field) is not a variant field, even when its values look alike.
Return ONE JSON object and nothing else: {"variant_fields": [field names exactly as shown]}."""

UNITS_PROMPT = """Values of one product property are written in different units. For each unit listed, give how many of the BASE unit one of that unit equals (for example, with BASE "gb", "tb" is 1000). Return ONE JSON object and nothing else: {"factors": {"unit": number}}, with null for a unit you cannot convert."""

MEANING_PROMPT = """No store listing meets all of a shopper's requirements literally. Below, each requirement lists values that otherwise fitting listings show for that property. For each numbered value, answer true when a shopper asking for the required value would accept it as that value: the same thing in other words, or a shade, variant or kind of it (such as a colour shade within the required colour family). Answer false when it is a different thing, a different category the shopper did not ask for, or the opposite.
Return ONE JSON object and nothing else, with one entry per requirement label, for example {"R1": {"1": true, "2": false}}."""

MODES_PROMPT = """The store tool below ends a shopping task. Which of its argument values make it place the order, and which make it end without buying? Use only argument names and allowed values from its schema. Return ONE JSON object and nothing else: {"order_args": {...}, "decline_args": {...}}."""

FORMAT_PROMPT = """A shopping assistant needs one call of a store tool to do this: ACTION. The store's tools and their JSON argument schemas are listed below. Choose the tool and write its arguments, using only argument names and allowed values from that tool's schema. Write "<ITEM>" where a listing's id goes, "<QTY>" where a number of units goes and "<TEXT>" where free text goes; when an argument is a list with one entry per cart line, write a single entry with the placeholders. Leave out optional arguments the action does not need. Return ONE JSON object and nothing else: {"tool": the tool name, "args": {...}}, or {"tool": null} when no tool does this."""

FORMAT_ACTIONS = {
    "add": "put <QTY> units of the listing <ITEM> into the cart",
    "remove": "take the listing <ITEM> out of the cart completely",
    "basket": "make the cart hold exactly <QTY> units of the listing <ITEM> and nothing else",
    "order": "place the order for what the cart holds",
    "order_item": "buy <QTY> units of the listing <ITEM>",
    "decline": "end the task without buying anything, telling the shopper <TEXT>",
    "message": "send the shopper this message: <TEXT>",
}

REREAD_PROMPT = """A shopping assistant read the shopper's request below as an order for the products under PRODUCTS READ SO FAR, but it is now putting the listings below into the cart, which look like more than one kind of product. Read the request again: which different products does it ask to buy? List each one, in the request's order, as {"product": the kind of product in the request's words without the count, "quantity": how many units of ONE listing of it the request explicitly asks for (1 unless it literally asks for several; a pair, set or pack is 1), "quantity_quote": the exact words asking for that number ("" for 1), "quantity_word": the number as written inside quantity_quote ("" for 1), "category": the store shelf that holds it, copied exactly from THE STORE'S SHELVES when they are listed, else ""}. A listing the request does not ask for (another kind of product than any it names) is no reason to add a product. Then give, for each requirement label, the position of the product it is about, or null when it is about every product.
Return ONE JSON object and nothing else: {"items": [...], "assign": {"R1": 0}}."""

ITEM_MAP_PROMPT = """A shopper's order has the products listed below (position: product). For each store listing below, give the position of the product it is a listing of, judged by its category label and title, or -1 when it is none of them (another kind of product). Return ONE JSON object and nothing else: {"items": {"listing id": position}}."""

REFERENCE_PROMPT = """A shopper's requirement takes the value of one property from another store listing, described as REFERENCE (the shopper's own words are under REQUEST). Below are listings opened in the store, each with its labels and price, its text (title, description, seller's notes) and its details. Is one of them that listing? Go through every detail the description names (the maker, shop or brand name, often given only in the page's seller's notes; the kind of product; each feature) and check it against the page: a listing is that listing only when its page confirms every detail and does not say it is not that item. When several come close, choose the one whose page confirms all details, or none. If one is:
- when its page says to take PROPERTY from another listing instead of this one, copy that other listing's description into "next", in the page's own words, and leave "value" and "paired" null;
- otherwise copy into "value" the value its page states for PROPERTY (its details first, then its text), and into "paired" the value the page names as going with that value of its own, from a sentence about this listing or from a rule listing which value goes with which ("its red goes with pink"; "our rule: 12v with 100-240 v, and 24v with 220v" on a page whose own value is 12v gives 100-240 v). Leave "paired" null when the page pairs nothing with that value. Copy only what the page states: what the shopper wants done with these values is decided elsewhere.
Return ONE JSON object and nothing else: {"id": the listing id or null, "details": {each detail of the description: true when that listing's page confirms it, else false}, "value": the page's own value or null, "paired": the value the page pairs with it or null, "next": the other listing's description or null}."""

class Reply(tuple):
    """One call's answer as (call id, text), the way the navigator reads it; held_back tells whether the call was held back
    before it reached the store (no step used)."""

    def __new__(cls, call_id: str, text: str, held_back: bool = False) -> "Reply":
        reply = super().__new__(cls, (call_id, text))
        reply.held_back = held_back
        return reply

    def __getnewargs__(self) -> tuple[str, str, bool]:
        return (self[0], self[1], self.held_back)

    @property
    def id(self) -> str:
        return self[0]

    @property
    def text(self) -> str:
        return self[1]


_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

def _price(model: str, usage: Any) -> float:
    """What one answered call cost, from its token counts at PRICES (the provider's own cost field reads 0 on replica runs)."""
    if not isinstance(usage, dict):
        return 0.0
    rates = PRICES.get(model) or tuple(max(rate[i] for rate in PRICES.values()) for i in range(4)) + (0, 1.0)
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    count = lambda value: value if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else 0
    prompt, out = count(usage.get("prompt_tokens")), count(usage.get("completion_tokens"))
    read, write = count(details.get("cached_tokens")), count(details.get("cache_write_tokens"))  # both are part of the prompt's count
    factor = rates[5] if rates[4] and prompt >= rates[4] else 1.0
    return factor * (max(0, prompt - read - write) * rates[0] + read * rates[1] + write * rates[2] + out * rates[3])

def _brief(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _brief(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_brief(v, limit, depth + 1) for v in value]
    return value


_ID_HINT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")


def _compact(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > OBS_CHARS:
        text = text[:OBS_CHARS] + "...(cut)"
    return text


def _parse_json_object(text: str, salvage: bool = True) -> dict[str, Any] | None:
    """A JSON object in a model's answer; salvage=False refuses a cut-off object instead of closing it (tool-call arguments)."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except ValueError:
            pass
    if not salvage:
        return None
    salvaged = _salvage_json(text[start:]) if start >= 0 else None
    if salvaged is not None or start < 0:
        return salvaged
    found: list[tuple[int, dict[str, Any]]] = []  # the last complete top-level object (an answer written after prose)
    for match in re.finditer(r"\{", text):
        if found and match.start() < found[-1][0]:
            continue
        try:
            value, end = json.JSONDecoder().raw_decode(text, match.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            found.append((end, value))
    return found[-1][1] if found else None


def _salvage_json(text: str) -> dict[str, Any] | None:
    """A JSON object cut off mid-way (an answer that hit its token limit), closed after its last complete element."""
    cuts: list[tuple[int, str]] = []
    closers: list[str] = []
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            closers.append("}" if char == "{" else "]")
        elif char in "}]" and closers:
            closers.pop()
            cuts.append((index + 1, "".join(reversed(closers))))
        elif char == ",":
            cuts.append((index, "".join(reversed(closers))))
    for position, closing in reversed(cuts[-200:]):
        try:
            value = json.loads(text[:position] + closing)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _type_ok(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "null":
        return value is None
    return True


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    kinds = schema.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    if not kinds:
        return value
    if any(_type_ok(value, k) for k in kinds):
        if "integer" in kinds and isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    for kind in kinds:
        try:
            if kind == "integer":
                if isinstance(value, float) and value.is_integer():
                    return int(value)
                if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
                    return int(float(value))
            if kind == "number" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                number = float(value)
                return int(number) if number.is_integer() else number
            if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
            if kind == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
        except (TypeError, ValueError):
            continue
    return value


def _check(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, list[str]]:
    if not isinstance(schema, dict):
        return value, []
    problems: list[str] = []
    value = _coerce(value, schema)
    kinds = schema.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    if kinds and not any(_type_ok(value, k) for k in kinds):
        return value, [f"{path}: expected {'/'.join(kinds)}"]
    if "enum" in schema and value not in schema["enum"]:
        if isinstance(value, str):
            for option in schema["enum"]:
                if isinstance(option, str) and option.lower() == value.strip().lower():
                    value = option
                    break
        if value not in schema["enum"]:
            problems.append(f"{path}: must be one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: must equal {schema['const']!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and isinstance(schema["exclusiveMinimum"], (int, float)) and value <= schema["exclusiveMinimum"]:
            problems.append(f"{path}: must be above {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and isinstance(schema["exclusiveMaximum"], (int, float)) and value >= schema["exclusiveMaximum"]:
            problems.append(f"{path}: must be below {schema['exclusiveMaximum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{path}: allows at most {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            fixed = []
            for index, item in enumerate(value):
                item, sub = _check(item, schema["items"], f"{path}[{index}]")
                fixed.append(item)
                problems += sub
            value = fixed
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                problems.append(f"{path}.{name}: required")
        if schema.get("additionalProperties") is False:
            for name in list(value):
                if name not in properties:
                    problems.append(f"{path}.{name}: not an allowed argument")
        fixed = {}
        for name, item in value.items():
            if name in properties:
                item, sub = _check(item, properties[name], f"{path}.{name}")
                problems += sub
            fixed[name] = item
        value = fixed
    return value, problems


def _sentences(value: Any, out: list[str], depth: int = 0, least: int = 4) -> None:
    if depth > 4:
        return
    if isinstance(value, str):
        if len(value.split()) >= least:
            out.append(value)
    elif isinstance(value, list):
        for item in value[:10]:
            _sentences(item, out, depth + 1, least)


_ID_KEYS: set[str] = set()  # field names a result's rows proved to hold listing ids (one episode per process; reset per episode)
_PRICE_KEYS: set[str] = set()  # field names a cart total proved to be the price


def _id_rank(key: Any) -> int:
    """0 for a field named "id" (listing_id, productId), 1 for one ending in it (pid, uuid), 2 for any other containing it (width)."""
    words = _words(_key_name(key))
    return 0 if "id" in words else 1 if any(w.endswith("id") for w in words) else 2


def _id_value(value: Any, named: bool = True) -> str | None:
    """A field value that can be a listing id: a short token, or a whole number under an id-named field."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value) if named and value >= 0 else None
    if isinstance(value, str) and _ID_HINT.match(value) and (named or (re.search(r"\d", value) and re.search(r"[A-Za-z]", value))):
        return value
    return None


def _collect_ids(value: Any, out: set[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, int) and not isinstance(item, bool):
                ident = _id_value(item) if key in _ID_KEYS or _id_rank(key) < 2 else None  # not a width or a count
            else:
                ident = _id_value(item) if isinstance(item, str) and ("id" in str(key).lower() or key in _ID_KEYS) else None
            if ident:
                out.add(ident)
            else:
                _collect_ids(item, out, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _collect_ids(item, out, depth + 1)


DECLINE_NOTE = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
_OPS = ("is", "not", "at_most", "at_least", "below", "above", "between", "text")  # "text": a condition only a reading of the page can judge
_LIMITS = ("at_most", "at_least", "below", "above", "between")  # number limits ("below" and "above" exclude the limit itself)
_NUMBER = re.compile(r"(?:(?<![\w.])(-))?(\d+(?:\.\d+)?)\s*([a-z]*)")  # a minus counts only at the start of a token, so "10-20" stays a range


def _words(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value if value is not None else "").lower())


def _spelling(word: str) -> str:
    return re.sub(r"our(s?)$", r"or\1", word) if len(word) > 4 else word


def _same_word(a: str, b: str) -> bool:
    a, b = _spelling(a), _spelling(b)
    return a == b or a + "s" == b or b + "s" == a or a + "es" == b or b + "es" == a


def _value_words(value: Any) -> list[str]:
    """The words of a value as compared with another value: case and spacing do not count, a number and the unit or letters
    written onto it are two words ("240v" = "240 v"), and a range that repeats its unit reads as one ("100v-240v" = "100-240 v")."""
    text = str(value if value is not None else "").lower()
    text = re.sub(r"(\d)\s*([a-z]+)\s*(-|~|to)\s*(\d+(?:\.\d+)?)\s*\2(?![a-z])", r"\1-\4 \2", text)
    return _words(re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", text))


def _has_words(text: Any, wanted: list[str]) -> bool:
    have = _value_words(text)
    wanted = [w for x in wanted for w in _value_words(x)]
    return bool(wanted) and all(any(_same_word(w, h) for h in have) for w in wanted)


def _has_phrase(text: Any, target: Any) -> bool:
    have, wanted = _value_words(text), _value_words(target)
    size = len(wanted)
    return size > 0 and any(all(_same_word(wanted[i], have[j + i]) for i in range(size)) for j in range(len(have) - size + 1))


def _says(text: Any, target: Any) -> bool:
    wanted = " ".join(_value_words(target))
    return bool(wanted) and f" {wanted} " in f" {' '.join(_value_words(text))} "


def _elements(value: Any) -> list[str]:
    # a comma between a digit and exactly three digits is a thousands separator ("11,200 mAh"), not a list break
    return [part.strip() for part in re.split(r"[;/|]|(?<!\d),|,(?!\d{3}(?!\d))", str(value)) if part.strip()]


def _menu_values(value: Any) -> list[str]:
    """The values one fact of a listing shows the question menu, lower-cased: a short text itself; a list, or a longer text that
    lists label-length values, each of them (at most MENU_LIST_VALUES); never an id-like code."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)  # a fact the store records as a JSON number is a value like any other
    if isinstance(value, str) and len(value) <= MENU_VALUE_CHARS:
        parts = [value]
    else:
        parts = _elements(value) if isinstance(value, str) else [v.strip() for v in value if isinstance(v, str)] if isinstance(value, list) else []
        if len(parts) < (1 if isinstance(value, list) else 2) or any(len(part) > MENU_PART_CHARS for part in parts):
            return []  # one long text is prose, not values
    kept = [part.lower() for part in parts if part and not (re.search(r"\d", part) and re.fullmatch(r"[A-Za-z0-9_\-]+", part) and ("-" in part or "_" in part))]
    return list(dict.fromkeys(kept))[:MENU_LIST_VALUES]


def _states(value: Any, target: Any) -> bool:
    wanted = _words(target)
    if re.search(r"[,;/|]", str(target)):
        return _has_words(value, wanted)
    # a part joining two values with "+" ("ssd + hdd") is that combination, which states neither value alone
    joined = "+" not in str(target)
    return any(_has_phrase(part, target) and not (joined and "+" in part) for part in _elements(value))


def _near_word(a: str, b: str) -> bool:
    """Two forms of one word: the same word, one the start of the other, a long shared stem ("padded", "paddings"), or one
    letter apart in a word of four or more ("gray", "grey"; "canister", "cannister")."""
    if _same_word(a, b) or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))):
        return True
    if min(len(a), len(b)) < 4 or not (a.isalpha() and b.isalpha()):
        return False  # (a model number one digit off is another model)
    stem = len(_shared_start(a, b))
    if stem >= 4 and stem * 2 >= max(len(a), len(b)):
        return True
    tail = len(_shared_start(a[::-1], b[::-1]))
    return stem >= 1 and tail >= 1 and abs(len(a) - len(b)) <= 1 and stem + tail >= max(len(a), len(b)) - 1


def _shared_start(a: Any, b: Any) -> Any:
    """What two strings or word lists have in common from their start."""
    size = 0
    while size < min(len(a), len(b)) and a[size] == b[size]:
        size += 1
    return a[:size]


def _related(value: Any, target: Any, prop: Any) -> bool:
    """A value that may be the required one in another form: a form of the target's first own word (the property's words say
    nothing), or all its words among the target's, or the other way round ("n male" for "n-type male", "no wire" for
    "non-wire"; "v-neck" for "scoop neck" shares only its last word). Only the model check tells such a value from a different
    thing; until it does the value is unclear."""
    named = _words(prop)
    have = _words(value)
    words = _words(target)
    wanted = [w for w in words if len(w) >= 3 and not w.isdigit() and not any(_same_word(w, n) for n in named)][:1]
    within = lambda some, others: bool(some) and all(any(_near_word(w, x) or (not w.isdigit() and x.startswith(w)) for x in others) for w in some)
    return any(_near_word(w, h) for w in wanted for h in have) or (bool(wanted) and (within(have, words) or within(words, have)))


def _blank(value: Any) -> bool:
    """A value that states nothing: no word or number at all ("---", "[]"), or two lone letters joined by a mark ("N/A")."""
    return bool(re.fullmatch(r"[\W_]*(?:[^\W\d_][\W_]+[^\W\d_][\W_]*)?", str(value if value is not None else "")))


def _pointer_words(reference: Any, request: Any, values_seen: set[str], keys_seen: set[str], shelves: set[str]) -> tuple[list[str], list[str]]:
    """(names, values) of the description a pointer gives of another listing: its capitalised words (the shop's or maker's
    name, a shelf written out), and its other words that the opened listings state as values but never as a property's
    name (the details it is described by), each once in the description's order; a number inside a decimal or a fraction
    is left out (a search splits it otherwise than the listings' pages do). A word the request also uses tells no listing
    apart (the product's own name, "the"); a description quoted from the request counts only the request's other uses."""
    text = str(reference or "")
    named, asked = _words(text), _words(request)
    quoted = f" {' '.join(named)} " in f" {' '.join(asked)} "
    caps = [w for token in re.findall(r"[A-Z][\w'&-]*", text) for w in _words(token.split("'")[0])]
    split = {w for token in re.findall(r"\d+(?:[./]\d+)+", text) for w in _words(token)}

    def own(w: str, counted: bool) -> bool:
        return asked.count(w) <= named.count(w) if counted else w not in asked

    for counted in ((True,) if quoted else (False, True)):  # (a request that names the shop itself: its count decides)
        names = [w for w in dict.fromkeys(caps) if len(w) >= 3 and own(w, counted) and w not in shelves]
        if names:
            break
    values = [w for w in dict.fromkeys(named) if w not in names and w not in split and (len(w) >= 3 or w.isdigit()) and own(w, quoted)
              and any(_same_word(w, s) for s in values_seen) and not any(_same_word(w, k) for k in keys_seen)]
    return names, values


def _onward_words(note: Any, prop: Any, own: set[str]) -> str:
    """The words of a page's note that send a property on to another listing, or "": the sentence naming the property (its
    whole name, else the last words of it) with a capitalised name after that which is none of the page's own value words
    (own), from there on, without a closing clause of few plain words ("..., not this item"). Read from the shape every
    recorded chain page shares (races 5-10: 279 of 280 hops), so no wording is assumed; a name before the property only says
    whose page it is."""
    wanted = _words(prop)
    for size in range(len(wanted), 0, -1):
        for sentence in re.split(r"(?<=[.!?])\s+", str(note or "")):
            marks = [(m.end(), w) for m in re.finditer(r"[\w'&-]+", sentence) for w in _words(m.group())[:1]]
            spots = [marks[i + size - 1][0] for i in range(len(marks) - size + 1) if all(_same_word(a, marks[i + n][1]) for n, a in enumerate(wanted[-size:]))]
            rest = sentence[spots[0]:] if spots else ""
            if not any(w not in own for token in re.findall(r"\b[A-Z][\w'&-]*", rest) for w in _words(token.split("'")[0])):
                continue
            head, _, tail = rest.rstrip(" .!?;").rpartition(",")
            plain = len(_words(tail)) <= 5 and not re.search(r"\d|[A-Z()]", tail)
            return " ".join((head if head and plain else rest).lstrip(" ,:;-").split()).rstrip(" .!?;")[:160]
    return ""


# the first question. Without a property menu (nothing opened yet, the question comes from the story) every model planned 33 of
# the 48 race-3 replica prompts, so the reference model asks it; with a menu the strong model planned 23 and 24 of 99 against the
# reference model's 20 and 22, so the strong model asks it (each keeps the other as its fallback)
PLAN_FIRST_ROUND = ((REFERENCE_MODEL, 0.0),)
PLAN_FIRST_MENU_ROUND = ((JUDGE_MODEL, 0.0),)
# Request settings of the first question's planning: the strong model reasons by default (on the 51 recorded race-7 first plans,
# 312 reasoning tokens a plan, $0.0087 of it), and at effort low it reasons none ($0.0054), with the same first question in 37
# of 51 (two draws of the default setting agree on 86% of 205 race 1-4 prompts) and the same 5 hidden needs asked of 16
SETTINGS_FIRST_PLAN = {JUDGE_MODEL: {"reasoning": {"effort": "low"}}}
# the second and later questions: on the race-5 replica's recorded prompts of rounds 1-4 the reference model planned 48 hidden
# needs of 181 against the fast model's 29 and 34 (the strong model 44), and 64 against the fast model's 60 of 279 on races 1-4,
# at about twice the fast model's price; the fast model is its fallback
PLAN_LATER_ROUND = ((REFERENCE_MODEL, 0.0),)
PLAN_LATE_ROUND = ((REFERENCE_MODEL, 0.0),)  # third and later question rounds

# Which models answer each kind of call, first choice first (judge() adds the backup provider). No call changes model with the
# task's spend; at the exploration stop (spent_out) the calls stop instead.
ROUTES = {
    "setup": ("SETUP_MODEL", "BACKUP_MODEL"),
    "setup_retry": ("BACKUP_MODEL", "SETUP_MODEL"),
    "navigator": ("MAIN_MODEL", "BACKUP_MODEL", "MAIN_MODEL"),
    "judge": ("REFERENCE_MODEL", "JUDGE_MODEL"),  # the model check of a listing, the audit of a decline (on 227 recorded checks the reference model agreed 225 times, as precise against the grader)
    "read": ("READ_MODEL", "JUDGE_MODEL"),  # reading the shopper, call formats, finishing modes, unit factors
    "read_late": ("READ_MODEL",),  # reading the shopper when little time is left
    "match": ("JUDGE_MODEL", "READ_MODEL"),  # which near-miss values are the required ones
    # which product of the order a listing is, which fields are variant fields: on the 44 recorded race-7 prompts the fast model
    # gave the strong model's answer 42 times (one listing mapped to the product the answer key accepts it for, where the strong
    # model mapped it to none; one pack size taken for a variant field), at a twentieth of the price
    "label": ("READ_MODEL", "JUDGE_MODEL"),
    "reference": ("REFERENCE_MODEL", "JUDGE_MODEL"),  # which opened listing a pointer describes, and its value
    "reference_late": ("REFERENCE_MODEL", "READ_MODEL"),  # the same past the stop (no call there dearer than a planned question's)
    "values": ("VALUE_MODEL", "JUDGE_MODEL"),  # whether a stated value is the required one
    "executor": ("READ_MODEL", "JUDGE_MODEL"),  # carrying out one due instruction; deciding which listed instructions are undone
    "executor_late": ("READ_MODEL",),  # the same past the spend stop (the reserve pays only for the fast model)
}


def _key_spent(error: Any) -> bool:
    """A refused model call says the run's key is spent: its limit reached (403) or the account's credit (402), read the way
    the validator reads its own calls' failures (a 402 for credit held by calls in flight passes, and is not this)."""
    if not isinstance(error, dict):
        return False
    body = str(error.get("body") or "").lower()
    if error.get("status") == 403:
        return "key limit exceeded" in body
    return error.get("status") == 402 and ("insufficient credits" in body or "requires more credits" in body)


# The moments an instruction can fall due: the code's own actions (an order, a decline, a cart change), before any finish,
# before anything is chosen, or throughout. An instruction's content is free text the executor reads; the code reads only when
# it falls due, whether it needs a message and the shopper's answer, and a count.
INSTRUCTION_WHEN = ("before_order", "before_decline", "before_finish", "before_cart_change", "before_message", "after_change", "at_start", "any")
INSTRUCTION_COUNTS = ("open", "compare", "search", "cart", "message", "step")  # what a counted instruction counts (the code counts it)
EXECUTOR_CALLS = 8      # executor and catch-all calls a task may make at most
PAIR_CALLS = 6          # model judgements of a pair a task may make (only when the hub's page shows no model-like identifier)
EXECUTOR_TOKENS = 1500  # one executor or catch-all answer (a short JSON object)
GATE_HOLDS = 4          # finishes the gate may hold back in one task; past them it lets every finish go (no hold loop)
CLOSE_RESENDS = 2       # the same close (lines, prices, task version) the store did not take this often is not sent again
CLOSE_ROUNDS = 4        # rounds of decide, repair and send in one forced finish (a store that refuses for ever ends the task)
BASKET_POOL = 6         # listings per product tried when a basket-level rule breaks a basket
INSTRUCTION_TRIES = 3   # actions one instruction may take for one cart (a look-up, a message, a plainer message) before it is given up


def _instruction(raw: Any, text: Any, source: str) -> dict[str, Any] | None:
    """One instruction record from a reader's answer (an object, or a bare string); its quote is checked against the text it
    came from (kept either way, marked when its words are not found there). Unknown timing means before any finish."""
    if isinstance(raw, str):
        raw = {"quote": raw}
    if not isinstance(raw, dict):
        return None
    quote = " ".join(str(raw.get("quote") or raw.get("text") or "").split())[:400]
    words = _words(quote)
    if not words:
        return None
    when = str(raw.get("when") or "").strip().lower().replace(" ", "_")
    count = raw.get("count")
    have = set(_words(text))
    return {"quote": quote, "source": source, "when": when if when in INSTRUCTION_WHEN else "before_finish",
            "message": raw.get("message") if isinstance(raw.get("message"), bool) else None,
            "wait_reply": raw.get("wait_reply") is True,
            "count": int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) and count > 0 else None,
            "counts": raw.get("counts") if raw.get("counts") in INSTRUCTION_COUNTS else None,
            "limit": raw.get("limit") if raw.get("limit") in ("at_least", "at_most") else None,
            "tool": raw.get("tool") if isinstance(raw.get("tool"), str) and raw.get("tool").strip() else None,
            "names": raw.get("names") if raw.get("names") in ("forbidden", "required") else None,
            "content": " ".join(str(raw.get("content") or "").split())[:300],
            "quoted": sum(1 for w in words if w in have) >= 0.8 * len(words),
            "status": "pending", "done_for": None, "tries": []}


def _relation(raw: Any, text: Any, products: int) -> dict[str, Any] | None:
    """One relation between two products of the order (the dependent must suit whichever listing is chosen for the hub)."""
    if not isinstance(raw, dict):
        return None
    dep, hub = raw.get("dependent"), raw.get("hub")
    if not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v < products for v in (dep, hub)) or dep == hub:
        return None
    quote = " ".join(str(raw.get("quote") or "").split())[:300]
    return {"dependent": dep, "hub": hub, "quote": quote, "quoted": bool(_words(quote)) and all(w in set(_words(text)) for w in _words(quote))}


def _route(kind: str) -> list[str]:
    """The models for one kind of call, read when the call is made (a measuring script may swap a model constant)."""
    return [globals()[name] for name in ROUTES[kind]]


def _vote(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge planner samples: non-property items from the first sample, then (product, property) pairs by summed rank points (1, 1/2, 1/3 ...)."""
    if not samples:
        return []
    extra = [q for q in samples[0].get("questions") or [] if isinstance(q, dict) and not _words(q.get("property")) and str(q.get("question") or "").strip()]
    points: dict[tuple[Any, str], float] = {}
    order: list[tuple[Any, str]] = []
    for sample in samples:
        entries = [(q.get("item") if isinstance(q.get("item"), int) and not isinstance(q.get("item"), bool) else None, str(q.get("property") or "").strip()) for q in sample.get("questions") or [] if isinstance(q, dict) and _words(q.get("property"))]
        for rank, (item, name) in enumerate(entries):
            key = next((k for k in order if k[0] == item and _same_name(k[1], name)), None)
            if key is None:
                key = (item, name)
                order.append(key)
                points[key] = 0.0
            points[key] += 1 / (rank + 1)
    ranked = sorted(order, key=lambda k: (-points[k], order.index(k)))
    return extra + [{"item": k[0], "property": k[1], "question": ""} for k in ranked]


def _ask_text(prop: Any, product: Any = None, example: Any = "") -> str:
    """One plain question about one property, naming the product it belongs to when one is given (a multi-product order's
    hidden need is held per product; a single product's needs are read without one) and, with `example`, one value of it."""
    named = " ".join(re.sub(r"[^\w' &-]+", " ", str(product or "")).split()[:6]).lower()
    whose = f" of the {named}" if named else ""
    such = f", for example {' '.join(str(example).split())}" if str(example).strip() else ""
    return f"What is your preference for the {' '.join(_words(prop))}{whose}{such}?"


def _same_name(a: Any, b: Any) -> bool:
    """Two property names name one property: the same words in any order, allowing plurals and -our/-or spellings."""
    first, second = _words(a), _words(b)
    return bool(first) and len(first) == len(second) and all(any(_same_word(x, y) for y in second) for x in first) and all(any(_same_word(y, x) for x in first) for y in second)


def _names_product(label: Any, product: Any) -> bool:
    """A label that names a requested product: every word of it is a word of the product's name ("pumps" for "pumps with a logo pattern")."""
    first, second = _words(label), _words(product)
    return bool(first) and all(any(_same_word(x, y) for y in second) for x in first)


def _narrower(recorded: Any, asked: Any) -> bool:
    """A property name that keeps only part of another ("roast" for "coffee roast"): fewer words, each one of the other's."""
    have, wanted = _words(recorded), _words(asked)
    return bool(have) and len(have) < len(wanted) and all(any(_same_word(h, w) for w in wanted) for h in have)


def _same_property(a: Any, b: Any) -> bool:
    first, second = set(_words(a)), set(_words(b))
    return bool(first and second) and (first <= second or second <= first)


def _is_measure(target: Any) -> bool:
    return bool(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*[a-z]*\s*", re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(target).lower())))


def _same_measure(value: Any, target: Any, unit: str) -> bool:
    wanted = _measures(target)
    if len(wanted) != 1 or not _is_measure(target):
        return False
    number, want = wanted[0][0], wanted[0][1] or unit
    return any(n == number and (not want or not u or _same_unit(u, want)) for part in _elements(value) for n, u in _measures(part))


def _measures(value: Any) -> list[tuple[float, str]]:
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(value).lower())
    text = re.sub(r"(?<![a-z0-9.])[a-z]+\d[\w.]*", " ", text)  # a name with a number written onto it ("ddr5", "ip67") is no measure
    return [(-float(number) if sign else float(number), unit) for sign, number, unit in _NUMBER.findall(text)]


def _in_other_unit(value: Any, targets: list[Any], unit: str) -> bool:
    """A value whose numbers carry only units other than the required measure's ("38.1 cm" for 15 inches): only a conversion
    can tell whether it is the required value."""
    wanted = {_measures(t)[0][1] or unit for t in targets if _is_measure(t)} - {""}
    have = {u for _, u in _measures(value) if u}
    return bool(wanted and have) and not any(_same_unit(u, w) for u in have for w in wanted)


def _unit_parts(value: Any, targets: list[Any], unit: str) -> list[str]:
    """The parts of a value in another unit that a conversion decides (none when some number is in the required unit); a
    "/" between digits is a fraction ("1 1/4 in"), not a list break."""
    if not _in_other_unit(value, targets, unit):
        return []
    parts = re.split(r"[;|]|(?<!\d)/|/(?!\d)|(?<!\d),|,(?!\d{3}(?!\d))", str(value))
    return [p.strip() for p in parts if p.strip() and _in_other_unit(p, targets, unit)]


def _key_name(key: Any) -> str:
    """A fact key with camelCase boundaries spaced ("toeStyle" -> "toe Style"); other separators are kept, so "toe_shape" and "toe shape" stay apart."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(key))


def _same_unit(a: str, b: str) -> bool:
    if not a or not b:
        return True
    if _same_word(a, b) or (min(len(a), len(b)) >= 2 and (a.startswith(b) or b.startswith(a))):
        return True
    # a one-letter abbreviation and the spelled-out unit it stands for ("m" and "meter"): the long form is a word of four or
    # more letters starting with that letter (two- and three-letter forms such as "mm", "mah" or "gb" are other units)
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) == 1 and len(long_) >= 4 and long_.isalpha() and long_.startswith(short)


def _numeric(op: str, limits: list[float], unit: str, value: Any) -> str | None:
    found = _measures(value)
    if not found:
        return None
    if unit and any(u and _same_unit(u, unit) for _, u in found):
        # drop only numbers that carry another explicit unit ("16gb 5200mhz" against gb reads 16 gb); numbers without a unit
        # and one-letter marks such as the "x" of "55 x 70 inch" stay, so a range or a size counts as a whole
        found = [(n, u) for n, u in found if len(u) < 2 or _same_unit(u, unit)]
    units = [u for _, u in found if u]
    if unit and units and not _same_unit(units[-1], unit):
        return "unclear"
    low, high = min(n for n, _ in found), max(n for n, _ in found)
    if op == "at_most":
        met = high <= limits[0]
    elif op == "below":
        met = high < limits[0]
    elif op == "at_least":
        met = low >= limits[0]
    elif op == "above":
        met = low > limits[0]
    else:
        met = low >= min(limits[:2]) and high <= max(limits[:2])
    return "met" if met else "not met"


def _ident_phrases(text: str) -> set[str]:
    """Model-like identifiers in a text: one piece with at least 2 letters and 3 digits ("LIZ-7810" -> "liz7810"), or a
    capitalised word of 3+ letters followed by a number of 3+ digits ("KAT 6750" -> "kat 6750"). Shapes only, no words."""
    out: set[str] = set()
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-.]*[A-Za-z0-9]|[A-Za-z0-9]", str(text))
    for k, token in enumerate(tokens):
        flat = re.sub(r"[^0-9a-z]", "", token.lower())
        if sum(c.isalpha() for c in flat) >= 2 and sum(c.isdigit() for c in flat) >= 3:
            out.add(flat)
        if re.fullmatch(r"[A-Z][A-Za-z]{2,}", token) and k + 1 < len(tokens) and re.fullmatch(r"\d{3,}", tokens[k + 1]):
            out.add(token.lower() + " " + tokens[k + 1])
    return out


_SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n+")


_EXCEPTION = re.compile(r"\b(?:except(?:ing)?|excluding|but not|other than|apart from|not for|save for)\b", re.IGNORECASE)
_FAMILY_CODE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{2,}-?\d+)([xX]+)(?![A-Za-z0-9])")


def _exception_split(sentence: str) -> tuple[str, str]:
    """A sentence's text before a marker of exception, and the clause from that marker to the sentence's end."""
    found = _EXCEPTION.search(sentence)
    return (sentence[:found.start()], sentence[found.start():]) if found else (sentence, "")


def _family_heads(text: str) -> set[tuple[str, int]]:
    """Code families a text names ("every SIR21xx model"): the head in lower case without its hyphen, and the length of the run of x."""
    return {(re.sub(r"[^0-9a-z]", "", head.lower()), len(run)) for head, run in _FAMILY_CODE.findall(text)}


def _in_family(code: str, families: set[tuple[str, int]]) -> bool:
    return any(code.startswith(head) and len(code) == len(head) + run for head, run in families)


def _link_marks(strings: list[str]) -> tuple[set[tuple[str, int]], set[str], set[tuple[str, int]]]:
    """What a listing's texts name as going with it, by shape: the families it names, and the codes and families it names only
    as exceptions ("Works with every SIR21xx model, except SIR2140")."""
    families: set[tuple[str, int]] = set()
    codes: set[str] = set()
    out: set[tuple[str, int]] = set()
    for text in strings:
        for sentence in [x for x in _SENTENCE.split(text) if x.strip()]:
            kept, clause = _exception_split(sentence)
            families |= _family_heads(kept)
            codes |= _ident_phrases(clause)
            out |= _family_heads(clause)
    return families, codes, out


def _link_state(hub_strings: list[str], strings: list[str]) -> bool | None:
    """Whether a dependent listing's own texts go with a hub listing's identifiers: True when they name one of them, its
    family or its series, False when they rule it out or name only others, None when the hub shows no identifier or the
    dependent names no link at all (the texts cannot tell)."""
    hub_ids, series = _ident_phrases(" ".join(hub_strings)), _series_codes(hub_strings)
    if not hub_ids and not series:
        return None
    families, excepted, out = _link_marks(strings)
    plain = {re.sub(r"[^0-9a-z]", "", i) for i in hub_ids}
    if hub_ids & excepted or any(_in_family(i, out) for i in plain):
        return False
    vouched = _vouching_ids(strings)
    if hub_ids & vouched or series & _code_heads(strings) or any(_in_family(i, families) for i in plain):
        return True
    own = {c for s in strings if not _SENTENCE.search(s.strip()) for c in _ident_phrases(s)}
    return False if families or excepted or out or vouched - own or _code_heads(strings) else None


def _code_shape(code: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"([a-z]+)(\d+)", code)
    return (match.group(1), len(match.group(2))) if match else None


def _vouching_ids(strings: list[str]) -> set[str]:
    """The identifiers a listing's own text vouches for: from every sentence of its texts and every fact value, leaving out a
    sentence that names a sibling part (a code of the same letters-and-digits shape as one of the listing's own codes, those in
    its fields without sentences such as a title, but another value: "Part PE8989 also fits: X" on the page of part PE8488)."""
    own = {c for s in strings if not _SENTENCE.search(s.strip()) for c in _ident_phrases(s) if " " not in c}
    shapes = {_code_shape(c) for c in own} - {None}
    good: set[str] = set()
    for text in strings:
        for sentence in [x for x in _SENTENCE.split(text) if x.strip()]:
            ids = _ident_phrases(_exception_split(sentence)[0])  # (a code named only as an exception vouches for nothing)
            if not any(_code_shape(c) in shapes and c not in own for c in ids if " " not in c):
                good |= ids
    return good


def _series_codes(strings: list[str]) -> set[str]:
    """Short family codes a listing's running text names ("Compatible part series: ZE8."): capitals then a few digits, shapes only."""
    return {c for s in strings if _SENTENCE.search(s.strip()) for c in re.findall(r"(?<![A-Za-z0-9-])[A-Z]{1,4}\d{1,3}(?![A-Za-z0-9])", s)}


def _in_code_family(value: Any, text: Any) -> bool:
    """A required code the text covers by its family ("VIK-5710" by "every VIK-57xx model"): letters and digits, then a run of x."""
    code = re.sub(r"[^0-9a-z]", "", str(value).lower())
    heads = {re.sub(r"[^0-9a-z]", "", h.lower()) for h in re.findall(r"(?<![A-Za-z0-9])([A-Za-z]{2,}-?\d+)[xX]+(?![A-Za-z0-9])", str(text))}
    return bool(_code_shape(code)) and any(code.startswith(h) for h in heads)


def _code_heads(strings: list[str]) -> set[str]:
    """The family each part code of a listing belongs to, by the code's own form ("ZE8-10576" is of family ZE8)."""
    return {c for s in strings for c in re.findall(r"(?<![A-Za-z0-9-])([A-Z]{1,4}\d{1,3})-\d+", s)}


def _requirement(raw: Any, source: str) -> dict[str, Any] | None:
    """One requirement from a reader's answer. Its "field" says where the value is read ("price": the listing's price, not its
    facts), "change" how it meets an earlier one on the property, "when" the condition it depends on."""
    req = _plain_requirement(raw, source)
    if req is None or not isinstance(raw, dict):
        return req
    if raw.get("change") in ("add", "replace"):
        req["change"] = raw["change"]
    if raw.get("field") == "price" and req["op"] in _LIMITS:
        req.update(property="price", field="price", blocking=True)
    when = raw.get("when")
    if isinstance(when, dict) and not req.get("pending"):
        condition = _plain_requirement(when, "stated")
        if condition is None or condition.get("pending"):
            req["blocking"] = False  # a condition that cannot be read: the requirement may lapse, so nothing can be proven by it
        else:
            condition["negate"] = when.get("negate") is True
            req["when"] = condition
    return req


def _plain_requirement(raw: Any, source: str) -> dict[str, Any] | None:
    if isinstance(raw, str):
        name, _, rest = raw.partition(":")
        raw = {"property": name, "op": "is", "values": [rest], "quote": raw} if rest.strip() else None
    if not isinstance(raw, dict):
        return None
    prop = " ".join(_words(raw.get("property")))
    op = str(raw.get("op") or "is").strip().lower().replace(" ", "_")
    values = raw.get("values") if isinstance(raw.get("values"), list) else [raw.get("values")]
    item = raw.get("item") if isinstance(raw.get("item"), int) and not isinstance(raw.get("item"), bool) and raw.get("item") >= 0 else None
    # (a paired need whose reader left the description out still names the listing in its quote: on 54 readings of shopper lines, once)
    reference = " ".join(str(raw.get("reference") or (raw.get("quote") if op == "paired" else "") or "").split())[:160]
    if reference and prop and op in ("is", "within", "not", "text", "paired"):
        # the value is the one another listing states, or the one its page pairs with it: pending (not checked) until that
        # listing is found and read (a condition worded as free text that names such a listing relates to its value too)
        numbers = [n for v in values if not isinstance(v, bool) for n, _ in _measures(v)]
        tolerance = [abs(numbers[0])] if op == "within" and numbers else []
        relation = "within" if tolerance else {"not": "not", "paired": "paired", "text": "paired"}.get(op, "is")
        return {"property": prop, "op": relation, "values": tolerance, "unit": " ".join(_words(raw.get("unit"))), "blocking": False,
                "pending": True, "reference": reference, "described": reference, "item": item, "source": source, "quote": str(raw.get("quote") or "")[:200]}
    op = op if op in _OPS else "text"  # an operator the code does not know is a condition the model check judges, never "is"
    if op in _LIMITS:
        numbers = [n for v in values if not isinstance(v, bool) for n, _ in _measures(v)]
        limits = numbers[:2] if op == "between" else numbers[:1]
        if len(limits) >= (2 if op == "between" else 1):
            values = limits
        elif source == "revealed":
            # the shopper names the wanted value as a limit ("nothing that falls short of single"): no number to compare,
            # so the word is the value (a stated limit without a number stays dropped: its meaning is unclear)
            op, values = "is", [str(v).strip() for v in values if isinstance(v, (str, int, float)) and not isinstance(v, bool) and _words(v) and not _measures(v)]
        else:
            return None
    else:
        values = [str(v).strip() for v in values if isinstance(v, (str, int, float)) and not isinstance(v, bool) and _words(v)]
    if not prop or not values:
        return None
    return {"property": prop, "op": op, "values": values, "unit": " ".join(_words(raw.get("unit"))), "blocking": raw.get("blocking") is not False, "item": item, "source": source, "quote": str(raw.get("quote") or "")[:200]}


GROUND_WORDS = 6  # a limit read from the request is grounded when a word of its property stands within this many words of its number


def _grounded(req: dict[str, Any], text: Any) -> str:
    """Why a requirement the setup read from the request is not grounded in the request's words ("" when it is): a number
    limit needs a word of its property near one place the request writes its number ("camera at most 48 k" never grounds
    "price: at most 48"); a value needs a word of its values or of its property in the request. A pointer's value and a
    worded condition are not checked, nor a limit whose number the request does not write as such (a unit converted)."""
    if req.get("pending") or req["op"] == "text":
        return ""
    if req.get("field") == "price":  # (a price limit often names no property: "I won't pay over 315 cents")
        return "" if any(re.search(rf"(?<![\d.]){re.escape(format(abs(v), 'g'))}(?![\d])", re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(text or ""))) for v in req["values"]) else f"the number {format(req['values'][0], 'g')} is not written in the request"
    flat = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(text or "").lower())
    tokens = re.findall(r"\d+(?:\.\d+)?|[a-z]+", flat)
    named = [w for w in _value_words(req["property"]) if not w.isdigit()]

    def stem(a: str, b: str) -> int:
        return next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))

    def said(word: str, have: list[str]) -> bool:  # the same word, or one of its forms ("weighs" for weight)
        return any(_same_word(word, h) or (stem(word, h) >= 4 and stem(word, h) >= min(len(word), len(h)) - 2) for h in have)
    if req["op"] in _LIMITS:
        spots = [i for i, t in enumerate(tokens) if t[0].isdigit() and any(float(t) == abs(v) for v in req["values"])]
        if not spots or not named:
            return ""
        if any(said(w, tokens[max(0, i - GROUND_WORDS): i + GROUND_WORDS + 1]) for i in spots for w in named):
            return ""
        if any(i >= 2 and tokens[i - 1] == "than" and tokens[i - 2].endswith("er") for i in spots):
            return ""  # a comparative names the property by its adjective ("no taller than 75 cm" for height)
        if any(re.search(rf"(?<![\d.]){re.escape(format(abs(v), 'g'))}\s*[a-z]*\s+[a-z]+\s*(?:[.?!,;]|$)", flat) for v in req["values"]):
            return ""  # so does one word closing the clause after the number and its unit ("75 cm high.")
        return f"no word of the property '{req['property']}' stands near the number {format(req['values'][0], 'g')} in the request"
    values = [w for v in req["values"] for w in _value_words(v)]
    if any(said(w, tokens) for w in values + named):
        return ""
    return f"neither the value nor the property '{req['property']}' is written in the request"


def _one_value(value: Any) -> Any:
    """A value a judgement gave: a list of its parts ("gold", or "8 channel" in pieces) is one value, a number its text."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if isinstance(v, (str, int, float)) and not isinstance(v, bool)) or None
    return f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else value


_UPPER, _LOWER = ("at_most", "below"), ("at_least", "above")


def _replaces(new: dict[str, Any], old: dict[str, Any]) -> bool:
    """Whether a new requirement takes the place of an earlier one on the same property and product."""
    if new.get("change") == "add" or new.get("pending") or old.get("pending"):
        return new.get("change") != "add"
    a, b = new["op"], old["op"]
    if a in ("is", "text") and b in ("is", "text"):
        return True
    if a == "not" and b == "not":
        return False
    if {a, b} <= set(_UPPER) or {a, b} <= set(_LOWER) or "between" in (a, b) and a in _LIMITS and b in _LIMITS:
        return True
    if a in _LIMITS and b in _LIMITS:  # a floor beside a ceiling: both hold unless the later word leaves no room
        low, high = (new, old) if a in _LOWER else (old, new)
        return low["values"][0] > high["values"][0]
    if "is" in (a, b) and "not" not in (a, b):  # a limit and a value: the later word wins when it breaks the other
        limit, value = (new, old) if a in _LIMITS else (old, new)
        numbers = [n for v in value["values"] for n, _ in _measures(v)]
        check = {"at_most": lambda n: n <= limit["values"][0], "below": lambda n: n < limit["values"][0], "at_least": lambda n: n >= limit["values"][0], "above": lambda n: n > limit["values"][0]}.get(limit["op"])
        return bool(numbers and check and limit["unit"] == value["unit"] and not any(check(n) for n in numbers)) or not numbers
    return False


def _describe(req: dict[str, Any]) -> str:
    when = req.get("when")
    if when:
        return _describe({k: v for k, v in req.items() if k != "when"}) + f" ({'unless' if when.get('negate') else 'only when'} {_describe(when)} on the listing chosen for product {when.get('item')})"
    unit = f" {req['unit']}" if req["unit"] else ""
    if req.get("pending"):
        how = f"within {req['values'][0]:g}{unit} of the value on" if req["op"] == "within" and req["values"] else {"not": "different from the value on", "paired": "the value paired with the one on"}.get(req["op"], "the same as on")
        return f"{req['property']}: {how} {req['reference']}"
    if req["op"] in _LIMITS:
        numbers = [f"{v:g}" for v in req["values"]]
        limit = {"at_most": f"at most {numbers[0]}", "at_least": f"at least {numbers[0]}", "below": f"below {numbers[0]}", "above": f"above {numbers[0]}"}.get(req["op"]) or f"{numbers[0]} to {numbers[-1]}"
        return f"{req['property']}: {limit}{unit}" + (" (the limit itself excluded)" if req["op"] in ("below", "above") else "")
    if req["op"] == "text":
        return f"{req['property']}: " + "; ".join(req["values"]) + " (as the request words it)"
    return f"{req['property']}: " + ("not " if req["op"] == "not" else "") + " or ".join(req["values"])


def _within(req: dict[str, Any], number: float) -> bool:
    """A number against a limit requirement."""
    values = req["values"]
    return {"at_most": number <= values[0], "below": number < values[0], "at_least": number >= values[0], "above": number > values[0]}.get(req["op"], values[0] <= number <= values[-1])


def _key_match(wanted: list[str], key: Any) -> int:
    have = _words(key)
    found = sum(1 for w in wanted if any(_same_word(w, h) or (len(w) >= 4 and h.startswith(w)) for h in have))
    return 2 if wanted and found == len(wanted) else (1 if found else 0)


def _family(req: dict[str, Any], facts: dict[str, Any], level: int = 2) -> dict[str, Any]:
    wanted = _words(req["property"])
    return {k: v for k, v in facts.items() if _key_match(wanted, k) == level}


def _evidence(req: dict[str, Any], facts: dict[str, Any], text: str, vocabulary: set[str], judged: dict[str, bool] | None = None, variants: set[str] | None = None) -> tuple[str, str]:
    judged = judged or {}
    family, near = _family(req, facts), _family(req, facts, 1)
    op, values = req["op"], req["values"]
    if op == "text":  # no exact rule reads it: the field naming the property (if any) goes to the model check
        shown = next((f"{k}: {v}" for k, v in family.items()), None) or next((f"{k}: {v}" for k, v in near.items()), None)
        return "unclear", shown or "title or description"
    if op in _LIMITS:
        named = _words(req["property"])
        exact = {k: v for k, v in family.items() if len(_words(k)) == len(named) and all(_same_word(a, b) for a, b in zip(_words(k), named)) and _measures(v)}
        found = [(_numeric(op, values, req["unit"], part), f"{k}: {v}") for k, v in (exact or family).items() for part in _elements(v)]
        for state in ("met", "unclear", "not met"):
            hit = next((e for r, e in found if r == state), None)
            if hit:
                return state, hit
        near_hit = next((f"{k}: {v}" for k, v in near.items() if _measures(v)), None)
        worded = next((f"{k}: {v}" for k, v in family.items() if str(v).strip()), None)
        return ("unclear", near_hit or worded) if near_hit or worded else ("no evidence", "")

    def literal(value: Any) -> bool:
        return any(_same_measure(value, t, req["unit"]) or (not req["unit"] and _states(value, t)) if _is_measure(t) else _states(value, t) for t in values)

    def in_text() -> bool:
        return any(_same_measure(text, t, req["unit"]) if _is_measure(t) else _says(text, t) for t in values)

    def same(value: Any) -> bool:
        return any(judged.get(part.lower()) for part in _elements(value) + _unit_parts(value, values, req["unit"]))

    def open_question(value: Any) -> bool:
        return any(_related(value, t, req["property"]) for t in values) and not all(part.lower() in judged for part in _elements(value))

    hits = [(k, v, "") for k, v in family.items() if literal(v)] + [(k, v, " (same meaning)") for k, v in family.items() if not literal(v) and same(v)]
    labels = {k: v for k, v in facts.items() if " ".join(_words(k)) in (variants or set()) and k not in near}
    elsewhere = next((f"{k}: {v}" for k, v in labels.items() if literal(v)), None) if not family else None
    # a value in another unit that the value check has not judged yet ("38.1 cm" for 15 inches): only a conversion can tell
    # (one it judged is decided: the same value counts as a value of the same meaning above, another value as a miss). For
    # "not" the number of a label in another unit ("4k" against 1080p) is almost never the excluded value: it stays met
    converted = next(((k, v) for k, v in family.items() if _in_other_unit(v, values, req["unit"]) and not all(p.lower() in judged for p in _unit_parts(v, values, req["unit"]))), None)
    if op == "not":
        if hits:
            return "not met", f"{hits[0][0]}: {hits[0][1]}{hits[0][2]}"
        if not family and in_text():
            return "not met", "title or description"
        # a field of another name that states the excluded value ("is_unisex: unisex" beside no gender field, "options: black /
        # eu:38" beside "color: white") says the listing is that value: an exclusion holds only when none of its fields states it
        # (over the race-5 replica pack's "not" requirements this turns 519 verdicts, and the pack accepts 1 of those listings).
        # Only a worded value counts so: a number in another field ("power_consumption: 1-500" for storage not 500gb) is another measure
        stated = next((f"{k}: {v}" for k, v in facts.items() if k not in family and any(not _is_measure(t) and _states(v, t) for t in values)), None)
        if stated:
            return "not met", stated
        if family:  # a field names the property with another value: that is what confirms an exclusion (evidence level 0)
            key, value = next(iter(family.items()))
            return "met", f"{key}: {value}"
        return "met", ""
    if hits:
        named = _words(req["property"])
        exact = [(k, v, h) for k, v, h in hits if len(_words(k)) == len(named) and all(_same_word(a, b) for a, b in zip(_words(k), named))]
        key, value, how = (exact or hits)[0]
        others = [] if exact else [f"{k}: {v}" for k, v in family.items() if not literal(v) and not same(v) and any(p.lower() in vocabulary for p in _elements(v))]
        return ("contradicted", f"{key}: {value}{how}; but {others[0]}") if others else ("met", f"{key}: {value}{how}")
    if family:
        pending = next(((k, v) for k, v in family.items() if open_question(v)), None)
        key, value = pending or next(iter(family.items()))
        if not pending and req.get("named") is False:
            # the request never names this property (the name is a guess): another field stating the value may be the right one
            other = next((f"{k}: {v}" for k, v in facts.items() if k not in family and literal(v)), None)
            if other:
                return "unclear", f"{other} (the request does not name the property; another field states the value)"
        if converted and not pending:  # the number is in another unit ("50.8 mm" for 2 inches): only a conversion can tell
            return "unclear", f"{converted[0]}: {converted[1]} (in another unit than the requirement's: convert to compare)"
        if not pending and all(_blank(v) for v in family.values()):  # a placeholder ("N/A") proves no miss
            return "unclear", f"{key}: {value} (states nothing)"
        return ("unclear" if pending else "not met"), f"{key}: {value}"
    if in_text():
        return "met", "title or description"
    if elsewhere:
        return "met", elsewhere
    phrase = next((f"{k}: {v}" for k, v in facts.items() for t in values if _says(v, f"{req['property']} {t}") or _says(v, f"{t} {req['property']}")), None)
    if phrase:  # "GMO free" read as property GMO, value free: a field stating the whole phrase
        return "met", phrase
    near_hit = next((f"{k}: {v}" for k, v in near.items() if literal(v) or same(v)), None)
    return ("unclear", near_hit) if near_hit else ("no evidence", "")


def _evidence_level(req: dict[str, Any], state: str, evidence: str, facts: dict[str, Any], variants: set[str], keyed: bool) -> int:
    """How directly one check's evidence states a requirement, by the measured precision of a "met" verdict on the race-3 and
    race-4 packs: 0 a field named for the property (or the same field under another name) states the value, alone, qualified
    or in a comma list (0.92-0.99); 1 a variant label, a value judged of the same meaning, or the title or description where no
    opened listing records the field (about 0.9); 2 a "/"-joined list, the property merely absent for "not", or the text where
    listings do record the field (0.6-0.8); 3 a "+" join, a related value, a key naming only part of the property, or another
    field (about 0.5 or less); 9 a failed check."""
    if state in ("not met", "no evidence"):
        return 9
    if req.get("field"):
        return 0  # read from the listing's own price
    if not evidence:
        return 2  # "not": nothing states the excluded value
    if evidence == "title or description":
        return 3  # (the race-6 and race-7 replica packs accept 0.23 of the listings whose title alone states the value, keyed or not)
    key = next((k for k in sorted(facts, key=lambda k: -len(str(k))) if evidence.startswith(f"{k}: ")), None)
    if key is None:
        return 3  # (a relaxed or leniently judged requirement)
    value, family = str(facts[key]), _family(req, facts)
    if state == "unclear":
        return 3 if key in family or "(the request does not name" in evidence or _narrower(key, req["property"]) else 0
    if key not in family:
        return 1 if " ".join(_words(key)) in variants else 0  # a variant label; else a field that states the whole phrase
    if "(same meaning)" in evidence:
        return 1
    return 3 if "+" in value else 0  # a "/" list is a list like a comma list (accepted 0.89 and 0.87 on the race-6 and race-7 packs)


def _evidence_class(assessment: dict[str, Any] | None) -> int:
    """How far a listing's weakest evidence is from a field naming the property: 0 a field states it (levels 0 and 1), 1 the
    property is absent under an exclusion (level 2), 2 only the title, the description or a loose match states it (level 3).
    The race-11 grader accepted the absent field 0 of 175 times and earlier packs 60 to 80%, so a class ranks, it never rejects."""
    level = (assessment or {}).get("level", 0)
    return 0 if level < 2 else 1 if level == 2 else 2


def _candidate_state(assessment: dict[str, Any] | None) -> str:
    """The state a candidate shows: "fits", "may fit" (it fits but its evidence is contradicted), "unclear", "fails", or
    "not opened" (no details read)."""
    if not assessment:
        return "not opened"
    return "may fit" if assessment["state"] == "fits" and assessment["contradicted"] else assessment["state"]


def _quantity_from(value: Any, quote: Any, word: Any, source: Any) -> int | None:
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or value < 1:
        return None
    value = int(value)
    if value == 1:
        return 1
    quoted, spelled = _words(quote), _words(word)
    if not quoted or any(w not in set(_words(source)) for w in quoted):
        return None
    if len(spelled) != 1 or spelled[0] not in quoted or (spelled[0].isdigit() and int(spelled[0]) != value):
        return None
    return value


def _as_written(amount: Any, text: str) -> Any:
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        return amount
    written = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
    if not written or float(amount) in written:
        return amount
    for factor in (100, 0.01):
        scaled = round(amount * factor, 6)
        if scaled in written:
            return int(scaled) if scaled == int(scaled) else scaled
    return amount


def _strings(value: Any, depth: int = 0) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if depth > 4:
        return []
    items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
    return [s for item in items for s in _strings(item, depth + 1)]


def _event_value(value: Any) -> bool:
    """Shopper text riding beside a tool result: a sentence, a list of sentences, or a list of text-only objects."""
    if isinstance(value, str):
        return len(value.split()) >= 4
    if not isinstance(value, list) or not value:
        return False
    if all(isinstance(x, str) for x in value):
        return True
    return all(isinstance(x, dict) and not _record_id(x) and any(isinstance(v, str) and len(v.split()) >= 4 for v in x.values()) and not any(isinstance(v, (int, float, dict, list)) and not isinstance(v, bool) for v in x.values()) for x in value)


def _unwrap(observation: Any) -> Any:
    """The tool result inside an envelope that also carries shopper text (any key names, any text container)."""
    if isinstance(observation, dict) and len(observation) >= 2:
        inner = [v for v in observation.values() if isinstance(v, dict)]
        rest = [v for v in observation.values() if not isinstance(v, dict)]
        if len(inner) == 1 and rest and all(_event_value(v) for v in rest):
            return inner[0]
    return observation


def _event_texts(observation: Any) -> list[str]:
    """The shopper's sentences carried by a tool result, wherever the world puts them."""
    if not isinstance(observation, dict):
        return []
    inner = _unwrap(observation)
    wrapped = inner is not observation
    texts: list[str] = []
    for value in observation.values():
        if wrapped and value is inner:
            continue
        if wrapped and isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            texts += value[:10]
        elif wrapped and isinstance(value, list):
            texts += [s for x in value if isinstance(x, dict) for s in x.values() if isinstance(s, str)]
    return [t for t in texts if len(t.split()) >= 4]


def _succeeded(observation: Any) -> bool:
    observation = _unwrap(observation)
    flags = [v for v in observation.values() if isinstance(v, bool)] if isinstance(observation, dict) else []
    return True in flags and False not in flags


def _has_list(value: Any, depth: int = 0) -> bool:
    if isinstance(value, list):
        return True
    return isinstance(value, dict) and depth < 3 and any(_has_list(v, depth + 1) for v in value.values())


def _find(value: Any, match: Callable[[dict[str, Any]], Any], limit: int, depth: int = 0) -> list[dict[str, Any]]:
    """The objects inside a JSON value that `match` accepts, in reading order, without looking inside a match (at most
    `limit` levels down)."""
    if depth > limit:
        return []
    if isinstance(value, dict):
        if match(value):
            return [value]
        children: Any = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return []
    return [found for child in children for found in _find(child, match, limit, depth + 1)]


def _cart_lines(view: Any, known: set[str], prices: dict[str, Any] | None = None, tracked: dict[str, Any] | None = None) -> dict[str, int | None]:
    """Cart lines in a cart view: each object naming one known listing, with its count when the numbers beside it tell (a
    number equal to the listing's known price is its price; among numbers still in doubt, the count the code saw go in
    wins; else the smallest, when it is clearly the smallest)."""
    def ids(line: dict[str, Any]) -> set[str]:
        return {v for v in line.values() if isinstance(v, str) and v in known}

    lines: dict[str, int | None] = {}
    for line in _find(view, lambda d: len(ids(d)) == 1, 6):
        ident = ids(line).pop()
        numbers = sorted(v for v in line.values() if isinstance(v, int) and not isinstance(v, bool) and v > 0)
        price = (prices or {}).get(ident)
        counts = [n for n in numbers if n != price] if len(numbers) >= 2 and price is not None and any(n != price for n in numbers) else numbers
        if len(counts) == 1 and len(numbers) >= 2:
            lines[ident] = counts[0]  # the other number is the listing's price
        elif len(counts) >= 2 and (tracked or {}).get(ident) in counts:
            lines[ident] = (tracked or {})[ident]
        else:
            unique = len(counts) >= 2 and counts[0] < counts[1] and counts[0] <= 99
            lines[ident] = counts[0] if unique else None
    return lines


def _fact_size(value: dict[str, Any]) -> int:
    return sum(1 for x in value.values() if isinstance(x, (str, int, float, bool)) or (isinstance(x, list) and x and all(isinstance(y, (str, int, float)) for y in x)) or isinstance(x, dict))


def _flat_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """A listing's facts as text values: list values joined, true/false as yes/no, one level of nested fields named "outer inner"."""
    out: dict[str, Any] = {}
    for key, value in facts.items():
        name = _key_name(key)
        if isinstance(value, bool):
            out[name] = "yes" if value else "no"
        elif isinstance(value, (str, int, float)):
            out[name] = value
        elif isinstance(value, list) and value and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in value):
            out[name] = ", ".join(str(x) for x in value)
        elif isinstance(value, dict):
            for inner, part in value.items():
                if isinstance(part, (str, int, float)) and not isinstance(part, bool):
                    out[f"{name} {_key_name(inner)}"] = part
    return out


def _record_id(record: dict[str, Any]) -> str | None:
    learned = next((ident for key in _ID_KEYS if key in record for ident in [_id_value(record[key], "id" in str(key).lower())] if ident), None)
    if learned:
        return learned
    # a field named "id" before "brand id" (another record's id can come first), any other id-named field last
    keys = sorted((k for k in record if "id" in str(k).lower()), key=lambda k: (_id_rank(k), len(_words(_key_name(k)))))
    return next((record[k] for k in keys if isinstance(record[k], str) and _ID_HINT.match(record[k])), None)


def _listing(record: Any, labels: set[str] | None = None) -> tuple[str, dict[str, Any], str] | None:
    record = _unwrap(record)
    if not isinstance(record, dict):
        return None
    blocks = [(k, v) for k, v in record.items() if isinstance(v, dict) and _fact_size(v) >= 2]
    item = _record_id(record)
    if not blocks or item is None:
        return None
    # the facts are the biggest block (more text values on a tie: an availability block can come first); other blocks are kept as "block field"
    main = max(blocks, key=lambda kv: (_fact_size(kv[1]), sum(isinstance(x, str) for x in kv[1].values())))
    facts = _flat_facts(main[1])
    for key, value in blocks:
        if value is not main[1]:
            for name, part in _flat_facts(value).items():
                facts.setdefault(f"{_key_name(key)} {name}", part)
    text = " ".join(v for v in record.values() if isinstance(v, str) and v != item and v.strip().lower() not in (labels or set()))
    return item, facts, text


def _record_dicts(observation: Any) -> list[dict[str, Any]]:
    """Every listing record in a result: the result itself, or the records inside it (one tool call may open several listings)."""
    return _find(_unwrap(observation), _listing, 4)


def _kinds(spec: Any) -> list[str]:
    """The JSON types a schema allows, looking through anyOf/oneOf, without null."""
    if not isinstance(spec, dict):
        return []
    kinds = spec.get("type")
    kinds = [kinds] if isinstance(kinds, str) else list(kinds or [])
    for key in ("anyOf", "oneOf"):
        for option in spec.get(key) or []:
            kinds += _kinds(option)
    return [k for k in kinds if k != "null"]


def _branch(spec: Any) -> dict[str, Any]:
    """The schema itself, or its first non-null anyOf/oneOf branch when it states no type of its own."""
    if not isinstance(spec, dict):
        return {}
    if spec.get("type") or spec.get("properties") or spec.get("enum"):
        return spec
    for key in ("anyOf", "oneOf"):
        for option in spec.get(key) or []:
            if _kinds(option):
                return _branch(option)
    return spec


def _count_key(schema: Any, taken: Any = ()) -> str | None:
    """The argument a count goes in: an integer slot before a decimal one, a required slot before an optional one."""
    schema = _branch(schema)
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    numbers = [k for k, s in props.items() if k not in taken and {"integer", "number"} & set(_kinds(s)) and "enum" not in _branch(s)]
    pool = [k for k in numbers if "integer" in _kinds(props[k])] or numbers
    return ([k for k in pool if k in required] or pool or [None])[0]


def _fill(schema: Any, item: Any, count: int | None, fixed: dict[str, Any]) -> dict[str, Any] | None:
    """Arguments for one object schema: the fixed values, the listing id in its only free-text slot (the only required one when
    there are several) and the count in its count slot; None when a required argument stays unknown."""
    schema = _branch(schema)
    props = schema.get("properties") or {}
    required = [k for k in schema.get("required") or [] if k in props]
    args = {k: v for k, v in fixed.items() if k in props}
    texts = [k for k, s in props.items() if k not in args and "string" in _kinds(s) and "enum" not in _branch(s)]
    slot = texts if len(texts) == 1 else [k for k in texts if k in required]
    if len(slot) != 1:
        return None
    args[slot[0]] = item
    key = _count_key(schema, args)
    if count is not None and key:
        args[key] = count
    return None if any(k not in args for k in required) else args


def _lines_slot(schema: Any) -> tuple[str, dict[str, Any]] | None:
    """(argument, line schema) when a tool takes the cart as a list of line objects, which sets the whole basket at once."""
    for key, spec in (_branch(schema).get("properties") or {}).items():
        spec = _branch(spec)
        line = _branch(spec.get("items")) if "array" in _kinds(spec) else {}
        if "object" in _kinds(line) or line.get("properties"):
            return key, line
    return None


_ITEM, _QTY, _TEXT, _LINES = "\x00item", "\x00qty", "\x00text", "\x00lines"
_PLACEHOLDERS = {"<ITEM>": _ITEM, "<QTY>": _QTY, "<TEXT>": _TEXT}


def _holds(value: Any, marker: str) -> bool:
    if isinstance(value, dict):
        return any(_holds(v, marker) for v in value.values())
    if isinstance(value, list):
        return any(_holds(v, marker) for v in value)
    return value == marker


def _template(value: Any) -> Any:
    """A model-written call with placeholders as a call format; a list whose single entry holds the item repeats once per cart line."""
    if isinstance(value, str):
        return _PLACEHOLDERS.get(value.strip(), value)
    if isinstance(value, dict):
        return {k: _template(v) for k, v in value.items()}
    if isinstance(value, list):
        items = [_template(v) for v in value]
        if len(items) == 1 and isinstance(items[0], dict) and _holds(items[0], _ITEM):
            return [_LINES, items[0]]
        return items
    return value


def _learned(args: Any, known: set[str]) -> Any:
    """A successful call's arguments as a call format: each known listing id and the one count beside it become slots."""
    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            out = {k: walk(v) for k, v in value.items()}
            ids = [k for k, v in value.items() if isinstance(v, (str, int)) and not isinstance(v, bool) and str(v) in known]
            if len(ids) == 1:
                out[ids[0]] = _ITEM
                counts = [k for k, v in value.items() if isinstance(v, int) and not isinstance(v, bool) and k != ids[0]]
                if len(counts) == 1:
                    out[counts[0]] = _QTY
            return out
        if isinstance(value, list):
            items = [walk(v) for v in value]
            if items and all(isinstance(x, dict) and _holds(x, _ITEM) for x in items):
                return [_LINES, items[0]]
            return items
        return value

    format_ = walk(args)
    return format_ if _holds(format_, _ITEM) else None


def _apply(format_: Any, lines: list[tuple[str, int]], text: str = "") -> Any:
    """Fill a call format: one cart line for a flat call, every line for a list of lines."""
    def fill(value: Any, line: tuple[str, int] | None) -> Any:
        if value == _ITEM:
            return line[0] if line else None
        if value == _QTY:
            return line[1] if line else None
        if value == _TEXT:
            return text
        if isinstance(value, dict):
            return {k: fill(v, line) for k, v in value.items()}
        if isinstance(value, list):
            if len(value) == 2 and value[0] == _LINES:
                return [fill(value[1], one) for one in lines]
            return [fill(v, line) for v in value]
        return value

    return fill(format_, lines[0] if len(lines) == 1 else None)


def _with_counts(args: Any, schema: Any, lines: list[tuple[str, int]]) -> Any:
    """Write each line's count into the schema's count slot, for a call format without a count slot of its own (it was left out,
    so the store's default of one unit would apply, or written as a fixed number)."""
    if not isinstance(args, dict) or not lines:
        return args
    slot = _lines_slot(schema)
    if slot and isinstance(args.get(slot[0]), list):
        key = _count_key(slot[1])
        for entry, (_, count) in zip(args[slot[0]], lines):
            if key and isinstance(entry, dict):
                entry[key] = count
        return args
    key = _count_key(schema)
    if key and len(lines) == 1:
        args[key] = lines[0][1]
    return args


def _worked(observation: Any) -> bool:
    """A tool result that reports success, or carries data rather than only an error text or a false flag."""
    inner = _unwrap(observation)
    if not isinstance(inner, dict):
        return inner is not None and not isinstance(inner, str)
    flags = [v for v in inner.values() if isinstance(v, bool)]
    if flags:
        return True in flags and False not in flags
    return any(v is not None and not isinstance(v, str) for v in inner.values())


def _arg_lines(args: Any, known: set[str]) -> list[tuple[str, int | None]]:
    """The cart lines a call names: each object holding one known listing id, with the one count beside it."""
    def ids(line: dict[str, Any]) -> list[str]:
        return [str(v) for v in line.values() if isinstance(v, (str, int)) and not isinstance(v, bool) and str(v) in known]

    lines: list[tuple[str, int | None]] = []
    for line in _find(args, lambda d: len(ids(d)) == 1, 5):
        ident = ids(line)[0]
        counts = [v for v in line.values() if isinstance(v, int) and not isinstance(v, bool) and str(v) != ident]
        lines.append((ident, counts[0] if len(counts) == 1 else None))
    return lines


def _list_id_key(rows: list[dict[str, Any]]) -> str | None:
    """The field that holds the listing id across a result's rows: present in every row, a different value in each
    (a brand_id repeats); an id-named field first, else a token mixing letters and digits (such as an sku)."""
    if len(rows) < 2:
        return None
    candidates = []
    for key in set.intersection(*(set(row) for row in rows)):
        named = "id" in str(key).lower()
        values = [_id_value(row.get(key), named) for row in rows]
        if all(values) and len(set(values)) == len(values):
            candidates.append((not named, _id_rank(key), len(_words(key)), str(key)))
    return min(candidates)[-1] if candidates else None


def _rows(observation: Any) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []

    def walk(value: Any, depth: int) -> None:
        if depth > 4:
            return
        if isinstance(value, list):
            key = _list_id_key([x for x in value if isinstance(x, dict)])
            if key:
                _ID_KEYS.add(key)
            for item in value:
                ident = (_id_value(item.get(key), "id" in key.lower()) if key and isinstance(item, dict) else None) or (_record_id(item) if isinstance(item, dict) else None)
                if ident:
                    found.append((ident, item))
                else:
                    walk(item, depth + 1)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item, depth + 1)

    walk(_unwrap(observation), 0)
    return found


def _price_stock(row: dict[str, Any]) -> tuple[float | None, bool | None]:
    numbers = [v for v in row.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    flags = [v for v in row.values() if isinstance(v, bool)]
    price = float(numbers[0]) if len(numbers) == 1 else None
    if price is None:
        proven = [row[k] for k in _PRICE_KEYS if isinstance(row.get(k), (int, float)) and not isinstance(row.get(k), bool)]
        price = float(proven[0]) if proven else None  # several numbers: the field a cart total proved
    return price, (flags[0] if len(flags) == 1 else None)


class Episode:
    # The task's budgets, one record each: total (None when unknown), used, source; what is reserved is derived (budget_view).
    # The attributes the rest of the code reads are views of these records.
    max_steps = property(lambda self: self.budgets["steps"]["total"], lambda self, v: self.budgets["steps"].__setitem__("total", v))
    step = property(lambda self: self.budgets["steps"]["used"], lambda self, v: self.budgets["steps"].__setitem__("used", v))
    message_budget = property(lambda self: self.budgets["messages"]["total"], lambda self, v: self.budgets["messages"].__setitem__("total", v))
    messages_used = property(lambda self: self.budgets["messages"]["used"], lambda self, v: self.budgets["messages"].__setitem__("used", v))
    budget_now = property(lambda self: self.budgets["money"]["total"], lambda self, v: self.budgets["money"].__setitem__("total", v))
    # no step limit was stated (by the request, the shopper or the store): the step record still holds the platform's own
    cap_unstated = property(lambda self: self.budgets["steps"]["source"] == "policy",
                            lambda self, v: self.budgets["steps"].__setitem__("source", "policy" if v else "stated"))

    def __init__(self, problem_data: dict[str, Any]):
        self.started = time.monotonic()
        self.budgets: dict[str, dict[str, Any]] = {
            "messages": {"total": None, "used": 0, "source": "default"},
            "steps": {"total": 30, "used": 0, "source": "policy"},
            "money": {"total": None, "used": 0, "source": "unstated"},
            "time": {"total": TASK_SECONDS, "used": 0, "source": "platform"},
        }
        _ID_KEYS.clear()
        _PRICE_KEYS.clear()
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.tools = [t for t in self.policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}} for t in self.tools}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.policy_steps = self.max_steps  # no allowance can go past the platform's own limit
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.order_attempts = 0
        self.refusals = 0
        self.order_checked = False
        self.decline_checked = False
        self.cart_used = False
        self.seen_ids: set[str] = set()
        self.reqs: list[dict[str, Any]] = []
        self.free: list[str] = []
        self.dropped: list[str] = []
        self.heard: set[str] = set()
        self.cart_ids: list[str] = []
        self.records: list[Any] = []
        self.menu_order: list[str] = []
        self.variant_names: dict[str, bool] = {}
        self.relaxed: dict[str, dict[int, bool]] = {}  # requirement -> {product: whether its listings record the property}
        self.lenient: dict[tuple[str, str], bool] = {}
        self.unit_factors: dict[str, dict[str, float | None]] = {}
        self.labels_cache: tuple[int, set[str]] = (-1, set())
        self.variant_failures = 0
        self.last_cart_view: Any = None
        self.cart_state: dict[str, str] = {}
        self.add_signature: str | None = None
        self.asked: list[str] = []
        self.planner_on = True
        self.audited = False
        self.qty_warned: set[str] = set()
        self.items: list[dict[str, Any]] = [{"product": "", "quantity": 1, "category": ""}]  # the products the order needs, in the request's order
        self.item_map: dict[str, int] = {}  # listing -> position in items, -1 for another kind of product
        self.shelf_values: set[str] = set()  # the store's shelf labels, when a tool lists them
        self.held: dict[str, int | None] = {}
        self.cart_counts: str | None = None  # "set" when a cart call sets a line's count (0 takes it out), "add" when it adds to it; None until a result shows which
        self.cart_probe: tuple[str, int, int, int] | None = None  # (listing, count before, count sent, cart version after) of the last cart call
        self.last_echo: dict[str, int | None] | None = None  # the whole basket the last cart call's own result listed
        self.reread_done = False  # the request's products were read a second time (once per episode)
        self.basket_keys: set[str] = set()  # result keys whose lists have shown cart lines
        self.order_errors = 0  # orders the store answered with an error rather than a refusal or a finish
        self.cart_doubt = False  # the cart may not hold what the code believes (an order came back with an error)
        self.code_declining = False  # the code itself sends a decline (its decline checks are done)
        self.cart_changed_step = -1  # the step of the last cart change that the store took
        self.cheaper_opened = False  # the code opened cheaper seen listings before its own order (once per episode)
        self.stating_opened = False  # the code opened seen listings stating the required values before its own order (once)
        self.notes_tool: str | None = None  # the tool that showed a listing's seller notes
        self.notes_changed: set[str] = set()  # listings whose notes arrived since the last call was read
        self.notes_hinted: set[str] = set()  # listings the navigator was told to read the notes of
        self.facts_tools: dict[str, int] = {}  # tool -> one-listing calls of it that returned a property table
        self.page_tools: set[str] = set()  # tools seen to return a listing without one
        self.pool_searched: set[int] = set()  # products the sweep searched the required values of
        self.hop_batch: list[dict[str, Any]] = []  # further calls of the hop move just chosen
        self.facts_read: set[str] = set()  # listings a property table was read for, by any tool
        self.pick_reads: set[str] = set()  # picks the commit read the properties of
        self.note_tools: dict[str, int] = {}  # tool -> listings whose observation by it carried free-text fields another tool's lacked
        self.opened_with: dict[str, set[str]] = {}  # tool -> listings its results showed
        self.commit_paused = False  # the code's commit stopped to read more first (not a failure)
        self.line_pages_sent: set[str] = set()  # ordered lines whose page the code has opened or tried to (page_tool)
        self.called_on: dict[str, set[str]] = {}  # one-listing tool -> listings it was called on (a page with no table is no listing, so only the call tells)
        self.opening_lines = False  # the last check before an order is opening its lines' pages
        self.shelf_proof: tuple[int, bool] = (-1, False)  # (catalog size when looked, a shelf label names the one product)
        self.ready_commit = False  # the code orders early because every product has its verified choice
        self.cart_version = 0
        self.view_version = -1
        self.view_lines: dict[str, int | None] = {}
        self.qty_blocked = -1
        self.order_args: dict[str, Any] | None = {}
        self.decline_args: dict[str, Any] | None = {}
        self.mode_asked = False
        self.judged: dict[str, dict[str, bool]] = {}
        self.catalog: dict[str, dict[str, Any]] = {}
        self.searches: dict[str, dict[str, Any]] = {}
        self.board_said = ""
        self.cheaper_asked: tuple[int, int] | None = None
        self.best: dict[str, str] | None = None
        self.cart_modes: dict[str, dict[str, Any]] = {"add": {}, "remove": {}}
        self.decline_blocks = 0
        self.commit_wanted = False
        self.decline_hinted = False
        self.commit_failures = 0
        self.close_refused: dict[tuple, int] = {}  # closes the store did not take, by close_key: a resend is a decision, and the loop ends
        self.committing = False
        self.last_ask_step = -99
        self.last_ask_facts = 0
        self.ask_due = False
        self.pending_reply = ""
        self.verdicts: dict[str, str] = {}  # listing|ledger version -> why it does not fit ("" when it does)
        self.unanswered: set[str] = set()  # the same keys: a model check asked and not answered (no verdict either way)
        self.verdict_kinds: dict[str, str] = {}  # the same key -> "exact" (the exact checks failed it) or "model" (the model check did)
        self.ledger_version = 0
        self.cart_checked: set[str] = set()
        self.said: list[str] = []
        self.recent: list[str] = []
        self.roles: dict[str, str] = {}
        self.plan: dict[str, Any] = {}
        self.instructions: list[dict[str, Any]] = []  # the instruction ledger: from the request, the shopper and the store
        self.instruction_version = 0                  # grows with every instruction added
        self.other_tools: dict[str, str] = {}         # tools that fit none of the roles -> what they do (setup)
        self.change_count = 0                         # changes the shopper or the store announced (after_change instructions)
        self.last_change: list[str] = []
        self.one_at_a_time_seen = False               # the shopper answers one matter per message in this world
        self.pending_remark: dict[str, str] = {}      # finish kind -> the remark the executor wrote for it
        self.remark_written: str | None = None        # the finish argument the executor's remark went into (for the structural check)
        self.relations: list[dict[str, Any]] = []     # products that must suit whichever listing is chosen for another product
        self.pairs: dict[tuple[str, str], bool | None] = {}  # (hub listing, dependent listing) -> whether the pages show they go together
        self.pair_calls = 0
        self.pair_held: set[Any] = set()
        self.message_log: list[dict[str, Any]] = []   # every message sent: step, cart version, text, the reply's lines
        self.ack_lines: dict[str, str] = {}           # a shopper line -> what it said about our last message (accepted / refused / none)
        self.last_heard: list[str] = []               # the reply lines of the last message sent
        self.last_ack: str | None = None              # the acknowledgement read in the last reply (None: no reading of it)
        self.ready_blocked: tuple | None = None       # (cart, requirements) at which the shopper refused the code's basket
        self.discharging = False                      # a message the gate sends for a due instruction (the reserve pays for it)
        self.message_window = False                   # the executor could not write a due message: the navigator may send one
        self.catch_all_done: int | None = None
        self.best_effort_told = False
        self.executor_calls = 0
        self.gate_holds = 0
        self.tool_calls: dict[str, int] = {}          # store calls made, per tool (what a counted instruction counts)
        self.compared: set[str] = set()               # listings shown side by side by the view tool
        self.cost = 0.0
        self.spent = 0.0  # dollars, priced from token counts (_price)
        self.spend_stopped = False
        self.call_settings: dict[str, dict[str, Any]] = {}  # request settings of the call being made (the setup reading's, SETUP_SETTINGS)
        self.phase = "exploration"  # the spend phase the next call is booked to
        self.phase_spend = {"setup": 0.0, "planner": 0.0, "reserve": 0.0}  # exploration is what these leave of the spend
        self.stop_spent = 0.0  # the task's spend when it reached its stop
        self.cap = TASK_CAP  # the task's budget (set_cap, once the setup has read the task)
        self.reserve_top = 0.0  # a call's spend with the core reserve then owed, when that crossed the cap (reserve_cap)
        self.finish_check = False  # the call now asked is a listing check past the stop: it leaves the whole core reserve
        self.late_call = False  # the call now asked is one the stop still allows (such a reading, a planned question)
        self.call_skipped = False  # the last call asked was not started: its price could cross the cap less the core reserve
        self.answer_tokens = EXPLORE_TOKENS  # the answer allowance the forecast prices the call now asked with
        self.nav_prompt: dict[str, int] = {}  # model -> the prompt tokens of its last navigator turn (cached for the next one)
        self.key_spent = False  # a model call was refused because the run's key is spent (here or, by its mark, in another task)
        self.trace: list[dict[str, Any]] = []
        self.fallback: dict[str, Any] = {"closest": None, "margin": None, "counts": []}  # the request's own rule for when nothing fits (setup)
        self.linked_cache: dict[Any, list[str]] = {}
        self.link_rounds = 0  # commits that went looking for a dependent that goes with the chosen hub
        self.conditional: list[dict[str, Any]] = []  # requirements that depend on a chosen listing ("when"): checked on the basket, never on one listing alone
        self.messages: list[dict[str, Any]] = []
        self.listings_cache: tuple[int, dict[str, tuple[dict[str, Any], str]]] = (-1, {})
        self.cursor_args: dict[str, set[str]] = {}
        self.page_tokens: dict[str, set[str]] = {}
        self.formats: dict[str, dict[str, Any]] = {}
        self.format_asked: set[str] = set()
        self.helped: str | None = None
        self.asked_item: int | None = None  # the product the last planned question named
        self.asked_property: str | None = None  # the property it asked about
        self.asked_open: list[tuple[int | None, str, str]] = []  # (product, property, question) the shopper wanted taken one matter at a time: no answer yet
        self.plan_menu = False  # the planner's prompt carries a property menu
        self.drop_step: int | None = None  # when the shopper last dropped a blocking requirement
        self.drop_held: int | None = None
        self.drop_text = ""
        self.pending_holds: set[int] = set()  # steps at which an order waited for the navigator to find a listing a pointer names
        self.details_opened: set[str] = set()  # listings opened with the details tool (a store may count only those as opened)
        self.hop_state: dict[str, dict[str, Any]] = {}  # pointer description -> the code's search for it: searches sent, steps spent
        self.compare_size = HOP_COMPARE  # listings the compare tool takes at once
        self.hop_sent: set[str] = set()  # calls the code sent towards such listings (none is sent twice)
        self.hop_tried: set[str] = set()  # listings it sent to be read
        self.hop_tools_tried: set[str] = set()  # tools no role covers that were tried on a page of a chain (hop_tool)
        self.thin_pages: set[str] = set()  # identified pages that gave nothing: their details are read, and they are judged once more
        self.open_pages = False  # an order is waiting: the chain's pages a compare only showed are opened first (pages_unopened)
        self.note_slot = False  # a tool answers with an empty field its rows lack: a store's slot for notes, before any note is seen
        self.hops_ended: set[str] = set()  # pointers the code's own look-up has no next call for: their unnoted pages are judged now
        self.hop_failures = 0  # its calls in a row that the store answered with an error (it does not take the code's calls)
        self.stock_out_step: int | None = None  # when a cart line was last found sold out
        self.late_checked: set[str] = set()
        self.evidence_nudged: set[str] = set()  # line|ledger version: an order already held for better-evidenced listings
        self.coverage_holds: set[str] = set()  # searches (and their pages read) an order already waited for
        self.better_fit_held: set[tuple[str, str]] = set()  # (cart line, opened better fit): an order already held for the swap
        self.price_moves: list[str] = []  # listings whose price a result showed changed, not yet judged again
        self.stale_held: set[str] = set()  # cart lines an order already waited for, to read their changed price
        self.decline_rows_done: set[int] = set()  # products a decline was already held for, to open their seen rows
        self.short_held = -1  # the cart version an order missing a product's line was held back for
        self.completing = False  # the code's commit completes the navigator's cart: its verified lines stay
        self.menu_pairs: list[tuple[int, str]] = []
        self.recorded: dict[int, set[str]] = {}  # every property name each product's opened listings record
        self.menu_sections: dict[int, str] = {}  # the planner's menu, per product
        self.menu_values: dict[int, dict[str, list[str]]] = {}  # product -> menu property -> example values, most recorded first
        self.pointer_pages: set[str] = set()  # listings a pointer's description was found to be: read for another listing's value
        self.needs_found: dict[int, int] = {}  # product -> needs the shopper revealed when asked about it
        self.navigator_lines = 0
        self.search_records: set[int] = set()  # ids of stored results that came from the search tool
        self.model_failures: dict[str, int] = {}  # refusals by the service (an HTTP 400/401/403/404) per model
        self.model_ok: set[str] = set()  # models that answered at least once
        self.price_key: str | None = None  # the price field, once a cart total proved it
        self.reference_seen: dict[str, set[str]] = {}  # property|product|reference -> listings already checked against it
        self.search_note = ""
        # what this store showed of its own behaviour, this episode: batch sizes per tool, searches that repeat or ignore an argument
        self.learned: dict[str, Any] = {"limits": {}, "no_limit": set(), "rest": {}, "told": set(), "held": set(), "empty": set(), "dry": {}, "pages": {}, "totals": {}, "cap": {}, "words": {}, "read": set()}
        self.quiet = False                            # store_step is reading: no value check goes with it
        self.review_call: dict[str, Any] | None = None  # the navigator's close waiting for the review's reads
        self.price_arg: str | None = None             # the search argument that sets the highest price (setup)
        self.review_held = False                      # the one review of the task has started
        self.review_reads = 0
        self.store_reads: dict[int, int] = {}         # store_step's calls per product
        self.ceiling_calls: dict[int, list[dict[str, Any]]] = {}  # searches under a pick's price, per product (their next pages are read too)
        self.overlapping: list[tuple[int | None, str]] = []  # (product, property) a single question about which touched several of the shopper's points

    @property
    def quantity(self) -> int:
        """The first (or only) product's quantity."""
        return self.items[0]["quantity"]

    @quantity.setter
    def quantity(self, value: int) -> None:
        self.items[0]["quantity"] = value

    def qty(self, item: int | None) -> int:
        return self.items[item]["quantity"] if isinstance(item, int) and 0 <= item < len(self.items) else self.quantity

    def multi(self) -> bool:
        return len(self.items) > 1

    def product(self, item: int | None) -> str:
        return self.items[item]["product"] if isinstance(item, int) and 0 <= item < len(self.items) else self.items[0]["product"]

    def reqs_for(self, item: int | None, blocking: bool = False) -> list[dict[str, Any]]:
        """The requirements on one product: its own and those about every product."""
        return [r for r in self.reqs if r.get("item") in (None, item) and (r["blocking"] or not blocking)]

    def item_of(self, listing: str) -> int | None:
        """Which requested product a listing is: its position in items, -1 for another kind of product, None while unknown."""
        if listing in self.item_map:
            return self.item_map[listing]
        labels = {" ".join(_words(label)) for label in (self.catalog.get(listing) or {}).get("labels") or set()}
        shelves = [" ".join(_words(item.get("category"))) for item in self.items]
        hits = [index for index, shelf in enumerate(shelves) if shelf and shelf in labels]
        if len(hits) == 1:
            self.item_map[listing] = hits[0]
            return hits[0]
        if not hits and labels & self.shelf_values and all(shelves):
            self.item_map[listing] = -1  # it carries another shelf's label (only when every product's own shelf is known)
            return -1
        if len(self.items) == 1:
            return -1 if self.other_kind(listing) else 0
        return None

    def shelf_named(self) -> bool:
        """A label carried by two or more seen listings names the one requested product: the store shows its shelves in its rows
        and this is the product's (a title is one listing's, so it proves nothing)."""
        size, found = self.shelf_proof
        if found or size == len(self.catalog) or not self.items:
            return found
        product = self.items[0].get("product")
        counts: dict[str, int] = {}
        for entry in self.catalog.values():
            for label in entry.get("labels") or ():
                if _names_product(label, product):
                    counts[label] = counts.get(label, 0) + 1
        found = any(n >= 2 for n in counts.values())
        self.shelf_proof = (len(self.catalog), found)
        return found

    def other_kind(self, listing: str) -> bool:
        """A one-product task: the store's rows show which shelf is the product's, and no label of this listing names the product
        (a listing a pointer led to, or a row of a shop's other shelves)."""
        labels = (self.catalog.get(listing) or {}).get("labels") or ()
        return bool(labels) and self.shelf_named() and not any(_names_product(label, self.items[0].get("product")) for label in labels)

    def row_product(self, listing: str) -> int | None:
        """The product a seen row is, known or read from its own labels, with no model call: an unmapped row of a multi-product
        order is one to read too."""
        index = self.item_of(listing)
        if index is None:
            return self.label_product(listing)
        return index if index >= 0 else None

    def reference_listings(self) -> set[str]:
        """The listings pointers led to: what they are for is their values, not the order."""
        return {i for r in self.reqs for i in r.get("found") or []}

    def product_of(self, listing: str) -> int | None:
        """The position of the requested product a listing is, or None (another kind of product, or not known yet)."""
        index = self.item_of(listing)
        return index if isinstance(index, int) and index >= 0 else None

    def label_product(self, listing: str) -> int | None:
        """The product one of the listing's own labels names word for word (plurals allowed), when exactly one product is named."""
        labels = (self.catalog.get(listing) or {}).get("labels") or set()
        named = {index for index, item in enumerate(self.items) for label in labels if _same_name(label, item.get("product"))}
        if not named:  # ("pumps" names "pumps with a logo pattern")
            named = {index for index, item in enumerate(self.items) for label in labels if _names_product(label, item.get("product"))}
        return named.pop() if len(named) == 1 else None

    def map_items(self, listings: list[str]) -> None:
        """Ask the model, in one batch, which product of a multi-product order each listing is, when shelf labels do not tell.
        A listing whose own label names one product is that product without asking (the model answered the same on every such
        listing of the replica runs)."""
        todo = [i for i in listings if self.item_of(i) is None][:20]
        if not todo or not self.multi() or self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS:
            return
        named = {i: self.label_product(i) for i in todo}
        self.item_map.update({i: index for i, index in named.items() if index is not None})
        todo = [i for i in todo if named[i] is None]
        if not todo:
            self.trace.append({"kind": "item_map", "asked": 0, "mapped": len(named)})
            return
        known = self.listings()
        lines = [f"- {i}: " + " | ".join(sorted((self.catalog.get(i) or {}).get("labels") or set())[:6]) + " | " + (known.get(i, ({}, ""))[1])[:160] for i in todo]
        products = "\n".join(f"{index}: {item['product']}" for index, item in enumerate(self.items))
        result = self.finish_call(lambda: self.judge(ITEM_MAP_PROMPT + "\n\nPRODUCTS IN THE ORDER:\n" + products + "\n\nLISTINGS:\n" + "\n".join(lines), JUDGE_TOKENS, _route("label"))) or {}
        found = result.get("items") if isinstance(result.get("items"), dict) else {}
        for listing in todo:
            value = found.get(listing)
            if isinstance(value, int) and not isinstance(value, bool) and -1 <= value < len(self.items):
                self.item_map[listing] = value
        self.trace.append({"kind": "item_map", "asked": len(todo), "mapped": sum(1 for i in todo if i in self.item_map)})

    def learn_price(self, view: Any, lines: dict[str, int | None]) -> None:
        """The price field, proven by a cart view: the line field whose values times the line counts add up to the shown total."""
        if self.price_key or not lines or not isinstance(view, dict) or any(count is None for count in lines.values()):
            return
        if not any(count > 1 for count in lines.values()):
            return  # with every count at 1 a unit price and a line total add up the same
        totals = [v for v in view.values() if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0]

        def ident(line: dict[str, Any]) -> str | None:
            return next((str(v) for v in line.values() if isinstance(v, (str, int)) and not isinstance(v, bool) and str(v) in lines), None)

        found = [(ident(line) or "", line) for line in _find(view, ident, 4)]
        if not totals or not found:
            return
        for key in set.intersection(*(set(line) for _, line in found)):
            values = [line.get(key) for _, line in found]
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values) or all(v == lines[i] for (i, _), v in zip(found, values)):
                continue
            if any(abs(sum(v * lines[i] for (i, _), v in zip(found, values)) - total) < 0.01 for total in totals):
                self.price_key = key
                _PRICE_KEYS.add(key)
                self.trace.append({"kind": "price_key", "key": key})
                return

    def pointer_terms(self, req: dict[str, Any], listings: dict[str, tuple[dict[str, Any], str]] | None = None) -> tuple[list[str], list[str]]:
        """(names, values) of the description a pending requirement gives of the listing it points at (_pointer_words), with
        the values and property names the opened listings show."""
        listings = self.listings() if listings is None else listings
        values = {w for facts, _ in listings.values() for v in facts.values() for part in _elements(v) if len(_words(part)) <= 2 for w in _words(part)}
        keys = {w for facts, _ in listings.values() for k in facts for w in _words(k)}
        return _pointer_words(req["reference"], self.policy.get("query"), values, keys, {w for shelf in self.shelf_values for w in _words(shelf)})

    def same_pointer(self, old: dict[str, Any], new: dict[str, Any]) -> bool:
        """A requirement that already points (or pointed) at the listing a newly read one names: the same property and product,
        and its first description gives the same names."""
        if not old.get("described") or not _same_name(old["property"], new["property"]) or old.get("item") != new.get("item"):
            return False
        names = lambda text: _pointer_words(text, self.policy.get("query"), set(), set(), set())[0]
        return bool(names(new["reference"])) and names(old["described"]) == names(new["reference"])

    def reference_key(self, req: dict[str, Any]) -> str:
        return f"{req['property']}|{req.get('item')}|{req['reference']}"

    def note_fields(self, name: str, item: str, entry: dict[str, Any], row: dict[str, Any]) -> None:
        """Files the fields one tool's observation of a listing shows; its free-text fields (four words or more) that another
        tool's observation of the same listing lacks (the search's rows when they showed it: they list, so what they lack is
        shown only on opening) are that listing's notes (a seller's notes, where a pointer's value or next page is told), and
        that tool one that shows notes. The search itself never counts as one: it only lists."""
        if name in {self.roles.get(r) for r in ("cart", "cart_view", "message", "order", "decline")} | self.cart_tools():
            return
        shown = entry.setdefault("shown", {})
        shown[name] = ({str(k) for k in row}, {str(k): " ".join(v.split()) for k, v in row.items() if isinstance(v, str) and v != item and len(_words(v)) >= 4})
        search = self.roles.get("search")
        self.note_slot = self.note_slot or (name != search and search in shown and any(v in (None, "") and str(k) not in shown[search][0] for k, v in row.items()))
        for tool, (_, free) in shown.items():
            others = [keys for t, (keys, _) in shown.items() if t != tool and (t == search or search not in shown)]
            extra = [text for key, text in free.items() if any(key not in keys for keys in others)]
            if tool == search or not extra:
                continue
            self.note_tools.setdefault(tool, set()).add(item)
            if [n for n in extra if n not in (entry.get("notes") or [])]:
                entry["notes"] = list(dict.fromkeys((entry.get("notes") or []) + extra))[-6:]
                for seen in self.reference_seen.values():
                    seen.discard(item)  # its page now says more: a pointer is checked against it again
                self.notes_changed.add(item)

    def one_listing_tools(self) -> list[str]:
        """The tools no role covers whose one argument is a listing's id."""
        return [n for n in self.other_tools if len(_branch(self.schemas.get(n)).get("properties") or {}) == 1 and _fill(self.schemas.get(n), "x", None, {})]

    def reader(self) -> str | None:
        """The tool that reads a listing's properties: the one seen to return a property table most often; else the details
        role, unless that returned a listing without one and another tool reading one listing is still untried (a store may
        split a page and its properties over two tools, and the setup can name only one)."""
        details = self.roles.get("details")
        if self.facts_tools:
            return max(sorted(self.facts_tools), key=lambda t: (self.facts_tools[t], t == details))
        if details in self.page_tools:
            return next((n for n in self.one_listing_tools() if n not in self.page_tools), details)
        return details

    def learn_reader(self, name: str, args: dict[str, Any], observation: Any, opened: list[Any]) -> None:
        """Which one-listing tool shows properties, from what its results hold."""
        if opened:
            self.facts_read.update(listing[0] for listing in opened)
        if name != self.roles.get("details") and name not in self.other_tools:
            return
        if len(args) != 1 or not isinstance(next(iter(args.values())), str):
            return
        self.called_on.setdefault(name, set()).add(next(iter(args.values())))
        if opened:
            self.facts_tools[name] = self.facts_tools.get(name, 0) + 1
        elif isinstance(observation, dict) and _worked(observation) and _record_id(observation):
            self.page_tools.add(name)

    def hop_limit(self) -> int:
        """Steps one pointer's look-up may take: a task with room for it reads more of a shop's rows."""
        return min(16, max(HOP_STEPS, (self.max_steps - self.endgame_steps()) // 4))

    def hop_tool(self) -> str | None:
        """A tool no role covers that reads one listing (its one argument the listing's id), not tried on a page yet: a store may
        show a listing's notes there and count only that as opening it (race 7: the offer; 3 of its 32 chains count nothing else)."""
        for name in self.other_tools:
            props = _branch(self.schemas.get(name)).get("properties") or {}
            if name not in self.hop_tools_tried and len(props) == 1 and _fill(self.schemas.get(name), "x", None, {}):
                return name
        return None

    def pointer_tool(self) -> str | None:
        """The tool that opens a page of a pointer chain: one seen to show listings' notes (the details tool among them first,
        then a tool reading one listing, then the compare tool; the one seen doing so on most listings), before the compare
        tool or when none is seen yet a tool reading one listing that is still to be tried (hop_tool), else the details tool."""
        details, view = self.roles.get("details"), self.roles.get("view")
        tools = sorted(self.note_tools, key=lambda t: (t != details, t == view, -len(self.note_tools[t]), t))
        if tools and tools[0] != view:
            return tools[0]
        return self.hop_tool() or (tools[0] if tools else details)

    def page_call(self, item: str, opener: str | None = None) -> dict[str, Any] | None:
        """The call that opens one listing's page with the pointer tool."""
        opener = opener or self.pointer_tool()
        if opener == self.roles.get("view"):
            return self.view_call(opener, [item])
        return self.checked_call(opener, _fill(self.schemas.get(opener), item, None, {}))

    def page_opener(self) -> str | None:
        """The tool that opens a chain's page for the record: the pointer tool, but never the compare tool while a details tool
        exists: a store grades pages as opened by the details tool on 83 of 90 race-9 tasks (compare_counts), and a chain read
        through a compare alone left them unopened."""
        opener = self.pointer_tool()
        if opener != self.roles.get("view"):
            return opener
        return next(iter(sorted(self.hop_tools_tried)), None) or self.roles.get("details") or opener  # (a tool reading one listing, tried on a page without notes)

    def pages_unopened(self) -> list[str]:
        """Pages of the chains followed so far that the pointer tool has not opened."""
        opener = self.page_opener()
        done = self.details_opened if opener == self.roles.get("details") else self.opened_with.get(opener, set())
        pages = dict.fromkeys(i for r in self.reqs for i in r.get("found") or [] if i not in done)
        return [i for i in pages for call in [self.page_call(i, opener)] if call and call["name"] + _compact(call["args"]) not in self.hop_sent]

    def page_unread(self, item: str) -> bool:
        """The store shows listings' notes through some tool, and none showed this listing's: a pointer's value, its next page or
        a rule of which value goes with which is told there, so the page is not read yet (on the race-7 replica 5 chain pages
        were judged from their facts alone, and 3 chains ended there)."""
        entry = self.catalog.get(item) or {}
        return bool(self.note_tools) and not entry.get("notes") and not set(self.note_tools) & set(entry.get("shown") or {})

    def pointer_listings(self) -> dict[str, tuple[dict[str, Any], str]]:
        """The opened listings, with the listings known only from a tool that shows notes (a compare's or an offer's rows
        without a details block): a pointer's page may give its value in the notes alone."""
        listings = self.listings()
        for item, entry in self.catalog.items():
            if item not in listings and entry.get("notes"):
                text = str(entry.get("text") or "")
                listings[item] = ({}, (text + " " + " ".join(n for n in entry["notes"] if n not in text)).strip())
        return listings

    def reference_candidates(self, req: dict[str, Any], ids: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> list[str]:
        """Opened listings that may be the one a requirement points at: those carrying a name the description gives (a capitalised
        word such as a maker's or shop's name, often found only on that listing's page and a few look-alikes), else those
        sharing one of its rarest words among the opened listings."""
        named, asked = _words(req["reference"]), _words(self.policy.get("query"))
        # a word the request also uses outside the description (the product's own name, "the") tells no listing apart
        wanted = {w for w in named if len(w) >= 3 and asked.count(w) <= named.count(w)}
        names, values = self.pointer_terms(req, listings)
        docs = {i: set(_words(text + " " + " ".join(str(v) for v in facts.values()) + " " + " ".join((self.catalog.get(i) or {}).get("labels") or []))) for i, (facts, text) in listings.items()}
        # the shop's look-alikes (same-seller decoys) state fewer of the described values: the pages stating most go first, so a
        # batch the judgement sees holds the page that confirms every detail (which alone it takes)
        stated = lambda i: -sum(1 for v in values if any(_same_word(v, d) for d in docs.get(i, set())))
        if names:  # every name: the shop's own listings (of the shelf it names), not another shop's look-alike
            return sorted([i for i in ids if all(any(_same_word(n, d) for d in docs.get(i, set())) for n in names)], key=stated)
        counts = {w: sum(1 for d in docs.values() if w in d) for w in wanted}
        present = [n for n in counts.values() if n]
        if not present:
            return []
        rare = {w for w, n in counts.items() if 0 < n <= max(3, min(present))}
        return sorted([i for i in ids if docs.get(i, set()) & rare], key=stated)

    def listing_block(self, item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> str:
        facts, text = listings[item]
        entry = self.catalog.get(item) or {}
        labels = " | ".join(sorted(entry.get("labels") or set())[:8])
        # the price is a number, so the text (a page's strings) leaves it out: a description may still tell a listing by it
        # ("the cheaper of the two ... that X has listed")
        price = f"; price: {entry['price']:g}" if isinstance(entry.get("price"), (int, float)) else ""
        return f"{item} [labels: {labels}{price}]\nTEXT: {text[:2500]}\nDETAILS: {_compact(facts)[:1500]}"

    def without_pointer_details(self, reqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Requirements without those that only repeat a detail identifying the listing a pointer describes ("…X's pepper,
        with weight 1050 g" read as a weight requirement): every word of their property and values is in that description."""
        described = [set(_words(r["reference"])) for r in reqs + self.reqs if r.get("pending") and r.get("reference")]
        def inside(r: dict[str, Any]) -> bool:
            words = set(_words(r["property"])) | {w for v in r["values"] for w in _words(v)}
            return any(words and words <= d for d in described)
        kept = [r for r in reqs if r.get("pending") or not inside(r)]
        if len(kept) < len(reqs):
            self.trace.append({"kind": "pointer_details_dropped", "requirements": [_describe(r) for r in reqs if r not in kept]})
        return kept

    def resolve_references(self, opened: list[str]) -> str:
        """Find, among opened listings, the listing a pending requirement points at and take its value; when that listing's page
        sends the value on to another listing, point the requirement there (a chain of pages) and look among the opened ones."""
        notes = []
        listings = self.pointer_listings()
        query = str(self.policy.get("query") or "")
        for req in [r for r in self.reqs if r.get("pending")]:
            ids = [i for i in opened if i in listings]
            for _ in range(8):  # batches of five; a redirect can point at a listing opened some time ago
                key = self.reference_key(req)
                seen = self.reference_seen.setdefault(key, set())
                # (a page of the chain so far is never the next one, even opened again)
                pages = [i for i in self.reference_candidates(req, ids, listings) if i not in seen and i not in (req.get("found") or []) and not self.page_unread(i)]
                noted = [i for i in pages if (self.catalog.get(i) or {}).get("notes")]
                ended = key in self.hops_ended  # the code's look-up has no next call: the pages read so far are all there is
                if (self.note_tools or self.note_slot) and not noted and not ended and pages:
                    break  # every chain page has a note (441 of 442, races 5-10; notes are on 1-3% of listings): rows are left to read first
                late = self.spent_out()  # past the stop a judgement is made only while the allowance has room
                last = ended and late and not self.late_room()
                no_model = (late and not self.late_room()) or self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS
                try:
                    pick = self.stated_page(req, noted or (pages if last else []), listings, last)
                    reading = self.page_reading(req, pick, listings) if pick else None
                except Exception as exc:  # (the judgement below still reads the page)
                    self.trace.append({"kind": "hop_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
                    pick, reading = None, None
                spare = None
                if reading and not no_model:  # the model tells a page from its look-alikes (42 of 45 against the code's 63 of 73): the code's reading only backs it up
                    pick, reading, spare = None, None, reading
                fresh = [pick] if pick else (noted or pages)[:5]  # (the page whose reading needs a model is judged alone)
                if reading:
                    late = False  # read by code: no call, so nothing the stop or the clock holds back
                elif not fresh or self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS or (late and not self.late_room()):
                    self.thin_pages.update(i for i in noted if not listings[i][0] and i not in self.details_opened and i not in self.facts_read)  # (a store call tells a page's details, which the code can read)
                    break  # (the listings stay unjudged: a later look may still judge them)
                seen.update(fresh)
                prompt = REFERENCE_PROMPT + "\n\nREQUEST:\n" + query + f"\n\nREFERENCE: {req['reference']}\nPROPERTY: {req['property']}\n\nLISTINGS:\n" + "\n\n".join(self.listing_block(i, listings) for i in fresh)
                self.late_call = late
                try:
                    result = reading or self.judge(prompt, JUDGE_TOKENS, _route("reference_late" if late else "reference")) or spare
                    if result is None and self.call_skipped:
                        seen.difference_update(fresh)  # not started for its price: the listings stay unjudged for a later look
                        break
                    result = result or {}
                    if "value" not in result and "next" not in result and result.get("id") in fresh and (not late or self.late_room()):
                        # an answer cut off by the token cap (every whole answer gives both) that names a listing: it may have
                        # been about to give the value or the next page, so the batch is judged once more (on the paid runs' 15
                        # cut-off answers the 5 naming a listing all named the hop): by the route's next model, past the stop
                        # on the late route again (its capped answers keep the allowance's worst case)
                        self.trace.append({"kind": "reference_cut", "id": result.get("id"), "checked": fresh})
                        result = self.judge(prompt, JUDGE_TOKENS, _route("reference_late") if late else _route("reference")[1:]) or {}
                finally:
                    self.late_call = False
                ident, onward = result.get("id"), result.get("next")
                value, partner = _one_value(result.get("value")), _one_value(result.get("paired"))
                if isinstance(ident, str) and ident in listings:  # a value the page does not state (such as "not specified") is none of its values
                    page_facts, page_text = listings[ident]
                    stated = " ".join([str(page_text)] + [f"{k} {v}" for k, v in (page_facts or {}).items()])
                    if _words(value) and not _has_words(stated, [str(value)]):
                        self.trace.append({"kind": "reference_value_unstated", "id": ident, "value": value})
                        value = None
                    if _words(partner) and not _has_words(stated, [str(partner)]):
                        self.trace.append({"kind": "reference_value_unstated", "id": ident, "value": partner})
                        partner = None
                details = result.get("details") if isinstance(result.get("details"), dict) else {}
                self.trace.append({"kind": "reference", "reference": req["reference"], "checked": fresh, "id": ident, "value": value, "paired": partner, "next": onward, "details": details})
                chosen = partner if req["op"] == "paired" else value  # the page's facts are the model's, the relation is the reader's
                # it is that listing only when the judgement names the details it checked and finds every one confirmed (the
                # candidates carry every name the description gives): a look-alike neither sets the value nor sends the chain on
                confirmed = isinstance(ident, str) and ident in fresh and bool(details) and False not in details.values()
                if confirmed and not _words(chosen) and not (isinstance(onward, str) and _words(onward)):
                    tool = self.pointer_tool() if self.note_tools else self.notes_tool
                    if tool and not (self.catalog.get(ident) or {}).get("notes") and ident not in self.notes_hinted:
                        self.notes_hinted.add(ident)  # it matches but its page gives no value: its seller's notes may
                        notes.append(f"[reference] {ident} matches {req['reference']}, but its page gives no {req['property']}: read its seller's notes with {tool}.")
                    if ident not in self.details_opened and ident not in self.facts_read:
                        self.thin_pages.add(ident)  # its facts may state what its notes leave out (the own value a rule of pairs needs)
                if not confirmed or not (_words(chosen) or _words(value) or (isinstance(onward, str) and _words(onward))):
                    continue  # none of these (or a near miss that fails a detail): the next unchecked ones
                self.pointer_pages.add(ident)  # read for the value it gives (or the page it sends on to), not as a candidate
                if ident not in self.thin_pages:  # (a page to be judged again joins it then)
                    req.setdefault("found", []).append(ident)  # the chain so far (each hop is opened with the details tool)
                if not _words(chosen) and isinstance(onward, str) and _words(onward) and req.get("hops", 0) < 10:
                    req["hops"] = req.get("hops", 0) + 1
                    req["reference"] = " ".join(onward.split())[:160]
                    # no page of the chain so far is the one it sends on to (each sends the value onward, a loop never ends)
                    self.reference_seen.setdefault(self.reference_key(req), set()).update(req["found"])
                    self.ledger_version += 1
                    find = "It is looked up for you next" if self.looked_up(req) else "Find that listing and open its details"
                    notes.append(f"[reference] {ident} sends {req['property']} on to another listing: {req['reference']}. {find}; only a listing whose page shows every one of those details is it (a look-alike's page may point elsewhere: ignore it).")
                    ids = list(listings)  # the listing it sends to may be open already
                    continue
                if not _words(chosen):
                    break  # (a page that pairs nothing with its value leaves the requirement pending: never its own value, never set apart)
                if req["op"] == "within":
                    found = [(n, u) for n, u in _measures(chosen) if not req["unit"] or (u and _same_unit(u, req["unit"]))]
                    found = found or [(n, u) for n, u in _measures(chosen) if not u]  # a bare number is in the requirement's own unit
                    if not found or not req["values"]:
                        break  # no amount in the requirement's unit: not the value it points at
                    center, unit = found[0]
                    req.update({"op": "between", "values": [center - req["values"][0], center + req["values"][0]], "unit": req["unit"] or unit})
                else:  # that value or the one its page pairs with it, or (a pointer that sets the value apart) any value but that one
                    req.update({"op": "not" if req["op"] == "not" else "is", "values": [" ".join(str(chosen).split())[:80]]})
                req.update({"pending": False, "blocking": True, "resolved_from": ident})
                self.ledger_version += 1
                notes.append(f"[reference] {ident} is {req['reference']}, so the requirement is now {_describe(req)}.")
                break
        return " ".join(notes)

    def stated_page(self, req: dict[str, Any], pages: list[str], listings: dict[str, tuple[dict[str, Any], str]], last: bool = False) -> str | None:
        """The page a pointer describes, told by code: the only one of these whose details (not its text: a look-alike's note
        says "not 6000 mah") state every value and number the description gives, and no other value for a property it names.
        last (no model can be asked and no page is left to read): the best of them. None leaves it to the judgement."""
        said = _value_words(req["reference"])
        values = set(self.pointer_terms(req, listings)[1]) | {w for w in said if w[0].isdigit()}

        def stated(item: str) -> int:
            facts = listings[item][0]
            for key, value in facts.items():  # ("motorcycle model thunder" against its own "motorcycle model: raider")
                if len(_value_words(key)) >= 2 and f" {' '.join(_value_words(key))} " in f" {' '.join(said)} " and not set(_value_words(value)) & set(said):
                    return -1
            return sum(1 for v in values if any(_same_word(v, h) for x in facts.values() for h in _value_words(x)))
        counts = {i: stated(i) for i in pages if listings[i][0]}
        most = max(counts.values(), default=-1)
        top = [i for i, n in counts.items() if n == most]
        if values and most == len(values) and (len(top) == 1 or last):
            return top[0]
        return top[0] if last and top and most >= 0 else None

    def page_reading(self, req: dict[str, Any], item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> dict[str, Any] | None:
        """What a page gives for a pointer's property, as the reference judgement answers it, read by code: the listing its note
        sends on to, else the value its details state under the property; None when only a model can tell (a value paired with
        its own, or told in prose alone). The model reads a page at 1,200 prompt tokens; this needs none, also past the stop."""
        facts = listings[item][0]
        onward = _onward_words(" ".join((self.catalog.get(item) or {}).get("notes") or []), req["property"], {w for v in facts.values() for w in _words(v)})
        stated = next((str(v) for v in _family(req, facts).values() if str(v).strip()), None)
        if not onward and (req["op"] == "paired" or not stated):
            return None
        self.trace.append({"kind": "reference_read", "id": item, "value": None if onward else stated, "next": onward or None})
        return {"id": item, "details": {"read by code": True}, "value": None if onward else stated, "paired": None, "next": onward or None}

    def row_words(self, item: str) -> list[str]:
        entry = self.catalog.get(item) or {}
        return _words(str(entry.get("text") or "") + " " + " ".join(entry.get("labels") or []))

    def hop_rows(self, req: dict[str, Any], names: list[str], values: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> list[str]:
        """Seen, unread listings that may be the one a pending requirement points at, not yet judged against its description,
        and those opened without their notes (page_unread): rows naming every one of its names, then the other rows of the
        code's searches for it; the rows stating more of the described values first, then more of the description's other
        words (nothing opened yet may tell which words are values), then in the order they were seen."""
        key = self.reference_key(req)
        seen = self.reference_seen.get(key, set())
        described = [w for w in dict.fromkeys(_words(req["reference"])) if w not in names and (len(w) >= 3 or w.isdigit())]
        searched = {i for q in (self.hop_state.get(key) or {}).get("queries", []) for i in (self.searches.get(q) or {}).get("seen", [])}
        ranked = []
        for order, (item, entry) in enumerate(self.catalog.items()):
            unread = item in listings and self.page_unread(item)
            if not entry.get("text") or (not unread and (item in listings or item in seen or item in self.hop_tried)):
                continue
            have = self.row_words(item) + [w for v in (listings[item][0] if item in listings else {}).values() for w in _value_words(v)]  # (a shop's rows differ by their properties)
            named = all(any(_same_word(n, h) for h in have) for n in names)
            stated = lambda words: sum(1 for v in words if any(_same_word(v, h) for h in have))
            if named or item in searched:
                ranked.append((not named, -stated(values), -stated(described), order, item))
        return [item for *_, item in sorted(ranked)]

    def search_key(self, name: str, args: dict[str, Any]) -> str:
        """The key observe() files a search under (its arguments without the paging ones)."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        paging = {k for k, spec in props.items() if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec and k != getattr(self, "price_arg", None)} | self.cursor_args.get(name, set())
        return name + _compact({k: v for k, v in args.items() if k not in paging})

    def search_call(self, name: str, text: str) -> dict[str, Any] | None:
        """A search for these words: the tool's only free-text argument, else an earlier search's arguments with its words
        replaced (a store whose search also needs a mode or a shelf); words past the number the store was seen to read stay out."""
        most = self.learned["words"].get(name)
        text = " ".join(text.split()[:most]) if most else text
        call = self.checked_call(name, _fill(self.schemas.get(name), text, None, {}))
        if call:
            return call
        for query in reversed(list(self.searches.values())):
            args = query.get("args") or {}
            slots = [k for k, v in args.items() if isinstance(v, str) and v == query["text"]]
            if query.get("tool") == name and len(slots) == 1 and (self.search_kind(name, args), slots[0]) not in self.learned.setdefault("blind", set()):
                return self.checked_call(name, {**args, slots[0]: text})
        return None

    def next_page(self, call: dict[str, Any]) -> dict[str, Any] | None:
        """The call that reads a search's next page: the page bookmark its last result handed out, in the argument that takes
        bookmarks back (the one seen doing so, else the tool's only other optional text argument)."""
        query = self.searches.get(self.search_key(call["name"], call["args"]))
        if not query or query["ended"] or len(query.get("follow") or []) != 1:
            return None
        total = query["totals"][-1] if query["totals"] else None
        if total is not None and len(query["seen"]) >= total:
            return None
        schema = _branch(self.schemas.get(call["name"]))
        props, required = schema.get("properties") or {}, schema.get("required") or []
        slots = sorted(self.cursor_args.get(call["name"]) or ()) or [k for k, s in props.items() if k not in call["args"] and k not in required and "string" in _kinds(_branch(s)) and "enum" not in _branch(s)]
        return self.checked_call(call["name"], {**call["args"], slots[0]: query["follow"][0]}) if len(slots) == 1 else None

    def batch_slot(self, name: str) -> str | None:
        """The one argument of a tool that takes a list of listing ids."""
        props = _branch(self.schemas.get(name)).get("properties") or {}
        slots = [k for k, s in props.items() if "array" in _kinds(_branch(s)) and "string" in _kinds(_branch(_branch(s).get("items")))]
        return slots[0] if len(slots) == 1 else None

    def batch_size(self, name: str) -> int | None:
        """How many listings one call of this tool takes: what the store showed once a larger list failed, within the schema's
        own maximum (a schema's maximum alone was too high on 3 of the 5 worlds with a limit, and absent on one)."""
        seen = self.learned["limits"].get(name) or {}
        slot = self.batch_slot(name)
        most = _branch((_branch(self.schemas.get(name)).get("properties") or {}).get(slot)).get("maxItems") if slot else None
        sizes = [x for x in ((seen.get("ok") or seen["bad"] - 1) if seen.get("bad") else None, most if isinstance(most, int) and most > 0 else None) if x]
        return min(sizes) if sizes else None

    def view_call(self, name: str, items: list[str]) -> dict[str, Any] | None:
        """A call of the tool that opens several listings at once: its one argument taking a list of ids."""
        slot, most = self.batch_slot(name), self.batch_size(name)
        return self.checked_call(name, {slot: items[:most] if most else items}) if slot else None

    def split_batch(self, call_id: str, name: str, args: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
        """One call as the calls the store takes: a list longer than the size learned goes in parts, no step wasted on a failure,
        as many parts as the reading phase's steps leave room for (a name repeated counts once)."""
        slot = self.batch_slot(name)
        ids = args.get(slot) if slot and self.learned["limits"].get(name, {}).get("bad") else None
        size = self.batch_size(name)
        if not isinstance(ids, list) or not size or len(set(ids)) <= size:
            return [(call_id, name, args)]
        ids = list(dict.fromkeys(ids))
        parts = [ids[i : i + size] for i in range(0, len(ids), size)]
        room = max(1, self.steps_left() - self.th('reading_ends'))
        if room < len(parts):
            self.search_note += f" [store] {name} takes {size} listings at a time: not sent, no steps for them: {', '.join(i for part in parts[room:] for i in part)}."
        return [(call_id, name, {**args, slot: part}) for part in parts[:room]]

    def learn_batch(self, name: str, args: dict[str, Any], observation: Any) -> list[list[str]]:
        """Read one call of a list tool for the store's limit, and return the lists to send next. A list of known listings that
        shows none of them, or only its first ones, was too long, whatever the failure says (schemas and texts misled on every
        world with a limit): half goes next, while the reading phase has steps; what is left follows once a size worked. One
        listing that fails is no limit, and neither is a list of ids the store never showed."""
        slot = self.batch_slot(name)
        ids = list(dict.fromkeys(args.get(slot) or [])) if slot and name not in self.cart_tools() else []
        limits, rest = self.learned["limits"], self.learned["rest"]
        if not ids or name in self.learned["no_limit"]:
            return []
        shown: set[str] = set()
        _collect_ids(observation, shown)
        got = [i for i in ids if i in shown]
        seen = limits.setdefault(name, {"ok": None, "bad": None})
        known = all(i in self.catalog or i in self.seen_ids for i in ids)
        if got == ids[: len(got)] and got and (len(got) == len(ids) or known):
            seen["ok"] = max(seen["ok"] or 0, len(got))
            if len(got) < len(ids):  # the store showed only the first ones
                seen["bad"] = min(seen["bad"] or len(ids), len(got) + 1)
                rest[name] = ids[len(got) :] + rest.get(name, [])
            elif seen["bad"] and seen["ok"] >= seen["bad"]:
                seen["bad"] = None
            if seen["bad"]:
                self.tell(f"limit|{name}", f"[store] {name} failed on {seen['bad']} listings and worked on {seen['ok']}: it takes {seen['ok']} at a time here (its failure text was not about the listings); the code splits longer lists.")
                self.compare_size = min(self.compare_size, seen["ok"])
            waiting = rest.pop(name, [])
            size = self.batch_size(name) or len(waiting)
            parts = [waiting[i : i + size] for i in range(0, len(waiting), size)]
            return [] if self.call_cap(name) else parts[: max(0, self.steps_left() - self.th('reading_ends'))]
        if got or not known:
            return []
        if len(ids) == 1:
            if not seen["ok"]:
                limits.pop(name, None)  # down to one listing and still failing: the failure was never about the size
                self.learned["no_limit"].add(name)
            rest.pop(name, None)
            return []
        if seen["ok"] and len(ids) <= seen["ok"]:
            return []  # this many worked before: something else failed
        seen["bad"] = min(seen["bad"] or len(ids), len(ids))
        self.compare_size = min(self.compare_size, seen["bad"] - 1)
        if self.call_cap(name) or self.steps_left() <= self.th('reading_ends'):
            rest.pop(name, None)
            self.tell(f"limit|{name}", f"[store] {name} failed on {len(ids)} listings: try fewer at a time (its failure text may not be about the listings).")
            return []
        size = max(seen["ok"] or 0, (len(ids) + 1) // 2)
        rest[name] = ids[size:] + rest.get(name, [])
        return [ids[:size]]

    def tell(self, key: str, text: str) -> None:
        """One note to the navigator, once: every word stays in the conversation."""
        if key not in self.learned["told"]:
            self.learned["told"].add(key)
            self.search_note = (self.search_note + " " if self.search_note else "") + text

    def paged(self, name: str, args: dict[str, Any]) -> bool:
        """The call reads a later page: an argument takes a bookmark or page number the tool handed out."""
        tokens, cursors = self.page_tokens.get(name) or set(), self.cursor_args.get(name) or set()
        return any(not isinstance(v, bool) and isinstance(v, (str, int)) and v not in (0, 1) and (k in cursors or str(v) in tokens) for k, v in args.items())

    def search_tools(self) -> set[str | None]:
        """The tools seen to work as searches: not a list tool or a reader whose rows were filed like a search's."""
        others = {self.roles.get(r) for r in ("details", "view", "message", "cart_view", "order", "decline")} | self.cart_tools() | {self.notes_tool}
        return {t for t in {q["tool"] for q in self.searches.values()} | {self.roles.get("search")} if t not in others and not self.batch_slot(t)}

    def search_kind(self, name: str, args: dict[str, Any]) -> str:
        """A search's method: its tool and its mode (an argument with fixed values that is no shelf)."""
        props = _branch(self.schemas.get(name)).get("properties") or {}
        return name + _compact({k: v for k, v in sorted(args.items()) if isinstance(_branch(props.get(k)).get("enum"), list) and " ".join(_words(v)) not in self.shelf_values})

    def search_target(self, args: dict[str, Any]) -> int | None:
        """The product a search is for: the only one, or the one its words name best (its shelf, its name, its required values);
        None when they do not tell."""
        if not self.multi():
            return 0
        said = [w for v in args.values() if isinstance(v, str) for w in _words(v)]
        scores = []
        for index, item in enumerate(self.items):
            own = _words(item.get("product")) + _words(item.get("category")) + [w for r in self.reqs_for(index) for v in r["values"] for w in _words(v)]
            scores.append(sum(1 for w in said if any(_same_word(w, x) for x in own)))
        top = max(scores, default=0)
        return scores.index(top) if top and scores.count(top) == 1 else None

    def unopened_rows(self, index: int) -> list[str]:
        """Rows of one product seen but not opened that may still fit, those its required values' searches showed and the cheapest first."""
        listings, pool = self.listings(), self.pool(index)
        rows = [i for i, e in self.catalog.items() if i not in listings and self.row_product(i) == index and e.get("stock") is not False
                and not self.over_budget(i, index) and not self.row_conflict(i, index, listings)]
        return sorted(rows, key=lambda i: (i not in pool, self.catalog[i].get("price") is None, self.catalog[i].get("price") or 0, i))

    def search_instead(self, name: str, args: dict[str, Any], target: int | None) -> str:
        """What the store has shown would work instead of one more search: rows not opened, the next page of the search that
        brought the most new rows, another mode of this tool. "" when it has shown nothing."""
        options = []
        rows = [i for index in ([target] if target is not None else range(len(self.items))) for i in self.unopened_rows(index)][:3]
        if rows:
            options.append("open " + ", ".join(rows) + " (seen, not opened)")
        queries = sorted([q for q in self.searches.values() if not q["ended"] and q.get("new")], key=lambda q: -q["new"])
        page = next((c for q in queries for c in [self.next_page({"name": q["tool"], "args": q["args"]})] if c), None)
        if page:
            options.append(f"read the next page of the search that found the most: {_compact(page['args'])[:120]}")
        props = _branch(self.schemas.get(name)).get("properties") or {}
        others = [(k, o) for k, v in args.items() if " ".join(_words(v)) not in self.shelf_values for o in _branch(props.get(k)).get("enum") or [] if o != v]
        shelves = [t for t, spec in self.schemas.items() if t != name and any(" ".join(_words(o)) in self.shelf_values for x in (_branch(spec).get("properties") or {}).values() for o in _branch(x).get("enum") or [])]
        if others or shelves:
            options.append(f"use {others[0][0]}={others[0][1]!r}" if others else f"use {shelves[0]}")
        return "; or ".join(options)

    def search_held(self, name: str, args: dict[str, Any]) -> str:
        """Why a search is held back (no step used), once for each state of things: the same call found nothing before; a page
        past the end of its search; or searches of this method for this product have DRY_SEARCHES times in a row added no row while the store
        has shown something better to do. The repeat is of a result, not of arguments: a page of a search that still adds rows
        passes, and so does the same call sent again."""
        learned = self.learned
        if self.committing or name not in self.search_tools():
            return ""
        signature = name + _compact(args)
        target = self.search_target(args)
        method = (self.search_kind(name, args), target, self.ledger_version)  # (a requirement that changed is a new reason to search)
        count = learned["dry"].get(method, 0)
        query = self.searches.get(self.search_key(name, args))
        page = self.paged(name, args)
        if signature in learned["empty"]:
            mark, reason = (signature, "empty"), "this exact search showed no rows before and nothing has changed"
        elif page and query and query.get("stop"):
            mark, reason = (signature, "end"), "this search has no more pages"
        elif not page and count >= DRY_SEARCHES:
            mark, reason = (method, count), f"the last {count} searches of this kind added no listing you had not seen"
        else:
            return ""
        instead = self.search_instead(name, args, target)
        if mark in learned["held"] or (mark == (method, count) and not instead):
            return ""  # (with nothing better known a search may still find something)
        learned["held"].add(mark)
        return f"Not run (no step used): {reason}. " + (f"Instead: {instead}. " if instead else "") + "If you still want it, send it again."

    def learn_search(self, name: str, args: dict[str, Any], observation: Any, known: set[str]) -> None:
        """Read one search's result for how this store searches, by comparing the arguments with what came back; a behaviour
        nobody listed is met the same way, since nothing here names one: rows that add nothing, a page that ends a search, an
        argument that changed nothing, totals that stop at a round cap or that more words raise, rows that cannot be reached."""
        learned = self.learned
        if name not in self.search_tools():
            return
        top = _unwrap(observation)
        rows = _rows(top)
        ids = [i for i, _ in rows]
        fresh = [i for i in ids if i not in known]
        paged = self.paged(name, args)
        target = self.search_target(args)
        method = (self.search_kind(name, args), target, self.ledger_version)
        query = self.searches.get(self.search_key(name, args))
        numbers = [v for v in top.values() if isinstance(v, int) and not isinstance(v, bool)] if isinstance(top, dict) else []
        total = max(numbers) if numbers and max(numbers) >= len(rows) else None
        given = {str(v) for v in args.values()}
        onward = isinstance(top, dict) and any(v and str(v) not in given and ((isinstance(v, str) and " " not in v and re.search(r"\d", v)) or v in numbers and v != total) for v in top.values())
        if query:
            query["new"] = query.get("new", 0) + len(fresh)
            if paged and (not query.get("added") if rows else not onward):
                query["stop"] = query["ended"] = True  # the pages end here, whatever its total says (a page with no rows may still hand on)
        if not rows:
            learned["empty"].add(name + _compact(args))
        if name + _compact(args) in self.hop_sent:
            return  # (the code's own search for a listing a requirement points at is not this product's search)
        learned["dry"][method] = 0 if fresh else learned["dry"].get(method, 0) + 1
        if rows and not fresh:
            instead = self.search_instead(name, args, target)
            self.tell(f"dry|{method[:2]}", f"[search] no new rows: this search only repeated {len(ids)} rows already seen." + (f" Instead: {instead}." if instead else " Change the words or the method."))
        if query and not paged and len(rows) >= 3 and (query.get("follow") or (total or 0) > len(rows)):  # (a short list may simply hold the same rows)
            self.learn_arguments(name, args, ids, total)
        if query and total is not None and not paged and rows:
            self.learn_totals(name, query, ids, total)
        if query and rows and query["ended"] and total and len(query["seen"]) * 2 <= total and len(query["offsets"]) >= 2:
            self.tell(f"reach|{name}", f"[search] this search stopped after {len(query['seen'])} rows of {total}: paging reaches no more; narrower words or filters reach the others.")

    def learn_arguments(self, name: str, args: dict[str, Any], ids: list[str], total: int | None) -> None:
        """Two full first pages, alike, whose calls differ in one argument. A shelf or mode that changes, or text given against
        none, changed nothing: that argument is ignored here. Two texts prove less (a rare word can rule both, on any store): only
        twice alike, with two different words after the same first ones, do they show that only those first words are read."""
        props = _branch(self.schemas.get(name)).get("properties") or {}
        numbers = {k for k, spec in props.items() if _branch(spec).get("type") == "integer"} | (self.cursor_args.get(name) or set())
        plain = {k: v for k, v in args.items() if k not in numbers}
        pages = self.learned["pages"].setdefault(name, [])
        for old, seen, count in pages:
            changed = [k for k in set(plain) | set(old) if plain.get(k) != old.get(k)]
            if seen != ids or count != total or len(changed) != 1:
                continue
            key = changed[0]
            new_words, old_words = _words(plain.get(key)), _words(old.get(key))
            shared = len(_shared_start(new_words, old_words))
            if not ({*new_words} & {*old_words}) and new_words and old_words and "enum" not in _branch(props.get(key)):  # wholly different words, the same rows: twice, and this method does not read them
                kinds = self.learned.setdefault("unread", {}).setdefault((self.search_kind(name, args), key), set())
                kinds.add(frozenset((tuple(new_words), tuple(old_words))))
                if len(kinds) >= 2:
                    self.learned.setdefault("blind", set()).add((self.search_kind(name, args), key))
                    self.tell(f"blind|{self.search_kind(name, args)}|{key}", f"[search] {key} changed nothing here: different words gave the same rows, so this method ignores them; use another.")
            if "enum" in _branch(props.get(key)) or not new_words or not old_words:
                self.tell(f"ignored|{self.search_kind(name, args)}|{key}", f"[search] {key} changed nothing here: {old.get(key)!r} and {plain.get(key)!r} gave the same rows, so this search ignores it.")
            elif shared and shared < min(len(new_words), len(old_words)) and f"any|{name}" not in self.learned["told"]:
                sightings = self.learned.setdefault("first", {}).setdefault(name, {}).setdefault(shared, set())
                sightings.add(frozenset((new_words[shared], old_words[shared])))
                if len(sightings) >= 2:
                    self.learned["words"][name] = shared
                    self.tell(f"words|{name}", f"[search] only the first {shared} words of {key} count here: later words changed nothing, so put the deciding words first.")
            break
        pages.append((plain, ids, total))

    def learn_totals(self, name: str, query: dict[str, Any], ids: list[str], total: int) -> None:
        """Totals as far as they can be trusted: different searches with one and the same large total have hit a cap (it says "so
        many or more"); more words that raise the total mean any word matches, so they widen a search."""
        learned = self.learned
        rows = len(ids)
        same = learned["totals"].setdefault(name, {}).setdefault(total, [])
        if tuple(ids) not in same:
            same.append(tuple(ids))
        if len(same) >= 2 and total >= 10 * max(rows, 1) and learned["cap"].get(name) != total:
            learned["cap"][name] = total
            for q in self.searches.values():
                if q["tool"] == name:
                    q["totals"] = [None if t == total else t for t in q["totals"]]
        for other in self.searches.values():
            first = other["totals"][0] if other["totals"] else None
            same_filters = {k: v for k, v in other["args"].items() if v != other["text"]} == {k: v for k, v in query["args"].items() if v != query["text"]}
            if other is not query and other["tool"] == name and first and first != learned["cap"].get(name) and same_filters and len(other["words"]) < len(query["words"]) \
                    and all(any(_same_word(w, x) for x in query["words"]) for w in other["words"]) and first < total != learned["cap"].get(name):
                self.tell(f"any|{name}", f"[search] more words gave more rows ('{other['text']}' {first}, '{query['text']}' {total}): any word matches here, best first, so extra words widen a search; use few, rare words.")
                break

    def guarded(self, work: Callable[[], Any], fallback: Any) -> Any:
        """What the store could teach is an extra: a failure in reading it loses that, never the task."""
        try:
            return work()
        except Exception as exc:
            self.trace.append({"kind": "learn_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return fallback

    def sweep_due(self) -> bool:
        """The last steps before the commit: its window, by the stated limit or the soft wrap-up."""
        left = SOFT_STEPS - self.step if self.cap_unstated and self.max_steps > SOFT_STEPS else self.steps_left()
        return left <= self.commit_steps_needed() + SWEEP_STEPS

    def held_pick(self, index: int) -> tuple[str | None, str]:
        """The navigator's cart line of a product as the review sees it, else the code's pick."""
        line = next((i for i in self.held_lines() if self.item_of(i) == index), None)
        if not line:
            return self.best_candidate(index=index)
        return line, "fits" if line in self.strong_fits(index) else "unclear"

    def value_searches(self, index: int, ceiling: float | None) -> list[dict[str, Any]]:
        """The searches not made yet that may show a better row of a product: the last search that showed it under the pick's
        price (where the store's search takes a price limit), then each required value with the product's own words, the values
        the shopper revealed first (the searches made before the answers cannot have looked for them)."""
        tool = self.roles.get("search")
        if not tool:
            return []
        calls = []
        last = next((q for q in reversed(list(self.searches.values())) if q.get("tool") == tool and any(self.item_of(i) == index for i in q["seen"])), None)
        if last and ceiling is not None and self.price_arg:
            props = (self.schemas.get(tool) or {}).get("properties") or {}
            paging = {k for k, spec in props.items() if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec} | self.cursor_args.get(tool, set())
            capped = self.checked_call(tool, {**{k: v for k, v in (last.get("args") or {}).items() if k not in paging and k != self.price_arg}, self.price_arg: int(ceiling)})
            if capped:
                calls.append(capped)
        named = " ".join(_words(self.product(index)))
        for req in sorted(self.reqs_for(index, blocking=True), key=lambda r: r.get("source") != "revealed"):
            if req["op"] == "is" and req["values"]:
                call = self.search_call(tool, f"{req['values'][0]} {named}".strip())
                if call:
                    calls.append(call)
        return [c for c in calls if self.search_key(c["name"], c["args"]) not in self.searches]

    def store_step(self, review: bool = False) -> bool:
        """Store calls cost no judgement: while the endgame's steps stay whole, read what can still change the choice of a
        product: its unopened rows that state its required values, then those its searches for them showed, then its other rows,
        cheapest first (those under its pick when a cheapest rule or the budget asks for less). Only when no row of it is left
        unread does the next page of its search go out, and never for a product whose pick fits with no rule to weigh it. With
        review the navigator's cart is what is weighed, rows under a line that fits are read with no rule too, the searches not
        made yet follow (value_searches), and what is read stays out of the prompt. True when a call went out."""
        details = self.reader()
        if not details or self.key_out() or self.finished or self.dead or self.time_left() < WRAP_UP_SECONDS / 2:
            return False
        if self.steps_left() - 1 < self.commit_steps_needed(self.th('repair') + 1) + self.pages_due():
            return False  # (the ladder itself leaves the commit room for the lines' pages; these reads must not eat it)
        if self.cap_unstated and self.max_steps > SOFT_STEPS and self.step >= SOFT_STEPS - self.commit_steps_needed():
            return False  # a limit the request does not state may be lower than the policy's (the soft wrap-up)
        picks = {index: self.held_pick(index) if review else self.best_candidate(index=index) for index in range(len(self.items))}
        basket = {index: item for index, (item, _) in picks.items() if item}
        over = len(basket) == len(self.items) and self.basket_total(basket) > self.budget_value()
        priority = self.plan.get("priority")
        for index, (pick, state) in sorted(picks.items(), key=lambda p: (self.store_reads.get(p[0], 0), p[1][1] == "fits", -((self.catalog.get(p[1][0]) or {}).get("price") or 0) * self.qty(p[0]) if over else 0)):
            if not review and priority not in ("cheapest", "best") and not over and state == "fits":
                continue  # (on the code's own path rows under a fit lost more tasks than they won in replay: 2 gained, 6 lost; only a proven swap weighs them)
            price = (self.catalog.get(pick) or {}).get("price")
            room = self.budget_value() - self.basket_total({i: x for i, x in basket.items() if i != index})
            under = price is not None and (over or priority == "cheapest" or (review and state == "fits" and priority != "best"))
            rows = [i for i in self.unopened_rows(index) if (self.catalog[i].get("price") or 0) * self.qty(index) <= room and (not under or (self.catalog[i].get("price") or 0) < price)]
            pool = self.pool(index)
            wanted = [r for r in self.reqs_for(index, blocking=True) if r["op"] == "is" and r["values"]]
            stated = lambda i: sum(1 for r in wanted if any(_says(self.catalog[i].get("text") or "", v) for v in r["values"]))
            rows.sort(key=lambda i: -stated(i))
            fresh = lambda c: bool(c) and c["name"] + _compact(c["args"]) not in self.learned["read"]
            named = [r for r in rows if r in pool or stated(r)]
            call = next((c for r in named + [r for r in rows if r not in named] for c in [self.checked_call(details, _fill(self.schemas.get(details), r, None, {}))] if fresh(c)), None)
            looks: list[dict[str, Any]] = []
            if not call and (state != "fits" or priority in ("cheapest", "best") or over):  # (a pick that fits with nothing to weigh it against needs no more rows)
                pages = [c for q in self.unread_searches(index)[:1] for c in [self.next_page({"name": q["tool"], "args": q["args"]})] if c]
                pages += [c for done in self.ceiling_calls.get(index, []) for c in [self.next_page(done)] if c and len(self.searches.get(self.search_key(done["name"], done["args"]), {}).get("offsets") or []) < 3]
                looks = self.value_searches(index, price if under else None) if review else []
                if not review and priority in ("cheapest", "best") and not pool and index not in self.pool_searched:
                    looks = self.value_searches(index, None)[:1]  # (no search of its values yet: what the rule weighs is unknown)
                call = next((c for c in pages + looks if fresh(c)), None)
            if not call:
                continue
            self.learned["read"].add(call["name"] + _compact(call["args"]))
            if call in looks:
                self.pool_searched.add(index)
            if self.price_arg and self.price_arg in call["args"]:
                self.ceiling_calls.setdefault(index, []).append(call)
            self.store_reads[index] = self.store_reads.get(index, 0) + 1
            sent = [{"id": f"store_{self.step}", "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["args"])}}]
            if not review:
                self.messages.append({"role": "assistant", "content": "Before ordering I read what the store has shown and I have not opened.", "tool_calls": sent})
            self.quiet = review or self.spent_out()
            try:
                replies = self.run_calls(sent, planned=True)
            finally:
                self.quiet = False
            if not review:
                for reply in replies:
                    self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})
            self.trace.append({"kind": "store_step", "call": call, "steps_left": self.steps_left(), "review": review})
            return True
        return False

    def review_due(self) -> bool:
        """The navigator's first close, with steps to spare: every close passes the one review (review_step)."""
        return REVIEW_READS > 0 and not (self.review_held or self.committing or self.code_declining or self.order_attempts or self.several_products() or self.key_out()) and bool(self.roles.get("details")) \
            and self.steps_left() - 2 >= self.commit_steps_needed(self.th('repair') + 1) and self.time_left() > WRAP_UP_SECONDS + CHECK_SECONDS

    def review_hold(self, name: str, args: dict[str, Any]) -> str:
        self.review_held = True
        self.review_call = {"name": name, "args": args}
        self.trace.append({"kind": "review", "steps_left": self.steps_left()})
        return "Not run yet (no step used): the code reads the rows that can still change this choice, then sends this call for you."

    def stronger(self, index: int, line: str) -> str | None:
        """The listing that takes a navigator's cart line's place: what ranks ahead of it by the request's rule (rival), or, for a
        line that does not fit on its own fields, the cheapest listing that does; none it may not be (a model check that answers
        against it), and otherwise the line stays."""
        for _ in range(2):
            strong = self.strong_fits(index)
            found = self.rival(line) or (None if line in strong or not strong else min(strong, key=lambda i: ((self.catalog[i].get("price") or 0), i)))
            if not found or not self.verify_item(found, "ORDER"):
                return found
        return None

    def review_swap(self, call: dict[str, Any]) -> dict[str, Any]:
        """The cart lines a proven listing outranks are changed by the code; the call that closes the cart it leaves."""
        held = {self.item_of(i): i for i in self.held_lines()}
        if None in held or -1 in held:
            return call
        basket = {index: self.stronger(index, line) or line for index, line in held.items()}
        if basket == held:
            return call
        basket, _ = self.basket_rules(basket, {index: (i, "fits") for index, i in basket.items()})
        planned = self.cart_plan(basket)
        finish = self.order_call(basket[min(basket)])
        if self.basket_total(basket) > self.budget_value() or not planned or finish is None or len(planned) + self.th('repair') + 1 > self.steps_left():
            return call
        self.trace.append({"kind": "review_swap", "held": held, "basket": basket})
        self.committing = True
        try:
            for action in planned:
                version = self.cart_version
                sent = [{"id": f"swap_{self.step}", "type": "function", "function": {"name": action["name"], "arguments": json.dumps(action["args"])}}]
                self.messages.append({"role": "assistant", "content": "A listing proven to fit ranks ahead of a cart line, so I swap them.", "tool_calls": sent})
                replies = self.run_calls(sent, planned=True)
                for reply in replies:
                    self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})
                if self.finished or any(reply.held_back for reply in replies) or (action["name"] in self.cart_tools() and self.cart_version == version):
                    break
        finally:
            self.committing = False
        return finish

    def review_step(self) -> None:
        """One step of the review of the navigator's close: the reads first (no model and no prompt for them), then the swaps,
        then its own call again, through every check it had left."""
        if self.review_reads < REVIEW_READS and self.guarded(lambda: self.store_step(review=True), False):
            self.review_reads += 1
            return
        call, self.review_call = self.review_call, None
        if self.finish_kind(call["name"], call["args"]) == "order":
            call = self.guarded(lambda: self.review_swap(call), call)
        if self.finished:
            return
        sent = [{"id": f"review_{self.step}", "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["args"])}}]
        self.messages.append({"role": "assistant", "content": f"I read {self.review_reads} more store results, and I close now.", "tool_calls": sent})
        for reply in self.run_calls(sent):
            self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})

    def sweep_paused(self) -> bool:
        """The commit reads on before settling for less: a step taken, the next commit chooses again."""
        if not self.guarded(self.store_step, False):
            return False
        self.commit_paused = True
        return True

    def hop_search(self, names: list[str], values: list[str], state: dict[str, Any], pages_only: bool = False, numbers: tuple[str, ...] = ()) -> tuple[dict[str, Any], str] | None:
        """(call, search key) of the next search for a described listing: its names (the rarest among the rows seen, the shop's,
        first) with up to HOP_VALUES described numbers and values (a store indexes a listing's facts: on the true hops of races
        5-10 the shop plus the numbers put the page in the first two rows on 170 of 279, the shop alone on 70; a store wanting
        every word then finds nothing on 10 of 198, and the next search costs one step), then its rarest name alone (every
        listing of the shop: another capitalised word may be a shelf's name, which a store's search need not find a listing
        by); each search's next pages before the next search. A search anyone made already counts as sent. pages_only: only
        the next page of a search made for this listing."""
        search = self.roles.get("search")
        if not search:
            return None
        rows = [set(self.row_words(i)) for i in self.catalog]
        ordered = sorted(names, key=lambda n: sum(1 for have in rows if any(_same_word(n, h) for h in have)))
        picked = list(numbers) + [v for v in values if not re.search(r"\d", v)] + [v for v in values if re.search(r"\d", v) and not v.isdigit() and v not in numbers]
        for words in dict.fromkeys([tuple((ordered + picked[:HOP_VALUES])[:HOP_TERMS]), tuple(ordered[:1])]):
            call = self.search_call(search, " ".join(words)) if words else None
            if not call:
                continue
            key = self.search_key(call["name"], call["args"])
            if key not in state["queries"] and key not in self.searches:
                if pages_only:
                    continue
                return call, key
            page = self.next_page(call)
            if page and search + _compact(page["args"]) not in self.hop_sent:
                return page, key
        return None

    def hop_move(self, commit: bool = False) -> tuple[dict[str, Any], str, str | None, list[str]] | None:
        """(call, what it is for, pointer key, listings read) of the code's next call towards a listing a requirement points
        at: the details of a page that gave nothing (thin_pages); the pointer tool on a page found as a hop but not opened with
        it (a page whose notes a compare already showed only before an order, from the steps left then: a store may count only
        that as opening it, on 3 of 32 race-7 chains); else, for the first pending requirement still looked for, the next page
        of its search (up to HOP_PAGES steps), a compare of the best unjudged rows that may be it (the pointer tool on a single
        one), else its next search. None without one, or when the steps are the order's (the commit may spend two of its spare
        steps, as on its other reads before ordering). Past the stop the searches and compares go on: they cost no money, and
        stated_page and page_reading read a page without a model (7 of 9 unresolved race-9 chains ended at the stop)."""
        details = self.reader()
        if not details or self.finished or self.dead or self.time_left() < LAST_CALL_SECONDS * 3 or self.hop_failures >= HOP_FAILURES:
            return None
        if self.steps_left() - 1 < self.commit_steps_needed() - (2 if commit else 0):
            return None
        for ident in sorted(self.thin_pages - self.details_opened - self.facts_read):
            call = self.checked_call(details, _fill(self.schemas.get(details), ident, None, {}))
            if call and call["name"] + _compact(call["args"]) not in self.hop_sent:
                return call, f"{ident} is a listing a requirement points at, but its page told nothing, so I will read its details.", None, [ident]
        # every page of the chain is opened with the tool seen to show listings' notes (where a pointer is told), else the details tool
        for ident in self.pages_unopened():  # (a store call, no judgement: also past the stop)
            if commit or self.open_pages or self.spent_out() or not set(self.note_tools) & set((self.catalog.get(ident) or {}).get("shown") or {}):
                return self.page_call(ident, self.page_opener()), f"{ident} is a listing a requirement points at, so I will open its page.", None, [ident]
        if self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS:
            return None  # none in the wrap-up (past the stop the store's calls are free, and the code tells a page by its details)
        listings = self.listings()
        for req in [r for r in self.reqs if r.get("pending")]:
            key = self.reference_key(req)
            state = self.hop_state.setdefault(key, {"queries": [], "steps": 0})
            names, values = self.pointer_terms(req, listings)
            if (state["steps"] >= self.hop_limit() and not self.spent_out()) or not names:
                continue  # (past the stop no navigator is left to take over, and the steps have no other use)
            numbers = tuple(dict.fromkeys(re.findall(r"\d[\w.]*\w|\d", req["reference"])))
            rows = self.hop_rows(req, names, values, listings)
            # (the numbers narrow a shop's rows, so the first ones are read before its next pages)
            found = self.hop_search(names, values, state, pages_only=True) if state["queries"] and state["steps"] < HOP_PAGES and not (numbers and rows and len(state["queries"]) == 1) else None
            if found:
                return found[0], f"I will read the next page for {req['reference']}.", key, []
            view = self.roles.get("view")
            take = rows[: self.compare_size]
            call = self.view_call(view, take) if view and len(take) > 1 else None
            if call:
                return call, f"I will compare listings that may be {req['reference']}, to find which one it is.", key, take
            opener = self.page_opener()
            if rows and not view and opener != details:
                # a store shows pages and properties through two tools: a pointer's details are properties, so the rows' tables are
                # read first (a batch: each is a step) and only the rows they fit get the page, which is what a store counts as opened
                fresh = [i for i in rows if i not in listings]
                pick = self.stated_page(req, [i for i in rows if i in listings and listings[i][0]], listings)
                if pick:
                    return self.page_call(pick), f"I will open {pick}, whose properties fit {req['reference']}.", key, [pick]
                calls = [c for c in (self.checked_call(details, _fill(self.schemas.get(details), i, None, {})) for i in fresh[:3]) if c]
                room = self.steps_left() - self.commit_steps_needed() + (2 if commit else 0)
                if calls and room >= 1:
                    calls = calls[:room]
                    self.hop_batch = calls[1:]
                    return calls[0], f"I will read the properties of {fresh[0]}, which may be {req['reference']}.", key, fresh[:len(calls)]
                found = self.hop_search(names, values, state, pages_only=True) if state["queries"] else None
                if found:  # (none of the tables read fits: the shop's other rows come before the pages of those)
                    return found[0], f"I will read the next page for {req['reference']}.", key, []
            call = self.page_call(rows[0]) if rows else None
            if call:
                return call, f"I will open {rows[0]}, which may be {req['reference']}.", key, rows[:1]
            found = self.hop_search(names, values, state, numbers=numbers)
            if found:
                return found[0], f"I will search for {req['reference']}.", key, []
        return None

    def safe_hop_move(self, commit: bool = False) -> tuple[dict[str, Any], str, str | None, list[str]] | None:
        try:
            return self.hop_move(commit)
        except Exception as exc:
            self.trace.append({"kind": "hop_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return None

    def hop_reply(self, reply: Reply) -> str:
        """What the navigator reads of a call the code made to follow a pointer: what the code noted, not the rows, which stay in
        its records (on the race-7 replica they were 38% of the navigator's prompt on pointer tasks, about $0.003 a task)."""
        if reply.held_back:
            return reply.text
        _, marked, note = reply.text.partition("\n[note] ")
        return "(the code read this result to follow the pointer)" + ("\n[note] " + note if marked else "")

    def find_hop(self, commit: bool = False) -> bool:
        """Send the code's next call towards a listing a requirement points at (hop_move); True when one was sent."""
        move = self.safe_hop_move(commit)
        open_keys = {self.reference_key(r) for r in self.reqs if r.get("pending")} - self.hops_ended
        if not move and open_keys and not self.finished and not self.dead:
            self.hops_ended |= open_keys  # no call left for these: the pages it read without a note get their judgement now
            self.resolve_references(list(self.pointer_listings()))
            move = self.safe_hop_move(commit)
        if not move:
            if commit and any(r.get("pending") for r in self.reqs):
                self.trace.append({"kind": "pending_at_order", "requirements": [_describe(r) for r in self.reqs if r.get("pending")]})  # (assess shows it as unknown on every listing)
            return False
        call, why, key, items = move
        batch, self.hop_batch = [call] + self.hop_batch, []
        if key:
            state = self.hop_state[key]
            state["steps"] += len(batch)
            if call["name"] == self.roles.get("search"):
                state["queries"].append(self.search_key(call["name"], call["args"]))
        self.hop_sent.update(c["name"] + _compact(c["args"]) for c in batch)
        self.hop_tried.update(items)
        if call["name"] in self.other_tools:
            self.hop_tools_tried.add(call["name"])
        for item in self.thin_pages.intersection(items):  # its details are read now, and it is judged once more
            self.thin_pages.discard(item)
            for seen in self.reference_seen.values():
                seen.discard(item)
        sent = [{"id": f"hop_{self.step}_{n}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for n, c in enumerate(batch)]
        self.messages.append({"role": "assistant", "content": why, "tool_calls": sent})
        records = len(self.records)
        for reply in self.run_calls(sent, planned=True):
            self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": self.hop_reply(reply) + "\n" + self.status()})
        shown: set[str] = set()
        if len(self.records) > records:  # (a call held back used no step and tells nothing)
            self.hop_failures = 0 if _worked(self.records[-1]) else self.hop_failures + 1
            _collect_ids(self.records[-1], shown)
        if len(batch) == 1 and len(items) > 1 and len(self.records) > records and not set(items) & shown:
            # the compare tool took none of them (too many at once): fewer next time. Judged by the ids its answer shows, not by
            # the listings read from it: a listing whose page has no details block (37% of the race-5 catalog) is shown all
            # the same, and a batch of such listings had halved the batch size for the rest of the task
            self.hop_tried.difference_update(items)
            self.compare_size = max(min(2, len(items) - 1), len(items) // 2)  # (3 refused is no proof against 2: a store may take two)
            self.hop_failures = 0  # (too many at once is no failure of the look-up, and the size only falls)
        self.trace.append({"kind": "hop", "call": call, "steps_left": self.steps_left()})
        return True

    def time_left(self) -> float:
        self.budgets["time"]["used"] = time.monotonic() - self.started
        return self.budgets["time"]["total"] - self.budgets["time"]["used"]

    def set_budget(self, name: str, total: Any, source: str) -> None:
        """A budget's total from a stated source (the request, the shopper, the store, a refusal); traced."""
        record = self.budgets[name]
        if record["total"] != total:
            self.trace.append({"kind": "budget", "budget": name, "from": record["total"], "to": total, "source": source})
        record["total"], record["source"] = total, source

    def budget_view(self, name: str) -> dict[str, Any]:
        """One budget record with what is used and reserved now."""
        record = dict(self.budgets[name])
        if name == "messages":
            record["total"] = self.message_total()
            record["reserved"] = self.reserved_messages()
        elif name == "steps":
            record["reserved"] = self.endgame_steps()
        elif name == "money":
            record["used"] = sum(((self.catalog.get(i) or {}).get("price") or 0) * (self.held.get(i) or 0) for i in self.held_lines())
            record["reserved"] = 0
        else:
            self.time_left()
            record = dict(self.budgets[name], reserved=FORCE_SECONDS)
        return record

    def reserved_steps(self) -> int:
        """Steps kept before the finish for instructions still to carry out with a message or a tool."""
        tools = sum(1 for i in self.instructions if i.get("tool") and not i.get("message") and self.owed(i, "order"))
        return self.reserved_messages() + tools

    def endgame_steps(self) -> int:
        """Steps the endgame needs: one fresh basket's cart calls (one for a basket tool, one per product otherwise) and the
        steps kept for instructions; the finish itself is the ladder's first step."""
        carts = 1 if self.basket_tool() or not self.roles.get("cart") else len(self.items)
        return carts + self.reserved_steps()

    def th(self, name: str) -> int:
        """A threshold of the step ladder for this task now, in steps left (or, for "start", a step number). The finish takes
        one step (finish_only) and one more is kept for the last cart fix (last); every other phase sits LADDER[name] steps above
        the steps the endgame needs. A step budget under LADDER_FULL steps compresses the upper phases in proportion, so a short
        task keeps its working steps (never below the steps the endgame needs plus one)."""
        total = self.max_steps or 0
        if name == "start":
            return min(QUESTIONS_START_MAX, max(4, -(-total // 4)))
        if name == "finish_only":
            return FINISH_ONLY_STEPS
        if name == "last":
            return FINISH_ONLY_STEPS + 1
        base = self.endgame_steps()
        offset = LADDER[name] if total >= LADDER_FULL else max(LADDER["cart_only"], round(LADDER[name] * total / LADDER_FULL))
        return offset + base

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def explore_spent(self) -> float:
        """Exploration's own spend: what the setup, the planner and the reserve did not book."""
        return self.spent - sum(self.phase_spend.values())

    def unsearched_products(self) -> list[int]:
        """Products none of whose listings has been opened yet."""
        listings = self.listings()
        return [index for index in range(len(self.items)) if not any(self.item_of(i) == index for i in listings)]

    def floor_owed(self) -> float:
        """Exploration still owed of its floor while the search runs (none past the stop)."""
        return 0.0 if self.spend_stopped else max(0.0, EXPLORE_FLOOR - self.explore_spent())

    def spent_out(self, upcoming: float = 0.0) -> bool:
        """The task's search and checks are over: the run's key is spent; or its whole spend reached the cap less the core
        reserve, or would with the navigator turn about to start (upcoming: its forecast); or, once exploration has had its floor
        and every product has an opened listing, the cap less the core reserve and the questions still planned (protected_spend).
        The navigator is not called any more; the planned questions, the listing checks the finish needs (finish_call) and the
        reserve go on (never with the key spent)."""
        if not self.spend_stopped:
            explore = self.explore_spent()
            searched = explore >= EXPLORE_FLOOR and not self.unsearched_products()
            if self.key_out() or self.spent + upcoming >= self.cap - self.core_reserve() or (searched and self.spent >= self.cap - self.protected_spend()):
                self.spend_stopped, self.stop_spent = True, self.spent
                self.trace.append({"kind": "spend_stop", "spent": round(self.spent, 5), "upcoming": round(upcoming, 5), "exploration": round(explore, 5), "phases": {k: round(v, 5) for k, v in self.phase_spend.items()},
                                   "protected": round(self.protected_spend(), 5), "core": round(self.core_reserve(), 5), "unsearched": len(self.unsearched_products()),
                                   "key": self.key_spent, "step": self.step, "seconds": round(time.monotonic() - self.started, 1)})
        return self.spend_stopped

    def key_out(self) -> bool:
        """The run's key is spent (a call of this task was refused for it, or another task left its mark): no model is called
        and no message goes to the shopper any more, since reading a message bills the same key and the validator stops the
        whole run when that reading fails; the code finishes the task."""
        return self.key_spent

    def mark_key_out(self) -> None:
        """A call was refused because the key is spent: so for this task and, by the mark, for the run's other tasks."""
        self.key_spent = True

    def charge(self, usd: float) -> None:
        """One call's price, on the task's spend and its phase (exploration is what the other phases leave). A call outside the
        setup and the reserve that started with the core reserve inside the cap and ended past it (one that cost more than its
        forecast) keeps the core reserve on top of its spend (reserve_cap); a call started past it moves nothing."""
        before = self.spent
        self.spent += usd
        if self.phase in self.phase_spend:
            self.phase_spend[self.phase] += usd
        if self.phase not in ("setup", "reserve") and before + self.core_reserve() <= self.cap:
            self.reserve_top = max(self.reserve_top, self.spent + self.core_reserve())

    def reserve_cap(self) -> float:
        """The spend the reserve pays up to (before the overrun): the cap, or the spend of a call that crossed it with the core
        reserve then owed added (charge)."""
        return max(self.cap, self.reserve_top)

    def set_cap(self) -> None:
        """The task's budget from its own facts, once the setup has read them: BUDGET_SHARE of the setup's spend, the planned
        questions, the core reserve and SEARCH_STEP_SPEND for each step left to the search, never less than TASK_CAP. A request
        that states no step limit is counted at SOFT_STEPS (where its search wraps up)."""
        steps = min(self.max_steps, SOFT_STEPS) if self.cap_unstated else self.max_steps
        messages = self.message_total() if self.roles.get("message") else 0
        questions = self.message_limit() if self.planner_on and self.roles.get("message") else 0
        search = max(0, steps - messages - self.instructions_owed() - ENDGAME_STEPS)
        need = (self.phase_spend["setup"] + (FIRST_QUESTION_SPEND + NEXT_QUESTION_SPEND * (questions - 1) if questions else 0.0)
                + self.core_reserve() + SEARCH_STEP_SPEND * search)
        self.cap = max(TASK_CAP, BUDGET_SHARE * need)
        self.trace.append({"kind": "task_cap", "cap": round(self.cap, 5), "need": round(need, 5), "steps": steps, "messages": messages, "questions": questions,
                           "search_steps": search, "setup": round(self.phase_spend["setup"], 5), "core": round(self.core_reserve(), 5)})

    def core_reserve(self) -> float:
        """What the cap always keeps from exploration and questions: the finish and its reading, the instructions still to
        carry out that need a message or a tool, and the standing checks the gate's executor still decides."""
        return FINISH_SPEND + INSTRUCTION_SPEND * self.instructions_owed() + CHECK_SPEND * self.checks_owed()

    def protected_spend(self) -> float:
        """What the cap keeps from exploration once it has had its floor: the core reserve and the planner's share for the
        questions still planned."""
        left = max(0, self.message_limit() - self.messages_used) if self.planner_on and self.roles.get("message") else 0
        return self.core_reserve() + min(QUESTION_SPEND, QUESTION_COST * left)

    def reserve_room(self, reading: bool = False) -> bool:
        """The reserve may still pay for a call (an instruction's discharge, a reading of what the finish brought back), up to the
        overrun past the cap; a reading outside a discharge leaves what the instructions still owed need, inside the cap."""
        outside = reading and not self.discharging
        keep = self.core_reserve() - FINISH_SPEND if outside else 0.0
        return not self.key_out() and self.spent < self.reserve_cap() + (0.0 if outside else RESERVE_OVERRUN) - FINISH_SPEND / 2 - keep

    def spend_edge(self) -> float:
        """The spend at which no call of the current phase starts, and which its answer may not cross: the cap; for a call the
        reserve pays for, the reserve's cap and the overrun past it; for a listing check past the stop, that less the whole core
        reserve (its answer is shortened to fit, never into the reserve)."""
        if self.phase != "reserve":
            return self.cap
        return self.reserve_cap() + RESERVE_OVERRUN - (self.core_reserve() if self.finish_check else 0.0)

    def finish_room(self) -> bool:
        """Past the stop a listing check the finish needs may still be paid: beside the whole core reserve, up to the overrun."""
        return not self.key_out() and self.spent < self.reserve_cap() + RESERVE_OVERRUN - self.core_reserve()

    def finish_call(self, ask: Callable[[], Any]) -> Any:
        """A listing check (which product a listing is, its stated values, its variant fields, the model check): before the stop
        an exploration call like any other; past it the finish needs its answer (an opened fit left unmapped or unchecked is
        not ordered), so the reserve pays for it while finish_room allows, and the stop does not refuse it. Every such check
        has a short answer, and is priced with SHORT_ANSWER_TOKENS of it."""
        self.call_skipped = False
        late, self.answer_tokens = self.late_call, SHORT_ANSWER_TOKENS
        try:
            if not self.spent_out():
                return ask()
            if not self.finish_room():
                return None
            self.late_call = self.finish_check = True
            return self.in_phase("reserve", ask)
        finally:
            self.late_call, self.finish_check, self.answer_tokens = late, False, EXPLORE_TOKENS

    def forecast(self, model: str, prompt_tokens: int, navigator: bool, answer: int) -> float:
        """The price an exploration call is expected to reach, before it starts: for a navigator turn its last turn's prompt read
        from the cache and the rest written to it (an anthropic prompt is cached up to its last message), otherwise the whole
        prompt at the prompt rate; and the answer allowance of its kind of call. On the recorded race-7 calls this is 1.0-3.1
        times the real price (median by kind of call; the navigator 1.34), where the whole prompt at the prompt rate and
        EXPLORE_TOKENS of answer for every call was 2.9-13.8 times."""
        cached = min(prompt_tokens, self.nav_prompt.get(model, 0)) if navigator else 0
        written = prompt_tokens - cached if navigator and model.startswith("anthropic/") else 0
        return _price(model, {"prompt_tokens": prompt_tokens, "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written}, "completion_tokens": answer})

    def in_phase(self, name: str, work: Callable[[], Any]) -> Any:
        before, self.phase = self.phase, name
        try:
            return work()
        finally:
            self.phase = before

    def infer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int, temperature: float = 0) -> dict[str, Any] | None:
        # a model the service refused twice (not served or not allowed) and that never answered is skipped; an empty answer or a
        # timeout says nothing about the model, so it is not counted
        models = [m for m in models if m in self.model_ok or self.model_failures.get(m, 0) < 2] or models
        self.call_skipped = False
        for model in models:
            left = self.time_left()
            if left < FORCE_SECONDS - 10 or self.key_out() or self.spent >= self.spend_edge() or (self.spent_out() and not self.late_call):
                return None
            _llm.timeout = int(max(8, min(75, MODEL_SECONDS.get(model, 75), left - FORCE_SECONDS + 15)))
            sent = messages
            if model.startswith("anthropic/") and len(messages) > 2:
                sent = [dict(m) for m in messages]
                for index in (0, len(sent) - 1):
                    if isinstance(sent[index].get("content"), str) and sent[index]["content"]:
                        sent[index]["content"] = [{"type": "text", "text": sent[index]["content"], "cache_control": {"type": "ephemeral"}}]
            body: dict[str, Any] = {"model": model, "messages": sent, "temperature": temperature, "max_tokens": min(max_tokens, MODEL_TOKENS.get(model, max_tokens)), **MODEL_SETTINGS.get(model, {}), **self.call_settings.get(model, {})}
            prompt_tokens = int((len(json.dumps(sent)) + (len(json.dumps(tools)) if tools else 0)) / PROMPT_CHARS)
            navigator = messages is self.messages
            if not self.late_call and self.phase not in self.phase_spend:
                expected = self.forecast(model, prompt_tokens, navigator, self.answer_tokens)
                if navigator and self.spent_out(expected):
                    return None  # the navigator's next turn could cross the cap less the core reserve: the search stops before it
                if not navigator and self.spent + expected >= self.cap - self.core_reserve():
                    self.call_skipped = True  # this one call could cross it: it is skipped, and the search goes on
                    self.trace.append({"kind": "call_skipped", "model": model, "expected": round(expected, 5), "spent": round(self.spent, 5)})
                    return None
            room = self.spend_edge() - self.spent - self.forecast(model, prompt_tokens, navigator, 0)
            per_out = _price(model, {"prompt_tokens": 0, "completion_tokens": 1000}) / 1000
            if per_out > 0 and room < per_out * body["max_tokens"]:  # the answer could cross the task's cap: shorten it to fit, or skip
                fit = int(room / per_out)
                if fit < 300:
                    self.trace.append({"kind": "cap_skip", "model": model, "spent": round(self.spent, 5)})
                    return None
                body["max_tokens"] = fit
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            asked = time.monotonic()
            result = _llm.post_verbose("/inference/chat/completions", body)
            data = result.data
            if isinstance(data, dict):
                self.charge(_price(model, data.get("usage")))  # an answer cut off without content is billed too
                if navigator and isinstance((data.get("usage") or {}).get("prompt_tokens"), int):
                    self.nav_prompt[model] = data["usage"]["prompt_tokens"]
            elif time.monotonic() - asked >= _llm.timeout - 1:  # stopped by the time limit: the provider may bill a whole answer
                self.charge(_price(model, {"prompt_tokens": len(json.dumps(sent)) // 3, "completion_tokens": body["max_tokens"]}))
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                if isinstance(usage.get("cost"), (int, float)):
                    self.cost += float(usage["cost"])
                message = data["choices"][0].get("message") or {}
                if message.get("content") or message.get("tool_calls"):
                    self.model_ok.add(model)
                    return message
            self.trace.append({"kind": "inference_failure", "model": model, "error": str(result.error)[:300] if result.error else "empty"})
            if _key_spent(result.error):
                self.mark_key_out()  # every model bills the same key: no other model is tried
                return None
            status = result.error.get("status") if isinstance(result.error, dict) else None
            if status in (400, 401, 403, 404):
                self.model_failures[model] = self.model_failures.get(model, 0) + 1
        return None

    def env_call(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        payload: dict[str, Any] = dict(self.binding)
        payload["step"] = self.step + 1
        if len(actions) == 1:
            payload["action"] = actions[0]
        else:
            payload["actions"] = actions
        for _ in range(3):
            left = self.time_left()
            if left < LAST_CALL_SECONDS:
                return None
            _env.timeout = int(max(3, min(95, left - 1)))
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                if isinstance(data.get("step"), int):
                    self.step = data["step"]
                else:
                    self.step += len(actions)
                if data.get("finished"):
                    self.finished = True
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "env_failure", "status": status, "body": body[:300]})
            if status == 402:
                self.dead = True
                return None
            if status == 409:
                if "finished" in body:
                    self.finished = True
                    return None
                match = re.search(r"expected (\d+)", body)
                if match:
                    self.step = int(match.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                return None
            if status in (403, 422, 500, 503, 504):
                return None
        return None

    def setup(self) -> None:
        tool_lines = []
        for tool in self.tools:
            function = tool["function"]
            tool_lines.append(f"- {function['name']}: {function.get('description', '')} | arguments: {_compact(function.get('parameters') or {})}")
        static = SETUP_PROMPT + "\n\nTOOLS:\n" + "\n".join(tool_lines)
        variable = "REQUEST:\n" + str(self.policy.get("query") or "")
        if self.policy.get("opening") is not None:
            variable += "\n\nOPENING OBSERVATION:\n" + _compact(self.policy.get("opening"))
        content = [{"type": "text", "text": static, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": variable}]
        query = str(self.policy.get("query") or "")
        plan: dict[str, Any] = {}
        for models in (_route("setup"), _route("setup_retry")):
            self.call_settings = SETUP_SETTINGS
            try:
                message = self.infer([{"role": "user", "content": content}], None, models, SETUP_TOKENS)
            finally:
                self.call_settings = {}
            plan = _parse_json_object((message or {}).get("content") or "") or {}
            if isinstance(plan.get("roles"), dict) and plan["roles"]:
                break
            self.trace.append({"kind": "setup_retry", "keys": sorted(plan)[:20]})  # without roles nothing guards the run: read it once more
            if self.time_left() < TASK_SECONDS - 60:
                break
        self.plan = plan
        roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        others = plan.get("other_tools") if isinstance(plan.get("other_tools"), list) else []
        self.other_tools = {o["tool"]: " ".join(str(o.get("does") or "").split())[:120] for o in others
                            if isinstance(o, dict) and o.get("tool") in self.schemas and o.get("tool") not in self.roles.values()}
        self.order_args, self.decline_args = self.finish_modes(plan)
        searcher = (_branch(self.schemas.get(self.roles.get("search"))).get("properties") or {}) if self.roles.get("search") else {}
        self.price_arg = plan.get("price_arg") if plan.get("price_arg") in searcher else None
        measure, direction = " ".join(_words(plan.get("priority_property"))), str(plan.get("priority_direction") or "").strip().lower()
        self.best = {"property": measure, "direction": direction} if plan.get("priority") == "best" and measure and direction in ("least", "most") else None
        cart_args = plan.get("cart_args") if isinstance(plan.get("cart_args"), dict) else {}
        self.cart_modes = {k: self.mode_values(self.roles.get("cart"), cart_args.get(k)) for k in ("add", "remove")}
        budgets = plan.get("budgets") if isinstance(plan.get("budgets"), dict) else {}
        budget = budgets.get("messages")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget >= 0:
            self.set_budget("messages", int(budget), "request")
        items = []
        for raw in plan.get("items") if isinstance(plan.get("items"), list) else []:
            if not isinstance(raw, dict) or not _words(raw.get("product")):
                continue
            quantity = _quantity_from(raw.get("quantity"), raw.get("quantity_quote"), raw.get("quantity_word"), query) or 1
            if raw.get("quantity") not in (None, 1, quantity):
                self.trace.append({"kind": "quantity", "proposed": raw.get("quantity"), "quote": raw.get("quantity_quote"), "kept": quantity})
            items.append({"product": " ".join(str(raw.get("product")).split())[:80], "quantity": quantity, "category": " ".join(str(raw.get("category") or "").split())})
        if not items:  # an older reply shape, or a failed setup
            quantity = _quantity_from(plan.get("quantity"), plan.get("quantity_quote"), plan.get("quantity_word"), query) or 1
            items = [{"product": " ".join(str(plan.get("product") or "").split())[:80], "quantity": quantity, "category": ""}]
        self.items = items
        self.reqs = self.without_pointer_details([r for r in (_requirement(raw, "stated") for raw in plan.get("requirements") or []) if r])
        for req in [r for r in self.reqs if r.get("when")]:
            self.reqs.remove(req)
            req["item"] = req["item"] if self.multi() and req["item"] is not None and req["item"] < len(self.items) else None
            req["when"]["item"] = req["when"].get("item") if isinstance(req["when"].get("item"), int) and 0 <= req["when"].get("item") < len(self.items) else None
            if self.multi() and req["when"]["item"] is None:
                req["blocking"] = False  # the listing that decides it cannot be named: nothing can be proven by it
                self.reqs.append({k: v for k, v in req.items() if k != "when"})
            elif not _grounded(req, query):
                self.conditional.append(req)
        ungrounded = [(r, _grounded(r, query)) for r in self.reqs if _grounded(r, query)]
        if ungrounded:  # a reading the request's words do not support is dropped (the shopper's answers can still state it)
            self.reqs = [r for r in self.reqs if not any(r is u for u, _ in ungrounded)]
            self.trace.append({"kind": "ungrounded", "dropped": [{"requirement": _describe(r), "why": why} for r, why in ungrounded]})
        self.relations = [r for r in (_relation(raw, query, len(self.items)) for raw in plan.get("relations") or [] if isinstance(plan.get("relations"), list)) if r]
        if self.relations:  # a relation the reader also wrote as a product requirement: one listing alone can never show it
            said = [set(_words(r["quote"])) for r in self.relations]
            self.reqs = [r for r in self.reqs if not (r["op"] == "text" and any(set(w for v in r["values"] for w in _words(v)) <= q for q in said))]
            self.trace.append({"kind": "relations", "relations": self.relations})
        asked_words = _words(query)
        for req in self.reqs:
            if req.get("item") is not None and (req["item"] >= len(self.items) or len(self.items) == 1):
                req["item"] = None
            req["named"] = any(_same_word(w, h) for w in _words(req["property"]) for h in asked_words)  # False: the property name is a guess
        self.learn_shelves()
        if plan and "instructions" not in plan:
            self.legacy_reading(plan, budgets)  # an answer in the older shape (a recorded episode): its keys become records
        self.add_instructions(plan.get("instructions"), "request", query)  # (after the roles: a tool an instruction names is checked against them)
        self.fallback = self.fallback_rule(plan)
        if self.fallback["closest"] == "wanted":  # naming the closest belongs to the close when nothing fits: it takes no question from the order path
            for ins in self.instructions:
                if ins.get("names") == "required" and ins.get("when") in ("before_order", "before_finish", "after_change", "any"):
                    ins["when"] = "before_decline"
        cap = _as_written(plan.get("price_cap"), query)
        priced = {r.get("item") for r in self.reqs if r.get("field")}
        if cap is not None:
            self.set_budget("money", cap, "request")
        elif None in priced or priced >= set(range(len(self.items))):
            pass  # every product has its own price limit: no total is missing
        elif plan and self.roles.get("message"):  # no price limit stated: learning it is the first thing to ask (a record like any other)
            self.instructions.insert(0, dict(_instruction({"quote": "(no price limit is stated)", "when": "at_start", "message": True, "wait_reply": True,
                                                           "content": "the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"}, "", "code"), id=0))
        self.budgets["money"]["currency"] = plan.get("currency")
        steps = budgets.get("steps")
        if isinstance(steps, (int, float)) and not isinstance(steps, bool) and steps > 0:  # the request states its cap
            self.set_budget("steps", min(self.max_steps, int(steps)), "request")
        else:
            pass  # the step record keeps the policy's limit as its source: the soft wrap-up (SOFT_STEPS) applies

    def fallback_rule(self, plan: dict[str, Any]) -> dict[str, Any]:
        """The request's own rule for when nothing fits: the setup's reading, or, when it gave none, what a decline instruction
        says about naming listings ("required": the closest is wanted; "forbidden": no substitute)."""
        raw = plan.get("if_nothing_fits") if isinstance(plan.get("if_nothing_fits"), dict) else {}
        closest = raw.get("closest") if raw.get("closest") in ("wanted", "forbidden") else None
        if closest is None:
            rules = [i.get("names") for i in self.instructions if i.get("when") in ("before_decline", "before_finish")]
            closest = "forbidden" if "forbidden" in rules else ("wanted" if "required" in rules else None)
        margin = raw.get("margin")
        margin = margin / 100 if isinstance(margin, (int, float)) and not isinstance(margin, bool) and 1 <= margin <= 100 else margin
        margin = margin if isinstance(margin, (int, float)) and not isinstance(margin, bool) and 0 < margin < 1 else None
        counts = [c for c in raw.get("counts") or [] if c in ("stock", "budget")] if isinstance(raw.get("counts"), list) else []
        return {"closest": closest, "margin": margin, "counts": counts}

    def legacy_reading(self, plan: dict[str, Any], budgets: dict[str, Any]) -> None:
        """A reader's answer in the shape before the instruction ledger (the setup answers of recorded episodes): its message and
        step numbers become the budget records, each fact to learn first an instruction to ask at the start, and a decline rule
        against naming listings an instruction for the decline. Its quoted rules carried no behaviour then and are not read."""
        for key, name in (("message_budget", "messages"), ("step_budget", "steps")):
            if name not in budgets and isinstance(plan.get(key), (int, float)) and not isinstance(plan.get(key), bool):
                budgets[name] = plan[key]
                if name == "messages" and plan[key] >= 0:
                    self.set_budget("messages", int(plan[key]), "request")
        items = [{"quote": str(a), "when": "at_start", "message": True, "content": str(a)} for a in plan.get("ask_first") or [] if str(a).strip()]
        if plan.get("decline_no_names") in (True, False):
            items.append({"quote": "(the request's rule for declining)", "when": "before_decline", "message": False, "names": "forbidden" if plan["decline_no_names"] else "required"})
        plan["instructions"] = items

    def learn_shelves(self) -> None:
        """The store's shelf labels: the allowed values of the tool argument that lists every product's category."""
        shelves = {" ".join(_words(item["category"])) for item in self.items if item["category"]}
        for spec in self.schemas.values():
            for prop in ((spec or {}).get("properties") or {}).values():
                options = {" ".join(_words(v)) for v in prop.get("enum") or [] if isinstance(v, str)} if isinstance(prop, dict) else set()
                if shelves and shelves <= options:
                    self.shelf_values = options  # the argument that lists the store's shelves

    def reread_products(self, listing: str) -> None:
        """A one-product order, but the navigator is putting another kind of listing in the cart: read the request's products
        once more (setup may have read two products as one) and, when it asks for several, make the order a multi-product
        one, with each requirement on its product."""
        self.reread_done = True
        reqs = [r for r in self.reqs if r.get("blocking") or r.get("pending")]
        labels = {f"R{n + 1}": r for n, r in enumerate(reqs)}
        known = self.listings()
        about = lambda i: f"{i}: " + " | ".join(sorted((self.catalog.get(i) or {}).get("labels") or set())[:6]) + " | " + known.get(i, ({}, ""))[1][:160]
        prompt = REREAD_PROMPT + "\n\nREQUEST:\n" + str(self.policy.get("query") or "") + "\n\nPRODUCTS READ SO FAR:\n" + self.products_text()
        prompt += "\n\nLISTINGS IN OR GOING INTO THE CART:\n" + "\n".join(about(i) for i in self.held_lines() + [listing])
        prompt += "\n\nREQUIREMENTS:\n" + ("\n".join(f"{label}: {_describe(r)}" for label, r in labels.items()) or "(none)")
        shelves = sorted(self.shelf_values)
        if shelves:
            prompt += "\n\nTHE STORE'S SHELVES:\n" + ", ".join(shelves)
        result = self.judge(prompt, JUDGE_TOKENS, _route("setup")) or {}
        query = str(self.policy.get("query") or "")
        items = []
        for raw in result.get("items") if isinstance(result.get("items"), list) else []:
            if isinstance(raw, dict) and _words(raw.get("product")):
                quantity = _quantity_from(raw.get("quantity"), raw.get("quantity_quote"), raw.get("quantity_word"), query) or 1
                category = " ".join(str(raw.get("category") or "").split())
                items.append({"product": " ".join(str(raw.get("product")).split())[:80], "quantity": quantity, "category": category if not shelves or " ".join(_words(category)) in self.shelf_values else ""})
        self.trace.append({"kind": "reread", "listing": listing, "items": items})
        if len(items) < 2:
            return
        assign = result.get("assign") if isinstance(result.get("assign"), dict) else {}
        for label, req in labels.items():
            index = assign.get(label)
            req["item"] = index if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(items) else None
        self.items = items
        self.item_map = {}
        self.learn_shelves()
        self.ledger_version += 1
        self.search_note = (self.search_note + " " if self.search_note else "") + "[order] The request asks for several products: " + "; ".join(f"product {n}: {i['product']} ({i['quantity']} unit(s))" for n, i in enumerate(items)) + ". Each product needs its own line in the cart."

    def second_kind(self, listing: str, going: list[str]) -> bool:
        """In a one-product order, the listing going into the cart may be a second product: it is on another shelf than the
        order's product, or it joins a different listing (held, or in the same call) that no shelf label shows to be the
        same kind (a swap leaves its shelf the same)."""
        if self.multi() or self.reread_done or self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS or not self.roles.get("cart"):
            return False
        if self.item_of(listing) == -1:
            return True
        labels = lambda i: {" ".join(_words(label)) for label in (self.catalog.get(i) or {}).get("labels", set())}
        others = [i for i in self.held_lines() + list(going) if i != listing]
        if self.shelf_values:
            shelf = lambda i: labels(i) & self.shelf_values
            return any(not (shelf(i) and shelf(i) == shelf(listing)) for i in others)
        return any(not (labels(i) & labels(listing)) for i in others)  # no shelf list: a shared row label (category, brand) is the same kind

    def mode_values(self, name: str | None, proposed: Any) -> dict[str, Any]:
        properties = (self.schemas.get(name or "") or {}).get("properties") or {}
        out: dict[str, Any] = {}
        for key, value in (proposed if isinstance(proposed, dict) else {}).items():
            spec = properties.get(key)
            if not isinstance(spec, dict):
                continue
            if "enum" in spec:
                match = next((option for option in spec["enum"] if str(option).lower() == str(value).strip().lower()), None)
                if match is not None:
                    out[key] = match
            elif spec.get("type") == "boolean" and isinstance(value, bool):
                out[key] = value
        return out

    def ask_modes(self, name: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        tool = next((t["function"] for t in self.tools if t["function"]["name"] == name), None)
        if tool is None:
            return None
        result = self.judge(MODES_PROMPT + "\n\nTOOL:\n" + _compact(tool), JUDGE_TOKENS, _route("read")) or {}
        order_args, decline_args = self.mode_values(name, result.get("order_args")), self.mode_values(name, result.get("decline_args"))
        if any(key in decline_args and decline_args[key] != value for key, value in order_args.items()):
            return order_args, decline_args
        return None

    def finish_modes(self, plan: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        order, decline = self.roles.get("order"), self.roles.get("decline")
        order_args = self.mode_values(order, plan.get("order_args"))
        decline_args = self.mode_values(decline, plan.get("decline_args"))
        if not order or order != decline:
            return order_args, decline_args
        if any(key in decline_args and decline_args[key] != value for key, value in order_args.items()):
            return order_args, decline_args
        asked = self.ask_modes(order)
        self.trace.append({"kind": "finish_modes", "source": "asked" if asked else "unknown", "modes": asked})
        return asked if asked else (None, None)

    def finish_kind(self, name: str | None, args: dict[str, Any]) -> str | None:
        order, decline = self.roles.get("order"), self.roles.get("decline")
        if not name or name not in (order, decline):
            return None
        if order != decline:
            return "order" if name == order else "decline"
        for kind, mode in (("order", self.order_args), ("decline", self.decline_args)):
            if mode and all(args.get(key) == value for key, value in mode.items()):
                return kind
        return "unknown"

    def how(self, kind: str) -> str:
        name = self.roles.get(kind)
        mode = self.order_args if kind == "order" else self.decline_args
        return f"{name} with {_compact(mode)}" if name and mode else (name or f"the {kind} tool")

    def brief(self) -> str:
        plan = self.plan
        if not plan:
            return ""
        lines = ["A first reading of the request (check it against the request itself):"]
        if self.multi():
            lines.append(f"- the order has {len(self.items)} different products; buy each as ONE listing at its own quantity, one cart line per product, all in one order:")
            for index, item in enumerate(self.items):
                shelf = f" (shelf: {item['category']})" if item["category"] else ""
                lines.append(f"  - product {index}: {item['product']}{shelf}, exactly {item['quantity']} unit(s) of one listing")
        else:
            if self.items[0]["product"]:
                lines.append(f"- product: {self.items[0]['product']}" + (f" (shelf: {self.items[0]['category']})" if self.items[0]["category"] else ""))
            lines.append(f"- quantity: exactly {self.quantity} unit(s) of one listing")
        for req in self.reqs + self.conditional:
            lines.append(f"- stated requirement{self.about(req)}: {_describe(req)}" + self.req_flag(req))
        if plan.get("price_cap") is not None:
            lines.append(f"- price cap (total): {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip())
        if plan.get("in_stock_only"):
            lines.append("- only items in stock / on hand")
        if plan.get("priority") in ("cheapest", "best"):
            detail = f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else ""
            scope = " for the whole order's total" if plan.get("priority_scope") == "order" and self.multi() else ""
            lines.append(f"- choose the {plan.get('priority')}{detail} among all fitting items{scope}")
        else:
            lines.append("- no cheapest/best rule was found: any item that meets every requirement within the budget is right, so do not spend steps hunting for the lowest price (unless the request itself says otherwise)")
        for want in plan.get("soft_wants") or []:
            lines.append(f"- nice to have, NOT required (only a tie-breaker): {want}")
        for ins in self.instructions:
            lines.append("- instruction: " + self.instruction_line(ins))
        for tool, does in self.other_tools.items():
            lines.append(f"- another tool: {tool} ({does})")
        if self.other_tools:
            lines.append("- a store may show a listing's page and its table of properties through two tools: requirements are judged on the table, so read it for every listing you may order")
        for r in self.relations:
            lines.append(f"- relation: product {r['dependent']} ({self.product(r['dependent'])}) must work with whichever product {r['hub']} ({self.product(r['hub'])}) listing is chosen: choose that one first, then a listing whose own page names its model (code or model name): {r['quote'][:200]}")
        if self.message_budget is not None:
            lines.append(f"- messages to the shopper allowed: {self.message_budget}")
        lines.append(f"- steps allowed: {self.max_steps}")
        if self.roles.get("order") and self.roles.get("order") == self.roles.get("decline") and self.order_args and self.decline_args:
            lines.append(f"- finishing: {self.how('order')} places the order; {self.how('decline')} declines. Always set it explicitly.")
        return "\n".join(lines)

    def status(self) -> str:
        left = self.steps_left()
        parts = [f"[status] steps used {self.step} of {self.max_steps}, {left} left" + (f" (no step limit was stated: the store may stop earlier, so aim to order by step {SOFT_STEPS})" if self.cap_unstated and self.max_steps > SOFT_STEPS else "")]
        kept = self.reserved_messages()
        if self.message_budget is not None:
            parts.append(f"messages sent {self.messages_used} of {self.message_budget}" + (f" ({kept} kept for the instructions)" if kept else ""))
        else:
            parts.append(f"messages sent {self.messages_used}" + (f" ({kept} kept for the instructions)" if kept else ""))
        parts.append(f"about {int(max(0, self.time_left() - FORCE_SECONDS))} s left")
        text = "; ".join(parts) + "."
        steps_view = self.budget_view("steps")
        if steps_view["reserved"] > 1:
            text += f" The endgame keeps {steps_view['reserved']} steps (cart calls and the instructions' messages) before the finish."
        text += "\n[requirements now] " + self.ledger()
        text += self.relation_hint()
        if self.instructions:
            text += "\n[instructions] " + "; ".join(f"{n + 1}) {self.instruction_line(ins, short=True)}" for n, ins in enumerate(self.instructions[-8:]))
        lines = "one verified line per product" if self.multi() else "one verified line"
        if left <= self.th('last'):
            text += " LAST STEPS: order now with the best candidate you have in the cart (a decline almost never scores)."
        elif left <= self.th('finish_now') or self.time_left() < WRAP_UP_SECONDS:
            text += f" FINISH NOW: fix the cart to {lines}, re-check stock, then order your best candidates."
        elif self.order_attempts == 0 and (ready := self.safe_ready()):
            if self.multi():
                text += f" READY: {ready} are your choices; put them in the cart (one line per product, each at its quantity) and order now. Every pending change shows up at the first order attempt, and a refused order can still be fixed."
            else:
                text += f" READY: {ready} is your choice; put it in the cart (if it is not there yet) and order it now. Every pending change shows up at the first order attempt, and a refused order can still be fixed."
        elif self.order_attempts == 0 and left <= self.th('low'):
            text += f" Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when {self.th('reading_ends')} steps remain, and a refused order needs about {self.th('repair')} steps to repair."
        return text

    def about(self, req: dict[str, Any]) -> str:
        if not self.multi():
            return ""
        return f" for product {req['item']} ({self.product(req['item'])})" if req.get("item") is not None else " for every product"

    def looked_up(self, req: dict[str, Any]) -> bool:
        """The code looks for the listing a pending requirement points at itself: a details tool, a name in its description
        (the shop's), and steps left in its search for it."""
        names = _pointer_words(req["reference"], self.policy.get("query"), set(), set(), {w for shelf in self.shelf_values for w in _words(shelf)})[0]
        return bool(self.roles.get("details") and names) and (self.hop_state.get(self.reference_key(req)) or {}).get("steps", 0) < self.hop_limit()

    def req_flag(self, req: dict[str, Any]) -> str:
        gives = "the value its page pairs with its own" if req["op"] == "paired" else "its value"
        if req.get("pending") and self.looked_up(req):
            return f" (another listing gives this value: that listing is found and opened for you, and {gives} becomes the requirement)"
        if req.get("pending"):
            return f" (another listing gives this value: find that listing, open its details, and {gives} becomes the requirement)"
        return "" if req["blocking"] else " (no listing can show this: not blocking)"

    def req_text(self, req: dict[str, Any]) -> str:
        tag = "stated" if req["source"] == "stated" else ("announced by the store" if req.get("origin") == "store" else "revealed by the shopper")
        if req.get("pending"):  # the listing it points at now is the one its description names; the chain so far is listed
            chain = f", after {' -> '.join(req['found'])}" if req.get("found") else ""
            return f"{_describe(req)} ({tag}; pending{chain}: " + ("the code looks up that listing" if self.looked_up(req) else "find the listing it points at") + ")"
        return f"{_describe(req)} ({tag}" + ("" if req["blocking"] else "; no listing can show this: not blocking") + ")"

    def ledger(self) -> str:
        if self.multi():
            parts = []
            for index, item in enumerate(self.items):
                own = [self.req_text(r) for r in self.reqs + self.conditional if r.get("item") == index]
                parts.append(f"product {index} ({item['product']}, exactly {item['quantity']} unit(s) of one listing): " + ("; ".join(own) or "no requirement yet"))
            common = [self.req_text(r) for r in self.reqs + self.conditional if r.get("item") is None]
            text = " | ".join(parts) + (" | every product: " + "; ".join(common) if common else "")
        else:
            text = "; ".join(self.req_text(r) for r in self.reqs + self.conditional) or "see the request"
        if self.budget_now is not None:
            text += f"; total price of the whole order at most {self.budget_now} {self.plan.get('currency') or ''}".rstrip()
        if not self.multi():
            text += f"; quantity exactly {self.quantity}"
        if self.dropped:
            text += "; no longer required: " + "; ".join(self.dropped)
        if self.free:
            text += "; shopper does not care about: " + ", ".join(self.free)
        return text

    def judge(self, prompt: str, max_tokens: int = JUDGE_TOKENS, models: list[str] | None = None, temperature: float = 0) -> dict[str, Any] | None:
        models = list(models or _route("judge"))
        if BACKUP_MODEL not in models:
            models.append(BACKUP_MODEL)  # another provider, for when the listed models are not served
        message = self.infer([{"role": "user", "content": prompt}], None, models, max_tokens, temperature)
        return _parse_json_object((message or {}).get("content") or "")

    def interpret(self, lines: list[str], question: str | None) -> str:
        fresh = [line for line in lines if line not in self.heard]
        if not fresh or self.time_left() < FORCE_SECONDS + 5:
            return ""  # too late to read: the lines stay unread, so they are read if they come again
        late = self.spent_out()  # past the stop every delivered text is still read, from the reserve, while the task's cap allows
        if late and not self.reserve_room(reading=True):
            return ""
        self.heard.update(fresh)
        prompt = INTERPRET_PROMPT + "\n\nTHE SHOPPER'S REQUEST (for context):\n" + str(self.policy.get("query") or "")
        prompt += "\n\nPRODUCTS IN THE ORDER:\n" + self.products_text()
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        if self.said:
            prompt += "\n\nEARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(self.said[-12:])
        if question:
            prompt += "\n\nTHE ASSISTANT ASKED:\n" + question
        prompt += "\n\nTHE SHOPPER NOW SAID:\n" + "\n".join(fresh)
        # past the stop a reading is made with the fast model only (a change often comes with a refused order); a reading is not
        # an exploration call, so the price check before one (infer) never skips it
        self.late_call = True
        try:
            result = self.judge(prompt, JUDGE_TOKENS, _route("read_late") if late or self.time_left() < WRAP_UP_SECONDS else _route("read"))
        finally:
            self.late_call = False
        if not result:
            self.heard.difference_update(fresh)
            return ""
        self.said.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        notes = []
        speaker = result.get("speaker") if result.get("speaker") in ("shopper", "store") else ("shopper" if question is not None else "store")
        if result.get("one_at_a_time") is True:
            self.one_at_a_time_seen = True
        ack = str(result.get("ack") or "none").strip().lower()
        for line in fresh:
            self.ack_lines[line] = ack if ack in ("accepted", "refused") else "none"
        self.last_ack = ack if ack in ("accepted", "refused") else "none"  # what the shopper said to our last message, read now
        added = self.add_instructions(result.get("instructions"), speaker, "\n".join(fresh))
        if added:
            notes.append("new instruction: " + "; ".join(added))
        if isinstance(result.get("allowance"), dict):
            written = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", "\n".join(fresh))}  # (a reply about a listing is no new limit: a number no line writes is the reader's own)
            changed = self.apply_allowance({k: v for k, v in result["allowance"].items() if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) in written}, speaker)
            if changed:
                notes.append(changed)
        for drop in result.get("dropped") or []:
            removed = self.drop_requirement(drop, keep_stated=question is not None)
            if removed:
                self.dropped.append(removed)
                notes.append(f"dropped: {removed}")
        pointed = False
        needs = [dict(raw, property=self.asked_property) if question is not None and isinstance(raw, dict) and _narrower(raw.get("property"), self.asked_property)
                 and _has_phrase(" ".join(fresh), self.asked_property) else raw for raw in result.get("needs") or []]  # the shopper's own name for it
        for req in self.without_pointer_details([r for r in (_requirement(raw, "revealed") for raw in needs) if r]):
            if req:
                if not self.multi() or (req.get("item") is not None and req["item"] >= len(self.items)):
                    req["item"] = None
                elif req.get("item") is None and question is not None and self.asked_item is not None:
                    req["item"] = self.asked_item  # the answer is about the product the question named
                if question is not None and req.get("item") is not None:
                    self.needs_found[req["item"]] = self.needs_found.get(req["item"], 0) + 1
                if req.get("pending") and any(self.same_pointer(r, req) for r in self.reqs):
                    continue  # the shopper named the same listing again: the requirement keeps the chain followed so far
                pointed = pointed or bool(req.get("pending"))
                req["origin"] = speaker
                if _grounded(req, "\n".join(fresh)) and req.get("field"):
                    continue  # a price limit whose number these lines never state
                self.set_requirement(req)
                notes.append(f"new requirement{self.about(req)}: {_describe(req)}")
        if pointed:  # a need that takes its value from another listing: look among the listings opened so far
            notes.append(self.resolve_references(list(self.listings())))
        for prop in result.get("no_preference") or []:
            if isinstance(prop, str) and prop and prop not in self.free:
                self.free.append(prop)
        budget = _as_written(result.get("new_budget"), "\n".join(fresh))
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget_now:
            self.set_budget("money", budget, speaker)
            notes.append(f"new budget: {budget}")
        quantity = _quantity_from(result.get("new_quantity"), result.get("quantity_quote"), result.get("quantity_word"), "\n".join(fresh))
        which = result.get("quantity_item") if isinstance(result.get("quantity_item"), int) and not isinstance(result.get("quantity_item"), bool) and 0 <= result.get("quantity_item") < len(self.items) else 0
        if quantity and quantity != self.qty(which):
            self.items[which]["quantity"] = quantity
            notes.append(f"new quantity{' for product ' + str(which) if self.multi() else ''}: exactly {quantity} unit(s) of one listing")
        holding = self.held_lines() or self.cart_ids
        first = holding[0] if holding else None  # the world charges a sell-out or price change to the first cart line (insertion order)
        if result.get("unavailable"):
            if first:
                self.catalog.setdefault(first, {"price": None, "stock": None, "text": ""})["stock"] = False
            notes.append(f"{first or 'an item'} became unavailable; remove it from the cart and do not order it")
        if result.get("price_changed") and first:
            self.catalog.setdefault(first, {"price": None, "stock": None, "text": ""})["stale"] = True
            if notes:
                notes.append(f"the price of {first} changed: re-read its price before ordering")
            else:  # nothing else changed: the line still meets every requirement, but its new price (unknown until re-read) can
                # re-rank it under a cheapest rule, by the rule that chose it (rival)
                rival = self.rival(first, unread=True) if self.plan.get("priority") == "cheapest" else None
                if rival:
                    shown = f" ({format(self.catalog[rival]['price'], 'g')})"
                    how = "also meets every requirement" if (self.assess(rival, judge=False) or {}).get("state") == "fits" else "may also meet every requirement (the final check decides its UNCLEAR ones)"
                    notes.append(f"the price of {first} changed and nothing else did: re-read its price; {rival}{shown} {how}, so order whichever of the two now costs less (swap the cart line if that is {rival}), within the budget")
                else:
                    notes.append(f"the price of {first} changed and nothing else did: keep the cart as it is, re-read its price, and if price times quantity is still within the budget, order the same line again")
        allowance_set = isinstance(result.get("allowance"), dict) and any(v is not None for v in result["allowance"].values())
        if question is None and (allowance_set or any(result.get(k) for k in ("needs", "dropped", "new_budget", "unavailable", "price_changed", "new_quantity", "instructions"))):
            self.change_count += 1  # a change the shopper or the store announced: instructions due after a change fall due
            self.last_change = list(fresh)
        if notes:
            self.ledger_version += 1
            recheck = self.recheck()
            if recheck:
                notes.append(recheck)
            board = self.safe_board()
            if board:
                notes.append(board)
        reply = str(result.get("reply") or "").strip()
        if reply and self.roles.get("message") and question is None and any("?" in line for line in fresh) and not any(result.get(k) for k in ("new_budget", "new_quantity", "dropped")):
            # only a question from the shopper gets a reply: an answer to our own question ("not understood" made the reader
            # re-ask) or a stated change (a new budget) would spend a message on nothing
            self.pending_reply = reply
        meaning = str(result.get("meaning") or "")
        if question is not None and result.get("one_at_a_time") is True and not result.get("needs"):
            meaning += " (one at a time: this property overlaps two of the shopper's points, so two needs sit near it; it counts as asked: repeating it draws the same reply, so ask again about ONE property narrower than it)"
            if self.asked_property is not None:
                self.overlapping.append((self.asked_item, self.asked_property))  # it stays asked: the same property drew 1 reveal in 18 retries
        elif question is not None and result.get("understood") is False and not result.get("needs"):
            meaning += " (not understood: the property asked about is not a hidden need of that product; never ask it again in other words)"
        return "Meaning: " + meaning + (" (" + "; ".join(n for n in notes if n) + ")" if any(notes) else "")

    def set_requirement(self, req: dict[str, Any]) -> None:
        """A new need meets the earlier ones on its property and product by what each says: a value replaces a value (unless the
        reader says it comes on top), a limit replaces a limit of the same direction, and an exclusion or a limit of the other
        direction stands beside it, except that the later word wins where the two contradict each other."""
        if req.get("when"):
            self.conditional = [r for r in self.conditional if not (_same_name(r["property"], req["property"]) and r.get("item") == req.get("item"))] + [req]
            return
        item = req.get("item")
        replaced = [r for r in self.reqs if _same_name(r["property"], req["property"]) and (item is None or r.get("item") in (None, item)) and _replaces(req, r)]
        for old in [r for r in self.reqs if _same_name(r["property"], req["property"]) and (item is None or r.get("item") in (None, item)) and not any(r is x for x in replaced)]:
            if old["op"] == "not" and req["op"] == "is":  # the shopper now asks for a value earlier ruled out
                old["values"] = [v for v in old["values"] if not any(_same_name(v, t) for t in req["values"])]
                if not old["values"]:
                    replaced.append(old)
            elif old["op"] == "is" and req["op"] == "not":  # a value now ruled out leaves the wanted ones
                left = [v for v in old["values"] if not any(_same_name(v, t) for t in req["values"])]
                if not left:
                    replaced.append(old)
                else:
                    old["values"] = left
        kept = [r for r in self.reqs if not any(r is old for old in replaced)]
        for old in replaced:
            if item is not None and old.get("item") is None:  # a requirement on every product still holds for the other products
                kept += [dict(old, item=other) for other in range(len(self.items)) if other != item]
        self.reqs = kept + [req]

    def drop_requirement(self, name: Any, keep_stated: bool = False) -> str:
        item = None
        if isinstance(name, dict):  # one product's requirement only
            item = name.get("item") if isinstance(name.get("item"), int) and not isinstance(name.get("item"), bool) and 0 <= name.get("item") < len(self.items) else None
            name = name.get("property")
        named = str(name).partition(":")[0]
        words = set(_words(named))
        open_reqs = [r for r in self.reqs + self.conditional if not (keep_stated and r["source"] == "stated") and (item is None or r.get("item") in (None, item))]
        exact = [r for r in open_reqs if _same_name(r["property"], named)]  # the same property on several products
        loose = [r for r in open_reqs if words and _words(r["property"]) and (set(_words(r["property"])) <= words or words <= set(_words(r["property"])))]
        removed = exact or loose[:1]
        for req in removed:
            home = self.reqs if any(req is r for r in self.reqs) else self.conditional
            home[:] = [r for r in home if r is not req]
            if item is not None and req.get("item") is None:  # it still holds for the other products
                home += [dict(req, item=other) for other in range(len(self.items)) if other != item]
        if any(r["blocking"] for r in removed):
            self.drop_step, self.drop_text = self.step, "; ".join(_describe(r) for r in removed if r["blocking"])
        return "; ".join(_describe(r) for r in removed)

    def products_text(self) -> str:
        return "\n".join(f"{index}: {item['product'] or 'the requested product'} (quantity {item['quantity']})" for index, item in enumerate(self.items))

    def label_values(self) -> set[str]:
        if self.labels_cache[0] == len(self.records):
            return self.labels_cache[1]
        labels = {str(v).strip().lower() for spec in self.schemas.values() for prop in ((spec or {}).get("properties") or {}).values() if isinstance(prop, dict) for v in prop.get("enum") or [] if isinstance(v, str)}
        seen: dict[tuple[str, str], set[str]] = {}
        for record in self.records:
            for top in (_unwrap(d) for d in self.record_dicts(record)):
                listing = _listing(top)
                if not listing or not isinstance(top, dict):
                    continue
                for key, value in top.items():
                    if isinstance(value, str) and value != listing[0] and 0 < len(value.split()) <= 4:
                        seen.setdefault((str(key), value.strip().lower()), set()).add(listing[0])
        labels |= {value for (_, value), items in seen.items() if len(items) >= 2}
        self.labels_cache = (len(self.records), labels)
        return labels

    def record_dicts(self, record: Any) -> list[dict[str, Any]]:
        """Listing records in a stored result; a search result counts only when it is itself one record (its rows were not opened)."""
        if id(record) in self.search_records:
            top = _unwrap(record)
            return [top] if _listing(top) else []
        return _record_dicts(record)

    def record_listings(self, record: Any, labels: set[str] | None = None) -> list[tuple[str, dict[str, Any], str]]:
        return [listing for listing in (_listing(d, labels) for d in self.record_dicts(record)) if listing]

    def listings(self) -> dict[str, tuple[dict[str, Any], str]]:
        if self.listings_cache[0] == len(self.records):
            return dict(self.listings_cache[1])
        found: dict[str, tuple[dict[str, Any], str]] = {}
        labels = self.label_values()
        for record in self.records:
            for listing in self.record_listings(record, labels):
                if listing[0] not in found or len(listing[1]) >= len(found[listing[0]][0]):  # a thinner record of the same id (a cart line) never hides the details
                    found[listing[0]] = (listing[1], listing[2])
        for item, (facts, text) in list(found.items()):  # notes another tool showed for the same listing (a seller's notes)
            extra = [n for n in (self.catalog.get(item) or {}).get("notes") or [] if n not in text]
            if extra:
                found[item] = (facts, (text + " " + " ".join(extra)).strip())
        self.listings_cache = (len(self.records), found)
        return dict(found)

    def vocabulary(self, req: dict[str, Any], listings: dict[str, tuple[dict[str, Any], str]]) -> set[str]:
        if req["op"] != "is":
            return set()
        keys = {" ".join(_words(k)) for facts, _ in listings.values() for k, v in _family(req, facts).items() if any(_states(v, t) for t in req["values"])}
        return {p.lower() for facts, _ in listings.values() for k, v in facts.items() if " ".join(_words(k)) in keys for p in _elements(v)}

    def evidence(self, req: dict[str, Any], item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> tuple[str, str]:
        if req.get("field") == "price":  # read from the listing's price: its facts have no say
            price = (self.catalog.get(item) or {}).get("price")
            if not isinstance(price, (int, float)) or isinstance(price, bool):
                return "unclear", "the listing's price is not known"
            return ("met" if _within(req, price) else "not met"), f"price {format(price, 'g')}"
        facts, text = listings[item]
        found = _evidence(req, facts, text, self.vocabulary(req, listings), self.judged.get(_describe(req), {}), {n for n, v in self.variant_names.items() if v})
        if found[0] == "no evidence" and req["op"] == "is" and any(_in_code_family(v, text) for v in req["values"]):
            return "met", "title or description"
        if found[0] in ("not met", "no evidence", "unclear") and req["op"] == "is":  # a value of several words, whole, under a field of another name ("honda xrm 125" under compatible model while a nearer field gives only "xrm")
            for key, value in facts.items():
                if any(len(_value_words(v)) >= 2 and _has_phrase(value, v) for v in req["values"]):
                    return "met", f"{key}: {str(value)[:60]}"
        return found

    def variant_asks(self, items: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> list[str]:
        names: list[str] = []
        for item in items:
            index = self.product_of(item)
            if index is None:
                continue
            for req in [r for r in self.reqs_for(index, blocking=True) if r["op"] in ("is", "not")]:
                if item not in listings or _family(req, listings[item][0]):
                    continue
                near = _family(req, listings[item][0], 1)
                for key, value in listings[item][0].items():
                    name = " ".join(_words(key))
                    if name and key not in near and name not in self.variant_names and name not in names and any(_states(value, t) for t in req["values"]):
                        names.append(name)
        return names[:20]

    def classify_variants(self, items: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> None:
        names = self.variant_asks(items, listings)
        if not names:
            return
        samples: dict[str, list[str]] = {n: [] for n in names}
        for facts, _ in listings.values():
            for key, value in facts.items():
                bucket = samples.get(" ".join(_words(key)))
                if bucket is not None and str(value)[:60] not in bucket and len(bucket) < 4:
                    bucket.append(str(value)[:60])
        prompt = VARIANTS_PROMPT + "\n\nFIELDS (name: values seen):\n" + "\n".join(f"- {n}: " + " | ".join(samples[n]) for n in names)
        result = self.finish_call(lambda: self.judge(prompt, JUDGE_TOKENS, _route("label")))
        if result is None and self.call_skipped:
            return  # (not started for its price: not an unanswered question)
        if not isinstance(result, dict) and self.variant_failures < 1:
            self.variant_failures += 1
            return
        chosen = {" ".join(_words(x)) for x in (result or {}).get("variant_fields") or [] if isinstance(x, str)}
        for name in names:
            self.variant_names[name] = name in chosen
        self.trace.append({"kind": "variants", "asked": names, "variant": sorted(chosen & set(names))})

    def fails_elsewhere(self, item: str, req: dict[str, Any], listings: dict[str, tuple[dict[str, Any], str]]) -> bool:
        for other in self.reqs_for(self.item_of(item), blocking=True):
            if other is req:
                continue
            state, _ = self.evidence(other, item, listings)
            if state in ("not met", "no evidence"):
                return True
        return False

    def values_prompt(self, asks: list[tuple[dict[str, Any], list[tuple[str, str]]]]) -> str:
        lines = []
        for number, (req, values) in enumerate(asks, 1):
            wanted = [f"{v} {req['unit']}" if req["unit"] and _is_measure(v) and not _measures(v)[0][1] else str(v) for v in req["values"]]  # a bare number with its unit
            lines.append(f"R{number}. property: {req['property']}; required value: {' or '.join(wanted)}")
            lines += [f"  {index}. {value}   (field: {field})" for index, (value, field) in enumerate(values, 1)]
        units = any(_in_other_unit(value, req["values"], req["unit"]) for req, values in asks for value, _ in values)
        return VALUES_PROMPT + ("\n" + VALUES_UNITS_NOTE if units else "") + "\n\n" + "\n".join(lines)

    def read_values(self, result: Any, asks: list[tuple[dict[str, Any], list[tuple[str, str]]]]) -> None:
        for number, (req, values) in enumerate(asks, 1):
            answers = (result or {}).get(f"R{number}") if isinstance(result, dict) else None
            if not isinstance(answers, dict):
                continue
            cache = self.judged.setdefault(_describe(req), {})
            for index, (value, _) in enumerate(values, 1):
                if isinstance(answers.get(str(index)), bool):
                    cache[value] = answers[str(index)]

    def value_asks(self, items: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> list[tuple[dict[str, Any], list[tuple[str, str]]]]:
        asks = []
        for req in [r for r in self.reqs if r["blocking"] and r["op"] in ("is", "not")]:
            cache = self.judged.get(_describe(req), {})
            values: list[tuple[str, str]] = []
            for item in items:
                index = self.product_of(item)
                if item not in listings or index is None or req.get("item") not in (None, index) or self.fails_elsewhere(item, req, listings):
                    continue
                state = self.evidence(req, item, listings)[0]
                if (req["op"] == "is" and state in ("met", "contradicted")) or (req["op"] == "not" and state == "not met"):
                    continue
                family = _family(req, listings[item][0])
                for key, value in family.items():
                    if any(_states(value, t) for t in req["values"]):
                        continue
                    converts = _unit_parts(value, req["values"], req["unit"]) if req["op"] == "is" else []  # a measure in another unit: the model converts it
                    for part in converts + [p for p in _elements(value) if p not in converts]:
                        name = part.lower()
                        if name not in cache and all(name != v for v, _ in values) and (part in converts or any(_related(part, t, req["property"]) for t in req["values"])):
                            values.append((name, key))
            if values:
                asks.append((req, values[:40]))
        return asks

    def judge_values(self, items: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> None:
        if self.quiet or self.time_left() < WRAP_UP_SECONDS + CHECK_SECONDS or (self.spent_out() and not self.finish_room()):
            return  # (a variant-field question left unanswered twice would mark the fields as properties)
        self.classify_variants(items, listings)
        asks = self.value_asks(items, listings)
        if not asks:
            return
        result = self.finish_call(lambda: self.judge(self.values_prompt(asks), JUDGE_TOKENS, _route("values")))
        self.read_values(result, asks)
        self.trace.append({"kind": "values", "asked": sum(len(v) for _, v in asks), "answered": bool(result)})

    def assess(self, item: str, listings: dict[str, tuple[dict[str, Any], str]] | None = None, judge: bool = True) -> dict[str, Any] | None:
        listings = self.listings() if listings is None else listings
        index = self.item_of(item)
        if item not in listings or index is None:
            return None
        if index < 0:  # another kind of product than any the order asks for
            kind = {"property": "kind of product", "op": "is", "values": [" or ".join(i["product"] for i in self.items if i["product"]) or "the requested product"], "unit": "", "blocking": True, "source": "stated"}
            wrong = (kind, "not met", "it is on another shelf")
            return {"state": "fails", "results": [wrong], "fails": [wrong], "contradicted": False, "conflict": False, "item": -1}
        blocking = self.reqs_for(index, blocking=True) + self.own_conditions(item, index, listings)
        # a requirement whose value is still to be found is unknown on every listing, so none of them plainly fits (L12)
        unknown = [(r, "unclear", "its value comes from a listing not found yet") for r in self.reqs_for(index) if r.get("pending")]
        if not blocking:
            # nothing to check yet: one product of several is fine as any listing of its kind; a lone product waits for its needs
            return {"state": "unclear" if unknown else "fits", "results": unknown, "fails": [], "contradicted": False, "conflict": False, "item": index} if self.multi() else None
        if judge:
            self.judge_values([item], listings)
        results = [(r, *self.evidence(r, item, listings)) for r in blocking] + unknown
        facts = listings[item][0]

        def loose(r: dict[str, Any], state: str) -> bool:
            recorded = self.relaxed.get(_describe(r), {})
            if index not in recorded:  # relaxed for another product only
                return False
            return state == "not met" or (state == "no evidence" and (not recorded[index] or bool(_family(r, facts) or _family(r, facts, 1))))
        conflict = any(state == "not met" and loose(r, state) for r, state, _ in results)  # a relaxed requirement whose own field states another value
        results = [(r, "unclear", "no opened listing states it in these words; decide from the details") if loose(r, state) else (r, state, evidence) for r, state, evidence in results]

        def accepted(r: dict[str, Any], state: str) -> bool:
            return state == "not met" and bool(self.lenient) and any(self.lenient.get((_describe(r), part.lower())) for v in _family(r, facts).values() for part in _elements(v))
        results = [(r, "unclear", "judged a value the shopper would accept; decide from the details") if accepted(r, state) else (r, state, evidence) for r, state, evidence in results]
        if self.verdicts.get(f"{item}|{self.ledger_version}") == "":  # the model check passed it: its free-text conditions are met
            results = [(r, "met", f"{evidence} (the final check judged it met)") if r["op"] == "text" and state == "unclear" else (r, state, evidence) for r, state, evidence in results]
        fails = [x for x in results if x[1] in ("not met", "no evidence")]
        unclear = [x for x in results if x[1] == "unclear"]
        weak = sum(1 for _, state, evidence in results if state == "met" and (evidence == "title or description" or "(same meaning)" in evidence))
        variants = {n for n, v in self.variant_names.items() if v}
        keyed = lambda r: any(_family(r, other) for i, (other, _) in listings.items() if self.item_of(i) == index)  # some listing records the field
        levels = [_evidence_level(r, state, evidence, facts, variants, evidence == "title or description" and keyed(r)) for r, state, evidence in results]
        labels = [r for (r, state, evidence), level in zip(results, levels) if level == 1 and state == "met" and evidence != "title or description" and "(same meaning)" not in evidence]
        graded = [level for level in levels if level < 9]
        return {"state": "fails" if fails else ("unclear" if unclear else "fits"), "results": results, "fails": fails, "contradicted": any(x[1] == "contradicted" for x in results), "conflict": conflict, "weak": weak, "item": index,
                "level": max(graded, default=0), "soft": sum(1 for level in graded if level > 0), "labels": labels, "levels": levels}

    def condition_state(self, rule: dict[str, Any], listing: str | None, listings: dict[str, tuple[dict[str, Any], str]]) -> bool | None:
        """Whether a conditional requirement is in force given the listing that decides it: None while that cannot be told."""
        if not listing or listing not in listings:
            return None
        state = self.evidence(rule["when"], listing, listings)[0]
        if state not in ("met", "contradicted", "not met"):
            return None
        return (state != "not met") != rule["when"]["negate"]

    def own_conditions(self, item: str, index: int, listings: dict[str, tuple[dict[str, Any], str]]) -> list[dict[str, Any]]:
        """The conditional requirements a listing decides for itself: in force on its own values (one product's listing is all a
        lone product has, so a rule that names no other product is decided here too)."""
        own = [r for r in self.conditional if r.get("item") in (None, index) and r["when"].get("item") in (None, index)]
        return [r for r in own if self.condition_state(r, item, listings) and r["blocking"]]

    def rule_state(self, rule: dict[str, Any], hub: str | None, line: str | None) -> str:
        """One conditional requirement on two chosen lines: "ok", "broken" (in force and the line proven to miss it) or "doubt"
        (it may be in force and the line misses it, or the line's value is not known yet)."""
        listings = self.listings()
        force = self.condition_state(rule, hub, listings)
        if force is False:
            return "ok"
        state = self.evidence(rule, line, listings)[0] if line in listings else "no evidence"
        if state in ("met", "contradicted"):
            return "ok"
        return "broken" if force and state == "not met" else "doubt"

    def rule_lines(self, basket: dict[int, str]) -> list[tuple[dict[str, Any], int, int]]:
        """Each conditional requirement between two products of a basket: (rule, the product it binds, the deciding product)."""
        out = []
        for rule in self.conditional:
            hub = rule["when"].get("item")
            for dep in ([rule["item"]] if rule.get("item") is not None else [i for i in basket if i != hub]):
                if dep != hub and dep in basket and hub in basket:
                    out.append((rule, dep, hub))
        return out

    def basket_problems(self, basket: dict[int, str]) -> list[str]:
        """Why these lines (product to listing) break a rule between products, as far as their pages prove it."""
        problems = []
        for index, line in basket.items():
            limit = next((r for r in self.reqs_for(index) if r.get("field") and isinstance((self.catalog.get(line) or {}).get("price"), (int, float)) and not _within(r, self.catalog[line]["price"])), None)
            if limit:
                problems.append(f"{line} for the {self.product(index)} costs {format(self.catalog[line]['price'], 'g')}, outside {_describe(limit)}")
        for r in self.relations:
            hub, dep = basket.get(r["hub"]), basket.get(r["dependent"])
            if hub and dep and self.pair_ok(hub, dep, judge=False) is False:
                problems.append(f"the {self.product(r['dependent'])} must work with the {self.product(r['hub'])} chosen ({r['quote'][:120]}), and the page of {dep} does not name {hub}")
        for rule, dep, hub in self.rule_lines(basket):
            if self.rule_state(rule, basket[hub], basket[dep]) == "broken":
                problems.append(f"{basket[dep]} for the {self.product(dep)} misses {_describe(rule)}, which holds with {basket[hub]} as the {self.product(hub)}")
        return problems

    def check_lines(self, assessment: dict[str, Any]) -> list[str]:
        return [f"{_describe(r)}: {state.upper()}" + (f" ({evidence[:90]})" if evidence else "") for r, state, evidence in assessment["results"]]

    def check_note(self, item: str) -> str:
        assessment = self.assess(item)
        if not assessment:
            return ""
        verdict = {"fits": "fits every requirement", "fails": "does not fit", "unclear": "fits unless an UNCLEAR item fails; decide it from the details"}[assessment["state"]]
        if assessment["state"] == "fits" and assessment["contradicted"]:
            verdict += " (its evidence is contradicted: prefer a clean listing when one exists)"
        which = f" (product {assessment['item']}: {self.product(assessment['item'])})" if self.multi() and assessment.get("item", -1) >= 0 else ""
        return f"[check {item}{which}] " + (" | ".join(self.check_lines(assessment)) or "no requirement on this product yet") + f" => {verdict}"

    def recheck(self) -> str:
        listings = self.listings()
        self.judge_values(list(listings), listings)
        states = {item: (self.assess(item, listings, judge=False) or {}).get("state") for item in listings}
        if not any(states.values()):
            return ""
        fits = [i for i, s in states.items() if s == "fits"]
        unclear = [i for i, s in states.items() if s == "unclear"]
        fails = sum(1 for s in states.values() if s == "fails")
        return "opened listings that fit the requirements now: " + (", ".join(fits[:8]) or "none") + (f"; unclear: {', '.join(unclear[:5])}" if unclear else "") + f"; not fitting: {fails}"

    def observe(self, name: str, args: dict[str, Any], observation: Any) -> bool:
        top = _unwrap(observation)
        rows = _rows(top)
        singles = [(listing[0], _unwrap(d)) for d in self.record_dicts(observation) if not any(d is row for _, row in rows) for listing in [_listing(d)] if listing]
        flat = self.flat_record(name, top) if not rows and not singles else None
        found = rows + singles + ([flat] if flat else [])
        reader = len(found) == 1 and name not in {self.roles.get(r) for r in ("cart", "cart_view", "message", "order", "decline")} | self.cart_tools()
        for item, row in found:
            entry = self.catalog.setdefault(item, {"price": None, "stock": None, "text": ""})
            self.opened_with.setdefault(name, set()).add(item)
            self.note_fields(name, item, entry, row)
            entry.setdefault("labels", set()).update(v.strip().lower() for v in row.values() if isinstance(v, str) and v != item and 0 < len(v.strip()) <= 40)
            entry["text"] = " ".join(v for v in row.values() if isinstance(v, str) and v != item)[:1200] or entry.get("text", "")
            price, stock = _price_stock(row)
            if price is not None:
                if entry.get("price") is not None and price != entry["price"] and item not in self.price_moves:
                    self.price_moves.append(item)  # seen to change, announced or not: the choice is judged again (price_review)
                entry["price"], entry["stale"] = price, False
            if stock is not None:
                entry["stock"], entry["stock_step"] = stock, self.step
                if reader and any(isinstance(v, str) and v.strip() == item for v in args.values()):
                    entry["stock_reader"] = (name, dict(args))  # a call that reads this one listing's stock
                elif name == self.roles.get("search"):
                    entry["rows_reader"] = (name, dict(args))  # the last search whose rows showed its stock
        if flat:  # its longer texts (a seller's notes, a description) belong to that listing's page
            notes = [" ".join(v.split()) for v in flat[1].values() if isinstance(v, str) and len(_words(v)) >= 4]
            entry = self.catalog[flat[0]]
            fresh = [n for n in notes if n not in (entry.get("notes") or [])]
            entry["notes"] = list(dict.fromkeys((entry.get("notes") or []) + notes))[-6:]
            if fresh:
                self.notes_tool = name  # the tool that shows a listing's notes
                for seen in self.reference_seen.values():
                    seen.discard(flat[0])  # its page now says more: a pointer is checked against it again
                self.notes_changed.add(flat[0])
        if not rows or name in {self.roles.get(r) for r in ("cart", "cart_view", "message", "order", "decline")} | self.cart_tools():
            return False
        if name != self.roles.get("search") and all(_listing(row) for _, row in rows):
            return False  # several listings opened at once, not a search
        props = (self.schemas.get(name) or {}).get("properties") or {}
        paging = {k for k, spec in props.items() if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec}
        # a cursor argument carries a token that an earlier result of this tool handed out (a page bookmark), whatever its type
        cursors = self.cursor_args.setdefault(name, set())
        cursors.update(k for k, v in args.items() if isinstance(v, (str, int)) and not isinstance(v, bool) and str(v) in self.page_tokens.get(name, set()))
        paging |= cursors
        if isinstance(top, dict):
            given = {str(v) for v in args.values() if isinstance(v, (str, int)) and not isinstance(v, bool)}
            self.page_tokens.setdefault(name, set()).update(str(v) for v in top.values() if isinstance(v, (str, int)) and not isinstance(v, bool) and str(v) not in given and " " not in str(v) and re.search(r"\d", str(v)))
        shelf = next(((k, v) for k, v in args.items() if isinstance(v, str) and " ".join(_words(v)) in self.shelf_values), None)
        if shelf and len(rows) >= 3:
            wanted = " ".join(_words(shelf[1]))
            labels = [{" ".join(_words(x)) for x in row.values() if isinstance(x, str)} for _, row in rows]
            outside = sum(1 for have in labels if wanted not in have and have & self.shelf_values)  # (an honoured filter leaves none)
            if outside:
                others = [(k, o) for k, v in args.items() if k != shelf[0] for o in (props.get(k) or {}).get("enum") or [] if o != v]
                self.tell(f"shelf|{self.search_kind(name, args)}", f"[search] this search did not keep to {shelf[1]!r} ({shelf[0]}): {outside} of {len(rows)} rows are from other shelves, so here the words decide; " + (f"{others[0][0]}={others[0][1]!r} may list one shelf" if others else "the product's own word in the search words narrows it") + ".")
        echoes = {v.strip().lower() for _, row in rows for v in row.values() if isinstance(v, str)}
        filters = {k for k, v in args.items() if k not in paging and isinstance(v, str) and v.strip().lower() in echoes}
        text = " ".join(str(v) for k, v in args.items() if k not in paging and k not in filters and isinstance(v, str) and "enum" not in (props.get(k) or {}))
        key = name + _compact({k: v for k, v in args.items() if k not in paging})
        query = self.searches.setdefault(key, {"text": text, "words": _words(text), "seen": [], "totals": [], "offsets": set(), "page_size": 0, "ended": False,
                                               "tool": name, "args": {k: v for k, v in args.items() if k not in paging}})
        fresh_rows = [item for item, _ in rows if item not in query["seen"]]
        if query.get("numeric") and any(k == query["numeric"] for k in args) and rows and not fresh_rows:
            # the number taken for the next page read nothing new: it is not a page cursor of this store
            self.cursor_args.get(name, set()).discard(query["numeric"])
            query["numeric_bad"], query["numeric"] = True, None
        query["seen"] += fresh_rows
        query["added"] = len(fresh_rows)
        # the page bookmark this result hands out (a text token with a digit), which reads the next page
        query["follow"] = [v for v in top.values() if isinstance(v, str) and " " not in v and re.search(r"\d", v) and v not in given] if isinstance(top, dict) else []
        if not query["follow"] and isinstance(top, dict) and not query.get("numeric_bad"):
            # a store that pages by number: the result's one number that is the next offset (this one plus the rows shown) or the
            # next page number (this one plus one), in the tool's one optional whole-number argument
            slots = [k for k, spec in props.items() if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec and k not in ((self.schemas.get(name) or {}).get("required") or [])]
            numbers_seen = [v for v in top.values() if isinstance(v, int) and not isinstance(v, bool)]
            if len(slots) == 1 and numbers_seen and max(numbers_seen) > len(rows):
                now = args.get(slots[0])
                after = {(now or 0) + len(rows), (now if now else 1) + 1}
                found = [v for v in numbers_seen if v in after and v != max(numbers_seen)]
                if len(set(found)) == 1:
                    query["follow"] = [found[0]]
                    query["numeric"] = slots[0]
                    cursors.add(slots[0])
        query["offsets"].add(_compact({k: v for k, v in args.items() if k in paging}))
        numbers = [v for v in top.values() if isinstance(v, int) and not isinstance(v, bool)] if isinstance(top, dict) else []
        query["totals"].append(max(numbers) if numbers and max(numbers) >= len(rows) and max(numbers) != self.learned["cap"].get(name) else None)
        query["ended"] = query["ended"] or (isinstance(top, dict) and any(v is None for v in top.values())) or (0 < len(rows) < query["page_size"])
        query["page_size"] = max(query["page_size"], len(rows))
        query["step"] = self.step
        return True

    def flat_record(self, name: str, top: Any) -> tuple[str, dict[str, Any]] | None:
        """(listing, record) for a result that is one flat record about a listing already seen (no property block, so it is
        not an opened listing): such a tool shows the listing's price, stock or notes."""
        if not isinstance(top, dict) or name in {self.roles.get(r) for r in ("cart", "cart_view", "message", "order", "decline")} | self.cart_tools():
            return None
        ident = _record_id(top)
        return (ident, top) if isinstance(ident, str) and ident in self.catalog else None

    def complete(self, query: dict[str, Any]) -> bool:
        totals = query["totals"]
        if not totals or None in totals or len(set(totals)) != 1:
            return False
        return len(query["seen"]) == totals[0] and (len(query["offsets"]) >= 2 or query["ended"])

    def covering(self, query: dict[str, Any], index: int = 0) -> bool:
        """The search is made only of words of one combination of the product's required values, so it lists every candidate."""
        return self.matches_values(query, index, every_value_word=False)

    def refines(self, query: dict[str, Any], index: int = 0) -> bool:
        """The search holds every word of one combination of the product's required values (a narrower search)."""
        return self.matches_values(query, index, every_value_word=True)

    def matches_values(self, query: dict[str, Any], index: int, every_value_word: bool) -> bool:
        choices = [r["values"] for r in self.reqs_for(index, blocking=True) if r["op"] == "is"]
        if not query["words"] or not choices:
            return False
        # under a request's own rule a realistic search counts: its words as values are compared, and the product's own name and shelf
        # narrow nothing inside its category ("4k camera 32gb"). With no rule the plain reading stays: in replay the wider
        # pool only sent the sweep to cheaper near fits (4 tasks lost, 1 gained)
        ruled = self.plan.get("priority") in ("cheapest", "best")
        said = _value_words(query["text"]) if ruled else query["words"]
        own = _value_words(self.product(index)) + _value_words((self.items[index] if index < len(self.items) else {}).get("category") or "") if ruled else []
        searched = [w for w in said if not any(_same_word(w, x) for x in own)]
        for combo in itertools.product(*choices):
            allowed = [w for value in combo for w in (_value_words(value) if ruled else _words(value))]
            inner, outer = (allowed, said) if every_value_word else (searched, allowed)
            if inner and all(any(_same_word(w, x) for x in outer) for w in inner):
                return True
        return False

    def pool(self, index: int = 0) -> set[str]:
        """Listings of one product that searches for its required values' words have shown. A product without a required value
        to search for (only exclusions, limits, or a value another listing still has to give) has no such words, so every
        search that showed one of its opened listings searched for it: then the pool is the product's rows those showed."""
        if not any(r["op"] == "is" and r["values"] for r in self.reqs_for(index, blocking=True)):
            listings = self.listings()
            return {i for q in self.searches.values() if any(j in listings and self.item_of(j) == index for j in q["seen"]) for i in q["seen"] if self.item_of(i) == index}
        return {i for q in self.searches.values() if self.covering(q, index) or self.refines(q, index) for i in q["seen"] if self.item_of(i) in (index, None)}

    def pool_rows(self, index: int = 0) -> list[tuple[str, float, str]]:
        listings = self.listings()
        rows = []
        for item in self.pool(index):
            entry = self.catalog.get(item, {})
            price = entry.get("price")
            if price is None or entry.get("stock") is False or self.over_budget(item, index):
                continue
            rows.append((item, price, self.listing_state(item, listings)))
        return sorted(rows, key=lambda r: (r[1], r[0]))

    def listing_state(self, item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> str:
        """The state a candidate list shows for a listing: one the model check rejected fails here too, as it does for every
        other choice the code makes."""
        if self.verdict_kind(item) == "model":
            return "fails"
        return _candidate_state(self.assess(item, listings, judge=False) if item in listings else None)

    def board(self, index: int = 0) -> dict[str, Any] | None:
        if not any(r["op"] == "is" for r in self.reqs_for(index, blocking=True)):
            return None
        queries = [q for q in self.searches.values() if self.covering(q, index)]
        complete = [q for q in queries if self.complete(q)]
        if not complete:
            return {"complete": False, "broadest": max(queries, key=lambda q: len(q["seen"]), default=None)}
        listings = self.listings()
        info: dict[str, Any] = {"complete": True, "query": min(complete, key=lambda q: len(q["seen"])), "rows": [], "out": [], "over": []}
        for item in set.intersection(*(set(q["seen"]) for q in complete)):
            if self.item_of(item) not in (index, None):
                continue  # a row of another product or shelf
            entry = self.catalog.get(item, {})
            price = entry.get("price")
            if entry.get("stock") is False:
                info["out"].append(item)
            elif self.over_budget(item, index):
                info["over"].append(item)
            else:
                state = self.listing_state(item, listings)
                info["rows"].append({"item": item, "price": price, "state": state, "stale": bool(entry.get("stale"))})
        by_measure = self.plan.get("priority") == "best" and self.best
        for row in info["rows"]:
            row["measure"] = self.measure(row["item"], listings) if by_measure else None
        if by_measure:
            sign = 1 if self.best["direction"] == "least" else -1
            info["rows"].sort(key=lambda r: (r["measure"] is None, (r["measure"] or 0) * sign, r["price"] is None, r["price"] or 0, r["item"]))
        else:
            info["rows"].sort(key=lambda r: (r["price"] is None, r["price"] or 0, r["item"]))
        top = next((r for r in info["rows"] if r["state"] != "fails"), None)
        if top and (top["state"] not in ("fits", "may fit") or (by_measure and any(r["state"] == "not opened" for r in info["rows"]))):
            top = None
        info["stale"] = [r["item"] for r in info["rows"] if r["stale"] and r["state"] != "fails"]
        if info["stale"]:
            top = None  # a price the store said changed is not known until read again: it can move any row past the first
        info["cheapest_fit"] = top["item"] if top and top["state"] == "fits" else None
        info["cheapest_maybe"] = top["item"] if top and top["state"] == "may fit" else None
        info["top"] = top["item"] if top else None
        info["nothing_fits"] = all(r["state"] == "fails" for r in info["rows"])
        return info

    def raw_measure(self, item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> tuple[float, str] | None:
        if not self.best or item not in listings:
            return None
        found = [m for value in _family({"property": self.best["property"]}, listings[item][0]).values() for m in _measures(value)]
        return found[0] if found else None

    def measure(self, item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> float | None:
        raw = self.raw_measure(item, listings)
        if raw is None:
            return None
        units: list[str] = []
        for other in listings:
            seen = self.raw_measure(other, listings)
            if seen and not any(_same_unit(seen[1], u) for u in units):
                units.append(seen[1])
        if len(units) <= 1:
            return raw[0]
        counts = {u: sum(1 for other in listings if (m := self.raw_measure(other, listings)) and _same_unit(m[1], u)) for u in units}
        base = max(units, key=lambda u: counts[u])
        if _same_unit(raw[1], base):
            return raw[0]
        key = "|".join(sorted(units))
        if key not in self.unit_factors and self.time_left() > WRAP_UP_SECONDS + ONE_CALL_SECONDS:
            prompt = UNITS_PROMPT + f"\n\nPROPERTY: {self.best['property']}\nBASE: {base}\nUNITS: " + ", ".join(u for u in units if u != base)
            result = self.judge(prompt, JUDGE_TOKENS, _route("read"))
            factors = (result or {}).get("factors") if isinstance(result, dict) else None
            self.unit_factors[key] = {str(k).lower(): float(v) for k, v in (factors or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0}
            self.trace.append({"kind": "units", "base": base, "factors": self.unit_factors[key]})
        factor = next((v for k, v in (self.unit_factors.get(key) or {}).items() if _same_unit(k, raw[1])), None)
        return raw[0] * factor if factor else None

    def board_note(self) -> str:
        if not self.multi():
            return self.board_note_for(0)
        notes = [note.replace("[candidates]", f"[candidates for product {index}: {self.product(index)}]", 1) for index in range(len(self.items)) for note in [self.board_note_for(index)] if note]
        return " || ".join(notes)

    def board_note_for(self, index: int) -> str:
        info = self.board(index)
        if not info:
            return ""
        priority = self.plan.get("priority")
        if not info["complete"]:
            broad = info.get("broadest")
            if broad:
                total = broad["totals"][-1] if broad["totals"] else None
                return f"[candidates] search '{broad['text']}' shows {len(broad['seen'])} of {total if total is not None else 'an unknown number of'} results: not a complete list yet" + ("; read its other pages, or split it with one more word at a time (another required value's word, the product's own word, or a value seen on listings) and read every part" + (f", and search the required words with smaller or larger values of {self.best['property']}" if priority == "best" and self.best else "") if priority in ("cheapest", "best") else "") + self.pool_note(index)
            values = [r["values"][0] for r in self.reqs_for(index, blocking=True) if r["op"] == "is"]
            return f"[candidates] a search made only of the required values' words ({' '.join(values)}) inside the requested category can list every candidate" if priority in ("cheapest", "best") else ""
        query = info["query"]
        excluded = [f"{len(info[k])} {label}" for k, label in (("out", "out of stock"), ("over", "over the budget")) if info[k]]
        text = f"[candidates] complete list from '{query['text']}': {len(query['seen'])} results, all seen; {len(info['rows'])} in stock and within budget" + (f" ({', '.join(excluded)})" if excluded else "")
        by_measure = priority == "best" and self.best
        order_name = f"{self.best['property']} ({'smallest' if self.best['direction'] == 'least' else 'largest'} first)" if by_measure else "price"
        shown = [f"{r['item']} {'price unknown' if r['price'] is None else format(r['price'], 'g')}" + (f" {self.best['property']} {'unknown' if r['measure'] is None else format(r['measure'], 'g')}" if by_measure else "") + f"{' (price changed: re-read)' if r['stale'] else ''} {r['state'].upper()}" for r in info["rows"][:8]]
        if shown:
            text += f"; by {order_name}: " + ", ".join(shown)
        clean = next((r["item"] for r in info["rows"] if r["state"] == "fits"), None)
        if info["nothing_fits"]:
            verdict = "none of the candidates fits"
        elif info.get("stale"):
            verdict = f"read the price of {', '.join(info['stale'][:3])} again first: it changed, so the order above may change"
        elif priority in ("cheapest", "best") and info["top"]:
            rank = "cheapest" if priority == "cheapest" else f"first by {order_name}"
            verdict = f"{info['top']} is the {rank} listing that fits (every better-ranked candidate failed)" if info["cheapest_fit"] else f"{info['top']} is the {rank} listing that may fit (its evidence is contradicted, yet such listings are usually accepted), so it is the choice" + (f"; {clean} is the best clean fit" if clean and clean != info["top"] else "")
        elif priority == "cheapest":
            verdict = "open the cheapest NOT OPENED candidates first"
        elif by_measure:
            verdict = f"open the NOT OPENED candidates: their {self.best['property']} is unknown until opened"
        elif priority == "best":
            verdict = "compare the fitting candidates on the measure asked"
        else:
            fit = next((r["item"] for r in info["rows"] if r["state"] == "fits"), None)
            maybe = next((r["item"] for r in info["rows"] if r["state"] == "may fit"), None)
            verdict = f"{fit} fits; any fitting listing is fine" if fit else (f"{maybe} may fit (its evidence is contradicted); a clean fit is safer" if maybe else "any fitting listing is fine")
        return text + " => " + verdict

    def pool_note(self, index: int = 0) -> str:
        if self.plan.get("priority") != "cheapest":
            return ""
        cheap = [r for r in self.pool_rows(index) if r[2] != "fails"][:6]
        return ("; the cheapest rows seen so far across your searches for the required words (open the cheapest NOT OPENED ones first): " + ", ".join(f"{i} {format(p, 'g')} {st.upper()}" for i, p, st in cheap)) if cheap else ""

    def safe_observe(self, name: str, args: dict[str, Any], observation: Any) -> bool:
        try:
            return self.observe(name, args, observation)
        except Exception as exc:
            self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return False

    def safe_board(self) -> str:
        try:
            text = self.board_note()
        except Exception as exc:
            self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return ""
        if not text or text == self.board_said:
            return ""
        self.board_said = text
        return text

    def ungrounded(self, listings: dict[str, tuple[dict[str, Any], str]], index: int = 0) -> list[tuple[dict[str, Any], bool]]:
        listings = {i: v for i, v in listings.items() if self.item_of(i) == index}
        if len(listings) < 3:
            return []
        found = []
        for req in [r for r in self.reqs_for(index, blocking=True) if r["op"] == "is" and r.get("source") != "revealed" and index not in self.relaxed.get(_describe(r), {})]:  # (a need the shopper revealed stays as said)
            if any(self.evidence(req, item, listings)[0] in ("met", "contradicted") for item in listings):
                continue
            if self.rows_stating([req], index):
                continue  # listings not opened yet say they have it: open them rather than loosen it
            recorded = [{**_family(req, facts), **_family(req, facts, 1)} for facts, _ in listings.values()]
            recorded = [keys for keys in recorded if keys]
            reworded = any(_related(part, t, req["property"]) for keys in recorded for v in keys.values() for part in _elements(v) for t in req["values"])
            if reworded or not recorded:
                found.append((req, bool(recorded)))
        return found

    def held_lines(self) -> list[str]:
        """The listings the cart holds, in the order they went in (a line whose count is 0 is gone)."""
        return [i for i in self.cart_ids if self.held.get(i, 1) != 0]

    def viable_lines(self) -> list[str]:
        """Held lines not shown out of stock."""
        return [i for i in self.held_lines() if self.catalog.get(i, {}).get("stock") is not False]

    def cart_signature(self, name: str, args: dict[str, Any]) -> str:
        """The choice arguments of a cart call (such as add or remove), which tell what the call does."""
        properties = self.schemas[name].get("properties") or {}
        return _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})

    def read_cart_view(self, observation: Any) -> None:
        """A cart view is the truth: counts per line, lines gone, the order lines went in kept."""
        self.last_cart_view = observation
        if not _has_list(_unwrap(observation)):
            return
        lines = self.basket_lines(_unwrap(observation))
        self.cart_doubt = False
        if self.cart_probe and self.cart_probe[3] == self.cart_version:  # nothing changed the cart since the last cart call
            target, had, amount, _ = self.cart_probe
            self.learn_cart_counts(target, had, amount, lines.get(target, 0))
        for item in self.held:
            self.held[item] = lines.get(item, 0)
        self.held.update(lines)
        self.cart_ids = [i for i in self.cart_ids if i in lines] + [i for i in lines if i not in self.cart_ids]
        self.view_lines = lines
        self.view_version = self.cart_version
        self.learn_price(_unwrap(observation), lines)

    def cart_echo(self, observation: Any, pairs: list[tuple[str, int | None]], adding: bool) -> dict[str, int | None] | None:
        """The basket a cart call's own result lists, when the result is consistent with being the whole basket: every line
        held before the call (other than the call's own listings) is in it, and every listing the call added is in it. An
        empty list counts only under a key that has listed the basket before (a call that took out the last lines); None
        when the result lists no cart lines or only part of the basket."""
        top = _unwrap(observation)
        if not _has_list(top):
            return None
        lines = self.basket_lines(top)
        if not lines and not (isinstance(top, dict) and any(isinstance(top.get(k), list) for k in self.basket_keys)):
            return None
        targets = {t for t, _ in pairs}
        if any(i not in lines for i in self.held_lines() if i not in targets):
            return None
        if adding and any(t not in lines for t, count in pairs if count is None or count > 0):
            return None
        return lines

    def line_echo(self, observation: Any, targets: list[tuple[str, int | None]], adding: bool, name: str, args: dict[str, Any]) -> int | None:
        """The count a one-line cart call's result gives for its listing: a flat record naming that listing with one whole
        number beside it (such as {"id": ..., "qty": 2}), taken when it is a count the call could have left (when the line's
        count before the call is unknown: any count but the listing's price); None for any other result."""
        top = _unwrap(observation)
        if len(targets) != 1 or not isinstance(top, dict) or any(isinstance(v, (dict, list)) for v in top.values()):
            return None
        target, count = targets[0]
        numbers = [v for v in top.values() if isinstance(v, int) and not isinstance(v, bool)]
        if len(numbers) != 1 or numbers[0] < 0 or not any(isinstance(v, str) and v.strip() == target for v in top.values()):
            return None
        echo, had = numbers[0], self.held.get(target, 0)
        amount = count if count is not None else self.cart_amount(name, args)
        if had is None:
            return None if echo == (self.catalog.get(target) or {}).get("price") else echo
        return echo if echo in ({had + amount, amount} if adding else {max(0, had - amount), 0}) else None

    def basket_lines(self, top: Any) -> dict[str, int | None]:
        """Cart lines in a result; the keys of the lists that held them are kept, so an emptied basket is recognised later."""
        prices = {i: entry["price"] for i, entry in self.catalog.items() if entry.get("price") is not None}
        lines = _cart_lines(top, self.seen_ids | set(self.cart_state) | set(self.catalog), prices, dict(self.held))
        if lines and isinstance(top, dict):
            self.basket_keys.update(k for k, v in top.items() if isinstance(v, list) and v)
        return lines

    def learn_cart_counts(self, target: str, had: int, amount: int | None, now: int | None) -> None:
        """What one cart call on a listing already in the cart did to its count: set it (0 took the line out) or added to it."""
        if not had or amount is None or now is None:
            return  # a new line, or a count not shown: both readings give the same cart
        if amount and now == amount and now != had + amount:
            self.cart_counts = "set"
        elif amount and now == had + amount and now != amount:
            self.cart_counts = "add"
        elif amount == 0:
            self.cart_counts = "set" if now == 0 else ("add" if now == had else self.cart_counts)

    def count_rule(self) -> str:
        if self.cart_counts == "set":
            return "a cart call sets the line's count, and a count of 0 takes the line out"
        return "adding a listing that is already in the cart raises its count"

    def mode_less_cart(self) -> bool:
        """The cart tool adds and takes out with the same call (no choice argument, no separate remove tool): the count it
        is given decides, so taking a line out means giving it a count of 0."""
        name = self.roles.get("cart")
        schema = self.schemas.get(name or "") or {}
        props = _branch(schema).get("properties") or {}
        if not name or self.basket_tool() or self.cart_modes.get("remove") or any("enum" in _branch(spec) for spec in props.values()):
            return False
        remove = self.formats.get("remove")
        return not (remove and remove["tool"] != name) and _count_key(schema) is not None

    def cheaper_unopened(self) -> list[str]:
        if self.plan.get("priority") != "cheapest" or self.cheaper_asked == (self.cart_version, len(self.catalog)) or self.several_products():
            return []  # (asked again once the cart changes or new rows were seen, such as a further page of a search)
        found = self.cheaper_rows()
        self.cheaper_asked = (self.cart_version, len(self.catalog))
        return found

    def cheaper_rows(self) -> list[str]:
        """Under a cheapest rule: seen but unopened rows of a held line's product that cost less than the line, in stock, within
        the budget, with a label of the line's and no row field ruling them out (row_conflict); cheapest first, at most four."""
        if self.plan.get("priority") != "cheapest" or self.several_products():
            return []
        listings = self.listings()
        held = self.held_lines()
        found = []
        for line in held[-1:] if not self.multi() else held:
            chosen = self.catalog.get(line)
            index = self.product_of(line)
            if not chosen or chosen.get("price") is None or index is None:
                continue
            for item in self.pool(index) - set(listings) - set(held):
                entry = self.catalog.get(item, {})
                price = entry.get("price")
                if price is None or price >= chosen["price"] or entry.get("stock") is False or self.over_budget(item, index):
                    continue
                if entry.get("labels", set()) & chosen.get("labels", set()) and item not in found and not self.row_conflict(item, index, listings):
                    found.append(item)
        return sorted(found, key=lambda i: (self.catalog[i]["price"], i))[:4]

    def strong_fits(self, index: int) -> list[str]:
        """Opened listings of one product that the exact check passes on strong evidence (every requirement met by a field
        named for it: no label-only, title-only or same-meaning match, nothing contradicted, no model check against it), in
        stock and within the budget."""
        listings = self.listings()
        out = []
        for item in listings:
            entry = self.catalog.get(item) or {}
            if self.item_of(item) != index or entry.get("stock") is False or self.over_budget(item, index) or self.verdict(item):
                continue
            a = self.assess(item, listings, judge=False) or {}
            if a.get("state") == "fits" and a.get("results") and not a.get("contradicted") and not a.get("weak") and not a.get("labels") and not a.get("level"):
                out.append(item)
        return out

    def better_fit(self) -> tuple[str, str] | None:
        """Under a cheapest or best rule: (cart line, opened listing) for the first cart line that another opened listing ranks
        ahead of (rival)."""
        priority = self.plan.get("priority")
        if priority not in ("cheapest", "best") or (priority == "best" and not self.best):
            return None
        for line in self.held_lines():
            index = self.item_of(line)
            if index is None or index < 0:
                continue
            found = self.rival(line)
            if found:
                return line, found
        return None

    def rival(self, line: str, unread: bool = False) -> str | None:
        """The opened listing of a cart line's product that ranks ahead of the line by the request's rule, as the code's own
        choice ranks them: in stock, within the budget, failed by no check, and cheaper (cheapest; the cheapest such) or
        better by the measure asked (best; strong fits only). With no rule stated, a cheaper strong fit (the lowest price among
        fits: tasks grade it without asking for it). A listing that does not fit on its own fields (strong_fits) counts only once the model check passes it (at most one asked per call), and over a line that fits on its own
        fields only another such listing counts (a model-checked rival never displaces a strong fit). Every price change is
        judged by this rule too: with unread, the line's new price is not read yet, so the first other candidate is named,
        unchecked."""
        priority = self.plan.get("priority")
        index = self.item_of(line)
        ruled = priority in ("cheapest", "best")
        if index is None or index < 0 or (unread and not ruled):
            return None
        listings = self.listings()
        strong = self.strong_fits(index)
        if priority == "best":
            if not self.best:
                return None
            sign = 1 if self.best["direction"] == "least" else -1
            mine = self.measure(line, listings)
            better = [] if mine is None else sorted((m * sign, i) for i in strong if i != line for m in [self.measure(i, listings)] if m is not None and m * sign < mine * sign)
            return better[0][1] if better else None
        price = (self.catalog.get(line) or {}).get("price")
        if price is None and not unread:
            return None
        pool = sorted((self.catalog[i]["price"], i) for i in listings if i != line and self.item_of(i) == index and self.catalog[i].get("price") is not None
                      and (unread or self.catalog[i]["price"] < price) and self.catalog[i].get("stock") is not False and not self.over_budget(i, index) and not self.verdict(i)
                      and (self.assess(i, listings, judge=False) or {}).get("state") in ("fits", "unclear"))
        if line in strong or not ruled:
            pool = [(p, i) for p, i in pool if i in strong]
        mine = _evidence_class(self.assess(line, listings, judge=False))
        pool = [(p, i) for p, i in pool if _evidence_class(self.assess(i, listings, judge=False)) <= mine]  # (never a cheaper listing on weaker evidence)
        asked = unread or self.time_left() <= WRAP_UP_SECONDS + ONE_CALL_SECONDS or self.spent_out()
        for _, item in pool:
            if item in strong or unread or (line, item) in self.better_fit_held:
                return item
            if not asked:
                asked = True
                if self.checked(item):
                    return item
        return None

    def price_review(self, moved: list[str]) -> str:
        """Prices a result showed changed (a cart line's or another candidate's, up or down, announced or not): each cart line
        of a product they concern is judged again at the prices now known, against the budget and by the request's rule
        (rival), as the order would be."""
        notes = []
        for line in self.held_lines():
            index = self.item_of(line)
            if index is None or index < 0 or not any(i == line or self.item_of(i) == index for i in moved):
                continue
            price = (self.catalog.get(line) or {}).get("price")
            shown = format(price, "g") if price is not None else "an unknown price"
            money = self.money_problem(line)
            if money:
                instead = self.ranked_candidate(index)[0]
                notes.append(f"{line} {money} now: take it out of the cart" + (f" and order {instead}, the listing that ranks first within the budget" if instead and instead != line else ""))
                continue
            found = self.rival(line)
            if found:
                known = " known so far (one still takes its value from a listing not found yet)" if any(r.get("pending") for r in self.reqs_for(index)) else ""
                notes.append(f"{line} costs {shown} now, and {found} ({format(self.catalog[found]['price'], 'g')}) ranks ahead of it by the request's rule and meets every requirement{known}: swap the cart line to {found}, then order")
            elif line in moved and self.plan.get("priority") in ("cheapest", "best"):
                notes.append(f"{line} costs {shown} now, within the budget, and still ranks first by the request's rule among the listings opened: keep it")
        return ("price change: " + "; ".join(notes)) if notes else ""

    def failing_cart_swap(self) -> str:
        held = self.held_lines()
        if not held or self.commit_failures >= 2 or self.several_products():
            return ""
        for line in held[-1:] if not self.multi() else held:
            assessment = self.assess(line, judge=False)
            if not assessment or assessment["state"] != "fails":
                continue
            index = self.product_of(line)
            best, state = self.best_candidate(relax=True, index=index if index is not None else 0)
            if not best or best == line:
                continue
            self.commit_wanted = True
            why = "; ".join(f"{_describe(r)}: {state_}" for r, state_, _ in assessment["fails"])
            return f"{line} fails the exact check ({why}), so I will order {best} ({state}) instead."
        return ""

    def near_misses(self, listings: dict[str, tuple[dict[str, Any], str]]) -> list[tuple[dict[str, Any], list[str]]]:
        found = []
        for item in listings:
            if self.money_problem(item) or self.verdict_kind(item) == "model":
                continue
            assessment = self.assess(item, listings, judge=False)
            if not assessment or assessment["state"] != "fails" or len(assessment["fails"]) != 1 or assessment.get("item", 0) < 0:
                continue
            req, state, _ = assessment["fails"][0]
            if req["op"] != "is" or state != "not met" or any(_is_measure(v) for v in req["values"]):
                continue
            values = [part.lower() for v in _family(req, listings[item][0]).values() for part in _elements(v)]
            if values:
                found.append((req, values))
        return found

    def judge_meaning(self, listings: dict[str, tuple[dict[str, Any], str]]) -> bool:
        asks: dict[str, tuple[dict[str, Any], list[str]]] = {}
        for req, values in self.near_misses(listings):
            bucket = asks.setdefault(_describe(req), (req, []))[1]
            for value in values:
                if (_describe(req), value) not in self.lenient and value not in bucket and len(bucket) < 20:
                    bucket.append(value)
        asks_list = [(req, values) for req, values in asks.values() if values]
        if not asks_list or self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS or self.spent_out():
            return False  # (an unanswered question would record every value as refused)
        lines = []
        for number, (req, values) in enumerate(asks_list, 1):
            lines.append(f"R{number}. property: {req['property']}; required value: {' or '.join(str(v) for v in req['values'])}")
            lines += [f"  {index}. {value}" for index, value in enumerate(values, 1)]
        result = self.judge(MEANING_PROMPT + "\n\n" + "\n".join(lines), JUDGE_TOKENS, _route("match"))
        accepted = []
        for number, (req, values) in enumerate(asks_list, 1):
            answers = (result or {}).get(f"R{number}") if isinstance(result, dict) else None
            for index, value in enumerate(values, 1):
                ok = isinstance(answers, dict) and answers.get(str(index)) is True
                self.lenient[(_describe(req), value)] = ok
                if ok:
                    accepted.append(f"{_describe(req)} ~ {value}")
        self.trace.append({"kind": "meaning", "asked": sum(len(v) for _, v in asks_list), "accepted": accepted})
        return bool(accepted)

    def best_candidate(self, relax: bool = False, index: int = 0, last_resort: bool = False) -> tuple[str | None, str]:
        """The best listing for one product: fitting first, then (relax) requirements no listing states in these words,
        then (last_resort) the closest listing of that product, since a decline almost never scores."""
        found = self.ranked_candidate(index)
        if found[0]:
            return found
        if relax and self.relax(index):
            return self.best_candidate(index=index, last_resort=last_resort)
        return self.near_fit(index) if last_resort else (None, "")

    def relax(self, index: int) -> bool:
        """When no listing of one product fits: relax its requirements that no opened listing states in these words, else let
        the model judge near-miss values; True when either may change the checks. Never when listings that meet every
        requirement exist but are over the budget or sold out: the exact values exist, so a looser reading is not wanted."""
        if self.priced_out(index):
            return False
        listings = self.listings()
        loose = self.ungrounded(listings, index)
        if loose:
            for r, recorded in loose:
                self.relaxed.setdefault(_describe(r), {})[index] = recorded
            self.clear_verdicts("exact")
            self.trace.append({"kind": "relaxed", "requirements": [_describe(r) for r, _ in loose]})
            return True
        if self.judge_meaning(listings):
            self.clear_verdicts("exact")
            return True
        return False

    def rows_stating(self, reqs: list[dict[str, Any]], index: int) -> list[str]:
        """Seen but unopened listings of a product whose row text states a value of every given "is" requirement (in stock,
        within the budget), cheapest first."""
        listings = self.listings()
        wanted = [r for r in reqs if r["op"] == "is" and r["values"]]
        if not wanted:
            return []
        rows = [i for i, e in self.catalog.items() if i not in listings and e.get("text") and self.item_of(i) == index and e.get("stock") is not False
                and not self.over_budget(i, index) and all(any(_says(e["text"], v) for v in r["values"]) for r in wanted)]
        return sorted(rows, key=lambda i: (self.catalog[i].get("price") is None, self.catalog[i].get("price") or 0, i))

    def row_conflict(self, item: str, index: int, listings: dict[str, tuple[dict[str, Any], str]]) -> bool:
        """The row text of a seen listing names, for a required value it does not state, another value that opened listings
        record for that property (not a same-meaning or related form of a required one): it is not the kind asked for."""
        text = (self.catalog.get(item) or {}).get("text") or ""
        for req in self.reqs_for(index, blocking=True):
            if req["op"] != "is" or not req["values"] or any(_is_measure(v) or _says(text, v) for v in req["values"]):
                continue  # (a measure may be written in another unit)
            judged = self.judged.get(_describe(req), {})
            if any(_says(text, p) for p in self.vocabulary(req, listings) if not judged.get(p) and not any(_states(p, v) or _related(p, v, req["property"]) for v in req["values"])):
                return True
        return False

    def title_only(self, item: str | None) -> bool:
        """A listing whose fit rests on its title, description or a loose match (no field names a required property)."""
        return bool(item) and _evidence_class(self.assess(item, judge=False)) >= 2

    def open_stating(self, chosen: dict[int, tuple[str, str]]) -> bool:
        """Before the code orders: for a product without a full fit on its fields, open (once) the cheapest seen listing whose row text states
        every one of its required values. True when it opened any."""
        details = self.reader()
        if not details:
            return False
        found = []
        for index in range(len(self.items)):
            pick, state = chosen.get(index) or (None, "")
            if state == "fits" and not self.title_only(pick):
                continue
            rows = self.rows_stating(self.reqs_for(index, blocking=True), index)
            found += rows[:1]
        calls = [c for c in (self.checked_call(details, _fill(self.schemas.get(details), i, None, {})) for i in found[:2]) if c]
        if not calls or self.steps_left() - len(calls) < self.commit_steps_needed() - 2 or self.time_left() < WRAP_UP_SECONDS / 2:
            return False
        self.stating_opened = True
        self.commit_paused = True
        sent = [{"id": f"stating_{self.step}_{n}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for n, c in enumerate(calls)]
        self.messages.append({"role": "assistant", "content": "No opened listing fits yet, so before ordering I will open listings whose descriptions state the required values.", "tool_calls": sent})
        for reply in self.run_calls(sent, planned=True):
            self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})
        self.trace.append({"kind": "commit_opened_stating", "items": [c["args"] for c in calls]})
        return True

    def revealed_conflict(self, index: int) -> dict[str, Any] | None:
        """A need the shopper revealed that nothing can meet together with the other requirements: the product was searched and
        its listings opened, none fits every requirement, some meet all the others and each of those has its own field for this
        need stating another value (a need no field records may be named differently in this store, so it is no proof), no seen
        row states the needed value, and no pointer is involved."""
        listings = self.listings()
        mine = [i for i in listings if self.item_of(i) == index and (self.catalog.get(i) or {}).get("stock") is not False and not self.over_budget(i, index)]
        blocking = self.reqs_for(index, blocking=True)
        if not mine or any(r.get("pending") or r.get("reference") or r.get("resolved_from") for r in blocking):
            return None
        if not any(self.item_of(i) == index for q in self.searches.values() for i in q["seen"]):
            return None  # never searched: the listings opened are no proof
        if any((self.assess(i, listings, judge=False) or {}).get("state") in ("fits", "unclear") for i in mine):
            return None
        for need in [r for r in blocking if r.get("source") == "revealed"]:
            rest = [r for r in blocking if r is not need]
            others = [i for i in mine if all(self.evidence(r, i, listings)[0] in ("met", "contradicted") for r in rest)]
            if others and all(self.evidence(need, i, listings)[0] == "not met" for i in others) and not self.rows_stating([need], index):
                return need
        return None

    def priced_out(self, index: int) -> bool:
        """Opened, in-stock listings of the product meet every requirement, and every one of them costs more than the budget
        (when nothing that meets the requirements fits the budget, a decline is right; a sold-out pick is not such a case:
        the next-best listing is wanted then)."""
        listings = self.listings()
        fits = [i for i in listings if self.item_of(i) == index and self.catalog.get(i, {}).get("stock") is not False
                and (self.assess(i, listings, judge=False) or {}).get("state") == "fits"]
        return bool(fits) and all(self.over_budget(i, index) for i in fits)

    def decline_proven(self, index: int) -> bool:
        """The code holds a proof that nothing of this product can be ordered: every listing meeting its requirements is over the
        budget or sold out, or a revealed need is met by none."""
        return not self.closest_open(index) and bool(self.priced_gap(index) or self.revealed_conflict(index) or self.stated_miss(index))

    def proven_miss(self, item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> bool:
        """A check proved the listing wrong: its own field states another value of a stated requirement (a missing field or an
        UNCLEAR reading proves nothing), or the model check found it wrong."""
        if self.verdict_kind(item) == "model":
            return True
        fails = (self.assess(item, listings, judge=False) or {}).get("fails") or []
        return any(state == "not met" and req.get("source") == "stated" for req, state, _ in fails)

    def stated_miss(self, index: int) -> bool:
        """The search for one product is over (no seen row stating its required values is left to open, or fewer than the two
        steps opening one and ordering it take: opening is a store call, so the spend stop does not end it) and every opened
        listing of it in stock and within the budget has a proven miss: no order of it can fit, so a decline is right."""
        listings = self.listings()
        mine = [i for i in listings if self.item_of(i) == index and (self.catalog.get(i) or {}).get("stock") is not False and not self.over_budget(i, index)]
        if not mine or not all(self.proven_miss(i, listings) for i in mine):
            return False
        return self.steps_left() < 2 or not self.rows_stating(self.reqs_for(index, blocking=True), index)

    def verified_pick(self, index: int, cheaper_than: float | None = None, held_only: bool = False) -> str | None:
        """The best opened listing of one product a check passed (the exact or model check under this ledger, or, for a line the
        cart holds, under an earlier one) that nothing now fails: in stock, within the budget, no failing verdict, no proven
        miss; cheapest first (only below cheaper_than when given; only cart lines with held_only)."""
        listings = self.listings()
        held = set(self.held_lines())
        passed = lambda i: self.verdicts.get(f"{i}|{self.ledger_version}") == "" or (i in held and (self.assess(i, listings, judge=False) or {}).get("state") != "fails"
                                                                                     and any(k.rsplit("|", 1)[0] == i and v == "" for k, v in self.verdicts.items()))
        found = [(price is None, price or 0, i) for i in listings for price in [(self.catalog.get(i) or {}).get("price")]
                 if self.item_of(i) == index and (i in held or not held_only) and (self.catalog.get(i) or {}).get("stock") is not False and not self.over_budget(i, index)
                 and not self.verdict(i) and passed(i) and not self.proven_miss(i, listings) and (cheaper_than is None or (price is not None and price < cheaper_than))]
        return min(found)[-1] if found else None

    def best_effort_fill(self, missing: list[int], basket: dict[int, str]) -> dict[int, str]:
        """For products a basket lacks while no decline is proven for them: the best listing known for each - an opened near fit,
        else the cheapest in-stock row seen of that product that keeps the order within the budget. {} when any product has none
        (then a decline is the only finish left)."""
        out: dict[int, str] = {}
        for index in missing:
            if self.decline_proven(index):
                return {}
            ranked = [i for i, _ in self.ranked_pool(index) if i not in basket.values()]
            item = ranked[0] if ranked else (self.closest(index)[0] if self.closest_open(index) else None)  # an opened listing that does not fail its checks
            if not item:
                taken = {**basket, **out}
                room = (self.budget_value() - self.basket_total(taken)) if self.budget_now is not None else None
                opened = self.listings()
                mapped = lambda i: self.item_map.get(i) == index or self.label_product(i) == index  # evidence it is this product, not a guess
                rows = [(e.get("price"), i) for i, e in self.catalog.items() if mapped(i) and i not in opened and e.get("stock") is not False and isinstance(e.get("price"), (int, float))
                        and not self.verdict(i) and i not in taken.values() and not self.over_budget(i, index) and (room is None or e["price"] * self.qty(index) <= room)]
                item = min(rows)[1] if rows else None
            if not item:
                return {}
            out[index] = item
        if out:
            self.trace.append({"kind": "best_effort", "lines": out})
        return out

    def priced_gap(self, index: int) -> bool:
        """The product has no listing an order can carry because of the budget or stock: priced out, and no other opened
        listing of it may fit (its closest listings are not wanted, since the exact values exist)."""
        return self.priced_out(index) and not self.ranked_candidate(index)[0] and not self.unproven_listing(index)

    def unproven_listing(self, index: int) -> str | None:
        """An opened listing of the product within the budget and in stock that nothing proves wrong, though a check could not
        confirm it (a field left unread, a check the money stop skipped): what is unknown is no proof that only dearer listings fit."""
        listings = self.listings()
        return next((i for i in listings if self.item_of(i) == index and (self.catalog.get(i) or {}).get("stock") is not False and not self.over_budget(i, index) and not self.proven_miss(i, listings)), None)

    def priced_rows(self, index: int) -> list[str]:
        """Seen but unopened, in-stock listings of a product over the budget whose row text states a value of every required
        "is" requirement, cheapest first: opening one that fits proves the product priced out."""
        listings = self.listings()
        wanted = [r for r in self.reqs_for(index, blocking=True) if r["op"] == "is" and r["values"]]
        rows = [i for i, e in self.catalog.items() if wanted and i not in listings and e.get("text") and self.item_of(i) == index and e.get("stock") is not False
                and self.over_budget(i, index) and all(any(_says(e["text"], v) for v in r["values"]) for r in wanted)]
        return sorted(rows, key=lambda i: (self.catalog[i].get("price") or 0, i))

    def basket_fault(self, lines: list[tuple[str, int]]) -> str:
        """Why an order of these lines (listing, units) can never fit, or "": an order needs a line for every product (fewer
        lines than products never fit, whatever the lines are) and the lines together within the budget. A decline is then
        the only finish that can be right."""
        items = list(dict.fromkeys(item for item, _ in lines))
        if len(items) < len(self.items):
            return f"only {len(items)} of the {len(self.items)} products have a listing an order can carry, and an order needs one line for each"
        total = sum((self.catalog.get(item, {}).get("price") or 0) * units for item, units in lines)
        if total > self.budget_value():
            return f"the best listings I found cost {format(total, 'g')} together, over the budget of {format(self.budget_value(), 'g')}"
        return ""

    def order_lines(self) -> list[str]:
        """The lines an order of the cart now carries: the latest cart view's when it is fresh, else the cart as followed."""
        return list(dict.fromkeys(i for i, n in self.view_lines.items() if n != 0)) if self.view_version == self.cart_version else list(dict.fromkeys(self.held_lines()))

    def short_cart(self) -> bool:
        """The cart holds lines but fewer than the products asked for, and the store takes a decline: no order of it can fit."""
        return bool(self.roles.get("decline")) and 0 < len(self.order_lines()) < len(self.items)

    def unopened_products(self) -> list[tuple[int, list[str]]]:
        """Products none of whose listings has been opened although searches showed some: (product, up to three to open)."""
        listings = self.listings()
        out = []
        for index in range(len(self.items)):
            if any(self.item_of(i) == index for i in listings):
                continue
            pool = self.pool(index)
            rows = [i for i, e in self.catalog.items() if i not in listings and self.item_of(i) == index and e.get("stock") is not False and not self.over_budget(i, index)]
            rows.sort(key=lambda i: (i not in pool, self.catalog[i].get("price") is None, self.catalog[i].get("price") or 0, i))
            if rows:
                out.append((index, rows[:3]))
        return out

    def ranked_candidate(self, index: int) -> tuple[str | None, str]:
        """The best opened listing of one product that is in stock, within the budget, unrejected and not failing its checks."""
        ranked = self.ranked_pool(index)
        return ranked[0] if ranked else (None, "")

    def ranked_pool(self, index: int) -> list[tuple[str, str]]:
        """Every such listing of one product, best first (by the request's rule), with its state."""
        listings = self.listings()
        pool = []
        for item in listings:
            if self.item_of(item) != index:
                continue
            if self.catalog.get(item, {}).get("stock") is False or self.over_budget(item, index) or self.verdict(item):
                continue
            assessment = self.assess(item, listings, judge=False)
            if not assessment or assessment["state"] == "fails":
                continue
            price = self.catalog.get(item, {}).get("price")
            pool.append((item, _candidate_state(assessment), price, self.measure(item, listings), assessment["conflict"], assessment.get("weak", 0), assessment.get("level", 0), assessment.get("soft", 0), _evidence_class(assessment)))
        if not pool:
            return []
        grade = {"fits": 0, "may fit": 1, "unclear": 2}
        held = set(self.held_lines())
        priority = self.plan.get("priority")
        # among relaxed candidates, one whose field is merely absent ranks above one whose field states another value (c[4]);
        # among equal fits, evidence from fields ranks above evidence from the title or description (c[5])
        if priority == "cheapest":
            key = lambda c: (c[1] == "unclear", c[4], c[8], c[2] is None, c[2] or 0, grade[c[1]], c[5], c[0] not in held)  # (a cheaper listing wins only against evidence no stronger than its own)
        elif priority == "best" and self.best:
            sign = 1 if self.best["direction"] == "least" else -1
            key = lambda c: (c[1] == "unclear", c[4], c[8], c[3] is None, (c[3] or 0) * sign, grade[c[1]], c[5], c[2] or 0)
        else:  # (c[6], c[7]: the weakest evidence grade, and how many requirements rest on less than a field named for them)
            stays = self.order_attempts > 0  # (after a first close the same cart passes 185 of 397, a changed one 66 of 194)
            key = lambda c: (c[4], grade[c[1]], c[6], stays and c[0] not in held, c[2] is None, c[2] or 0, c[7], c[5], c[0] not in held)  # (the lowest price among equal fits: at least 9 of 90 race-10 tasks grade it without asking for it)
        return [c[:2] for c in sorted(pool, key=key)]

    def near_fit(self, index: int) -> tuple[str | None, str]:
        """The closest listing of one product when none fits: fewest failed checks (a missing field before a contradicting
        one), in stock and within the budget, cheapest first; by the request's own rule when it asks for the closest (closest),
        and none when it rules out a substitute."""
        if self.fallback.get("closest") == "forbidden":
            return None, ""
        if self.fallback.get("closest") == "wanted":
            item = self.closest(index)[0]
            return (item, "near fit") if item else (None, "")
        listings = self.listings()
        pointed = self.reference_listings()
        ranked = []
        for item in listings:
            if self.item_of(item) != index or item in pointed:
                continue  # (a listing a pointer led to gave its values; it is not what was asked for)
            entry = self.catalog.get(item, {})
            price = entry.get("price")
            if entry.get("stock") is False or self.over_budget(item, index):
                continue
            if self.verdict_kind(item) == "model":
                continue  # the model check found it wrong in itself (such as another kind of product)
            if self.proven_miss(item, listings):
                continue  # its own field states another value of a stated requirement: never ordered as the closest
            assessment = self.assess(item, listings, judge=False)
            fails = (assessment or {}).get("fails") or []
            stated = any(state == "not met" and req.get("source") == "stated" for req, state, _ in fails)
            ranked.append((stated, len(fails), sum(1 for _, state, _ in fails if state == "not met"), price is None, price or 0, item))
        if not ranked:
            return None, ""
        return min(ranked)[-1], "near fit"

    def closest(self, index: int) -> tuple[str | None, str | None]:
        """The request asks for the closest listing when nothing fits: (the listing to order, the listing to name in a decline).
        Opened listings rank by how many requirements each misses, sold out and over the budget counting as one each when the
        request counts them (else behind every listing that only misses requirements); among the fewest, those within the
        request's margin of the cheapest stand level, and the one with the fewest proven misses goes, a cart line first, then
        the cheapest. A sold-out listing, or one costing more than the whole budget, can be named but not ordered (the store's
        grader takes no order over the budget)."""
        listings = self.listings()
        counts = self.fallback.get("counts") or []
        held = set(self.held_lines())
        pointed = self.reference_listings()
        rows = []
        for item in listings:
            if self.item_of(item) != index or item in pointed:
                continue
            if self.verdict_kind(item) == "model":
                continue  # the model check found it wrong in itself (such as another kind of product)
            entry = self.catalog.get(item) or {}
            fails = (self.assess(item, listings, judge=False) or {}).get("fails") or []
            sold, over = entry.get("stock") is False, self.over_budget(item, index) and not any(r.get("field") for r, _, _ in fails)  # (a price limit it misses is counted among its fails)
            price = entry.get("price")
            misses = (len(fails) or int(bool(self.verdict(item)))) + int(sold and "stock" in counts) + int(over and "budget" in counts)
            rows.append({"item": item, "rank": (int(sold and "stock" not in counts) + int(over and "budget" not in counts), misses), "price": price, "sold": sold,
                         "whole": price is not None and price * self.qty(index) > self.budget_value(),
                         "proven": sum(1 for _, state, _ in fails if state == "not met")})

        def first(pool: list[dict[str, Any]]) -> str | None:
            level = [r for r in pool if r["rank"] == min(x["rank"] for x in pool)]
            priced = [r["price"] for r in level if r["price"] is not None]
            top = min(priced) * (1 + (self.fallback.get("margin") or 0)) if priced else 0
            within = [r for r in level if r["price"] is not None and r["price"] <= top] or level
            return min(within, key=lambda r: (r["proven"], r["item"] not in held, r["price"] is None, r["price"] or 0, r["item"]))["item"]
        orderable = [r for r in rows if not r["sold"] and not r["whole"]]
        return (first(orderable) if orderable else None), (first(rows) if rows else None)

    def closest_open(self, index: int) -> bool:
        """The request asks for the closest listing and this product has one that can be ordered: no proof makes a decline right."""
        return self.fallback.get("closest") == "wanted" and bool(self.closest(index)[0])

    def closest_named(self) -> str | None:
        """The listing a decline names when the request asks for the closest: that of the first product none can be ordered of."""
        found = [self.closest(index) for index in range(len(self.items))]
        return next((named for item, named in found if named and not item), None) or next((named for _, named in found if named), None)

    def picks(self, relax: bool = True, last_resort: bool = True) -> dict[int, tuple[str, str]]:
        """The chosen listing (and its state) for every product that has one."""
        found = {}
        for index in range(len(self.items)):
            item, state = self.best_candidate(relax=relax, index=index, last_resort=last_resort)
            if item:
                found[index] = (item, state)
        return found

    def ready_choice(self) -> str | None:
        chosen: dict[int, tuple[str, str]] = {}
        for index in range(len(self.items)):
            if any(r.get("pending") for r in self.reqs_for(index)):
                return None  # a requirement still points at a listing not found yet: its value can change the choice
            item, state = self.best_candidate(index=index)
            if not item:
                return None
            if self.plan.get("priority") in ("cheapest", "best"):
                info = self.board(index)
                if not (info and info.get("complete") and info.get("top") == item):
                    return None
            elif state != "fits":
                return None
            chosen[index] = (item, state)
        if not self.multi():
            return chosen[0][0]
        chosen = self.within_budget(chosen, quiet=True)  # the budget is the whole order's total
        if self.basket_total({index: item for index, (item, _) in chosen.items()}) > self.budget_value():
            return None
        return " and ".join(f"{item} for product {index} ({self.product(index)})" for index, (item, _) in sorted(chosen.items()))

    def decline_rows(self) -> tuple[int, list[str], bool, list[str]] | None:
        """Before a decline: (product, rows, stating, over) for a product without a candidate whose seen but unopened rows (in
        stock, within the budget) state its required values, or, with none stating them, the rows its searches for those
        values listed, cheapest first; at most three, only as many as leave the code's endgame its
        steps, and once per product. While a product has no listing an order can carry, only such products are asked about
        (whatever the others show, no basket is whole without it); a product whose only candidates fail their checks also
        gets its cheapest row over the budget that states the values (over), whose fit would prove it priced out."""
        listings = self.listings()
        room = min(3, self.steps_left() - self.th('commit') - 1)
        gaps = [index for index in range(len(self.items)) if not self.best_candidate(index=index, last_resort=True)[0]]
        for index in (gaps or range(len(self.items))) if room > 0 else []:
            if index in self.decline_rows_done or self.best_candidate(index=index)[0]:
                continue  # held once already, or a candidate exists (the checks below name it)
            pool = self.pool(index)  # (a row not yet mapped to a product is this one's when a search for its values showed it)
            rank: dict[str, tuple[int, int]] = {}  # where the searches for the product's required values listed a row: earliest search, top row first
            for number, q in enumerate(self.searches.values()):
                if self.covering(q, index) or self.refines(q, index):
                    for position, i in enumerate(q["seen"]):
                        rank.setdefault(i, (number, position))
            sought = {i for q in self.searches.values() if self.refines(q, index) for i in q["seen"]}
            seen = sorted((i for i, e in self.catalog.items() if i not in listings and (self.item_of(i) == index or (self.item_of(i) is None and i in pool))
                           and e.get("stock") is not False and not self.over_budget(i, index)),
                          key=lambda i: (i not in sought, i not in rank, self.catalog[i].get("price") is None, self.catalog[i].get("price") or 0, rank.get(i, (0, 0)), i))
            wanted = [r for r in self.reqs_for(index, blocking=True) if r["op"] == "is" and r["values"]]
            stating = [i for i in seen if wanted and self.catalog[i].get("text") and all(any(_says(self.catalog[i]["text"], v) for v in r["values"]) for r in wanted)]
            over = [] if stating or index in gaps or self.priced_out(index) else self.priced_rows(index)[:1]
            rows = stating[:room] or seen[: room - len(over)] + over
            if rows:
                return index, rows, bool(stating), over
        return None

    def decline_blocker(self) -> str:
        """Why a decline is held back, or "": first the evidence it rests on (a product none of whose seen listings was
        opened, seen rows that may fit), then the basket: a decline is held back only while every product has a listing an
        order can carry, within the budget together (an order missing a product never fits, so then a decline is right)."""
        unopened = self.unopened_products() if self.steps_left() > self.th('commit') else []
        if unopened:
            index, rows = unopened[0]
            return f"no listing of {self.product(index) or 'the requested product'} has been opened, so nothing shows that none fits: open the details of {', '.join(rows)} first."
        seen = self.decline_rows()
        if seen:
            index, rows, stating, over = seen
            self.decline_rows_done.add(index)
            self.trace.append({"kind": "decline_rows", "product": index, "rows": rows, "stating": stating, "over": over})
            what = "their rows state the required values" if stating else "your searches for the required values listed them first"
            proof = f" {over[0]}'s row states the required values at a price over the budget: if its details meet every requirement, the listings that do all cost more than the budget, and the decline is right." if over else ""
            if rows == over:
                return f"a listing of {self.product(index) or 'the requested product'} that your searches showed has not been opened: open the details of {over[0]} first.{proof}"
            return f"listings of {self.product(index) or 'the requested product'} that your searches showed have not been opened, and {what}: open the details of {', '.join(rows)} first, and decline only if none of them fits.{proof}"
        if self.decide_finish()["action"] == "decline":
            return ""  # every listing that meets all of a product's requirements is over the budget or sold out: a decline is right
        chosen: dict[int, tuple[str, str]] = {}
        for index in range(len(self.items)):
            item, state = self.best_candidate(index=index)
            if not item:
                info = self.board(index)
                if info is not None and not info["complete"] and self.steps_left() > self.th('commit') + 2 and not self.decline_hinted:
                    self.decline_hinted = True
                    return "no search has proven that nothing fits yet. " + (self.board_note() or "Search the requested category with the required values' words") + ". Search that way first."
                item, state = self.best_candidate(relax=True, index=index, last_resort=True)
            for _ in range(3):
                if not item or state == "near fit" or not self.verify_item(item, "ORDER"):
                    break
                item, state = self.best_candidate(relax=True, index=index, last_resort=True)
            if not item:
                return ""  # a product without any listing an order can carry: no basket is whole, a decline is right
            chosen[index] = (item, state)
        if self.decide_finish(lines=[(item, self.qty(index)) for index, (item, _) in self.within_budget(chosen, quiet=True).items()], sent=False)["action"] == "decline":
            return ""  # even the cheapest fitting listings cost more than the budget together
        if self.commit_failures < 2 or self.decline_blocks < 4:  # a failed code commit does not make a decline right
            self.decline_blocks += 1
            why = {"fits": "fits every requirement", "may fit": "may fit (its evidence is contradicted, yet such listings are usually accepted)", "unclear": "may still fit (no check proves its UNCLEAR requirement fails)", "near fit": "is the closest listing (a decline almost never scores, an order of the closest listing sometimes does)"}
            if not self.multi():
                item, state = chosen[0]
                if self.decline_blocks >= 2 and self.commit_failures < 2:
                    self.commit_wanted = True
                    return f"{item} {why[state]}, and a decline is right only when every candidate is proven wrong, so I will order {item} now."
                return f"{item} {why[state]}, and a decline is right only when every candidate is proven wrong. Put it in the cart (quantity {self.quantity}) and send {self.how('order')}."
            named = "; ".join(f"{item} for product {index} {why[state]}" for index, (item, state) in sorted(chosen.items()))
            if self.decline_blocks >= 2 and self.commit_failures < 2:
                self.commit_wanted = True
                return f"{named}, and a decline is right only when every candidate is proven wrong, so I will order these now."
            amounts = ", ".join(f"{item} x{self.qty(index)}" for index, (item, _) in sorted(chosen.items()))
            return f"{named}, and a decline is right only when every candidate is proven wrong. Put {amounts} in the cart (one line per product) and send {self.how('order')}."
        return ""

    def checked_call(self, name: str | None, args: Any) -> dict[str, Any] | None:
        if not name or name not in self.schemas or not isinstance(args, dict):
            return None
        args, problems = _check(args, self.schemas[name], "args")
        return None if problems else {"name": name, "args": args}

    def format_call(self, format_: dict[str, Any] | None, lines: list[tuple[str, int]], text: str = "") -> dict[str, Any] | None:
        """A call from a call format; a format without a count slot of its own gets each line's count written in."""
        if not format_:
            return None
        args = _apply(format_["args"], lines, text)
        if not _holds(format_["args"], _QTY):
            args = _with_counts(args, self.schemas.get(format_["tool"]), lines)
        return self.checked_call(format_["tool"], args)

    def basket_tool(self) -> bool:
        """The cart tool takes the cart as a list of lines with nothing else to choose, so one call sets the whole basket."""
        schema = self.schemas.get(self.roles.get("cart") or "")
        slot = _lines_slot(schema)
        props = _branch(schema).get("properties") or {}
        return slot is not None and not any("enum" in _branch(spec) for key, spec in props.items() if key != slot[0])

    def cart_tools(self) -> set[str]:
        names = {self.roles.get("cart")} | {f["tool"] for action, f in self.formats.items() if action in ("add", "remove", "basket")}
        return {n for n in names if n}

    def model_format(self, action: str) -> dict[str, Any] | None:
        """A call format written by the model from the tool schemas, used only when neither the schema nor a learned call is enough."""
        if action in self.formats:
            return self.formats[action]
        if action in self.format_asked or self.time_left() < WRAP_UP_SECONDS:
            return None
        self.format_asked.add(action)
        tools = "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')} | arguments: {_compact(t['function'].get('parameters') or {})}" for t in self.tools)
        result = self.judge(FORMAT_PROMPT.replace("ACTION", FORMAT_ACTIONS[action]) + "\n\nTOOLS:\n" + tools, JUDGE_TOKENS, _route("read")) or {}
        name, args = result.get("tool"), result.get("args")
        format_ = _template(args) if isinstance(args, dict) else None
        sample = next(iter(self.catalog), "p-0a1b2c3d4e5f")
        probe = self.checked_call(name, _apply(format_, [(sample, 1)], "Thank you.")) if isinstance(name, str) and format_ is not None else None
        self.trace.append({"kind": "format", "action": action, "tool": name, "args": args, "valid": probe is not None})
        others = {self.roles.get(r) for r in ("search", "details", "view", "message", "cart_view", "order", "decline")} - {None}
        misplaced = (action in ("add", "remove", "basket") and name in others) or (action == "message" and not _holds(format_, _TEXT))
        if probe is None or misplaced or (action not in ("message", "order", "decline") and not _holds(format_, _ITEM)):
            return None
        self.formats[action] = {"tool": name, "args": format_}
        return self.formats[action]

    def cart_call(self, item: str, mode: str, count: int | None = None) -> dict[str, Any] | None:
        """One call that adds (mode "add") or takes out ("remove") one listing: built from the schema as e_w8 did, else from a call
        the navigator made work, else from a format the model writes."""
        name = self.roles.get("cart")
        own = self.qty(self.item_of(item))  # the quantity of the product this listing is
        schema = self.schemas.get(name or "") or {}
        count = count if count is not None else (own if mode == "add" else (self.held.get(item) or own))
        chosen = self.cart_modes.get(mode) or {}
        if not chosen and mode == "add" and self.add_signature:
            chosen = _parse_json_object(self.add_signature) or {}
        if name and (mode == "add" or chosen):  # taking out needs a known remove value: without one the same call would add
            slot = _lines_slot(schema)
            if slot:  # cart lines in a list next to a mode argument: a list of one line
                line = _fill(slot[1], item, count, {})
                args = {**{k: v for k, v in chosen.items() if k != slot[0]}, slot[0]: [line]} if line is not None else None
                if args is not None and any(k not in args for k in _branch(schema).get("required") or []):
                    args = None
            else:
                args = _fill(schema, item, count, chosen)
            call = self.checked_call(name, args)
            if call:
                return call
        call = self.from_format(self.formats.get(mode), item, count, mode)
        if call or (mode == "remove" and self.mode_less_cart() and self.cart_counts == "add"):
            return call
        call = self.from_format(self.model_format(mode), item, count, mode)
        if call is None and mode == "remove" and self.mode_less_cart() and self.cart_counts != "add" and not self.spare_id_tools():
            call = self.checked_call(name, _fill(schema, item, 0, {}))  # no other tool could take it out: the cart call with 0
        return call

    def from_format(self, format_: dict[str, Any] | None, item: str, count: int, mode: str) -> dict[str, Any] | None:
        """A cart call from a call format. Taking a line out with the cart call itself (no choice argument) needs a count of 0:
        the held count would set the line again or add to it; when the store has shown that its cart calls only add, it cannot."""
        if not format_:
            return None
        if mode == "remove" and format_["tool"] == self.roles.get("cart") and self.mode_less_cart():
            if self.cart_counts == "add":
                return None
            return self.checked_call(format_["tool"], _fill(self.schemas.get(format_["tool"]), item, 0, {}))
        return self.format_call(format_, [(item, count)])

    def spare_id_tools(self) -> list[str]:
        """Tools no role names that take a free-text argument (one of them could be a separate tool that takes items out)."""
        named = {v for v in self.roles.values() if v}
        return [t["function"]["name"] for t in self.tools if t["function"]["name"] not in named
                and any("string" in _kinds(s) and "enum" not in _branch(s) for s in ((t["function"].get("parameters") or {}).get("properties") or {}).values())]

    def basket_call(self, lines: list[tuple[str, int]]) -> dict[str, Any] | None:
        """One call that makes the cart hold exactly these lines (for a cart tool that takes the whole basket)."""
        name = self.roles.get("cart")
        schema = self.schemas.get(name or "") or {}
        slot = _lines_slot(schema)
        if slot:
            key, line = slot
            filled = [_fill(line, item, count, {}) for item, count in lines]
            args = dict(self.cart_modes.get("add") or {})
            if all(entry is not None for entry in filled):
                args[key] = filled
                if all(k in args for k in _branch(schema).get("required") or []):
                    call = self.checked_call(name, args)
                    if call:
                        return call
        learned = self.formats.get("basket")
        if learned and _holds(learned["args"], _LINES):
            call = self.format_call(learned, lines)
            if call:
                return call
        format_ = self.model_format("basket")
        return self.format_call(format_, lines) if format_ and _holds(format_["args"], _LINES) else None

    def cart_plan(self, picks: Any, removals: bool = True, fix_counts: bool = True) -> list[dict[str, Any]] | None:
        """Calls that leave exactly the chosen listings in the cart, each at its product's quantity (a listing id, or
        {product position: listing}); with removals=False other lines stay, with fix_counts=False a chosen line already in
        the cart keeps its count. None when a needed call cannot be built."""
        chosen = {0: picks} if isinstance(picks, str) else dict(picks)
        if not self.roles.get("cart"):
            return []
        wanted = [(item, self.qty(index)) for index, item in sorted(chosen.items())]
        in_cart = self.held_lines()
        if self.basket_tool():
            if sorted(in_cart) == sorted(item for item, _ in wanted) and all(self.held.get(item) in (None, count) for item, count in wanted):
                return []
            call = self.basket_call(wanted)
            return [call] if call else None
        steps: list[dict[str, Any]] = []
        keep = {item for item, _ in wanted}
        for other in [i for i in in_cart if i not in keep] if removals else []:
            removal = self.cart_call(other, "remove")
            if removal is None:
                return None
            steps.append(removal)
        for item, count in wanted:
            if item not in in_cart:
                addition = self.cart_call(item, "add", count)
                if addition is None:
                    return None
                steps.append(addition)
            elif removals and fix_counts and self.held.get(item) not in (None, count):
                # held at another count: take the line out and add the order's quantity (right whether adding raises a
                # line's count or sets it); with no way to take it out, add the missing units (adding raises the count)
                if self.cart_counts == "set":  # the store sets a line's count: one call puts it right
                    addition = self.cart_call(item, "add", count)
                    if addition is not None:
                        steps.append(addition)
                        continue
                held = self.held.get(item) or 0
                removal = self.cart_call(item, "remove")
                addition = self.cart_call(item, "add", count) if removal is not None else (self.cart_call(item, "add", count - held) if held < count else None)
                if removal is not None and addition is not None:
                    steps += [removal, addition]
                elif addition is not None:
                    steps.append(addition)
        return steps

    def budget_value(self) -> float:
        """The budget for the whole order (infinite when none is known)."""
        budget = self.budget_now
        return float(budget) if isinstance(budget, (int, float)) and not isinstance(budget, bool) else float("inf")

    def over_budget(self, item: str, index: int | None) -> bool:
        """The listing alone, at its product's quantity, costs more than the budget for the whole order, or more than a price
        limit on its product allows."""
        price = self.catalog.get(item, {}).get("price")
        return price is not None and (price * self.qty(index) > self.budget_value() or any(r.get("field") and not _within(r, price) for r in self.reqs_for(index)))

    def basket_total(self, basket: dict[int, str]) -> float:
        """What the chosen lines cost together, each at its product's quantity (a line of unknown price counts as 0)."""
        return sum((self.catalog.get(item, {}).get("price") or 0) * self.qty(index) for index, item in basket.items())

    def cheapest_fit(self, index: int) -> tuple[str | None, str]:
        listings = self.listings()
        found = []
        for item in listings:
            entry = self.catalog.get(item, {})
            if self.item_of(item) != index or entry.get("stock") is False or entry.get("price") is None:
                continue
            if self.verdict(item):
                continue  # a check already found it wrong
            assessment = self.assess(item, listings, judge=False)
            if assessment and assessment["state"] != "fails":
                found.append((entry["price"], item, assessment["state"]))
        return (min(found)[1], min(found)[2]) if found else (None, "")

    def within_budget(self, chosen: dict[int, tuple[str, str]], quiet: bool = False) -> dict[int, tuple[str, str]]:
        """When the chosen lines together cost more than the budget, take each product's cheapest fitting listing instead."""
        total = self.basket_total({index: item for index, (item, _) in chosen.items()})
        if total <= self.budget_value():
            return chosen
        cheaper = dict(chosen)
        for index in chosen:
            item, state = self.cheapest_fit(index)
            if item:
                cheaper[index] = (item, state)
        if not quiet:
            self.trace.append({"kind": "basket_over_budget", "total": total, "budget": self.budget_now})
        return cheaper

    def prepare_formats(self) -> None:
        """Ask the model once, early, for calls the schema cannot give and the endgame may need when little time is left: taking a
        listing out when the cart tool has no way to say so (likely a separate tool), and a finishing tool's required choice that
        setup left open."""
        cart = self.roles.get("cart")
        props = _branch(self.schemas.get(cart or "")).get("properties") or {}
        if cart and not self.basket_tool() and not self.cart_modes.get("remove") and not any("enum" in _branch(s) for s in props.values()):
            self.model_format("remove")
        for kind in ("order", "decline"):
            name = self.roles.get(kind)
            mode = (self.order_args if kind == "order" else self.decline_args) or {}
            schema = _branch(self.schemas.get(name or ""))
            fields = schema.get("properties") or {}
            if name and any(k not in mode and "enum" in _branch(fields.get(k)) for k in schema.get("required") or []):
                self.model_format(kind)

    def order_takes_item(self) -> bool:
        """No cart tool: the order call itself names the listing."""
        order = self.roles.get("order")
        if not order or self.roles.get("cart"):
            return False
        schema = _branch(self.schemas.get(order))
        props = schema.get("properties") or {}
        return any(k in props and "string" in _kinds(props[k]) and "enum" not in _branch(props[k]) for k in schema.get("required") or [])

    def finish_args(self, kind: str) -> dict[str, Any]:
        """Arguments that make the order (or decline) tool finish that way; a required choice is never guessed when the model can name it."""
        name = self.roles.get(kind) or ""
        mode = dict((self.order_args if kind == "order" else self.decline_args) or {})
        schema = _branch(self.schemas.get(name))
        props = schema.get("properties") or {}
        missing = [k for k in schema.get("required") or [] if k not in mode and "enum" in _branch(props.get(k))]
        if missing:
            format_ = self.model_format(kind)
            if format_ and format_["tool"] == name:
                mode.update({k: v for k, v in format_["args"].items() if k in missing and v in (_branch(props.get(k)).get("enum") or [])})
        return self.neutral_args(name, "Order the cart." if kind == "order" else DECLINE_NOTE, mode)

    def order_call(self, item: str | None) -> dict[str, Any] | None:
        name = self.roles.get("order")
        if not name:
            return None
        if not self.order_takes_item():
            return {"name": name, "args": self.finish_args("order")}
        if not item:
            return None
        count = self.qty(self.item_of(item))
        call = self.checked_call(name, _fill(self.schemas.get(name), item, count, dict(self.order_args or {})))
        if call:
            return call
        format_ = self.formats.get("order_item") or self.model_format("order_item")
        return self.format_call(format_, [(item, count)], "Order the listing.") if format_ and format_["tool"] == name else None

    def commit_help(self, picks: Any) -> None:
        """Tell the navigator once when the code could not build the calls that put its best candidates in the cart."""
        chosen = {0: picks} if isinstance(picks, str) else dict(picks)
        lines = ", ".join(f"{self.qty(index)} unit(s) of {item}" for index, item in sorted(chosen.items()))
        if self.helped == lines:
            return
        self.helped = lines
        item = lines
        cart = self.roles.get("cart")
        how = f"call {cart} so the cart holds exactly {lines} and nothing else, then send {self.how('order')}" if cart else f"send {self.how('order')} for {item}"
        self.messages.append({"role": "user", "content": f"[note] Steps are running out and I could not build the calls for {item} myself: {how}.\n" + self.status()})

    def deadline_due(self) -> bool:
        if self.finished or self.dead or not self.roles.get("order"):
            return False
        if self.commit_wanted:  # (asked for by a check: also after failed commits, once per request)
            self.commit_wanted = False
            return True
        if self.commit_failures >= 2:
            return False
        if self.order_attempts == 0:
            due = self.steps_left() <= self.commit_steps_needed() or self.time_left() < WRAP_UP_SECONDS or (self.spent_out() and self.questions_done())
            soft = self.cap_unstated and self.max_steps > SOFT_STEPS and self.step >= SOFT_STEPS - self.commit_steps_needed()
            if not due and self.commit_failures == 0 and (self.steps_left() <= self.th('ready_commit') or soft) and self.questions_done() and not self.judgement_left() \
                    and self.ready_blocked != (self.basket_key(), self.ledger_version) and self.safe_ready():
                self.trace.append({"kind": "ready_commit", "steps_left": self.steps_left()})
                self.ready_commit = True
                return True  # every product has its verified choice: order now, so a change shown at the order can still be repaired
            if due and self.multi() and self.steps_left() > self.th('cart_only') and self.time_left() > WRAP_UP_SECONDS / 2:
                missing = [index for index in range(len(self.items)) if not self.best_candidate(index=index, last_resort=True)[0]]  # a question only: nothing relaxed
                if missing:  # a product without any listing yet: give the navigator its last steps to find one
                    if self.helped != f"missing {missing}":
                        self.helped = f"missing {missing}"
                        self.messages.append({"role": "user", "content": "[note] Steps are running out and no listing has been opened yet for " + ", ".join(f"product {i} ({self.product(i)})" for i in missing) + f": search its shelf and open one fitting listing now; the order is placed for you when {self.th('cart_only')} steps remain.\n" + self.status()})
                    return False
            return due
        return self.refusals > 0 and (self.steps_left() <= self.th('cart_only') or (self.spent_out() and self.questions_done()))  # (the navigator no longer runs at the stop)

    def commit_reason(self) -> str:
        if self.ready_commit:
            return "Every product has its verified choice and steps remain to repair a refused order"
        if self.completing:
            return "The cart has no line for every product"
        return "Steps are running out"

    def judgement_left(self) -> bool:
        """The choice among fitting listings still rests on the navigator's judgement: a wish, or a requirement no listing
        field can show (the exact checks call every listing that meets the rest a fit)."""
        return bool(self.plan.get("soft_wants")) or any(not r["blocking"] and not r.get("pending") for r in self.reqs)

    def questions_done(self) -> bool:
        """No planned question to the shopper is still to go out."""
        return not self.planner_on or not self.roles.get("message") or self.messages_used >= self.message_limit() or self.key_out()

    def late_room(self, floor: bool = True) -> bool:
        """A call the order's process still needs may be made (a planned question and the reading of its answer, a pointer
        judgement): never with the key spent; only while it fits beside the core reserve and, before the stop while a product
        has no opened listing, the exploration floor still owed (the setup's cost cannot take the floor from the search: the
        questions give way first; once every product has an opened listing, the planned questions come first). floor=False:
        the same without the floor."""
        owed = self.floor_owed() if floor and self.unsearched_products() else 0.0
        return not self.key_out() and self.spent + QUESTION_COST <= self.cap - self.core_reserve() - owed

    def commit_steps_needed(self, least: int | None = None) -> int:
        """Steps the endgame keeps for itself (at least least): the cart calls a whole basket needs, the order, and spare steps
        for a refusal (the stock re-reads before the order use the spare steps only when they fit)."""
        least = self.th('commit') if least is None else least
        lines = len(self.items)
        calls = 1 if self.basket_tool() or not self.roles.get("cart") else lines + len(self.held_lines())
        return max(least, calls + self.th('last') + 1 + self.reserved_steps())  # (the cart calls, the order and spare steps for a refusal)

    def commit_now(self) -> bool:
        self.committing = True
        self.commit_paused = False
        try:
            acted, ordered = self.commit_steps()
        finally:
            self.committing = False
            self.ready_commit = self.completing = False
        if not ordered and not self.commit_paused:  # a pause (cheaper listings opened, a line sold out) is not a failure
            self.commit_failures += 1
        return acted

    def code_decline(self, opening: str) -> tuple[bool, bool]:
        """The code's own decline (it passes the decline checks, as the code's own orders pass the order checks)."""
        decline = self.roles.get("decline")
        if self.refused_out("decline"):
            return False, False
        args = self.finish_args("decline")
        call = {"id": f"decline_{self.step}", "type": "function", "function": {"name": decline, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": opening, "tool_calls": [call]})
        self.code_declining = True
        try:
            replies = self.close_calls([call])
        finally:
            self.code_declining = False
        for reply in replies:
            self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})
        return True, False

    def close_calls(self, calls: list[dict[str, Any]]) -> list[Reply]:
        """Send the code's own close: past the stop the reserve pays for reading what the store brings back."""
        return self.in_phase("reserve", lambda: self.run_calls(calls, planned=True)) if self.spent_out() else self.run_calls(calls, planned=True)

    def close_key(self, kind: str, lines: list[str] | None = None) -> tuple:
        """A close as the store saw it: how it ends, the lines at their prices now, the version of the task."""
        lines = self.order_lines() if lines is None else lines
        return kind, tuple(sorted((i, (self.catalog.get(i) or {}).get("price")) for i in lines)) if kind == "order" else (), self.ledger_version

    def refused_out(self, kind: str, lines: list[str] | None = None) -> bool:
        """The store did not take this same close CLOSE_RESENDS times, with nothing changed since: it is not sent again."""
        return self.close_refused.get(self.close_key(kind, lines), 0) >= CLOSE_RESENDS

    def reopen_close(self, lines: list[str], version: int, key: tuple) -> None:
        """A refused finish starts another decision round on every path. What no model could read is taken from the store
        (read_unread); a change opens again the once-only steps that would block a repair, since they hold per version of the
        task; a close refused CLOSE_RESENDS times goes to the code's commit, which decides without it."""
        unread = [line for line in lines if line not in self.heard]
        if unread:
            self.read_unread(unread)
        if self.ledger_version != version:
            self.stating_opened = self.cheaper_opened = self.decline_hinted = self.best_effort_told = False
            self.decline_rows_done.clear()
            self.stale_held.clear()
            self.better_fit_held.clear()
            self.commit_failures = 0
        if self.close_refused.get(key, 0) >= CLOSE_RESENDS and not self.committing:
            self.commit_wanted = True

    def read_unread(self, lines: list[str]) -> None:
        """What a refusal said when no model can read it (the money, the key or the time is out). Store calls cost nothing, so
        every cart line's price and stock are read again (stale: the re-read before the order takes them); a number written in
        the unit the request wrote its budget in is the new budget. A dropped or new need stays unread: the basket stands on
        the requirements it has."""
        for item in self.held_lines():
            self.catalog.setdefault(item, {"price": None, "stock": None, "text": ""})["stale"] = True
        budget = self.budget_in(lines)
        if budget is not None and budget != self.budget_now:
            self.set_budget("money", budget, "store")
            self.ledger_version += 1
        self.trace.append({"kind": "refusal_unread", "lines": lines[:3], "budget": budget})

    def budget_in(self, lines: list[str]) -> float | int | None:
        """The one number these lines write in the word that follows the budget's number in the request, or None."""
        budget = self.budget_now
        if not isinstance(budget, (int, float)) or isinstance(budget, bool):
            return None
        written = re.search(rf"(?<![\d.,]){re.escape(format(budget, 'f').rstrip('0').rstrip('.'))}(?![\d])\s+([^\W\d]+)", str(self.policy.get("query") or ""))
        if not written:
            return None
        found = re.findall(rf"(?<![\d.,])(\d+(?:\.\d+)?)\s+{re.escape(written.group(1))}\b", " ".join(lines), re.IGNORECASE)
        if len(set(found)) != 1:
            return None
        return float(found[0]) if "." in found[0] else int(found[0])

    def decide_finish(self, chosen: dict[int, tuple[str, str]] | None = None, lines: list[tuple[str, int]] | None = None, sent: bool = True) -> dict[str, Any]:
        """The one close decision, model-free, taken by every closer (commit_steps, force_finish, the decline checks):
        {"action": "order"}, or {"action": "decline", "kind": why, "text": what the code says, ...}. A decline stands only on a
        proof that no order can fit, and never while the request asks for the closest listing and one can be ordered. With no
        argument it weighs prices and stock; with chosen (each product's pick, {} before any) the picks' proven misses; with
        lines (listing, units) the basket itself and, for a basket about to be sent (sent), whether the store refused it as it stands."""
        if not self.roles.get("decline"):
            return {"action": "order"}
        products = range(len(self.items))
        named = lambda index: self.product(index) or "requested product"
        if chosen is None and lines is None:
            out = [index for index in products if self.priced_gap(index) and not self.closest_open(index)]
            if out:
                return {"action": "decline", "kind": "decline_priced_out", "products": out,
                        "text": f"Every listing I found that meets all the requirements of the {named(out[0])} costs more than the budget or is sold out, so I will decline."}
        if chosen is not None:
            missed = next((index for index in products if (chosen.get(index) or (None, ""))[1] in ("", "near fit") and self.stated_miss(index) and not self.closest_open(index)), None)
            if missed is not None:
                return {"action": "decline", "kind": "decline_stated_miss", "product": missed,
                        "text": f"Every listing of the {named(missed)} I opened fails a stated requirement and no other listing is left to open, so I will decline."}
            conflict = next(((index, need) for index in products if (not chosen or (chosen.get(index) or (None, ""))[1] == "near fit") and not self.closest_open(index) for need in [self.revealed_conflict(index)] if need), None)
            if conflict:
                return {"action": "decline", "kind": "decline_revealed", "product": conflict[0], "need": _describe(conflict[1]),
                        "text": f"No listing of the {named(conflict[0])} that meets the other requirements has {_describe(conflict[1])}, as the shopper asked, so I will decline."}
        if lines is not None:
            items = list(dict.fromkeys(item for item, _ in lines))
            fault = self.basket_fault(lines)
            if fault:
                return {"action": "decline", "kind": "decline_over_budget" if len(items) >= len(self.items) else "decline_short_basket", "total": sum((self.catalog.get(i, {}).get("price") or 0) * n for i, n in lines), "budget": self.budget_now,
                        "text": f"{fault[0].upper()}{fault[1:]}, so I will decline."}
            if sent and items and self.refused_out("order", items):
                return {"action": "decline", "kind": "decline_refused", "text": "The store did not take this order, and nothing has changed since, so I will decline."}
        return {"action": "order"}

    def close_decline(self, decision: dict[str, Any]) -> tuple[bool, bool]:
        """The code's commit acts on a decline decision."""
        self.trace.append({k: v for k, v in decision.items() if k not in ("action", "text")})
        return self.code_decline(decision["text"])

    def basket_rules(self, basket: dict[int, str], chosen: dict[int, tuple[str, str]]) -> tuple[dict[int, str], dict[int, tuple[str, str]]]:
        """A basket a basket-level rule breaks is changed for the nearest one that keeps every such rule and the budget (fewest
        lines changed, best-ranked listings first); with none, it stays (an order scores more often than a decline)."""
        try:
            if not self.basket_problems(basket):
                return basket, chosen
            ranked = {index: dict(self.ranked_pool(index)) for index in basket}
            pools = [list(dict.fromkeys([item] + list(ranked[index])))[:BASKET_POOL] for index, item in basket.items()]
            indexes = list(basket)
            trial = sorted(itertools.product(*pools), key=lambda lines: (sum(line != basket[index] for line, index in zip(lines, indexes)), sum(pool.index(line) for line, pool in zip(lines, pools))))
            for lines in trial[1:]:
                other = dict(zip(indexes, lines))
                if self.basket_total(other) <= self.budget_value() and not self.basket_problems(other):
                    self.trace.append({"kind": "basket_rule_swap", "basket": basket, "instead": other})
                    return other, {**chosen, **{index: (item, ranked[index].get(item) or (chosen.get(index) or (item, "near fit"))[1]) for index, item in other.items()}}
            self.trace.append({"kind": "basket_rule_broken", "basket": basket})
        except Exception as exc:
            self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
        return basket, chosen

    def cheaper_unclear(self, chosen: dict[int, tuple[str, str]]) -> dict[int, tuple[str, str]]:
        """When the cheapest is asked for: an opened listing cheaper than a product's pick whose checks are only UNCLEAR (no
        failure) ranks below every full fit; ask the model check about the cheapest such listing, and take it if it passes."""
        if self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS or self.spent_out():
            return chosen  # the model check could not run: an unchecked UNCLEAR listing is not taken
        listings = self.listings()
        out = dict(chosen)
        for index, (pick, state) in chosen.items():
            price = (self.catalog.get(pick) or {}).get("price")
            if price is None:
                continue
            cheaper = []
            for item in listings:
                entry = self.catalog.get(item) or {}
                if self.item_of(item) != index or entry.get("price") is None or entry["price"] >= price or entry.get("stock") is False or self.over_budget(item, index) or self.verdict(item):
                    continue
                if (self.assess(item, listings, judge=False) or {}).get("state") == "unclear":
                    cheaper.append((entry["price"], item))
            if cheaper:
                item = min(cheaper)[1]
                if self.checked(item):
                    out[index] = (item, "unclear, checked")
                    self.trace.append({"kind": "cheaper_unclear", "product": index, "item": item, "instead_of": pick})
        return out

    def open_cheaper(self, chosen: dict[int, tuple[str, str]]) -> bool:
        """When the cheapest is asked for: before the code orders, open (once) up to two seen but unopened listings of a
        chosen product that cost less than its pick, in stock and within the budget, from the rows its searches for the
        required values showed with a label of the pick's (as the cheaper-unopened check draws them): rows whose text states
        every required value first, none whose text names another value of a required property. True when it opened any."""
        details = self.reader()
        if not details:
            return False
        room = self.steps_left() - self.commit_steps_needed(self.th('repair') + 1) - 1  # each page read leaves one opening and the commit's steps
        pages = [c for index in sorted(chosen) for q in self.unread_searches(index)[:1] for c in [self.next_page({"name": q["tool"], "args": q["args"]})] if c][:max(0, room)]
        if pages and self.time_left() >= WRAP_UP_SECONDS / 2:  # the rows the product's searches can still show come first (the next commit opens the cheaper ones)
            self.commit_paused = True
            sent = [{"id": f"pages_{self.step}_{n}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for n, c in enumerate(pages)]
            self.messages.append({"role": "assistant", "content": "The cheapest is asked for, so before ordering I will read the result pages my searches have left.", "tool_calls": sent})
            for reply in self.run_calls(sent, planned=True):
                self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})
            self.trace.append({"kind": "commit_read_pages", "calls": [c["args"] for c in pages]})
            return True
        listings = self.listings()
        found: list[str] = []
        for index, (pick, _) in chosen.items():
            entry = self.catalog.get(pick) or {}
            price = entry.get("price")
            if price is None:
                continue
            wanted = [r for r in self.reqs_for(index, blocking=True) if r["op"] == "is" and r["values"]]
            rows = [i for i in self.pool(index) - set(listings) for e in [self.catalog.get(i) or {}] if e.get("price") is not None and e["price"] < price
                    and e.get("stock") is not False and self.item_of(i) == index and not self.over_budget(i, index)
                    and e.get("labels", set()) & entry.get("labels", set()) and not self.row_conflict(i, index, listings)]
            states = lambda i: bool(wanted) and all(any(_says(self.catalog[i].get("text") or "", v) for v in r["values"]) for r in wanted)
            found += sorted(rows, key=lambda i: (not states(i), self.catalog[i]["price"], i))[:2]
        calls = [c for c in (self.checked_call(details, _fill(self.schemas.get(details), i, None, {})) for i in sorted(found, key=lambda i: self.catalog[i]["price"])[:4]) if c]
        calls = calls[: max(0, self.steps_left() - self.commit_steps_needed(self.th('repair') + 1))]  # each opened row leaves the order, its cart calls and its repair their steps
        if not calls or self.time_left() < WRAP_UP_SECONDS / 2:
            return False
        self.cheaper_opened = True
        self.commit_paused = True
        sent = [{"id": f"cheaper_{self.step}_{n}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for n, c in enumerate(calls)]
        self.messages.append({"role": "assistant", "content": "The cheapest is asked for, so before ordering I will open cheaper listings I have seen but not opened.", "tool_calls": sent})
        for reply in self.run_calls(sent, planned=True):
            self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})
        self.trace.append({"kind": "commit_opened_cheaper", "items": [c["args"] for c in calls]})
        return True

    def reread_stock(self, items: list[str]) -> bool:
        """Before the code's own order: read again the stock of each line whose stock was last read before the cart last
        changed, with the call that read that one listing (a store may sell out a cart line without saying so; only a later
        read shows it), and the price of each cart line the store said changed price since it was last read. False when a
        line is now shown out of stock, or a price read now differs from the one the choice was made at (a price change can
        re-rank the candidates: the next commit chooses again at the prices now shown)."""
        calls, priced = [], []
        stale = [i for i in self.held_lines() if (self.catalog.get(i) or {}).get("stale")]
        details = self.roles.get("details")
        for item in dict.fromkeys(items + stale):
            entry = self.catalog.get(item) or {}
            reader = entry.get("stock_reader") or entry.get("rows_reader")  # one search can re-read several lines
            if not reader and item in stale and details:
                reader = (details, _fill(self.schemas.get(details), item, None, {}))
            call = self.checked_call(reader[0], reader[1]) if reader and (item in stale or entry.get("stock_step", -1) < self.cart_changed_step) else None
            if call and call not in calls + priced:
                (priced if item in stale else calls).append(call)
        prices = {i: (self.catalog.get(i) or {}).get("price") for i in dict.fromkeys(items + stale)}
        # a changed price decides the choice, so its read keeps only the steps a new choice needs (its cart calls and the
        # order); a stock re-read only with room to spare: after the re-reads and the order, a refused order keeps its repair steps
        if priced and self.steps_left() - len(priced) < (1 if self.refusals else self.commit_steps_needed() - 2):  # (after a refusal the read decides, even if only a resend or a decline is left)
            priced = []
        if calls and self.steps_left() - len(priced) - len(calls) - 1 <= self.th('cart_only'):
            calls = []
        calls = priced + calls
        if calls and self.time_left() > LAST_CALL_SECONDS * (len(calls) + 2):
            sent = [{"id": f"stock_{self.step}_{n}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for n, c in enumerate(calls)]
            said = "Before ordering, I will read again the price the store said changed." if priced and not calls[len(priced):] else "Before ordering, I will check that the cart's listings are still in stock."
            self.messages.append({"role": "assistant", "content": said, "tool_calls": sent})
            for reply in self.run_calls(sent, planned=True):
                self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})
        gone = [i for i in items if (self.catalog.get(i) or {}).get("stock") is False]
        if gone:
            self.trace.append({"kind": "stock_gone", "items": gone})
        moved = [i for i, price in prices.items() if (self.catalog.get(i) or {}).get("price") != price]
        if moved:
            self.trace.append({"kind": "price_moved", "items": moved})
        return not gone and not moved

    def refresh_cart(self) -> None:
        """Read the cart once more when an order came back with an error: the commit then builds on what the cart holds."""
        view = self.roles.get("cart_view")
        call = self.checked_call(view, {}) if view else None
        if not self.cart_doubt or not call or self.steps_left() < self.th('last') + 1:  # the view, one cart call and the order
            return
        self.cart_doubt = False
        message = {"id": f"cartview_{self.step}", "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["args"])}}
        self.messages.append({"role": "assistant", "content": "The store did not take the order, so I will look at the cart first.", "tool_calls": [message]})
        for reply in self.run_calls([message], planned=True):
            self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})

    def commit_steps(self) -> tuple[bool, bool]:
        self.refresh_cart()
        if self.find_hop(commit=True):
            self.commit_paused = True
            return True, False  # a step towards a listing a requirement points at: its value may change the choice
        held = self.viable_lines()
        if self.several_products() or (self.multi() and any(self.item_of(i) is None for i in held) and len(held) >= len(self.items)):
            # the navigator's own multi-product basket (or one holding a line of unknown product, a line for every product by
            # count): order it as it stands
            finish = self.order_call(None)
            if finish is None:
                return False, False
            if self.open_line_pages(held, self.steps_left() - 2):
                return True, False  # (a page can carry a note)
            item, state, steps, basket = "", "", [], {}
            opening = f"{self.commit_reason()}, so I will order the basket as it stands."
        else:
            decision = self.decide_finish()
            if decision["action"] == "decline":
                return self.close_decline(decision)
            chosen = self.pair_choice(self.picks())
            if self.relations and self.open_linked(chosen):
                return True, False  # the next commit chooses among them too
            if self.conditional and self.open_ruled(chosen):
                return True, False  # (the same)
            if not self.stating_opened and chosen and (len(chosen) < len(self.items) or any(state != "fits" or self.title_only(item) for item, state in chosen.values())) and self.open_stating(chosen):
                return True, False  # listings stating the required values were opened first (for a product without a full fit, or without any pick): the next commit chooses again
            if (len(chosen) < len(self.items) or any(state in ("near fit", "unclear") for _, state in chosen.values())) and self.sweep_paused():
                return True, False  # (no pick is settled on while rows it could be are unread)
            decision = self.decide_finish(chosen)
            if decision["action"] == "decline":
                return self.close_decline(decision)
            if self.plan.get("priority") == "cheapest" and not self.cheaper_opened and self.open_cheaper(chosen):
                return True, False  # cheaper listings were opened first: the next commit chooses among them too
            if self.plan.get("priority") == "cheapest":
                chosen = self.cheaper_unclear(chosen)
            if self.ready_commit or self.completing or self.review_held:  # the navigator's own verified line for a product stays its choice (a reviewed line also on an UNCLEAR check)
                for line in held:
                    index = self.product_of(line)
                    assessment = self.assess(line) if index is not None else None
                    if assessment and (assessment["state"] == "fits" or (self.review_held and assessment["state"] == "unclear")) and not self.verify_item(line, "ORDER") and not self.rival(line):
                        chosen[index] = (line, "fits")  # (unless another listing ranks ahead of it by the request's rule, as after a price change)
            for index, (item, state) in list(chosen.items()):
                if state != "near fit" and self.verify_item(item, "ORDER"):
                    again = self.best_candidate(relax=True, index=index, last_resort=True)
                    if again[0]:
                        chosen[index] = again
            for index, (item, state) in list(chosen.items()):
                price = (self.catalog.get(item) or {}).get("price")
                unverified = state in ("near fit", "unclear") and self.verdicts.get(f"{item}|{self.ledger_version}") != ""
                # a near fit or an unchecked UNCLEAR pick never displaces a verified fit; no pick displaces a cheaper verified cart line
                keep = self.verified_pick(index) if unverified else None
                if not keep and self.plan.get("priority") != "best" and price is not None:
                    keep = self.verified_pick(index, cheaper_than=price, held_only=True)
                if keep and keep != item:
                    self.trace.append({"kind": "verified_kept", "product": index, "item": keep, "instead_of": item, "state": state})
                    chosen[index] = (keep, "verified")
            if not chosen:
                return False, False
            picked = dict(chosen)
            for _ in range(3):  # a cheaper listing that fails its check is skipped by the next round (cheapest_fit reads the verdicts)
                chosen = self.within_budget(picked)
                if not any(item != picked[index][0] and state != "near fit" and self.verify_item(item, "ORDER") for index, (item, state) in chosen.items()):
                    break
            else:
                chosen = {index: picked[index] if self.verdict(item) else (item, state) for index, (item, state) in chosen.items()}
            basket = {index: item for index, (item, _) in chosen.items()}
            if self.multi():  # a product with no pick keeps the line the navigator holds for it (the cart call may replace the basket)
                kept = {index: i for i in held for index in [self.product_of(i)] if index is not None}
                basket = {**{index: i for index, i in kept.items() if index not in basket}, **basket}
                spare = [i for i in held if self.item_of(i) is None and i not in basket.values()]
                for index in [n for n in range(len(self.items)) if n not in basket]:
                    if not spare:
                        break  # a held line of a product not known yet may be any such product's (one with its count first)
                    basket[index] = next((i for i in spare if self.held.get(i) == self.qty(index)), spare[0])
                    spare.remove(basket[index])
            missing = [index for index in range(len(self.items)) if index not in basket]
            if missing:  # a short basket never fits: complete it with the best listings known where no decline is proven
                filled = self.best_effort_fill(missing, basket)
                basket.update(filled)
                chosen.update({index: (item, "best effort") for index, item in filled.items()})
            basket, chosen = self.basket_rules(basket, chosen)
            if self.read_picks(list(basket.values())):
                return True, False  # the next commit chooses with their properties read
            if self.open_line_pages(list(basket.values()), self.steps_left() - len(self.cart_plan(basket) or []) - 2):
                return True, False  # (the same: a page can carry a note)
            decision = self.decide_finish(lines=[(i, self.qty(index)) for index, i in basket.items()])
            if decision["action"] == "decline" and self.sweep_paused():
                return True, False  # cheaper rows are still unread: no decline on the listings opened so far
            if decision["action"] == "decline":
                # a basket missing a product, or over the budget even with the cheapest fitting listings found, never fits,
                # while a decline is right whenever no basket can
                return self.close_decline({**decision, "basket": basket})
            kept = {index: i for i in held for index in [self.product_of(i)] if index is not None}
            if self.cart_plan(basket) is None and len(kept) == len(self.items) and self.cart_plan(kept) is not None:
                basket, chosen = kept, {index: (i, (chosen.get(index) or (i, "near fit"))[1]) for index, i in kept.items()}  # no call takes a line out: the cart as it stands, never no order
            item, state = chosen[min(chosen)]
            planned = self.cart_plan(basket)
            if planned is not None and len(planned) + 1 > self.steps_left():
                planned = self.cart_plan(basket, fix_counts=False)  # no room to put a held count right: order the lines as they are
            finish = self.order_call(item)
            if planned is None or finish is None:
                self.trace.append({"kind": "commit_unbuilt", "basket": basket, "cart": planned is not None, "order": finish is not None})
                self.commit_help(basket)
                return False, False
            steps = planned
            if self.multi():
                opening = f"{self.commit_reason()}, so I will order my best candidates now: " + ", ".join(f"{i} ({s_}) for product {index}" for index, (i, s_) in sorted(chosen.items())) + "."
            else:
                opening = f"{self.commit_reason()}, so I will order my best candidate {item} ({state}) now."
        order = finish["name"]
        steps.append(finish)
        if len(steps) > self.steps_left():
            return False, False
        self.trace.append({"kind": "commit", "item": item, "state": state, "steps": steps})
        ordered = False
        for index, action in enumerate(steps):
            if action["name"] == order:
                self.order_checked = True
                lines = list(basket.values()) if basket else list(held)
                if not self.reread_stock(lines):
                    self.commit_paused = True
                    break  # a line is no longer in stock: the next commit picks again
            call = {"id": f"commit_{self.step}_{index}", "type": "function", "function": {"name": action["name"], "arguments": json.dumps(action["args"])}}
            note = opening if index == 0 else "Continuing."
            self.messages.append({"role": "assistant", "content": note, "tool_calls": [call]})
            version, errors = self.cart_version, self.order_errors
            replies = self.close_calls([call]) if action["name"] == order else self.run_calls([call], planned=True)
            for reply in replies:
                self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})
            ordered = ordered or (action["name"] == order and bool(replies) and not any(reply.held_back for reply in replies) and self.order_errors == errors)
            if self.finished or any(reply.held_back for reply in replies):
                break
            if action["name"] != order and action["name"] in self.cart_tools() and self.cart_version == version:
                break  # the store refused a cart call: the cart is not as planned, so the order is not sent
        return True, ordered

    def read_picks(self, items: list[str]) -> bool:
        """Before the code orders: read the properties of the lines it is about to order whose properties were never read, whatever
        path chose them, once each and only while the endgame's steps stay whole. True when a call went out."""
        reader = self.reader()
        listings = self.listings()
        unread = [i for i in items if i not in listings and i not in self.pick_reads]
        calls = [c for c in (self.checked_call(reader, _fill(self.schemas.get(reader), i, None, {})) for i in unread[:3]) if c] if reader else []
        room = self.steps_left() - self.commit_steps_needed()
        if not calls or room < 1 or self.time_left() < LAST_CALL_SECONDS * (len(calls) + 2):
            return False
        calls = calls[:room]
        self.pick_reads.update(unread[:len(calls)])
        self.commit_paused = True
        sent = [{"id": f"pick_{self.step}_{n}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for n, c in enumerate(calls)]
        self.messages.append({"role": "assistant", "content": "Before ordering I will read the properties of the listings I am about to order.", "tool_calls": sent})
        for reply in self.run_calls(sent, planned=True):
            self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})
        self.trace.append({"kind": "commit_read_picks", "items": [c["args"] for c in calls]})
        return True

    def page_tool(self) -> str | None:
        """The tool that shows a listing's page in a store that splits page and properties over two tools reading one listing
        (race 7 and 11); None when one tool shows both. Such a store can count only that tool as opening a listing."""
        reader = self.reader()
        return next((t for t in [self.roles.get("details"), *self.one_listing_tools()] if t and t != reader), None) if reader else None

    def pages_due(self) -> int:
        """Steps the pages of the order's lines still take: one a product, less the cart lines already opened (kept by store_step)."""
        tool = self.page_tool()
        if not tool:
            return 0
        done = self.called_on.get(tool, set())
        return max(0, len(self.items) - sum(1 for i in self.held_lines() if i in done))

    def line_page_calls(self, items: list[str], room: int) -> list[dict[str, Any]]:
        """Calls that open the pages of the given lines, once each, within room steps."""
        tool = self.page_tool()
        if not tool or room < 1 or self.time_left() < LAST_CALL_SECONDS * 3:
            return []
        done = self.called_on.get(tool, set())
        due = [i for i in dict.fromkeys(items) if i not in done and i not in self.line_pages_sent]
        return [c for c in (self.checked_call(tool, _fill(self.schemas.get(tool), i, None, {})) for i in due) if c][:room]

    def open_line_pages(self, items: list[str], room: int) -> bool:
        """Before the code orders: open the page of every line about to be ordered, in view of the navigator's record. The next
        commit chooses again, since a page can carry a note that changes the choice. True when a call went out."""
        calls = self.line_page_calls(items, room)
        if not calls:
            return False
        self.line_pages_sent.update(i for i in items if any(i in c["args"].values() for c in calls))
        self.commit_paused = True
        sent = [{"id": f"line_page_{self.step}_{n}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for n, c in enumerate(calls)]
        self.messages.append({"role": "assistant", "content": "Before ordering I will open the page of every listing I am about to order.", "tool_calls": sent})
        for reply in self.run_calls(sent, planned=True):
            self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})
        self.trace.append({"kind": "commit_line_pages", "items": [c["args"] for c in calls]})
        return True

    def last_line_pages(self) -> bool:
        """The last check of any order, whichever path sent it: open what is still unopened of its lines, out of the prompt (a
        navigator's call is waiting for its answer there), keeping one step for a refusal. True when what was read moved the
        ledger or the cart, so the order is no longer the one that was checked."""
        if self.opening_lines:
            return False
        lines = self.held_lines()
        calls = self.line_page_calls(lines, self.steps_left() - 2)
        if not calls:
            return False
        version = (self.ledger_version, self.cart_version)
        self.line_pages_sent.update(i for i in lines if any(i in c["args"].values() for c in calls))
        sent = [{"id": f"last_page_{self.step}_{n}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for n, c in enumerate(calls)]
        self.opening_lines, self.quiet = True, True
        try:
            self.run_calls(sent, planned=True)
        finally:
            self.opening_lines, self.quiet = False, False
        self.trace.append({"kind": "last_line_pages", "items": [c["args"] for c in calls]})
        return version != (self.ledger_version, self.cart_version)

    def safe_ready(self) -> str | None:
        try:
            return self.ready_choice()
        except Exception as exc:
            self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return None

    def item_records(self, item: str) -> list[str]:
        """The latest of the first eight different records that name a listing (for model prompts)."""
        found: list[str] = []
        for record in self.records:
            for value in _find(record, lambda d: any(isinstance(v, str) and v == item for v in d.values()), 6):
                text = _compact(_brief(value, 400))[:2500]
                if text not in found and len(found) < 8:
                    found.append(text)
            if len(found) >= 8:
                break
        return found[-4:]

    def unopened_problem(self, item: str) -> str:
        """A listing whose property details were never opened cannot be shown to meet its product's requirements: the price,
        stock or notes another tool shows are not its properties. Open it first (while steps allow)."""
        details = self.reader()
        if not details or item in self.listings() or self.steps_left() <= self.th('repair') or self.time_left() < WRAP_UP_SECONDS:
            return ""
        index = self.product_of(item)
        if index is None and self.multi():
            return ""
        blocking = [r for r in self.reqs_for(index if index is not None else 0, blocking=True) if not r.get("pending")]
        if not blocking:
            return ""
        return f"{item}'s property details have not been opened, so nothing shows it meets " + "; ".join(_describe(r) for r in blocking[:4]) + f". Open its details with {details} first."

    def verify_cart(self) -> str:
        if not self.cart_ids or self.several_products():
            return ""
        if not self.multi():
            return self.verify_item(self.cart_ids[-1], "ORDER")
        for line in self.held_lines():
            verdict = self.verify_item(line, "ORDER")
            if verdict:
                return verdict
        return ""

    def money_problem(self, item: str) -> str:
        entry = self.catalog.get(item, {})
        price = entry.get("price")
        budget = self.budget_value()
        count = self.qty(self.item_of(item))
        if entry.get("stock") is False:
            return "is shown out of stock"
        if price is not None and price * count > budget:
            return f"costs {format(price, 'g')} x {count} = {format(price * count, 'g')}, over the budget of {format(budget, 'g')}"
        limit = next((r for r in self.reqs_for(self.item_of(item)) if r.get("field") and price is not None and not _within(r, price)), None)
        if limit:
            return f"costs {format(price, 'g')}, outside the shopper's limit ({_describe(limit)})"
        return ""

    def verdict(self, item: str) -> str:
        return self.verdicts.get(f"{item}|{self.ledger_version}", "")

    def verdict_kind(self, item: str) -> str:
        """"exact", "model" (also for a verdict set without a kind), or "" when the listing has no failing verdict."""
        key = f"{item}|{self.ledger_version}"
        return self.verdict_kinds.get(key, "model") if self.verdicts.get(key) else ""

    def set_verdict(self, item: str, kind: str, text: str) -> None:
        key = f"{item}|{self.ledger_version}"
        self.verdicts[key] = text
        self.verdict_kinds[key] = kind

    def clear_verdicts(self, kind: str) -> None:
        """Forget the verdicts of one kind (the exact checks change when a requirement is relaxed)."""
        dropped = {k for k, v in self.verdicts.items() if v and self.verdict_kinds.get(k, "model") == kind}
        self.verdicts = {k: v for k, v in self.verdicts.items() if k not in dropped}
        self.verdict_kinds = {k: v for k, v in self.verdict_kinds.items() if k not in dropped}

    def checked(self, item: str) -> bool:
        """The final check passed the listing on an answer (or on its own fields). A check that could not run or got no answer
        holds no order back, but it confirms nothing: it never makes a listing that is not a full fit the choice."""
        return not self.verify_item(item, "ORDER") and self.verdicts.get(f"{item}|{self.ledger_version}") == ""

    def verify_item(self, item: str, purpose: str) -> str:
        key = f"{item}|{self.ledger_version}"
        if key in self.verdicts:
            return self.verdicts[key]
        if key in self.unanswered:
            return ""  # (not asked again in this ledger)
        unopened = self.unopened_problem(item)
        if unopened:
            return unopened  # not kept: it ends as soon as the details are opened
        assessment = self.assess(item)
        if assessment and assessment["state"] == "fails":
            verdict = f"The exact check says {item} does NOT meet every requirement: " + "; ".join(f"{_describe(r)}: {state}" + (f" ({evidence[:80]})" if evidence else "") for r, state, evidence in assessment["fails"]) + ". Choose another listing whose check fits."
            self.trace.append({"kind": "verify", "item": item, "exact": assessment["state"]})
            self.set_verdict(item, "exact", verdict)
            return verdict
        money = self.money_problem(item)
        if money:
            self.trace.append({"kind": "verify", "item": item, "code": money})
            return f"{item} {money}. Choose another listing."
        if assessment and assessment["state"] == "fits" and not assessment["contradicted"] and not assessment.get("weak") and not assessment.get("labels") and not _evidence_class(assessment):
            shown = self.item_map.get(item, -1) if self.shelf_values else -1  # mapped to a product on the store's shelf list
            named = self.label_product(item) if shown < 0 else None
            if named is not None and self.item_of(item) == named:
                shown = named  # or one of its own labels names the product it is (the model check passed every such listing of the replica runs)
            if shown >= 0 and not any(r.get("pending") for r in self.reqs_for(shown)):
                self.verdicts[key] = ""  # every requirement met by its own fields, and it is the requested kind: nothing left for the model check
                return ""
        if self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS:
            return ""
        records = self.item_records(item)
        if not records:
            return ""
        prompt = VERIFY_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + self.ledger()
        prompt += f"\n\nTHE ITEM TO {purpose}: {item}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n" + "\n".join(records)
        if assessment:
            lines = self.check_lines(assessment)
            for n, (r, _, evidence) in enumerate(assessment["results"]):
                if any(r is label for label in assessment.get("labels") or []):
                    lines[n] = f"{_describe(r)}: UNCLEAR (only a variant or option label states it: {evidence[:90]})"
                elif not r.get("field") and n < len(assessment.get("levels") or []) and assessment["levels"][n] in (2, 3) and assessment["results"][n][1] == "met":
                    lines[n] = f"{_describe(r)}: UNCLEAR (" + (f"no field names the property; only this states it: {evidence[:90]})" if evidence else "the listing has no field for this property, so the exclusion cannot be confirmed)")
            prompt += "\n\nEXACT CHECKS (final):\n" + "\n".join(lines)
        else:
            prompt += "\n\nEXACT CHECKS: this listing's property details were not opened, so treat every requirement above as UNCLEAR and decide it from the records."
        if self.last_cart_view is not None:
            prompt += "\n\nLATEST CART VIEW:\n" + _compact(self.last_cart_view)[:3000]
        labels = self.row_labels()
        if labels:
            prompt += "\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the item's category label must be the one that matches the request's wording as a whole, not a neighbouring one):\n" + labels
        result = self.finish_call(lambda: self.judge(prompt, JUDGE_TOKENS))
        self.trace.append({"kind": "verify", "item": item, "result": result})
        if not result:
            if not self.call_skipped:
                self.unanswered.add(key)  # no answer: nothing is known about the listing (a call skipped for its price may be asked later)
            return ""
        if result.get("fits") is not False:
            self.verdicts[key] = ""
            return ""
        problems = [str(p) for p in result.get("problems") or []][:5]
        verdict = f"An independent check says {item} does NOT meet every requirement: " + "; ".join(problems) + ". Unless you are certain the check is wrong, choose another fully verified fit."
        self.set_verdict(item, "model", verdict)
        return verdict

    def tools_for_turn(self) -> list[dict[str, Any]]:
        left = self.steps_left()
        order, decline, cart = self.roles.get("order"), self.roles.get("decline"), self.roles.get("cart")
        carts = self.cart_tools()  # every tool that changes the cart, including a separate remove tool once known
        names: set[str | None] | None = None
        if left <= self.th('finish_only'):
            names = {order, decline}
        elif left <= self.th('cart_only'):
            names = {order, decline} | carts
        elif left <= self.th('reading_ends') and (self.order_attempts == 0 or (self.refusals > 0 and left <= self.th('repair'))):
            # after a refused order, reading stays open a little longer: the change it showed may need a new search
            opening = {self.roles.get("details"), self.reader()}
            reading = ({self.roles.get(r) for r in ("search", "message")} | opening) - {self.roles.get("view")}
            if any(r.get("pending") for r in self.reqs):
                reading -= opening  # a listing a requirement points at must still be opened with it
            if left > self.th('cheaper_open') + 2 * (len(self.items) - 1) and self.cheaper_rows():
                reading -= opening  # cheaper seen rows that may fit are still unchecked, and the steps allow one opening
            if self.multi() and self.order_attempts == 0 and any(not self.best_candidate(index=index, last_resort=True)[0] for index in range(len(self.items))):
                reading -= {self.roles.get("search")} | opening  # a product without any listing yet still needs one
            names = ({t["function"]["name"] for t in self.tools} - reading) | {order, decline} | carts
        tools = self.tools
        if names is not None and order and cart:
            narrowed = [t for t in self.tools if t["function"]["name"] in names]
            if narrowed:
                tools = narrowed
        if self.roles.get("message") and not (self.message_window and self.messages_used < self.message_total()):
            # with no message left the tool does not come back when the planner stops: any message would be refused, and a
            # tool list that changes mid-episode loses the cached conversation (on the replica runs the tool came back with no
            # message left in 261 of 270 episodes). With messages left it comes back as before.
            tools = [t for t in tools if t["function"]["name"] != self.roles.get("message")]
        return tools

    def property_menu(self) -> tuple[str, int]:
        """The properties the opened listings record, per product, with example values (for the question planner). A hidden
        need is a value the listings meeting the requirements differ on, so each product's menu counts its opened listings that
        meet the requirements so far (all of them while fewer than MENU_FITS do), each listing once, never a page read only
        for the value a pointer takes from it; a fact listing several values adds each of them. The code's own fill-in order
        keeps the product's other varying properties after those."""
        fact_sets = 0

        def tally(found: dict[str, dict[str, list[str]]], listings: list[str]) -> tuple[dict[str, int], dict[str, list[str]]]:
            counts: dict[str, int] = {}
            samples: dict[str, list[str]] = {}
            for listing in listings:
                for name, values in found[listing].items():
                    counts[name] = counts.get(name, 0) + 1
                    bucket = samples.setdefault(name, [])
                    bucket += [v for v in values if v not in bucket][:max(0, 5 - len(bucket))]
            return counts, samples

        fits = self.fitting_items()
        members: dict[int, dict[str, dict[str, list[str]]]] = {}  # product -> listing -> property -> its values
        for listing in (found for record in self.records for found in self.record_listings(record)):
            index = self.product_of(listing[0])
            if index is None or not any(isinstance(v, str) and 0 < len(v) <= MENU_VALUE_CHARS for v in listing[1].values()):
                continue
            fact_sets += 1
            values = {re.sub(r"[_.]+", " ", str(k)).strip().lower(): _menu_values(v) for k, v in listing[1].items()}
            values = {k: v for k, v in values.items() if v}
            if listing[0] not in self.pointer_pages and len(values) >= len(members.get(index, {}).get(listing[0], {})):
                members.setdefault(index, {})[listing[0]] = values  # a listing opened again counts once, with its fullest record
        sections = []
        self.menu_pairs = []
        self.menu_sections = {}
        self.menu_values = {}
        self.recorded = {index: {name for values in found.values() for name in values} for index, found in members.items()}
        for index in sorted(members):
            fitting = [i for i in members[index] if i in fits]
            counted = fitting if len(fitting) >= MENU_FITS else list(members[index])
            counts, samples = tally(members[index], counted)
            total, everywhere = tally(members[index], list(members[index])) if len(counted) < len(members[index]) else (counts, samples)
            split = lambda k: len({frozenset(members[index][i][k]) for i in fitting if k in members[index][i]}) > 1
            splitting = {k for k in counts if split(k)} if len(fitting) >= 3 else set()
            ranked = sorted(counts, key=lambda k: (-counts[k], -total[k]))[:28]  # ties: the more of the product's listings record it
            order = [k for k in ranked if k in splitting] + [k for k in ranked if k not in splitting and len(samples.get(k, [])) > 1]
            # fill-ins: then the product's other varying properties (a message left unsent asks nothing)
            reserve = [k for k in sorted(total, key=lambda k: -total[k])[:28] if k not in order and len(everywhere[k]) > 1]
            self.menu_pairs += [(index, k) for k in sorted(order + reserve, key=lambda k: self.composite(k, list(members[index].values())))]
            self.menu_values[index] = {k: samples.get(k) or everywhere[k] for k in ranked + [k for k in total if k not in ranked]}

            def note(k: str) -> str:
                return f" [differs among the {len(fitting)} listings that meet the stated requirements]" if k in splitting else ""
            lines = [f"- {k} (in {counts[k]} listings){note(k)}: " + " | ".join(samples.get(k, [])) for k in ranked]
            sections.append((f"product {index} ({self.product(index)}):\n" if self.multi() else "") + "\n".join(lines))
            self.menu_sections[index] = sections[-1]
        self.menu_order = [k for _, k in self.menu_pairs]
        return "\n".join(sections), fact_sets

    def composite(self, name: str, found: list[dict[str, list[str]]]) -> bool:
        """Most of these listings fill this property with several values, or with what another of their properties already
        records (a bundle of variant labels). A hidden need sits on such a property about a quarter as often as on a plain one
        (eight worlds), so the code's own fill-in asks them last; the planner still sees them, its picks of them hit nearly as often."""
        def bundled(record: dict[str, list[str]]) -> bool:
            words = set(_words(" ".join(record[name])))
            return len(record[name]) > 1 or bool(re.search(r"[;|+]|,(?!\d)|(?<!\d)/|/(?!\d)", " ".join(record[name]))) or any(0 < len(part) < len(words) and part <= words for part in (set(_words(" ".join(v))) for k, v in record.items() if k != name))
        records = [record for record in found if name in record]
        return sum(1 for record in records if bundled(record)) * 2 >= len(records) > 0

    def fitting_items(self) -> set[str]:
        if not any(r["blocking"] for r in self.reqs):
            return set()
        listings = self.listings()
        return {item for item in listings if (self.assess(item, listings, judge=False) or {}).get("state") not in (None, "fails")}

    def row_labels(self) -> str:
        counts: dict[str, dict[str, int]] = {}

        def walk(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                if len(value) >= 4:
                    for key, item in value.items():
                        if isinstance(item, str) and 0 < len(item) <= 40 and not (re.search(r"\d", item) and re.fullmatch(r"[A-Za-z0-9_\-]+", item)):
                            bucket = counts.setdefault(str(key), {})
                            bucket[item] = bucket.get(item, 0) + 1
                for item in value.values():
                    if isinstance(item, (dict, list)):
                        walk(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    walk(item, depth + 1)

        for record in self.records:
            walk(record, 0)
        lines = []
        for key, bucket in counts.items():
            total = sum(bucket.values())
            if total >= 4 and 2 <= len(bucket) <= 8 and max(bucket.values()) >= 2:
                lines.append(f"- {key}: " + " | ".join(f"{v} (x{n})" for v, n in sorted(bucket.items(), key=lambda kv: -kv[1])))
        return "\n".join(lines[:12])

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get("message")
        schema = self.schemas.get(name or "") or {}
        properties = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        strings = [k for k, v in properties.items() if "string" in _kinds(v) and "enum" not in _branch(v)]
        target = next((k for k in required if k in strings), None) or (strings[0] if len(strings) == 1 else None)
        if target and not any(k != target for k in required):
            return {target: text}
        format_ = self.model_format("message") if name else None  # e.g. a second required argument: let the model write the call once
        if format_ and format_["tool"] == name:
            return _apply(format_["args"], [], text)
        return None

    def message_total(self) -> int:
        """Messages allowed: the budget the request (or a later line) states, or the store showed by refusing one, else the
        default, plus one for each further product of a multi-product order (each product can hold its own hidden need)."""
        return self.message_budget if self.message_budget is not None else DEFAULT_MESSAGES + len(self.items) - 1

    def message_limit(self) -> int:
        """Messages the questions (and replies) may use: the allowance less the messages kept for due instructions."""
        return max(0, self.message_total() - self.reserved_messages())

    def ask_planned(self, force: bool = False) -> bool:
        return self.in_phase("planner", lambda: self.ask_planned_now(force))

    def ask_planned_now(self, force: bool = False) -> bool:
        tool = self.roles.get("message")
        limit = self.message_limit()
        if not tool or self.messages_used >= limit:
            self.planner_on = False
            return False
        menu, fact_sets = self.property_menu()
        # (only what the choice cannot start without goes before the facts: a limit on the money; anything else waits, first in line)
        urgent = any(re.search(r"\b(?:money|budget|spend|price|cost|limit)\b", text, re.IGNORECASE) for text in self.must_learn()) and not self.asked
        if fact_sets < 2 and self.step < self.th('start') and not urgent and not force:
            return False  # a question chosen before any listing is opened is a guess from the story
        if force:
            urgent = False
        if self.steps_left() <= min(self.th('questions_end'), max(4, self.max_steps // 4)) or self.time_left() < WRAP_UP_SECONDS + QUESTION_SECONDS or not self.late_room(floor=False):
            self.planner_on = False
            return False
        if not self.late_room():
            return False  # only the floor still owed holds it back (a product has no opened listing): the question waits, it is not given up
        if self.asked and not force and not self.ask_due and self.step < self.last_ask_step + 2 and fact_sets < self.last_ask_facts + 2 and self.steps_left() > self.th('low'):
            return False
        target = self.next_product() if self.multi() else None
        if target is not None and target not in self.recorded and not urgent and not force and self.step < self.th('start') + self.th('start'):
            return False  # no listing of that product opened yet: its menu would be a guess
        count = limit - self.messages_used
        self.late_call = self.spent_out()  # past the stop the question is still planned (and its answer read)
        try:
            planned = self.plan_questions(count, menu, target)[:1]
        finally:
            self.late_call = False
        calls = []
        labels = []
        for index, item in enumerate(planned):
            args = self.message_args(item["question"])
            if not args:
                continue
            args, problems = _check(args, self.schemas[tool], "args")
            if problems:
                continue
            calls.append({"id": f"ask_{self.messages_used + index + 1}_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}})
            labels.append((item["property"], item["question"], item.get("item")))
        if not calls or self.key_out():  # (a key found spent while planning: whatever the menu offers, no message goes out)
            self.planner_on = False
            return False
        self.messages.append({"role": "assistant", "content": "I will ask the shopper about " + ", then ".join(l[0] for l in labels) + ".", "tool_calls": calls})
        self.asked_item, self.asked_property = labels[0][2], labels[0][0]
        replies = self.run_calls(calls, planned=True)
        self.asked_item, self.asked_property = None, None  # the reply is read: a later message is not about that product unless it says so
        for index, reply in enumerate(replies):
            call_id, text = reply
            if index == len(replies) - 1:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            if index < len(labels) and not reply.held_back:
                which = f" (product {labels[index][2]})" if self.multi() and labels[index][2] is not None else ""
                self.asked.append(f"- {labels[index][0]}{which}: asked \"{labels[index][1]}\" -> {text.split(chr(10) + '[status]')[0][:400]}")
        self.last_ask_step, self.last_ask_facts, self.ask_due = self.step, fact_sets, False
        self.planner_on = self.messages_used < limit
        return True

    def next_product(self) -> int:
        """The product a multi-product order's next question is about: one whose hidden need no answer has revealed yet
        before one whose need is known, then the one asked about least, one the request hints at, one whose listings are
        opened (questions are scarce: each spent on a product already answered or never chosen is lost)."""
        asked: dict[int, int] = {}
        for item, _ in self.asked_pairs():
            if item is not None:
                asked[item] = asked.get(item, 0) + 1
        hints = [set(w for w in _words(h) if len(w) >= 3) for h in self.must_learn()]

        def hinted(index: int) -> bool:
            named = {w for w in _words(self.product(index)) if len(w) >= 3}
            return any(named & hint for hint in hints)
        return min(range(len(self.items)), key=lambda i: (self.needs_found.get(i, 0) > 0, asked.get(i, 0), not hinted(i), i not in self.recorded, i))

    def asked_pairs(self) -> list[tuple[int | None, str]]:
        """(product, property) of each planned question the shopper answered: a reply asking for one matter at a time answers
        nothing, so a property whose only question drew one is not asked yet."""
        found = []
        for line in self.asked:
            match = re.match(r"- (.*?)(?: \(product (\d+)\))?: asked", line)
            if match:
                found.append((int(match.group(2)) if match.group(2) else None, match.group(1)))
        for item, prop, _ in self.asked_open:
            same = [pair for pair in found if pair[0] == item and _same_property(pair[1], prop)]
            if len(same) == 1:
                found.remove(same[0])
        return found

    def shelf_name(self, index: int) -> str:
        """A product of a multi-product order the way the store names it (a shopper's topic names each product by its shelf):
        its shelf from the setup, else the label all its opened listings carry that the most listings seen in the store carry
        (a shelf holds many listings, a seller or a title few; two at least, not every one seen, and no opened listing of
        another product); the request's own words for it while no one label stands out."""
        if self.items[index].get("category"):
            return self.items[index]["category"]
        labels = lambda i: (self.catalog.get(i) or {}).get("labels") or set()
        own: list[set[str]] = []
        others: set[str] = set()
        for listing in self.listings():
            product = None if listing in self.pointer_pages else self.product_of(listing)
            if product == index:
                own.append(labels(listing))
            elif product is not None:
                others |= labels(listing)
        seen = [labels(i) for i in self.catalog]
        counts = {label: sum(1 for found in seen if label in found) for label in (set.intersection(*own) - others if own else set())}
        counts = {label: n for label, n in counts.items() if 2 <= n < len(seen)}
        ranked = sorted(counts.values(), reverse=True)
        return max(counts, key=lambda label: counts[label]) if ranked and (len(ranked) == 1 or ranked[0] > ranked[1]) else self.product(index)

    def listed_name(self, index: int | None, prop: str) -> tuple[str, str]:
        """The name the product's opened listings record a property under (the most recorded one naming it) and one value of it."""
        values = self.menu_values.get(index if index is not None else 0, {})
        key = next((k for k in values if _same_name(k, prop)), None) or next((k for k in values if _same_property(k, prop)), None)
        return (key, next((v for v in values[key] if len(v) <= MENU_PART_CHARS), "")) if key else (prop, "")

    def plan_questions(self, count: int, menu: str, target: int | None = None) -> list[dict[str, Any]]:
        prompt = PLAN_PROMPT.replace("QUESTION_COUNT", str(count)) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nPRODUCTS IN THE ORDER:\n" + self.products_text()
        if target is not None:
            prompt += f"\n\nTHE NEXT QUESTION IS ABOUT: product {target} ({self.product(target)}). Choose properties of this product only (its own listings are shown below); a question about the whole order is allowed only for something under MUST BE LEARNED FROM THE SHOPPER that is not a product property."
            menu = self.menu_sections.get(target, "")
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        if self.asked:
            prompt += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.asked)
        if self.must_learn():
            prompt += "\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(self.must_learn())
        prompt += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n" + (menu or "(none seen yet)")
        self.plan_menu = bool(menu)  # opened listings record properties to choose from
        samples = self.plan_samples(prompt)
        self.trace.append({"kind": "plan_questions", "result": samples[0] if samples else None, "samples": len(samples), "votes": [[str(q.get("property")) for q in (r.get("questions") or []) if isinstance(q, dict)] for r in samples]})
        planned = self.select_questions(samples, count, target)
        if target is not None and not planned:
            planned = self.select_questions(samples, count, None)  # nothing planned for that product: any product's question
        return planned

    def select_questions(self, samples: list[dict[str, Any]], count: int, target: int | None) -> list[dict[str, Any]]:
        """Up to `count` questions from the planner's samples (then from the menu): about the target product when one is set,
        one property each, none the request settles, none the shopper already answered for that product."""
        asked = self.asked_pairs()
        reopened = [(item, prop, text) for item, prop, text in self.asked_open if not any(i == item and _same_property(p, prop) for i, p in asked)]
        overlaps, narrower = self.narrower_first(target)
        chosen = narrower + _vote(samples) + [{"item": index, "property": name} for index, name in getattr(self, "menu_pairs", []) if target is None or index == target]
        planned: list[dict[str, Any]] = []
        for entry in chosen:
            prop, question = str(entry.get("property") or "").strip(), str(entry.get("question") or "").strip()
            index = entry.get("item") if isinstance(entry.get("item"), int) and not isinstance(entry.get("item"), bool) and 0 <= entry.get("item") < len(self.items) else None
            if target is not None and _words(prop):
                if index not in (None, target):
                    continue  # this round's question is about the target product
                index = target
            if not self.multi():
                index = None
            elif index is None and _words(prop):
                owners = {i for i, names in self.recorded.items() for name in names if _same_name(name, prop)}
                if len(owners) > 1:
                    continue  # recorded on several products' listings: the question must name one of them
                index = owners.pop() if owners else None  # recorded on none: a question about the whole order (such as its budget)
            if not question and (not _words(prop) or prop.lower() == "none" or any(_same_name(prop, r["property"]) for r in self.reqs_for(index))):
                continue
            if any(_same_property(prop, other) and (item in (None, index) or index is None) for item, other in [a for a in asked if not (entry in narrower and a in overlaps)] + [(p.get("item"), str(p.get("property") or "")) for p in planned]):
                continue
            named = self.shelf_name(index) if self.multi() and index is not None else None
            again = [text for item, other, text in reopened if item == index and _same_property(prop, other)] if _words(prop) else []
            if again:  # its one question drew no answer: once more, by the listings' own name for it and with one of its values
                prop, example = self.listed_name(index, prop)
                question = _ask_text(prop, named, example)
                if question in again:
                    continue  # the same words would draw the same reply
            elif _words(prop):
                question = _ask_text(prop, named)
            planned.append({"property": prop or question, "item": index, "question": question})
            if len(planned) >= count:
                break
        return planned

    def narrower_first(self, target: int | None) -> tuple[list[tuple[int | None, str]], list[dict[str, Any]]]:
        """After a one-at-a-time reply: the (product, property) pairs that drew it, and the product's other listed properties
        sharing a word with one of them, which ask a narrower matter (the same words mostly draw that reply again). One such
        question per reply: once asked, the choice is the planner's again."""
        try:
            overlaps = [(item, str(prop)) for item, prop in getattr(self, "overlapping", set())]
            shared = lambda a, b: any(_same_word(x, y) for x in _words(a) for y in _words(b))
            found: list[dict[str, Any]] = []
            for item, prop in overlaps:
                if (target is None or item in (None, target)) and not any(i == item and (i, p) not in overlaps and shared(p, prop) for i, p in self.asked_pairs()):
                    found += [{"item": item, "property": name} for name in self.menu_values.get(item if item is not None else 0, {}) if shared(name, prop) and not any(_same_name(name, p) for _, p in overlaps)]
            return overlaps, found
        except Exception as exc:
            self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return [], []

    def plan_samples(self, prompt: str) -> list[dict[str, Any]]:
        first = PLAN_FIRST_MENU_ROUND if self.plan_menu else PLAN_FIRST_ROUND
        draws = first if not self.asked else (PLAN_LATER_ROUND if len(self.asked) == 1 else PLAN_LATE_ROUND)
        if self.time_left() < WRAP_UP_SECONDS + VOTE_SECONDS:
            draws = draws[:1]
        backups = (JUDGE_MODEL, READ_MODEL) if not self.asked else (READ_MODEL, JUDGE_MODEL)  # a later question falls back on the fast model first
        if self.spent_out():  # past the stop every question is planned on the later questions' route, the fast model its fallback
            draws, backups = PLAN_LATER_ROUND, (READ_MODEL,)
        self.call_settings = SETTINGS_FIRST_PLAN if not self.asked else {}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(draws)) as pool:
                futures = [pool.submit(self.judge, prompt, JUDGE_TOKENS, [model] + [m for m in backups if m != model], t) for model, t in draws]
                results = [f.result() for f in futures]
        finally:
            self.call_settings = {}
        return [r for r in results if isinstance(r, dict)]

    def send_reply(self, text: str) -> bool:
        tool = self.roles.get("message")
        limit = self.message_limit()
        if not tool or self.messages_used >= limit or self.steps_left() <= self.th('repair') or self.time_left() < WRAP_UP_SECONDS:
            return False
        args = self.message_args(text)
        if not args:
            return False
        args, problems = _check(args, self.schemas[tool], "args")
        if problems:
            return False
        call = {"id": f"reply_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "I will answer the shopper's question.", "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        for index, (call_id, reply_text) in enumerate(replies):
            if index == len(replies) - 1:
                reply_text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text})
        return True

    def cart_amount(self, name: str, args: dict[str, Any]) -> int:
        properties = self.schemas[name].get("properties") or {}
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                value = args.get(key, spec.get("default", 1))
                return value if isinstance(value, int) and not isinstance(value, bool) else 1
        return 1

    def cart_target(self, name: str, args: dict[str, Any]) -> str | None:
        properties = self.schemas[name].get("properties") or {}
        return next((v.strip() for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)

    def cart_targets(self, name: str, args: dict[str, Any]) -> list[str]:
        """Listings a cart call puts into the cart (none for a call that takes things out)."""
        if name != self.roles.get("cart"):
            return []
        if self.basket_tool():
            lines = _arg_lines(args, set(self.catalog) | self.seen_ids)
            if len({i for i, _ in lines}) >= 2 and not self.multi():
                return []  # several products the plan did not see: not checked against the one product's list
            held = set(self.held_lines())
            return [i for i, _ in lines if i not in held]
        if self.add_signature is not None and self.cart_signature(name, args) != self.add_signature:
            return []
        target = self.cart_target(name, args)
        return [target] if target else []

    def several_products(self) -> bool:
        """The navigator itself put several different listings in the basket with one call: a multi-product order that the
        one-product checks and the endgame must leave whole (the order is placed as it stands)."""
        return not self.multi() and self.navigator_lines >= 2 and len(self.held_lines()) >= 2  # (a multi-product plan checks each line)

    def track_cart(self, name: str, args: dict[str, Any], observation: Any, by_code: bool = False) -> str:
        """Follow the cart through one cart call and keep the call as a format once it worked; returns a note when the count is off."""
        properties = self.schemas[name].get("properties") or {}
        signature = self.cart_signature(name, args)
        known = set(self.catalog) | self.seen_ids
        lines = _arg_lines(args, known)
        ok = _worked(observation)
        note = ""
        if name == self.roles.get("cart") and self.basket_tool():
            if ok:  # the call replaced the whole basket
                if not by_code:
                    self.navigator_lines = len({item for item, _ in lines})
                for item in self.held:
                    self.held[item] = 0
                self.cart_ids = [item for item, _ in lines]
                for item, count in lines:
                    self.held[item] = count
                    self.cart_state[item] = signature
                format_ = _learned(args, known)
                if format_ is not None:
                    self.formats["basket"] = {"tool": name, "args": format_}
                for listing, count in lines if (len(lines) == 1 or self.multi()) else []:
                    want = self.qty(self.item_of(listing))
                    if count is not None and count != want:
                        note = f"The cart now holds {count} unit(s) of {listing}; the order needs exactly {want}."
                        break
        else:
            if name == self.roles.get("cart"):
                if self.add_signature is None:
                    self.add_signature = signature
                adding = signature == self.add_signature
            else:
                adding = name != (self.formats.get("remove") or {}).get("tool")
            pairs = lines or [(v.strip(), None) for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()]
            targets = pairs if lines else pairs[:1]  # as e_w8: one target when no listing id is known
            self.last_echo = self.cart_echo(observation, [(t, c if c is not None else self.cart_amount(name, args)) for t, c in targets], adding) if ok else None
            if self.last_echo is not None and adding and name == self.roles.get("cart"):
                for target, count in targets:  # the basket the store shows tells whether the call set the count or added to it
                    self.learn_cart_counts(target, self.held.get(target) or 0, count if count is not None else self.cart_amount(name, args), self.last_echo.get(target, 0))
            echo = self.line_echo(observation, targets, adding, name, args) if ok and self.last_echo is None else None
            for value, _ in pairs if ok else []:  # a call the store refused leaves the cart as it was
                self.cart_state[value] = signature
                if not adding and echo and value == targets[0][0]:
                    continue  # the store says units of the line are left: it stays where it was
                self.cart_ids = [i for i in self.cart_ids if i != value] + ([value] if adding else [])
            for index, (target, count) in enumerate(targets):
                if not ok:
                    continue
                if adding:
                    known_count = self.held.get(target, 0)
                    amount = count if count is not None else self.cart_amount(name, args)
                    if name == self.roles.get("cart"):
                        self.cart_probe = (target, known_count or 0, amount, self.cart_version + 1)
                    if name == self.roles.get("cart") and (self.cart_counts == "set" or (self.cart_counts is None and self.mode_less_cart())):
                        self.held[target] = amount  # (both such stores seen set the count; a basket echo or cart view shows otherwise)
                    else:
                        self.held[target] = None if known_count is None else known_count + amount
                    if echo is not None and index == 0:
                        self.held[target] = echo  # the store's own count of the line now
                    want = self.qty(self.item_of(target))
                    if index == 0 and self.held[target] not in (None, want) and not (self.cart_counts == "set" and self.held[target] == 0):
                        note = f"The cart line for {target} now holds {self.held[target]} unit(s) ({self.count_rule()}); the order needs exactly {want}."
                else:
                    self.held[target] = echo if echo is not None and index == 0 else None
            format_ = _learned(args, known) if ok else None
            if format_ is not None:
                named = [action for action in ("add", "remove") if self.cart_modes.get(action) and _compact(self.cart_modes[action]) == signature]
                self.formats[named[0] if named else ("add" if adding else "remove")] = {"tool": name, "args": format_}
        if ok:
            self.cart_version += 1
            self.cart_changed_step = self.step
        return note

    def quantity_warning(self, name: str, args: dict[str, Any]) -> str:
        if self.basket_tool() and self.multi():
            for listing, count in _arg_lines(args, set(self.catalog) | self.seen_ids):
                index = self.product_of(listing)
                if count is not None and index is not None and count != self.qty(index):
                    return f"product {index} ({self.product(index)}) needs exactly {self.qty(index)} unit(s) of one listing, but this call puts {count} of {listing} in the cart."
            return ""
        if self.basket_tool():
            lines = _arg_lines(args, set(self.catalog) | self.seen_ids)
            if len(lines) == 1 and lines[0][1] is not None and lines[0][1] != self.quantity:
                return f"the request needs exactly {self.quantity} unit(s) of one listing, but this call puts {lines[0][1]} in the cart."
            return ""
        if self.add_signature is not None and self.cart_signature(name, args) != self.add_signature:
            return ""
        lines = _arg_lines(args, set(self.catalog) | self.seen_ids)  # the listing named and the count beside it, at any depth
        target = lines[0][0] if lines else self.cart_target(name, args)
        amount = lines[0][1] if lines and lines[0][1] is not None else self.cart_amount(name, args)
        held = self.held.get(target or "", 0) or 0
        want = self.qty(self.item_of(target) if target else None)
        if amount == 0 and self.mode_less_cart() and self.cart_counts != "add":
            return ""  # a count of 0 takes the line out (or, if the store only adds, changes nothing): the result will show which
        if self.cart_counts == "set":
            return "" if amount == want else f"the request needs exactly {want} unit(s) of one listing, but this call sets the line to {amount}."
        if held + amount == want:
            return ""
        if held and self.cart_counts is None and self.mode_less_cart():  # no result has shown yet whether this call sets or adds
            return "" if amount == want and self.last_echo is not None else f"this listing is already in the cart with {held} unit(s); if this store's cart call adds, it makes {held + amount}, and if it sets the line, {amount}; the order needs exactly {want} unit(s) of one listing. View the cart after it."
        if held:
            return f"this listing is already in the cart with {held} unit(s); adding {amount} more makes {held + amount}, but the order needs exactly {want} unit(s) of one listing."
        return f"the request needs exactly {want} unit(s) of one listing, but this call asks for {amount}."

    def cart_quantity_problem(self) -> str:
        if self.several_products() or (not self.multi() and self.view_version != self.cart_version):
            return ""
        if self.view_version == self.cart_version and not any(n != 0 for n in self.view_lines.values()):
            return "the latest cart view shows an empty cart: put your choice in the cart first, then order."
        lines = self.view_lines
        if self.multi():
            if self.view_version != self.cart_version:  # no fresh cart view: the lines the code's own cart state holds
                lines = {i: self.held.get(i) for i in self.held_lines()}
            per: dict[int, list[tuple[str, int | None]]] = {}
            for listing, count in lines.items():
                index = self.product_of(listing)
                if index is not None:
                    per.setdefault(index, []).append((listing, count))
            for index, entries in sorted(per.items()):
                if len(entries) > 1:
                    return f"the latest cart view holds {len(entries)} lines for product {index} ({self.product(index)}); keep ONE listing for it at {self.qty(index)} unit(s), take the others out, view the cart again, then order."
                listing, count = entries[0]
                if count is not None and count != self.qty(index):
                    return f"the latest cart view shows {count} unit(s) of {listing} for product {index} ({self.product(index)}), but it needs exactly {self.qty(index)} unit(s) of one listing. Fix that line, view the cart again, then order."
            missing = [index for index in range(len(self.items)) if index not in per]
            if missing and lines and all(isinstance(self.item_of(listing), int) for listing in lines):
                names = ", ".join(f"product {index} ({self.product(index)}, {self.qty(index)} unit(s))" for index in missing)
                last = "decline instead: an order without a line for every product never fits" if self.short_cart() else "send the order again as it stands"
                return f"the cart has no line for {names}; the order needs one line per product. Add your best listing for it, view the cart again, then order (if no listing of it can fit, {last})."
            basket = {index: entries[0][0] for index, entries in per.items()}
            total = self.basket_total(basket)
            if total > self.budget_value():
                cheaper = self.within_budget({index: (item, "") for index, item in basket.items()}, quiet=True)
                swaps = "; ".join(f"{item} for product {index}" for index, (item, _) in sorted(cheaper.items()) if item != basket.get(index))
                return f"the cart's lines cost {format(total, 'g')} together, over the budget of {format(self.budget_value(), 'g')} for the whole order." + (f" Cheaper fitting listings: {swaps}." if swaps else " Swap a line for a cheaper fitting listing.") + " Fix the cart, view it again, then order."
            return ""
        kept = [i for i, n in lines.items() if n != 0]
        if len(kept) > 1:  # one product is one line: a second line (a sold-out pick, a line a removal left behind) fails the order
            return f"the latest cart view holds {len(kept)} lines ({', '.join(kept)}), but the order needs exactly one line, the chosen listing at {self.quantity} unit(s). Take the other lines out{self.remove_hint()}, view the cart again, then order."
        chosen = self.cart_ids[-1] if self.cart_ids and self.cart_ids[-1] in lines else (next(iter(lines)) if len(lines) == 1 else None)
        count = lines.get(chosen) if chosen else None
        if count is None or count == self.quantity:
            return ""
        return f"the latest cart view shows {count} unit(s) of {chosen}, but the order needs exactly {self.quantity} unit(s) of one listing. Fix that line so it holds exactly {self.quantity} ({self.count_rule()}), view the cart again, then order."

    def remove_hint(self) -> str:
        return " (a cart call with a count of 0 takes a line out)" if self.mode_less_cart() and self.cart_counts != "add" else ""

    def audit_decline(self) -> str:
        if self.audited:
            return ""
        self.audited = True
        listings = self.listings()
        self.judge_values(list(listings), listings)
        states = {item: self.assess(item, listings, judge=False) for item in listings}
        fits = [item for item, a in states.items() if a and a["state"] == "fits"]
        if fits:
            return "The exact checks find that these opened listings meet every requirement: " + ", ".join(fits[:6]) + ". Re-check their stock and price, and buy the right one (the cheapest when the request asks for it); decline only if each one is out of stock or over the budget."
        unclear = [item for item, a in states.items() if a and a["state"] == "unclear"]
        if not unclear or self.time_left() < WRAP_UP_SECONDS + CHECK_SECONDS:
            return ""
        blocks = []
        for item in unclear[:8]:
            blocks.append(f"{item}: " + "\n".join(self.item_records(item)[-2:])[:2500] + "\nEXACT CHECKS: " + " | ".join(self.check_lines(states[item])))
        prompt = AUDIT_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
        prompt += "\n\nLISTINGS:\n" + "\n\n".join(blocks)
        result = self.judge(prompt, JUDGE_TOKENS)
        self.trace.append({"kind": "audit", "result": result})
        fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        if not fits:
            return ""
        return "An independent check thinks these listings may meet every requirement: " + ", ".join(fits) + (f" ({result.get('reason')})" if result.get("reason") else "") + ". Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."

    def neutral_args(self, name: str, text: str, mode: dict[str, Any] | None = None) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        properties = schema.get("properties") or {}
        args: dict[str, Any] = dict(mode or {})
        for key in schema.get("required") or []:
            spec = _branch(properties.get(key))
            if key in args:
                continue
            if "enum" in spec and spec["enum"]:
                args[key] = spec["enum"][0]
            elif "string" in _kinds(spec):
                args[key] = text[: spec.get("maxLength", len(text))]
            elif spec.get("type") == "boolean":
                args[key] = True
            elif spec.get("type") in ("integer", "number"):
                args[key] = max(1, spec.get("minimum", 1))
        for key, spec in properties.items():
            if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec and len(properties) == 1:
                args[key] = text[: spec.get("maxLength", len(text))]
        return args

    def clean_decline(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        """The structural check of a decline's text: no listing id when the instructions forbid naming listings (or, when they
        say nothing, when one slipped in); untouched when they ask for one to be named."""
        rule = self.no_names("decline")
        rule = True if self.fallback.get("closest") == "forbidden" else (False if rule is None and self.fallback.get("closest") == "wanted" else rule)
        if rule is False:
            return self.name_closest(args, name) if self.fallback.get("closest") == "wanted" else args
        written, self.remark_written = self.remark_written, None
        dirty = False
        for key, value in args.items():
            if isinstance(value, str):
                for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value):
                    if token and token in self.seen_ids:
                        dirty = True
        if written and not dirty:
            return args  # the remark the instructions shaped names no listing: it stays
        if rule or dirty:
            neutral = DECLINE_NOTE
            cleaned = {}
            properties = (self.schemas.get(name) or {}).get("properties") or {}
            for key, value in args.items():
                spec = properties.get(key) or {}
                if isinstance(value, str) and "enum" not in spec:
                    cleaned[key] = neutral[: spec.get("maxLength", len(neutral))]
                else:
                    cleaned[key] = value
            return cleaned
        return args

    def name_closest(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        """A decline where the request asks for the closest listing says which one, unless its text already names one."""
        item = self.closest_named()
        slot = self.remark_slot("decline", args)
        if not item or not slot or any(token in self.seen_ids for value in args.values() if isinstance(value, str) for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value)):
            return args
        said = f" The closest listing is {item}."
        room = ((self.schemas.get(name) or {}).get("properties") or {}).get(slot, {}).get("maxLength")
        text = str(args.get(slot) or DECLINE_NOTE)
        return dict(args, **{slot: (text[: max(0, room - len(said))] if room else text) + said})

    def force_finish(self) -> None:
        """The last close, in rounds: each one decides again from what the store last said, so a refusal starts the next
        round; they end when the store takes a close, when nothing more can be sent, or after CLOSE_ROUNDS."""
        self.gate_holds = max(self.gate_holds, GATE_HOLDS)  # a forced finish never waits
        for number in range(CLOSE_ROUNDS):
            if self.finished or self.dead or self.steps_left() <= 0 or self.time_left() < LAST_CALL_SECONDS:
                return
            refusals = self.refusals
            repaired = number < CLOSE_ROUNDS - 1 and self.repair_close()  # (the last round always sends)
            if not repaired:
                self.final_close()
            if self.refusals == refusals and not repaired:
                return

    def repair_close(self) -> bool:
        """After a refused close, or with a whole cart over the budget or breaking a basket rule, the code's own commit chooses
        and repairs the basket while steps allow (store calls; its checks keep their spend guards). True when it used a step
        (a commit whose calls were all held back has done nothing, and the round's send still goes)."""
        if not self.roles.get("order") or self.steps_left() < self.th('last') or self.time_left() < LAST_CALL_SECONDS * 4:
            return False
        try:
            lines = [(i, self.held.get(i) or self.qty(self.item_of(i))) for i in self.viable_lines()]
            basket = {index: i for i in self.viable_lines() for index in [self.product_of(i)] if index is not None}
            whole = len({i for i, _ in lines}) >= len(self.items)
            if not self.refusals and not (lines and whole and (self.basket_fault(lines) or self.basket_problems(basket))):
                return False  # (a short cart is the forced finish's own to fill)
            step = self.step
            return self.commit_now() and self.step != step
        except Exception as exc:
            self.trace.append({"kind": "commit_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return False

    def legal_call(self, name: str, args: dict[str, Any], said: str) -> bool:
        """One call of a forced finish, by the legal path of every finish (the gates ran with final=True first), so its result
        is recorded and read like any other. False when no result came."""
        call = {"id": f"forced_{self.step}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": said, "tool_calls": [call]})
        self.committing = self.order_checked = True
        self.code_declining = self.finish_kind(name, args) == "decline"
        try:
            replies = self.close_calls([call])
        except Exception as exc:
            self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            replies = []
        finally:
            self.committing = self.code_declining = False
        for reply in replies:
            self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})
        self.trace.append({"kind": "forced_finish", "action": {"name": name, "args": args}, "reply": _compact(replies)[:500]})
        return bool(replies) and not any(reply.held_back for reply in replies)

    def final_send(self, action: dict[str, Any], kind: str) -> None:
        """The forced finish's one send: its due instructions, then the legal path; a call the screen holds back goes straight
        to the store (a forced finish has no other turn)."""
        self.gate(kind, True, final=True)  # due instructions first, as on every finish
        if self.finished or self.steps_left() <= 0 or self.refused_out(kind):
            return
        if self.legal_call(action["name"], action["args"], "I will finish the task now."):
            return
        if self.finished or self.dead or self.steps_left() <= 0:
            return
        action = dict(action, args=self.with_remark(kind, action["name"], action["args"]))
        if kind == "decline":
            action["args"] = self.clean_decline(action["args"], action["name"])
        version, key = self.ledger_version, self.close_key(kind)
        reply = self.env_call([action])
        self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        self.read_reply(reply)
        if not self.finished and reply:
            self.refusals += 1
            self.close_refused[key] = self.close_refused.get(key, 0) + 1
            self.reopen_close(_event_texts(reply.get("observation")), version, key)

    def final_close(self) -> None:
        """One round of the forced finish: the decision on the basket the cart can hold in the steps left, then its send."""
        order = self.roles.get("order")
        decline = self.roles.get("decline")
        if any((self.catalog.get(i) or {}).get("stale") for i in self.held_lines()):
            self.reread_stock([])  # a cart line's price changed unread: the basket is judged at the price now shown (a store call, no model)
        viable = self.viable_lines()
        item: str | None = None
        decision = self.decide_finish({})
        if decision["action"] == "decline":
            self.trace.append({k: v for k, v in decision.items() if k not in ("action", "text")})
            self.final_send({"name": decline, "args": self.finish_args("decline")}, "decline")
            return
        placed = {self.item_of(i) for i in viable}
        missing = [index for index in range(len(self.items)) if index not in placed]
        if order and missing and None not in placed and self.steps_left() >= (1 if self.order_takes_item() else 2):  # an order that names the item needs no cart step
            if not self.stating_opened and self.steps_left() >= self.th('repair') + 1:
                try:
                    self.open_stating({index: self.best_candidate(index=index, last_resort=True) for index in missing})  # (no relaxing yet)
                except Exception as exc:
                    self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            try:
                chosen = {index: self.best_candidate(relax=True, index=index, last_resort=True)[0] for index in missing}
                chosen = {index: listing for index, listing in chosen.items() if listing}
                unfilled = [index for index in missing if index not in chosen]
                if unfilled:  # no opened listing: the best row seen, where no decline is proven
                    chosen.update(self.best_effort_fill(unfilled, {**{self.product_of(i): i for i in viable if self.product_of(i) is not None}, **chosen}))
                basket = {**{index: i for i in viable for index in [self.product_of(i)] if index is not None}, **chosen}
                fixed = self.basket_rules(basket, {index: (listing, "near fit") for index, listing in basket.items()})[0]
                chosen = {index: fixed.get(index, listing) for index, listing in chosen.items()}  # (only the lines still to place can move)
                basket = {**basket, **chosen}
                item = chosen[min(chosen)] if chosen else None
                plan = self.cart_plan(basket) if chosen else None
                if chosen and (plan is None or len(plan) >= self.steps_left()):
                    plan = self.cart_plan(basket, fix_counts=False)  # no room to put a held count right
                if chosen and (plan is None or len(plan) >= self.steps_left()):
                    plan = self.cart_plan(basket, removals=False)  # no room (or no call) to take other lines out: add the picks and order
            except Exception:
                item, plan, chosen = None, None, {}
            if chosen and plan is not None and len(plan) < self.steps_left():
                if self.held_lines():
                    self.gate("cart_change", True, final=True)
                for call in plan:
                    version = self.cart_version
                    if not self.legal_call(call["name"], call["args"], "I will put the cart right first.") or self.cart_version == version:
                        break
                else:
                    viable = viable + [listing for listing in chosen.values() if listing not in viable]
        finish = self.order_call(viable[0] if viable else item) if order and (viable or (self.cart_used and not self.cart_ids)) else None
        lines = [(i, self.held.get(i) or self.qty(self.item_of(i))) for i in viable]
        # every listing meeting a product's requirements is over the budget or sold out, the cart misses a product or costs more
        # than the budget, or the store refused it as it stands: such an order never fits, a decline may
        decision = self.decide_finish() if finish else {"action": "order"}
        if finish and decision["action"] == "order" and viable:
            decision = self.decide_finish(lines=lines)
        if decision["action"] == "decline":
            finish = None
            self.trace.append({k: v for k, v in decision.items() if k not in ("action", "text")})
        if finish:
            action = finish
        elif decline:
            action = {"name": decline, "args": self.finish_args("decline")}
        else:
            return
        kind = self.finish_kind(action["name"], action["args"])
        self.final_send(action, kind)

    def note_observation(self, name: str, observation: Any, refused: str | None = None) -> str:
        notes = []
        if isinstance(observation, dict):
            _collect_ids(observation, self.seen_ids)
            if observation.get("error") == "budget_exceeded":
                if name == self.roles.get("message") or self.message_budget is not None:
                    self.set_budget("messages", self.messages_used if self.message_budget is None else min(self.message_budget, self.messages_used), "store refusal")
                notes.append("That budget is used up; do not call this tool again.")
            if refused:
                notes.append(f"THE {refused.upper()} WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (remove any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again.")
            elif len(observation) >= 2 and name != self.roles.get("message"):
                lines = _event_texts(observation)
                if lines and all(line in self.heard for line in lines):
                    notes.append("This result repeats a message from the shopper that was already read and applied above; it asks for nothing new.")
                elif lines:
                    notes.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(lines[:4]))
        elif isinstance(observation, list):
            _collect_ids(observation, self.seen_ids)
        return " ".join(notes)

    # ------------------------------------------------------------------ the instruction ledger, the reserve and the gate
    # Every procedural instruction (from the request, a shopper line or a store notice) is a record; the code decides when it
    # falls due (before an order, a decline, a cart change, any finish) and keeps a message for it; the executor, one bounded
    # model call, decides what carrying it out means now and returns one action the code validates and sends; the shopper's
    # reply settles it for the cart as it is. No instruction's wording is known to the code.

    def add_instructions(self, items: Any, source: str, text: Any) -> list[str]:
        added: list[str] = []
        known = {i["quote"].lower() for i in self.instructions}
        for raw in items[:12] if isinstance(items, list) else []:
            ins = _instruction(raw, text, source)
            if not ins or ins["quote"].lower() in known:
                continue
            known.add(ins["quote"].lower())
            ins["id"] = len(self.instructions) + 1
            if ins["tool"] and ins["tool"] not in self.schemas:
                ins["tool"] = self.roles.get(ins["tool"])  # a role named instead of a tool, else no tool
            ins["tool_base"] = self.tool_calls.get(ins["tool"], 0) if ins["tool"] else 0
            self.instructions.append(ins)
            self.instruction_version += 1
            if ins["limit"] == "at_most" and ins["count"] and ins["counts"] in ("message", "step"):
                self.apply_allowance({"messages_total": ins["count"]} if ins["counts"] == "message" else {"steps_total": ins["count"]}, source)
                ins.update(status="satisfied", done_for="budget")
            added.append(ins["quote"][:120])
            self.trace.append({"kind": "instruction", "id": ins["id"], "source": source, "when": ins["when"], "message": ins["message"],
                               "wait_reply": ins["wait_reply"], "count": ins["count"], "quoted": ins["quoted"], "quote": ins["quote"][:200]})
        return added

    def apply_allowance(self, allowance: dict[str, Any], source: str) -> str:
        """A new limit on messages or steps a shopper line or a store notice states: it binds from now on, up or down."""
        num = lambda v: int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else None
        before = (self.message_budget, self.max_steps, self.budgets["time"]["total"])
        total, left = num(allowance.get("messages_total")), num(allowance.get("messages_left"))
        if total is not None:
            self.set_budget("messages", max(self.messages_used, total), source)
        elif left is not None:
            self.set_budget("messages", self.messages_used + left, source)
        total, left = num(allowance.get("steps_total")), num(allowance.get("steps_left"))
        if total is not None and total > 0:
            self.set_budget("steps", min(self.policy_steps, max(self.step + 1, total)), source)
        elif left is not None and left > 0:
            self.set_budget("steps", min(self.policy_steps, self.step + left), source)
        seconds = num(allowance.get("seconds_left"))
        if seconds is not None and seconds > 0:  # never past the platform's own time
            self.set_budget("time", min(TASK_SECONDS, time.monotonic() - self.started + seconds), source)
        if (self.message_budget, self.max_steps, self.budgets["time"]["total"]) == before:
            return ""
        if self.message_budget != before[0] and self.messages_used >= self.message_limit():
            self.planner_on = self.planner_on and self.messages_used < self.message_limit()
        self.trace.append({"kind": "allowance", "source": source, "messages": [before[0], self.message_budget], "steps": [before[1], self.max_steps]})
        return f"new allowance ({source}): messages {self.message_budget if self.message_budget is not None else 'unstated'}, steps {self.max_steps}"

    def must_learn(self) -> list[str]:
        """What the planner must learn from the shopper (a derived view): the request's instructions to find something out
        before choosing."""
        return [i["content"] or i["quote"] for i in self.instructions if i["when"] == "at_start" and i["message"] is True and i["status"] != "impossible"]

    def no_names(self, action: str = "decline") -> bool | None:
        """Whether the instructions for this finish forbid naming listings in what it says (True), ask for one (False), or say
        nothing about it (None) - a derived view of the ledger."""
        said = [i["names"] for i in self.instructions if i.get("names") and self.instruction_applies(i, action)]
        return True if "forbidden" in said else (False if "required" in said else None)

    def basket_key(self) -> tuple:
        """The cart as the shopper would be told it: its lines, their counts and the prices known now."""
        return tuple(sorted((i, self.held.get(i), (self.catalog.get(i) or {}).get("price")) for i in self.held_lines()))

    def instruction_mark(self, action: str) -> Any:
        if action == "decline":
            return "decline"
        if action == "cart_change":
            return ("cart", self.cart_version)
        return self.basket_key()

    def instruction_mark_for(self, ins: dict[str, Any], action: str) -> Any:
        """What a discharge of this instruction is done for: the change it follows, the message it precedes, else the action's."""
        if ins["when"] == "after_change":
            return ("change", self.change_count)
        if ins["when"] == "before_message":  # the next message that is not the gate's own
            return ("message", sum(1 for m in self.message_log if m["by"] != "gate"))
        if ins["when"] in ("at_start", "any"):
            return "once"  # carried out once for the task
        return self.instruction_mark(action)

    def instruction_applies(self, ins: dict[str, Any], action: str) -> bool:
        when = ins["when"]
        if when == "after_change":
            return self.change_count > 0  # at the next action of any kind once something changed
        if action == "message":
            return when == "before_message"
        if action == "cart_change":
            return when == "before_cart_change"
        if action == "order":
            return when in ("before_order", "before_finish", "at_start", "any")
        if action == "decline":
            return when in ("before_decline", "before_finish", "at_start", "any")
        return False

    def reserved_messages(self) -> int:
        """Messages kept for instructions that will need one before the order, and are not done for the cart as it is."""
        if not self.roles.get("message"):
            return 0
        # (only instructions due before the order keep a message from the questions: one due "throughout" or at the start is
        # usually met by the questions themselves, and one due only at a decline would take a question from every task)
        need = sum(1 for i in self.instructions if i["message"] is True and ((i["when"] in ("before_order", "before_finish", "after_change") and self.owed(i, "order"))
                                                                              or (i["when"] == "before_message" and self.owed(i, "message"))))
        return max(0, min(need, self.message_total() - self.messages_used))

    def instruction_line(self, ins: dict[str, Any], short: bool = False) -> str:
        when = {"before_order": "before the order", "before_decline": "when declining", "before_finish": "before the finish",
                "before_cart_change": "before a cart change", "before_message": "before a message", "after_change": "after a change",
                "at_start": "before choosing", "any": "throughout"}.get(ins["when"], ins["when"])
        if ins["source"] != "request":
            when += f", from the {ins['source']}"
        done = ins["status"] == "satisfied" and ins.get("done_for") in (self.instruction_mark_for(ins, "order"), self.instruction_mark_for(ins, "decline"), "budget")
        state = "done for this cart" if done else ("not possible: " + ins.get("why", "")[:60] if ins["status"] == "impossible" else "pending")
        how = ", carried out for you right before the finish" if ins["message"] and ins["when"] != "before_cart_change" else ""
        return f"{ins['quote'][:90 if short else 300]} ({when}{how}; {state})"

    def owed(self, ins: dict[str, Any], action: str) -> bool:
        """The one rule for an instruction still to carry out before this action: it applies to it, is not given up, not done
        for the cart (or decline) as it is now, is not a limit (kept at the calls themselves: call_cap, the budgets), and is
        not a standing constraint that needs no message (one that holds throughout: shown in the status, kept by the navigator)."""
        if ins["when"] == "at_start" and ins["message"] is True and any(m["by"] == "planner" for m in self.message_log):
            return False  # the planner asks what is to be learned first (must_learn); the gate is its safety net only
        return (ins["status"] != "impossible" and self.instruction_applies(ins, action) and ins.get("done_for") not in (self.instruction_mark_for(ins, action), "budget")
                and ins.get("limit") != "at_most" and not (ins["when"] in ("any", "at_start") and ins["message"] is not True and ins.get("limit") != "at_least"))

    def due_instructions(self, action: str) -> list[dict[str, Any]]:
        return [i for i in self.instructions if self.owed(i, action)]

    def question_ready(self) -> bool:
        """A planned question could go out now (the same limits ask_planned keeps, without planning one)."""
        return bool(self.planner_on and self.roles.get("message") and self.messages_used < self.message_limit() and self.late_room()
                    and self.steps_left() > min(self.th('questions_end'), max(4, self.max_steps // 4)) and self.time_left() >= WRAP_UP_SECONDS + QUESTION_SECONDS)

    def counted(self, counts: str) -> int:
        """How many of what a counted instruction counts the task has done (the code's own calls included)."""
        tools = {"search": {self.roles.get("search")}, "cart": self.cart_tools(), "message": {self.roles.get("message")}}.get(counts)
        if counts == "open":
            return len(self.details_opened)
        if counts == "compare":
            return len(self.compared)
        if counts == "step":
            return self.step
        return sum(self.tool_calls.get(t, 0) for t in tools or () if t)

    def counted_tools(self, counts: str) -> set[str]:
        return {"open": {self.roles.get("details")}, "compare": {self.roles.get("view")}, "search": {self.roles.get("search")},
                "cart": self.cart_tools(), "message": {self.roles.get("message")}}.get(counts, set()) - {None}

    def call_cap(self, name: str) -> str:
        """Why one more call of this tool would break an "at most" instruction ("" when it would not)."""
        for ins in self.instructions:
            if ins.get("limit") == "at_most" and ins.get("count") and ins.get("counts") not in (None, "message", "step") \
                    and name in self.counted_tools(ins["counts"]) and self.counted(ins["counts"]) >= ins["count"]:
                return f"the instruction \"{ins['quote'][:160]}\" allows at most {ins['count']} of these, and that many are done."
        return ""

    def instructions_owed(self) -> int:
        """Instructions still to carry out before the order that need a message or a tool (their spend is kept past the stop).
        A standing check (no message, no tool, such as an availability check) sends nothing and reads no reply: its executor
        calls are kept apart (checks_owed)."""
        return sum(1 for i in self.instructions if self.owed(i, "order") and (i.get("message") is True or i.get("tool"))
                   and not (i.get("counts") and i.get("count") and i.get("limit") == "at_least"))

    def standing_check(self, ins: dict[str, Any]) -> bool:
        """An instruction the gate's executor decides without a message or a tool (such as an availability check); one the code
        counts itself asks no model."""
        return ins.get("message") is not True and not ins.get("tool") and not (ins.get("counts") and ins.get("count") and ins.get("limit") == "at_least")

    def checks_owed(self) -> int:
        """Standing checks still to carry out before the order: each is decided by executor calls (recorded: $0.0004
        per check, p90 $0.0016), kept in the core reserve like the instructions that need a message or a tool."""
        return sum(1 for i in self.instructions if self.owed(i, "order") and self.standing_check(i))

    def counted_discharge(self, ins: dict[str, Any], mark: Any, final: bool) -> tuple[str, str]:
        """An "at least" instruction the code counts itself: met, held for the navigator while it makes progress, or given up."""
        done, need = self.counted(ins["counts"]), ins["count"]
        if done >= need:
            ins.update(status="satisfied", done_for=mark)
            return "ok", ""
        short = need - done
        if final or ins.get("held_at") == done or self.steps_left() <= short + self.commit_steps_needed():
            return self.give_up(ins, f"{done} of {need} done")
        ins["held_at"] = done
        tools = ", ".join(sorted(self.counted_tools(ins["counts"]))) or "the tools"
        return "hold", f"The instruction \"{ins['quote'][:200]}\" needs at least {need}; {done} are done so far, so do {short} more (with {tools}) before this finish."

    def gate(self, action: str, planned: bool, final: bool = False) -> str:
        """Every order, decline and (when an instruction names it) cart change passes here, from the navigator, the code's
        commit or the forced finish: the instructions due for it are carried out first. Returns why the action waits, or ""
        to let it go. A forced finish (final) never waits."""
        if action not in ("order", "decline", "cart_change", "message") or self.key_out():
            return ""
        due = self.due_instructions(action)
        if action in ("order", "decline") and self.catch_all_done != self.instruction_version and self.time_left() > FORCE_SECONDS + ONE_CALL_SECONDS:
            due += [i for i in self.catch_all(action, due) if i not in due]
        final = final or self.gate_holds >= GATE_HOLDS  # past the cap no finish is held back again
        notes: list[str] = []
        # work first, then telling, then asking for the yes; the standing checks last, so the reserve never runs out on a message
        due.sort(key=lambda i: (self.standing_check(i), i["message"] is True, bool(i["wait_reply"])))
        for ins in due:
            outcome, note = self.discharge(ins, action, planned or final, final)
            if note:
                notes.append(note)
            if outcome in ("changed", "hold") and not final:
                if planned:
                    self.commit_paused = True  # not a failed commit: the code chooses again with what the shopper said
                    if outcome == "changed":
                        self.ready_blocked = (self.basket_key(), self.ledger_version)  # the navigator works on it first
                self.gate_holds += 1
                self.trace.append({"kind": "gate_hold", "action": action, "instruction": ins["id"], "outcome": outcome})
                return "Not run yet (no step used). " + " ".join(notes)
        if due:
            self.trace.append({"kind": "gate_pass", "action": action, "instructions": [i["id"] for i in due]})
        return ""

    def discharge(self, ins: dict[str, Any], action: str, by_code: bool, final: bool = False) -> tuple[str, str]:
        """Carry out one due instruction: ("ok" | "changed" | "hold" | "impossible", a note for the navigator). by_code: the
        code's own finish (it cannot do the navigator's work); final: a forced finish (nothing may wait)."""
        mark = self.instruction_mark_for(ins, action)
        if ins.get("counts") and ins.get("count") and ins.get("limit") == "at_least":
            return self.counted_discharge(ins, mark, final)  # the code counts; no model is asked
        if ins.get("tool") and self.tool_calls.get(ins["tool"], 0) > ins.get("tool_base", 0) and not ins["message"]:
            ins.update(status="satisfied", done_for=mark)  # the tool it requires was used
            return "ok", ""
        for _ in range(INSTRUCTION_TRIES):
            if self.time_left() < FORCE_SECONDS + ONE_CALL_SECONDS or self.steps_left() < self.th('last'):
                return self.give_up(ins, "no time or steps left")
            act = self.execute(ins, action)
            kind = act.get("action") if act else None
            self.trace.append({"kind": "executor_action", "instruction": ins["id"], "action": kind, "detail": {k: act.get(k) for k in ("text", "tool", "why")} if act else None})
            if kind == "none":
                ins.update(status="satisfied", done_for=mark)
                return "ok", ""
            if kind == "hold":
                if by_code or ins.get("held_for") == mark:  # once per cart: the navigator heard it and chose to finish
                    ins.update(status="satisfied", done_for=mark)
                    return "ok", ""  # only the navigator can do the missing work; the code's own finish goes on
                ins["held_for"] = mark
                return "hold", f"Before this finish, the instruction \"{ins['quote'][:200]}\" needs more first: {str(act.get('why') or '')[:300]}"
            if kind == "remark" or (kind == "say" and ins["message"] is False):
                # what the finish itself says (a decline's remark): kept for the finish call, which the structural check still
                # reads; with no text argument on the finish, or for another action, nothing is sent
                text = " ".join(str(act.get("text") or "").split())
                if text and action in ("order", "decline") and self.remark_slot(action):
                    self.pending_remark[action] = text
                ins.update(status="satisfied", done_for=mark)
                return "ok", ""
            many = isinstance(act, dict) and (str(act.get("text") or "").count("?") > 1 or (isinstance(act.get("matters"), int) and not isinstance(act.get("matters"), bool) and act["matters"] > 1))
            if kind == "say" and self.one_at_a_time_seen and many and not ins.get("narrowed_for") == mark:
                ins["narrowed_for"] = mark  # the shopper answers one matter at a time: the executor writes it once more, narrower
                ins["tries"].append({"step": self.step, "text": str(act.get("text"))[:300], "reply": "(not sent: more than one matter, and this shopper answers one matter per message)", "ack": "none", "mark": None})
                continue
            call = self.instruction_call(act)
            if call is None:  # no answer (the call cap, the reserve, an unreadable answer) or no valid action
                if not by_code and ins["message"] and self.roles.get("message") and self.messages_used < self.message_total() and ins.get("window_for") != mark:
                    ins.update(window=True, window_for=mark)
                    self.message_window = True
                    return "hold", f"The instruction \"{ins['quote'][:200]}\" must be carried out before this finish: send the shopper that message yourself now (one message), read the reply, then finish again."
                return self.give_up(ins, "no valid action")
            version, before = self.ledger_version, len(self.message_log)
            self.last_ack = None
            self.discharging = True
            try:
                replies = self.in_phase("reserve", lambda: self.run_calls([call], planned=True))
            finally:
                self.discharging = False
            shown = " ".join(r[1].split("\n[status]")[0][:400] for r in replies)
            if any(r.held_back for r in replies):
                return self.give_up(ins, shown[:120])
            sent = call["function"]["name"] == self.roles.get("message") and len(self.message_log) > before
            if not sent:
                ack = "none"
            elif self.last_ack is not None:
                ack = self.last_ack  # this reply was read now
            else:  # every line of it was heard before (a repeated reply is not read again): what those words meant then
                ack = next((self.ack_lines.get(line) for line in self.last_heard if self.ack_lines.get(line) in ("accepted", "refused")), "none")
            ins["tries"].append({"step": self.step, "text": act.get("text") or _compact(act.get("args"))[:300], "reply": shown[:300], "ack": ack, "mark": mark if sent else None})
            note = f"[before this {action.replace('_', ' ')} the code carried out the instruction \"{ins['quote'][:120]}\": {call['function']['name']} -> {shown[:500]}]"
            if ack == "refused" or self.ledger_version != version:
                return "changed", note + " The shopper's answer changes something: read it, fix the choice or the cart, then finish again."
            if ins["wait_reply"] and sent and ack != "accepted":
                if sum(1 for t in ins["tries"] if t.get("mark") == mark) >= 2:
                    break  # two messages without a clear yes: given up for this cart (the order still goes)
                continue  # no clear yes: the executor writes it once more, more plainly
            if not sent:
                continue  # a look-up the instruction needed: the executor decides again with its result
            ins.update(status="satisfied", done_for=mark)
            return "ok", note
        return self.give_up(ins, "no clear answer from the shopper")

    def give_up(self, ins: dict[str, Any], why: str) -> tuple[str, str]:
        ins.update(status="impossible", why=why)
        self.trace.append({"kind": "instruction_status", "instruction": ins["id"], "status": "impossible", "why": why})
        return "impossible", ""

    def read_reply(self, reply: Any) -> None:
        """The listener reads what a call the code sent on its own (a forced finish's cart call or finish) brought back."""
        observation = (reply or {}).get("observation") if isinstance(reply, dict) else None
        lines = _event_texts(observation) if observation is not None else []
        if lines and not self.finished and self.time_left() > FORCE_SECONDS / 2:
            try:  # a forced finish never fails for a reading
                self.in_phase("reserve", lambda: self.interpret(lines, None))
            except Exception as exc:
                self.trace.append({"kind": "read_error", "error": f"{type(exc).__name__}: {exc}"[:200]})

    def remark_slot(self, kind: str, args: dict[str, Any] | None = None) -> str | None:
        """The free-text argument of the tool that sends this finish (its remark or note), or None: a string argument with no
        allowed values, not a mode, not one the call fills with a listing id, and not a required one when an optional one exists."""
        name = self.roles.get("order" if kind == "order" else "decline")
        schema = self.schemas.get(name or "") or {}
        props, required = schema.get("properties") or {}, set(schema.get("required") or [])
        modes = set(self.order_args or {}) | set(self.decline_args or {})
        ids = set(self.catalog) | self.seen_ids
        slots = [k for k, v in props.items() if k not in modes and "string" in _kinds(_branch(v)) and "enum" not in _branch(v)
                 and not (args and str(args.get(k)) in ids) and not any(w in ("id", "ids") for w in _words(_key_name(k)))]
        optional = [k for k in slots if k not in required]
        slots = optional or slots
        return slots[0] if len(slots) == 1 else None

    def with_remark(self, kind: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A finish call with the remark the executor wrote for it (the code's own text stays only when none was written)."""
        text = self.pending_remark.pop(kind, None)
        slot = self.remark_slot(kind, args)
        if text and slot and name == self.roles.get("order" if kind == "order" else "decline"):
            spec = ((self.schemas.get(name) or {}).get("properties") or {}).get(slot) or {}
            args = dict(args, **{slot: text[: spec.get("maxLength", len(text))]})
            self.remark_written = slot
            self.trace.append({"kind": "finish_remark", "finish": kind, "text": text[:200]})
        return args

    def instruction_call(self, act: Any) -> dict[str, Any] | None:
        """The executor's action as one checked tool call, or None (also for no answer at all)."""
        if not isinstance(act, dict):
            return None
        if act.get("action") == "say":
            text = " ".join(str(act.get("text") or "").split())
            args = self.message_args(text) if text and self.roles.get("message") else None
            name = self.roles.get("message")
            if not args or self.messages_used >= self.message_total():
                return None
        elif act.get("action") == "call":
            name = act.get("tool")
            finishing = {self.roles.get("order"), self.roles.get("decline")} | self.cart_tools()
            if not isinstance(name, str) or name not in self.schemas or name in finishing or not isinstance(act.get("args"), dict):
                return None
            if name == self.roles.get("message") and self.messages_used >= self.message_total():
                return None
            args = act["args"]
        else:
            return None
        args, problems = _check(args, self.schemas[name], "args")
        if problems:
            return None
        return {"id": f"instruction_{self.step}_{len(self.message_log)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

    def instruction_state(self) -> str:
        lines = []
        for line in self.held_lines():
            price, count = (self.catalog.get(line) or {}).get("price"), self.held.get(line)
            lines.append(f"- {line}: count {count if count is not None else 'unknown'}, price {format(price, 'g') if isinstance(price, (int, float)) else 'unknown'} each")
        key = self.basket_key()
        total = sum((p or 0) * (n or 0) for _, n, p in key)
        known = all(isinstance(p, (int, float)) and n is not None for _, n, p in key)
        text = "THE CART NOW:\n" + ("\n".join(lines) or "(empty)") + (f"\nTOTAL: {format(total, 'g')}" if lines and known else "")
        listings = self.listings()
        pages = [self.listing_block(line, listings) for line in self.held_lines() if line in listings]
        if pages:
            text += "\n\nTHE CART LINES' PAGES:\n" + "\n".join(p[:1200] for p in pages[:5])
        if self.budget_now is not None:
            text += f"\nBUDGET (the whole order): {self.budget_now} {self.plan.get('currency') or ''}".rstrip()
        text += "\n\nMESSAGES SENT (oldest first):\n" + ("\n".join(f"- step {m['step']} ({'after' if m['cart'] == key else 'before'} the last cart change, by the {m['by']}): {m['text'][:300]} -> shopper: {m['reply'][:200]}" for m in self.message_log[-8:]) or "(none)")
        text += f"\n\nMESSAGES LEFT: {max(0, self.message_total() - self.messages_used) if self.roles.get('message') else 0}\nSTEPS LEFT: {self.steps_left()} (the finish itself takes one)"
        return text

    def execute(self, ins: dict[str, Any], action: str) -> dict[str, Any] | None:
        if self.executor_calls >= EXECUTOR_CALLS or not self.reserve_room():
            return None
        self.executor_calls += 1
        usable = [t for t in self.tools if t["function"]["name"] not in ({self.roles.get("order"), self.roles.get("decline")} | self.cart_tools())]
        prompt = EXECUTOR_PROMPT + f"\n\nTHE INSTRUCTION ({ins['source']}): {ins['quote']}" + (f"\nWHAT IT ASKS: {ins['content']}" if ins["content"] else "")
        prompt += "\nIT NEEDS THE SHOPPER'S ANSWER: " + ("yes" if ins["wait_reply"] else "no")
        if ins["message"] is False:
            prompt += "\nIT NEEDS NO MESSAGE: do not message the shopper for it (\"say\" is not allowed)"
        if ins.get("tool"):
            prompt += f"\nIT NEEDS THE TOOL: {ins['tool']}" + (" (already used)" if self.tool_calls.get(ins["tool"], 0) > ins.get("tool_base", 0) else " (not used yet)")
        if ins["when"] == "after_change" and self.last_change:
            prompt += "\nTHE LATEST CHANGE: " + " / ".join(self.last_change)[:400]
        if self.one_at_a_time_seen:
            prompt += "\nTHE SHOPPER ANSWERS ONE MATTER PER MESSAGE: a message carries only this instruction's matter, with at most one question"
        prompt += "\nTHE FINISH TAKES A TEXT: " + ("yes" if action in ("order", "decline") and self.remark_slot(action) else "no")
        if ins["tries"]:
            prompt += "\nEARLIER ATTEMPTS FOR IT:\n" + "\n".join(f"- {t['text'][:300]} -> {t['reply'][:200]}" for t in ins["tries"][-3:])
        prompt += f"\n\nTHE FINISH ABOUT TO HAPPEN: {'a change of the cart' if action == 'cart_change' else action}"
        prompt += "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "") + "\n\nPRODUCTS IN THE ORDER:\n" + self.products_text()
        prompt += "\n\nREQUIREMENTS NOW: " + self.ledger() + "\n\n" + self.instruction_state()
        prompt += "\n\nTOOLS:\n" + "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')} | arguments: {_compact(t['function'].get('parameters') or {})}" for t in usable)
        self.late_call = True  # an instruction's discharge is paid from the reserve, past the spend stop too
        try:
            answer = self.in_phase("reserve", lambda: self.judge(prompt, EXECUTOR_TOKENS, _route("executor_late" if self.spent_out() else "executor")))
        finally:
            self.late_call = False
        return answer if isinstance(answer, dict) and answer.get("action") in ("say", "call", "remark", "hold", "none") else None

    def catch_all(self, action: str, due: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Before the first finish, one bounded check over the ledger: which recorded instructions the code's timing did not
        make due are still undone for it. It may pick only open ledger entries: none given up, none done for this cart, no
        limit, none due only at the other finish. No call when there is nothing to pick."""
        other = "before_decline" if action == "order" else "before_order"
        open_ = [i for i in self.instructions if i not in due and i["status"] != "impossible" and i.get("done_for") not in (self.instruction_mark_for(i, action), "budget")
                 and i.get("limit") != "at_most" and i["when"] not in (other, "before_message") and not (i["when"] == "after_change" and not self.change_count)]
        if not open_ or self.executor_calls >= EXECUTOR_CALLS or not self.reserve_room():
            return []
        self.catch_all_done = self.instruction_version  # (again only when new instructions arrive)
        self.executor_calls += 1
        listed = "\n".join(f"{i['id']}. {i['quote'][:300]} (from the {i['source']}; " + ("due now" if i in due else i["status"]) + ")" for i in due + open_)
        prompt = CATCH_ALL_PROMPT + f"\n\nTHE FINISH ABOUT TO HAPPEN: {action}\n\nINSTRUCTIONS:\n{listed}\n\n" + self.instruction_state()
        self.late_call = True
        try:
            answer = self.in_phase("reserve", lambda: self.judge(prompt, EXECUTOR_TOKENS, _route("executor_late" if self.spent_out() else "executor")))
        finally:
            self.late_call = False
        undone = answer.get("undone") if isinstance(answer, dict) else None
        ids = {n for n in (undone if isinstance(undone, list) else []) if isinstance(n, int) and not isinstance(n, bool)}
        chosen = [i for i in open_ if i["id"] in ids]
        self.trace.append({"kind": "catch_all", "action": action, "undone": [i["id"] for i in chosen]})
        return chosen

    # ------------------------------------------------------------------ relations between products (a capability, built once)
    def listing_strings(self, listing: str) -> list[str]:
        """Every text of one listing the store has shown (each field's text, each fact value)."""
        out: list[str] = []
        for record in self.records:
            for found in self.record_dicts(record):
                if isinstance(found, dict) and _record_id(found) == listing:
                    out += [v for v in found.values() if isinstance(v, str) and v != listing]
                    for value in found.values():
                        if isinstance(value, dict):
                            out += [str(x) for x in _flat_facts(value).values() if str(x).strip()]
        row = (self.catalog.get(listing) or {}).get("text")  # (a search's rows may be the only place a store shows a listing's description)
        return list(dict.fromkeys(out + ([row] if row else [])))

    def pair_ok(self, hub: str, dependent: str, judge: bool = True) -> bool | None:
        """Whether a dependent listing goes with a hub listing: its own page names one of the hub's identifiers (precision 1.000,
        recall 0.999 on the 60 stated links of the race-5/6 packs), or its part code is of the family the hub's text names (146 of
        156 known pairs and 3 of 164 others on the race-7 pack, where the first rule finds 8); when the hub's page shows neither, one model
        judgement of the two pages (at most PAIR_CALLS a task); None when that cannot be had."""
        key = (hub, dependent)
        if key in self.pairs:
            return self.pairs[key]
        hub_strings, strings = self.listing_strings(hub), self.listing_strings(dependent)
        hub_ids, series = _ident_phrases(" ".join(hub_strings)), _series_codes(hub_strings)
        if hub_ids or series:
            works = bool(_link_state(hub_strings, strings))  # (a dependent naming no link stays out: such pairs are 20 of 7,000 sampled over the race-5 to race-11 packs)
            if dependent in self.listings():  # (a row only seen may say more, or less, once opened)
                self.pairs[key] = works
            return works
        if not judge or self.pair_calls >= PAIR_CALLS or self.spent_out():
            return None
        self.pair_calls += 1
        listings = self.listings()
        if hub not in listings or dependent not in listings:
            return None
        answer = self.judge(PAIR_PROMPT + "\n\nHUB:\n" + self.listing_block(hub, listings) + "\n\nCANDIDATE:\n" + self.listing_block(dependent, listings), 400, _route("judge"))
        works = answer.get("works") if isinstance(answer, dict) else None
        self.pairs[key] = works if isinstance(works, bool) else None
        return self.pairs[key]

    def link_prior(self, chosen: dict[int, tuple[str, str]]) -> dict[int, tuple[str, str]]:
        """With no relation stated, a world may still pair products (hidden links): a candidate whose own page names the listing
        chosen for another product, of the same state and costing no more than the pick, is preferred (a stated cheapest rule
        outranks the prior). Judged by the pages alone, no model call."""
        if self.relations or len(chosen) < 2:
            return chosen
        out = dict(chosen)
        price = lambda i: (self.catalog.get(i) or {}).get("price")
        swapped: set[int] = set()
        for hub in list(chosen):
            if hub in swapped:
                continue
            h = out[hub][0]
            for dep, (d, state) in list(out.items()):
                if dep == hub or dep in swapped or self.pair_ok(h, d, judge=False):
                    continue
                better = next((c for c, st in self.ranked_pool(dep)[:8] if st == state and c != d and self.pair_ok(h, c, judge=False)
                               and (price(c) is not None and price(d) is not None and price(c) <= price(d))), None)
                if better:
                    out[dep] = (better, state)
                    swapped.add(dep)
                    self.trace.append({"kind": "link_prior", "hub": h, "instead": d, "chosen": better})
        return out

    def pair_choice(self, chosen: dict[int, tuple[str, str]]) -> dict[int, tuple[str, str]]:
        """The picks with every rule between products applied: relations first, then conditional requirements."""
        return self.condition_choice(self.relation_choice(chosen))

    def condition_choice(self, chosen: dict[int, tuple[str, str]]) -> dict[int, tuple[str, str]]:
        """A pair of lines a conditional requirement rules out or leaves in doubt: first the bound product moves to its best
        listing that settles the rule (a listing meeting the requirement passes whichever way the rule goes), else the deciding
        product moves to one under which the rule lapses. Unchanged when no such listing has been opened."""
        out = dict(chosen)
        for rule, dep, hub in self.rule_lines({index: item for index, (item, _) in chosen.items()}):
            if self.rule_state(rule, out[hub][0], out[dep][0]) == "ok":
                continue
            line = next(((d, st) for d, st in self.ranked_pool(dep)[:8] if self.rule_state(rule, out[hub][0], d) == "ok"), None)
            if line:
                self.trace.append({"kind": "condition_choice", "product": dep, "item": line[0], "instead_of": out[dep][0]})
                out[dep] = line
                continue
            lapse = next(((h, st) for h, st in self.ranked_pool(hub)[:8] if self.rule_state(rule, h, out[dep][0]) == "ok"), None)
            if lapse:
                self.trace.append({"kind": "condition_choice", "product": hub, "item": lapse[0], "instead_of": out[hub][0]})
                out[hub] = lapse
            else:
                self.trace.append({"kind": "condition_unproven", "product": dep})
        return out

    def relation_choice(self, chosen: dict[int, tuple[str, str]]) -> dict[int, tuple[str, str]]:
        """Hub-first choice: the hub listing (among its best few) for which the most relations have a dependent that goes with
        it, each dependent the best such listing; under a cheapest rule the cheapest such basket. Unchanged when nothing proves out."""
        if not self.relations:
            return self.link_prior(chosen)
        hubs = sorted({r["hub"] for r in self.relations})
        out = dict(chosen)
        for hub in hubs:
            rels = [r for r in self.relations if r["hub"] == hub]
            if hub in out and all(r["dependent"] in out and self.pair_ok(out[hub][0], out[r["dependent"]][0]) for r in rels):
                continue
            price = lambda i: (self.catalog.get(i) or {}).get("price") or 0
            best = None
            for rank, (h, h_state) in enumerate(self.ranked_pool(hub)[:4]):
                deps = {}
                for r in rels:
                    found = next(((d, st) for d, st in self.ranked_pool(r["dependent"])[:8] if self.pair_ok(h, d)), None)
                    if found:
                        deps[r["dependent"]] = found
                total = price(h) * self.qty(hub) + sum(price(d) * self.qty(i) for i, (d, _) in deps.items())
                score = (-len(deps), total if self.plan.get("priority") == "cheapest" or self.plan.get("priority_scope") == "order" else rank)
                if deps and (best is None or score < best[0]):
                    best = (score, (h, h_state), deps)
            if best:
                out[hub] = best[1]
                out.update(best[2])
                self.trace.append({"kind": "pair_choice", "hub": best[1][0], "dependents": {i: d for i, (d, _) in best[2].items()}})
            else:
                self.trace.append({"kind": "pair_unproven", "hub": hub})
        return out

    def pair_problem(self) -> str:
        """For the navigator's order: a held dependent line whose page names none of the held hub line's identifiers, or held
        lines a conditional requirement rules out."""
        held = {self.product_of(i): i for i in self.held_lines()}
        for rule, dep, hub in self.rule_lines({k: v for k, v in held.items() if k is not None}):
            if self.rule_state(rule, held[hub], held[dep]) == "broken" and (held[hub], held[dep]) not in self.pair_held:
                self.pair_held.add((held[hub], held[dep]))
                return (f"{held[dep]} for the {self.product(dep)} misses {_describe(rule)}, which holds with {held[hub]} as the {self.product(hub)}. Choose a {self.product(dep)} that meets it, "
                        f"or a {self.product(hub)} under which it lapses; to keep the cart as it is, send {self.how('order')} again.")
        for r in self.relations:
            hub, dep = held.get(r["hub"]), held.get(r["dependent"])
            if hub and dep and self.pair_ok(hub, dep) is False and (hub, dep) not in self.pair_held:
                self.pair_held.add((hub, dep))
                ids = ", ".join(self.hub_marks(hub))
                return (f"the request says the {self.product(r['dependent'])} must work with the {self.product(r['hub'])} chosen ({r['quote'][:120]}), and the page of {dep} names none of "
                        f"{hub}'s identifiers ({ids}). Choose a {self.product(r['dependent'])} listing whose own page names one of them (search with them), or another {self.product(r['hub'])}; "
                        f"to keep the cart as it is, send {self.how('order')} again.")
        return ""

    def hub_marks(self, hub: str) -> list[str]:
        """What a page must name to go with this hub listing."""
        strings = self.listing_strings(hub)
        return sorted(_ident_phrases(" ".join(strings)) | _series_codes(strings))[:6]

    def linked_rows(self, hub: str, index: int) -> list[str]:
        """Seen, unopened, buyable rows of a product whose own text goes with the hub listing, cheapest first."""
        key = (hub, index, len(self.catalog), len(self.records))  # (asked on every status line: a hub's rows change only with what was seen)
        if key not in self.linked_cache:
            listings, strings = self.listings(), self.listing_strings(hub)
            goes = lambda text: bool(text) and _link_state(strings, [text]) is True
            rows = [i for i, e in self.catalog.items() if i not in listings and goes(e.get("text")) and self.item_of(i) in (index, None) and e.get("stock") is not False and not self.over_budget(i, index)]
            self.linked_cache = {key: sorted(rows, key=lambda i: (self.catalog[i].get("price") is None, self.catalog[i].get("price") or 0, i))}
        return self.linked_cache[key]

    def open_linked(self, chosen: dict[int, tuple[str, str]]) -> bool:
        """Before the code orders: a relation whose chosen hub has no opened dependent that goes with it. Rows seen that go with it
        are opened (store calls, free of model spend), else its dependent's listings are searched for by the hub's mark. True
        when a call went out; at most LINK_ROUNDS times a task, and never into the steps the endgame needs."""
        if self.link_rounds >= LINK_ROUNDS or self.time_left() < WRAP_UP_SECONDS / 2:
            return False
        for r in self.relations:
            hub = (chosen.get(r["hub"]) or (None, ""))[0]
            if not hub or any(self.pair_ok(hub, d, judge=False) for d, _ in self.ranked_pool(r["dependent"])[:8]):
                continue
            details, search = self.reader(), self.roles.get("search")
            rows = self.linked_rows(hub, r["dependent"])[:2]
            calls = [c for c in (self.checked_call(details, _fill(self.schemas.get(details), i, None, {})) for i in rows) if c] if details else []
            if not calls and search:
                tried = {q.get("text") for q in self.searches.values()}
                # (a store may not index the codes: then the dependent's own rows are searched, and their texts name the family)
                mark = next((m for m in self.hub_marks(hub) + [self.product(r["dependent"])] if m and m not in tried), None)
                call = self.search_call(search, mark) if mark else None
                calls = [call] if call else []
            if not calls or self.steps_left() - len(calls) < self.commit_steps_needed():
                continue
            self.link_rounds += 1
            self.commit_paused = True
            sent = [{"id": f"linked_{self.step}_{n}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for n, c in enumerate(calls)]
            self.messages.append({"role": "assistant", "content": f"The {self.product(r['dependent'])} must go with {hub}, so before ordering I will look at the listings that do.", "tool_calls": sent})
            for reply in self.run_calls(sent, planned=True):
                self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})
            self.trace.append({"kind": "commit_opened_linked", "hub": hub, "items": [c["args"] for c in calls]})
            return True
        return False

    def open_ruled(self, chosen: dict[int, tuple[str, str]]) -> bool:
        """Before the code orders: a conditional requirement in force whose bound product has no opened listing meeting it. Its
        rows stating the value are opened, else the value is searched for (store calls, free of model spend). True when a call
        went out; LINK_ROUNDS shared with open_linked, and never into the steps the endgame needs."""
        if self.link_rounds >= LINK_ROUNDS or self.time_left() < WRAP_UP_SECONDS / 2:
            return False
        listings = self.listings()
        details, search = self.reader(), self.roles.get("search")
        for rule, dep, hub in self.rule_lines({index: item for index, (item, _) in chosen.items()}):
            if not self.condition_state(rule, chosen[hub][0], listings) or not rule["values"] or any(self.rule_state(rule, chosen[hub][0], d) == "ok" for d, _ in self.ranked_pool(dep)[:8]):
                continue
            rows = [i for i in self.unopened_rows(dep) if any(_says(self.catalog[i].get("text") or "", v) for v in rule["values"])][:2]
            calls = [c for c in (self.checked_call(details, _fill(self.schemas.get(details), i, None, {})) for i in rows) if c] if details else []
            if not calls and search:
                tried = {q.get("text") for q in self.searches.values()}
                word = next((str(v) for v in rule["values"] if str(v) not in tried), None)
                call = self.search_call(search, word) if word else None
                calls = [call] if call else []
            if not calls or self.steps_left() - len(calls) < self.commit_steps_needed():
                continue
            self.link_rounds += 1
            self.commit_paused = True
            sent = [{"id": f"ruled_{self.step}_{n}", "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for n, c in enumerate(calls)]
            self.messages.append({"role": "assistant", "content": f"With {chosen[hub][0]} chosen, the {self.product(dep)} must meet {_describe(rule)}, so before ordering I will look at listings that do.", "tool_calls": sent})
            for reply in self.run_calls(sent, planned=True):
                self.messages.append({"role": "tool", "tool_call_id": reply.id, "content": reply.text + "\n" + self.status()})
            self.trace.append({"kind": "commit_opened_ruled", "product": dep, "calls": [c["args"] for c in calls]})
            return True
        return False

    def relation_hint(self) -> str:
        if not self.relations and not self.conditional:
            return ""
        held = {self.product_of(i): i for i in self.held_lines()}
        parts = []
        for r in self.relations:
            hub = held.get(r["hub"])
            ids = self.hub_marks(hub)[:4] if hub else []
            rows = self.linked_rows(hub, r["dependent"])[:3] if hub else []
            parts.append(f"the {self.product(r['dependent'])} must work with the {self.product(r['hub'])} chosen" +
                         (f" ({hub}: a {self.product(r['dependent'])} goes with it when its own page names {', '.join(ids)}" + (f"; seen rows that do: {', '.join(rows)}" if rows else "") + ")" if ids else " (choose that one first; a listing goes with it when its page names its model)"))
        parts += [f"product {r.get('item')}: {_describe(r)}" for r in self.conditional]
        return "\n[relations] " + "; ".join(parts)

    def run_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[Reply]:
        """Screen the calls of one turn (the navigator's, or the code's own when planned), hold back those a check stops, send
        the rest to the store and answer each with its result and the notes the code adds, in the order the calls came."""
        replies: list[Reply] = []
        allowed = {t["function"]["name"] for t in self.tools_for_turn()}
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        cart_tools = self.cart_tools()
        runnable: list[tuple[str, str, dict[str, Any]]] = []
        planned_messages = 0
        for call in calls:
            name, args, held = self.screen_call(call, len(calls), planned, allowed, cart_tools, planned_messages)
            if held:
                replies.append(Reply(call["id"], held, True))
                continue
            if name == self.roles.get("message"):
                planned_messages += 1
            runnable.append((call["id"], name, args))
        if runnable and not planned and not self.committing:
            runnable = self.keep_fitting_lines(runnable, replies)
        if runnable and runnable[0][1] in finishing:
            call_id, name, args = runnable[0]
            held = self.finish_gate(name, args)
            if held:
                replies.append(Reply(call_id, held, True))
                return replies
            if self.finish_kind(name, args) == "order" and not self.committing and self.short_cart():
                # past its hold, an order without a line for every product never fits: where no decline is proven, the navigator
                # is told once which known listings complete it; else (or on a second try) the decline goes instead
                held_products = {self.product_of(i) for i in self.held_lines()}
                missing = [index for index in range(len(self.items)) if index not in held_products]
                filled = self.best_effort_fill(missing, {p: i for i in self.held_lines() for p in [self.product_of(i)] if p is not None}) if not self.best_effort_told else {}
                if filled and self.steps_left() > len(filled) + 1:
                    self.best_effort_told = True
                    replies.append(Reply(call_id, "Not run yet (no step used): the order needs a line for every product, and no decline is proven: put " + ", ".join(f"{i} for product {index} ({self.product(index)})" for index, i in sorted(filled.items())) + " in the cart, then order.", True))
                    return replies
                if self.best_effort_told and self.best_effort_fill(missing, {p: i for i in self.held_lines() for p in [self.product_of(i)] if p is not None}):
                    self.commit_wanted = self.completing = True  # told once already: the code completes the basket and orders
                    replies.append(Reply(call_id, "Not run yet (no step used): the order needs a line for every product; the code completes the basket with the best listings known and orders.", True))
                    return replies
                self.trace.append({"kind": "decline_short_cart", "lines": self.order_lines()})
                name, args = self.roles["decline"], self.finish_args("decline")
                runnable[0] = (call_id, name, args)
            kind = self.finish_kind(runnable[0][1], runnable[0][2])
            kind = kind if kind != "unknown" else ("order" if self.held_lines() else "decline")
            held = self.gate(kind, planned)
            if held:
                replies.append(Reply(call_id, held, True))
                return replies
            name, args = runnable[0][1], self.with_remark(kind, runnable[0][1], runnable[0][2])
            if kind == "decline":
                args = self.clean_decline(args, name)  # the structural check, after any remark the instructions shaped
            elif kind == "order" and self.guarded(self.last_line_pages, False):
                replies.append(Reply(call_id, "Not run yet (no step used): opening the lines' pages brought news; check it, then send the order again.", True))
                return replies
            runnable[0] = (call_id, name, args)
        message_tool = self.roles.get("message")
        if runnable and not self.discharging and message_tool and any(name == message_tool for _, name, _ in runnable) and any(i["when"] == "before_message" for i in self.instructions):
            held = self.gate("message", planned)  # an instruction to carry out before the shopper gets a message
            room = self.message_total() if self.message_window or (planned and self.discharging) else self.message_limit()
            if not held and self.messages_used >= room:
                held = "Not run (no step used): the message budget is used up. Do not message the shopper again."
            if held:
                for call_id, name, _ in [r for r in runnable if r[1] == message_tool]:
                    replies.append(Reply(call_id, held, True))
                runnable = [r for r in runnable if r[1] != message_tool]
        if runnable and (not planned or self.committing) and not self.discharging and self.held_lines() and any(name in cart_tools for _, name, _ in runnable):
            held = self.gate("cart_change", planned)  # an instruction to check with the shopper before the cart changes
            if held:
                for call_id, name, _ in [r for r in runnable if r[1] in cart_tools]:
                    replies.append(Reply(call_id, held, True))
                runnable = [r for r in runnable if r[1] not in cart_tools]
        runnable = [part for entry in runnable for part in self.guarded(lambda: self.split_batch(*entry), [entry])]
        if runnable and runnable[0][1] not in finishing and finishing:
            keep = self.steps_left() - 1
            if keep < len(runnable):
                for call_id, name, _ in runnable[max(0, keep):]:
                    replies.append(Reply(call_id, "Not run (no step used): only the last step is left, and it must be the order (or the decline).", True))
                runnable = runnable[: max(0, keep)]
        sent = 0
        while sent < len(runnable):  # (a list the store refused as too long comes back on the queue, shorter)
            chunk = runnable[sent : sent + 16]
            sent += len(chunk)
            if self.finished or self.dead or self.steps_left() <= 0:
                for call_id, _, _ in chunk:
                    replies.append(Reply(call_id, "Not run: the episode is over or no steps are left.", True))
                continue
            chunk = chunk[: self.steps_left()]
            actions = [{"name": name, "args": args} for _, name, args in chunk]
            before = self.step
            reply = self.env_call(actions)
            self.trace.append({"kind": "env", "step": before + 1, "actions": actions, "reply": reply})
            if reply is None:
                for call_id, _, _ in chunk:
                    replies.append(Reply(call_id, "The call failed (network or server). Try again."))
                continue
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            for index, (call_id, name, args) in enumerate(chunk):
                if index >= len(observations):
                    replies.append(Reply(call_id, "Not run: the episode finished before this call.", True))
                    continue
                last = index == len(observations) - 1
                replies.append(Reply(call_id, self.record_result(name, args, observations[index], bool(reply.get("finished")), last, cart_tools)))
                slot = self.batch_slot(name)
                runnable += [(call_id, name, {**args, slot: part}) for part in (self.guarded(lambda: self.learn_batch(name, args, observations[index]), []) if slot else [])]
        if any((self.catalog.get(i) or {}).get("stock") is False for i in self.held_lines()):
            self.stock_out_step = self.step  # a cart line is sold out: what replaces it late must be a checked listing
        if len(replies) > len(calls):  # the parts of a split call answer it as one
            joined: dict[str, list[Reply]] = {}
            for reply in replies:
                joined.setdefault(reply.id, []).append(reply)
            replies = [Reply(call_id, "\n".join(r.text for r in mine), all(r.held_back for r in mine)) for call_id, mine in joined.items()]
        order = {c["id"]: i for i, c in enumerate(calls)}
        replies.sort(key=lambda r: order.get(r[0], 0))
        return replies

    def coverage_gap(self, index: int) -> tuple[str, str]:
        """(key, what to read next) when the search of one product's required values (or of its own word with a required value)
        has rows left unread: its next page, up to two pages past the first and beyond them while every page left (and one
        opening) fits in the steps the gate keeps; or, when the only such search shows mostly other
        products, the search of the product's own word with a required value. ("", "") when nothing is left to read."""
        info = self.board(index)
        if not info or info.get("complete"):
            return "", ""
        queries = [q for q in self.searches.values() if self.covering(q, index) or self.refines(q, index)]
        values = [r["values"][0] for r in self.reqs_for(index, blocking=True) if r["op"] == "is" and r["values"]]
        named = " ".join(_words(self.product(index)))
        if values and named and not any(self.refines(q, index) and len(q["words"]) > len(_words(values[0])) for q in queries):
            rows = [i for q in queries if self.covering(q, index) for i in q["seen"]]
            mine = sum(1 for i in rows if self.item_of(i) == index)
            if rows and mine * 2 < len(rows):  # the value's words alone mostly found other products: their pages would too
                return f"mixed|{index}", f"your search for '{values[0]}' shows mostly other products: search '{named} {values[0]}' (the product's own word with the required value), read its first pages, open any cheaper row that may fit, then order."
        total = lambda q: q["totals"][-1] if q["totals"] and q["totals"][-1] is not None else None
        unread = self.unread_searches(index, queries)
        if unread:
            q = unread[0]
            return f"{q['text']}|{len(q['offsets'])}", f"your search '{q['text']}' has rows you have not read ({len(q['seen'])} seen of {total(q) if total(q) is not None else 'more'}): read its next page (the rows after the {len(q['seen'])} you have seen), open any cheaper row that may fit, then order."
        return "", ""

    def unread_searches(self, index: int, queries: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """The searches of one product's required values with pages left to read, the narrower first (the product's own word with
        the value lists fewer rows): up to two pages past the first, and beyond them only while every page left and one opening
        fit in the steps the coverage gate keeps (steps left over its REPAIR + 2 floor)."""
        if queries is None:
            queries = [q for q in self.searches.values() if self.covering(q, index) or self.refines(q, index)]
        total = lambda q: q["totals"][-1] if q["totals"] and q["totals"][-1] is not None else None
        spare = self.steps_left() - (self.th('repair') + 2)
        pages = lambda q: -(-(total(q) - len(q["seen"])) * max(1, len(q["offsets"])) // max(1, len(q["seen"])))  # pages left at this page size
        unread = [q for q in queries if not q["ended"] and (total(q) is None or len(q["seen"]) < total(q))
                  and (len(q["offsets"]) < 3 or (total(q) is not None and pages(q) + 1 <= spare))]
        return sorted(unread, key=lambda q: (not self.refines(q, index), total(q) if total(q) is not None else 10 ** 9, -q.get("step", 0)))

    def better_evidenced(self) -> str:
        """A preference, once per cart line and ledger: a fitting line whose evidence is weaker than another fitting listing's of
        its product. Without a cheapest or best rule, a listing at the same or a better state that states the requirements more
        directly; with a cheapest rule, only a line whose evidence is of the weakest kind (a related value, a "+" join, a key
        naming part of the property), and then the cheapest listing whose evidence is not."""
        priority = self.plan.get("priority")
        if priority == "best":
            return ""
        listings = self.listings()
        grade = {"fits": 0, "may fit": 1, "unclear": 2}
        for line in self.held_lines():
            index = self.product_of(line)
            mine = self.assess(line, listings, judge=False) if index is not None else None
            if not mine or mine["state"] == "fails" or f"{line}|{self.ledger_version}" in self.evidence_nudged:
                continue
            if priority == "cheapest":
                if mine.get("level", 0) < 3:
                    continue
                pool = []
                for item in listings:
                    entry = self.catalog.get(item) or {}
                    theirs = self.assess(item, listings, judge=False) if item != line and self.item_of(item) == index else None
                    if theirs and theirs["state"] != "fails" and theirs.get("level", 0) < 3 and entry.get("stock") is not False and entry.get("price") is not None and not self.over_budget(item, index) and not self.verdict(item):
                        pool.append((entry["price"], item))
                best = min(pool)[1] if pool else None
            else:
                best = self.ranked_candidate(index)[0]
                theirs = self.assess(best, listings, judge=False) if best and best != line else None
                if not theirs or (grade.get(_candidate_state(theirs), 3), theirs.get("level", 0), theirs.get("soft", 0)) >= (grade.get(_candidate_state(mine), 3), mine.get("level", 0), mine.get("soft", 0)):
                    best = None
            if not best:
                continue
            self.evidence_nudged.add(f"{line}|{self.ledger_version}")
            r, _, evidence = max(zip(mine["results"], mine["levels"]), key=lambda x: x[1])[0]
            self.trace.append({"kind": "evidence_nudge", "line": line, "instead": best})
            shown = f"{best} ({format(self.catalog[best]['price'], 'g')})" if priority == "cheapest" else best
            return f"{line} rests on weak evidence for {_describe(r)} ({evidence[:80] or 'no field states it'}), and {shown} states every requirement more directly; prefer it unless you are sure of {line}."
        return ""

    def cart_removals(self, name: str, args: dict[str, Any]) -> list[str]:
        """Held listings a per-line cart call takes out: a count of 0 on a cart without modes, or a call in the remove mode
        that leaves none of the line (a basket call that replaces the whole basket is not counted; taking off fewer units
        than the line holds only lowers its count)."""
        others = {self.roles.get(r) for r in ("search", "details", "view", "message", "cart_view", "order", "decline")} - {self.roles.get("cart")}
        if name not in self.cart_tools() or name in others or self.basket_tool():
            return []  # (a tool with another role is never a removal, even when a format learned it as one)
        lines = _arg_lines(args, set(self.catalog) | self.seen_ids)
        target = lines[0][0] if lines else self.cart_target(name, args)
        if not target or target not in self.held_lines():
            return []
        amount = lines[0][1] if lines and lines[0][1] is not None else self.cart_amount(name, args)
        if name == self.roles.get("cart") and self.mode_less_cart():
            return [target] if amount == 0 and self.cart_counts != "add" else []
        remove = self.cart_modes.get("remove")
        if name == self.roles.get("cart"):
            removing = bool(remove) and _compact(remove) == self.cart_signature(name, args)
        else:
            removing = name == (self.formats.get("remove") or {}).get("tool")
        props = (self.schemas.get(name) or {}).get("properties") or {}
        counted = bool(lines and lines[0][1] is not None) or any(isinstance(s, dict) and s.get("type") == "integer" and "enum" not in s for s in props.values())
        held = self.held.get(target)
        return [target] if removing and not (counted and held and amount < held) else []

    def fitting_line(self, line: str) -> bool:
        """A held line that meets every requirement, in stock and within the budget, and the only such line of its product."""
        index = self.item_of(line)
        entry = self.catalog.get(line) or {}
        if index is None or index < 0 or entry.get("stock") is False or self.over_budget(line, index) or self.verdict(line):
            return False
        listings = self.listings()
        fits = lambda item: (self.assess(item, listings, judge=False) or {}).get("state") == "fits"
        return fits(line) and not any(other != line and self.item_of(other) == index and fits(other) for other in self.held_lines())

    def keep_fitting_lines(self, runnable: list[tuple[str, str, dict[str, Any]]], replies: list[Reply]) -> list[tuple[str, str, dict[str, Any]]]:
        """Hold back a navigator call that takes a verified fitting line out of the cart while nothing in the same reply puts
        in another listing of its product: the line stays right after a price change, and a cart without it orders too little."""
        adds = {self.item_of(t) for _, name, args in runnable if not self.cart_removals(name, args) for t in self.cart_targets(name, args)}
        kept = []
        for call_id, name, args in runnable:
            line = next((i for i in self.cart_removals(name, args) if self.item_of(i) not in adds and self.fitting_line(i)), None)
            if line:
                self.trace.append({"kind": "kept_line", "item": line})
                replies.append(Reply(call_id, f"Not run (no step used): {line} meets every requirement, is in stock and within the budget, so it stays in the cart. Take it out only in the same reply that puts in the listing replacing it; after a price change, re-read its price and order it again.", True))
                continue
            kept.append((call_id, name, args))
        return kept

    def screen_call(self, call: dict[str, Any], batch: int, planned: bool, allowed: set[str], cart_tools: set[str], planned_messages: int) -> tuple[str | None, dict[str, Any], str]:
        """(tool name, checked arguments, why the call is held back or ""), checks in a fixed order: a known tool, no message once
        the run's key is spent, a tool allowed now, valid arguments, a finishing call alone and with its mode stated, messages left, not a repeat, a cart addition that
        passes its checks, a cart count that fits the order."""
        function = call.get("function") or {}
        name = function.get("name")
        raw = function.get("arguments")
        message_tool = self.roles.get("message")
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        if name not in self.schemas:
            return name, {}, f"Not run (no step used): '{name}' is not one of the tools. Use one of: {', '.join(sorted(allowed))}."
        if name == message_tool and self.key_out():  # every path a message can take passes here
            return name, {}, "Not run (no step used): no message can reach the shopper any more. Do not message the shopper again."
        capped = self.call_cap(name)
        if capped and name not in finishing:
            return name, {}, f"Not run (no step used): {capped}"
        if not planned and name not in allowed:
            if name == message_tool and self.messages_used + planned_messages < self.message_limit():
                reason = "questions to the shopper are planned and sent for you, and the messages the instructions ask for are sent for you right before the finish"
            elif name == message_tool and self.messages_used + planned_messages >= self.message_limit():
                return name, {}, "Not run (no step used): the message budget is used up. Do not message the shopper again."  # as when the tool was offered
            else:
                reason = "too few steps are left for this; only these tools can be used now: " + ", ".join(sorted(allowed))
            return name, {}, f"Not run (no step used): {reason}."
        if isinstance(raw, dict):
            args = raw
        else:
            args = _parse_json_object(raw or "{}", salvage=False) if (raw or "").strip() else {}
            if args is None:
                return name, {}, "Not run (no step used): the arguments were not a valid JSON object."
        args, problems = _check(args, self.schemas[name], "args")
        if problems:
            return name, args, "Not run (no step used): invalid arguments: " + "; ".join(problems[:6])
        if name in finishing and batch > 1:
            return name, args, "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls."
        if self.finish_kind(name, args) == "unknown" and self.order_args and self.decline_args and not self.mode_asked:
            self.mode_asked = True
            return name, args, f"Not run (no step used): say explicitly how this ends: {self.how('order')} places the order, {self.how('decline')} declines."
        limit = self.message_total() if (planned and self.discharging) or (not planned and self.message_window) else self.message_limit()
        if name == message_tool and self.messages_used + planned_messages >= limit:
            return name, args, "Not run (no step used): the message budget is used up. Do not message the shopper again."
        signature = name + _compact(args)
        if name not in finishing and name != message_tool and name not in cart_tools and signature in self.recent:
            return name, args, "Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else."
        repeat = self.guarded(lambda: self.search_held(name, args), "") if not planned else ""
        if repeat:
            return name, args, repeat
        if name in cart_tools and self.steps_left() > self.th('cart_only') and not self.committing:  # the commit checked its picks itself
            going = [i for i, _ in _arg_lines(args, set(self.catalog) | self.seen_ids)] if self.basket_tool() else self.cart_targets(name, args)
            other = next((t for t in going if self.second_kind(t, going)), None)
            if other:
                try:
                    self.reread_products(other)
                except Exception as exc:  # the order stays as it was read
                    self.trace.append({"kind": "reread_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            target = next((t for t in self.cart_targets(name, args) if f"{t}|{self.ledger_version}" not in self.cart_checked), None)
            if target:
                self.cart_checked.add(f"{target}|{self.ledger_version}")
                verdict = self.verify_item(target, "ADD TO THE CART")
                if verdict:
                    return name, args, "Not run (no step used). " + verdict + " If you are certain it fits, send the same call again."
        if name in cart_tools and not planned and not self.committing and self.stock_out_step is not None and self.steps_left() <= self.th('repair'):
            listings = self.listings()
            late = next((t for t in self.cart_targets(name, args) if not self.cart_removals(name, args) and t not in self.late_checked
                         and (t not in listings or (self.assess(t, listings, judge=False) or {}).get("state") == "fails")), None)
            if late:
                self.late_checked.add(late)
                index = self.product_of(late)
                best, state = self.best_candidate(index=index if index is not None else 0, last_resort=True)
                instead = f" Put {best} ({state}) in instead, or, if you are sure, send the same call again." if best and best != late else " If you are sure, send the same call again."
                return name, args, f"Not run (no step used): with {self.steps_left()} steps left after a sell-out, the replacement must be a listing whose details are open and whose check fits, and {late} is {'not opened' if late not in listings else 'failing its check'}.{instead}"
        if name == self.roles.get("cart") and not self.committing:  # the commit sets each count itself
            warning = self.quantity_warning(name, args)
            if warning and signature not in self.qty_warned:
                self.qty_warned.add(signature)
                return name, args, "Not run (no step used): " + warning + " If you are sure, send the same call again."
        return name, args, ""

    def finish_gate(self, name: str, args: dict[str, Any]) -> str:
        """Why an order (or a decline) is held back once, or "": checks in a fixed order, each holding the call at most once
        per situation; the code's own commit passes the order checks."""
        kind = self.finish_kind(name, args)
        if kind == "order" and not self.committing and self.short_cart() and self.steps_left() > self.th('last') and (self.commit_failures < 2 or self.short_held != self.cart_version):
            # the code's commit completes the basket or declines; once it cannot, the navigator is told once per cart, and an
            # order sent again unchanged (or with no steps left to change it) is replaced by the decline (run_calls)
            self.short_held = self.cart_version
            self.commit_wanted = self.completing = self.commit_failures < 2
            what = "I will put the best listing found for each product in the cart and order, or decline if a product has none" if self.commit_wanted else f"put a fitting listing of each missing product in the cart and send {self.how('order')} again, or decline if a product has none"
            return f"Not run (no step used): the cart holds {len(self.order_lines())} line(s) for the {len(self.items)} products the request asks for, and an order needs one line for each (a shorter order never fits): {what}."
        if kind == "order" and not self.committing and self.steps_left() > self.th('last'):
            problem = self.cart_quantity_problem()
            if problem and self.qty_blocked != self.cart_version:
                self.qty_blocked = self.cart_version
                return "Not run (no step used): " + problem
        if kind == "order" and not self.committing and self.steps_left() > self.th('repair') and (self.relations or self.conditional):
            problem = self.pair_problem()
            if problem:
                return "Not run yet (no step used): " + problem
        if kind == "order" and not self.committing and self.steps_left() > self.th('last'):
            held = {self.product_of(i) for i in self.held_lines()}
            pending = [r for r in self.reqs if r.get("pending") and (r.get("item") is None or not self.multi() or r.get("item") in held)]
            if pending and self.safe_hop_move():  # the code looks it up before the next turn (it has no call once the steps are the order's)
                r = pending[0]
                return f"Not run yet (no step used): the requirement {_describe(r)}{self.about(r)} still takes its value from another listing ({r['reference']}), which has not been found yet; it is looked up for you first, since its value may change the choice."
            if not pending and not self.open_pages and len(self.pages_unopened()) and self.steps_left() > self.th('repair') + len(self.pages_unopened()):
                self.open_pages = True  # (once: a store may count only this as opening the pages of the chain)
                return "Not run yet (no step used): the code opens the pages of the listings a requirement pointed at first; order again after that."
            if pending and self.steps_left() > self.th('ready_commit') and self.step not in self.pending_holds and len(self.pending_holds) < 3:
                self.pending_holds.add(self.step)  # a store whose search the code cannot write: only the navigator can find it
                r = pending[0]
                return f"Not run yet (no step used): the requirement {_describe(r)}{self.about(r)} still takes its value from another listing ({r['reference']}), which has not been found yet. Find that listing and open its details (and any listing its page sends you to) before ordering; with {self.th('ready_commit')} steps or fewer left the order goes through."
        if kind == "order" and not self.committing and self.plan.get("priority") in ("cheapest", "best") and self.drop_step is not None and self.drop_held != self.drop_step and self.steps_left() >= self.th('repair') + 2:
            if not any(q.get("step", -1) > self.drop_step for q in self.searches.values()):
                self.drop_held = self.drop_step
                return f"Not run yet (no step used): the shopper no longer requires {self.drop_text}, so more listings fit now and a cheaper (or better) one may exist. Search again with the requirements that remain (fewer words than before, in the requested category), open the cheapest new hits, and order the one that fits best by the request's rule."
        if kind == "order" and not self.committing and self.planner_on and self.asked and not self.ask_due and self.messages_used < self.message_limit() and self.steps_left() > self.th('question_hold') and self.time_left() > WRAP_UP_SECONDS + QUESTION_SECONDS:
            self.ask_due = True
            return "Not run yet (no step used): one more planned question to the shopper goes out first, since its answer may change the choice."
        if kind == "order" and not self.committing and self.steps_left() > self.th('repair') and self.roles.get("details") in {t["function"]["name"] for t in self.tools_for_turn()}:
            stale = next((i for i in self.held_lines() if (self.catalog.get(i) or {}).get("stale") and i not in self.stale_held), None)
            if stale:  # the choice and the budget are judged at the new price, which no result has shown yet
                self.stale_held.add(stale)
                self.trace.append({"kind": "stale_price_hold", "line": stale})
                return (f"Not run yet (no step used): the store said the price of {stale} changed, and no result has shown its new price yet. Open it with {self.roles['details']} to read the new price: "
                        "the budget, and under a cheapest rule which listing costs least, are judged at the new price; then order the listing that ranks first by the request's rule.")
        if kind == "order" and self.review_due():
            return self.review_hold(name, args)
        if kind == "order" and not self.committing and self.steps_left() > self.th('repair') and self.time_left() > WRAP_UP_SECONDS + ONE_CALL_SECONDS:
            found = self.better_fit()
            if found and found not in self.better_fit_held:
                self.better_fit_held.add(found)
                line, item = found
                self.trace.append({"kind": "better_fit_hold", "line": line, "item": item})
                if self.plan.get("priority") == "cheapest":
                    why = f"the request asks for the cheapest fitting listing, and {item} ({format(self.catalog[item]['price'], 'g')}) costs less than {line} ({format(self.catalog[line]['price'], 'g')}) in the cart"
                else:
                    why = f"the request asks for the best by {self.best['property']}, and {item} measures better on it than {line} in the cart"
                return (f"Not run yet (no step used): {why}; {item} is opened, in stock, within the budget and meets every requirement" + (" on its own fields. " if item in self.strong_fits(self.item_of(item)) else " (the independent check passed it). ") + 
                        f"Swap them: take {line} out of the cart and put {item} in ({self.qty(self.item_of(item))} unit(s)), then order. If you are sure {line} is the right choice, send {self.how('order')} again.")
        if kind == "order" and not self.committing and self.steps_left() > self.th('repair') and self.time_left() > WRAP_UP_SECONDS + ONE_CALL_SECONDS:
            cheaper = self.cheaper_unopened()
            if cheaper:
                shown = ", ".join(f"{i} ({format(self.catalog[i]['price'], 'g')})" for i in cheaper)
                return f"Not run yet (no step used). The request asks for the cheapest fitting listing, and your searches for the required values showed cheaper listings in the same category that you have not opened: {shown}. Open them and buy the cheapest one that fits; if none fits, send {self.how('order')} again."
        if kind == "order" and not self.committing and self.plan.get("priority") in ("cheapest", "best") and self.steps_left() >= self.th('repair') + 2 and (len(self.coverage_holds) < 3 or self.steps_left() > self.th('reading_ends')):
            held = sorted({self.product_of(i) for i in self.held_lines()} - {None}) if self.multi() else [0]
            for index in held:
                key, gap = self.coverage_gap(index)
                if gap and key not in self.coverage_holds:
                    self.coverage_holds.add(key)
                    self.trace.append({"kind": "coverage_hold", "product": index, "key": key})
                    return f"Not run yet (no step used): the {self.plan.get('priority')} {self.product(index) or 'listing'} must be chosen over every row your searches can show, and " + gap
        if kind == "order" and not self.committing and self.steps_left() > self.th('repair'):
            try:
                better = self.better_evidenced()
            except Exception as exc:
                self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
                better = ""
            if better:
                return "Not run yet (no step used): " + better + f" To keep the cart as it is, send {self.how('order')} again."
        if kind == "order" and self.order_checked and not self.committing and self.steps_left() > self.th('last'):
            try:
                swap = self.failing_cart_swap()
            except Exception as exc:
                self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
                swap = ""
            if swap:
                return "Not run (no step used): " + swap
        if kind == "order" and not self.order_checked and self.steps_left() > self.th('last') and self.time_left() > WRAP_UP_SECONDS:
            self.order_checked = True
            verdict = self.verify_cart()
            if verdict:
                return "Not run yet (no step used). " + verdict
            lines_rule = "the cart holds one line for each product the request asks for, each at its required quantity, and nothing else" if self.several_products() or self.multi() else "the cart holds exactly one line, the chosen item at the required quantity, and nothing else"
            return f"Not run yet (no step used). Pre-order check: (1) {lines_rule}; (2) after your last cart change you re-read this item's stock and price with a tool that shows them, and it is in stock and price times quantity is within the current budget; (3) its own properties meet every stated requirement, every need the shopper revealed and every change announced; (4) if the cheapest or best was asked for, no better fitting item exists among everything you could find; (5) every instruction listed under [instructions] that is about your own work is done (the messages they ask for are sent for you right before the finish). If anything is unchecked, check it now; otherwise send {self.how('order')} again."
        if kind == "decline" and self.review_due():
            return self.review_hold(name, args)
        if kind == "decline" and not self.code_declining:
            try:
                blocker = self.decline_blocker()
            except Exception as exc:
                self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
                blocker = ""
            if blocker:
                return "Not run (no step used): " + blocker
            if not self.decline_checked and self.steps_left() > self.th('commit') and self.time_left() > WRAP_UP_SECONDS + CHECK_SECONDS:
                self.decline_checked = True
                audit = self.audit_decline()
                if audit:
                    return "Not run yet (no step used). " + audit
                return f"Not run yet (no step used). Before declining, be sure: you searched in several ways (different words, filters, pages), you checked every candidate's own properties, and each one fails a stated requirement, a revealed need, the budget or stock. Remember that close synonyms and any key spelling count. If something does fit, buy it instead; if nothing fits, send {self.how('decline')} again."
        return ""

    def record_result(self, name: str, args: dict[str, Any], observation: Any, finished: bool, last: bool, cart_tools: set[str]) -> str:
        """Take in one call's result (the cart, refusals, the shopper's words, listings and searches) and write the text the
        navigator reads: the result itself and the notes it calls for."""
        message_tool = self.roles.get("message")
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        count_note = ""
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
        if name == message_tool:
            self.messages_used += 1
        kind = self.finish_kind(name, args)
        if kind == "order":
            self.order_attempts += 1
        if name in cart_tools:
            self.cart_used = True
            self.order_checked = False
        if name in cart_tools or name == message_tool or name in finishing:
            self.recent = []
        else:
            self.recent = (self.recent + [name + _compact(args)])[-2:]
        self.records.append(observation)
        if name == self.roles.get("search"):
            self.search_records.add(id(observation))
        if name in cart_tools:
            self.last_echo = None
            count_note = self.track_cart(name, args, observation, by_code=self.committing)
        if name == self.roles.get("cart_view") or (name in cart_tools and self.last_echo is not None):
            self.read_cart_view(observation)  # a cart call whose result lists the whole basket is a cart view too
        refused = None
        version, close = self.ledger_version, None
        if name in finishing and not finished:
            self.refusals += 1  # an order or decline that did not end the episode, whatever the result's key names
            self.order_checked = False
            close = self.close_key("decline" if kind == "decline" else "order")
            self.close_refused[close] = self.close_refused.get(close, 0) + 1
            refused = ("decline" if kind == "decline" else "order") if _event_texts(observation) else None
            if kind == "order" and refused is None and not _worked(observation):
                self.order_errors += 1  # the store did not take the order (such as an empty basket): the cart is not as believed
                self.cart_doubt = True
        notes = [self.note_observation(name, observation, refused), count_note]
        heard, events, question = self.shopper_lines(name, args, observation)
        if name == message_tool and not heard and not events:
            # the store gave no shopper words back: it did not deliver the message, so the messages it took are all it takes
            self.messages_used = max(0, self.messages_used - 1)
            self.set_budget("messages", self.messages_used, "store (message not delivered)")
            self.planner_on = False
            notes.append(f"[note] The store did not deliver this message: the shopper takes no more messages ({self.messages_used} were sent). Do not message the shopper again.")
        if events:
            notes.append(self.interpret(events, None))
        if heard:
            notes.append(self.interpret(heard, question))
        if close:
            self.reopen_close(heard, version, close)
        if name == message_tool:
            self.last_heard = list(heard)
            self.message_log.append({"step": self.step, "cart": self.basket_key(), "text": question or "", "reply": " / ".join(heard)[:300], "by": "gate" if self.discharging else ("planner" if self.asked_property is not None else "navigator")})
            if self.message_window and not self.discharging and self.message_log[-1]["by"] == "navigator":
                # the navigator sent the message the executor could not write: one message, settled by the reply
                self.message_window = False
                for ins in self.instructions:
                    if ins.get("window"):
                        ins["window"] = False
                        if not ins["wait_reply"] or self.last_ack == "accepted":
                            ins.update(status="satisfied", done_for=self.basket_key())
        known = set(self.catalog)
        searched = self.safe_observe(name, args, observation)  # first: it files the shelf labels that map listings to products
        self.guarded(lambda: self.learn_search(name, args, observation, known), None)
        opened = self.record_listings(observation)
        if name == self.roles.get("details"):
            self.details_opened.update(listing[0] for listing in opened)
        self.learn_reader(name, args, observation, opened)
        if name == self.roles.get("view"):
            self.compared.update(listing[0] for listing in opened)
        if opened and self.multi():
            self.map_items([listing[0] for listing in opened])
        if opened and any(r.get("pending") for r in self.reqs):
            notes.append(self.resolve_references([listing[0] for listing in opened]))
        if self.notes_changed:
            changed, self.notes_changed = list(self.notes_changed), set()
            if any(r.get("pending") for r in self.reqs):
                notes.append(self.resolve_references(changed))
        if len(opened) > 1 and any(r["blocking"] for r in self.reqs):
            self.judge_values([listing[0] for listing in opened[:4]], self.listings())  # one value judgement for a batch, not one per listing
        notes += [self.check_note(listing[0]) for listing in opened[:4]]
        if self.price_moves:
            moved, self.price_moves = self.price_moves, []
            if not self.committing:  # (the code's own commit chooses again at the prices now known)
                try:
                    notes.append(self.price_review(moved))
                except Exception as exc:
                    self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
        if searched or opened:
            notes.append(self.safe_board())
        if self.search_note:
            notes.append(self.search_note)
            self.search_note = ""
        text = _compact(observation)
        if last and finished:
            text += " [the episode is finished]"
        note = " ".join(n for n in notes if n)
        return text + ("\n[note] " + note if note else "")

    def shopper_lines(self, name: str, args: dict[str, Any], observation: Any) -> tuple[list[str], list[str], str | None]:
        """(the shopper's answer lines, other shopper lines riding along, the question asked): a message reply may carry the
        answer next to a change announced in the same breath; any other result may carry the shopper's words beside it."""
        if name != self.roles.get("message"):
            return _event_texts(observation), [], None
        heard: list[str] = []
        events: list[str] = []
        if isinstance(observation, str):
            heard = [observation]
        elif isinstance(observation, dict) and "error" not in observation:
            said = [v for v in observation.values() if isinstance(v, list) and v and all(isinstance(x, str) for x in v)]
            answers = _strings([v for v in observation.values() if not any(v is x for x in said)])
            if said and answers:
                _sentences(said, events)
                _sentences(answers, heard, least=1)  # a reply to our message may be short ("Noted, thanks.")
            else:
                _sentences(list(v for v in observation.values() if isinstance(v, (str, list))), heard, least=1)
        return heard, events, " ".join(str(v) for v in args.values() if isinstance(v, str))

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.in_phase("setup", self.setup)
        self.trace.append({"kind": "plan", "plan": self.plan, "roles": self.roles})
        try:
            self.in_phase("setup", self.prepare_formats)
        except Exception as exc:
            self.trace.append({"kind": "format_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
        opening = self.policy.get("opening")
        user = "SHOPPER REQUEST:\n" + str(self.policy.get("query") or "")
        if opening is not None:
            user += "\n\nOPENING OBSERVATION:\n" + _compact(opening)
        brief = self.brief()
        if brief:
            user += "\n\n" + brief
        user += "\n\n" + self.status()
        if opening is not None:
            self.records.append(opening)
            self.safe_observe("opening", {}, opening)
            _collect_ids(opening, self.seen_ids)
        self.set_cap()
        system = SYSTEM_PROMPT.replace("<READING_ENDS>", str(self.th('reading_ends'))).replace("<COMMIT>", str(self.th('commit')))
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        idle = 0
        failures = 0
        counter = 0
        for _ in range(MAX_TURNS):
            if self.finished or self.dead or self.steps_left() <= 0:
                break
            if self.time_left() < FORCE_SECONDS:
                self.force_finish()
                break
            if self.review_call:
                self.review_step()
                continue
            if self.order_attempts == 0 and (self.spent_out() or self.sweep_due()) and self.questions_done() and self.steps_left() > self.commit_steps_needed() and self.guarded(self.store_step, False):
                continue  # (before the commit the stop calls for: what these reads find can still change the choice)
            try:
                committed = self.deadline_due() and self.commit_now()
            except Exception as exc:
                self.trace.append({"kind": "commit_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
                committed = False
            if committed:
                continue
            if self.spent_out():  # no navigator or check is called any more: the code orders its best checked candidate (or declines)
                if not self.questions_done():
                    self.ask_planned(force=True)  # once its planned questions have gone out (each round asks one, or ends them)
                    continue
                if self.find_hop():  # a found hop is still opened, and a pending chain followed while late_room() allows
                    continue
                if self.guarded(self.store_step, False):  # store calls cost no money
                    continue
                self.force_finish()
                break
            if self.pending_reply:
                text, self.pending_reply = self.pending_reply, ""
                if self.send_reply(text):
                    continue
            if self.planner_on and self.ask_planned():
                continue
            if self.find_hop():  # the listing a requirement points at is the code's to find, before the navigator's turn
                continue
            message = self.infer(self.messages, self.tools_for_turn(), _route("navigator"), NAV_TOKENS)
            if message is None:
                failures += 1
                if failures >= 2:
                    self.force_finish()
                    break
                continue
            calls = []
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                counter += 1
                function = call.get("function") or {}
                calls.append({"id": call.get("id") or f"call_{counter}", "type": "function", "function": {"name": function.get("name"), "arguments": function.get("arguments") if isinstance(function.get("arguments"), str) else json.dumps(function.get("arguments") or {})}})
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = _compact(content)
            committing = ({self.roles.get("order"), self.roles.get("decline")} | self.cart_tools()) - {None}
            if self.planner_on and any(c["function"]["name"] in committing for c in calls) and self.question_ready():
                # a planned question goes out first: the navigator's calls are answered (not run, no step used), then asked
                self.trace.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in calls], "held": "question first"})
                self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                for c in calls:
                    self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": "Not run yet (no step used): a planned question to the shopper goes out first, since its answer may change the choice; read it, then send your call again."})
                self.ask_planned(force=True)
                continue
            if not calls:
                idle += 1
                self.messages.append({"role": "assistant", "content": content or "(no tool call)"})
                self.messages.append({"role": "user", "content": "Every reply must call at least one of the supplied tools. Continue.\n" + self.status()})
                if idle >= 3:
                    self.force_finish()
                    break
                continue
            idle = 0
            self.trace.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in calls]})
            self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            replies = self.run_calls(calls)
            for index, (call_id, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        if not self.finished and not self.dead:
            self.force_finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "spent": round(self.spent, 5), "seconds": round(time.monotonic() - self.started, 1)})
        return self.trace



def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            if not episode.finished and not episode.dead:
                episode.force_finish()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
        return episode.trace


__all__ = ["agent_main"]
