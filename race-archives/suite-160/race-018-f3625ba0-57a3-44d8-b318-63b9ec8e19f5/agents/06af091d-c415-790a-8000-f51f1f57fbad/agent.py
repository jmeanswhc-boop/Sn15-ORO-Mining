"""cam50: a shopping agent for the ORO store worlds.

A light model walks the store with the store's own tools and reads every result as the store wrote it; the agent's own
parse of the records rides along as a short code-derived note and acts on its own only while it reads the store well.
The agent asks the shopper one property per message, picked by a planner from the request's hints and the properties
the opened listings carry, and re-asks once in plainer words after a puzzled reply. Ordering is the default ending: a
held cart is ordered, and a decline goes out only after one review pass in which the model re-reads the request and the
shopper's answers and searches once more. Spend never removes a tool or a question: judgements move to the light model
past the soft line, and only the stop line ends the model's turns. Passing service faults are retried after short
waits inside the episode's own clock.

cam50 over cam47: the width of a question adapts within the episode. A message names up to KEY_WIDTH planned properties
of ONE product (never two products), short and in the store's own property names. A crowding complaint ("one at a
time") narrows every later message to one property and re-sends the complained properties one by one, first first; a
puzzled reply halves the width and re-sends once in plainer words; a shrug frees the property it covers and the next
message leads with a property never led before, so a shrug about the first property cannot hide the rest. Every message
the cap allows is spent, less the ones held for finish duties, and a message held for a swap that is no longer needed
when the order goes out carries one more planned question.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

MAIN_MODEL = "anthropic/claude-haiku-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"
SETUP_MODEL = "anthropic/claude-sonnet-5.5"
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
READ_MODEL = "anthropic/claude-haiku-5.5"

TASK_SECONDS = 265.0
FINAL_SECONDS = 285.0       # the finishing call alone may still go out until this many seconds (the task limit is 300 s)
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
OBS_CHARS = 16000
RAW_CUT = 700               # characters of one string of a result the model reads (a whole flat record fits)
RAW_CHARS = 13500           # characters of one result the model reads in all
NOTE_CHARS = 800            # characters of the agent's own parse note under a result
HISTORY_CHARS = 140000      # conversation size past which the older tool results are folded to their head
MAX_TURNS = 80
# The race spend governor. A race runs about ninety episodes on one $5 allowance of the miner's key, and every episode
# runs in a child process of its own, so only lines kept inside one episode can hold the run under the cap. The
# validator's reasoning judge and the shopper simulator bill the same key, about $0.008 per episode (RESERVE).
# Spend is booked per call as the billed usage.cost; only a reply without a cost is booked at a size estimate (size_cost).
# The hard line scales with the step cap: HARD_SHORT up to SHORT_CAP steps, rising linearly to HARD_LONG at LONG_CAP,
# plus BUNDLE_EXTRA per product of a bundle beyond the first; the soft line is SOFT_SHARE of it, the stop line STOP_EXTRA
# above it (single product):
#   step cap      <=40    50     60     70     >=80     (cam47 lines)
#   soft          .024   .026   .027   .029   .030     past it: every judgement on the light model
#   hard          .040   .043   .045   .048   .050     past it: optional looks go, old results fold shorter; search and questions stay
#   stop          .060   .060   .060   .060   .060     past it: no model call; the agent finishes in code (a held cart is ordered)
# The lines cut the tail, not the median: an episode of race-11 mechanics books about $0.037 (worldsim race11 mode;
# cam's live race-11 spend was $3.8-3.9 a run, about $0.035 an episode on the model), so a hard line of .030 would stop
# the typical episode. Arithmetic for the run: 90 * mean + 90 * RESERVE <= $4.6 needs a mean of at most $0.043;
# race-11 mechanics give 90 * .037 + .72 = $4.05. The bound with every episode at its stop line (2/3 short caps, 1/3
# long caps of 50-115 averaging about .064, bundles +.004 to +.012) is about .058 an episode, $6.0 a run: the stop lines
# cap a runaway episode; they do not by themselves hold an all-runaway run under $5.
HARD_SHORT = 0.040          # hard line for a step cap of SHORT_CAP or less
HARD_LONG = 0.050           # hard line for a step cap of LONG_CAP or more
SHORT_CAP, LONG_CAP = 40, 80
BUNDLE_EXTRA = 0.004        # added to the hard line for every product of a bundle beyond the first (three at most)
SOFT_SHARE = 0.6            # soft line = this share of the hard line: past it every judgement reads on the light model
STOP_EXTRA = 0.008          # (kept for the table above; the stop line is STOP_LINE)
STOP_LINE = 0.06            # past this spend the model is never asked again and the agent finishes in code (orders a held cart)
# cam47: spend never takes a tool or a question away. Past the soft line the judgements go to the light model; past the
# hard line the optional looks go too and old results are folded shorter; only the stop line ends the model's turns.
RESERVE = 0.008             # per-episode spend of the validator's judge and the shopper simulator on the same key
TURN_SLACK = 8              # model tool turns per episode at most max_steps + this (a turn is a paid call; holds take turns without steps)
LLM_FAILS = 6               # failed inference requests in a row (429, 5xx, network, empty) after which the model is given up
PASSING = (408, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529)   # statuses worth a retry after a short wait
WAIT_BUDGET = 30.0          # seconds of retry waits one episode may spend in all (inference and store together)
LONG_TEXT = 450            # free-text characters a result row carries on average from which the store is wordy (race 15: about 520)
WORDY_SHARE = 0.85          # the spend lines in a wordy store, as a share of the usual ones (the checks read light there from the start)
PAGE_CHARGE = 0.0015        # a page the agent opens itself for fitment text: its text rides in every later turn, so it is booked as spend
CACHE_READ, CACHE_WRITE = 0.1, 1.25   # prompt-cache prices as shares of the input price (a prefix sent before is read, the rest is written)
# $ per million tokens (in, out) of the models this agent calls, as billed through the proxy, for the size estimate that
# books a reply carrying no cost (a reply with a cost is booked at its billed cost only)
PRICE_IN_OUT = {"anthropic/claude-haiku-5.5": (0.1, 0.5), "anthropic/claude-sonnet-5.5": (2.0, 10.0), "openai/gpt-6-luna": (2.0, 8.0)}
CHARS_PER_TOKEN = 4.0
LONG_WORLD = 40             # a step cap at or above this is a long world: the store changes more than once and the shopper has
                            # more to reveal; the extra care it buys is bounded so it can never delay the first order
LONG_FINAL_CHECKS = 2       # in a long world the final look may run before a second finishing call with a new cart or new requirements
REPAIR_RESERVE = 7          # steps kept after the first order for the repair of a refusal (a swap, a re-read, the order again)
EXTRA_ROOM = 8              # steps beyond the finish reserve that any optional care (holds, repeated looks) needs to be worth a step
ABSTAIN_BLOCK = 3           # steps after a refusal during which a decline is never sent while an opened sheet fits
POINTER_SEARCHES = 2        # zero-result searches a pointer-type need may spend before the category and the budget decide
POINTER_OPENS = 3           # pages opened per round while resolving a pointer-type need
CHANGE_REVIEWS = 2
BUNDLE_READS = 5           # reads, searches and page turns the agent takes itself per bundle product
FINISH_READS = 12          # reads and page turns the finish in code may still take for bundle products without a fit
BUNDLE_PAGES = 3           # pages the agent turns itself per bundle product without a fit
WALK_STEPS = 4             # steps the cheapest descent may spend on its own searches, pages and reads
WALK_FROM_STEPS = 14       # the descent starts only with this many steps left: later it would starve the finish
PAGE_FROM_STEPS = 12       # a short worded list is paged to its end by the agent only with this many steps left
SWEEP_PER = 4              # sheets the agent opens itself after a search that showed unread candidates (fewer when steps are short)
SWEEP_SHARE = 0.3          # the sweeps' steps per episode at most, as a share of the step cap
SWEEP_ROOM = 9             # steps beyond the finish reserve a sweep leaves for the model's own turns
WALK_READS = 3             # sheets opened per descent round
REFILL_STEPS = 4           # after a sellout on a cheapest request: pages turned and sheets opened to find another fit at the same price
FIT_BY = 0.4               # pace: a verified fit should be known by this share of the step cap (about 15 of 38)
COMMIT_BY = 0.6            # pace: the cart should be ordered by this share of the step cap (about 23 of 38), never later than cap - REPAIR_RESERVE
COMMIT_ASKED_BY = 0.5      # pace: once the questions are done, a known fit goes into the cart by this share of the cap (about 19 of 38)
EXPLORE_VALVE = 3          # past the commit target one reading in this many held ones goes through
COMMIT_STEPS = 12          # with this many steps left (less in short episodes) and a fit known, nothing but committing is worth a step
COMMIT_SHARE = 0.32        # the same, as a share of a short step cap
LATE_READS = 2             # sheets a reply may open past the order mark
DECLINE_MARK = 0.7         # on a step cap of 50 or more, no decline before this share of the cap while a product is unfound
UNSTATED_QUESTIONS = 3     # questions when no message cap is stated or reported (race 14: cap 3; race 13: cap 2, refused with "budget_exceeded")
KEY_WIDTH = 5              # attribute keys the first question names; the width then follows the replies (race 16: lists were
                           # refused, so a complaint narrows to one; race 17: lists were answered, need named anywhere in them)
QUESTION_CHARS = 120       # a question longer than this drops its last keys (long, wordy messages puzzle the shopper)
SIMPLER_ASKS = 2           # questions re-sent in plainer words after a puzzled reply, per episode
KEY_READY = 3              # opened sheets of a product before its question goes out (until the asking mark)
ASK_TIME = 0.6             # ... or once this share of the episode's clock is left (slow model calls must not starve the questions)
ASK_MARK = 0.45            # share of the step cap after which a product's question goes out with whatever keys are known
QUESTION_CAP = 5           # questions per episode at most, one property each (race 12: top agents asked 4.85)
JUDGE_TOKENS = 1200        # the least output cap of a reading call (the reading models think first, and the thinking counts)
AUTO_ORDERS = 2            # orders the agent sends itself after repairing the cart, per episode
ONE_FIELD = ("Label keys that differ only in case, spaces, hyphens or underscores are one field, so 'clock shape: rectangle' beside "
             "'clock_shape: rectangular, square' is one field that lists square and states it; a list value that contains the required "
             "value states it; only a key with a dotted namespace ('clothing.material' beside 'material') is a separate claim that can clash.")

SYSTEM_PROMPT = """You are a meticulous shopping assistant. You serve one shopper inside a simulated store, and you can act only through the supplied tools. The task is scored 1 or 0. It scores 1 only if your finish is fully right: the ordered item meets every requirement the shopper stated, every hidden need the shopper reveals when asked, and every change the shopper announces while you work; the quantity and the budget are right; the cheapest (or best) choice is made when the request asks for it; and every rule the request states is followed. If nothing in the store can meet everything, the right finish is to decline, exactly the way the request says.

THE WORLD
- Every tool call uses one step, including calls that fail or return errors, and the number of steps is limited. Model thinking is free; steps are not.
- You may put several tool calls in one reply when none depends on another's result (for example, opening the details of several listings at once). They run in order. Never put an order or a decline together with other calls.
- If a tool shows the details of several listings in one call (a compare tool), read candidates with it in batches: one step for several listings.
- Every result shows the store's own text first, then a short note marked "agent's own parse (code-derived)". That note is written by code and can be incomplete or wrong for this store's way of writing records; where it and the store's text disagree, the store's text is what counts, and listings the note misses are still there.
- Tool names, arguments and the shape of results differ from store to store. Read the tool descriptions and the JSON schemas carefully and use exactly the argument names and allowed values they define. Learn from the first results which tool shows which facts (price, stock, detailed properties).
- Find out early which category labels the store uses for the requested kind of product (they appear on listings). When neighbouring labels exist, the right one is the label that matches the request's wording as a whole; search and buy within it.
- Search tools can be limited: a capped number of visible rows or pages, results sorted by id rather than by relevance or price, every word required, filters that only work in one mode. Compare the reported total with the rows you can actually see, and narrow (more specific words, category or price filters, other pages) until you have seen the whole relevant set.

THE SHOPPER
- The shopper holds needs that the request does not spell out. A need comes to light only when a message asks about exactly that property, and each message is answered about the first thing it asks. So one property per message, as a short plain question that names the property; an example value or two may help. No instructions, no lists, no two questions at once.
- "Don't mind", "no preference", "whatever you think" free that property. A stated value or limit becomes a firm requirement from then on, exactly as spoken.
- Never send a message that only acknowledges, apologises or confirms ("understood", "I'll ask one at a time"): it spends a message and reveals nothing. A message with two questions is answered about the first only.
- When messages remain, your questions are planned and sent for you once two or three sheets of the requested kind have been read; the answers appear here with their meaning. Take every revealed need as a firm requirement and search again with it.
- If you ask yourself: first what the request's story points at (who will use it, how it must look, where it goes), then the sheet properties whose values differ between the cheapest candidates. Never ask about something the request already fixes, and never exceed the message budget.

RULES IN THE REQUEST
- The request can set rules of procedure: how many listings to compare, what to check, what to tell the shopper, how to decline. Each one is graded like a requirement: follow it literally and completely.
- Something the shopper would merely like ("ideally", "would be nice") is not a requirement; use it only to choose among listings that already meet everything, and never against a cheapest rule.
- When only the shopper can supply something (no price limit stated, or two wishes that may not both be possible), ask for it before choosing.

CHANGES WHILE YOU WORK
- The shopper's words can appear inside any tool result (a list of sentences, a note field) or in the answer to an order. A new budget, a dropped or added requirement, a price rise, a stockout: each is binding at once. Re-check your pick and your backups, and when the cheapest is wanted, choose again over the new set.
- An order call that does not end the episode has not bought anything, whatever its text says. It was refused, it failed, or it waits for a further step (a confirmation with a token it returned). Read why, apply every change, fix the cart, re-check stock and price, and finish again. A price change alone does not make a listing unacceptable while price times quantity stays within the limit.
- Stock and price can also change silently: re-read the pick with a tool that shows them right before every order attempt.

HOW A LISTING IS JUDGED
- A requirement counts as met only when the listing's own property for it states the value. Labels vary in spelling and prefixes; any label that names the property counts. Values match by meaning: one member of a comma list is enough, singular and plural are the same, close synonyms count, case does not. A word that merely begins the same way does not count, nor does a broader or different class (girls' is not women's; plain unisex is not female).
- The title or description is evidence only when no label covers the property. A label that states something else beats the title.
- A value under a different, related property does not count. Numbers are compared after converting units; limits include their edge; a range counts only when the whole range meets the limit. Exclusions and closed lists are strict.
- Only the stated requirements, the revealed needs and the request's rules decide. Never drop a listing over something nobody asked for (an age group, a size range, an audience, a brand or a style the request does not name): when such a property could matter, ask the shopper about it while messages remain; otherwise it is no reason to reject a listing that meets everything asked.
- The listing must sit in the category that matches the requested kind as a whole; never demand that a property repeat the category's name. In-stock means in stock as last reported; price times quantity must be within the limit (equal is fine).
- A comma list that contains the required value simply matches. Evidence is mixed only when the listing's own labels for one property state incompatible values; then, if no cheapest rule decides, prefer the listing whose evidence is clean. """ + ONE_FIELD + """

CHEAPEST OR BEST
- "The cheapest that fits" means the lowest price among ALL fitting listings. One fitting listing is not an answer. Lower the price cap until every remaining match is visible, sort what you see by price, and check the candidates in rising price order: the first that fully fits, with every cheaper one checked and rejected, is the pick. A cheaper listing is rejected only by a stated requirement or a revealed need its own record fails, never by your own sense of what suits the story. Keep the next fitting ones as backups.
- "The best by some measure" means comparing every fitting listing on that measure and taking the extreme; a listing that states no value for the measure cannot win.

FINISHING
- Keep a ranked list of verified fits with at least one backup, because the first pick can sell out at checkout.
- Before every order attempt: the cart holds exactly the lines the request needs (one line, the pick at the required quantity, unless the request needs several products that go together); their stock and price were re-read after the last cart change; they are in stock and within the limit.
- Things the request tells you to do or say at the end (confirm, tell the total, a closing word) are carried out for you right before the finish when they appear under [to do before finishing]; one message is kept free for them.
- Decline only when a thorough search has found nothing that meets everything. A decline is worth 0 whenever some listing could have met the stated requirements and the revealed needs, so between a fitting order and a decline, order. When the request forbids substitutes, the decline remark names no listing: no ids, titles or prices. The agent itself refuses a decline while an opened listing still fits, and it orders that listing for you when you insist.
- A refusal that only states a new budget is not a rejection of the item: when the held line's price is within the new budget, the same order is sent again (the agent does this for you); when it is over, the cheapest opened listing that fits under the new budget replaces it.
- A need that points at another listing ("the value shown on the page for the <kind> from <shop> that comes with <properties>", or "anything other than" such a value) is resolved by the agent: it finds that page, reads the value there and turns the need into a plain value or exclusion. The shop and the properties that pick out that other listing are never requirements on the item you buy; do not search for them.
- Get the first order attempt in while at least seven steps remain, so that a refusal can still be repaired. From seven steps left only cart, reading and finishing tools remain. Change the cart once, not back and forth.
- Every reply carries at least one tool call. Keep the words short: the requirements in force, the verified fits with price and stock, the next move."""

SETUP_PROMPT = """Before a shopping assistant starts, read the shopper's request and the store's tool list below. Return ONE compact JSON object (no indentation) and nothing else, with these keys:

"product": the kind of product wanted, in the request's own words.
"quantity": units of one listing to order (an integer; 1 unless the request asks for several; a pair or a set is 1).
"requirements": a list of firm conditions on the product, each "property: constraint" with the shopper's exact value, limit, exclusion or closed list. Not the price, stock, quantity or category, not wishes, not delivery or arrival times; words that the product's or category's own name already carries (the season of a "winter jacket") are covered by it, but an occasion or use the request states outright as what the product is for, when the product's name does not carry it, is a requirement (boots that "are for hiking" give "occasion: hiking"); the person the item is meant for is story unless the request demands a value for it. A property the request gives a value for with words such as should be, has to, got to, must, only or nothing else is a requirement however casually it is said, also when the shopper adds that they will check it themselves; a stated value never goes under "hints".
"hints": a list of objects {"property": ..., "likely": ..., "why": ...}: the product properties the request's story points at without stating a value, named as a listing would name them (for example "water resistance" when the shopper will swim with the watch, "pile height" when a door must open over the rug), with the value the story implies in "likely" (or "" when it implies none) and the request's own words in "why". Hints are what to ask the shopper about first and what to prefer among items that already fit; they are never a reason to reject an item. Empty when the request is plain.
"price_cap": the largest total price allowed, as a number in the unit the request uses, or null.
"currency": the price unit the request uses, or null.
"in_stock_only": true when only items on hand count.
"priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only when it explicitly asks for the best by some measure; otherwise "none". Remarks that money is tight or that every penny counts are not a request for the cheapest; they only explain the price cap.
"priority_detail": for "best", what "best" is measured by; otherwise "".
"message_budget": the number of messages the request allows the assistant to send, as an integer, or null.
"step_budget": the number of moves or steps the request allows, as an integer, or null.
"rules": a list of every procedural rule the request states for the assistant (how many listings to look at, what to check or tell, how to decline, what not to do, what to include), quoted word for word. Remarks about what the shopper will do themselves are story, not rules.
"soft_wants": wishes the shopper does not require (ideally, preferably, if possible); never in "requirements".
"ask_first": a list of things the agent can only learn from the shopper and should ask about before choosing: for example the budget when no price limit is stated, which of two wishes matters more when the request says both may not be possible (then also put both wishes in soft_wants, not in requirements), or anything the request says to check with the shopper. Empty when none.
Every sentence of the request must land somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or pure story. When unsure whether a sentence is a rule, put it in "rules".
"duties": a list of the things the request tells the assistant to do or say at some moment, as opposed to conditions on the product: each an object {"do": the request's own words, "when": "finish" (before ordering or declining), "order" (only before ordering), "decline" (only before declining), "replace" (before swapping one item for another) or "anytime", "tell": true when doing it means sending the shopper a message (telling, confirming, checking with them, summing up, a closing word)}. Empty when the request states none.
"companions": when the request needs more than one product bought together, a list of the other products, each "product: how it must go with the main one"; otherwise an empty list.
"items": when the request needs more than one product, one object per product {"product": its name as the request says it, "quantity": the units of it the request wants (1 unless it says more)}, the main product first; otherwise an empty list.
"decline_no_names": true when the request's rule for declining forbids offering, naming or suggesting substitutes or alternatives; false when it asks you to name something; null when the request gives no rule.
"roles": an object naming the tool for each role, using the exact tool names from the list, or null when no tool fits:
  "search", "details", "view", "message", "cart", "remove", "cart_view", "order", "decline".
  "message" is the tool that sends a message to the shopper. "cart" is the tool that adds to the cart (or adds and removes); "remove" is a separate tool that takes listings out of the cart, if there is one. "order" is the tool that buys or checks out. "decline" is the tool that ends the task without buying.
"cart_modes": when the cart tool takes a mode argument, {"add": the value that puts a listing in, "remove": the value that takes one out}; null otherwise.
"finish_args": when one tool both buys and ends without buying, an object {"order": {...}, "decline": {...}} with the argument values that select each, for example {"order": {"mode": "buy"}, "decline": {"mode": "stop"}}; null otherwise.
"""

INTERPRET_PROMPT = """You read what a shopper just said to a shopping assistant and turn it into changes of the requirements. Answer with ONE JSON object and nothing else, with these keys:
"meaning": one short sentence saying what the shopper means.
"needs": a list of new firm requirements on the product, each written as "property: constraint" with the exact value, limit, exclusion or closed list the shopper gave.
"no_preference": a list of properties the shopper said do not matter.
"dropped": a list of earlier requirements the shopper no longer wants.
"new_budget": a new total price limit as a number, whether higher or lower, even when the shopper calls it good news; else null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), else false.
"price_moved": true when the line says the price of an item changed, else false.
"requirements_now": the complete list of firm requirements on the product as they stand after everything the shopper has said so far, re-derived from the request, the earlier shopper lines and this one, each as "property: constraint". Keep every earlier requirement unless the shopper dropped or replaced it; for a replaced value keep only the newest one; add every newly revealed need. Do not include price, stock, quantity or the product category, and never add a requirement from the request's story or from your own reading of the situation: only what the request states outright and what the shopper has said.
"reply": when the shopper asks the assistant something or asks for a confirmation, a short, plain reply to send back that answers it and names the change in substance (for example confirming the new value); otherwise "".
"duties": things the line newly tells the assistant to do or say before finishing or before replacing an item, each {"do": the words, "when": "finish", "order", "decline", "replace" or "anytime", "tell": true when it means messaging the shopper}; otherwise [].
"messages_left": when the line says how many more messages the assistant may send from now on, that number; else null.
"steps_left": when the line says how many more moves or steps the assistant has from now on, that number; else null.
"unavailable_items": the listing ids or names the line says can no longer be bought, as written; else [].
"unclear": true when the shopper did not understand the assistant's message or asks for it to be put another way, else false.
"one_at_a_time": true when the shopper asks for one question or one point per message, else false.
One line can carry several changes: report every one. Insistence idioms ("X isn't up for trade", "X is non-negotiable", "it has to be X") mean the shopper firmly wants X, never that X is excluded. A requirement replaced by another is both dropped (the old one) and a new need (the new one). When the shopper says which of two wishes matters more, the preferred wish is a need. When the shopper states a budget, it is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and charitably. Folksy, figurative or roundabout phrasing that names a value means the shopper firmly wants that value; treat a value as excluded only when the shopper clearly says to avoid it. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit. A new budget replaces the old one whether it is higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference, or anything works, put the property asked about in no_preference."""

VERIFY_PROMPT = """You check one listing for a shopping assistant before it orders it; the order is worth 1 only if the listing meets every requirement below. Judge it from the records shown alone, by these reading rules:
- Labels vary in spelling and prefixes, and any label naming the property counts; a value inside a comma- or slash-separated list counts; close synonyms and the other number of a word count. A look-alike word or a broader or different class does not. """ + ONE_FIELD + """
- The title or description decides only where no label covers the property, and a label that states something else beats the title.
- A value under a different, related property does not count.
- Numbers are compared after converting units, limits include their edge, and a range must lie wholly inside.
- Exclusions and closed lists must be respected.
- The item must be the requested kind of product; when listings carry a category label, it must be the category whose label matches the request's wording as a whole, not a narrower or neighbouring one.
- When the item's own property for a requirement states a different value, the requirement fails, even if the title suggests otherwise.
- Price times quantity must be within the budget (equal is fine). If the records show it out of stock, it fails.
- A mere wish ("ideally", "would be nice") listed as not required is not a requirement, and a property no requirement names (an audience, a size range, a brand) never fails the item.
When records disagree (for example on price or stock), the newest record counts.
Return ONE JSON object and nothing else: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement that is not met or not shown]}. Use fits=false only for a requirement that the records clearly fail, or that no record addresses at all; do not fail an item for a missing record of stock or price."""

PLAN_PROMPT = """You choose the questions a shopping assistant still asks the shopper. The shopper holds hidden needs: each is the value of one property of the item, and it comes to light only when a message asks about exactly that property; a shopper may answer only the first property a message names, and a property that does not matter earns a shrug. Each message names a short list of ONE product's properties (one property once the shopper wants one at a time) and only a few messages can go out, so the order decides everything: the properties first in your list must be the ones most likely to hold a hidden need. Choose up to QUESTION_COUNT different properties, best first:
0. first, anything listed below as MUST BE LEARNED FROM THE SHOPPER that has not been asked yet; ask it plainly. For the budget, ask how much they want to spend at most. For a choice between two wishes, name both values with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?";
1. then properties that the listings' details below actually carry, preferring (a) a property the story in the request hints at indirectly (who the item is for or who will wear or use it, how it must look in photos or to others, where or how it will be used), and (b) properties that describe what the product is like and whose values vary across the listings (who it is made for, pattern, shape, type or style variant, closure, length, function) rather than logistics (warranty, condition, pack size, variant or size lists, years). When CORE PROPERTIES are listed below (most of the listings opened carry them), choose only among them. "Who it is for" is the property that records the wearer or user (a gender, age group or audience label): when the story names the person the item is for and the listings carry such a property with differing values, ask it early.
Never pick a property the request already settles (its stated requirements, the price, stock, quantity), never repeat a property already asked, never spend two questions on near-duplicate properties (two kinds of style, two spellings of one property), and avoid properties no listing carries. Use the property names as the listings name them, in a few plain words. Name the product each property belongs to, as the request names it; when the request needs several products, spread the questions over the products (each product at least once while questions last, the main one first). The agent words every question itself in one fixed short form naming the property and the product, so give no question text, except for an item of step 0, whose question you write out in full (mark it "first": true).
Return ONE JSON object and nothing else: {"questions": [{"property": "...", "product": "...", "first": false, "question": ""}, ...]}."""

AUDIT_PROMPT = """A shopping assistant is about to decline, saying that no listing meets every requirement. Check the listing records below against the requirements in force by these reading rules: labels vary in spelling and any label naming the property counts; a member of a comma list, the other number of a word and close synonyms count, a broader or different class does not; """ + ONE_FIELD + """ the title or description decides only where no label covers the property; numbers are compared after converting units and limits include their edge; the price must be within the budget and the item must not be shown out of stock.
Return ONE JSON object and nothing else: {"possible_fits": [ids of listings that appear to meet every requirement, best first, at most 5], "reason": "one sentence"}."""

WRAP_PROMPT = """A shopping assistant is about to ACTION. The request (and the shopper since) told it to do or say certain things at this point; they are listed below with the messages it has already sent. Decide which of them are still undone and write ONE short, friendly message to the shopper that carries out all of those at once: when it must tell the basket or the total, name each item with its price and the total; when it must confirm, say what it is about to do and ask them to confirm; when it must give a closing word, add it. Do not ask about product properties and do not name listings when the action is a decline that must not suggest substitutes.
Return ONE JSON object and nothing else: {"message": the message to send, or "" when every listed thing is already done, "why": "one short sentence"}."""

REVIEW_PROMPT = """Something just changed while a shopping assistant works (stated below): a new budget or requirement, an item that can no longer be bought, or a refused order. Look at the cart and the listings it has opened, judged by their own records against the requirements as they stand now (labels name properties; a member of a comma list counts; numbers are compared after converting units; price times quantity must be within the budget; an item shown sold out or refused as gone cannot be bought).
Return ONE JSON object and nothing else: {"cart_still_right": true or false, "why": "one short sentence", "take_instead": [ids of opened listings that meet everything now, best first (cheapest first when the cheapest is wanted), at most 3]}."""

TOGETHER_PROMPT = """A shopping assistant is about to order several products that the shopper needs to use together. From the records below, judge whether the products in the cart work with each other as the request requires (model, size, fitting, connector or standard named on each one's own records) and whether every product the request needs is there.
Return ONE JSON object and nothing else: {"works_together": true or false, "problems": [short strings], "missing": [products the request needs that the cart lacks]}."""

FINAL_PROMPT = """Final check: a shopping assistant is about to ACTION. The task scores 1 only if the ordered item meets every requirement the request states and every need the shopper revealed, and if every step the request (or the shopper since) explicitly demanded has been carried out. Read the request, the requirements in force, the cart lines with the facts seen about them, the steps already recorded with whether they were done, and the exchange with the shopper. Name only what is grounded in these texts: a condition is unmet only when a cart line's own facts fail it or say nothing about it; a step is undone only when the request's or the shopper's own words demand it at this point and nothing recorded shows it done. Do not invent steps the texts do not ask for; when the texts ask for nothing more, return empty lists.
Return ONE JSON object and nothing else: {"unmet_conditions": [short strings, at most 4], "undone_steps": [short strings in the request's own words, at most 2], "ok": true or false}."""

PRICE_FIELD_PROMPT = """Below are a few rows of a store's search results, each showing two fields. Which field is the listing's price (what one unit costs)? Return ONE JSON object and nothing else: {"price": the exact field name}"""

BLIND_PROMPT = """A shopping assistant could not read the store's listing records by code, so you read them. Below are the shopper's request, the requirements in force and the raw text of the store's recent results. Name the one listing that meets every requirement (each property as its own facts state it, in stock, price times quantity within the budget, and the cheapest of those that fit when the request asks for the cheapest), or none when no listing shown does. Return ONE JSON object and nothing else: {"pick": "<the listing id exactly as written in the results>" or null, "why": "<one short sentence>"}"""

POINTER_PROMPT = """A shopper's requirement takes its value from another listing's page rather than stating it. Below are the requirement in the shopper's own words and the page that was found for the listing it points at (that page's labelled facts, remarks and description). Work out the concrete value the shopper requires for the named property:
- when the requirement says the value must equal (match, be the same as) what that page shows, the value is the page's own value for that property;
- when it says "anything other than", "unlike" or "different from" that value, the page's own value is EXCLUDED;
- when it says the value must pair with, go with, work alongside or suit the page's value, apply the pairing rule that the page itself states (in its remarks or description) to the page's own value for the property; the result is the required value;
- when the page carries no value for the property but names another listing to consult (or the requirement says "or else on the listing that page points to"), name that other listing in "follow" and leave "value" empty.
Return ONE JSON object and nothing else: {"value": the concrete value as the page spells it, or "", "excluded": true when that value is to be avoided, "follow": "" or the id, title or description of the other listing to open next, "why": "one short sentence"}."""

# cam50: how a reply that drew no firm value reads the message it answers. A crowding complaint wants one point per
# message; a puzzled reply wants the same point again in other words. The reader's flags decide first; these cues only
# back them up on replies with no firm value.
CROWD_CUE = re.compile(r"(?i)\bone (?:at a time|by one|\w+ per)\b|\bsplit\b|\bat once\b|\bsingle\b")
PUZZLE_CUE = re.compile(r"(?i)\bagain\b|\bonce more\b|\banother (?:way|pass)\b|\bsimpl|\bplain|\brepeat")


def single_name(product: str) -> str:
    """The product as one plain noun phrase for a question: no leading article, no trailing mark, and only the first of
    alternatives ('pants or jeans' -> 'pants'), since a question naming two things reads to the shopper as two questions."""
    text = re.sub(r"^(?:the|a|an|your|some)\s+", "", re.sub(r"[\s?.!:]+$", "", str(product or "").strip()), flags=re.I)
    first = re.split(r"\s+or\s+|\s*/\s*", text, maxsplit=1)[0].strip()
    return first or text or "item"


def pointer_from_need(need: str) -> dict[str, Any] | None:
    """A need whose value sits on another listing's page: "<property>: ... the value shown on the page for the <kind> from
    <shop> that comes with <attribute value> and <attribute value>, or else on the listing that page points to", in any
    of its forms (match it, pair with it, anything other than it). Returns the parts the agent needs to find that page:
    the property, the shop or brand word, the kind, the attributes, whether the value is excluded, and the whole text."""
    text = str(need or "").strip()
    prop, sep, rest = text.partition(":")
    if not sep:
        prop, rest = "", text
    low = rest.lower()
    end = r"(?:,(?!\s*with\b)|\.(?!\d)|\?|$| or else| otherwise|\s*\()"
    chained = re.search(r",\s*(?:following|then following|and following|then follow|and follow)\b", low)
    if chained:                                 # "..., following each listing it names": the description runs to that clause, commas and all
        end = r"(?:,\s*(?:following|then following|and following|then follow|and follow)\b)"
    page = re.search(r"(?:shown|given|listed|stated|written|found|appears?|displayed)\s+(?:on|at|in)\s+the\s+(?:page|listing|sheet|record|product page)\s+(?:for|of)\s+(?:the\s+)?(.+?)" + end, low)
    if not page:
        page = re.search(r"\bthe\s+(?:page|listing|sheet|record)\s+(?:for|of)\s+(?:the\s+)?(.+?)" + end, low)
    if not page:
        page = re.search(r"\bthe\s+(?:page|listing|sheet|record)\s+((?:for\s+)?[a-z][\w&\-]*'s\s+(?:listing|page)\b.*?)" + end, low)
    if not page:
        return None
    body = page.group(1)
    owner = re.search(r"\b([A-Za-z][\w&\-]*)['\u2019]s\s+(?:listing|page|offer)", rest[page.start(1):page.end(1)])
    shop = owner or re.search(r"\b(?:from|by|sold by|at|of)\s+([A-Za-z][\w&\-]*)", rest[page.start(1):page.end(1)], re.I)
    if owner:                                   # "X's listing of the <kind> - the one with <attribute>": the kind and attributes follow
        tail = body[owner.end():]
        kind_m = re.search(r"^\s*(?:of|for)\s+(?:the\s+)?([a-z][\w\s\-]*?)(?=\s*[\u2013\u2014,;-]|\s+(?:the one|that|which|with)\b|$)", tail)
        attrs_m = re.search(r"\b(?:the one\s+)?with\s+(.+?)(?=\s*[\u2013\u2014;]|\s+or\s+wherever|$)", tail)
        body = (kind_m.group(1) if kind_m else "") + ((" with " + attrs_m.group(1)) if attrs_m else "")
    kind = re.split(r"\s+(?:from|by|sold by|that|which|with)\b|,", body, 1)[0].strip()
    kind = re.split(r"\s+(?:you|at)\b", kind, 1)[0].strip()   # "... you'll find at <shop>": where it is sold, not what it is
    attrs_text = re.search(r"\b(?:that|which)?\s*(?:comes?|come|is|are|listed|sold|offered)?\s*with\s+(.+)$", body)
    attrs: list[str] = []
    if attrs_text:
        attrs = [a.strip() for a in re.split(r"\s+and\s+|,\s*", attrs_text.group(1)) if a.strip()]
    excluded = bool(re.search(r"anything other than|anything but|other than|unlike|different from|not the same as|anything except|avoid|\bnot\s+(?:whatever|what(?:'s| is)|the (?:value|one)|anything)|\bnever\b|\bnothing like\b", low))
    pairing = bool(re.search(r"works? alongside|pairs? with|goes with|go with|suits?|matches?|complement|alongside|partner", low))
    # a figure the value may stray from the page's own ("inside 1.41 inches of whatever ..."), or a side it may lie on
    band = re.search(r"\b(?:inside|within|no more than|at most)\s+(\d+(?:\.\d+)?)\s*([a-z%\"']*)\s+(?:of|from)\s+(?:whatever|the value|what)", low)
    side = "min" if re.search(r"\bor (?:higher|more|above|greater|bigger|larger|longer|up)\b|\bat least (?:whatever|the value|what)", low) else \
        "max" if re.search(r"\bor (?:lower|less|below|smaller|shorter|under|down)\b|\bat most (?:whatever|the value|what)", low) else ""
    return {"prop": prop.strip().lower(), "text": text, "shop": shop.group(1) if shop else "", "kind": kind, "attrs": attrs,
            "excluded": excluded, "pairing": pairing, "value": "", "done": False, "tries": 0, "opened": [], "page": "",
            "band": (float(band.group(1)), band.group(2)) if band else None, "side": side}


CODE_TOKEN = re.compile(r"(?<![\w-])[A-Za-z]{2,5}\d[0-9A-Za-z]{1,8}(?![\w-])")   # a model or part code: letters, then a digit, then letters or digits


def code_family(token: str) -> str:
    """The letter run a code starts with (its family): 'luk' for LUK1720."""
    match = re.match(r"[A-Za-z]+", token)
    return match.group(0).lower() if match else ""


def code_covers(listed: str, wanted: str) -> bool:
    """Whether a code written on a listing covers the code the shopper named: equal codes, or a listing code whose x runs
    after the first digit stand for any character (SIR21xx covers SIR2120)."""
    family = code_family(listed)
    rest = listed[len(family):]
    if family != code_family(wanted) or len(rest) != len(wanted) - len(family):
        return False
    return all(a.lower() == b.lower() or (a in "xX" and i > 0) for i, (a, b) in enumerate(zip(rest, wanted[len(family):])))


def money_unit_words(query: str, currency: Any) -> set[str]:
    """The words the request writes its money in: the setup's currency and the word that follows the request's first
    figure when it is not a plain word of the sentence (usd-cents, ¢, credits)."""
    words = {w for w in re.findall(r"[^\s\d.,;:!?()]+", str(currency or "").lower()) if len(w) > 1 or not w.isalpha()}
    after = re.search(r"\d[\d,.]*\s*([^\s\d.,;:!?()]+)", str(query or ""))
    if after and (re.search(r"[^a-z]", after.group(1).lower()) or after.group(1).lower() in words):
        words.add(after.group(1).lower())
    return words


def lone_amount(text: str, units: set[str]) -> float | None:
    """The one figure a line writes in the request's money unit ("would 4344 usd-cents be workable"); None when it
    writes none or several different ones."""
    if not units:
        return None
    found = {float(m.group(1).replace(",", "")) for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*([^\s\d.,;:!?()]+)", str(text or ""))
             if m.group(2).lower().strip("'\"") in units}
    return found.pop() if len(found) == 1 else None


_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

COUNT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def stated_message_cap(text: str) -> int | None:
    """The number of messages a request allows, read from its own words: a count right before 'messages', 'questions'
    or 'times' in a sentence about messaging or asking the shopper. The least such count; None when no sentence says."""
    found = []
    count = r"(\d{1,2}|" + "|".join(COUNT_WORDS) + r")"
    for sentence in re.split(r"(?<=[.!?;])\s+|\n", text):
        low = sentence.lower()
        if not re.search(r"messag|question|\bask|answer|\bping|\btext|contact|reach out|get back", low):
            continue
        for match in re.finditer(r"\b" + count + r"\s+(?:[a-z-]+\s+){0,2}?(messages?|questions?|times|pings?|texts?|repl(?:y|ies))\b(?!\s+(?:at a time|at once|per |each|in (?:a|one|each) ))", low):
            word = match.group(1)
            value = int(word) if word.isdigit() else COUNT_WORDS.index(word)
            if match.group(2).startswith("time") and not re.search(r"messag|\bping|\btext|contact|\bask", low):
                continue
            found.append(value)
    return min(found) if found else None


def status_code_reply(observation: Any) -> bool:
    """A message result whose only text is one bare token ('CAPPED', 'budget_exceeded'): a store status code, never the
    shopper's words (a shopper always answers in a sentence)."""
    texts = [observation] if isinstance(observation, str) else [v for v in observation.values() if isinstance(v, str)] if isinstance(observation, dict) else []
    texts = [t.strip() for t in texts if t.strip()]
    return len(texts) == 1 and len(texts[0]) <= 40 and bool(re.fullmatch(r"[A-Z][A-Z0-9_]{2,}|[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+", texts[0]))


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


SHOWN_CHARS = 9000          # a result as the model reads it: the agent keeps the whole record, the conversation a short form
ROW_TEXT, SHEET_TEXT = 100, 600   # free text kept per field in a list of rows / in a single record
FIT_CUE = re.compile(r"(?i)compatib|\bfits?\b|\bexcept\b|\bmodel\b|\bpart\b|\bseries\b|taken from|different product|\bsold by\b")


ROW_TAIL, SHEET_TAIL = 180, 400   # fitment sentences kept beyond the cut, per field of a row / of a single record
ROWS_FULL = 14              # rows of one list shown with their free text; later rows show their short fields and facts only


def _short_text(text: str, limit: int, tail_cap: int = SHEET_TAIL) -> str:
    """A long free text cut for the model: its head, plus the sentences that carry fitment or a pointer to another
    listing (a model or part code, "compatible with", "except", "taken from a different product")."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    extra = [p.strip() for p in re.split(r"(?<=[.;!?])\s+", text[limit:]) if FIT_CUE.search(p) or len(CODE_TOKEN.findall(p)) >= 1 and re.search(r"[A-Z]{2,}-?\d{3}", p)]
    tail = " ".join(e[:200] for e in extra)[:tail_cap]
    return head + "..." + (" " + tail if tail else "")


def _raw_text(value: Any) -> str:
    """A result as the store wrote it, for the model to read: compact JSON (a string that holds JSON is opened one level,
    so it is not escaped twice), each string of a row list cut at RAW_CUT characters and any other string at four times
    that, so a whole record survives; the whole capped at RAW_CHARS. A bare string result is shown as it is."""
    if isinstance(value, str):
        inner = _json_inside(value)
        if inner is None:
            return value if len(value) <= RAW_CHARS else value[:RAW_CHARS] + f"...(cut: {len(value) - RAW_CHARS} more characters)"
        value = inner

    def walk(v: Any, cut: int, depth: int) -> Any:
        if depth > 10:
            return "..."
        if isinstance(v, str):
            inner = _json_inside(v) if depth < 6 else None
            if inner is not None:
                return walk(inner, cut, depth + 1)
            return v if len(v) <= cut else v[:cut] + f"...(+{len(v) - cut} chars)"
        if isinstance(v, dict):
            return {str(k): walk(x, cut, depth + 1) for k, x in v.items()}
        if isinstance(v, list):
            inner_cut = RAW_CUT if len(v) >= 3 else cut
            return [walk(x, inner_cut, depth + 1) for x in v]
        return v
    try:
        text = json.dumps(walk(value, 4 * RAW_CUT, 0), ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= RAW_CHARS else text[:RAW_CHARS] + f"...(cut: {len(text) - RAW_CHARS} more characters)"


def _json_inside(text: str) -> Any:
    """The container a string holds as JSON text ('{"id": ...}' or '[...]'), else None."""
    t = text.strip()
    if len(t) < 2 or t[0] not in "[{" or t[-1] not in "]}":
        return None
    try:
        inner = json.loads(t)
    except ValueError:
        return None
    return inner if isinstance(inner, (dict, list)) else None


def _apparent_records(value: Any, depth: int = 0) -> int:
    """How many listing records a raw result seems to hold, read without any parser: the longest list of mappings or of
    long strings anywhere in it (a string that holds JSON or several lines is looked into), or one for a lone record."""
    if depth > 8:
        return 0
    if isinstance(value, str):
        inner = _json_inside(value)
        if inner is not None:
            return _apparent_records(inner, depth + 1)
        blocks = [b for b in re.split(r"\n\s*\n", value) if len(b.strip()) >= 40]
        return len(blocks) if len(blocks) >= 2 else int(len(value) >= 80 and value.count(":") + value.count("=") >= 3)
    if isinstance(value, list):
        own = sum(1 for x in value if isinstance(x, dict) and len(x) >= 3 or isinstance(x, str) and len(x) >= 40)
        return max([own] + [_apparent_records(x, depth + 1) for x in value[:40]])
    if isinstance(value, dict):
        inner = max([_apparent_records(x, depth + 1) for x in value.values()] or [0])
        return inner or int(len(value) >= 4)
    return 0


def _parse_json_object(text: str) -> dict[str, Any] | None:
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
            return None
    return None


def shape_args(schema: dict[str, Any], args: Any, path: str = "args") -> tuple[Any, list[str]]:
    """Arguments bent to a JSON schema where that is harmless (numbers from strings, enum case, lengths, bounds, unknown
    keys dropped when forbidden), with the problems that remain and would waste a step."""
    if not isinstance(schema, dict):
        return args, []
    problems: list[str] = []
    kinds = schema.get("type")
    kinds = [kinds] if isinstance(kinds, str) else (kinds or [])
    kind = next((k for k in kinds if k != "null"), None)
    value = args
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        value = int(value)
    elif kind == "integer" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(?:\.0+)?\s*", value):
        value = int(float(value))
    elif kind == "number" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", value):
        value = float(value)
    elif kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    elif kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
        value = value.strip().lower() == "true"
    checks = {"string": lambda v: isinstance(v, str), "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool), "boolean": lambda v: isinstance(v, bool),
              "object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list), "null": lambda v: v is None}
    if kinds and not any(checks.get(k, lambda v: True)(value) for k in kinds):
        return value, [f"{path} should be {'/'.join(kinds)}"]
    options = schema.get("enum")
    if isinstance(options, list) and options and value not in options:
        if isinstance(value, str):
            value = next((o for o in options if isinstance(o, str) and o.lower() == value.strip().lower()), value)
        if value not in options:
            problems.append(f"{path} must be one of {options}")
    if isinstance(value, str):
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]].rstrip()
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            problems.append(f"{path} is too short")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            value = schema["minimum"]
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            value = schema["maximum"]
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        fixed = []
        for index, item in enumerate(value):
            item, more = shape_args(schema["items"], item, f"{path}[{index}]")
            fixed.append(item)
            problems += more
        value = fixed
    if isinstance(value, dict):
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for name in schema.get("required") or []:
            if name not in value or value.get(name) in (None, ""):
                problems.append(f"{path}.{name} is required")
        fixed = {}
        for name, item in value.items():
            if name not in properties:
                if schema.get("additionalProperties") is False:
                    continue
                fixed[name] = item
                continue
            item, more = shape_args(properties[name], item, f"{path}.{name}")
            problems += more
            fixed[name] = item
        value = fixed
    return value, problems


def _sentences(value: Any, out: list[str], depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(value, str):
        if len(value.split()) >= 4:
            out.append(value)
    elif isinstance(value, list):
        for item in value[:10]:
            _sentences(item, out, depth + 1)


ENVELOPE_KEYS = r"^(?:data|result|payload|observation|response|body)$"


def _line_list(value: Any) -> list[str] | None:
    """Shopper or store lines riding in a result: a list of sentences, or a list of small event objects that each carry
    one sentence (no listing fields beside it)."""
    if not isinstance(value, list) or not value:
        return None
    if all(isinstance(v, str) for v in value):
        lines = [v for v in value if len(v.split()) >= 4]
        return lines or None
    if all(isinstance(v, dict) and 0 < len(v) <= 3 for v in value):
        lines = []
        for v in value:
            texts = [x for x in v.values() if isinstance(x, str) and len(x.split()) >= 4]
            if len(texts) != 1 or any(isinstance(x, (int, float, list, dict)) and not isinstance(x, bool) for x in v.values()):
                return None
            lines.append(texts[0])
        return lines
    return None


def settle_observation(obs: Any) -> Any:
    """One plain form for a result, whatever the store's packing: single-key envelopes around a dict are opened, and lines
    that ride deeper than the top level (nested under a payload, or as event objects) are lifted into one top-level list
    beside the payload ("refusal" when the store called them that). A result already in the plain form is left as is."""
    if isinstance(obs, dict) and len(obs) == 2 and any(_line_list(v) and all(isinstance(x, str) for x in v) for v in obs.values()):
        return obs
    lifted: list[str] = []
    named = {"refusal": False}

    def walk(value: Any, depth: int) -> Any:
        if not isinstance(value, dict) or depth > 3:
            return value
        out = {}
        for key, item in value.items():
            lines = _line_list(item)
            if lines:
                lifted.extend(lines)
                named["refusal"] = named["refusal"] or str(key).lower() == "refusal"
                continue
            out[key] = walk(item, depth + 1) if isinstance(item, dict) else item
        return out

    body = walk(obs, 0)
    while isinstance(body, dict) and len(body) == 1 and re.match(ENVELOPE_KEYS, str(next(iter(body))).lower()) and isinstance(next(iter(body.values())), (dict, type(None), str)):
        body = next(iter(body.values()))
    if not lifted:
        return body
    if body in (None, {}, ""):
        return lifted
    return {"payload": body, ("refusal" if named["refusal"] else "notes"): lifted}


# ----------------------------------------------------------------------------------------------- record encodings
# A store may pack a listing as a JSON object, as a JSON text, or as one flat line of labelled fields
# ("label: value | label: value | facts: k=v; k=v"), as "Label: value" lines, or wrapped in a one-key object. Every result
# is unpacked into plain objects before any code reads it, by structure alone (a label is a short word group followed
# by ": "), never by particular label names. A text that shows no such structure stays as it is.
_LABEL = re.compile(r"[A-Za-z_][A-Za-z0-9_ .\-/()]{0,38}")
_PAIR_LABEL = re.compile(r"[^\s=;|:{}\[\]][^=;|:{}\[\]\n]{0,119}")
_FLAT_SEPARATORS = (" | ", " || ", " ; ", "\t")
LEAD_LABEL = "text"         # the field name given to an unlabelled lead segment of a packed record


def _label_head(part: str, mark: str = ":") -> tuple[str, str] | None:
    """A segment that opens a field ('label: value'): the label and the rest, else None."""
    pattern = _LABEL if mark == ":" else _PAIR_LABEL
    head = pattern.match(part.lstrip())
    if not head:
        return None
    label = head.group(0).rstrip()
    rest = part.lstrip()[head.end():]
    lead = len(rest) - len(rest.lstrip(" "))
    rest = rest.lstrip(" ")
    if not rest.startswith(mark) or not label or len(label.split()) > (4 if mark == ":" else 12) or re.search(r"[.!?]\s", label):
        return None
    after = rest[len(mark):]
    if mark == ":" and after and not after[0].isspace():
        return None                               # "12:30", "http://", "a:b" are values, not fields
    if mark == ":" and lead:
        return None
    return label, after.strip()


def _scalar(text: str) -> Any:
    """A field's text as the value a JSON record would hold: flags, the empty marker, plain numbers, nested records."""
    value = text.strip()
    if value in ("None", "null", "NULL", "Null"):
        return None
    if value in ("True", "true", "TRUE"):
        return True
    if value in ("False", "false", "FALSE"):
        return False
    if re.fullmatch(r"-?(?:0|[1-9]\d{0,14})(?:\.\d+)?", value):
        return float(value) if "." in value else int(value)
    return unpack_records(value, 1, "field") if len(value) >= 2 else value


GROUP_KEYS: set[str] = set()   # fields that held a nested label group in some record of this episode (reset per episode)


def _lone_pair(text: str) -> dict[str, Any] | None:
    """A field value that is one 'label: value' pair, short and without a sentence: a one-entry group."""
    head = _label_head(text)
    if not head or not head[1] or len(head[1]) > 120 or re.search(r"[.!?]\s+[A-Z]|[.!?]$", head[1]) or ": " in head[1]:
        return None
    return {head[0]: _scalar(head[1])}


def _pair_group(text: str) -> dict[str, Any] | None:
    """A nested 'k=v; k=v' (or 'k: v; k: v', 'k: v, k: v') group as a map, else None. Segments that open no pair
    continue the value before them. A 'k: v' group needs two pairs or more and short values without sentences (a
    description that starts with a label stays a text)."""
    if "=" not in text and ": " not in text:
        return None
    for mark in ("=", ":"):
        for glue in ("; ", ";", ", "):
            parts = text.split(glue)
            if len(parts) == 1 and glue != "; ":
                continue
            pairs: list[list[str]] = []
            for part in parts:
                head = _label_head(part, mark)
                if head and not (pairs and head[0] in {p[0] for p in pairs}):
                    pairs.append([head[0], head[1]])
                elif pairs:
                    pairs[-1][1] += glue + part
                else:
                    pairs = []
                    break
            if not pairs or (len(pairs) == 1 and (len(parts) > 1 or mark == ":" or len(pairs[0][0].split()) > 4)):
                continue                          # a lone 'x: y' inside a field is text; a lone 'k=v' is a one-entry group
            if mark == ":" and any(len(v) > 160 or re.search(r"[.!?]\s+[A-Z]", v) for _, v in pairs):
                continue
            return {k: _scalar(v) for k, v in pairs}
    return None


def _flat_fields(text: str) -> list[list[str]] | None:
    """A one-line record split into its fields: a separator splits only where the next segment opens a field, so a
    title holding the separator stays whole. At least two fields."""
    for glue in _FLAT_SEPARATORS:
        if glue not in text:
            continue
        fields: list[list[str]] = []
        for part in text.split(glue):
            head = _label_head(part)
            if head and not (fields and head[0] in {f[0] for f in fields}):
                fields.append([head[0], head[1]])
            elif fields:
                fields[-1][1] += glue + part
            else:
                fields.append([LEAD_LABEL, part.strip()])   # an unlabelled lead segment (a heading) before the fields
        if sum(1 for f in fields if f[0] != LEAD_LABEL) >= 2:
            return fields
    return None


def _line_fields(text: str) -> list[Any] | None:
    """'Label: value' lines (one field a line; indented or bulleted lines under an empty label form a nested group;
    other lines continue the value above). At least two fields, and most lines open one."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    fields: list[Any] = []
    opened = 0
    for line in lines:
        indented = line[:1].isspace() or line.lstrip().startswith(("- ", "* ", "\u2022 "))
        bare = re.sub(r"^\s*(?:[-*\u2022]\s+)?", "", line)
        head = _label_head(bare)
        if indented and not head and fields and (isinstance(fields[-1][1], dict) or fields[-1][1] == "") and ": " in bare:
            label, _, rest = bare.partition(": ")
            head = (label.strip(), rest.strip()) if 0 < len(label) <= 120 else None   # a nested label may be long
        if indented and head and fields and isinstance(fields[-1][1], dict):
            fields[-1][1][head[0]] = _scalar(head[1])
            opened += 1
        elif indented and head and fields and fields[-1][1] == "":
            fields[-1][1] = {head[0]: _scalar(head[1])}
            opened += 1
        elif head and not indented and not any(f[0] == head[0] for f in fields):
            fields.append([head[0], head[1]])
            opened += 1
        elif fields and isinstance(fields[-1][1], str):
            fields[-1][1] = (fields[-1][1] + " " + line.strip()).strip()
        elif not fields and not indented:
            fields.append([LEAD_LABEL, line.strip()])   # an unlabelled heading line before the fields
        else:
            return None
    return fields if len(fields) - (fields[0][0] == LEAD_LABEL) >= 2 and opened * 2 > len(lines) else None


def record_from_text(text: str, where: str = "whole") -> dict[str, Any] | None:
    """One text as a record, when it is one: a JSON object text, a flat labelled line, or labelled lines. Inside a field
    of another record only a JSON text or a flat line of three fields or more counts (a description stays a text)."""
    body = text.strip()
    if len(body) < 5:
        return None
    if body[:1] == "{" and body[-1:] == "}":
        try:
            value = json.loads(body)
        except ValueError:
            value = None
        if isinstance(value, dict):
            return value
    fields = _flat_fields(body) if "\n" not in body else None
    if fields is None and "\n" in body and where != "field":
        fields = _line_fields(body) or _flat_fields(" ".join(body.split()))
    if fields is None or (where == "field" and len(fields) < 3):
        return None
    record: dict[str, Any] = {}
    for label, raw in fields:
        if isinstance(raw, dict):
            record[label] = raw
            GROUP_KEYS.add(label)
            continue
        group = _pair_group(raw) if ("=" in raw or ": " in raw) and len(raw) <= 4000 else None
        if group is None and label in GROUP_KEYS:
            group = _lone_pair(raw)               # the same field held a group elsewhere: a single pair is a group of one
        if group is not None:
            GROUP_KEYS.add(label)
        record[label] = group if group is not None else _scalar(raw)
    return record


def unpack_records(value: Any, depth: int = 0, where: str = "whole") -> Any:
    """A result with every packed record opened into an object (see the note above): JSON texts are read, labelled texts
    become objects, lists of such texts become lists of objects, and a one-key wrapper around a packed record gives way
    to the record. 'where' is the text's place: the whole result or a list member ('whole', 'item') may be any packing;
    one field among others ('field') only a JSON text or a flat line of three fields. Anything else is unchanged."""
    if depth > 6:
        return value
    if isinstance(value, str):
        body = value.strip()
        if len(body) >= 2 and ((body[0] == "{" and body[-1] == "}") or (body[0] == "[" and body[-1] == "]") or (body[0] == body[-1] == '"')):
            try:
                decoded = json.loads(body)
            except ValueError:
                decoded = None
            if isinstance(decoded, (dict, list)) or (isinstance(decoded, str) and decoded != value):
                return unpack_records(decoded, depth + 1, where)
        if not (": " in value or ":\n" in value):
            return value
        record = record_from_text(value, where)
        return unpack_records(record, depth + 1, where) if record is not None else value
    if isinstance(value, list):
        items = [unpack_records(v, depth + 1, "item") for v in value]
        for item in items:                        # a field that holds a group in one row holds a group of one in another
            if isinstance(item, dict):
                for key, inner in list(item.items()):
                    if isinstance(inner, str) and key in GROUP_KEYS and ": " in inner:
                        pair = _lone_pair(inner)
                        if pair is not None:
                            item[key] = pair
        return items
    if isinstance(value, dict):
        lone = len(value) == 1
        out = {k: unpack_records(v, depth + 1, "whole" if lone else "field") for k, v in value.items()}
        if lone:
            (key, inner), = out.items()
            if isinstance(value[key], str) and isinstance(inner, dict) and not re.match(ENVELOPE_KEYS, str(key).lower()):
                return inner                      # {"value": "<packed record>"}: the record itself
        return out
    return value


def _key_stem(key: Any) -> str:
    """A field name reduced to its letters and digits, so 'item_pid', 'Item PID' and 'item-pid' are one name."""
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _names_id(key: Any, id_keys: set[str] | None = None) -> bool:
    """Whether a field name denotes the listing identifier: a name the tools or the rows taught us, else one carrying 'id'."""
    stem = _key_stem(key)
    return bool(id_keys and stem in id_keys) or "id" in str(key).lower()


def _collect_ids(value: Any, out: set[str], depth: int = 0, id_keys: set[str] | None = None) -> None:
    if depth > 6:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and _names_id(key, id_keys) and _ID_HINT.match(item):
                out.add(item)
            else:
                _collect_ids(item, out, depth + 1, id_keys)
    elif isinstance(value, list):
        for item in value:
            _collect_ids(item, out, depth + 1, id_keys)


def lowered(text: Any) -> str:
    return str(text if text is not None else "").lower()


def stem(word: str) -> str:
    """One number for a word: 'stripes' -> 'stripe', 'boxes' -> 'box', 'ladies' -> 'lady'."""
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def word_list(text: Any, glue_units: bool = False) -> list[str]:
    text = re.sub(r"'s\b", "", lowered(text))
    if glue_units:
        text = re.sub(r"(\d)\s+(?=[a-z%])", r"\1", text)
    return re.findall(r"[a-z0-9]+", text)


SMALL_WORDS = ("a", "an", "the", "of", "and", "or", "with", "be", "is", "are", "in", "for", "to")
TITLE_LIKE = r"title|heading|name|option|variant|description|about"
GROUP_LIKE = r"categor|aisle|shelf|department|section|collection"
NOT_A_PROPERTY = r"avail|stock|on hand|categor|item type|kind of|\buse\b|purpose|usage|budget|price|cost|deliver|shipping|arriv|quantity|message|step|\bbrand\b"


def kin(one: str, other: str) -> bool:
    """Two words that are forms of each other (tunic/tunik, stripe/striped), never two different numbers."""
    one, other = stem(one), stem(other)
    if one == other:
        return True
    if re.search(r"\d", one + other):
        return False
    short, long = sorted((one, other), key=len)
    if long.startswith(short):
        return len(short) >= 4 and len(long) - len(short) <= 3
    shared = 0
    for a, b in zip(one, other):
        if a != b:
            break
        shared += 1
    return shared >= max(5, len(short) - 2)


def states_value(value: str, text: str) -> bool:
    """Every content word of the value has a kin among the text's words, or the value written solid sits inside the text."""
    for glue in (True, False):
        wanted = [w for w in word_list(value, glue) if w not in SMALL_WORDS]
        have = word_list(text, glue)
        if wanted and all(any(kin(w, h) for h in have) for w in wanted):
            return True
    solid = re.sub(r"[^a-z0-9]+", "", value.lower())
    return len(solid) >= 8 and solid in re.sub(r"[^a-z0-9]+", "", lowered(text))


def label_fits(label: str, prop: str, product_words: set[str]) -> bool:
    """A sheet label stands for the property when it holds the property's words (the product's own words may be dropped)
    and at most two more; 'type' in the property must appear in the label unless the label is exactly the core."""
    lw = {stem(w) for w in word_list(label)}
    pw = {stem(w) for w in word_list(prop)}
    core = pw - {"type"}
    if core and not core <= lw:
        core = core - product_words
        pw = pw - product_words
    if not core or not core <= lw:
        return False
    if "type" in pw and "type" not in lw and lw != core:
        return False
    return len(lw - pw) <= 2


def bound_of(value: str) -> tuple[float | None, float | None] | None:
    """The numeric bound a value states: 'at least 2 l' -> (2, None); 'under 500' -> (None, 500); '2-5 kg' -> (2, 5)."""
    text = value.lower().replace(",", "")
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]
    if not nums:
        return None
    more = r"more|bigger|larger|higher|longer|wider|taller|heavier|above|over"
    less = r"less|smaller|fewer|lower|shorter|narrower|lighter|under|below"
    if len(nums) >= 2 and re.search(r"between|\d\s*(?:to|-|and)\s*\d", text) and not re.search(r"up to|at (?:most|least)", text):
        return (min(nums[:2]), max(nums[:2]))
    if re.search(r"\bno(?:t)?\s+(?:%s)\b" % more, text):
        return (None, nums[0])
    if re.search(r"\bno(?:t)?\s+(?:%s)\b" % less, text):
        return (nums[0], None)
    if re.search(r"at least|or (?:%s)|minimum|\bmin\b|(?:%s) than|\babove\b|\bover\b|>=|\d\+" % (more, more), text):
        return (nums[0], None)
    if re.search(r"at most|up to|or (?:%s)|maximum|\bmax\b|(?:%s) than|\bunder\b|\bbelow\b|within|<=" % (less, less), text):
        return (None, nums[0])
    return None


def figures_in(text: str) -> list[tuple[float, str]]:
    return [(float(m.group(1)), stem(m.group(2))) for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*([a-z%\"]*)", lowered(text).replace(",", ""))]


def titled(key: Any) -> bool:
    """A key holding a listing's name: one that says so, or the world's learned title field."""
    return bool(re.search(TITLE_LIKE, str(key), re.I)) or (bool(FIELDS.get("title")) and str(key) == FIELDS.get("title"))


def name_key(key: Any) -> bool:
    """A key holding only a listing's name (not its free description, which carries fitment and remarks)."""
    return bool(re.search(r"(?i)title|name|heading", str(key))) or (titled(key) and not re.search(r"(?i)descr|about", str(key)))


def grouped(key: Any) -> bool:
    """A key holding a listing's category: one that says so, or the world's learned category field."""
    return bool(re.search(GROUP_LIKE, str(key), re.I)) or (bool(FIELDS.get("group")) and str(key) == FIELDS.get("group"))


def field_id(label: str) -> str:
    """Two labels spell the same field when they differ only in case, spaces, hyphens or underscores ('clock shape' and
    'clock_shape'); a dotted namespace ('clothing.material' beside 'material') makes a separate claim."""
    return re.sub(r"[\s_\-]+", " ", str(label).strip().lower())


def field_groups(pairs: list[tuple[str, str]]) -> list[list[str]]:
    """The texts of the given label -> value pairs, grouped by field."""
    groups: dict[str, list[str]] = {}
    for label, text in pairs:
        groups.setdefault(field_id(label), []).append(text)
    return list(groups.values())


def labels_agree(sheet: dict[str, Any], prop: str, value: str, product_words: set[str]) -> bool:
    """True when the sheet has a label for the property and every field standing for it states the value (or one of its
    spellings); copies of one field count as a whole, so any copy stating it is enough."""
    named = [(str(k), str(v)) for k, v in sheet.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)
             and not titled(k) and not grouped(k) and label_fits(str(k), prop, product_words)]
    options = [a for a in re.split(r"\s+or\s+|(?<=[a-z])\s*/\s*(?=[a-z])|,\s*(?!\d)", value) if a.strip()] or [value]
    return bool(named) and all(any(states_value(a, text) for a in options for text in copies) for copies in field_groups(named))


def clashing_claims(named: list[tuple[str, str]], options: list[str]) -> bool:
    """Separate claims about one property (a namespaced label and the plain one, 'clothing.material' and 'material')
    where one states the value and another does not. Copies of one field are read together, and labels for different
    properties ('heel type' beside 'heel height') never clash."""
    cores: dict[str, dict[str, bool]] = {}
    for label, text in named:
        fid = field_id(label)
        core = fid.rsplit(".", 1)[-1].strip()
        held = cores.setdefault(core, {})
        held[fid] = held.get(fid, False) or any(states_value(a, text) for a in options)
    return any(len(fields) > 1 and any(fields.values()) and not all(fields.values()) for fields in cores.values())


def read_sheet(sheet: dict[str, Any], prop: str, value: str, product_words: set[str]) -> tuple[str, str]:
    """How a listing's own sheet answers one requirement: 'yes', 'no' or 'unshown', with the evidence. A label that stands
    for the property decides; the title counts only when no label does; an excluded value present loses; a bound is met
    by a figure inside it and lost by a comparable figure outside it. Advice for the model, not a grade."""
    pairs = [(str(k), str(v)) for k, v in sheet.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)
             and not re.fullmatch(r"(?i)\s*(?:n/?a|none|not specified|unknown|null|-|)\s*", str(v))]
    named = [(k, v) for k, v in pairs if not titled(k) and not grouped(k) and label_fits(k, prop, product_words)]
    texts = [v for _, v in named] or [v for k, v in pairs if titled(k)]
    shown = "; ".join(f"{k}: {v[:60]}" for k, v in named[:2]) or "no label for it"
    bound = bound_of(value)
    excluded = re.match(r"^(?:not|no|without|anything but|anything except|anything other than|other than|unlike|different from|except|excluding|avoid)\s+(.+)$", value)
    if excluded and bound is None:
        # with no label for the property, an excluded value any other field of the sheet states still loses: the
        # store may file the property under a neighbouring name (a collar under 'neckline')
        seen = texts if named else [v for _, v in pairs]
        hit = next((k for k, v in pairs if v in seen and states_value(excluded.group(1), v)), "")
        if hit and not named:
            shown = f"{hit}: {sheet.get(hit, '')!s:.60}"
        return ("no" if hit else "yes"), shown
    if bound is not None:
        low, high = bound
        unit = next((u for _, u in figures_in(value) if u), "")
        figures = [f for t in texts for f in figures_in(t)]
        comparable = [n for n, u in figures if not unit or not u or kin(u, unit)]
        if any((low is None or n >= low) and (high is None or n <= high) for n in comparable):
            return "yes", shown
        return ("no" if comparable and named else "unshown"), shown
    options = [a for a in re.split(r"\s+or\s+|(?<=[a-z])\s*/\s*(?=[a-z])|,\s*(?!\d)", value) if a.strip()] or [value]
    if named and clashing_claims(named, options):
        return "no", shown
    if any(states_value(a, t) for a in options for t in texts):
        return "yes", shown
    if not named:
        return "unshown", shown
    units = {u for _, u in figures_in(value) if u}
    if units and any(u and u not in units for t in texts for _, u in figures_in(t)):
        return "unshown", shown
    return "no", shown


_SHELF_KEY = re.compile(r"(?i)categor|aisle|shelf|department|section")


def words_slot_of(episode: Any) -> str | None:
    return (episode.search_shape() or {}).get("words")


def label_words(text: Any) -> list[str]:
    return [stem(w) for w in word_list(text)]


FIELDS: dict[str, Any] = {}     # the price and stock keys of the world in play, learned once from its rows (reset per episode)


def money_value(value: Any) -> float | None:
    """A price as a number: a plain number, or a short text holding one figure with a currency sign or word around it."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and len(value) <= 32 and re.match(r"\s*(?:[A-Za-z]{1,3}\s+|[^\w\s]{1,3}\s*)?\d", value):
        figures = re.findall(r"\d[\d,]*(?:\.\d+)?", value)
        if len(figures) == 1:
            try:
                return float(figures[0].replace(",", ""))
            except ValueError:
                return None
    if isinstance(value, dict) and 2 <= len(value) <= 3:
        # a price written as a small object: one figure beside a short unit or currency word ({"amount": 4.5, "currency": "USD"})
        figures = [v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        words = [v for v in value.values() if isinstance(v, str)]
        if len(figures) == 1 and len(words) == len(value) - 1 and all(0 < len(w) <= 12 for w in words):
            return float(figures[0])
    return None


def major_amount(value: Any) -> bool:
    """A price object whose figure has cents after the point: written in the major unit (dollars, not cents)."""
    if not isinstance(value, dict) or money_value(value) is None:
        return False
    figure = money_value(value) or 0.0
    return not float(figure).is_integer() and abs(round(figure, 2) - figure) < 1e-9


def stock_value(value: Any) -> bool | None:
    """Whether a stock field's value says the listing can be bought: a flag, a count, or a short word."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str) and 0 < len(value) <= 30:
        return not re.search(r"(?i)\b(?:not|no|out|sold|none|zero)\b|\bun\w+|^0$", value)
    return None


def _price(record: dict[str, Any]) -> float | None:
    key = FIELDS.get("price")
    if key and key in record:
        value = money_value(record[key])
        return value * FIELDS.get("scale", 1.0) if value is not None and isinstance(record[key], (str, dict)) else value
    for key, value in record.items():
        if re.search(r"(?i)price|cost", str(key)) and not re.search(r"(?i)total", str(key)) and money_value(value) is not None:
            return money_value(value)
    return None


def _in_stock(record: dict[str, Any]) -> bool | None:
    key = FIELDS.get("stock")
    if key and key in record:
        return stock_value(record[key])
    for key, value in record.items():
        if re.search(r"(?i)stock|avail", str(key)) and isinstance(value, (bool, int, float)):
            return bool(value) if isinstance(value, bool) else value > 0
    return None


def as_pairs(value: Any) -> dict[str, Any] | None:
    """A list of small records that each carry one label and one value, read as a label -> value map: the label is the
    key whose texts all differ, the value the other key."""
    if not (isinstance(value, list) and len(value) >= 2 and all(isinstance(x, dict) and 2 <= len(x) <= 4 for x in value[:80])):
        return None
    shared = set.intersection(*(set(x) for x in value[:80]))
    texts = [k for k in shared if all(isinstance(x[k], str) and 0 < len(x[k]) <= 60 for x in value[:80])]
    labels = [k for k in texts if len({x[k].lower() for x in value[:80]}) == len(value[:80])]
    if not labels:
        return None
    label = next((k for k in labels if re.search(r"(?i)label|name|key|attr|prop|field", str(k))), labels[0])
    worth = [k for k in shared if k != label and all(isinstance(x[k], (str, int, float)) and not isinstance(x[k], bool) for x in value[:80])]
    if not worth:
        return None
    worth.sort(key=lambda k: (not re.search(r"(?i)val|content|setting|text|data", str(k)), list(value[0]).index(k)))
    return {x[label]: x[worth[0]] for x in value[:80]}


def normal(value: Any, depth: int = 0) -> Any:
    """A result with every label/value pair list turned into a label -> value map (a sheet in any of its shapes)."""
    if depth > 6:
        return value
    if isinstance(value, dict):
        return {k: normal(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        pairs = as_pairs(value)
        return pairs if pairs is not None and len(pairs) >= 2 else [normal(v, depth + 1) for v in value]
    return value


def _shelf(record: dict[str, Any]) -> str:
    group = FIELDS.get("group")
    if group and isinstance(record.get(group), str) and record[group].strip():
        return record[group]
    return next((v for k, v in record.items() if _SHELF_KEY.search(str(k)) and isinstance(v, str) and v.strip()), "")


class Episode:
    def __init__(self, problem_data: dict[str, Any]):
        self.started = time.monotonic()
        FIELDS.clear()
        GROUP_KEYS.clear()
        self.last_target = ""                     # the listing id the last reading call asked for
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.tools = [t for t in self.policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}} for t in self.tools}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.step, self.finished, self.dead, self.cost = 0, bool(self.policy.get("finished")), False, 0.0
        self.llm_dead = False                     # the model is given up (402, a refused key, or LLM_FAILS failures in a row)
        self.llm_fails = 0                        # inference failures in a row
        self.waited = 0.0                         # seconds spent waiting before retries (inference and store)
        self.banned_models: set[str] = set()      # models the service refused for this episode (403, 404)
        self.tool_turns = 0                       # model tool turns sent
        self.packed_results = 0                   # results whose records came packed as text (opened by unpack_records)
        self.judge_cache: dict[str, Any] = {}     # one reading per distinct prompt
        self.cached_size, self.cached_tools = 0, ""  # the conversation prefix the last tool turn left in the prompt cache
        self.roles: dict[str, str] = {}
        self.mode_cache: dict[str, dict[str, tuple[str, str]]] = {}
        self.virtual: dict[str, tuple[str, dict[str, Any]]] = {}   # tools the agent shows the model that the store serves as another tool's mode
        self.plan: dict[str, Any] = {}
        self.finish_args: dict[str, dict[str, Any]] = {}
        self.fact_tools: set[str] = set()
        self.trace: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self.start_requirements()
        self.start_store()
        self.start_cart()
        self.start_talk()
        self.start_duties()

    def start_duties(self) -> None:
        """What the request and the shopper told the agent to do (not what to buy): the duties with when they fall due and
        whether they take a message, the messages sent, the reviews after a change, and the products bought together."""
        self.duties: list[dict[str, Any]] = []
        self.sent_texts: list[str] = []
        self.reviews_left = CHANGE_REVIEWS + (1 if self.long_world() else 0)   # a long world sees a second change worth a look
        self.change_reason = ""
        self.handed_at = -1                       # the change after which the cart was only handed back (no checked fit yet)
        self.second_recoveries = 0                # recoveries repeated for one change in a long world, once a checked fit appeared
        self.final_marks: set[str] = set()        # the finishing calls (cart state and requirements) the final look already covered
        self.early_held = False                   # the one step-free hold on a finishing call sent before the fit mark
        self.pointers: list[dict[str, Any]] = []  # needs whose value sits on another listing's page, with what was found
        self.pointer_searches = 0                 # zero-result searches spent on pointer pages
        self.shed_words: set[str] = set()         # words that describe a pointed-at listing (its shop, its attributes): never search words for the item
        self.repaired_by_code = False             # the last change was handled by the agent's own cart repair (no review needed)
        self.swap_asked: set[str] = set()
        self.companions: list[str] = []
        self.together_checked = False
        self.bundle_names: list[str] = []         # the products a request for several products names (empty for one product)
        self.bundle_unsure = False                # a refused order said a line is gone without naming it: the held lines are re-read
        self.bundle_reads = 0                     # sheets and searches the agent spent itself on the products of a bundle
        self.bundle_searched: set[str] = set()
        self.bundle_removed: set[str] = set()     # lines the agent took out of a bundle cart (never twice)

    def start_requirements(self) -> None:
        """What the shopper wants: the stated, revealed, freed and dropped requirements, the limit in force, and the
        version that every cached verdict is keyed by."""
        self.stated: list[str] = []
        self.revealed: list[str] = []
        self.free: list[str] = []
        self.dropped: list[str] = []
        self.current: list[str] | None = None
        self.budget_now: Any = None
        self.ledger_version = 0
        self.verdicts: dict[str, str] = {}

    def start_store(self) -> None:
        """What the store has shown: records, result rows, opened sheets, ids, search coverage and counts."""
        self.records: list[Any] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.sheets: dict[str, dict[str, Any]] = {}
        self.seen_ids: set[str] = set()
        self.id_field: str | None = None          # the row/sheet key that carries the listing identifier, once learned
        self.coverage: dict[str, dict[str, Any]] = {}
        self.searches_done = 0
        self.recent: list[str] = []
        self.empty_order_warned: set[int] = set()
        self.walk_spent = 0
        self.swept = 0                            # sheets the agent opened itself in its sweeps after searches
        self.sweep_seen = 0                       # the search count at the last sweep
        self.walk_floor: float | None = None      # below this price nothing in stock fits (shown by an empty search)
        self.walk_wide = False                    # the no-text shelf list under the ceiling is long and unsorted: its first rows prove nothing
        self.walk_searched: set[float] = set()
        self.walk_told: set[str] = set()
        self.fallback_told: set[str] = set()
        self.search_counts: dict[str, list[tuple[str, int, tuple[str, ...]]]] = {}
        self.coarse_counts: dict[str, list[tuple[str, int, tuple[str, ...], str]]] = {}   # the same without the ceiling: words changed along with the limit or page
        self.deaf_modes: set[str] = set()         # search settings under which the words change nothing
        self.page_cover: dict[str, dict[str, Any]] = {}   # each worded search: its total, the rows shown, the last page
        self.pages_turned = 0
        self.refill_spent = 0                     # steps the agent spent looking for another fit at the cheapest price after a sellout
        self.gone_held: set[str] = set()          # sold-out listings whose return to the cart was held once
        self.pages: dict[str, str] = {}           # each listing's page text as opened (description, fine print, remarks)
        self.row_texts: dict[str, str] = {}       # long free text a result row carried (a full description)
        self.page_reads = 0                       # pages the agent opened itself to read fitment text
        self.fit_told: set[str] = set()           # cart adds held once over fitment text
        self.long_holds = 0                       # declines held on a long step cap before its decline mark
        self.budget_holds = 0                     # declines held right after a budget note
        self.text_tools: dict[str, int] = {}      # reading tools by how often their result carried page text
        self.fit_cache: tuple[Any, dict[str, Any]] = (None, {})
        self.code_misses = 0                      # searches with a requested code in them that found nothing
        self.page_stock: dict[str, bool] = {}     # each listing's stock flag as its page last showed it
        self.reread_after: set[str] = set()       # held lines re-read after a refusal that named no figure
        self.viewed: set[str] = set()             # listings whose own page or sheet a reading tool opened (by the model or the agent)
        self.text_sizes: dict[str, int] = {}      # free-text characters each result row carried (its title, description, remarks)
        self.floor_rounds = 0                     # declines held because too few listings were opened
        self.blind_results = 0                    # search or reading results in which the code found no listing record
        self.last_blind = False                   # the newest result was one of those: the model reads its text itself
        self.blind_tried = False                  # the model's reading of the raw results was asked for once at the finish
        self.parse_seen = 0                       # searches and readings whose raw text showed listing records
        self.parse_low = 0                        # ... of which the agent's parse found far fewer ids than the text shows
        self.raw_results: list[Any] = []          # the store's own text of the latest results (for the model's own pick)

    def start_cart(self) -> None:
        """The cart and the finish: lines, add signature, views, refusals and the one-time holds on orders and declines."""
        self.cart_ids: list[str] = []
        self.list_pages: dict[str, dict[str, Any]] = {}   # listing calls with a next page, per call shape
        self.bundle_paged: dict[str, int] = {}
        self.fallback_tried = False   # pages a bundle product's search was turned by the agent
        self.code_probe = False                   # the main product's code is being looked up (no recursion through its fits)
        self.mode_units: dict[str, int] = {}      # units a line went in with through a cart tool with modes (include / discard)
        self.cart_qty: dict[str, int] = {}        # units per held line, as last set (whole-basket tools set them all at once)
        self.cart_state: dict[str, str] = {}
        self.cart_used, self.cart_changed_at, self.cart_view_at, self.last_cart_view = False, 0, -1, None
        self.order_attempts, self.refusals = 0, 0
        self.order_checked = self.decline_checked = self.audited = self.search_guarded = False
        self.audit_version = -1                    # the requirement version the last decline audit read
        self.gone: set[str] = set()
        self.shelf_held: set[str] = set()
        self.cart_checked: set[str] = set()
        self.qty_warned: set[str] = set()
        self.rule_told: set[str] = set()
        self.fits_shown = ""
        self.explore_holds = 0                    # step-free holds on reading past the commit target (one in EXPLORE_VALVE goes through)
        self.swap_done = False                    # the one agent-side cart swap to the cheapest fit at order time
        self.commit_done = False                  # the one agent-side cart add of the paced fit at the commit mark
        self.market_changed_at = 0                # the step of the last price move in the store: what was read before it is stale
        self.changes = 0                          # changes seen (a gone line, a refusal, a price or budget move, a new wish)
        self.recovered_at = -1                    # the change the agent already repaired the cart after
        self.seen_at: dict[str, int] = {}         # the last step each listing was shown in any result
        self.fresh_viewed: set[str] = set()       # picks the agent re-read itself before the order
        self.agent_placed: set[str] = set()       # cart lines the agent put in itself (their facts were checked by code, so no checklist turn)
        self.discard_held: set[str] = set()       # fits whose removal from the cart was held once
        self.model_discarded: set[str] = set()    # listings the model itself took out of the cart (never carted for it again by the agent)
        self.order_holds: dict[str, int] = {}     # soft holds on the order call, per cart state (at most one each)
        self.auto_orders = 0                      # orders the agent sent itself right after repairing the cart
        self.refusal_at = -1                      # the step of the last refused order (a decline is blocked for ABSTAIN_BLOCK steps after it while a sheet fits)
        self.budget_moved_at = -1                 # the step at which the shopper last moved the budget
        self.decline_blocks = 0                   # declines the gate has intercepted
        self.gate_ordered = False                 # the gate already carted and ordered a fit itself
        self.budget_search_done = False           # the one category-under-the-budget search the agent runs before a decline may stand
        self.held_first = ""                      # the closest candidate the agent held before any decline (its notes come at the first add)
        self.hold_unfit = ""                      # a held line that still fails after the notes: never ordered by code
        self.relaxed_at = -1                      # step at which the shopper last dropped a requirement
        self.first_add_at = -1                    # step of the first line put in the cart (the store's changes come on the call after it)
        self.add_looked = False                   # the one cart look the agent takes between the first add and an order
        self.reranked_at = -1                     # step of the last re-rank after a note
        self.decline_refused_at = -1              # step at which a decline came back with the shopper's words instead of ending
        self.verify_swaps = 0                     # independent checks the recovery spent on an unchecked sheet-fit
        self.last_said = ""                      # the model's latest reasoning text (a veto must name a contradicted property)
        self.final_checked = False                # the one light look before the first finishing call
        self.model_declined = False               # the model itself sent a decline that passed the review

    def start_talk(self) -> None:
        """The exchange with the shopper: the message budget and count, questions asked, lines heard, and a reply owed."""
        self.messages_used = 0
        self.message_budget: int | None = None
        self.final_reply: dict[str, Any] | None = None   # the store's answer to the latest finishing call sent in code
        self.forced_ask = False                   # the one question forced before a first commit has been tried
        self.blind_retry = False                  # with no cap known: the first message drew no firm value, so one more may go
        self.planner_on = True
        self.asked: list[str] = []
        self.said: list[str] = []
        self.heard: set[str] = set()
        self.told_once: set[str] = set()
        self.pending_reply = ""
        self.flags: dict[str, bool] = {}           # what the reader made of the latest lines (unavailable, price moved, unclear, one at a time)
        self.question_flags: dict[str, dict[str, bool]] = {}
        self.heard_now = ""
        self.refused_text = ""
        self.multi_held: set[str] = set()         # cart states whose order was held once for holding several listings of one product
        self.key_width = KEY_WIDTH                # keys per question now; narrows when replies show the shopper reads one key at a time
        self.width_proven = False                 # a reply gave a firm value for a key named after the first: lists work here
        self.key_msgs: list[tuple[str, list[str], int]] = []   # (product, field ids named, index of the key answered or -1) per question
        self.key_firm: dict[str, set[str]] = {}   # product -> field ids (and need properties) the shopper gave a firm value for
        self.last_reply = ""                      # cam50: the store's text of the latest reply to a planned question
        self.reply_kind = ""                      # cam50: how the latest reply to a planned question read (firm, shrug, first, crowd, puzzled)
        self.last_keys: list[str] = []            # the keys the latest question named, in order
        self.probed = False                       # the store's allowance report was read once
        self.width_locked = False                 # cam50: a crowding complaint was heard: one key per message for the rest of the episode
        self.requeue: list[tuple[str, list[str], bool]] = []   # cam50: (product, keys, plain words) to re-send before any new question
        self.buy_asks: set[str] = set()           # cam50: the moments (cart, order) at which free messages were spent on questions
        self.simpler_sent = 0                     # questions re-sent in plainer words after a puzzled reply
        self.simpler_keys: set[str] = set()
        self.lines_unsure = False                 # a line is gone but the words did not say which: the held lines are re-read

    def time_left(self) -> float:
        """Seconds left of the episode's own clock: one fixed span for every step cap (the network stops a task at 300 s
        whatever its cap), with the finish reserve inside it."""
        return TASK_SECONDS - (time.monotonic() - self.started)

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def infer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        """One model call, tried on the given models in turn. Never raises and never stalls: a 402 (the key's money is
        gone) or a refused key gives the model up for the episode at once, and LLM_FAILS failures in a row (429, 5xx,
        network, an empty completion) do the same; after that every call returns None at once and the agent finishes in
        code. The spend is booked as the billed cost; a reply without a cost is booked at a size estimate with prompt-cache
        prices."""
        for index, model in enumerate(models):
            if self.llm_dead:
                return None
            if model in self.banned_models:
                continue
            tries = 0
            while True:
                outcome = self.infer_once(messages, tools, model, max_tokens)
                if isinstance(outcome, dict):
                    return outcome
                if outcome == "next" or self.llm_dead:
                    if self.llm_fails >= LLM_FAILS and not self.llm_dead:
                        self.llm_dead = True
                        self.trace.append({"kind": "model_given_up", "status": None, "fails": self.llm_fails})
                    break
                status = outcome
                passing = status is None or status in PASSING
                # a 429 hands over at once to a later model that costs no more than this one (a judge to the light
                # model); a dearer fallback is only tried after this model's own retries
                price = PRICE_IN_OUT.get(model, (3.0, 15.0))[0]
                handover = status == 429 and any(m != model and m not in self.banned_models and PRICE_IN_OUT.get(m, (3.0, 15.0))[0] <= price
                                                 for m in models[index + 1:])
                if passing and tries < 2 and not handover and self.pause(tries, model + str(len(messages)) + str(tries)):
                    tries += 1
                    continue                          # the same model once more after a short wait
                self.llm_fails += 1
                if self.llm_fails >= LLM_FAILS:
                    self.llm_dead = True
                    self.trace.append({"kind": "model_given_up", "status": status, "fails": self.llm_fails})
                break
        return None

    def infer_once(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, model: str, max_tokens: int) -> Any:
        """One request to one model: the message on success; "next" when this model should be passed over (out of time,
        the output cap ran out while thinking, or the model is refused); else the failure's status (None for a network
        fault or an empty reply)."""
        left = self.time_left()
        if left < FORCE_SECONDS - 10:
            self.llm_dead = self.llm_dead or left < 3
            return "next"
        _llm.timeout = int(max(8, min(75, left - FORCE_SECONDS + 15)))
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 2:
            # breakpoints on the playbook, on the last message already sent before (its prefix is byte-identical
            # from turn to turn, so the next turn reads it from the cache) and on the newest message
            sent = [dict(m) for m in messages]
            for index in sorted({0, len(sent) - 2, len(sent) - 1}):
                if isinstance(sent[index].get("content"), str) and sent[index]["content"]:
                    sent[index]["content"] = [{"type": "text", "text": sent[index]["content"], "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            result = _llm.post_verbose("/inference/chat/completions", body)
            data, error = result.data, result.error
        except Exception as exc:                # a client fault is one more failure, never a crash
            data, error = None, {"status": None, "body": f"{type(exc).__name__}: {exc}"}
        message: dict[str, Any] = {}
        if isinstance(data, dict) and isinstance(data.get("choices"), list) and data["choices"]:
            first = data["choices"][0]
            message = (first.get("message") if isinstance(first, dict) else None) or {}
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            billed = usage.get("cost")
            estimate = self.size_cost(model, body, message, messages)   # always run: it keeps the cache prefix current
            if isinstance(billed, (int, float)) and not isinstance(billed, bool) and billed >= 0:
                self.cost += float(billed)          # the billed cost is the spend; the estimate never overrides it
            else:
                self.cost += estimate               # a reply without a cost is booked at the size estimate
            if message.get("content") or message.get("tool_calls"):
                self.llm_fails = 0
                return message
            if isinstance(first, dict) and first.get("finish_reason") == "length":
                # the cap ran out while the model was still thinking: no fault of the service, so it never counts
                # towards giving the model up; the next model (or the caller) goes on
                self.trace.append({"kind": "inference_failure", "model": model, "status": None, "error": "length"})
                return "next"
            self.trace.append({"kind": "inference_failure", "model": model, "status": None, "error": "empty"})
            self.llm_fails += 1                   # an empty completion: the same request would come back empty again
            return "next"
        status = (error or {}).get("status") if isinstance(error, dict) else None
        self.trace.append({"kind": "inference_failure", "model": model, "status": status,
                           "error": str((error or {}).get("body") or "")[:300] if error else "empty"})
        if status in (401, 402):
            self.llm_dead = True                  # the key is refused or its money is gone: no model can answer
            self.trace.append({"kind": "model_given_up", "status": status, "fails": self.llm_fails})
            return "next"
        if status in (403, 404):
            self.banned_models.add(model)         # this model is refused for this key: the next one goes on
            if {MAIN_MODEL, BACKUP_MODEL} <= self.banned_models:
                self.llm_dead = True
            return "next"
        if status == 400:
            self.llm_fails += 1                   # a body this model would not take: the next model goes on, no wait
            return "next"
        return status

    def size_cost(self, model: str, body: dict[str, Any], message: dict[str, Any], stored: list[dict[str, Any]]) -> float:
        """The price of a call from its size (about four characters a token, tool schemas included). A tool turn on a
        cached model reads the prefix the previous tool turn wrote (the stored conversation up to its newest message,
        byte-identical, with the same tools) at CACHE_READ and writes the rest at CACHE_WRITE; any other call is priced
        whole."""
        rate_in, rate_out = PRICE_IN_OUT.get(model, (3.0, 15.0))
        try:
            tools = json.dumps(body.get("tools") or [], ensure_ascii=False)
            size_in = len(json.dumps(stored, ensure_ascii=False)) + len(tools)
            size_out = len(str(message.get("content") or "")) + len(json.dumps(message.get("tool_calls") or [], ensure_ascii=False))
            prefix = len(json.dumps(stored[:-1], ensure_ascii=False)) + len(tools)
        except (TypeError, ValueError):
            return 0.0
        price_in = float(size_in)
        if model.startswith("anthropic/") and body.get("tools"):
            read = min(self.cached_size, size_in) if tools == self.cached_tools else 0
            price_in = CACHE_READ * read + CACHE_WRITE * (size_in - read)
            self.cached_size, self.cached_tools = prefix, tools
        return (price_in * rate_in + size_out * rate_out) / CHARS_PER_TOKEN / 1e6

    def spent(self) -> float:
        """Spend booked so far: the model calls plus a charge for every page opened for fitment text."""
        return self.cost + self.page_reads * PAGE_CHARGE

    def env_trouble(self, result: Any, payload: dict[str, Any]) -> str:
        """What a failed environment reply means: 'stop' (budget spent, episode over, or an error not worth repeating),
        'again' (the step counter was realigned, or the reply was lost and the same step is resent)."""
        error = result.error or {}
        status, body = error.get("status"), str(error.get("body") or "")
        self.trace.append({"kind": "env_failure", "status": status, "body": body[:300]})
        if status == 402:
            self.dead = True
            return "stop"
        if status == 409:
            self.finished = self.finished or "finished" in body
            expected = re.search(r"expected (\d+)", body)
            if "finished" in body or not expected:
                return "stop"
            self.step = int(expected.group(1)) - 1
            payload["step"] = self.step + 1
            return "again"
        if status in PASSING or status is None:
            return "wait"                             # a passing fault of the service: the same step again after a short wait
        return "stop" if status in (400, 401, 403, 404, 422) else "again"

    def pause(self, attempt: int, seed: str) -> bool:
        """A short wait before a retry, with a jitter drawn from the call's own text (never from the clock), inside the
        episode's wait budget and never into the finish reserve. False when no wait fits."""
        jitter = (sum(ord(c) for c in seed[-64:]) % 97) / 97.0
        wait = min(6.0, (1.0, 2.5, 5.0)[min(attempt, 2)] + jitter)
        if self.waited + wait > WAIT_BUDGET or self.time_left() - wait < FORCE_SECONDS + 5:
            return False
        self.waited += wait
        time.sleep(wait)
        return True

    def env_call(self, actions: list[dict[str, Any]], final: bool = False) -> dict[str, Any] | None:
        """One envelope with one action or a group; three tries at most, never past the clock. The finishing call may run
        into the margin between the episode's own clock and the hard task limit (FINAL_SECONDS)."""
        payload: dict[str, Any] = {**self.binding, "step": self.step + 1}
        talk = self.roles.get("message")
        if talk and any(a.get("name") == talk for a in actions) and self.messages_used >= self.message_limit():
            self.trace.append({"kind": "message_cap_guard", "used": self.messages_used, "limit": self.message_limit()})
            return None                               # never a message past the cap, whichever path sent it
        actions = [self.as_real(a) for a in actions]
        payload.update({"action": actions[0]} if len(actions) == 1 else {"actions": actions})
        waits = 0
        for attempt in range(5):
            left = self.time_left() + (FINAL_SECONDS - TASK_SECONDS if final else 0.0)
            if left < 2:
                return None
            _env.timeout = int(max(3, min(95, left - 1)))
            try:
                result = _env.post_verbose("/environment/call", payload)
            except Exception as exc:                # a client fault is a passing fault, never a crash
                self.trace.append({"kind": "env_failure", "status": None, "body": f"{type(exc).__name__}: {exc}"[:300]})
                if waits >= 2:
                    return None
                self.pause(waits, json.dumps(actions)[:200] + str(attempt))
                waits += 1
                continue
            if isinstance(result.data, dict):
                data = result.data
                names = [a.get("name") for a in actions]
                # the store's own text is kept beside the agent's parse of it: the model reads the first, the code the second
                if "observation" in data:
                    data = {**data, "raw_observation": data["observation"],
                            "observation": settle_observation(self.unpacked(data["observation"], names[0] if len(names) == 1 else None))}
                if isinstance(data.get("observations"), list):
                    data = {**data, "raw_observations": list(data["observations"]),
                            "observations": [settle_observation(self.unpacked(o, names[i] if i < len(names) else None)) for i, o in enumerate(data["observations"])]}
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + len(actions)
                self.finished = self.finished or bool(data.get("finished"))
                if final:
                    self.final_reply = data
                return data
            verdict = self.env_trouble(result, payload)
            if verdict == "stop":
                return None
            if verdict == "wait":
                if waits >= 2:
                    return None
                self.pause(waits, json.dumps(actions)[:200] + str(attempt))   # no room for a wait: the retry goes at once
                waits += 1
            elif attempt >= 2:
                return None
        return None

    def unpacked(self, observation: Any, name: Any) -> Any:
        """A result with its packed records opened (unpack_records); a message's reply is the shopper's text and stays as
        sent. A result whose records came packed is counted, so the prompts know the store writes records as text."""
        if name is not None and name == self.roles.get("message"):
            return observation
        opened = unpack_records(observation)
        if opened != observation and isinstance(observation, (dict, list, str)):
            self.packed_results += 1
        return opened

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
        message = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, BACKUP_MODEL], 2000)
        plan = _parse_json_object((message or {}).get("content") or "") or {}
        # a reading that names the keys in other words still fills the plan
        for ours, alias in (("product", "item"), ("price_cap", "price_limit"), ("step_budget", "step_limit"), ("in_stock_only", "in_stock_required")):
            if plan.get(ours) is None and plan.get(alias) is not None:
                plan[ours] = plan[alias]
        if plan.get("priority") is None and isinstance(plan.get("preference"), str):
            plan["priority"] = "cheapest" if "cheap" in plan["preference"].lower() else "best" if plan["preference"].lower().startswith("best") else "none"
        self.plan = plan
        roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        # the reading's roles stand unless the tool's parameters cannot serve the role at all: a name or a description
        # never vetoes what the reading (or the schema) shows; the name cues only fill roles the reading left empty
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas and self.shape_allows(k, self.tool_shape(v))}
        for role, name in self.roles_by_name().items():
            self.roles.setdefault(role, name)      # a failed or partial reading still leaves every role filled
        self.split_view_mode()
        if self.roles.get("order") and self.roles.get("order") == self.roles.get("decline"):
            self.code_finish_modes()               # one finishing tool for both endings: its modes are read off its values
        modes = plan.get("finish_args")
        if isinstance(modes, dict):
            self.finish_args = {k: v for k, v in modes.items() if k in ("order", "decline") and isinstance(v, dict)}
            for kind, values in list(self.finish_args.items()):    # a mode naming a field the tool does not have is no mode
                fields = (self.schemas.get(self.roles.get(kind) or "") or {}).get("properties") or {}
                kept = {k: v for k, v in values.items() if k in fields}
                if kept:
                    self.finish_args[kind] = kept
                else:
                    del self.finish_args[kind]
            if len(self.finish_args) < 2 and self.roles.get("order") and self.roles.get("order") == self.roles.get("decline"):
                self.finish_args = {}
                self.code_finish_modes()
        budget = plan.get("message_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget >= 0:
            self.message_budget = int(budget)
        for key, value in self.policy_counts():
            if re.search(r"(?i)messag|question|note|chat|contact|\bask|allowance", key) and not re.search(r"(?i)step|turn|call|move|action|tool", key):
                if self.message_budget is None or value < self.message_budget:
                    self.message_budget = value       # a cap the episode states beside the request (its own field)
        worded = stated_message_cap(str(self.policy.get("query") or ""))
        if worded is not None and (self.message_budget is None or worded < self.message_budget):
            self.message_budget = worded              # the request's own words on how many messages it allows
        self.stated = [str(r) for r in plan.get("requirements") or [] if r]
        for need in self.stated:
            self.note_pointer(need)
        for duty in plan.get("duties") or []:
            if isinstance(duty, dict):
                self.add_duty(duty.get("do"), duty.get("when"), duty.get("tell"))
            elif isinstance(duty, str):
                self.add_duty(duty, None, None)
        self.companions = [str(c) for c in plan.get("companions") or [] if isinstance(c, str) and c.strip()][:3]
        self.bundle_names = self.split_products(plan)
        self.budget_now = plan.get("price_cap")
        if plan and self.budget_now is None and self.roles.get("message"):
            ask = [str(a) for a in plan.get("ask_first") or [] if a]
            plan["ask_first"] = ["the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"] + ask
        steps = plan.get("step_budget")
        if isinstance(steps, (int, float)) and not isinstance(steps, bool) and 0 < steps < self.max_steps:
            self.max_steps = int(steps)

    def policy_counts(self) -> list[tuple[str, int]]:
        """Whole numbers the episode's own fields carry beside the request and the tools (one level of nesting), with
        their key paths split into words ('maxMessages' -> 'max messages')."""
        out: list[tuple[str, int]] = []
        stack = [("", self.policy, 0)]
        while stack:
            path, value, depth = stack.pop()
            if isinstance(value, dict) and depth < 3:
                for key, inner in value.items():
                    if key not in ("query", "tools", "max_steps", "max_calls_per_turn"):
                        words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(key)).replace("_", " ").replace("-", " ").lower()
                        stack.append((path + " " + words, inner, depth + 1))
            elif isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 50:
                out.append((path.strip(), value))
        return out

    def add_duty(self, text: Any, when: Any, tell: Any) -> bool:
        """Record one duty once; 'when' falls back to the finish and a duty naming the shopper's ear takes a message."""
        text = str(text or "").strip()
        if not text or any(d["do"].lower() == text.lower() for d in self.duties):
            return False
        moment = str(when or "").strip().lower()
        moment = moment if moment in ("finish", "order", "decline", "replace", "anytime") else "finish"
        if moment == "finish" and re.search(r"(?i)replac|swap|substitut|switch", text):
            moment = "replace"
        talks = bool(tell) if isinstance(tell, bool) else bool(re.search(r"(?i)tell|confirm|let (?:me|them) know|check with|ask|message|summar|thank|update", text))
        if moment == "finish" and re.search(r"(?i)come up empty|empty.?handed|nothing (?:fits|works|matches)|no luck|can'?t find|cannot find|find nothing|if (?:there'?s|there is) nothing|none (?:fit|work)", text):
            moment = "decline"
        if talks and moment == "decline" and self.finish_says("decline"):
            talks = False          # the decline tool carries its own words to the shopper; no separate message is owed
        self.duties.append({"do": text, "when": moment, "tell": talks, "done": False})
        return True

    def finish_says(self, kind: str) -> bool:
        """Whether the tool that ends the episode this way takes free text for the shopper (a remark, note or reason)."""
        schema = self.schemas.get(self.roles.get(kind) or "") or {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        return any(isinstance(v, dict) and v.get("type") == "string" and not v.get("enum")
                   and re.search(r"(?i)remark|note|message|reason|text|comment|explan|justif|say|summary", str(k))
                   for k, v in props.items())

    def duties_due(self, kind: str) -> list[dict[str, Any]]:
        return [d for d in self.duties if not d["done"] and d["when"] in ("finish", kind)]

    def message_limit(self) -> int:
        """The messages the agent may send in all. A cap the request states or the store reports is kept to; with none
        known, a refused message would already break the cap, so the agent stays at the least any task may grant: one
        message (naming every product's keys), a second only when the first drew no firm value at all, and one more for
        a reply the request itself demands at the finish."""
        if self.message_budget is not None:
            return self.message_budget
        # cam47: one property per message, so an unstated cap allows UNSTATED_QUESTIONS questions (a store that caps lower
        # says so with a status code, which then becomes the cap), plus the message a duty at the finish takes
        return self.unstated_questions() + int(any(d["tell"] for d in self.duties))

    def unstated_questions(self) -> int:
        """Questions when no cap is stated or reported: UNSTATED_QUESTIONS, or one per product of a larger bundle (each
        message names one property of one product), never more than QUESTION_CAP."""
        return min(QUESTION_CAP, max(UNSTATED_QUESTIONS, len(self.bundle_names)))

    def held_messages(self) -> int:
        """Messages kept back: one for the finish while a duty that takes a message is still to be done, and one for asking
        before a replacement while the request wants that and none was asked yet."""
        finish = any(d["tell"] and not d["done"] and d["when"] in ("finish", "order", "decline") for d in self.duties)
        swap = any(d["when"] == "replace" for d in self.duties) and not self.swap_asked
        return int(finish) + int(swap)

    def loop_floor(self) -> int:
        """Steps left at which the model's turns stop and the agent finishes by itself: the finish reserve counted for an
        empty cart (a fresh add, the order, a decline after a refusal, a message owed). The batch cut uses the same floor,
        so a reply that spends its last allowed step never leaves the finish short."""
        return self.finish_reserve(empty_cart=True)

    def commit_steps(self) -> int:
        """Steps left under which, with a fit known, only committing is worth a step (12 of 38; fewer in short episodes)."""
        return min(COMMIT_STEPS, round(self.max_steps * COMMIT_SHARE))

    def pace_targets(self) -> tuple[int, int]:
        """The steps by which a fit should be verified and the order placed, from the step cap in force."""
        commit_by = min(round(self.max_steps * COMMIT_BY), self.max_steps - REPAIR_RESERVE)
        return max(4, round(self.max_steps * FIT_BY)), max(6, commit_by)

    def paced_fit(self) -> str:
        """A listing the agent could commit to now: its own sheet states every requirement in force, it is not gone, and
        no independent check has failed it under the requirements as they stand. Cheapest first; "" when there is none."""
        for ident, _ in self.live_fits():
            if not self.verdicts.get(f"{ident}|{self.ledger_version}"):
                return ident
        return ""

    def pace(self) -> tuple[str, str]:
        """Where the episode stands against the pace: ("", fit) on pace, ("fit", "") when no fit is known past the fit
        target, ("commit", fit) when a fit is known past the commit target and nothing was ordered yet."""
        fit_by, commit_by = self.pace_targets()
        fit = self.paced_fit()
        if not fit:
            return ("fit" if self.step >= fit_by else "", "")
        if self.order_attempts == 0 and self.step >= commit_by:
            return "commit", fit
        return "", fit

    def commit_time(self) -> str:
        """The fit to commit to when the steps left are few enough that only committing is worth one ("" otherwise)."""
        return self.paced_fit() if self.steps_left() <= self.commit_steps() else ""

    def finish_reserve(self, empty_cart: bool = False, swap: bool = True) -> int:
        """Steps the finish needs: the order or decline plus one repair, a message for every duty still owed, and, when the
        cart is empty, one more for putting the pick in (a fresh line can be refused at the order and still needs a decline)."""
        cart = bool(self.roles.get("cart"))
        fresh = 1 if empty_cart and cart and not self.cart_ids else 0
        # a held single line can be refused at the order by a change (a lower budget, a sellout): the swap that repairs
        # it takes the line out and puts the replacement in before the order goes again, two steps the reserve keeps
        room = 2 if swap and cart and (self.cart_ids or fresh) and not self.companions and not self.bundle_names and self.max_steps >= 15 else 0
        return 2 + self.held_messages() + fresh + room

    def long_world(self) -> bool:
        """A long episode (the step cap at LONG_WORLD or above): the store changes more than once, the shopper has more to
        reveal, and a wrong early finish throws away the steps that would have found the right ending."""
        return self.max_steps >= LONG_WORLD

    def wordy_world(self) -> bool:
        """A store whose rows carry long free text (LONG_TEXT characters a row on average over at least five rows): every
        result and every check prompt is several times larger, so the spend lines tighten and the checks read light."""
        sizes = list(self.text_sizes.values())
        return len(sizes) >= 5 and sum(sizes) / len(sizes) >= LONG_TEXT

    def spend_caps(self) -> tuple[float, float, float]:
        """The soft, hard and stop spend lines for this episode's step cap (see the table at HARD_SHORT); WORDY_SHARE of
        them in a store with long row texts."""
        share = min(1.0, max(0.0, (self.max_steps - SHORT_CAP) / float(LONG_CAP - SHORT_CAP)))
        hard = HARD_SHORT + (HARD_LONG - HARD_SHORT) * share + BUNDLE_EXTRA * min(3, max(0, len(self.bundle_names) - 1))
        if self.wordy_world():
            hard *= WORDY_SHARE
        return (SOFT_SHARE * hard, hard, max(STOP_LINE, hard + 0.005))

    def room_for_extras(self) -> bool:
        """Whether optional care (a hold, a repeated look, a second recovery) may take a step or a model turn: well clear
        of the finish reserve, and never while it would push the first order past the commit mark."""
        if self.steps_left() < self.finish_reserve(empty_cart=True) + EXTRA_ROOM:
            return False
        return self.order_attempts > 0 or self.step < self.pace_targets()[1]

    def spend_level(self) -> int:
        """0 below the soft line, 1 past it (optional checks go light), 2 past the hard line (finishing work only)."""
        soft, hard, _ = self.spend_caps()
        spent = self.spent()
        return 2 if spent >= hard else 1 if spent >= soft else 0

    def light_checks(self) -> bool:
        """Whether the optional checks should take the light reader first: past the soft line, or in a long world once
        half the soft line is spent (the long world has many more checks ahead of it)."""
        return self.spend_level() >= 1 or self.spent() >= self.spend_caps()[0] / 2 or self.wordy_world()

    # role cues read on a tool's name and description together (each short; the shapes of the parameters decide the rest)
    ROLE_CUES = {"order": r"check.?out|purchas|\bbuy(?!er)|\border|settle|\bpay",
                 "decline": r"declin|abstain|give.?up|walk.?away|nothing|without|\bno\b|\bstop",
                 "message": r"messag|shopper|customer|\bask|question|contact",
                 "cart": r"cart|basket|\bbag|trolley|\badd\b|reserv",
                 "details": r"detail|spec|sheet|propert|attribut|fact",
                 "view": r"\bview|page|\bfull|record|card|\bopen",
                 "search": r"search|find|look.?up|query|browse|catalog",
                 "stock": r"stock|availab|on.?hand",
                 "compare": r"compar|side by side|weigh|versus"}
    FINAL_CUE = r"final|terminal|\bends?\b|no (?:more|further)|nothing (?:can|else)|clos(?:e|es|ing)\b|finish"
    ADD_CUE = r"add|put|includ|insert|plus|hold"

    def tool_shape(self, name: str) -> dict[str, Any]:
        """What a tool's schema says about it: its free text, choice and number slots, what it requires, and whether its
        description calls it final."""
        tool = next((t["function"] for t in self.tools if t["function"]["name"] == name), {})
        schema = self.schemas.get(name) or {}
        props = {k: v for k, v in (schema.get("properties") or {}).items() if isinstance(v, dict)}
        kinds = lambda v: v.get("type") if isinstance(v.get("type"), list) else [v.get("type")]
        free = [k for k, v in props.items() if "string" in kinds(v) and not v.get("enum")]
        desc = str(tool.get("description") or "").lower()
        return {"name": name.replace("_", " ").replace("-", " ").lower(), "desc": re.sub(self.FINAL_CUE, " ", desc), "props": props,
                "free": free, "enums": [k for k, v in props.items() if v.get("enum")],
                "numbers": [k for k, v in props.items() if not v.get("enum") and ("integer" in kinds(v) or "number" in kinds(v))],
                "required": [k for k in schema.get("required") or [] if k in props],
                "basket": any(v.get("type") == "array" and isinstance(v.get("items"), dict) and (v["items"].get("properties") or {}) for v in props.values()),
                "ids": [k for k, v in props.items() if v.get("type") == "array" and isinstance(v.get("items"), dict) and v["items"].get("type") == "string"],
                "flags": [k for k, v in props.items() if "boolean" in kinds(v)],
                "final": bool(re.search(self.FINAL_CUE, desc + " " + name.lower())) and not (re.search(self.ROLE_CUES["search"], name.lower() + " " + desc)
                                                                                         and not re.search(self.ROLE_CUES["order"], name.lower()))}

    @staticmethod
    def shape_allows(role: str, shape: dict[str, Any]) -> bool:
        """Whether a tool's parameters can carry a role at all, read off the schema alone: a message needs a free-text
        slot, a cart line or a reading needs a slot for the listing (text, a list of ids, a number or a basket), a cart
        view takes no required argument, and a search takes some argument."""
        target = bool(shape["free"] or shape["ids"] or shape["numbers"])
        if role == "message":
            return bool(shape["free"])
        if role in ("cart", "remove"):
            return target or bool(shape["basket"])
        if role == "cart_view":
            return not shape["required"]
        if role in ("details", "view", "stock"):
            return target
        if role == "search":
            return bool(shape["props"])
        return True

    def role_score(self, role: str, shape: dict[str, Any]) -> float:
        """How well one tool fits one role: its cues on the name and the description, then its parameter shape."""
        cue = self.ROLE_CUES.get(role, self.ROLE_CUES["cart"] if role in ("cart_view", "remove") else "")
        on_name, on_desc = bool(re.search(cue, shape["name"])), bool(re.search(cue, shape["desc"]))
        score = 2.5 * on_name + 2.5 * on_desc
        n, free, final = len(shape["props"]), shape["free"] + shape["ids"], shape["final"]
        removing = bool(re.search(self.REMOVES, shape["name"]) or re.search(self.REMOVES + r"|\bout\b", shape["desc"]) and not shape["enums"])
        if role in ("order", "decline"):
            score += 3.0 if final else -1.0
            if role == "order" and re.search(self.ROLE_CUES["decline"], shape["name"] + " " + shape["desc"]):
                score -= 4.0                      # "end without buying" names buying too
            if role == "decline" and not (on_name or on_desc):
                score -= 3.0
            if len(free) > 1 or shape["numbers"] and role == "decline":
                score -= 2.0
        elif final:
            score -= 10.0
        if role == "message":
            score += (1.0 if n == 1 else 0.0) - (10.0 if not free else 0.0) - (3.0 if shape["numbers"] or shape["enums"] else 0.0)
        if role == "cart":
            score += (2.0 if shape["enums"] or shape["numbers"] or shape["basket"] or shape["ids"] else 0.0) - (10.0 if not free and not shape["basket"] else 0.0) - (2.0 if removing and not shape["enums"] else 0.0) - (0.0 if on_name or on_desc else 3.0)
        if role == "remove":
            score += (3.0 if removing else -10.0) - (10.0 if not free or shape["enums"] else 0.0)
        if role == "cart_view":
            score += (2.0 if not shape["required"] else -10.0) - (3.0 if free else 0.0) - (3.0 if re.search(self.ROLE_CUES["order"], shape["name"] + " " + shape["desc"]) else 0.0) - (0.0 if on_name or on_desc else 3.0)
        if role in ("details", "view", "stock"):
            score += (1.0 if len(shape["required"]) == 1 and free else -10.0 if not free else -2.0) - (3.0 if n > 2 else 0.0) - (1.0 if shape["ids"] else 0.0)
        if role == "compare":
            score += 0.0 if n >= 1 else -10.0
        if role == "search":
            score += (2.0 if n >= 2 or shape["numbers"] else 0.0) - (10.0 if not n else 0.0) - (3.0 if len(shape["required"]) == 1 and n == 1 and not on_name else 0.0)
        return score

    def roles_by_name(self) -> dict[str, str]:
        """Tool roles read off each tool's schema: the parameter shapes and the cues of its name and description, each tool
        taking the role it fits best, so the guards work when the setup reading fails or names tools that do not exist."""
        names = [t["function"]["name"] for t in self.tools]
        shapes = {n: self.tool_shape(n) for n in names}
        roles = ("order", "decline", "message", "cart", "remove", "cart_view", "stock", "compare", "details", "view", "search")
        # the schema decides which roles a tool can take at all; among those, the shape and the cues rank the pairs
        pairs = sorted(((self.role_score(r, shapes[n]), -names.index(n), -roles.index(r), r, n) for r in roles for n in names
                        if self.shape_allows(r, shapes[n])), reverse=True)
        found: dict[str, str] = {}
        for floor in (2.0, 1.0):
            for score, _, _, role, name in pairs:
                if score < floor or role in found or name in found.values():
                    continue
                found[role] = name
        if "view" in found and "details" not in found:
            found["details"] = found.pop("view")    # the one reading tool reads the sheets
        if "view" not in found and "details" not in found:
            single = [n for n in names if n not in found.values() and self.role_score("details", shapes[n]) > -5]
            if single:
                found["details"] = single[0]
        found.pop("stock", None)
        found.pop("compare", None)
        for one, other in (("order", "decline"), ("decline", "order")):
            tool = found.get(other)
            if one not in found and tool and shapes[tool]["final"] and (any(len(shapes[tool]["props"][k]["enum"]) >= 2 for k in shapes[tool]["enums"])
                                                                        or any(k in shapes[tool]["required"] for k in shapes[tool]["flags"])):
                found[one] = tool                     # one finishing tool whose mode picks buying or ending without a purchase
        finals = [n for n in names if shapes[n]["final"]]
        if len(finals) == 1 and ("order" not in found or "decline" not in found):
            found.setdefault("order", finals[0])  # the only ending: it buys a filled basket and ends without buying on an empty one
            found.setdefault("decline", finals[0])
        return found

    def empty_for_decline(self) -> None:
        """One ending tool for both outcomes and no switch telling them apart: a decline first empties the basket."""
        order, decline = self.roles.get("order"), self.roles.get("decline")
        if not order or order != decline or self.finish_args.get("decline") or self.dead:
            return
        for ident in list(self.cart_ids):
            call = self.remove_args(ident)
            if not call or self.steps_left() <= 1:
                return
            reply = self.env_call([{"name": call[0], "args": call[1]}], final=True)
            self.trace.append({"kind": "empty_for_decline", "item": ident, "reply": reply})
            if reply is None:
                return
            self.cart_ids = [i for i in self.cart_ids if i != ident]

    def cart_modes(self, name: str | None) -> dict[str, tuple[str, str]]:
        """Each choice slot of a cart tool that sets its mode: (the adding value, the removing value). Read once per tool:
        the values' own words, else the description's words right after each value, else the first value adds."""
        if not name:
            return {}
        cached = self.mode_cache.get(name)
        if cached is not None:
            return cached
        shape = self.tool_shape(name)
        modes: dict[str, tuple[str, str]] = {}
        for key in shape["enums"]:
            values = [str(v) for v in shape["props"][key]["enum"] if not any(m.get(key) == v for real, m in self.virtual.values() if real == name)]
            if len(values) < 2:
                continue
            told = (self.plan.get("cart_modes") or {}) if isinstance(self.plan.get("cart_modes"), dict) else {}
            add = told.get("add") if told.get("add") in values else None
            out = told.get("remove") if told.get("remove") in values else None
            out = out or next((v for v in values if re.search(self.REMOVES, v.replace("_", " ").lower()) and v != add), None)
            add = add or next((v for v in values if re.search(self.ADD_CUE, v.replace("_", " ").lower()) and v != out), None)
            if out is None:
                for v in values:
                    after = shape["desc"].partition(v.lower())[2][:40]
                    if v != add and after and re.search(r"\bout\b|\boff\b|" + self.REMOVES, after):
                        out = v
                        break
            if add is None and out is None and not re.search(r"mode|action|step|kind|op|type|do", key.lower()):
                continue                          # a choice slot that is not a mode (a size, a colour)
            add = add or next(v for v in values if v != out)
            out = out or next((v for v in values if v != add), None)
            if out:
                modes[key] = (add, out)
        self.mode_cache[name] = modes
        return modes

    VIEW_NAME = "view_cart_lines"

    def split_view_mode(self) -> None:
        """A cart tool that also lists the cart in one of its modes ("with mode 'look' it lists the basket instead") is
        shown to the model as two tools: the cart itself and a separate argument-free cart view, which the agent sends as
        the cart tool in that mode."""
        cart = self.roles.get("cart")
        if not cart or (self.roles.get("cart_view") and self.roles.get("cart_view") != cart) or self.VIEW_NAME in self.schemas:
            return
        shape = self.tool_shape(cart)
        for key in shape["enums"]:
            values = [str(v) for v in shape["props"][key]["enum"]]
            if len(values) < 3:
                continue
            desc = str(next((t["function"].get("description") for t in self.tools if t["function"]["name"] == cart), "") or "").lower()
            view = next((v for v in values if re.search(r"\b(?:list|show|view|look|see)", v.replace("_", " ").lower())), None)
            view = view or next((v for v in values if re.search(r"\b(?:lists|shows|views)\b", desc.partition("'" + v.lower() + "'")[2].split("'")[0][:60])), None)
            if view is None:
                continue
            self.virtual[self.VIEW_NAME] = (cart, {key: view})
            self.tools.append({"type": "function", "function": {"name": self.VIEW_NAME, "description": "Show the cart's lines and total (no arguments).",
                                                                  "parameters": {"type": "object", "properties": {}, "required": []}}})
            self.schemas[self.VIEW_NAME] = {"type": "object", "properties": {}, "required": []}
            self.roles["cart_view"] = self.VIEW_NAME
            self.mode_cache.pop(cart, None)
            return

    def as_virtual(self, name: Any, raw: Any) -> tuple[Any, Any]:
        """A model call of the cart tool in its listing mode, renamed to the virtual cart view."""
        for virtual, (real, mode) in self.virtual.items():
            if name == real:
                args = raw if isinstance(raw, dict) else (_parse_json_object(raw or "{}") or {})
                if all(args.get(k) == v for k, v in mode.items()):
                    return virtual, "{}"
        return name, raw

    def as_real(self, action: dict[str, Any]) -> dict[str, Any]:
        """The action as the store knows it: a virtual tool becomes the real tool in its mode."""
        if isinstance(action, dict) and action.get("name") in self.virtual:
            real, mode = self.virtual[action["name"]]
            return {**action, "name": real, "args": dict(mode)}
        return action

    def removing_call(self, name: str, args: dict[str, Any]) -> bool:
        """Whether a call takes a line out: the cart tool in its removing mode, or the separate removing tool."""
        if name == self.roles.get("remove"):
            return True
        return name == self.roles.get("cart") and any(args.get(k) == out for k, (_, out) in self.cart_modes(name).items())

    @staticmethod
    def flat_sheet(sheet: dict[str, Any]) -> dict[str, Any]:
        """One flat label -> value table from a stored record: its own scalar fields plus the facts table it carries."""
        flat = {k: v for k, v in sheet.items() if not isinstance(v, (dict, list))}
        stack = [(v, 1) for v in sheet.values() if isinstance(v, dict)]
        while stack:
            value, depth = stack.pop(0)
            flat.update({k: v for k, v in value.items() if isinstance(v, (str, int, float)) and k not in flat})
            stack += [(v, depth + 1) for v in value.values() if isinstance(v, dict) and depth < 3]
        return flat

    def wider_shelf(self, item: str) -> str:
        """A request naming two kinds ('pants or jeans') means the store's combined category ('Pants & Jeans') when it has one;
        a pick from a category carrying fewer of those words ('Jeans') is questioned once."""
        kind = {stem(w) for w in word_list(self.plan.get("product") or "", False) if len(w) >= 3 and w not in SMALL_WORDS and w not in ("for", "pair", "set")}
        if len(kind) < 2 or self.bundle_names:
            return ""                                   # several products named together are a bundle, not one combined kind
        share = lambda label: len(kind & {stem(w) for w in word_list(label, False)})
        mine = _shelf(self.rows.get(item) or {}) or _shelf(self.sheets.get(item) or {})
        labels = {_shelf(r) for r in list(self.rows.values()) + list(self.sheets.values())} - {""}
        if not mine or not labels:
            return ""
        top = max(labels, key=share)
        if share(top) < 2 or share(mine) >= share(top):
            return ""
        return (f"{item} is in the store category {mine!r}, while the request asks for {self.plan.get('product')!r} and the store has the category {top!r}, which carries "
                "every word of that request. Two kinds named together mean the store's combined category: choose from it. If you are certain, send the same call again.")

    def brief(self) -> str:
        """The plan rendered for the model, one line per fact, the request itself being the authority."""
        plan = self.plan
        if not plan:
            return ""
        many = lambda key, label: [f"- {label}{x}" for x in plan.get(key) or []]
        priority = plan.get("priority")
        rule = (f"- choose the {priority}" + (f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else "") + " among all fitting items"
                if priority in ("cheapest", "best") else
                "- no cheapest/best rule was found: any item that meets every requirement within the budget is right, so do not spend steps hunting for the lowest price (unless the request itself says otherwise)")
        lines = (["A first reading of the request (check it against the request itself):"]
                 + ([f"- product: {plan.get('product')}"] if plan.get("product") else [])
                 + [f"- quantity: {plan.get('quantity') or 1}"]
                 + many("requirements", "stated requirement: ")
                 + ([f"- price cap (total): {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip()] if plan.get("price_cap") is not None else [])
                 + (["- only items in stock / on hand"] if plan.get("in_stock_only") else [])
                 + [rule]
                 + many("soft_wants", "nice to have, NOT required (only a tie-breaker): ")
                 + many("ask_first", "must be learned from the shopper first: ")
                 + many("rules", "rule: ")
                 + ([f"- this request needs several products bought together: {plan.get('product')} and " + "; ".join(self.companions)
                     + " (one cart line per product; each must go with the others by its own records)"] if self.companions else [])
                 + [f"- duty ({d['when']}): {d['do']}" for d in self.duties]
                 + ([f"- messages to the shopper allowed: {self.message_budget}"] if self.message_budget is not None else [])
                 + [f"- steps allowed: {self.max_steps}"])
        hinted = [h for h in plan.get("hint_properties") or [] if isinstance(h, str)]
        if hinted:
            lines.append("- properties the story may hint at (ask about these first, one per message): " + ", ".join(hinted))
        return "\n".join(lines)

    def pressure_line(self, left: int) -> str:
        """What the status presses for as the steps or seconds run out ("" while there is room)."""
        if left <= 2:
            return " LAST STEPS: order now if the cart holds a fit; decline only when no listing fits at all."
        if self.spend_level() == 2 and self.order_attempts == 0:
            return " Keep each reply short: search and read only what the decision still needs, then cart the best verified fit and order it."
        if left <= 5 or self.time_left() < WRAP_UP_SECONDS:
            return " FINISH NOW: fix the cart to one verified line, re-check its stock, then order (or decline if nothing fits)."
        fit = self.commit_time()
        if fit and self.order_attempts == 0:
            return (f" COMMIT NOW: {fit} states every requirement on its own sheet and is in stock; put it in the cart (if it is not there) and order it."
                    " With this few steps left, no question, search or reading is worth a step; a refused order still needs about 4 steps to repair.")
        behind, fit = self.pace()
        if behind == "commit" and self.plan.get("priority") != "cheapest":
            return (f" BEHIND THE PACE: {fit} is a verified fit and the order should be in by step {self.pace_targets()[1]}; put it in the cart and order now."
                    " Further searching or reading is only for a listing whose check failed.")
        if behind == "commit":
            return (f" BEHIND THE PACE: {fit} is the cheapest verified fit so far and the order should be in by step {self.pace_targets()[1]};"
                    " read only the cheaper listings not yet opened, then cart the cheapest fit and order.")
        if behind == "fit":
            return (f" BEHIND THE PACE: no listing has been verified yet and a fit should be known by step {self.pace_targets()[0]};"
                    " open the details of the best candidates now and settle on one.")
        if self.order_attempts == 0 and left <= REPAIR_RESERVE + 7:
            return (" Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end"
                    f" when {REPAIR_RESERVE} steps remain, and a refused order needs about 4 steps to repair.")
        return ""

    def status(self) -> str:
        """The line appended to every tool result: steps, messages and seconds left, the requirements in force, the story's
        hints, and the finish pressure."""
        left = self.steps_left()
        sent = f"messages sent {self.messages_used}" + (f" of {self.message_budget}" if self.message_budget is not None else "")
        clock = f"about {int(max(0, self.time_left() - FORCE_SECONDS))} s left"
        fit_by, commit_by = self.pace_targets()
        behind, fit = self.pace()
        standing = ("behind, no verified fit yet" if behind == "fit" else f"behind, {fit} fits: commit" if behind == "commit"
                    else f"on pace, {fit} fits" if fit else "on pace")
        pacing = f"pace: a fit by step {fit_by}, the order by step {commit_by} ({standing})"
        text = f"[status] steps used {self.step} of {self.max_steps}, {left} left; {sent}; {clock}; {pacing}.\n[requirements now] " + self.ledger()
        hints = self.hints()
        if hints:
            text += ("\n[story hints] " + "; ".join(f"{p} (likely {v})" if v else p for p, v in hints)
                     + ": ask about these before anything else, then prefer the fit that matches; never reject an item or decline over a hint.")
        undone = [d["do"] for d in self.duties if not d["done"] and d["when"] in ("finish", "order", "decline")]
        if undone:
            text += "\n[to do before finishing] " + "; ".join(undone[:4])
        pending = [p for p in self.pointers if not p["done"]]
        if pending:
            text += ("\n[pointer need] the value of " + ", ".join(p["prop"] or "a property" for p in pending) + " sits on another listing's page; the agent finds that page and reads it for you."
                     " Do not search for that listing's shop or attributes as if they were requirements on the item.")
        return text + self.fitment_line() + self.bundle_status() + self.pressure_line(left)

    def ledger(self) -> str:
        if self.current is not None:
            items = list(self.current)
        else:
            items = [f"{r} (stated)" for r in self.stated if r not in self.dropped]
            items += [f"{r} (revealed by the shopper)" for r in self.revealed if r not in self.dropped]
        text = "; ".join(items) if items else "see the request"
        if self.budget_now is not None:
            text += f"; total price at most {self.budget_now} {self.plan.get('currency') or ''}".rstrip()
        text += f"; quantity {self.plan.get('quantity') or 1}"
        if self.dropped:
            text += "; no longer required: " + "; ".join(self.dropped)
        if self.free:
            text += "; shopper does not care about: " + ", ".join(self.free)
        return text

    def intent(self, name: str, args: dict[str, Any]) -> str | None:
        """'order' or 'decline' for a call of a finishing tool; one tool may do both, told apart by its mode argument."""
        order, decline = self.roles.get("order"), self.roles.get("decline")
        if name not in (order, decline):
            return None
        if order != decline:
            return "order" if name == order else "decline"
        for kind in ("decline", "order"):
            want = self.finish_args.get(kind) or {}
            if want and all(lowered(args.get(k)) == lowered(v) for k, v in want.items()):
                return kind
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("enum") and args.get(key) is not None:
                return "order" if args.get(key) == spec["enum"][0] else "decline"
        for key, spec in properties.items():          # a yes/no switch on the one finishing tool: off ends without buying
            if isinstance(spec, dict) and "boolean" in str(spec.get("type")) and isinstance(args.get(key), bool):
                return "order" if args[key] else "decline"
        if self.cart_used and not self.cart_ids and not self.finish_args:
            return "decline"                          # one finishing tool without a mode: an empty cart ends without buying
        return "order"

    def id_keys(self) -> set[str]:
        """Names that denote the listing identifier: the key learned from the rows, and the identifier arguments of the
        reading and cart tools (their required free string, or the one free string they take)."""
        keys = {_key_stem(self.id_field)} if self.id_field else set()
        for role in ("details", "view", "cart"):
            tool = self.roles.get(role)
            schema = self.schemas.get(tool or "") or {}
            props = schema.get("properties") or {}
            free = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v]
            required = [k for k in schema.get("required") or [] if k in free]
            for key in (required or free)[:1]:
                keys.add(_key_stem(key))
        return keys

    def id_in(self, record: dict[str, Any]) -> str | None:
        """The identifier a row or sheet carries: under the learned key, under a key the tools name, else under a key with 'id'."""
        if not isinstance(record, dict):
            return None
        if self.id_field and isinstance(record.get(self.id_field), str) and _ID_HINT.match(record[self.id_field]):
            return record[self.id_field]
        keys = self.id_keys()
        for key, value in record.items():
            if isinstance(value, str) and _ID_HINT.match(value) and _key_stem(key) in keys:
                return value
        return next((v for k, v in record.items() if isinstance(v, str) and "id" in k.lower() and _ID_HINT.match(v)), None)

    def learn_id_field(self, rows: list[dict[str, Any]]) -> None:
        """When no row names its identifier in any known way, the one key whose values are distinct identifier-like tokens
        across the rows (no spaces, a shared leading form) is the identifier."""
        if self.id_field or len(rows) < 2 or any(self.id_in(r) for r in rows):
            return
        shared = set.intersection(*(set(r) for r in rows))
        best = None
        for key in shared:
            values = [r.get(key) for r in rows]
            if not all(isinstance(v, str) and " " not in v.strip() and _ID_HINT.match(v) for v in values):
                continue
            if len(set(values)) != len(values):
                continue
            heads = {re.match(r"[A-Za-z]*[-_:]?", v).group(0) for v in values}
            score = (len(heads) == 1, -len(key))
            if best is None or score > best[0]:
                best = (score, key)
        if best:
            self.id_field = best[1]

    @staticmethod
    def facts_like(value: Any) -> bool:
        return isinstance(value, dict) and len(value) >= 1 and all(isinstance(x, (str, int, float)) for x in value.values())

    def carries_facts(self, record: dict[str, Any]) -> bool:
        """A record with a nested label -> value block (at any of the first three levels)."""
        stack = [(v, 1) for v in record.values() if isinstance(v, dict)]
        while stack:
            value, depth = stack.pop()
            if self.facts_like(value):
                return True
            stack += [(v, depth + 1) for v in value.values() if isinstance(v, dict) and depth < 3]
        return False

    def keep_sheet(self, ident: str, record: dict[str, Any]) -> None:
        self.sheets[ident] = {k: (_brief(v, 160) if not isinstance(v, dict) else v) for k, v in record.items()}

    def learn_fields(self, rows: list[dict[str, Any]]) -> None:
        """The world's price and stock keys, decided once from the first rows that show them: the price is the money-like
        field whose values differ from row to row (a name that says price or cost first, a model reading when two fields
        are alike), the stock the field holding flags, counts or stock words."""
        rows = [r for r in rows if isinstance(r, dict)]
        if rows and len(rows) >= 2 and not FIELDS.get("title"):
            texts = [k for k in rows[0] if k != self.id_field and all(isinstance(r.get(k), str) and 0 < len(r[k]) <= 240 for r in rows)]
            named = [k for k in texts if len({r[k] for r in rows}) == len(rows) and sum(len(r[k].split()) for r in rows) / len(rows) >= 2.5]
            if named and not any(re.search(TITLE_LIKE, str(k), re.I) for k in texts):
                FIELDS["title"] = max(named, key=lambda k: (-len(set(word_list(self.plan.get("product") or "")) - set(" ".join(r[k] for r in rows).lower().split())), sum(len(r[k]) for r in rows)))
            product = {stem(w) for w in word_list(self.plan.get("product") or "") if len(w) > 2}
            if not FIELDS.get("group") and product and not any(re.search(GROUP_LIKE, str(k), re.I) for k in texts):
                shelves = [k for k in texts if k != FIELDS.get("title") and len({r[k] for r in rows}) < len(rows)
                           and all(len(r[k].split()) <= 4 and not r[k].rstrip().endswith(".") for r in rows)
                           and sum(1 for r in rows if product & {stem(w) for w in word_list(r[k])}) * 2 >= len(rows)]
                shelves.sort(key=lambda k: (not re.search(r"(?i)group|famil|path|type|kind", str(k)), len({r[k] for r in rows})))
                if shelves:
                    FIELDS["group"] = shelves[0]
        if len(rows) < 1 or (FIELDS.get("price") and FIELDS.get("stock")):
            return
        keys = [k for k in rows[0] if k != self.id_field and sum(1 for r in rows if k in r) * 2 >= len(rows)]
        if not FIELDS.get("price"):
            scored = []
            for key in keys:
                values = [r.get(key) for r in rows if key in r]
                money = [money_value(v) for v in values]
                if any(m is None for m in money) or any(isinstance(v, bool) for v in values):
                    continue
                named = bool(re.search(r"(?i)price|cost|amount|fee|\bpay|charge", str(key)))
                if not named and all(isinstance(v, str) and not re.search(r"[^\d.,\s]", v) for v in values):
                    continue                          # a number written as bare text is a code, not a price
                residue = {re.sub(r"[\d.,]+", "#", v).strip().lower() for v in values if isinstance(v, str)}
                if len(residue) > 1 and len(rows) > 1:
                    continue                          # a price's words around the figure are the same on every row; a text's are not
                distinct = len(set(money)) > 1 or len(rows) == 1
                small = all(m is not None and float(m).is_integer() and 0 <= m <= 10 for m in money)
                score = 3.0 * named + 1.0 * distinct - 2.0 * small - 2.0 * bool(re.search(r"(?i)total|count|qty|quant|rating|page", str(key)))
                score += 1.0 * any(isinstance(v, str) and re.search(r"[^\d.,\s]", v) for v in values)
                scored.append((score, -keys.index(key), key))
            scored.sort(reverse=True)
            if scored and scored[0][0] > 0:
                pick = scored[0][2]
                if len(scored) > 1 and scored[1][0] >= scored[0][0] - 0.5 and not self.llm_dead:
                    sample = _compact([{k: r.get(k) for k in (scored[0][2], scored[1][2])} for r in rows[:3]])
                    read = self.judge(PRICE_FIELD_PROMPT + "\n\nROWS:\n" + sample, 120, [READ_MODEL])
                    if isinstance(read, dict) and read.get("price") in (scored[0][2], scored[1][2]):
                        pick = read["price"]
                FIELDS["price"] = pick
                # prices written in the major unit ("$1.20") for a request that counts in the minor one (cents): one scale
                minor = bool(re.search(r"(?i)cent|\bcts?\b|¢|minor|penn|pence", str(self.plan.get("currency") or "") + " " + " ".join(re.findall(r"\d\s*(\S+)", str(self.policy.get("query") or ""))[:3])))
                written = [str(r.get(pick)) for r in rows if isinstance(r.get(pick), str)]
                if minor and written and all(re.fullmatch(r"\s*[^\d\s.,]{1,3}\s*\d[\d,]*\.\d{2}\s*", w) for w in written):
                    FIELDS["scale"] = 100.0
                objects = [r.get(pick) for r in rows if isinstance(r.get(pick), dict)]
                if minor and objects and len(objects) == len(rows) and any(major_amount(o) for o in objects):
                    FIELDS["scale"] = 100.0           # price objects in dollars and cents for a request that counts cents
        if not FIELDS.get("stock"):
            scored = []
            for key in keys:
                values = [r.get(key) for r in rows if key in r]
                if key == FIELDS.get("price") or any(stock_value(v) is None for v in values):
                    continue
                named = bool(re.search(r"(?i)stock|avail|on.?hand|order|buy|sell", str(key)))
                flags = all(isinstance(v, bool) for v in values)
                words = all(isinstance(v, str) and len(v.split()) <= 3 for v in values)
                counts = all(isinstance(v, int) and not isinstance(v, bool) for v in values)
                score = 3.0 * named + 2.0 * flags + 1.0 * (words and named) + 1.0 * (counts and named) - 3.0 * (not named and not flags)
                scored.append((score, -keys.index(key), key))
            scored.sort(reverse=True)
            if scored and scored[0][0] > 0:
                FIELDS["stock"] = scored[0][2]

    def records_in(self, observation: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
        """The single listing record a result carries (the shallowest dict naming an id and holding a label -> value
        block) and its longest list of row records, wherever the store wrapped them."""
        lists: list[list[dict[str, Any]]] = []
        single = None
        stack = [(observation, 0)]
        while stack:
            value, depth = stack.pop(0)
            if depth > 4:
                continue
            if isinstance(value, list):
                dicts = [v for v in value if isinstance(v, dict)]
                if dicts and len(dicts) * 2 >= len(value):
                    lists.append(dicts)
                continue
            if not isinstance(value, dict):
                continue
            if single is None and depth <= 3 and self.id_in(value) and self.carries_facts(value):
                single = value
                continue
            stack += [(v, depth + 1) for v in value.values() if isinstance(v, (dict, list))]
        lists.sort(key=len, reverse=True)
        return single, (lists[0] if lists else None)

    def remember(self, name: str, observation: Any) -> None:
        """Keep a compact record of every opened listing (one carrying a facts-like table) and of every result row; count
        the searches (results whose rows mostly carry no facts)."""
        if not isinstance(observation, (dict, list)):
            return
        observation = normal(observation)
        many: list[dict[str, Any]] = []
        layer = [observation]
        for _ in range(3):                    # sheets side by side: a list of them, or a map keyed by id
            nxt = []
            for v in layer:
                for x in (v.values() if isinstance(v, dict) else v if isinstance(v, list) else []):
                    if isinstance(x, dict) and self.id_in(x) and self.carries_facts(x):
                        many.append(x)
                    elif isinstance(x, (dict, list)):
                        nxt.append(x)
            layer = nxt
        if many:
            self.fact_tools.add(name)
        if len(many) > 1:                     # a batch or side-by-side reading: every sheet it carries is kept
            for record in many:
                self.remember(name, record)
            return
        single, rows = self.records_in(observation)
        if rows is not None and name != self.roles.get("cart_view"):
            self.learn_id_field(rows)
            self.learn_fields(rows)
            if single is None and self.id_field:
                single = self.records_in(observation)[0]
        if single is None and isinstance(observation, dict) and self.last_target:
            # a reading tool's reply: the key holding the id the call asked for is the id field from now on
            for record in [observation] + [v for v in observation.values() if isinstance(v, dict)]:
                key = next((k for k, v in record.items() if v == self.last_target), None)
                if key and self.carries_facts(record):
                    self.id_field = self.id_field or key
                    single = record
                    break
        if single is not None:
            ident = self.id_in(single)
            if ident:
                self.keep_sheet(ident, single)
                self.learn_fields([single])
        if rows is None or name == self.roles.get("cart_view"):
            return
        detailed = 0
        for row in rows:
            key = self.id_in(row)
            if not key:
                continue
            self.rows[key] = {k: v for k, v in row.items() if (isinstance(v, (str, int, float, bool)) and not (isinstance(v, str) and len(v) > 160))
                              or (isinstance(v, dict) and money_value(v) is not None)}
            long_text = " \n".join(v[:3000] for k, v in row.items() if isinstance(v, str) and len(v) > 160 and not name_key(k))
            if key not in self.text_sizes:
                self.text_sizes[key] = sum(len(v) for v in row.values() if isinstance(v, str))
            if long_text:
                self.row_texts[key] = long_text
            if self.carries_facts(row):
                self.keep_sheet(key, row)
                detailed += 1
        if detailed * 2 < len(rows):
            self.searches_done += 1

    def sheets_stating(self, need: str) -> tuple[list[str], list[str]]:
        """Opened listings whose own sheet answers a told need 'yes', and those whose sheet answers it 'no'."""
        prop, _, value = need.partition(":")
        prop, value = prop.strip().lower(), value.strip().lower()
        if not prop or not value or bound_of(value) is not None:
            return [], []
        kind = set(label_words(self.plan.get("product") or ""))
        verdicts = {ident: read_sheet(self.flat_sheet(rec), prop, value, kind)[0] for ident, rec in self.sheets.items()}
        return [i for i, v in verdicts.items() if v == "yes"], [i for i, v in verdicts.items() if v == "no"]

    def hints(self) -> list[tuple[str, str]]:
        """The story's hints from the setup, (property, likely value): properties the request points at without stating a
        value. They choose the questions and break ties among fits; they never reject an item (on the story pack pass 6b
        lost five tasks by hunting for story details as if they were requirements)."""
        settled = [prop for prop, _ in self.demands()] + list(self.free)
        out: list[tuple[str, str]] = []
        for item in self.plan.get("hints") or []:
            prop = str(item.get("property") if isinstance(item, dict) else item or "").strip().lower()
            likely = str(item.get("likely") or "").strip().lower() if isinstance(item, dict) else ""
            words = set(label_words(prop)) - {"type"}
            if not words or any(words <= set(label_words(s)) or (set(label_words(s)) - {"type"} and set(label_words(s)) - {"type"} <= words) for s in settled):
                continue
            if prop not in [p for p, _ in out]:
                out.append((prop, likely))
        return out[:4]

    def preferred(self, fits: list[tuple[str, float | None]]) -> str:
        """Among the proven fits, the cheapest whose own facts state a hint's likely value, else the cheapest."""
        if self.plan.get("priority") in ("cheapest", "best"):
            return fits[0][0]                                        # a cheapest or best rule outranks any hint
        product = {stem(w) for w in word_list(self.plan.get("product") or "", False)}
        for ident, _ in fits:
            if any(v and read_sheet(self.flat_sheet(self.sheets.get(ident) or {}), p, v, product)[0] == "yes" for p, v in self.hints()):
                return ident
        return fits[0][0]

    REMOVES = r"remov|delet|drop|discard|releas|clear|minus|subtract|take out|decreas|cancel"

    def basket_spec(self, name: str | None) -> tuple[str, str, str | None] | None:
        """A cart tool that sets the whole basket in one call: (the list slot, the listing slot and the quantity slot of its
        line objects). None for a tool that takes one listing per call."""
        props = (self.schemas.get(name or "") or {}).get("properties") or {}
        for key, spec in props.items():
            inner = (spec.get("items") or {}).get("properties") if isinstance(spec, dict) and spec.get("type") == "array" and isinstance(spec.get("items"), dict) else None
            if not isinstance(inner, dict):
                continue
            texts = [k for k, v in inner.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v]
            ident = next((k for k in texts if re.search(r"(?i)id|item|sku|listing|product|code|ref", k)), texts[0] if texts else None)
            qty = next((k for k, v in inner.items() if isinstance(v, dict) and v.get("type") in ("integer", "number")), None)
            if ident:
                return key, ident, qty
        return None

    def line_quantity(self, ident: str) -> int:
        """Units wanted of one line: its product's own quantity in a request for several products, else the request's."""
        if self.bundle_names:
            name = self.product_of(ident)
            for item in self.plan.get("items") or []:
                if isinstance(item, dict) and name and lowered(str(item.get("product") or item.get("name") or "")) == lowered(name):
                    q = item.get("quantity")
                    if isinstance(q, (int, float)) and not isinstance(q, bool) and q >= 1:
                        return int(q)
        return self.quantity_wanted()

    def basket_args(self, lines: list[str]) -> dict[str, Any]:
        """The whole-basket call that leaves exactly these lines in the cart, each at its wanted quantity."""
        cart = self.roles.get("cart")
        spec = self.basket_spec(cart)
        if not spec:
            return {}
        key, id_key, qty_key = spec
        args: dict[str, Any] = {}
        for other, rule in ((self.schemas.get(cart or "") or {}).get("properties") or {}).items():
            if other != key and isinstance(rule, dict) and rule.get("enum"):
                modes = self.cart_modes(cart)
                args[other] = modes[other][0] if other in modes else rule["enum"][0]
        args[key] = [dict({id_key: i}, **({qty_key: self.cart_qty.get(i) or self.line_quantity(i)} if qty_key else {})) for i in dict.fromkeys(lines)]
        return args

    def cart_args(self, ident: str) -> dict[str, Any]:
        """The cart tool's arguments that put one listing in at the asked quantity: an enum mode that is not a removal, the
        quantity in any number slot, the listing in the first free text slot (a whole-basket tool: the lines it keeps plus
        this one)."""
        if self.basket_spec(self.roles.get("cart")):
            keep = [i for i in self.cart_ids if i != ident and i not in self.gone] if (self.bundle_names or self.companions) else []
            if self.bundle_names:
                keep = [i for i in keep if not self.product_of(ident) or self.product_of(i) != self.product_of(ident)]
            self.cart_qty.pop(ident, None)
            return self.basket_args(keep + [ident])
        properties = (self.schemas.get(self.roles.get("cart") or "") or {}).get("properties") or {}
        wanted = self.line_quantity(ident)
        args: dict[str, Any] = {}
        for key, rule in properties.items():
            if not isinstance(rule, dict):
                continue
            kinds = rule.get("type") if isinstance(rule.get("type"), list) else [rule.get("type")]
            if rule.get("enum"):
                modes = self.cart_modes(self.roles.get("cart"))
                args[key] = modes[key][0] if key in modes else rule["enum"][0]
            elif "integer" in kinds or "number" in kinds:
                args[key] = max(wanted, int(rule.get("minimum", 1) or 1))
            elif "string" in kinds and ident not in args.values():
                args[key] = ident
            elif "array" in kinds and isinstance(rule.get("items"), dict) and rule["items"].get("type") == "string" and not any(i == ident or i == [ident] for i in args.values()):
                counted = any(isinstance(r, dict) and r.get("type") in ("integer", "number") for r in properties.values())
                args[key] = [ident] * (1 if counted else max(1, wanted))   # a list of ids with no count slot: one entry per unit
        return args

    def quantity_wanted(self) -> int:
        q = self.plan.get("quantity")
        return int(q) if isinstance(q, (int, float)) and not isinstance(q, bool) and q >= 1 else 1

    def in_force(self) -> list[str]:
        return list(self.current) if self.current is not None else [r for r in self.stated + self.revealed if r not in self.dropped]

    def sheet_settles(self, item: str) -> bool:
        """An opened listing whose own sheet reads 'yes' on every condition in force, each a word value stated by every label
        for that property (no figures, bounds or exclusions), affordable, in stock and in the main category."""
        demands = self.demands()
        if item not in self.sheets or not demands or len(demands) != len([r for r in self.in_force() if str(r).strip()]):
            return False
        if item not in {i for i, _ in self.fits_by_facts()}:
            return False
        flat, kind = self.flat_sheet(self.sheets[item]), set(label_words(self.plan.get("product") or ""))
        for prop, value in demands:
            if bound_of(value) is not None or re.search(r"\d", value) or re.match(r"^(?:not|no|without|anything|except|excluding|avoid)\b", value):
                return False      # figures, bounds and exclusions stay with the judge
            if not labels_agree(flat, prop, value, kind):
                return False
        return True

    def demands(self) -> list[tuple[str, str]]:
        """The requirements in force as (property, value) pairs for the sheet reader. A short bracket that is another spelling
        of the value ('none (sleeveless)') stays as an alternative; a bracket that is commentary is dropped."""
        items = self.in_force()
        out = []
        for item in items:
            prop, sep, value = str(item).partition(":")
            prop, value = prop.strip().lower(), value.strip().lower()
            value = re.sub(r"\s*\(([^)]*)\)", lambda m: (" or " + m.group(1).strip()) if 0 < len(m.group(1).split()) <= 3
                           and not re.search(r"\bi\b|\bi'|\bme\b|\bmy\b|check|^\s*(?:no|not|without)\b", m.group(1)) else "", value).strip(" .")
            value = re.sub(r"^(?:from|by|made (?:in|of|from|by))\s+(?=\S)", "", value)   # a leading linking word is no part of the value
            if sep and prop and value and not re.search(NOT_A_PROPERTY, prop):
                out.append((prop, value))
        return out

    def affordable(self, record: dict[str, Any]) -> bool:
        """Within the limit in force at the asked quantity, and not last shown out of stock."""
        price, cap = _price(record), self.budget_now
        if price is None and isinstance(cap, (int, float)) and not isinstance(cap, bool):
            ident = self.id_in(record)
            price = self.latest_price(ident) if ident else None     # a sheet without a price: the newest price shown elsewhere
        over = isinstance(cap, (int, float)) and not isinstance(cap, bool) and price is not None and price * self.quantity_wanted() > cap + 1e-9
        return not over and _in_stock(record) is not False

    # ---------- fitment codes (a model code in the request is matched against listing pages, never searched) ----------
    def wanted_codes(self) -> list[str]:
        """The model codes the request and the shopper named (LUK1720, SIR2120): letters, then at least three digits."""
        text = " ".join([str(self.policy.get("query") or "")] + [str(r) for r in self.stated + self.revealed + (self.current or [])])
        out: list[str] = []
        for token in CODE_TOKEN.findall(text):
            if len(re.findall(r"\d", token)) >= 3 and token.upper() not in out:
                out.append(token.upper())
        if out or self.bundle_names:
            sources = list(self.cart_ids)
            if self.bundle_names and not self.code_probe and not any(self.product_of(i) == self.bundle_names[0] for i in sources):
                self.code_probe = True              # no main line held yet: the main product's best verified listing gives the code
                try:
                    sources += [i for i, _ in self.product_fits(self.bundle_names[0])][:1]
                finally:
                    self.code_probe = False
            for ident in sources:                   # a held line's own model number is the code its companions must fit
                text = re.sub(r"\b([A-Z]{2,5})-(\d{3,8})\b", r"\1\2", (self.pages.get(ident) or "") + " \n" + (self.row_texts.get(ident) or "") + " \n" + " \n".join(
                    v for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v) >= 12 and not re.search(r"(?i)title|name|heading", str(k))))
                for sentence in re.split(r"(?<=[.;!?])\s+|\n", text):
                    tokens = CODE_TOKEN.findall(sentence)
                    labelled = re.match(r"\s*[A-Za-z][A-Za-z ]{1,24}:\s*\S+\s*[.;]?\s*$", sentence) and not re.search(r"(?i)compatib|\bfits?\b|\bwith\b|\bpart\b|\balso\b", sentence)
                    if (len(tokens) == 1 and (":" not in sentence or labelled) and (out or not re.search(r"(?i)\bfits?\b|\bwith\b", sentence)) and len(sentence.strip()) <= 40 and "x" not in tokens[0][len(code_family(tokens[0])):].lower()
                            and len(re.findall(r"\d", tokens[0])) >= 3 and tokens[0].upper() not in out):
                        out.append(tokens[0].upper())
        return out[:8]

    def note_page(self, name: str, observation: Any) -> None:
        """Keep the free text of a listing's page (description, fine print, remarks): the fitment lines live there."""
        if not isinstance(observation, dict) or name not in (self.roles.get("view"), self.roles.get("details")):
            return
        for table in self.dicts_in(observation, 3):
            ident = self.id_in(table)
            if not ident:
                continue
            texts = [v for k, v in table.items() if isinstance(v, str) and len(v) >= 12 and not name_key(k)]
            if _in_stock(table) is not None:
                self.page_stock[ident] = bool(_in_stock(table))
            if texts or any(re.search(r"(?i)descr", str(k)) for k in table):
                self.pages[ident] = " \n".join(texts)
                self.text_tools[name] = self.text_tools.get(name, 0) + (1 if texts else 0)

    def page_text(self, ident: str) -> str:
        row = self.rows.get(ident) or {}
        extra = " \n".join(v for k, v in row.items() if isinstance(v, str) and len(v) >= 12 and not name_key(k))
        text = (self.pages.get(ident) or "") + " \n" + extra + " \n" + (self.row_texts.get(ident) or "")
        # a model code written with a hyphen ("BEK-4110") or a wildcard code after a space ("every BEX 35xx model") is one code
        return re.sub(r"\b([A-Z]{2,5})(?:-(\d{3,8}|\d+[xX]+)\b|\s(\d+[xX]+)\b)", lambda m: m.group(1) + (m.group(2) or m.group(3)), text)

    def title_codes(self, ident: str, families: set[str]) -> list[str]:
        record = dict(self.rows.get(ident) or {})
        record.update({k: v for k, v in (self.sheets.get(ident) or {}).items() if isinstance(v, str)})
        title = " ".join(str(v) for k, v in record.items() if isinstance(v, str) and name_key(k))
        return [t.upper() for t in CODE_TOKEN.findall(title) if code_family(t) not in families]

    def fit_statements(self) -> dict[str, list[tuple[list[str], list[str]]]]:
        """Every fitment line read, keyed by the listing it speaks for: (codes it fits, codes it excludes). A line that
        names another listing's part code ("Part QU94639 also fits: LUK1920") speaks for that listing."""
        wanted = self.wanted_codes()
        families = {code_family(w) for w in wanted}
        key = (tuple(wanted), len(self.pages), len(self.rows), len(self.row_texts), sum(len(v) for v in self.pages.values()))
        if self.fit_cache[0] == key:
            return self.fit_cache[1]
        out: dict[str, list[tuple[list[str], list[str]]]] = {}
        if families:
            parts: dict[str, str] = {}
            for ident in set(self.rows) | set(self.sheets) | set(self.pages):
                for code in self.title_codes(ident, families):
                    parts.setdefault(code, ident)
            for ident in set(self.pages) | set(self.rows):
                own = set(self.title_codes(ident, families))
                for sentence in re.split(r"(?<=[.;!?])\s+|\n", self.page_text(ident)):
                    tokens = CODE_TOKEN.findall(sentence)
                    family = [t for t in tokens if code_family(t) in families]
                    if not family:
                        continue
                    cut = re.search(r"(?i)\b(?:except|excluding|not)\b", sentence)
                    include = [t for t in family if not cut or sentence.find(t) < cut.start()]
                    exclude = [t for t in family if cut and sentence.find(t) > cut.start()]
                    first = min(sentence.find(t) for t in family)
                    foreign = [t.upper() for t in tokens if code_family(t) not in families and t.upper() not in own and len(re.findall(r"\d", t)) >= 3 and sentence.find(t) < first]
                    target = (parts.get(foreign[0]) or "part:" + foreign[0]) if foreign else ident
                    out.setdefault(target, []).append((include, exclude))
            for ident in set(self.rows) | set(self.sheets) | set(self.pages):
                for code in self.title_codes(ident, families):
                    if "part:" + code in out:
                        out.setdefault(ident, []).extend(out["part:" + code])
        self.fit_cache = (key, out)
        return out

    def code_fit(self, ident: str) -> bool | None:
        """True when a fitment line for the listing covers a requested code (and does not exclude it), False when its
        fitment lines name only other codes, None when no fitment line for it was read."""
        wanted = self.wanted_codes()
        lines = self.fit_statements().get(ident) or []
        if not wanted or not lines:
            return None
        return any(any(code_covers(t, w) for t in include) and not any(code_covers(t, w) for t in exclude) for include, exclude in lines for w in wanted)

    def code_group(self, ident: str) -> str:
        return self.product_of(ident) if self.bundle_names else ""

    def codes_bind(self, group: str) -> bool:
        """Whether the listings of a product carry fitment lines at all (then only a listing whose line matches can fit)."""
        statements = self.fit_statements()
        return any(not i.startswith("part:") and self.code_group(i) == group for i in statements)

    def code_ok(self, ident: str) -> bool:
        if self.series_verdict(ident) is False:
            return False
        if not self.wanted_codes():
            return True
        verdict = self.code_fit(ident)
        if verdict is not None:
            return verdict
        return not self.codes_bind(self.code_group(ident))

    def page_tool(self) -> str | None:
        view, details = self.roles.get("view"), self.roles.get("details")
        if view and details and self.text_tools.get(details, 0) > self.text_tools.get(view, 0):
            return details
        return view or details

    def fitment_unread(self) -> list[str]:
        """Candidates whose page was never opened while codes are in play: cart lines first, then sheets that no
        requirement rules out, then shown rows of a product with no match yet; cheapest first within each."""
        if not self.wanted_codes():
            return []
        kind = set(label_words(self.plan.get("product") or ""))
        def open_to(ident: str, record: dict[str, Any]) -> bool:
            described = bool(self.row_texts.get(ident)) or len(str(next((v for k, v in (self.rows.get(ident) or {}).items() if re.search(r"(?i)descr", str(k))), ""))) >= 12
            return ident not in self.pages and not described and ident not in self.gone and self.affordable(record) and self.code_fit(ident) is None
        first = [i for i in self.cart_ids if open_to(i, self.sheets.get(i) or self.rows.get(i) or {})]
        sheets = []
        for ident, sheet in self.sheets.items():
            if ident in first or not open_to(ident, sheet):
                continue
            demands = self.product_demands(self.code_group(ident)) if self.bundle_names and self.code_group(ident) else self.demands()
            if "no" in [read_sheet(self.flat_sheet(sheet), prop, value, kind)[0] for prop, value in demands]:
                continue
            sheets.append(((self.latest_price(ident) or 0.0), ident))
        matched = {self.code_group(i) for i in set(self.sheets) | set(self.rows) if self.code_fit(i)}
        rows = [((_price(r) or 0.0), i) for i, r in self.rows.items() if i not in self.sheets and i not in first and open_to(i, r)
                and (not self.bundle_names or (self.code_group(i) and self.code_group(i) not in matched))]
        return first + [i for _, i in sorted(sheets)] + [i for _, i in sorted(rows)][:4]

    def read_fitment(self) -> bool:
        """The agent opens the pages of up to three candidates per turn while a code is in play (the fitment lines are on
        the page, not on the sheet). Returns whether it spent steps."""
        tool = self.page_tool()
        if not tool or self.finished or self.dead or not self.wanted_codes():
            return False
        if (self.steps_left() <= self.finish_reserve() + 6 or self.time_left() < WRAP_UP_SECONDS + 40 or self.spend_level() == 2
                or self.page_reads >= 6 + self.max_steps // 5):
            return False
        known = any(not i.startswith("part:") for i in self.fit_statements())
        if (self.max_steps < LONG_WORLD or self.page_reads >= 4) and not known:
            return False
        todo = self.fitment_unread()[:3]
        if not todo:
            return False
        calls = self.open_calls(todo, tool)
        if not calls:
            return False
        self.page_reads += len(calls)
        self.own_rounds(calls, "Reading the pages of " + ", ".join(todo) + " for their fitment lines (the requested code is matched there).")
        return True

    def fitment_line(self) -> str:
        wanted = self.wanted_codes()
        if not wanted or not (self.code_misses or any(not i.startswith("part:") for i in self.fit_statements())):
            return ""
        ids = sorted(set(self.sheets) | set(self.pages))
        match = [i for i in ids if self.code_fit(i)]
        other = [i for i in ids if self.code_fit(i) is False]
        unread = self.fitment_unread()[:6]
        return ("\n[fitment] requested code(s): " + ", ".join(wanted) + ". Never search a code; search the product's words and read each candidate's page (description), where lines such as a compatibility list, a model number or a wildcard family name the codes it fits."
                + (" Pages that cover the code: " + ", ".join(match[:8]) + "." if match else "")
                + (" Pages that name only other codes (they fit other models; take one only for a product the code does not bind): " + ", ".join(other[:8]) + "." if other else "")
                + (" Not read yet: " + ", ".join(unread) + "." if unread else ""))

    def main_group(self) -> str:
        """The store category the seen listings point to: the label sharing most words with the requested kind, then the
        one seen most often."""
        kind = set(label_words(self.plan.get("product") or ""))
        counts: dict[str, int] = {}
        for record in list(self.rows.values()) + list(self.sheets.values()):
            if _shelf(record):
                counts[_shelf(record)] = counts.get(_shelf(record), 0) + 1
        if not counts:
            return ""
        return max(counts, key=lambda g: (len(kind & set(label_words(g))), counts[g]))

    def fits_by_facts(self, strict: bool = True) -> list[tuple[str, float | None]]:
        """Opened listings judged on their own sheets: strict, every requirement read 'yes'; loose, none read 'no' (most
        'yes' first). Affordable, in stock and in the main category; cheapest first."""
        demands = self.demands()
        if strict and not demands and not self.dropped:
            return []                               # (every requirement dropped by the shopper: any buyable listing of the kind fits)
        kind, group = set(label_words(self.plan.get("product") or "")), self.main_group()
        found = []
        for ident, sheet in self.sheets.items():
            own = _shelf(sheet) or _shelf(self.rows.get(ident) or {})
            if (group and own and own != group) or not self.affordable(sheet):
                continue
            verdicts = [read_sheet(self.flat_sheet(sheet), prop, value, kind)[0] for prop, value in demands]
            if (strict and any(v != "yes" for v in verdicts)) or (not strict and "no" in verdicts):
                continue
            found.append((ident, self.latest_price(ident), verdicts.count("yes")))
        found.sort(key=lambda t: ((-t[2]) if not strict else 0, t[1] is None, t[1] or 0.0))
        return self.code_first([(i, p) for i, p, _ in found])

    def series_tag(self, ident: str) -> str:
        """The part series a listing's title carries in brackets ("[TA6-43208]" -> "TA6"), "" when none."""
        titles = " ".join(v for record in (self.rows.get(ident) or {}, self.sheets.get(ident) or {}) for k, v in record.items() if isinstance(v, str) and name_key(k))
        found = re.search(r"\[([A-Za-z]{1,4}\d{1,3})-\d+\]", titles)
        return found.group(1).upper() if found else ""

    def series_fitted(self, ident: str) -> list[str]:
        """The part series a listing's own text says fit it ("Fits part series TA4, TA6.")."""
        out: list[str] = []
        for clause in re.findall(r"(?i)\bseries\b([^.;\n]*)", self.page_text(ident)):
            out += [t.upper() for t in re.findall(r"\b[A-Za-z]{1,4}\d{1,3}\b", clause)]
        return list(dict.fromkeys(out))

    def series_verdict(self, ident: str) -> bool | None:
        """In a bundle, whether a listing and the held lines agree on part series: a part whose bracketed series a held
        line lists as fitting it (or a line listing the held part's series). None when no series is stated on either side."""
        if not self.bundle_names or not self.cart_ids:
            return None
        verdicts = []
        own, fitted = self.series_tag(ident), self.series_fitted(ident)
        for held in self.cart_ids:
            if held == ident or self.code_group(held) == self.code_group(ident):
                continue
            held_fitted, held_tag = self.series_fitted(held), self.series_tag(held)
            if own and held_fitted:
                verdicts.append(own in held_fitted)
            elif fitted and held_tag:
                verdicts.append(held_tag in fitted)
        return all(verdicts) if verdicts else None

    def code_first(self, fits: list[Any]) -> list[Any]:
        """With a code in play, only the fits whose page covers it, as long as there is one; else every fit whose page does
        not name only other codes, else all (a product the code does not bind keeps its fits)."""
        if fits and self.bundle_names and self.cart_ids:
            fits = [f for f in fits if self.series_verdict(f[0]) is not False] or fits
        if not fits or not self.wanted_codes():
            return fits
        return [f for f in fits if self.code_ok(f[0])] or [f for f in fits if self.code_fit(f[0]) is not False] or fits

    def latest_price(self, item: str) -> float | None:
        """The newest price any record shows for the listing (a cart view after a price move counts), else its row's."""
        def look(value: Any, depth: int = 0) -> float | None:
            if depth > 6:
                return None
            if isinstance(value, dict):
                if item in value.values() and _price(value) is not None:
                    return _price(value)
                values = value.values()
            elif isinstance(value, list):
                values = value
            else:
                return None
            for inner in values:
                hit = look(inner, depth + 1)
                if hit is not None:
                    return hit
            return None
        for record in reversed(self.records):
            hit = look(record)
            if hit is not None:
                return hit
        return _price(self.sheets.get(item) or self.rows.get(item) or {})

    def rule_check(self, item: str, adding: bool) -> str:
        """The store's rule on a listing about to be carted or ordered (the v3 line reads the pick's facts first, and a
        fact named for a property that states another value loses whatever the title says)."""
        sheet = self.sheets.get(item)
        if sheet is None:
            if adding and self.sheets and self.roles.get("details") and self.steps_left() > 8:
                return f"The details of {item} have not been opened: the store grades a listing by its own facts, which only the details show. Open them first."
            return ""
        own = self.product_of(item) if self.bundle_names else ""
        product = self.name_words(own) if own else {stem(w) for w in word_list(self.plan.get("product") or "", False)}
        fails = []
        for prop, value in self.own_demands(item):
            verdict, shown = read_sheet(self.flat_sheet(sheet), prop, value, product)
            if verdict == "no":
                fails.append(f"{prop}: {value} (its own fact says {shown})")
        if not fails:
            return ""
        return f"By the store's own facts {item} does not meet: " + "; ".join(fails[:4]) + ". The store reads a requirement from the listing's own fact for that property, not from the title or a look-alike word."

    def cheaper_note(self, item: str) -> str:
        """Cheapest tasks: opened cheaper listings that fit by their sheets, and cheaper unread rows in the same category."""
        if self.plan.get("priority") != "cheapest":
            return ""
        price = self.latest_price(item)
        if price is None:
            return ""
        notes = []
        fits = [f"{k} ({p:g})" for k, p in self.fits_by_facts() if k != item and p is not None and p < price and k not in self.gone and not self.shunned(k)]
        if fits and self.steps_left() > 4:
            notes.append(f"these opened listings state every requirement on their own sheets and cost less than {item} ({price:g}): " + ", ".join(fits[:4]) + "; order the cheapest of them instead (swap the cart) unless its sheet fails something the shopper said. " + ONE_FIELD)
        group = _shelf(self.sheets.get(item) or self.rows.get(item) or {})
        unread = sorted((_price(r), k) for k, r in self.rows.items() if k not in self.sheets and _price(r) is not None and _price(r) < price
                        and self.affordable(r) and (not group or (_shelf(r) or group) == group))
        if unread and self.steps_left() > max(9, self.commit_steps()):
            notes.append("these listings within the limit cost less and were never read: " + ", ".join(f"{k} ({p:g})" for p, k in unread[:5]) + "; read them cheapest first before ordering")
        if not notes:
            return ""
        return "The request asks for the cheapest fitting listing: " + "; ".join(notes) + "."

    def search_shape(self) -> dict[str, Any] | None:
        """The search tool's slots for a category browse below a price: category, ceiling, page and words, plus the
        enum settings (taken from an earlier search that set both category and ceiling, else one named for filtering;
        a lowest-price-first order when the store offers one)."""
        tool = self.roles.get("search")
        props = (self.schemas.get(tool or "") or {}).get("properties") or {}
        if not props:
            return None
        def types(spec: dict[str, Any]) -> set[Any]:
            return set(spec.get("type") if isinstance(spec.get("type"), list) else [spec.get("type")])
        def slot(sign: str, wanted: set[str], never: str = "$^") -> str | None:
            for key, spec in props.items():
                name = key.lower().replace("_", " ")
                if isinstance(spec, dict) and "enum" not in spec and re.search(sign, name) and not re.search(never, name) and types(spec) & wanted:
                    return key
            return None
        shape: dict[str, Any] = {"tool": tool, "group": slot(GROUP_LIKE + "|dept", {"string"}),
                                 "ceiling": slot(r"max|cap|ceiling|budget|limit|under|upper|spend|price", {"number", "integer"}, r"min|low|from|floor|least"),
                                 "page": slot(r"page|cursor|offset|continue|next", {"integer", "number"}),
                                 "words": slot(r"text|query|keyword|term|words|search|\bq\b|phrase|string", {"string"}), "modes": {}}
        if not shape["ceiling"]:
            return None
        enums = {k: list(v["enum"]) for k, v in props.items() if isinstance(v, dict) and v.get("enum")}
        for entry in reversed(self.trace):
            for action in (entry.get("actions") or []) if entry.get("kind") == "env" else []:
                args = action.get("args") or {}
                if action.get("name") == tool and all(args.get(shape[k]) is not None for k in ("group", "ceiling") if shape[k]):
                    shape["modes"] = {k: args[k] for k in enums if k in args}
                    break
            if shape["modes"]:
                break
        for key, values in enums.items():
            low_first = next((v for v in values if re.search(r"(?i)price", str(v)) and re.search(r"(?i)asc|low|cheap|increas", str(v))), None)
            if low_first is not None:
                shape["modes"][key] = low_first
                shape["sorted"] = True
            elif key not in shape["modes"]:
                chosen = next((v for v in values if re.search(r"(?i)field|filter|facet|attrib|structur|browse|categor|exact", str(v))), None)
                if chosen is not None or key in ((self.schemas.get(tool or "") or {}).get("required") or []):
                    shape["modes"][key] = chosen if chosen is not None else values[0]
        return shape

    def match_count(self, observation: Any) -> tuple[int | None, list[str]]:
        """The total a search reports and the ids of the rows it shows."""
        for table in Episode.dicts_in(observation):
            rows = next((v for v in table.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), None)
            total = next((v for k, v in table.items() if isinstance(v, int) and not isinstance(v, bool) and re.search(r"(?i)match|total|count|hits|found|results", str(k))), None)
            if rows is None and total is None:
                continue
            return total, [i for i in (self.id_in(r) for r in rows or []) if i]
        return None, []

    def open_calls(self, idents: list[str], tool: str | None = None) -> list[dict[str, Any]]:
        """Calls that open the given listings' sheets: one call for several when the reading tool takes a list."""
        reader = self.roles.get("details") or self.roles.get("view")
        if not tool and reader not in self.fact_tools and self.fact_tools:
            reader = sorted(self.fact_tools)[0]   # the tool whose results did carry the sheets reads them
        tool = tool or reader
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        many = next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array"), None)
        if many:
            size = max(1, int(props[many].get("maxItems") or len(idents) or 1))
            batches = [{many: idents[i:i + size]} for i in range(0, len(idents), size)]
        else:
            key = next((k for k in schema.get("required") or [] if (props.get(k) or {}).get("type") == "string"), None)
            key = key or next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v), None)
            batches = [{key: i} for i in idents] if key else []
        out = []
        for n, args in enumerate(batches):
            shaped, problems = shape_args(schema, args, "args")
            if not problems:
                out.append({"id": f"walk_open_{self.step}_{n}", "type": "function", "function": {"name": tool, "arguments": json.dumps(shaped)}})
        return out

    def sweep_pool(self) -> list[str]:
        """Unread rows worth a sheet: affordable and in stock as shown, in the main category, not gone; those whose own row
        words state the most required values first, then the ones naming the product's words, then the cheaper."""
        group, wanted = self.main_group(), [v for _, v in self.demands() if bound_of(v) is None and not re.match(r"^(?:not|no|without)\b", v)]
        kind = [w for w in label_words(self.plan.get("product") or "") if w not in SMALL_WORDS]
        def rank(ident: str) -> tuple[int, int, float]:
            row = self.rows[ident]
            words = " ".join(str(v) for v in row.values() if isinstance(v, str)) + " " + self.row_texts.get(ident, "")
            price = _price(row)
            return (-sum(1 for v in wanted if states_value(v, words)), -sum(1 for w in kind if states_value(w, words)), price if price is not None else float("inf"))
        pool = [i for i, r in self.rows.items() if i not in self.sheets and i not in self.gone and self.affordable(r)
                and not (group and _shelf(r) and _shelf(r) != group)]
        return sorted(pool, key=rank)

    def sweep_reads(self) -> None:
        """After a search that showed unread candidates, the agent itself opens the sheets of the most promising ones (a few
        per search, fewer as the steps left shrink or the spend nears its soft line), so the choice is made over a wider set
        of read sheets than the model would open by itself. Never past the commit mark, never into the finish reserve, and
        never once an order was tried; a reading tool that takes a list reads several sheets in one step."""
        if self.searches_done <= self.sweep_seen or self.finished or self.dead:
            return
        self.sweep_seen = self.searches_done
        if (self.order_attempts or self.cart_ids or self.bundle_names or self.companions or self.spend_level() >= 1
                or self.time_left() < WRAP_UP_SECONDS + 60 or self.step >= self.pace_targets()[1] or self.max_steps < 2 * PAGE_FROM_STEPS):
            return
        soft = self.spend_caps()[0]
        headroom = 1.0 - min(1.0, self.spent() / soft) if soft > 0 else 0.0
        room = min(self.steps_left() - self.finish_reserve(empty_cart=True) - SWEEP_ROOM, round(self.max_steps * SWEEP_SHARE) - self.swept)
        known = len(self.live_fits())
        per = SWEEP_PER if headroom > 0.5 else SWEEP_PER // 2
        per = per if known < 2 or self.plan.get("priority") == "cheapest" else max(1, per // 2)
        pool = self.sweep_pool()
        tool = self.roles.get("details") or self.roles.get("view")
        props = ((self.schemas.get(tool or "") or {}).get("properties") or {})
        many = any(isinstance(v, dict) and v.get("type") == "array" for v in props.values())
        count = min(per * (2 if many else 1), len(pool)) if many else min(per, room, len(pool))
        if room <= 0 or count <= 0:
            return
        calls = self.open_calls(pool[:count])[:max(0, room)]
        if not calls:
            return
        self.trace.append({"kind": "sweep", "ids": pool[:count], "steps": len(calls), "spent": round(self.spent(), 5)})
        before = self.walk_spent
        self.own_rounds(calls, "Opening the sheets of the most promising unread listings from the search, so the choice is made over every likely fit: " + ", ".join(pool[:count]) + ".")
        self.swept += self.walk_spent - before
        self.walk_spent = before                    # the sweep has its own allowance; the cheapest descent keeps its steps

    def own_rounds(self, calls: list[dict[str, Any]], said: str) -> None:
        """The agent's own reading calls, one envelope each."""
        for call in calls:
            if self.finished or self.dead or self.steps_left() <= self.loop_floor():
                break
            self.own_round([call], said)

    def own_round(self, calls: list[dict[str, Any]], said: str) -> None:
        """Calls the agent makes on its own, kept in the conversation like a model turn."""
        self.messages.append({"role": "assistant", "content": said, "tool_calls": calls})
        replies = self.run_calls(calls, planned=True)
        for index, (call_id, text) in enumerate(replies):
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        self.walk_spent += len(calls)

    def committed_to(self, calls: list[dict[str, Any]]) -> str:
        """The listing a reply commits to: the one it adds to the cart, or the cart's line when it orders."""
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name")
            args = _parse_json_object(function.get("arguments") or "{}") or {}
            if name == self.roles.get("cart"):
                properties = (self.schemas.get(name) or {}).get("properties") or {}
                if self.removing_call(name, args):
                    continue
                spec = self.basket_spec(name)
                if spec and isinstance(args.get(spec[0]), list):
                    lines = [l.get(spec[1]) for l in args[spec[0]] if isinstance(l, dict) and isinstance(l.get(spec[1]), str)]
                    fresh = [i for i in lines if i not in self.cart_ids]
                    if fresh or lines:
                        return (fresh or lines)[-1]
                    continue
                target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), "")
                if target:
                    return target
            if name in (self.roles.get("order"), self.roles.get("decline")) and self.intent(name, args) == "order" and self.cart_ids:
                return self.cart_ids[-1]
        return ""

    def live_fits(self) -> list[tuple[str, float]]:
        """Listings whose own sheets state every requirement in force, priced, not gone: cheapest first."""
        return sorted(((i, p) for i, p in self.fits_by_facts() if p is not None and i not in self.gone), key=lambda t: t[1])

    def below(self, price: float, shelf: str) -> list[tuple[float, str]]:
        """Rows never opened, in the shelf, in stock and within the limit, priced below `price`: cheapest first."""
        out = []
        for ident, row in self.rows.items():
            cost = _price(row)
            if ident in self.sheets or ident in self.gone or cost is None or cost >= price - 1e-9 or not self.affordable(row):
                continue
            if shelf and _shelf(row) and _shelf(row) != shelf:
                continue
            out.append((cost, ident))
        return sorted(out)

    def walk_room(self) -> bool:
        return (self.walk_spent < WALK_STEPS and self.steps_left() > self.finish_reserve(empty_cart=True) + 4
                and self.time_left() > WRAP_UP_SECONDS + 40 and self.spend_level() < 2 and not self.finished and not self.dead)

    def page_out(self) -> None:
        """A worded search that matched a short list (about a hundred rows or fewer) but has not shown it all is paged
        to its end by the agent itself, so the cheapest rows cannot hide on a later page. Bounded by the walk's steps."""
        shape = self.search_shape() or {}
        page_slot, tool = shape.get("page"), shape.get("tool")
        if not page_slot or not tool or self.pages_turned >= 2 or not self.walk_room():
            return                                              # paging is worth a step only while the walk has steps left
        for key, cover in reversed(list(self.page_cover.items())):
            settings = key.rsplit("|", 1)[0]
            if settings in self.deaf_modes or cover["total"] > 100 or len(cover["seen"]) >= cover["total"] or not cover["seen"]:
                continue
            per_page = max(1, len(cover["seen"]) // max(1, cover["page"]))
            last_page = -(-cover["total"] // per_page) + 1
            while self.walk_room() and self.pages_turned < 2 and len(cover["seen"]) < cover["total"] and cover["page"] < last_page:
                args = dict(cover["args"], **{page_slot: cover["page"] + 1})
                args, problems = shape_args(self.schemas[tool], args, "args")
                if problems:
                    break
                before, page_before = len(cover["seen"]), cover["page"]
                self.own_round([{"id": f"walk_page_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}],
                               f"That search matched {cover['total']} listings and showed {before}: turning to the next page so no cheaper fit hides there.")
                self.pages_turned += 1
                self.recent = [r for r in self.recent if r != tool + _compact(args)]   # the model may still turn this page itself
                if cover["page"] == page_before:                # the bookkeeping after the call records the page turned; failing that, count it here
                    cover["page"] = page_before + 1
                if len(cover["seen"]) == before:
                    break
            return

    def read_cheapest(self, floor: float, shelf: str) -> bool:
        """Open the cheapest unread rows under `floor` on the shelf, one call each, within the walk's steps. False when
        there was nothing to open."""
        unread = self.below(floor, shelf)
        room = min(WALK_READS, WALK_STEPS - self.walk_spent, self.steps_left() - self.finish_reserve(empty_cart=True) - 4)
        calls = self.open_calls([i for _, i in unread[:max(0, room)]])[:max(0, room)]
        for call in calls:                                      # one call per request: every reading costs its own step anyway
            if not self.walk_room():
                break
            self.own_round([call], "Opening the next cheapest listing below that price to check it against every requirement.")
        return bool(calls)

    def descend(self, pick: str) -> bool:
        """Cheapest requests: before the cart takes `pick`, the agent itself looks under the cheapest fit known. It searches
        the pick's shelf with no words and a price ceiling just under that fit; when the list it gets is short (about two
        pages) or sorted by price, it opens the cheapest unread rows there and repeats under each cheaper fit, until a
        search under the ceiling shows nothing (no cheaper listing exists). When the list is long and unsorted, its first
        rows prove nothing, so instead the worded search the model made (the requirement words, a hundred hits or fewer)
        is paged to its end and the cheapest rows seen there are opened. At most WALK_STEPS steps in all, only while
        plenty of steps remain (the finish reserve decides, not whether a sellout happened). True when the reply
        committing to `pick` is set aside because a cheaper fit is known."""
        price = self.latest_price(pick)
        if price is None or self.plan.get("priority") != "cheapest" or not (self.roles.get("details") or self.roles.get("view")):
            return False
        if self.steps_left() < WALK_FROM_STEPS:
            return False
        shelf = _shelf(self.sheets.get(pick) or {}) or _shelf(self.rows.get(pick) or {}) or self.main_group()
        shape = self.search_shape()
        rounds = 0
        while shape and self.walk_room() and rounds < 4:
            rounds += 1
            fits = self.live_fits()
            floor = min([price] + [p for _, p in fits])
            if self.walk_floor is not None and self.walk_floor >= floor - 1e-9:
                break                                           # already shown: nothing fits below this price
            if not self.walk_wide and round(floor, 2) not in self.walk_searched:
                self.walk_searched.add(round(floor, 2))
                props = (self.schemas.get(shape["tool"]) or {}).get("properties") or {}
                args: dict[str, Any] = dict(shape["modes"])
                if shape["group"] and shelf:
                    args[shape["group"]] = shelf
                whole = "integer" in str((props.get(shape["ceiling"]) or {}).get("type"))
                ceiling = (int(floor) - (1 if floor == int(floor) else 0)) if whole else round(floor - 0.01, 2)
                args[shape["ceiling"]] = ceiling
                if shape["words"] and shape["words"] in ((self.schemas.get(shape["tool"]) or {}).get("required") or []):
                    args[shape["words"]] = " ".join(word_list(self.plan.get("product") or shelf)[-2:]) or shelf
                args, problems = shape_args(self.schemas[shape["tool"]], args, "args")
                if problems:
                    break
                self.own_round([{"id": f"walk_find_{self.step}", "type": "function", "function": {"name": shape["tool"], "arguments": json.dumps(args)}}],
                               f"The request wants the cheapest fitting item: I look for anything in stock below {floor:g}.")
                total, shown = self.match_count(self.records[-1] if self.records else None)
                if (total == 0 or not shown) and not self.below(floor, shelf):
                    self.walk_floor = floor
                    break
                if total is not None and shown and total > 2 * len(shown) and not shape.get("sorted"):
                    self.walk_wide = True                       # a long unsorted list: the worded search's pages decide instead
                    self.page_out()
            if not self.below(floor, shelf):
                total, shown = self.match_count(self.records[-1] if self.records else None)
                if not self.walk_wide and total is not None and total <= len(shown):
                    self.walk_floor = floor                     # every row under the ceiling was shown and none fits
                break
            if not self.read_cheapest(floor, shelf) or self.walk_wide:
                break                                           # pages read: a further no-text search would be as wide
        if self.walk_room() and not self.walk_wide and shape:
            self.page_out()                                     # steps left in the walk: the worded list's further pages may still hide a cheaper row
            fits = self.live_fits()
            self.read_cheapest(min([price] + [p for _, p in fits]), shelf)
        fits = self.live_fits()
        self.trace.append({"kind": "descent", "pick": pick, "price": price, "floor": self.walk_floor, "wide": self.walk_wide, "fits": fits[:4], "spent": self.walk_spent})
        best = fits[0] if fits else None
        line = ""
        held = False
        if best and best[0] != pick and best[1] < price - 1e-9 and f"{pick}>{best[0]}" not in self.walk_told:
            self.walk_told.add(f"{pick}>{best[0]}")
            line = (f"[cheapest search, done by the agent] {best[0]} ({best[1]:g}) states every requirement on its own sheet and costs less than {pick} ({price:g}). "
                    f"Put {best[0]} in the cart instead, unless its sheet fails something the shopper said (then say what). " + ONE_FIELD)
            held = True
        elif self.walk_floor is not None and best and best[0] == pick and pick not in self.walk_told:
            self.walk_told.add(pick)
            line = f"[cheapest search, done by the agent] Nothing in stock below {self.walk_floor:g} is listed on this shelf, so {pick} is the cheapest fit."
        if line:
            if self.messages and self.messages[-1].get("role") == "tool":
                self.messages[-1]["content"] += "\n" + line
            else:
                self.messages.append({"role": "user", "content": line})
        return held

    def explore_hold(self) -> str:
        """Once a fit is verified on a request that does not ask for the cheapest and the order is overdue, a reply made
        only of searching or reading is set aside without a step, pointing at the fit; one such reply in EXPLORE_VALVE
        goes through, since the model may know why the fit is not enough ("" when nothing is held)."""
        if self.plan.get("priority") == "cheapest" or self.order_attempts:
            return ""
        behind, fit = self.pace()
        if behind != "commit":
            return ""
        self.explore_holds += 1
        if self.explore_holds % EXPLORE_VALVE == 0:
            return ""
        where = "is in the cart" if fit in self.cart_ids else "is not in the cart yet"
        return (f"{fit} states every requirement on its own sheet, it is in stock, and the order should already be in (step {self.pace_targets()[1]} of {self.max_steps}); it {where}. "
                f"Put {fit} in the cart (if it is not there) and order it now; reading more is worth a step only for a listing whose check failed.")

    def early_finish_hold(self, kind: str) -> str:
        """A finishing call sent before the fit mark (FIT_BY of the cap) is held once, without a step, unless the finish is
        already grounded: an order on a line that passed an independent check under the requirements in force with the
        questions asked and no story hint open, or a decline after the audit of the opened listings found no fit and the
        questions were asked. A hidden need that surfaces only when asked, or a listing two reads away, is what an early
        finish throws away ("" when nothing is held)."""
        if self.early_held or self.order_attempts or self.refusals or self.gone or self.max_steps < 2 * PAGE_FROM_STEPS:
            return ""
        if self.step >= self.pace_targets()[0] or self.steps_left() <= 6 or self.time_left() < WRAP_UP_SECONDS + 30 or self.spend_level() == 2:
            return ""
        if not self.room_for_extras() or (kind == "decline" and self.fits_by_facts()):
            return ""                               # a decline with a fit opened goes to the gate instead
        asked = self.questions_done() and not (self.plan.get("ask_first") and not self.asked)
        if kind == "order" and asked and self.bundle_names and self.cart_ids and all(i in self.agent_placed for i in self.cart_ids):
            return ""                                   # a basket the agent filled itself, one verified line per product
        if kind == "order":
            pick = self.cart_ids[-1] if self.cart_ids else ""
            grounded = bool(pick) and self.verdicts.get(f"{pick}|{self.ledger_version}") == "" and pick in dict(self.live_fits())
            if grounded and asked and not self.hints():
                return ""
            what = ("the shopper has not been asked yet what the request leaves open (a need they reveal only when asked decides which listing fits)" if not asked
                    else "the cart line has not passed a check against every requirement in force" if not grounded else "a story hint is still open")
        elif kind == "decline":
            if asked:
                return ""                               # the audit of the opened listings follows on the decline path itself
            what = "the shopper has not been asked yet what the request leaves open"
        else:
            return ""
        self.early_held = True
        return (f"Only step {self.step} of {self.max_steps} is used and a fit is expected by step {self.pace_targets()[0]}: finishing this early loses the task when {what}. "
                "First " + ("ask the shopper the question still open, " if not asked else "") + "open the details of two or three more candidates and check the pick's own sheet against every requirement; "
                "then send this call again if it still holds.")

    def long_order_hold(self, pick: str) -> str:
        """In a long world, before an order on a line whose check predates a revealed or changed requirement: the line is
        re-checked against the requirements as they now stand, without a step; the finding holds the order once per cart
        state ("" when the line still passes or nothing was revealed since)."""
        if not self.long_world() or not pick or not self.revealed or f"{pick}|{self.ledger_version}" in self.verdicts:
            return ""
        if self.spend_level() == 2 or self.time_left() < WRAP_UP_SECONDS + 20 or not self.room_for_extras():
            return ""
        verdict = self.rule_check(pick, False) or self.verify_item(pick, "ORDER")
        if not verdict:
            return ""
        return verdict + " The requirements changed after this line was last checked, so check its own records against what the shopper revealed; if you are certain it fits, send the same call again."

    def fallback_note(self, target: str) -> str:
        """After a sold-out pick or a refused order: when the cart is about to take a dearer listing than the cheapest one
        already shown to fit, say so once."""
        if not (self.gone or self.refusals) or self.plan.get("priority") != "cheapest" or target in self.fallback_told:
            return ""
        looked = self.refill_candidates()                   # before a dearer line goes in: more candidates at the cheapest price may be a page away
        fits = [f for f in self.live_fits() if f[0] != target]
        price = self.latest_price(target)
        if not fits or price is None or fits[0][1] >= price - 1e-9:
            return ""
        self.fallback_told.add(target)
        return ((looked + " " if looked else "") + f"{fits[0][0]} ({fits[0][1]:g}) was already read and states every requirement on its own sheet, and it is cheaper than {target} ({price:g}). "
                f"Take {fits[0][0]} unless something the shopper said rules it out; if so, send this call again and say why. " + ONE_FIELD)

    def cheapest_known(self) -> float | None:
        """The lowest price at which any opened listing in the main category stating every requirement was seen within the
        limit, whether or not it can still be bought (sold-out and gone ones count: the price is what another fit must match)."""
        demands = self.demands()
        if not demands:
            return None
        kind, group, cap = set(label_words(self.plan.get("product") or "")), self.main_group(), self.budget_now
        prices = []
        for ident, sheet in self.sheets.items():
            own = _shelf(sheet) or _shelf(self.rows.get(ident) or {})
            if group and own and own != group:
                continue
            if all(read_sheet(self.flat_sheet(sheet), prop, value, kind)[0] == "yes" for prop, value in demands):
                price = self.latest_price(ident)
                over = isinstance(cap, (int, float)) and not isinstance(cap, bool) and price is not None and price * self.quantity_wanted() > cap + 1e-9
                if price is not None and not over:
                    prices.append(price)
        return min(prices) if prices else None

    def refill_room(self, need: int = 1) -> bool:
        return (self.refill_spent + need <= REFILL_STEPS and self.steps_left() - need >= self.finish_reserve(empty_cart=True) + 3
                and self.time_left() > WRAP_UP_SECONDS + 30 and self.spend_level() < 2 and not self.finished and not self.dead)

    def refill_candidates(self) -> str:
        """After a sellout or a refusal on a cheapest request, when no listing in stock is known to fit at the cheapest
        price seen to fit: before anything dearer is proposed, the agent turns the unseen pages of the model's own worded
        search (a hundred hits or fewer) and opens the cheapest unread rows at that price or below. A task whose pick
        sells out twice needs a third fit, and the pages the model left unread are where it hides. Bounded by
        REFILL_STEPS. Returns a short note ("" when nothing was done)."""
        if self.plan.get("priority") != "cheapest" or not (self.gone or self.refusals) or self.refill_spent >= REFILL_STEPS:
            return ""
        floor = self.cheapest_known()
        if floor is None or any(p <= floor + 1e-9 for _, p in self.live_fits()):
            return ""
        shape = self.search_shape() or {}
        tool, page_slot = shape.get("tool"), shape.get("page")
        done = 0
        if tool and page_slot:
            for key, cover in reversed(list(self.page_cover.items())):
                settings, words = key.rsplit("|", 1)
                if not words or settings in self.deaf_modes or cover["total"] > 100 or len(cover["seen"]) >= cover["total"] or not cover["seen"]:
                    continue
                per_page = max(1, len(cover["seen"]) // max(1, cover["page"]))
                last_page = -(-cover["total"] // per_page) + 1
                pages = 0
                while pages < 2 and self.refill_room() and len(cover["seen"]) < cover["total"] and cover["page"] < last_page:
                    before, page_before = len(cover["seen"]), cover["page"]
                    observation = self.agent_action(tool, dict(cover["args"], **{page_slot: cover["page"] + 1}),
                                                    f"The cheapest fit so far can no longer be bought: turning to the next page of the earlier search for another at {floor:g} or less.", "refill")
                    self.refill_spent += 1
                    pages += 1
                    if observation is None:
                        break
                    if cover["page"] == page_before:        # the bookkeeping after the call records the page turned; failing that, count it here
                        cover["page"] = page_before + 1
                    done += 1
                    if len(cover["seen"]) == before:
                        break
                break
        unread = [i for _, i in self.below(floor + 0.005, self.main_group())]   # rows at the floor or below never opened, in stock, affordable
        for ident in unread[:2]:
            if not self.refill_room():
                break
            calls = self.open_calls([ident])
            if not calls:
                break
            function = calls[0]["function"]
            if self.agent_action(function["name"], _parse_json_object(function["arguments"]) or {},
                                 f"Opening {ident} to see whether it states every requirement at {floor:g} or less.", "refill") is None:
                break
            self.refill_spent += 1
            done += 1
        if not done:
            return ""
        fits = [f for f in self.live_fits() if f[1] <= floor + 1e-9]
        if fits:
            return f"The agent looked further for a fit at {floor:g} or less and found {fits[0][0]} ({fits[0][1]:g}) in stock, stating every requirement on its own sheet."
        return f"The agent looked further (pages and sheets) for a fit at {floor:g} or less and found none in stock."

    def removal_target(self, name: str, args: dict[str, Any]) -> str:
        """The listing a call takes out of the cart: the cart tool in a removing mode, or a separate removing tool ("" otherwise)."""
        if name not in self.schemas:
            return ""
        properties = self.schemas[name].get("properties") or {}
        if not self.removing_call(name, args):
            return ""
        return next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), "")

    def remove_args(self, ident: str) -> tuple[str, dict[str, Any]] | None:
        """A call that takes one listing out of the cart: the cart tool's removing mode, else the separate removing tool."""
        cart = self.roles.get("cart")
        if cart and self.basket_spec(cart):
            shaped, problems = shape_args(self.schemas[cart], self.basket_args([i for i in self.cart_ids if i != ident]), "args")
            return (cart, shaped) if not problems else None
        modes = self.cart_modes(cart)
        if cart and modes:
            args = dict(self.cart_args(ident))
            for key, (_, out) in modes.items():
                args[key] = out
            props = self.schemas[cart].get("properties") or {}
            for key, value in list(args.items()):
                if isinstance(value, int) and not isinstance(value, bool) and "enum" not in (props.get(key) or {}) and self.mode_units.get(ident, 0) > value:
                    args[key] = self.mode_units[ident]
            shaped, problems = shape_args(self.schemas[cart], args, "args")
            return (cart, shaped) if not problems else None
        name = self.roles.get("remove")
        if name:
            props = (self.schemas.get(name) or {}).get("properties") or {}
            key = next((k for k in (self.schemas[name].get("required") or []) if (props.get(k) or {}).get("type") == "string"), None)
            key = key or next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v), None)
            if key:
                shaped, problems = shape_args(self.schemas[name], {key: ident}, "args")
                return (name, shaped) if not problems else None
            many = next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array"), None)
            if many:
                shaped, problems = shape_args(self.schemas[name], {many: [ident]}, "args")
                return (name, shaped) if not problems else None
        return None

    def model_veto(self, item: str) -> bool:
        """The model may keep the agent from swapping to `item` only by naming, in its latest reasoning, a requirement
        property that the listing's own sheet contradicts (code-checkable); a bare "labels conflict" is not a veto."""
        sheet = self.sheets.get(item)
        if sheet is None or not self.last_said:
            return False
        said = lowered(self.last_said)
        kind = set(label_words(self.plan.get("product") or ""))
        for prop, value in self.demands():
            if read_sheet(self.flat_sheet(sheet), prop, value, kind)[0] == "no" and any(len(w) > 3 and w in said for w in word_list(prop)):
                return True
        return False

    FLAGS = r"conflict|mixed|ambiguous|not a |not an |bundle|\bsets?\b|combo|kit|mismatch|inconsisten|decoy|wrong"

    def flagged_by_model(self, item: str) -> bool:
        """True when a clause of the model's latest reasoning names this listing together with a doubt word (a conflict, a
        mixed sheet, a set or bundle, a mismatch): a listing the model itself called into question is never chosen for it."""
        said = lowered(self.last_said)
        if item.lower() not in said:
            return False
        clauses = re.split(r"[.;!?,:]|\bbut\b|\bso\b|\bwhile\b|\bwhereas\b|\bthen\b", said)
        return any(item.lower() in clause and re.search(self.FLAGS, clause) for clause in clauses)

    def shunned(self, item: str) -> bool:
        """A listing the agent never picks on the model's behalf: one the model took out of the cart itself, or one its
        latest words called into question."""
        return item in self.model_discarded or self.flagged_by_model(item)

    def digit_noise(self, item: str) -> int:
        """How many other fields of the listing's sheet carry a figure shaped like a requirement's figure but different
        from it (a '1080p' or '1920x1080' beside a stated '3440 x 1440'): a tie-breaker between fits, lower is cleaner."""
        sheet = self.flat_sheet(self.sheets.get(item) or {})
        kind = set(label_words(self.plan.get("product") or ""))
        noise = 0
        for prop, value in self.demands():
            mine = set(re.findall(r"\d+", value))
            if not mine:
                continue
            lengths = {len(t) for t in mine}
            for key, text in sheet.items():
                name = str(key)
                if label_fits(name, prop, kind) or grouped(name) or name in (FIELDS.get("price"), FIELDS.get("stock")) or re.search(r"(?i)price|cost|stock|avail|quant|qty|rating|review|count|\bid\b|_id|pid|sku", name):
                    continue
                for token in re.findall(r"\d+(?:\.\d+)?[a-z]{0,2}\b", lowered(str(text))):
                    digits = re.sub(r"\D", "", token)
                    if digits and len(digits) in lengths and digits not in mine:
                        noise += 1
                        break
        return noise

    def hand_back(self, current: str, fits: list[tuple[str, float]]) -> str:
        """Nothing an independent check passed is left to swap to: the agent only takes the gone line out (one step) and
        lists the listings whose own sheets state every requirement, cheapest first, for the model to choose and order."""
        if current:
            if self.steps_left() < self.finish_reserve(swap=False) + 1:
                return ""
            removal = self.remove_args(current)
            if removal is None:
                return ""
            self.recovered_at = self.changes
            if self.agent_action(removal[0], removal[1], f"{current} can no longer be bought: taking it out of the cart.", "recover") is None:
                return ""
            self.cart_ids = [i for i in self.cart_ids if i != current]
            self.cart_changed_at = self.step
        else:
            self.recovered_at = self.changes
        kind = set(label_words(self.plan.get("product") or ""))
        lines = []
        for ident, price in fits[:4]:
            shown = "; ".join(read_sheet(self.flat_sheet(self.sheets.get(ident) or {}), prop, value, kind)[1] for prop, value in self.demands()[:3])
            lines.append(f"{ident} ({price:g}): {shown}")
        return ((f"The agent took {current} out of the cart. " if current else "") + "No remaining listing has passed an independent check yet; these state every requirement on their own sheets, cheapest first: "
                + " | ".join(lines) + ". Choose the one that truly fits, put it in the cart and order it, or decline. " + ONE_FIELD)

    def discard_hold(self, name: str, args: dict[str, Any]) -> str:
        """A model call that takes a fitting listing out of the cart is set aside: every time for a line the agent put in
        itself while its own sheet still states every requirement (the model may take it out only by naming a requirement
        property the sheet contradicts), once for any fit on a cheapest request; otherwise "" ("" also once a repeat is due)."""
        if self.steps_left() <= 2:
            return ""
        target = self.removal_target(name, args)
        if not target or target in self.gone or target not in dict(self.live_fits()):
            return ""
        price = self.latest_price(target)
        shown = "; ".join(read_sheet(self.flat_sheet(self.sheets.get(target) or {}), prop, value, set(label_words(self.plan.get("product") or "")))[1]
                          for prop, value in self.demands()[:3])
        if target in self.agent_placed:
            if self.model_veto(target):
                return ""                               # a contradicted property was named: the model may take it out
            return (f"{target} ({price:g}) was put in by the agent because its own sheet states every requirement in force ({shown}); taking it out "
                    "on a guess leaves nothing to order. Order it, or name the requirement property its sheet truly contradicts and then take it out. " + ONE_FIELD)
        if self.plan.get("priority") != "cheapest" or target in self.discard_held:
            return ""
        self.discard_held.add(target)
        return (f"{target} ({price:g}) states every requirement on its own sheet ({shown}), so taking it out would lose the cheapest fit. "
                + ONE_FIELD + " If a requirement property on its sheet truly states another value, name that property and send the same call again.")

    def enforce_cheapest(self) -> str:
        """Cheapest requests, at order time: when a cheaper in-stock listing already states every requirement on its own
        sheet, the agent itself swaps the single cart line for it (two store steps, once per episode) and the order goes
        on. The model can stop it only by naming a property the listing's sheet contradicts. Returns the note or ""."""
        if self.plan.get("priority") != "cheapest" or self.swap_done or self.companions or len(self.cart_ids) != 1 or not self.parse_trusted():
            return ""                               # (a parse that misreads this store never overrides the model's pick)
        current = self.cart_ids[0]
        now = self.latest_price(current)
        if now is None or self.steps_left() < self.finish_reserve(swap=False) + 2 or self.time_left() < WRAP_UP_SECONDS / 2:
            return ""
        fits = [(i, p) for i, p in self.live_fits() if i != current and p < now - 1e-9 and self.affordable(self.sheets.get(i) or self.rows.get(i) or {})]
        fits = [(i, p) for i, p in fits if not self.model_veto(i) and not self.shunned(i)]
        if not fits:
            return ""
        best, price = fits[0]
        removal = self.remove_args(current)
        cart = self.roles.get("cart")
        if removal is None or not cart:
            return ""
        self.swap_done = True
        actions = ([] if self.basket_spec(cart) else [{"name": removal[0], "args": removal[1]}]) + [{"name": cart, "args": self.cart_args(best)}]
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        for action in actions:                      # one envelope each: every store takes a single action, not all take a group
            reply = self.env_call([action])
            self.trace.append({"kind": "cheapest_swap", "from": current, "to": best, "price": price, "action": action, "reply": reply})
            if reply is None or self.finished or self.dead:
                return ""
            self.book_result(action["name"], action["args"], reply.get("observation"), reply, cart, self.roles.get("message"), finishing)
        self.cart_ids = [i for i in self.cart_ids if i != current and i != best] + [best]
        self.cart_used, self.cart_changed_at, self.order_checked = True, self.step, True
        return (f"The cart now holds {best} ({price:g}) instead of {current} ({now:g}): its own sheet states every requirement and it costs less, "
                f"which the request asks for. " + ONE_FIELD + " Ordering it.")

    def forced_pick(self, fits: list[tuple[str, float | None]]) -> str:
        """The fit a finish in code takes: the one a best rule ranks first, else the preferred one."""
        if self.plan.get("priority") == "best" and self.measure_words():
            priced = [(i, p if p is not None else 0.0) for i, p in fits]
            return self.ranked_by_rule(priced)[0][0]
        return self.preferred(fits)

    def enforce_best(self) -> None:
        """A finish in code under a best rule: when an opened in-stock fit within the limit states a better measure on its
        own sheet than the single held line, the cart is swapped to it (once) before the order."""
        if (self.plan.get("priority") != "best" or self.swap_done or self.companions or self.bundle_names or len(self.cart_ids) != 1 or not self.measure_words()
                or not self.parse_trusted()):
            return
        current = self.cart_ids[0]
        fits = [(i, p) for i, p in self.live_fits() if self.affordable(self.sheets.get(i) or self.rows.get(i) or {}) and not self.model_veto(i) and not self.shunned(i)]
        ranked = self.ranked_by_rule(fits)
        if not ranked or ranked[0][0] == current or self.steps_left() < 4:
            return
        best = ranked[0][0]
        held, top = self.measure_of(current), self.measure_of(best)
        if top is None or (held is not None and held == top):
            return
        removal, cart = self.remove_args(current), self.roles.get("cart")
        if removal is None or not cart:
            return
        self.swap_done = True
        for action in ([] if self.basket_spec(cart) else [{"name": removal[0], "args": removal[1]}]) + [{"name": cart, "args": self.cart_args(best)}]:
            reply = self.env_call([action])
            self.trace.append({"kind": "best_swap", "from": current, "to": best, "action": action, "reply": reply})
            if reply is None or self.finished or self.dead:
                return
        self.cart_ids = [best]
        self.cart_used, self.cart_changed_at = True, self.step

    def agent_action(self, name: str, args: dict[str, Any], said: str, tag: str) -> Any:
        """One store call the agent makes on its own (one envelope, one step), kept in the conversation like a model turn
        and booked like any result. Returns the observation, or None when the call could not run."""
        if name not in self.schemas or self.finished or self.dead or self.steps_left() <= 0:
            return None
        args, problems = shape_args(self.schemas[name], args, "args")
        if problems:
            return None
        call_id = f"{tag}_{self.step}"
        self.messages.append({"role": "assistant", "content": said, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        reply = self.env_call([{"name": name, "args": args}])
        self.trace.append({"kind": tag, "action": {"name": name, "args": args}, "reply": reply})
        if reply is None:
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": "The call failed (network or server)."})
            return None
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        observation = self.book_result(name, args, reply.get("observation"), reply, self.roles.get("cart"), self.roles.get("message"), finishing)
        raw = reply.get("raw_observation")
        text = self.result_text(observation if raw is None else raw, observation, name)
        if name != self.roles.get("message"):
            heard, _ = self.shopper_lines(name, args, observation, self.roles.get("message"))
            heard += [t for t in self.notice_texts(observation) if t not in heard]
            if heard:
                notes = self.hear_in(heard)
                meaning = self.interpret(heard, None)
                if notes or meaning:
                    text += "\n[note] the result carries the shopper's words: " + " ".join(notes + ([meaning] if meaning else []))
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        return observation

    def result_text(self, raw: Any, parsed: Any, name: str) -> str:
        """What the model reads for one result: the store's own text (_raw_text), then, apart from it, a short note of
        what the agent's parse made of it, marked as code-derived. A message's reply is the shopper's text alone."""
        text = _raw_text(raw)
        if raw is not None and name != self.roles.get("message"):
            self.raw_results = (self.raw_results + [raw])[-10:]
        if name == self.roles.get("message"):
            return text
        note = self.parse_note(raw, parsed, name)
        return text + ("\n" + note if note else "")

    def parse_note(self, raw: Any, parsed: Any, name: str) -> str:
        """The agent's parse of a result as one short line: the listing ids it found with price and stock. A search or a
        reading whose parse found far fewer ids than the raw text shows records is counted as a low-confidence parse, and
        the note then tells the model to read the text itself (its reading wins)."""
        if name in {self.roles.get("order"), self.roles.get("decline"), self.roles.get("message")} - {None}:
            return ""
        single, rows = self.records_in(normal(parsed)) if isinstance(parsed, (dict, list)) else (None, None)
        records = ([single] if isinstance(single, dict) else []) + [r for r in rows or [] if isinstance(r, dict)]
        found: list[tuple[str, dict[str, Any]]] = []
        for record in records:
            ident = self.id_in(record)
            if ident and all(ident != i for i, _ in found):
                found.append((ident, record))
        apparent = _apparent_records(raw)
        searching = name == self.roles.get("search")
        looked = searching or name in self.reading_tools()
        # a search is a low-confidence parse when it read under half the records its text shows; a reading (one listing,
        # whose page may carry lists of its own) only when it read no listing at all
        low = looked and apparent >= 1 and (len(found) * 2 < apparent if searching else not found)
        if looked and apparent >= 1:
            self.parse_seen += 1
            self.parse_low += int(low)
        if low:
            return (f"[agent's own parse (code-derived): it found {len(found)} listing id(s) where the store's text above shows about {apparent} "
                    "record(s), so it cannot vouch for this result. Read the store's text yourself; where it and any agent note disagree, the store's text wins.]")
        if not found:
            return ""
        parts = []
        for ident, record in found[:24]:
            price, stock = _price(record), _in_stock(record)
            bits = [f"{price:g}" if price is not None else "price ?"] + (["in stock"] if stock is True else ["out of stock"] if stock is False else [])
            parts.append(f"{ident} ({', '.join(bits)})")
        more = f"; and {len(found) - 24} more" if len(found) > 24 else ""
        return ("[agent's own parse (code-derived, may be wrong; the store's text above wins): " + "; ".join(parts) + more)[: NOTE_CHARS - 1] + "]"

    def parse_trusted(self) -> bool:
        """Whether the agent's parse of this store may act on its own (cart a fit, swap a line, fill a basket): not when
        at least half of the searches and readings so far were low-confidence parses. Then the model's reading leads."""
        return self.parse_seen < 2 or self.parse_low * 2 < self.parse_seen

    def ids_shown(self, observation: Any) -> set[str]:
        """The known listings (rows, sheets, cart lines) that a result names anywhere, whatever the key is called."""
        known = set(self.rows) | set(self.sheets) | set(self.cart_ids)
        found: set[str] = set()
        def walk(value: Any, depth: int) -> None:
            if depth > 6 or len(found) >= len(known):
                return
            if isinstance(value, str):
                if value in known:
                    found.add(value)
            elif isinstance(value, dict):
                for inner in value.values():
                    walk(inner, depth + 1)
            elif isinstance(value, list):
                for inner in value[:60]:
                    walk(inner, depth + 1)
        walk(observation, 0)
        return found

    def fresh_look(self, pick: str) -> str:
        """Before the order, the pick's stock and price must have been seen since the last change in the store (not since
        the last cart change: a cart change moves no prices). When no result since then shows the pick, the agent re-reads
        it once itself (one step, no model turn). Returns a note for the model ("" when nothing was done)."""
        if not pick or pick in self.fresh_viewed or not self.market_changed_at or self.seen_at.get(pick, -1) >= self.market_changed_at:
            return ""                                   # no price has moved, or the pick was shown since the last move: nothing is stale
        if self.steps_left() < self.finish_reserve() + 1 or self.time_left() < WRAP_UP_SECONDS / 2:
            return ""
        calls = self.open_calls([pick]) if (self.roles.get("details") or self.roles.get("view")) else []
        if not calls:
            return ""
        self.fresh_viewed.add(pick)
        function = calls[0]["function"]
        observation = self.agent_action(function["name"], _parse_json_object(function["arguments"]) or {}, f"Re-reading {pick} since the store changed.", "fresh_look")
        if observation is None:
            return ""
        price = self.latest_price(pick)
        return f"The agent re-read {pick} after the last change in the store" + (f": price {price:g}" if price is not None else "") + ("; it is shown sold out" if pick in self.gone else "") + "."

    def questions_done(self) -> bool:
        """Whether the shopper has been asked what was going to be asked: the planner is off and at least one question went
        out (or none can: no message tool, or no message to spend), or the messages the planner may use are spent."""
        if self.messages_used >= self.message_limit() - self.held_messages():
            return True
        asked = self.messages_used > 0 or not self.roles.get("message") or self.message_limit() - self.held_messages() <= 0
        return not self.planner_on and asked

    def commit_cart(self) -> bool:
        """Requests that do not ask for the cheapest: when a fit is known (its own sheet states every requirement in force,
        it is in stock, affordable, not gone, and no check has failed it), nothing is ordered yet, the cart is empty and the
        questions are done, the agent itself puts the fit in the cart at the commit mark (or at COMMIT_ASKED_BY of the cap once
        the questions are done): the model then has only to order it, or to name a requirement property its sheet
        contradicts. Once per episode; never on bundles. True when the cart was filled."""
        if self.commit_done or self.plan.get("priority") == "cheapest" or self.companions or self.order_attempts or self.cart_ids or not self.parse_trusted():
            return False
        cart = self.roles.get("cart")
        if not cart or self.change_reason or self.spend_level() == 2 or self.time_left() < WRAP_UP_SECONDS or self.max_steps < 2 * PAGE_FROM_STEPS:
            return False                                # in a short episode the two steps a wrong pick costs are the finish's
        # the agent commits only when nothing is left to learn: no story hint still open, nothing the request says to ask
        # first still unasked, and every planned question sent (carting before a hidden need surfaces loses the task)
        overdue = self.step >= self.pace_targets()[1]
        if not overdue and (self.hints() or (self.plan.get("ask_first") and not self.asked) or not self.questions_done()):
            return False
        if self.step < round(self.max_steps * COMMIT_ASKED_BY):
            return False
        if self.steps_left() < self.finish_reserve(empty_cart=True) + 1:
            return False
        fit = self.paced_fit()
        if not fit or self.shunned(fit) or self.seen_at.get(fit, -1) < self.market_changed_at or self.model_veto(fit):
            return False                                # a listing the model took out or called into question is not put in for it
        if self.long_world() and f"{fit}|{self.ledger_version}" not in self.verdicts:
            # a long world reveals needs late: the fit is checked against the requirements as they now stand before it goes in
            if self.rule_check(fit, True) or self.verify_item(fit, "ADD TO THE CART"):
                return False
        self.commit_done = True
        observation = self.agent_action(cart, self.cart_args(fit), f"{fit} states every requirement on its own sheet and the order is due: putting it in the cart.", "commit_cart")
        if observation is None or fit not in self.cart_ids:
            return False
        self.agent_placed.add(fit)
        price = self.latest_price(fit)
        shown = "; ".join(read_sheet(self.flat_sheet(self.sheets.get(fit) or {}), prop, value, set(label_words(self.plan.get("product") or "")))[1]
                          for prop, value in self.demands()[:3])
        self.messages.append({"role": "user", "content": (f"[commit, done by the agent] {fit}" + (f" ({price:g})" if price is not None else "") + f" is in the cart: its own sheet states every requirement in force ({shown}) and the order is due by step {self.pace_targets()[1]} of {self.max_steps}. "
                              "Order it now. Only if a requirement property on its sheet truly states another value, name that property, take it out and choose another fit. " + ONE_FIELD)})
        return True

    def recover(self) -> str:
        """After a line sold out or an order was refused: when another listing fits (its own sheet states every requirement,
        in stock, affordable, not gone, never one the model took out or doubted; cheapest first when the cheapest is wanted,
        else only one an independent check passed), the agent itself takes the gone line out and puts that one in (two
        steps), so the order can follow at once instead of a long re-search; with no checked fit it only takes the gone
        line out and lists the sheet-fits for the model. Once per change in the store; never on bundles, never when the request
        says to ask before replacing and nothing was asked yet. Returns the note for the model ("" when nothing was done)."""
        if self.companions or not (self.gone or self.refusals):
            return ""
        again = False
        if self.recovered_at == self.changes:
            # in a long world the cart handed back after a change is repaired once more when the model has since checked a fit:
            # the straight path (recover, re-verify, order) beats a long re-search there
            again = self.long_world() and self.handed_at == self.changes and self.second_recoveries < 1 and not self.cart_ids and self.room_for_extras()
            if not again:
                return ""
        cart = self.roles.get("cart")
        if not cart or len(self.cart_ids) > 1 or self.spend_level() == 2 or self.time_left() < WRAP_UP_SECONDS / 2:
            return ""
        current = self.cart_ids[0] if self.cart_ids else ""
        if current and current not in self.gone and self.affordable(self.sheets.get(current) or self.rows.get(current) or {}) and current in dict(self.live_fits()):
            return ""                                   # the line still fits: the refusal asked for something else than a swap
        if any(d["when"] == "replace" for d in self.duties) and not self.swap_asked:
            return ""                                   # the request wants the shopper asked first: the model's add does that
        usable = lambda: [(i, p) for i, p in self.live_fits() if i != current and i not in self.gone and not self.model_veto(i) and not self.shunned(i)]
        fits = usable()
        looked = ""
        if self.plan.get("priority") == "cheapest":
            floor = self.cheapest_known()
            if floor is not None and not any(p <= floor + 1e-9 for _, p in fits):
                looked = self.refill_candidates()           # a third fit at the cheapest price may sit on an unread page: look before settling for a dearer one
                fits = usable()
        if not fits and not looked and self.steps_left() >= max(10, self.finish_reserve(swap=False) + 7) and self.time_left() > WRAP_UP_SECONDS + 20:
            group = self.main_group()
            unread = [i for _, i in sorted(((_price(r) or 0.0), i) for i, r in self.rows.items()
                                           if i not in self.sheets and i not in self.gone and self.affordable(r) and (not group or not _shelf(r) or _shelf(r) == group))][:3]
            calls = self.open_calls(unread) if unread else []
            if calls:
                self.own_rounds(calls, "The held line is gone and no other opened listing fits: reading the next unread candidates (" + ", ".join(unread) + ").")
                fits = usable()
                looked = "The agent read " + ", ".join(unread) + "."
        if not fits:
            return looked
        if self.plan.get("priority") == "cheapest":
            best = fits[0][0]
        else:
            # without a cheapest rule the agent swaps only to a listing an independent check passed under the requirements as
            # they stand (the cleanest sheet first, then the cheapest); a sheet-fit nobody checked is the model's call
            order = {i: n for n, (i, _) in enumerate(self.ranked_by_rule(fits))} if self.measure_words() else {}
            rank = lambda i: (order.get(i, 0), self.digit_noise(i), self.latest_price(i) or 0, i)
            passed = sorted((i for i, _ in fits if self.verdicts.get(f"{i}|{self.ledger_version}") == ""), key=rank)
            if order and passed and passed[0] != min(order, key=order.get) and self.verify_swaps < 2 and self.spend_level() < 2:
                passed = []                             # under a best-by-measure rule the best remaining fit gets its own check first
            if not passed and self.verify_swaps < 2 and (self.questions_done() or order):
                # the cheapest sheet-fit gets one independent check now: a straight swap beats a long re-search after a change
                self.verify_swaps += 1
                candidate = sorted((i for i, _ in fits), key=rank)[0]
                if not self.verify_item(candidate, "SWAP INTO THE CART") and self.verdicts.get(f"{candidate}|{self.ledger_version}") == "":
                    passed = [candidate]
                elif order:
                    passed = sorted((i for i, _ in fits if i != candidate and self.verdicts.get(f"{i}|{self.ledger_version}") == ""), key=rank)
            if not passed:
                if again:
                    return ""
                self.handed_at = self.changes
                return ((looked + " ") if looked else "") + self.hand_back(current, fits)
            best = passed[0]
        need = (1 if current else 0) + 1
        if self.steps_left() < self.finish_reserve(swap=False) + need:
            return ""
        if again:
            self.second_recoveries += 1
        self.recovered_at = self.changes
        said = f"{current or 'the pick'} can no longer be bought; {best} states every requirement on its own sheet: swapping the cart to it."
        if current:
            removal = self.remove_args(current)
            if removal is None:
                return ""
            if self.agent_action(removal[0], removal[1], said, "recover") is None:
                return ""
            self.cart_ids = [i for i in self.cart_ids if i != current]
        observation = self.agent_action(cart, self.cart_args(best), said, "recover")
        if observation is None or best not in self.cart_ids:
            return ""
        self.cart_changed_at = self.step
        self.agent_placed.add(best)
        price = self.latest_price(best)
        return ((looked + " " if looked else "") + f"The agent took {current or 'the gone line'} out and put {best}" + (f" ({price:g})" if price is not None else "") + " in: its own sheet states every requirement in force"
                + (", and it is the cheapest such listing read" if self.plan.get("priority") == "cheapest" else "") + ". Order it now, or name the requirement property its sheet contradicts. " + ONE_FIELD)

    def order_now(self, why: str, force: bool = False) -> str:
        """Right after the agent repaired the cart, it sends the order itself through the usual order path (the item check,
        the fresh look, the final look, the duties owed), instead of handing the turn back for a long re-check. The order's
        result, or the hold that stopped it, is kept in the conversation like a model turn. Returns a note ("" when not sent).
        With `force` (the decline gate in the last steps) the finish reserve is not required, only a step."""
        order = self.roles.get("order")
        if (not order or self.finished or self.dead or self.auto_orders >= AUTO_ORDERS + 1 or not self.cart_ids
                or self.cart_ids[-1] not in self.agent_placed or self.cart_ids[-1] in self.gone
                or self.steps_left() < (1 if force else self.finish_reserve(swap=False)) or self.time_left() < (6 if force else WRAP_UP_SECONDS / 2)):
            return ""
        if any(d["when"] == "replace" for d in self.duties) and not self.swap_asked:
            return ""                                   # the shopper must be asked first; the model's turn does that
        self.auto_orders += 1
        args = self.neutral_args(order, "Order the cart.", "order")
        call_id = f"agent_order_{self.step}"
        call = {"id": call_id, "type": "function", "function": {"name": order, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": why + " Ordering it now.", "tool_calls": [call]})
        before = self.order_attempts
        replies = self.run_calls([call])
        for reply_id, text in replies:
            self.messages.append({"role": "tool", "tool_call_id": reply_id, "content": text})
        self.trace.append({"kind": "auto_order", "sent": self.order_attempts > before, "finished": self.finished})
        if self.finished:
            return "The agent then sent the order; the episode is finished."
        if self.order_attempts > before:
            return "The agent then sent the order itself; the store's answer is above."
        return "The agent then tried to order; what still has to happen first is above."

    def search_key(self, args: dict[str, Any]) -> tuple[str, str]:
        """A search's settings without its words or page, and its words."""
        shape = self.search_shape() or {}
        words_slot, page_slot = shape.get("words"), shape.get("page")
        settings = _compact({k: v for k, v in args.items() if k not in (words_slot, page_slot)})
        return settings, str(args.get(words_slot) or "").strip().lower() if words_slot else ""

    def note_search(self, args: dict[str, Any], observation: Any) -> None:
        """Learn at run time whether the words matter under these settings: two searches with the same settings and
        different words that report the same total and the same first rows mean the words are ignored there."""
        settings, words = self.search_key(args)
        total, shown = self.match_count(observation)
        if total is None or not words:
            return
        page_slot = (self.search_shape() or {}).get("page")
        page = args.get(page_slot) if page_slot and isinstance(args.get(page_slot), (int, float)) and not isinstance(args.get(page_slot), bool) else 1
        cover = self.page_cover.pop(settings + "|" + words, {"args": dict(args), "total": total, "seen": set(), "page": 0})
        cover["total"], cover["page"] = total, max(cover["page"], int(page))
        cover["seen"] |= set(shown)
        self.page_cover[settings + "|" + words] = cover                     # newest search last
        seen = self.search_counts.setdefault(settings, [])
        for other_words, other_total, other_first in seen:
            if other_words != words and other_total == total and other_first == tuple(shown[:3]) and total > 0:
                self.deaf_modes.add(settings)
        seen.append((words, total, tuple(shown[:3])))
        # the words may change together with the limit or the page: under the same mode and category, a different wording
        # that reports the same long total (or the same first rows) still means the words are ignored there
        ceiling = (self.search_shape() or {}).get("ceiling")
        coarse = _compact({k: v for k, v in args.items() if k not in (words_slot_of(self), page_slot, ceiling)})
        kin_seen = self.coarse_counts.setdefault(coarse, [])
        for other_words, other_total, other_first, other_settings in kin_seen:
            if other_words != words and other_total == total and total > 0 and (other_first == tuple(shown[:3]) or total >= 50):
                self.deaf_modes.add(settings)
                self.deaf_modes.add(other_settings)
        kin_seen.append((words, total, tuple(shown[:3]), settings))

    def plain_words(self, text: str) -> str:
        """Search words kept to what the shopper's own words, the requirements, the needs heard and the shelf name hold."""
        shelves = {_shelf(r) for r in list(self.rows.values()) + list(self.sheets.values())} - {""}
        titles = [str(v) for ident, _ in (self.fits_by_facts() or self.fits_by_facts(strict=False))
                  for k, v in (self.sheets.get(ident) or self.rows.get(ident) or {}).items() if isinstance(v, str) and titled(k)]
        allowed = " ".join([str(self.policy.get("query") or ""), str(self.plan.get("product") or ""), self.main_group()]
                           + [str(r) for r in self.stated + self.revealed + (self.current or [])] + sorted(shelves) + titles)
        keep = {stem(w) for w in word_list(allowed)} - self.shed_words
        words = word_list(text)
        kept = [w for w in words if stem(w) not in self.shed_words and (stem(w) in keep or any(kin(w, k) for k in keep if len(k) > 3))]
        return " ".join(kept) if kept and len(kept) < len(words) else text

    def search_guard(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Search hygiene: words for unstated things are dropped, and searches that vary only the words (or page a long
        list) under settings where the words are ignored are not run."""
        shape = self.search_shape() or {}
        words_slot, page_slot = shape.get("words"), shape.get("page")
        if name != self.roles.get("search") or not words_slot:
            return args, ""
        wanted = self.wanted_codes()
        if (wanted and isinstance(args.get(words_slot), str) and CODE_TOKEN.search(args[words_slot])
                and (self.code_misses or any(not i.startswith("part:") for i in self.fit_statements()))):
            families = {code_family(w) for w in wanted}
            stripped = " ".join(w for w in args[words_slot].split() if not any(code_family(t) in families for t in CODE_TOKEN.findall(w)))
            if not stripped.strip():
                return args, ("a model code is never in a listing's searchable text, so this search finds nothing. Search the product's own words "
                              "(its kind, its requirements) and read each candidate's page for the code.")
            args = dict(args, **{words_slot: stripped})
        if isinstance(args.get(words_slot), str) and args[words_slot].strip():
            args = dict(args, **{words_slot: self.plain_words(args[words_slot])})
        if self.shed_words:
            props = (self.schemas.get(name) or {}).get("properties") or {}
            for key, value in args.items():
                if isinstance(value, str) and "enum" not in (props.get(key) or {}) and key != shape.get("group"):
                    tokens = [stem(w) for w in word_list(value)]
                    if tokens and all(t in self.shed_words for t in tokens):
                        return args, (f"{value!r} describes the listing that a need points at (its shop or its attributes), not the item to buy; the agent reads that page itself. "
                                      "Search for the item's own kind and requirements instead.")
        settings, words = self.search_key(args)
        if settings in self.deaf_modes:
            known = self.search_counts.get(settings) or []
            total = known[-1][1] if known else 0
            paging = page_slot and isinstance(args.get(page_slot), (int, float)) and args.get(page_slot) not in (0, 1)
            if (words and all(words != w for w, _, _ in known)) or (paging and total > 100):
                props = (self.schemas.get(name) or {}).get("properties") or {}
                group_slot = shape.get("group")
                ways = []
                if group_slot and args.get(group_slot):
                    ways.append(f"run the same words with {group_slot} left out")
                for key, value in args.items():
                    spec = props.get(key) if isinstance(props.get(key), dict) else {}
                    others = [v for v in (spec.get("enum") or []) if v != value]
                    if others:
                        ways.append(f"set {key} to {others[0]!r} with the same words and turn its pages")
                        break
                if len(ways) < 2:
                    ways.append("use the other search mode with the same words and turn its pages")
                return args, (f"under these settings the store ignores the words ({total} matches whatever the words), so this would show the same list again. "
                              + "Instead " + ", or ".join(ways) + ", or keep these settings and lower the price limit.")
        return args, ""

    def order_note(self, item: str) -> str:
        """Once per listing and requirement version: what its own sheet contradicts, else the cheaper fits it must beat."""
        key = f"{item}|{self.ledger_version}|order"
        if item and key not in self.rule_told:
            self.rule_told.add(key)
            return self.rule_check(item, False) or self.cheaper_note(item)
        return ""

    def unseen_searches(self) -> list[str]:
        return [f"'{c['label']}' ({c['total']} matched, {len(c['seen'])} seen)" for c in self.coverage.values() if c["label"] and len(c["seen"]) < c["total"] <= 60][:3]

    def decline_note(self) -> str:
        """Before a decline, once per requirement version: the opened listings that state everything, or else the unread
        candidates and the searches with matches never seen."""
        key = f"decline|{self.ledger_version}"
        if key in self.rule_told or self.steps_left() <= 3 or self.time_left() < WRAP_UP_SECONDS:
            return ""
        fits = self.fits_by_facts()
        if fits:
            self.rule_told.add(key)
            return ("Not a decline yet: by their own sheets these opened listings state every requirement in force, within the limit and not shown out of stock: "
                    + ", ".join(f"{i} ({p:g})" if p is not None else i for i, p in fits[:5]) + ". Re-read the best one (sheet, stock, price), cart it and order it; decline only if each one truly fails.")
        reasons = []
        unread = self.shelf_candidates() if self.steps_left() > 6 else []
        if unread:
            reasons.append("these listings within the limit were never read: " + ", ".join(unread) + "; their sheets decide, their titles do not")
        unseen = self.unseen_searches() if self.steps_left() > 6 else []
        if unseen:
            reasons.append("these searches still have matches you never saw: " + ", ".join(unseen) + "; turn their further pages")
        if not reasons:
            return ""
        self.rule_told.add(key)
        return "No opened listing states every requirement yet, but before declining: " + "; ".join(reasons) + ". Decline only if each one fails."

    LOGISTICS = r"(?i)option|condition|warrant|model|title|categor|brand|price|stock|seller|sku|origin|series|assembl|care|instruction|dimension"

    def label_table(self) -> tuple[dict[str, int], dict[str, int], dict[str, dict[str, int]]]:
        """Per property label (its words, 'type' aside) across the opened sheets: how many sheets carry it, how many carry
        a figure under it, and the spellings seen."""
        carried, numeric, spellings, _ = self.label_table_values()
        return carried, numeric, spellings

    def label_table_values(self) -> tuple[dict[str, int], dict[str, int], dict[str, dict[str, int]], dict[str, set[str]]]:
        """The label table with, per label, the distinct values seen: a property whose value differs between the opened
        listings is one the shopper's hidden need can split them on."""
        carried: dict[str, int] = {}
        numeric: dict[str, int] = {}
        spellings: dict[str, dict[str, int]] = {}
        distinct: dict[str, set[str]] = {}
        for sheet in self.sheets.values():
            facts = next((v for v in sheet.values() if isinstance(v, dict)), {})
            present = {" ".join(label_words(str(k))): str(v) for k, v in facts.items() if str(v).strip() and not re.fullmatch(r"(?i)n/?a|none|not specified|unknown|null|-", str(v).strip())}
            for label, value in present.items():
                base = " ".join(w for w in label.split() if w != "type")
                if not base:
                    continue
                spellings.setdefault(base, {})[label] = spellings.get(base, {}).get(label, 0) + 1
                distinct.setdefault(base, set()).add(value.strip().lower()[:40])
                if re.search(r"\d", value):
                    numeric[base] = numeric.get(base, 0) + 1
            for base in {" ".join(w for w in label.split() if w != "type") for label in present} - {""}:
                if not re.search(self.LOGISTICS, base):
                    carried[base] = carried.get(base, 0) + 1
        return carried, numeric, spellings, distinct

    def core_labels(self) -> list[str]:
        """Labels most opened sheets carry (at least half, two at least), neither settled nor asked, word-valued first; the
        hidden needs sit among these."""
        carried, numeric, spellings, distinct = self.label_table_values()
        floor = max(2, (len(self.sheets) + 1) // 2)
        settled = [set(label_words(t)) - {"type"} for t in [p for p, _ in self.demands()] + [a[2:].split(": asked", 1)[0] for a in self.asked] + list(self.free)]
        def taken(words: set[str]) -> bool:
            return any(t and (words <= t or t <= words) for t in settled)
        ranked = sorted(carried, key=lambda k: (numeric.get(k, 0) * 2 > carried[k], len(distinct.get(k, ())) < 2, -carried[k]))
        return [max(spellings[b], key=lambda f: (spellings[b][f], "type" in f.split())) for b in ranked if carried[b] >= floor and not taken(set(b.split()))][:8]

    def keep_core(self, planned: list[dict[str, Any]], urgent: bool) -> list[dict[str, Any]]:
        """Each planned question about a core label keeps its wording; one about anything else gives way to the next core
        label (what must be learned first, such as a missing budget, keeps its place)."""
        core = self.core_labels()
        if not core:
            return planned
        hinted = [set(label_words(p)) - {"type"} for p, _ in self.hints()]
        labels: list[str | None] = []
        texts: list[set[str]] = []
        for index, item in enumerate(planned):
            text = set(label_words(str(item.get("property") or "") + " " + str(item.get("question") or "")))
            texts.append(text)
            label = None if urgent and index == 0 else next((c for c in core if c not in labels and set(c.split()) - {"type"} <= text), None)
            labels.append(label)
        out: list[dict[str, Any]] = []
        for index, (item, label) in enumerate(zip(planned, labels)):
            if urgent and index == 0 or any(h and h <= texts[index] for h in hinted):
                out.append(item)
                continue
            if label is None:
                prop = lowered(item.get("property"))
                if prop and sum(1 for sheet in self.sheets.values() if any(lowered(re.sub(r"[_.]+", " ", str(k))) == prop for k in self.flat_sheet(sheet))) >= 2:
                    out.append(item)                # a property the opened sheets carry is the planner's choice: kept
                    continue
                taken = set().union(*texts) if texts else set()
                label = next((c for c in core if c not in labels and not set(c.split()) <= taken), None)
                if label is None:
                    continue
                labels[index] = label
                item = {"property": label, "product": item.get("product") or "", "question": ""}
            out.append(item)
        return out

    def shelf_candidates(self) -> list[str]:
        """Rows never read, affordable, in the main category: those whose title or row words carry the most stated values
        first, then the cheaper."""
        group, wanted = self.main_group(), [v for _, v in self.demands() if bound_of(v) is None]
        def rank(ident: str) -> tuple[int, float]:
            row = self.rows[ident]
            words = " ".join(str(v) for v in row.values() if isinstance(v, str))
            price = _price(row)
            return (-sum(1 for v in wanted if states_value(v, words)), price if price is not None else float("inf"))
        pool = [i for i, r in self.rows.items() if i not in self.sheets and self.affordable(r) and not (group and _shelf(r) and _shelf(r) != group)]
        return [f"{i} ({_price(self.rows[i]):g})" if _price(self.rows[i]) is not None else i for i in sorted(pool, key=rank)[:5]]

    def judge(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None) -> dict[str, Any] | None:
        if self.llm_dead or self.spent() >= self.spend_caps()[2]:
            return None
        light = [READ_MODEL, JUDGE_MODEL] if self.light_checks() else [JUDGE_MODEL, MAIN_MODEL]
        chosen = models or light
        if self.spend_level() >= 1:
            chosen = [READ_MODEL]                   # past the soft line only the light model reads
        max_tokens = max(max_tokens, JUDGE_TOKENS)  # a reading model thinks before it answers: a short cap ends it before the answer
        key = "|".join(chosen) + "|" + str(max_tokens) + "|" + prompt
        if key in self.judge_cache:
            return self.judge_cache[key]
        message = self.infer([{"role": "user", "content": prompt}], None, chosen, max_tokens)
        result = _parse_json_object((message or {}).get("content") or "")
        if message is not None:
            self.judge_cache[key] = result
        return result

    @staticmethod
    def merge_new(into: list[str], values: Any) -> list[str]:
        """Append the strings not yet held; return the ones appended."""
        added = []
        for value in values or []:
            if isinstance(value, str) and value and value not in into:
                into.append(value)
                added.append(value)
        return added

    def hearing_prompt(self, fresh: list[str], question: str | None) -> str:
        parts = [INTERPRET_PROMPT, "THE SHOPPER'S REQUEST (for context):\n" + str(self.policy.get("query") or ""), "REQUIREMENTS SO FAR:\n" + self.ledger()]
        if self.said:
            parts.append("EARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(self.said[-12:]))
        if question:
            parts.append("THE ASSISTANT ASKED:\n" + question)
        parts.append("THE SHOPPER NOW SAID:\n" + "\n".join(fresh))
        return "\n\n".join(parts)

    def apply_reading(self, result: dict[str, Any]) -> list[str]:
        """Fold one reading of the shopper's words into the requirements, budget and freed properties; the notes say what
        changed (an empty list when nothing did)."""
        for ours, other in (("needs", "adds"), ("no_preference", "frees"), ("dropped", "drops"), ("new_budget", "limit"), ("unavailable", "sold_out")):
            if result.get(ours) is None and result.get(other) not in (None, [], ""):
                result[ours] = result[other]         # a reading that names the keys in other words still counts
        now = result.get("requirements_now")
        if isinstance(now, list) and all(isinstance(r, str) for r in now):
            # a requirement the reader re-derived that only restates the product's own name (a season or an audience that
            # is part of the category's name) is covered by the category: it is kept only when it was stated or heard
            kind = {stem(w) for w in label_words(self.plan.get("product") or "")}
            heard = {lowered(x) for x in list(self.stated) + list(self.revealed) + [n for n in result.get("needs") or [] if isinstance(n, str)]}
            def restates(req: str) -> bool:
                words = {stem(w) for w in word_list(str(req).partition(":")[2]) if w not in SMALL_WORDS}
                return bool(words) and bool(kind) and words <= kind and lowered(req) not in heard
            self.current = [r for r in now if r.strip() and not restates(r)]
        notes = [f"new requirement: {n}" for n in self.merge_new(self.revealed, result.get("needs"))]
        self.merge_new(self.free, result.get("no_preference"))
        newly_dropped = self.merge_new(self.dropped, result.get("dropped"))
        notes += [f"dropped: {d}" for d in newly_dropped]
        if newly_dropped:
            self.relaxed_at = self.step
            fresh_needs = {str(n) for n in result.get("needs") or []}
            gone_props = [{stem(w) for w in word_list(str(d).partition(":")[0])} for d in newly_dropped if ":" not in str(d)]
            if self.current is not None and gone_props:
                # a drop named by its property alone ("bundle size can go") lets every requirement on that property go
                self.current = [r for r in self.current if r in fresh_needs or {stem(w) for w in word_list(str(r).partition(":")[0])} not in gone_props]
        budget = result.get("new_budget")
        priced = isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0
        if priced and budget != self.budget_now:
            notes += self.apply_budget(float(budget), "the reader")
        self.flags = {k: bool(result.get(k)) for k in ("unavailable", "price_moved", "unclear", "one_at_a_time")}
        if result.get("unclear") or result.get("one_at_a_time"):
            self.flags["muddled"] = True
        if result.get("unavailable") and self.cart_ids and self.cart_used:
            notes.append("an item became unavailable; remove it from the cart and do not order it")
            said = " ".join([self.heard_now, self.refused_text] + [str(x) for x in result.get("unavailable_items") or []])
            named = [i for i in self.cart_ids if i in said and i not in self.gone]
            if not named and len(self.cart_ids) == 1 and self.cart_ids[0] not in self.gone:
                named = list(self.cart_ids)
            if named:
                self.gone.update(named)
                self.change_reason = ", ".join(named) + " can no longer be bought"
                self.changes += 1
            elif not any(i in self.gone for i in self.cart_ids):
                self.lines_unsure = True                # which line is gone is learned by re-reading the held lines
                self.bundle_unsure = bool(self.bundle_names)
                self.changes += 1
        for need in list(result.get("needs") or []) + list(now if isinstance(now, list) else []):
            if isinstance(need, str):
                self.note_pointer(need)
        if self.current is not None:
            self.current = [self.pointer_form(r) for r in self.current if not self.describes_pointer(r)]
        self.revealed = [self.pointer_form(r) for r in self.revealed if not self.describes_pointer(r)]
        if result.get("price_moved"):
            notes.append("a price changed: re-check that price times quantity stays within the limit and, when the cheapest is wanted, that the pick is still the cheapest fit")
        for need in [n for n in result.get("needs") or [] if isinstance(n, str) and ":" in n]:
            stating, _ = self.sheets_stating(need)
            if stating:
                notes.append(f"opened listings that state {need.strip()}: " + ", ".join(stating[:6]))
            elif self.sheets:
                notes.append(f"no opened listing states {need.strip()} yet: search for it")
        if (bool(result.get("dropped")) or priced) and self.plan.get("priority") == "cheapest":
            notes.append("the set of fitting items has changed: search again (items excluded before may fit now) and re-pick the cheapest that fits before ordering")
        for duty in result.get("duties") or []:
            if isinstance(duty, dict) and self.add_duty(duty.get("do"), duty.get("when"), duty.get("tell")):
                notes.append(f"new instruction: {duty.get('do')}")
        notes += self.take_allowance(result.get("messages_left"), result.get("steps_left"))
        if self.cart_ids and any(n.startswith(("new requirement", "dropped", "new budget", "an item became", "a price changed")) for n in notes):
            self.change_reason = "; ".join(notes[:3])
            self.changes += 1
        if any(n.startswith("a price changed") for n in notes):
            self.market_changed_at = self.step      # prices moved: what was read before is stale
        return notes

    def take_allowance(self, messages: Any, steps: Any) -> list[str]:
        """A limit announced while working binds from now on: N more messages or N more steps."""
        notes = []
        if isinstance(messages, (int, float)) and not isinstance(messages, bool) and messages >= 0:
            cap = self.messages_used + int(messages)
            if self.message_budget is None or cap < self.message_budget:
                self.message_budget = cap
                notes.append(f"messages allowed from now: {int(messages)}")
        if isinstance(steps, (int, float)) and not isinstance(steps, bool) and steps > 0:
            cap = self.step + int(steps)
            if cap < self.max_steps:
                self.max_steps = cap
                notes.append(f"steps allowed from now: {int(steps)}")
        return notes

    def notice_texts(self, observation: Any) -> list[str]:
        """Free text nested anywhere in a result outside the listing records (a store or shopper note under any key, at any
        depth): sentences of four words or more, never a listing's own text and never one heard before. The reader reads
        them for their meaning."""
        found: list[str] = []
        stack = [(observation, 0)]
        while stack:
            value, depth = stack.pop()
            if depth > 5:
                continue
            if isinstance(value, dict):
                listing = bool(self.id_in(value))     # a listing's own text is not a notice; a list of lines it carries may be
                for key, inner in value.items():
                    if titled(key) or grouped(key):
                        continue
                    if listing and not (isinstance(inner, list) and inner and all(isinstance(x, str) for x in inner)):
                        continue
                    if isinstance(inner, str) and len(inner.split()) >= 4 and not (money_value(inner) is not None and len(inner.split()) <= 5):
                        found.append(inner.strip())
                    elif isinstance(inner, (dict, list)):
                        stack.append((inner, depth + 1))
            elif isinstance(value, list):
                for inner in value[:40]:
                    if isinstance(inner, str) and len(inner.split()) >= 4 and not (money_value(inner) is not None and len(inner.split()) <= 5):
                        found.append(inner.strip())
                    elif isinstance(inner, (dict, list)):
                        stack.append((inner, depth + 1))
        return [t for t in dict.fromkeys(found) if t not in self.heard][:4]

    def interpret(self, lines: list[str], question: str | None) -> str:
        """The shopper's new lines read once each by a reader model; returns the meaning and what changed, for the note."""
        fresh = [line for line in lines if line not in self.heard]
        if not fresh:
            return ""
        self.heard.update(fresh)
        self.heard_now = " ".join(fresh)
        self.flags = {}
        if self.time_left() < WRAP_UP_SECONDS:
            return ""
        asked = re.search(r"(?i)what\s+(.+?)\s+would you like", question or "")
        for line in fresh:                      # a reply that points at another listing's page is kept as a pointer in its own words
            prop = asked.group(1).strip().lower() if asked else ""
            if len(self.last_keys) > 1:         # a question naming several keys: the pointer belongs to the key the reply names
                words = {stem(w) for w in word_list(line)}
                prop = next((k for k in self.last_keys if self.key_words(k) and self.key_words(k) <= words), "")
            if prop and re.search(r"(?i)\bthe\s+(?:page|listing|sheet)\b", line):
                self.note_pointer(prop + ": " + re.sub(r"(?i)^.*?\b(?:on|at|sitting at|landing on)\s+(?=(?:definitely\s+)?not\b|whatever|the value|anything)", "", line))
        prompt = self.hearing_prompt(fresh, question)
        self.said.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        result = self.judge(prompt, 1500, [READ_MODEL, JUDGE_MODEL])
        if not result:
            fallback = self.budget_by_unit(fresh, question) + self.drops_in(fresh, question)
            return ("Read by the agent: " + "; ".join(fallback)) if fallback else ""
        notes = self.apply_reading(result)
        self.question_flags[question or ""] = dict(self.flags)
        budget = result.get("new_budget")
        if not (isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0) and not result.get("price_moved") and not result.get("unavailable"):
            notes += self.budget_by_unit(fresh, question)
        if notes:
            before = self.ledger_version
            self.ledger_version += 1
            if all(n.startswith(("new budget", "a price changed")) for n in notes):
                for key, verdict in list(self.verdicts.items()):
                    item, _, version = key.rpartition("|")
                    if version == str(before):
                        self.verdicts[f"{item}|{self.ledger_version}"] = verdict   # money moved, no facts did
        reply = str(result.get("reply") or "").strip()
        if reply and self.roles.get("message") and any("?" in line for line in fresh) and any(d["tell"] and not d["done"] for d in self.duties):
            self.pending_reply = reply              # only a reply the request's own steps call for; never a re-explanation
        return "Meaning: " + str(result.get("meaning") or "") + (" (" + "; ".join(notes) + ")" if notes else "")

    def budget_by_unit(self, lines: list[str], question: str | None) -> list[str]:
        """The structural reading of a budget note: a shopper line outside an answer to a question that writes exactly one
        figure in the request's money unit sets the new limit, whatever the words around it."""
        if question is not None:
            return []
        units = money_unit_words(str(self.policy.get("query") or ""), self.plan.get("currency"))
        for line in lines:
            amount = lone_amount(line, units)
            if amount is not None and amount > 0:
                return self.apply_budget(amount, "the figure in the shopper's line")
        return []

    # ---------- changes read by the agent itself (no model turn) ----------
    def hear_in(self, lines: list[str]) -> list[str]:
        """The shopper's lines are read for their meaning by the reader that follows (with a structural fallback when it
        fails); nothing is read off their wording here."""
        return []

    def apply_budget(self, amount: float, source: str) -> list[str]:
        """A new total limit from the shopper: the ledger takes it, the requirement version moves on, and every verdict
        already passed carries over to the new version for the listings still affordable (a budget moves no facts)."""
        old = self.budget_now
        if isinstance(old, (int, float)) and not isinstance(old, bool) and abs(float(old) - amount) < 1e-9:
            return []
        before = self.ledger_version
        self.budget_now = amount
        self.budget_moved_at = self.step
        self.ledger_version += 1
        quantity = self.quantity_wanted()
        for key, verdict in list(self.verdicts.items()):
            item, _, version = key.rpartition("|")
            if version != str(before):
                continue
            price = self.latest_price(item)
            if price is None or price * quantity <= amount + 1e-9:
                self.verdicts[f"{item}|{self.ledger_version}"] = verdict
        if self.cart_ids:
            self.changes += 1
            self.change_reason = f"the budget is now {amount:g}" + (f" (was {old})" if old is not None else "")
        self.trace.append({"kind": "budget", "amount": amount, "was": old, "source": source, "step": self.step})
        direction = "up" if isinstance(old, (int, float)) and amount > float(old) else "down" if isinstance(old, (int, float)) else "set"
        return [f"new budget: {amount:g} ({direction}; read from {source})"]

    def post_add_look(self) -> str:
        """An order sent on the very call after the first add: the store hands the shopper's changes (a dropped requirement,
        a new budget) over on that next call, so the agent reads the cart once first. When the look brought a change, the
        cart is re-ranked and the model is told; else the order goes on. Returns the reply ("" when the order may go on)."""
        self.add_looked = True
        view = self.roles.get("cart_view")
        if not view or view not in self.schemas or self.bundle_names or self.companions:
            return ""
        changes, said = self.changes, len(self.said)
        if self.agent_action(view, self.neutral_args(view, ""), "Reading the cart before the order: the store's news about a new line comes with the next result.", "post_add") is None:
            return ""
        if self.changes == changes and len(self.said) == said:
            return ""
        reranked = self.rerank_after_note()
        if self.finished or self.dead:
            return reranked
        return ("Not run yet (one step used to read the cart). The shopper's words came with it: " + (self.said[-1][:300] if self.said else "")
                + ". " + (reranked or "Re-check the held line against the requirements and the budget now in force (the request's rule may now rank another opened listing first), then order."))

    def rerank_after_note(self) -> str:
        """Right after a note that moved the budget or dropped a requirement while one line is held: every opened sheet is
        judged again under the requirements and limit now in force, and when the request's rule (cheapest, or best by its
        measure) ranks another fit above the held line, or the held line no longer fits, the agent swaps to that fit and
        orders it. Never on a price or stock note alone (the held line's facts did not change), never on bundles.
        Returns the note ("" when nothing was done)."""
        if self.bundle_names or self.companions or len(self.cart_ids) != 1 or self.step not in (self.relaxed_at, self.budget_moved_at):
            return ""
        if self.reranked_at == self.step or self.spend_level() == 2 or self.time_left() < WRAP_UP_SECONDS / 2:
            return ""
        self.reranked_at = self.step
        cart = self.roles.get("cart")
        current = self.cart_ids[0]
        fits = [(i, p) for i, p in self.fits_open() if not self.shunned(i)]
        if not fits or not cart:
            return ""
        rule = self.plan.get("priority")
        if rule == "cheapest" and self.step == self.relaxed_at and not self.measure_words():
            # a dropped requirement lets in listings it had excluded, and those were seldom opened: the cheapest unread
            # rows below the cheapest fit now known are read before the swap is decided
            floor = min([p for _, p in fits if p is not None] + [self.latest_price(current) or float("inf")])
            unread = [i for _, i in self.below(floor, self.main_group())]
            room = min(REFILL_STEPS, self.steps_left() - self.finish_reserve(swap=True) - 4)
            if unread and room > 0 and self.time_left() > WRAP_UP_SECONDS:
                for call in self.open_calls(unread[:room])[:room]:
                    if self.finished or self.dead or self.steps_left() <= self.finish_reserve(swap=True) + 3:
                        break
                    self.own_round([call], "A requirement was dropped: opening the cheaper listings it had excluded before choosing again.")
                fits = [(i, p) for i, p in self.fits_open() if not self.shunned(i)] or fits
                if self.finished or self.dead or current not in self.cart_ids:
                    return ""
        holds = current in dict(fits)
        if holds and rule not in ("cheapest", "best"):
            return ""
        ranked = self.ranked_by_rule(fits) if (rule == "best" and self.measure_words()) else fits
        best = ranked[0][0] if rule in ("cheapest", "best") else self.preferred(fits)
        if best == current:
            return ""
        if rule == "cheapest" and self.step == self.relaxed_at:
            # a listing that states the requirements only in its title or description may be the cheapest fit now: with
            # such a cheaper one in sight the swap is not decided in code, the model weighs it
            loose = [p for i, p in self.fits_by_facts(strict=False) if p is not None and i not in self.gone and not self.shunned(i) and i != best]
            if loose and min(loose) < (self.latest_price(best) or 0.0) - 1e-9:
                return ""
        if holds and rule == "cheapest" and (self.latest_price(best) or 0.0) >= (self.latest_price(current) or 0.0) - 1e-9:
            return ""
        if self.steps_left() < self.finish_reserve(swap=False) + 2:
            return ""
        removal = self.remove_args(current)
        if removal is None or (not self.basket_spec(cart) and self.agent_action(removal[0], removal[1], f"After the shopper's note, {best} ranks above {current} under the requirements and limit now in force: swapping.", "rerank") is None):
            return ""
        self.cart_ids = [i for i in self.cart_ids if i != current]
        if self.agent_action(cart, self.cart_args(best), f"{best} is the fit the request's rule ranks first now: putting it in.", "rerank") is None or best not in self.cart_ids:
            return f"The agent took {current} out; put {best} in and order it."
        self.agent_placed.add(best)
        self.repaired_by_code = True
        price = self.latest_price(best)
        why = f"After the shopper's note, {best}" + (f" ({price:g})" if price is not None else "") + f" ranks first among the listings that fit everything now in force, so it replaced {current}."
        ordered = self.order_now(why) if self.questions_done() else ""
        return why + (" " + ordered if ordered else " Order it now.")

    def after_declined(self) -> str:
        """A decline answered with the shopper's words (a dropped requirement, a new budget) instead of ending: when a listing
        now fits every requirement in force, the agent puts the one the request's rule ranks first in the cart (taking out a
        line that fits less) and orders it. Returns the note ("" when nothing was done)."""
        cart = self.roles.get("cart")
        fits = [(i, p) for i, p in self.fits_open() if not self.shunned(i)]
        if not fits or not cart or self.auto_orders >= AUTO_ORDERS:
            return ""
        best = self.preferred(self.ranked_by_rule(fits) if self.measure_words() else fits)
        need = 1 + sum(1 for i in self.cart_ids if i != best) + (0 if best in self.cart_ids else 1)
        if self.steps_left() < need + 1:
            return ""
        for other in [i for i in self.cart_ids if i != best]:
            removal = self.remove_args(other)
            if removal is None or self.agent_action(removal[0], removal[1], f"Taking {other} out: {best} fits the requirements now in force.", "after_decline") is None:
                return ""
            self.cart_ids = [i for i in self.cart_ids if i != other]
        if best not in self.cart_ids:
            if self.agent_action(cart, self.cart_args(best), f"{best} fits every requirement now in force: putting it in the cart.", "after_decline") is None or best not in self.cart_ids:
                return ""
        self.agent_placed.add(best)
        self.hold_unfit = ""
        self.repaired_by_code = True
        return self.order_now(f"The shopper's words at the decline changed the requirements or the budget, and {best} now states every requirement in force within the limit.")

    def stock_reread(self) -> None:
        self.reread_lines()

    def reread_lines(self) -> None:
        """The held lines' pages are read again after a refusal that names no figure or names no line; a line whose own
        stock flag reads out of stock is gone (a state, not a word list)."""
        tool = self.page_tool()
        for pick in list(self.cart_ids):
            if not tool or pick in self.reread_after or pick in self.gone or self.steps_left() < 3 or self.time_left() < 12:
                return
            self.reread_after.add(pick)
            calls = self.open_calls([pick], tool)
            if not calls:
                return
            self.own_round(calls, f"Re-reading {pick} after the refused order: its stock flag shows whether it can still be bought.")
            if self.page_stock.get(pick) is False and pick not in self.gone:
                self.gone.add(pick)
                self.change_reason = f"{pick} can no longer be bought (its page shows it out of stock)"
                self.changes += 1
        self.lines_unsure = False

    def after_refusal(self) -> str:
        """Right after an order was refused with the cart's one line still buyable: when the refusal moved the budget or a
        price and the line still sits within the limit now in force with its sheet stating every requirement, the agent
        re-sends the order itself; a refusal is not a rejection of the item. Returns the note ("" when nothing was done)."""
        if self.finished or self.dead or self.refusal_at != self.step:
            return ""
        if self.decline_refused_at == self.step and not self.bundle_names and not self.companions:
            return self.after_declined()
        if not self.cart_ids:
            return ""
        moved = self.budget_moved_at == self.step or self.market_changed_at == self.step or self.relaxed_at == self.step
        if not moved and not any(i in self.gone for i in self.cart_ids):
            self.reread_lines()                     # a refusal naming no figure or no line: the held lines' own stock flags say which is gone
            if not any(i in self.gone for i in self.cart_ids):
                return ""
            moved = False
        if self.companions or self.bundle_names:
            # a price move alone leaves every held line acceptable while the total stays within the limit: order again
            total = sum((self.latest_price(i) or 0.0) for i in self.cart_ids) * self.quantity_wanted()
            cap = self.budget_now
            within = not isinstance(cap, (int, float)) or isinstance(cap, bool) or total <= cap + 1e-9
            if (self.market_changed_at == self.step and self.budget_moved_at != self.step and within and not any(i in self.gone for i in self.cart_ids)
                    and not self.bundle_missing() and self.auto_orders < AUTO_ORDERS):
                self.repaired_by_code = True
                self.agent_placed.update(self.cart_ids)
                return self.order_now("A price moved, and every held line still sits within the limit: the same cart is ordered again.")
            return ""
        if len(self.cart_ids) != 1:
            return ""
        pick = self.cart_ids[0]
        if pick in self.gone or not moved:
            return ""
        price = self.latest_price(pick)
        if price is None or not self.affordable(self.sheets.get(pick) or self.rows.get(pick) or {}):
            return ""
        passed = any(k.rpartition("|")[0] == pick and v == "" for k, v in self.verdicts.items())
        # a line the model chose itself, whose own sheet contradicts no requirement now in force, is re-sent as well: a note at
        # checkout changes the money or the wish list, not the line's facts (race 12: swaps and declines there lost the task)
        loose = pick != self.hold_unfit and pick in {i for i, _ in self.fits_by_facts(strict=False)}
        if pick not in dict(self.live_fits()) and not passed and self.verdicts.get(f"{pick}|{self.ledger_version}") != "" and not loose:
            return ""
        if self.relaxed_at == self.step and self.plan.get("priority") in ("cheapest", "best"):
            # a dropped requirement widens the fits: a listing it excluded may now rank before the held line
            fits = [(i, p) for i, p in self.live_fits() if i not in self.gone]
            if pick not in dict(fits):          # the held line passed on the loose reading: compare it on that reading too
                fits = [(i, p) for i, p in self.fits_by_facts(strict=False) if p is not None and i not in self.gone]
            ranked = self.ranked_by_rule(fits) if fits else []
            if ranked and ranked[0][0] != pick and (self.measure_words() or ranked[0][1] < price - 1e-9):
                return ""
        self.agent_placed.add(pick)
        self.repaired_by_code = True
        return self.order_now(f"The refusal moved the budget or the price, and {pick} ({price:g}) still sits within the limit now in force ({self.budget_now}) with its sheet stating every requirement.")

    # ---------- the decline gate ----------
    def measure_words(self) -> list[str]:
        """The property a 'best' rule is measured by, as words (the direction words taken out); empty without such a rule."""
        if self.plan.get("priority") != "best":
            return []
        detail = re.sub(r"\([^)]*\)", " ", str(self.plan.get("priority_detail") or ""))
        return [w for w in label_words(detail) if w not in SMALL_WORDS and not re.fullmatch(r"\w+est|most|least|top|value|measure|best", w)]

    def measure_of(self, ident: str) -> float | None:
        """The first figure a label for the measure states on the listing's own sheet (None when none does)."""
        words = " ".join(self.measure_words())
        kind = set(label_words(self.plan.get("product") or ""))
        for key, value in self.flat_sheet(self.sheets.get(ident) or {}).items():
            if words and not titled(key) and not grouped(key) and label_fits(str(key), words, kind) and figures_in(str(value)):
                return figures_in(str(value))[0][0]
        return None

    def ranked_by_rule(self, fits: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Fits in the order the request's rule prefers: the best stated measure first under a best rule (lowest when the
        detail asks for the least), a listing stating none after; then the cheaper; ties by id."""
        sign = 1.0 if re.search(r"(?i)\b(?:low|small|least|light(?!s)|min|short|thin)", re.sub(r"\([^)]*\)", " ", str(self.plan.get("priority_detail") or ""))) else -1.0
        def key(t: tuple[str, float]) -> tuple[int, float, float, str]:
            m = self.measure_of(t[0]) if self.measure_words() else None
            return (m is None, sign * m if m is not None else 0.0, t[1], t[0])
        return sorted(fits, key=key)

    def fits_open(self) -> list[tuple[str, float]]:
        """Opened listings that still fit everything now: every requirement stated on their own sheet, within the budget in
        force, not shown out of stock, not refused as gone, not contradicted by a named property."""
        return [(i, p) for i, p in self.live_fits() if not self.model_veto(i)]

    # ---------- hold first: a line in the cart before any decline ----------
    def closeness(self, ident: str) -> tuple[Any, ...] | None:
        """How near a listing comes to the request, as a sort key (smaller is nearer): buyable within the limit first, then
        the fewest requirements its own facts contradict, the fewest it leaves unstated, opened before unopened, then the
        request's rule (best measure, else the price). None for a listing that cannot be bought at all."""
        record = self.sheets.get(ident) or self.rows.get(ident)
        if not record or ident in self.gone or _in_stock(record) is False or self.page_stock.get(ident) is False:
            return None
        if not self.bundle_names:
            group, own = self.main_group(), _shelf(record) or _shelf(self.rows.get(ident) or {})
            if group and own and own != group:
                return None
        own = self.product_of(ident) if self.bundle_names else ""
        kind = self.name_words(own) if own else set(label_words(self.plan.get("product") or ""))
        flat = self.flat_sheet(record)
        verdicts = [read_sheet(flat, prop, value, kind)[0] for prop, value in self.own_demands(ident)]
        contradicted = verdicts.count("no") + (1 if self.wanted_codes() and self.code_fit(ident) is False else 0)
        unstated = len(verdicts) - verdicts.count("yes") - verdicts.count("no")
        price = self.latest_price(ident)
        measure = self.measure_of(ident) if self.measure_words() else None
        sign = 1.0 if re.search(r"(?i)\b(?:low|small|least|light(?!s)|min|short|thin)", str(self.plan.get("priority_detail") or "")) else -1.0
        rule = (measure is None, sign * measure if measure is not None else 0.0) if self.plan.get("priority") == "best" else (False, 0.0)
        return (not self.affordable(record), contradicted, self.shunned(ident), unstated, ident not in self.sheets, rule, price is None, price or 0.0, ident)

    def nearest(self) -> list[str]:
        ranked = sorted((key, ident) for ident in list(self.sheets) + [i for i in self.rows if i not in self.sheets] if (key := self.closeness(ident)) is not None)
        return [ident for _, ident in ranked]

    def hold_first(self, by_code: bool = False) -> str:
        """Before a decline with an empty cart: the shopper's changes (a requirement dropped, a budget raised or cut, a stock
        or price move) arrive when a first line goes into the cart, so a decline sent with nothing ever held is a decline made
        before the shopper has had a say. The agent puts the nearest candidate in, reads the words that come back, and when a
        listing fits the requirements as they now stand, keeps or swaps to the one the request's rule ranks first and orders it.
        Once per episode. Returns the note for the model ("" when nothing was done)."""
        cart, order = self.roles.get("cart"), self.roles.get("order")
        if self.held_first or self.cart_used or self.cart_ids or not cart or not order or self.finished or self.dead:
            return ""
        if self.steps_left() < (4 if self.roles.get("cart_view") else 3) or self.time_left() < (8 if by_code else WRAP_UP_SECONDS / 2):
            return ""
        if not by_code and not self.questions_done() and self.steps_left() > 12:
            self.ask_planned(force=True)                # the hidden needs first: the nearest line is chosen with them known
            if self.finished or self.dead:
                return ""
        near = self.nearest()
        if not near:
            return ""
        pick = near[0]
        self.held_first = pick
        if self.agent_action(cart, self.cart_args(pick), f"Nothing read fits every requirement yet; before any decline the nearest candidate {pick} goes into the cart, so the shopper's reaction to a held line is heard.", "hold_first") is None:
            return ""
        if pick not in self.cart_ids:
            return ""
        said_before = len(self.said)
        view = self.roles.get("cart_view")
        if view and view in self.schemas and self.steps_left() >= 3 and not self.finished and not self.dead:
            # the store hands the shopper's words over on the call after the add: one look at the cart collects them
            self.agent_action(view, self.neutral_args(view, ""), "Reading the cart: the shopper's reaction to the held line comes with the next result.", "hold_first")
        heard = self.said[-1] if len(self.said) > said_before or self.said else ""
        fits = [(i, p) for i, p in self.fits_open() if not self.shunned(i)]
        if by_code and not fits:
            fits = [(i, p) for i, p in (self.fits_by_facts(strict=False) or []) if p is not None and i not in self.gone]
        if not fits:
            self.hold_unfit = pick
            return (f"Not run (the decline was not sent). The agent put the nearest candidate {pick} in the cart first"
                    + (f"; the store answered with the shopper's words ({heard[:300]})" if heard else "")
                    + ". No opened listing meets every requirement as they now stand. Search for listings that meet the requirements now in force (any the shopper dropped no longer count) and open them; "
                    + f"order a fit (take {pick} out first unless it fits), or decline if none exists.")
        ranked = self.ranked_by_rule(fits) if self.measure_words() else fits
        best = self.preferred(ranked)
        steps = []
        if best != pick:
            removal = self.remove_args(pick)
            if removal is None or self.steps_left() < 3 or self.agent_action(removal[0], removal[1], f"{best} fits the requirements now in force; taking {pick} out.", "hold_first") is None:
                self.hold_unfit = pick
                return f"Not run (the decline was not sent). After the agent held {pick}, {best} fits every requirement now in force: take {pick} out, put {best} in and order it."
            self.cart_ids = [i for i in self.cart_ids if i != pick]
            if self.agent_action(cart, self.cart_args(best), f"{best} fits every requirement now in force: putting it in the cart.", "hold_first") is None or best not in self.cart_ids:
                return f"Not run (the decline was not sent). {best} fits every requirement now in force: put it in the cart and order it."
            steps.append(f"took {pick} out and put {best} in")
        self.hold_unfit = ""
        self.agent_placed.add(best)
        price = self.latest_price(best)
        why = (f"After the shopper's words at the first add, {best}" + (f" ({price:g})" if price is not None else "")
               + " states every requirement now in force on its own sheet and is the one the request's rule ranks first.")
        if by_code:
            return why
        ordered = self.order_now(why)
        return ("Not run (the decline was not sent). The agent held " + pick + (" and " + "; ".join(steps) if steps else "") + ". " + why + " " + (ordered or "Order it now."))

    def drops_in(self, lines: list[str], question: str | None) -> list[str]:
        """The structural reading of a relaxation note when no reader answered: a shopper line outside an answer that names
        the property of exactly one requirement in force, without its value and without a figure, lets that requirement go."""
        if question is not None:
            return []
        notes: list[str] = []
        for line in lines:
            if re.search(r"\d", line) or "?" in line:
                continue
            words = {stem(w) for w in word_list(line)}
            hits = []
            for requirement in self.in_force():
                prop, sep, value = str(requirement).partition(":")
                named = {stem(w) for w in word_list(prop)} - set(SMALL_WORDS)
                valued = {stem(w) for w in word_list(value)} - named - set(SMALL_WORDS)
                if sep and named and named <= words and not (valued and valued <= words):
                    hits.append(requirement)
            if len(hits) == 1:
                notes += self.drop_requirement(hits[0], "the shopper's line")
        return notes

    def drop_requirement(self, requirement: str, source: str) -> list[str]:
        if requirement in self.dropped:
            return []
        self.dropped.append(requirement)
        if self.current is not None:
            self.current = [r for r in self.current if r != requirement]
        self.ledger_version += 1
        self.relaxed_at = self.step
        if self.cart_ids:
            self.changes += 1
            self.change_reason = f"the shopper no longer requires {requirement}"
        self.trace.append({"kind": "drop", "requirement": requirement, "source": source, "step": self.step})
        return [f"dropped: {requirement} (read from {source})"]

    def decline_gate(self) -> str:
        """A decline is never sent while an opened listing fits: the first time the model is told which one; from the second
        time, after a refusal, or in the last steps, the agent carts that listing and orders it itself. With no fit open,
        one search of the whole category under the budget (and the cheapest unread rows) is made before the decline may
        stand. Returns the reply for the model ("" when the decline may go on)."""
        if self.finished or self.dead:
            return ""
        fits = self.fits_open() if self.parse_trusted() else []   # a parse that misreads the store never overrides the model's decline
        if not fits:
            looked = self.budget_search()
            fits = self.fits_open() if looked and self.parse_trusted() else []
            if looked and not fits:
                return "Not run yet (steps used by the agent). " + looked + " Decline only if each listing still fails; the agent holds no fit now."
            if not fits:
                return ""
        self.decline_blocks += 1
        best = self.preferred(fits)
        price = self.latest_price(best)
        kind = set(label_words(self.plan.get("product") or ""))
        shown = "; ".join(read_sheet(self.flat_sheet(self.sheets.get(best) or {}), prop, value, kind)[1] for prop, value in self.demands()[:3])
        blocked = self.refusal_at >= 0 and self.step - self.refusal_at < ABSTAIN_BLOCK
        left = self.steps_left()
        cart, order = self.roles.get("cart"), self.roles.get("order")
        settled = self.questions_done() or self.messages_used >= self.message_limit() or not self.roles.get("message")
        if self.decline_blocks >= 2 and not settled and not blocked and left > 4:
            return ""                               # the hidden needs were never asked for: the model's decline may be right
        if (self.decline_blocks >= 2 or blocked or left <= 4) and cart and order and not self.gate_ordered:
            self.gate_ordered = True
            need = (0 if best in self.cart_ids else 1) + sum(1 for i in self.cart_ids if i != best) + 1
            if left >= need:
                for other in [i for i in self.cart_ids if i != best]:
                    removal = self.remove_args(other)
                    if removal is None or self.agent_action(removal[0], removal[1], f"Taking {other} out so the fit can be ordered.", "gate") is None:
                        break
                    self.cart_ids = [i for i in self.cart_ids if i != other]
                if best not in self.cart_ids and self.cart_ids == []:
                    self.agent_action(cart, self.cart_args(best), f"{best} states every requirement in force; putting it in the cart instead of declining.", "gate")
                if self.cart_ids == [best]:
                    self.agent_placed.add(best)
                    ordered = self.order_now(f"A decline scores 0 while {best} ({price:g}) fits everything in force.", force=left <= 3)
                    return ("Not run (the decline was not sent): " + best + (f" ({price:g})" if price is not None else "") + " states every requirement on its own sheet (" + shown
                            + ") within the budget, so the agent carted it. " + (ordered or "Order it now."))
        return (f"Not run (no step used): a decline scores 0 while a listing that fits is open. {best}" + (f" ({price:g})" if price is not None else "")
                + f" states every requirement in force on its own sheet ({shown}), sits within the budget" + (f" ({self.budget_now})" if self.budget_now is not None else "")
                + " and is not shown out of stock. Put it in the cart and order it. " + ONE_FIELD + " If you send a decline again, the agent will order it for you.")

    def decline_pacing(self) -> str:
        """Declines that come too soon: within three steps of a budget note (the note asks for a new pick, not an end),
        and on a long step cap before DECLINE_MARK of it while a product was never found by a search or a candidate's
        page is unread with a code in play. Returns the reply ("" when the decline may go on to the gate)."""
        if self.relaxed_at >= 0 and self.step - self.relaxed_at < 3 and self.budget_holds < 2 and self.budget_moved_at < self.relaxed_at:
            self.budget_holds += 1
            gate = self.decline_gate() if not self.bundle_names and self.fits_open() else ""
            if gate or self.finished or self.dead:
                return gate or "Not run: the episode is over."
            fits = [f"{i} ({p:g})" for i, p in self.live_fits()[:3]]
            return ("Not run (no step used): the shopper has just dropped a requirement, which widens the set of fits; it asks for a pick, not for an end. "
                    + ("Fits already read under the requirements now in force: " + ", ".join(fits) + ". " if fits else "")
                    + "Search again without the dropped requirement, open the candidates it had excluded, and order the one the request's rule ranks first.")
        if self.budget_moved_at >= 0 and self.step - self.budget_moved_at < 3 and self.budget_holds < 2:
            self.budget_holds += 1
            gate = self.decline_gate() if not self.bundle_names and self.fits_open() else ""
            if gate or self.finished or self.dead:
                return gate or "Not run: the episode is over."
            looked = self.budget_search() if not self.bundle_names else ""
            fits = [f"{i} ({p:g})" for i, p in self.live_fits()[:3]]
            return (f"Not run (no step used): the shopper has just set the budget to {self.budget_now:g}, which asks for a pick under the new limit, not for an end. "
                    + (looked + " " if looked else "")
                    + ("Fits already read under it: " + ", ".join(fits) + ". " if fits else "")
                    + "If the held line is within it, order it again; if it is over, put the cheapest fitting listing under the new limit in the cart and order it; search the product under the new limit when none is known.")
        if self.max_steps < 50 or self.step >= DECLINE_MARK * self.max_steps or self.long_holds >= 3 or self.time_left() < WRAP_UP_SECONDS + 60:
            return ""
        names = self.bundle_names or [str(self.plan.get("product") or "the product")]
        unseen = [n for n in self.bundle_names if not any(self.product_of(i) == n for i in self.rows)]
        unread = self.fitment_unread()[:5]
        thin = not self.bundle_names and self.searches_done < 3
        if not unseen and not unread and not thin:
            return ""
        self.long_holds += 1
        todo = []
        if unseen:
            todo.append("search by its kind (category words, no codes) for " + ", ".join(unseen))
        if unread:
            todo.append("read the pages (descriptions) of " + ", ".join(unread) + " for the requested code")
        if thin:
            todo.append("search " + names[0] + " in other words (its category, its requirements, further pages)")
        return (f"Not run (no step used): only step {self.step} of {self.max_steps} is used; this long task leaves room to find a fit, and a decline scores 0 whenever one exists. First "
                + "; ".join(todo) + ". Decline only when each product truly has no listing that fits.")

    def budget_search(self) -> str:
        """Once per episode, before a decline may stand with no fit open: the agent lists the main category under the budget
        in force and opens its two cheapest unread rows, so a decline rests on the whole shelf and not on the model's
        searches alone. Returns what it found ("" when nothing was done)."""
        if self.budget_search_done or self.time_left() < WRAP_UP_SECONDS + 30:
            return ""
        if self.steps_left() < self.finish_reserve(empty_cart=True) + 4:
            return ""
        shape = self.search_shape()
        if not shape:
            return ""
        self.budget_search_done = True
        props = (self.schemas.get(shape["tool"]) or {}).get("properties") or {}
        args: dict[str, Any] = dict(shape["modes"])
        group = self.main_group()
        if shape["group"] and group:
            args[shape["group"]] = group
        cap = self.budget_now
        if isinstance(cap, (int, float)) and not isinstance(cap, bool):
            whole = "integer" in str((props.get(shape["ceiling"]) or {}).get("type"))
            args[shape["ceiling"]] = int(cap) if whole else float(cap)
        required = (self.schemas.get(shape["tool"]) or {}).get("required") or []
        if shape["words"] and (shape["words"] in required or not (shape["group"] and group)):
            args[shape["words"]] = " ".join(w for w in word_list(self.plan.get("product") or group) if stem(w) not in self.shed_words)[:80] or group
        args, problems = shape_args(self.schemas[shape["tool"]], args, "args")
        if problems:
            return ""
        before = len(self.rows)
        if self.agent_action(shape["tool"], args, "Before any decline: one look at the whole category under the budget in force.", "gate_search") is None:
            return ""
        opened = []
        floor = float(cap) + 1.0 if isinstance(cap, (int, float)) and not isinstance(cap, bool) else float("inf")
        for _, ident in self.below(floor, group)[:2]:
            if self.steps_left() < self.finish_reserve(empty_cart=True) + 2:
                break
            calls = self.open_calls([ident])
            if not calls:
                break
            function = calls[0]["function"]
            if self.agent_action(function["name"], _parse_json_object(function["arguments"]) or {}, f"Opening {ident}, a cheap unread row of the category, before any decline.", "gate_read") is None:
                break
            opened.append(ident)
        fits = self.fits_open()
        if fits:
            return f"The agent listed the category under the budget ({len(self.rows) - before} new rows) and opened {', '.join(opened) or 'nothing new'}: {fits[0][0]} ({fits[0][1]:g}) states every requirement on its own sheet."
        return f"The agent listed the category under the budget ({len(self.rows) - before} new rows) and opened {', '.join(opened) or 'nothing new'}; none of them states every requirement."

    # ---------- needs that point at another listing's page ----------
    def note_pointer(self, need: str) -> bool:
        """A requirement whose value sits on another page is kept as a pointer to resolve; the words that pick out that
        other listing (its shop, its attributes) are never used as search words for the item."""
        pointer = pointer_from_need(need)
        if pointer is None:
            return False
        if any(p["text"] == pointer["text"] or (p["prop"] and p["prop"] == pointer["prop"]) for p in self.pointers):
            return False
        self.pointers.append(pointer)
        kind = {stem(w) for w in word_list(self.plan.get("product") or "")}
        for word in word_list(pointer["shop"]) + [w for a in pointer["attrs"] for w in word_list(a)]:
            if len(word) >= 3 and stem(word) not in kind and stem(word) not in SMALL_WORDS:
                self.shed_words.add(stem(word))
        self.trace.append({"kind": "pointer", "need": pointer["text"], "shop": pointer["shop"], "kind_words": pointer["kind"], "attrs": pointer["attrs"], "excluded": pointer["excluded"]})
        self.stated = [r for r in self.stated if not self.describes_pointer(r)]
        self.revealed = [r for r in self.revealed if not self.describes_pointer(r)]
        if self.current is not None:
            self.current = [r for r in self.current if not self.describes_pointer(r)]
        return True

    def describes_pointer(self, requirement: str) -> bool:
        """A requirement that only describes the listing a pointer need points at (its shop as a brand, or one of the
        attributes that pick it out), read by mistake as a requirement on the item itself."""
        prop, sep, value = str(requirement).partition(":")
        if not sep or pointer_from_need(requirement) is not None:
            return False
        value = re.sub(r"(?i)^\s*(?:not|no|never|anything but|other than|except)\s+", "", value)   # "seller: not <the pointed shop>" still only names that page
        prop_words = {stem(w) for w in word_list(prop)} - {"type"}
        value_words = {stem(w) for w in word_list(value)} - set(SMALL_WORDS)
        if not value_words:
            return False
        for pointer in self.pointers:
            shop_words = {stem(w) for w in word_list(pointer["shop"])}
            if shop_words and value_words <= shop_words:
                return True
            if re.search(r"(?i)brand|seller|shop|maker|vendor|store|sold by", prop) and pointer["shop"] and states_value(pointer["shop"], value):
                return True
            for attr in pointer["attrs"]:
                attr_words = {stem(w) for w in word_list(attr, True)} | {stem(w) for w in word_list(attr)}
                mine = {stem(w) for w in word_list(prop + " " + value, True)} | prop_words | value_words
                if prop_words and prop_words <= attr_words and value_words <= attr_words:
                    return True
                if mine and mine <= attr_words:
                    return True
        return False

    def pointer_form(self, requirement: str) -> str:
        """A requirement written as a pointer that has since been resolved reads as its plain value (or exclusion)."""
        pointer = pointer_from_need(requirement)
        if pointer is None:
            return requirement
        for done in self.pointers:
            if done["done"] and done["value"] and (done["shop"] == pointer["shop"] or (done["prop"] and done["prop"] == pointer["prop"])):
                return self.resolved_need(done)
        return requirement

    @staticmethod
    def resolved_need(pointer: dict[str, Any]) -> str:
        """The plain requirement a resolved pointer stands for: the page's value, its exclusion, the band around its figure
        the shopper allows, or the side of it."""
        prop = pointer["prop"] or "value"
        value = str(pointer["value"])
        figure = re.search(r"(-?\d+(?:\.\d+)?)\s*([^\d\s,;]*)", value)
        if figure and not pointer["excluded"] and (pointer.get("band") or pointer.get("side")):
            number, unit = float(figure.group(1)), figure.group(2)
            if pointer.get("band"):
                width, band_unit = pointer["band"]
                unit = unit or band_unit
                return f"{prop}: between {round(number - width, 4):g} and {round(number + width, 4):g} {unit}".strip()
            return f"{prop}: {'at least' if pointer['side'] == 'min' else 'at most'} {number:g} {unit}".strip()
        return f"{prop}: {'not ' if pointer['excluded'] else ''}{value}"

    def pointer_rows(self, pointer: dict[str, Any]) -> list[str]:
        """Known listings that may be the page a need points at: any text field naming the shop, ranked by how many of the
        attributes their text already states; never one already opened for this pointer."""
        shop = lowered(pointer["shop"])
        ranked = []
        for ident in dict.fromkeys([*self.rows, *self.sheets]):
            if ident in pointer["opened"]:
                continue
            record = dict(self.rows.get(ident) or {})
            record.update(self.flat_sheet(self.sheets.get(ident) or {}))
            text = " ".join(str(v) for k, v in record.items() if isinstance(v, str) and not _names_id(k, self.id_keys()))
            words = set(word_list(text))
            if shop and not (shop in words or any(kin(shop, w) for w in words if len(w) >= 4)):
                continue
            stated = sum(1 for a in pointer["attrs"] if states_value(a, text))
            ranked.append((-stated, ident not in self.sheets, ident))
        return [i for _, _, i in sorted(ranked)]

    def attr_on_page(self, flat: dict[str, Any], attr: str, text: str) -> str:
        """How a page answers one attribute of the pointer ("water resistance 100 m"): 'yes' or 'no' by the label that
        opens the attribute's words when the page carries one (its value must state the rest), else 'yes' when the
        page's whole text states it and 'unshown' otherwise."""
        words = label_words(attr)
        for key, value in flat.items():
            label = label_words(key)
            if label and words[:len(label)] == label and len(words) > len(label):
                rest = " ".join(word_list(attr, True)[len(label):]) or " ".join(words[len(label):])
                return "yes" if states_value(rest, str(value)) or states_value(" ".join(words[len(label):]), str(value)) else "no"
        return "yes" if states_value(attr, text) else "unshown"

    def page_matches(self, ident: str, pointer: dict[str, Any]) -> bool:
        """An opened sheet is the pointed-at page when its own text names the shop and its labels (or, failing a label,
        its text) state every attribute given."""
        record = dict(self.rows.get(ident) or {})
        flat = self.flat_sheet(self.sheets.get(ident) or {})
        record.update(flat)
        text = " ".join(str(v) for k, v in record.items() if isinstance(v, str) and not _names_id(k, self.id_keys()))
        words = set(word_list(text))
        shop = lowered(pointer["shop"])
        named = not shop or shop in words or any(kin(shop, w) for w in words if len(w) >= 4)
        return named and all(self.attr_on_page(flat, a, text) == "yes" for a in pointer["attrs"])

    def page_contradicts(self, ident: str, pointer: dict[str, Any]) -> bool:
        """An opened sheet whose own label states another value for one of the pointer's attributes is not the page."""
        flat = self.flat_sheet(self.sheets.get(ident) or {})
        text = " ".join(str(v) for v in flat.values() if isinstance(v, str))
        return any(self.attr_on_page(flat, a, text) == "no" for a in pointer["attrs"])

    def shop_names(self) -> set[str]:
        """Shop, seller, brand and vendor names seen on any row or sheet, lowercased."""
        names: set[str] = set()
        for record in list(self.rows.values()) + [self.flat_sheet(s) for s in self.sheets.values()]:
            for key, value in record.items():
                if isinstance(value, str) and value.strip() and re.search(r"(?i)seller|brand|vendor|maker|shop|store|merchant", str(key)) and len(value) <= 40:
                    names.add(value.strip().lower())
        return names

    def code_pointer_value(self, pointer: dict[str, Any], page: str) -> tuple[str, str]:
        """What the page itself says, read without a model: the value under the page's own label for the property (for a
        match or an exclusion; a pairing rule needs the reader), else the sentence of the page's remarks that names the
        property together with another shop (the listing to follow). Returns (value, follow)."""
        flat = self.flat_sheet(self.sheets.get(page) or {})
        kind = set(label_words(self.plan.get("product") or ""))
        prop = pointer["prop"]
        prop_words = [w for w in word_list(prop) if w not in ("type",)]
        for sentence in re.split(r"(?<=[.!?;])\s+|\n", self.pages.get(page) or ""):
            low = sentence.lower()                # a remark that sends the property to another product outranks the page's own label
            if prop_words and all(w in low for w in prop_words) and re.search(r"\b(?:taken from|different product|not from this|instead|rather than)\b", low):
                return "", sentence.strip()[:200]
        if prop and not pointer["pairing"]:
            for key, value in flat.items():
                if (isinstance(value, (str, int, float)) and not isinstance(value, bool) and str(value).strip() and not titled(key)
                        and not grouped(key) and label_fits(str(key), prop, kind)
                        and not re.fullmatch(r"(?i)\s*(?:n/?a|none|not specified|unknown|null|-)\s*", str(value))):
                    return str(value).strip()[:80], ""
        others = self.shop_names() - {lowered(pointer["shop"])}
        prop_words = [w for w in word_list(prop) if w not in ("type",)]
        for key, value in flat.items():
            if not isinstance(value, str) or len(value) < 20:
                continue
            for sentence in re.split(r"(?<=[.!?;])\s+", value):
                low = sentence.lower()
                if prop_words and not all(w in low for w in prop_words):
                    continue
                if any(name in low for name in others if len(name) >= 3) or re.search(r"\b(?:use|see|consult|refer|instead|rather|points? to|listing|product)\b", low):
                    return "", sentence.strip()[:200]
        return "", ""

    def pointer_search(self, words: str) -> bool:
        """One worded search for the pointed-at listing, without a price ceiling (that listing may cost anything).
        True when it showed rows."""
        shape = self.search_shape() or {}
        tool = self.roles.get("search")
        if not tool:
            return False
        props = (self.schemas.get(tool) or {}).get("properties") or {}
        slot = shape.get("words") or next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v
                                           and re.search(r"text|query|keyword|term|words|search|\bq\b|phrase|string", k.lower())), None)
        if not slot:
            return False
        args: dict[str, Any] = {slot: words[:80]}
        for key, spec in props.items():
            if isinstance(spec, dict) and spec.get("enum") and key in ((self.schemas.get(tool) or {}).get("required") or []):
                chosen = next((v for v in spec["enum"] if re.search(r"(?i)text|typed|keyword|free|word|query", str(v))), spec["enum"][0])
                args[key] = chosen
        args, problems = shape_args(self.schemas[tool], args, "args")
        if problems:
            return False
        before = len(self.rows)
        observation = self.agent_action(tool, args, f"Looking for the listing a need points at: {words}.", "pointer_search")
        if observation is None:
            return False
        total, shown = self.match_count(observation)
        if not shown and len(self.rows) == before:
            self.pointer_searches += 1
            return False
        return True

    def resolve_pointers(self) -> None:
        """Before the model's turn: every need that points at another page is resolved by the agent while steps allow.
        It finds candidate listings (the shop's rows already seen, else one search for the shop and the kind, then the
        kind alone, at most POINTER_SEARCHES empty ones), opens up to POINTER_OPENS of them, and when a page names the shop
        and states the attributes, reads the value there with one reader call, following one further page when the first
        points on. The need then becomes a plain value or exclusion and the shop's words leave the search vocabulary."""
        for pointer in self.pointers:
            if pointer["done"] or pointer["tries"] >= 2:
                continue
            if self.steps_left() < self.finish_reserve(empty_cart=True) + 5 or self.time_left() < WRAP_UP_SECONDS + 40:
                return
            pointer["tries"] += 1
            page = ""
            for _ in range(2):
                candidates = [c for c in self.pointer_rows(pointer) if not (c in self.sheets and self.page_contradicts(c, pointer))]
                if not candidates and self.pointer_searches < POINTER_SEARCHES and not pointer.get("searched"):
                    pointer["searched"] = True
                    for words in dict.fromkeys([f"{pointer['shop']} {pointer['kind']}".strip(), pointer["kind"].strip()]):
                        if not words or self.pointer_searches >= POINTER_SEARCHES:
                            break
                        if self.pointer_search(words):
                            break
                    candidates = [c for c in self.pointer_rows(pointer) if not (c in self.sheets and self.page_contradicts(c, pointer))]
                for ident in candidates[:POINTER_OPENS]:
                    if self.steps_left() < self.finish_reserve(empty_cart=True) + 3:
                        break
                    if ident not in self.sheets:
                        calls = self.open_calls([ident])
                        if not calls:
                            break
                        function = calls[0]["function"]
                        if self.agent_action(function["name"], _parse_json_object(function["arguments"]) or {}, f"Opening {ident}: it may be the page a need points at.", "pointer_open") is None:
                            break
                    pointer["opened"].append(ident)
                    if ident in self.sheets and self.page_matches(ident, pointer):
                        page = ident
                        break
                if page or pointer.get("searched") or self.pointer_searches >= POINTER_SEARCHES:
                    break
            if not page:
                # every candidate opened and none states every attribute: the one page of the shop that contradicts none is taken
                loose = [c for c in pointer["opened"] if c in self.sheets and not self.page_contradicts(c, pointer) and self.page_matches(c, dict(pointer, attrs=[]))]
                page = loose[0] if len(loose) == 1 and (pointer.get("searched") or self.pointer_searches >= POINTER_SEARCHES) else ""
                if not page:
                    continue
            self.read_pointer_page(pointer, page, depth=0)

    def read_pointer_page(self, pointer: dict[str, Any], page: str, depth: int) -> None:
        """One reader call turns the pointed-at page into the concrete value the need requires (or excludes); a page that
        names a further listing is followed once."""
        pointer["page"] = page
        value, follow = self.code_pointer_value(pointer, page)
        result: dict[str, Any] | None = {"value": value, "follow": follow, "by": "code"}
        if (not value and not follow) or pointer["pairing"] or (value and pointer["excluded"] is False and re.search(r"(?i)pair|goes with|alongside|suit", pointer["text"])):
            record = dict(self.rows.get(page) or {})
            record.update(self.sheets.get(page) or {})
            prompt = (POINTER_PROMPT + "\n\nTHE REQUIREMENT IN THE SHOPPER'S WORDS:\n" + pointer["text"] + f"\nTHE PROPERTY WANTED: {pointer['prop'] or '(the one the requirement names)'}"
                      + "\n\nTHE PAGE FOUND (" + page + "):\n" + _compact(_brief(record, 600))[:5000])
            read = self.judge(prompt, 400, [READ_MODEL, JUDGE_MODEL])
            if isinstance(read, dict) and (str(read.get("value") or "").strip() or str(read.get("follow") or "").strip()):
                result = read
            elif not value and not follow:
                result = None
        self.trace.append({"kind": "pointer_read", "page": page, "result": result})
        if not isinstance(result, dict):
            return
        value = str(result.get("value") or "").strip()
        follow = str(result.get("follow") or "").strip()
        if value:
            pointer["value"] = value[:80]
            pointer["excluded"] = pointer["excluded"] or result.get("excluded") is True
            pointer["done"] = True
            self.apply_pointer(pointer)
            return
        if follow and depth < 3 and self.steps_left() >= self.finish_reserve(empty_cart=True) + 3:
            target = follow if follow in self.rows or follow in self.sheets else ""
            def best_match() -> str:
                wanted = {stem(w) for w in word_list(follow)} - set(SMALL_WORDS) - {stem(w) for w in word_list(pointer["prop"])} - {"use", "instead", "item", "product", "listing"}
                scored = sorted(((len(wanted & {stem(w) for w in word_list(" ".join(str(v) for v in r.values() if isinstance(v, str)))}), i)
                                 for i, r in self.rows.items() if i != page and i not in pointer["opened"]), reverse=True)
                return scored[0][1] if scored and scored[0][0] >= 2 and (len(scored) < 2 or scored[0][0] > scored[1][0] or scored[0][1] in self.sheets) else ""
            if not target:
                target = best_match()
            if not target and self.pointer_searches < POINTER_SEARCHES:
                names = [n for n in self.shop_names() if n in follow.lower() and n != lowered(pointer["shop"])]
                words = (names[0] + " " + pointer["kind"]).strip() if names else " ".join(word_list(follow)[:6])
                if words and self.pointer_search(words):
                    target = best_match()
            if not target:
                # two rows tie on the words: open the better ones and read the one that states the follow text in full
                wanted = {stem(w) for w in word_list(follow)} - set(SMALL_WORDS)
                tied = sorted(((len(wanted & {stem(w) for w in word_list(" ".join(str(v) for v in r.values() if isinstance(v, str)))}), i)
                               for i, r in self.rows.items() if i != page and i not in pointer["opened"]), reverse=True)[:2]
                for _, ident in tied:
                    if ident not in self.sheets and self.steps_left() >= self.finish_reserve(empty_cart=True) + 3:
                        calls = self.open_calls([ident])
                        if calls:
                            function = calls[0]["function"]
                            self.agent_action(function["name"], _parse_json_object(function["arguments"]) or {}, f"Opening {ident}: the page may point to it.", "pointer_open")
                    if ident in self.sheets:
                        text = " ".join(str(v) for v in self.flat_sheet(self.sheets[ident]).values() if isinstance(v, str))
                        if all(states_value(w, text) for w in re.findall(r"\(([^)]*)\)", follow) for w in w.split(",") if w.strip()):
                            target = ident
                            break
            if target and target not in self.sheets:
                calls = self.open_calls([target])
                if calls:
                    function = calls[0]["function"]
                    self.agent_action(function["name"], _parse_json_object(function["arguments"]) or {}, f"Opening {target}: the page points on to it.", "pointer_open")
            if target in self.sheets:
                pointer["opened"].append(target)
                self.read_pointer_page(pointer, target, depth + 1)

    def apply_pointer(self, pointer: dict[str, Any]) -> None:
        """The resolved need replaces its pointer wording everywhere the requirements are kept."""
        plain = self.resolved_need(pointer)
        replaced = False
        for held in (self.stated, self.revealed):
            for index, item in enumerate(held):
                if pointer_from_need(item) is not None:
                    held[index] = plain
                    replaced = True
        if self.current is not None:
            for index, item in enumerate(self.current):
                if pointer_from_need(item) is not None:
                    self.current[index] = plain
                    replaced = True
        if not replaced:
            self.revealed.append(plain)
            if self.current is not None:
                self.current.append(plain)
        self.ledger_version += 1
        stating, _ = self.sheets_stating(plain)
        line = (f"[pointer resolved by the agent] the page {pointer['page']} gives the value: the requirement is now {plain!r}"
                + (f"; opened listings that state it: {', '.join(stating[:6])}" if stating else "; no opened listing states it yet: search for the item with this value")
                + ". The shop and the attributes that picked out that page are not requirements on the item.")
        self.trace.append({"kind": "pointer_done", "need": plain, "page": pointer["page"]})
        if self.messages and self.messages[-1].get("role") == "tool":
            self.messages[-1]["content"] += "\n" + line
        else:
            self.messages.append({"role": "user", "content": line})

    def item_records(self, item: str) -> list[str]:
        found: list[str] = []

        def walk(value: Any, depth: int) -> None:
            if depth > 6 or len(found) >= 8:
                return
            if isinstance(value, dict):
                if any(isinstance(v, str) and v == item for v in value.values()):
                    text = _compact(_brief(value, 300))[:1800]
                    if text not in found:
                        found.append(text)
                    return
                for v in value.values():
                    walk(v, depth + 1)
            elif isinstance(value, list):
                for v in value:
                    walk(v, depth + 1)

        for record in self.records:
            walk(record, 0)
        return found[-4:]

    def verify_cart(self) -> str:
        if not self.cart_ids:
            return ""
        return self.verify_item(self.cart_ids[-1], "ORDER")

    def verify_item(self, item: str, purpose: str) -> str:
        if self.time_left() < WRAP_UP_SECONDS + 20:
            return ""
        key = f"{item}|{self.ledger_version}"
        if key in self.verdicts:
            return self.verdicts[key]
        records = self.item_records(item)
        if not records:
            return ""
        prompt = VERIFY_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + self.ledger()
        prompt += f"\n\nTHE ITEM TO {purpose}: {item}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n" + "\n".join(records)
        if self.last_cart_view is not None:
            prompt += "\n\nLATEST CART VIEW:\n" + _compact(self.last_cart_view)[:3000]
        labels = self.row_labels()
        if labels:
            prompt += "\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the item's category label must be the one that matches the request's wording as a whole, not a neighbouring one):\n" + labels
        result = None
        if self.sheet_settles(item):
            # the light reader may pass a listing whose own labels state every condition; any other answer goes to the judge
            first = self.judge(prompt, 1200, [READ_MODEL])
            if first and first.get("fits") is True and not first.get("problems"):
                result = first
        if result is None:
            result = self.judge(prompt, 1200)
        self.trace.append({"kind": "verify", "item": item, "result": result})
        if not result or result.get("fits") is not False:
            self.verdicts[key] = ""
            return ""
        problems = [str(p) for p in result.get("problems") or [] if not self.describes_pointer(str(p))][:5]
        if not problems and result.get("problems"):
            self.verdicts[key] = ""                 # it only faulted the words that pick out a pointed-at page
            return ""
        verdict = f"An independent check says {item} does NOT meet every requirement: " + "; ".join(problems) + ". Unless you are certain the check is wrong, choose another fully verified fit."
        self.verdicts[key] = verdict
        return verdict

    def tools_for_turn(self) -> list[dict[str, Any]]:
        """The tools offered this turn: all of them until the last steps narrow them to what can still finish well; the
        message tool stays with the question planner while it is on."""
        r, left = self.roles, self.steps_left()
        finish = {r.get("order"), r.get("decline")}
        stages = [(1, finish), (3, finish | {r.get("cart")})]
        if self.order_attempts == 0 or self.refusals > 0:
            removers = {r.get("remove")} - {None}
            stages.append((7, finish | {r.get("cart"), r.get("cart_view"), r.get("view")} | removers))   # a swap still fits
        names = next((allowed for limit, allowed in stages if left <= limit), None)
        offered = [t for t in self.tools if names is None or t["function"]["name"] in names] if r.get("order") and r.get("cart") else self.tools
        offered = offered or self.tools
        if self.planner_on and r.get("message"):
            offered = [t for t in offered if t["function"]["name"] != r["message"]]
        # cam47: the decline stays on offer whatever the code thinks fits; a decline call meets the review in run_calls
        return offered

    @staticmethod
    def dicts_in(value: Any, limit: int = 6) -> list[dict[str, Any]]:
        """Every mapping inside a record, outermost first, down to `limit` levels."""
        found, stack = [], [(value, 0)]
        while stack:
            item, depth = stack.pop()
            if depth > limit:
                continue
            if isinstance(item, dict):
                found.append(item)
                stack.extend((v, depth + 1) for v in reversed([v for v in item.values() if isinstance(v, (dict, list))]))
            elif isinstance(item, list):
                stack.extend((v, depth + 1) for v in reversed(item))
        return found

    def property_menu(self) -> tuple[str, int]:
        """The property labels the opened listings carry (a mapping with five or more short text values is one listing's
        facts), with how many listings carry each and sample values; and how many fact sets were seen."""
        tally: dict[str, list[str]] = {}
        carriers: dict[str, int] = {}
        fact_sets = 0
        for record in self.records:
            for table in self.dicts_in(record):
                texts = [(k, v) for k, v in table.items() if isinstance(v, str) and 0 < len(v) <= 70]
                if len(texts) < 5:
                    continue
                fact_sets += 1
                for key, text in texts:
                    if re.search(r"\d", text) and re.fullmatch(r"[A-Za-z0-9_\-]+", text) and ("-" in text or "_" in text):
                        continue                     # an identifier, not a property value
                    label = re.sub(r"[_.]+", " ", str(key)).strip().lower()
                    carriers[label] = carriers.get(label, 0) + 1
                    seen = tally.setdefault(label, [])
                    if text.lower() not in seen and len(seen) < 5:
                        seen.append(text.lower())
        top = sorted(carriers, key=lambda k: -carriers[k])[:28]
        return "\n".join(f"- {k} (in {carriers[k]} listings): " + " | ".join(tally.get(k, [])) for k in top), fact_sets

    def row_labels(self) -> str:
        """Short text values that recur under one key across result rows and records (a category, a gender, a type), with
        their counts: the store's own vocabulary for the checker."""
        per_key: dict[str, dict[str, int]] = {}
        for record in self.records:
            for table in self.dicts_in(record):
                if len(table) < 4:
                    continue
                for key, text in table.items():
                    if isinstance(text, str) and 0 < len(text) <= 40 and not (re.search(r"\d", text) and re.fullmatch(r"[A-Za-z0-9_\-]+", text)):
                        counts = per_key.setdefault(str(key), {})
                        counts[text] = counts.get(text, 0) + 1
        recurring = [(key, counts) for key, counts in per_key.items() if sum(counts.values()) >= 4 and 2 <= len(counts) <= 8 and max(counts.values()) >= 2]
        return "\n".join(f"- {key}: " + " | ".join(f"{v} (x{n})" for v, n in sorted(counts.items(), key=lambda kv: -kv[1])) for key, counts in recurring[:12])

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get("message")
        schema = self.schemas.get(name or "") or {}
        properties = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        strings = [k for k, v in properties.items() if isinstance(v, dict) and (v.get("type") == "string" or "string" in (v.get("type") or [])) and "enum" not in v]
        target = next((k for k in required if k in strings), None) or (strings[0] if len(strings) == 1 else None)
        if not target:
            return None
        if any(k != target for k in required):
            return None
        return {target: text}

    def planning_prompt(self, count: int, menu: str) -> str:
        """The planner's input: the request, the requirements in force, the story's hints, what was asked and answered,
        what must be learned first, the properties the opened listings carry and the core labels among them."""
        hints = self.hints()
        parts = [PLAN_PROMPT.replace("QUESTION_COUNT", str(count)), "THE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or ""),
                 "REQUIREMENTS SO FAR:\n" + self.ledger()]
        if hints:
            parts.append("HINTS FROM THE STORY (properties the request points at without stating a value; ask about these first, each in the listings' own words for it): "
                         + "; ".join(f"{p} (likely {v})" if v else p for p, v in hints))
        if self.asked:
            parts.append("ALREADY ASKED (with what the shopper said):\n" + "\n".join(self.asked))
        if self.plan.get("ask_first"):
            parts.append("MUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in self.plan.get("ask_first") or []))
        parts.append("PROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n" + (menu or "(none seen yet)"))
        if (len(menu.splitlines()) < 4 or not self.parse_trusted()) and self.raw_results:
            # the agent's parse found few properties: the store's own text of the latest results goes along, so the
            # property names are read from what the store wrote
            parts.append("THE STORE'S OWN TEXT OF THE LATEST RESULTS (read the property names the listings carry from it):\n"
                         + "\n---\n".join(_raw_text(r)[:1800] for r in self.raw_results[-3:])[:5000])
        core = self.core_labels()
        if core:
            parts.append("CORE PROPERTIES (carried by most of the listings opened, not settled or asked yet; ask only about these): " + ", ".join(core))
        return "\n\n".join(parts)

    def question_calls(self, planned: list[dict[str, Any]], tool: str) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
        """Message calls for the planned questions that fit the message tool's schema, with (property, question) labels."""
        calls, labels = [], []
        for index, item in enumerate(planned):
            text = str(item.get("question")).strip()
            args = self.message_args(text)
            checked, problems = shape_args(self.schemas[tool], args, "args") if args else (None, ["empty"])
            if problems:
                continue
            calls.append({"id": f"ask_{self.messages_used + index + 1}_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(checked)}})
            labels.append((str(item.get("property")), text))
        return calls, labels

    @staticmethod
    def key_words(label: str) -> set[str]:
        return {stem(w) for w in word_list(label) if w not in SMALL_WORDS and w != "type"}

    @staticmethod
    def related_keys(one: set[str], other: set[str]) -> bool:
        """Two keys name one property when they share a word, or one word starts the other ('pack' and 'packaging')."""
        return bool(one & other) or any(len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)) for a in one for b in other)

    def product_sheets(self, product: str) -> list[str]:
        if not self.bundle_names:
            return list(self.sheets)
        return [i for i in self.sheets if self.product_of(i) == product]

    def key_pool(self, product: str) -> list[tuple[str, str, int, int]]:
        """The attribute keys the opened sheets of one product carry, merged across spellings: (field id, the spelling to
        ask with, sheets carrying it, distinct values), most carried first. Names, groups, ids, money, stock, sellers and
        free text are not keys a hidden need sits on; nor is a key whose value never differs between sheets, one that holds
        the listing's price, or one that holds the word its title opens with (the maker)."""
        idents = self.product_sheets(product)
        seen: dict[str, list[tuple[str, str]]] = {}
        spelled: dict[str, dict[str, int]] = {}
        for ident in idents:
            sheet = self.sheets[ident]
            flat = self.flat_sheet(sheet)
            title = " ".join(str(v) for k, v in flat.items() if isinstance(v, str) and name_key(k))
            price = _price(dict(self.rows.get(ident) or {}, **flat))
            tables = [v for v in sheet.values() if isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 3]
            source = dict(tables[0]) if len(tables) == 1 else flat   # a sheet with one facts table: its keys are the properties
            for key, value in source.items():
                label = field_id(re.sub(r"\s*\([^()]*\)\s*$", "", re.sub(r"[.]+", " ", str(key))) or str(key))   # a bracketed tag on a label ('color (spec)') is no part of the name
                if (isinstance(value, bool) or not isinstance(value, (str, int, float)) or titled(key) or grouped(key) or _names_id(key, self.id_keys())
                        or re.search(NOT_A_PROPERTY, label) or re.search(self.LOGISTICS, label)
                        or re.search(r"(?i)merchant|seller|vendor|shop|sold ?by|retailer|store|\bid\b|currenc", label)):
                    continue
                text = str(value).strip()
                if (not text or len(text) > 60 or len(text.split()) > 6 or re.search(r"[.!?]\s+\S", text) or re.search(r"\S\s+\S+[.!]$", text)
                        or re.search(r"https?:|www\.|/\S+/", text)
                        or re.fullmatch(r"(?i)n/?a|none|not specified|unknown|null|-|0", text)):
                    continue
                if re.search(r"\d", text) and re.fullmatch(r"[A-Za-z0-9_\-]+", text) and ("-" in text or "_" in text):
                    continue
                figure = money_value(value)
                if price and figure and any(abs(figure * scale - price) < 1e-6 * max(1.0, price) for scale in (1, 100, 0.01)):
                    continue                          # the listing's price, in this unit or in cents
                words = self.key_words(label)
                if not words:
                    continue
                base = re.sub(r"[^a-z0-9]", "", label)  # 'operating system' and 'operatingsystem' are one field
                maker = bool(title) and lowered(title).startswith(text.lower() + " ")
                seen.setdefault(base, []).append((ident, "\x00" if maker else text.lower()[:40]))
                spelled.setdefault(base, {})[label] = spelled.get(base, {}).get(label, 0) + 1
        out = []
        for base, held in seen.items():
            carriers = len({i for i, _ in held})
            distinct = {v for _, v in held}
            if sum(1 for _, v in held if v == "\x00") * 2 >= len(held):
                continue                              # the word the titles open with: the maker, not a property
            if carriers >= 3 and len(distinct) < 2:
                continue                              # one value on every sheet: it splits nothing
            if carriers >= 3 and len(distinct) == len(held) and sum(len(v.split()) for _, v in held) >= 3 * len(held):
                continue                              # a different phrase on every sheet: a blurb, not a property
            if carriers >= 3 and len(distinct) * 5 >= 4 * len(held) and all(re.fullmatch(r"[\d.,\s]+", v) for _, v in held):
                continue                              # a bare figure that differs on nearly every sheet: a price or a count, not a choice
            spelling = max(spelled[base], key=lambda f: (len(f.split()), spelled[base][f], len(f)))   # the spelling with its words apart
            out.append((field_id(spelling), spelling, carriers, len(distinct)))
        out.sort(key=lambda t: (-t[2], t[3] < 2, t[1]))
        merged: list[tuple[str, str, int, int]] = []
        for item in out:                              # 'fruit' beside 'fruit type': one property, asked by the longer spelling
            twin = next((m for m in merged if self.key_words(m[1]) <= self.key_words(item[1]) or self.key_words(item[1]) <= self.key_words(m[1])), None)
            if twin is None:
                merged.append(item)
            elif len(item[1]) > len(twin[1]) and any(len(w) > 2 for w in set(item[1].split()) - set(twin[1].split())):
                merged[merged.index(twin)] = (twin[0], item[1], twin[2], twin[3])   # a longer spelling, unless it only adds a tag ('fa', 'ml')
        return merged

    def key_asked(self, product: str) -> dict[str, int]:
        """How often each field id was covered by a question about the product. A reply answers the first key of the list
        that holds a need, so the keys up to the one answered were covered. A reply without a firm value covers the key
        its list led with, and the keys behind it only once some reply has shown that this shopper reads lists whole. A
        crowding complaint or a puzzled reply (-3) covers nothing: the message was not taken in."""
        counts: dict[str, int] = {}
        for name, keys, hit in self.key_msgs:
            if name != product or hit == -3:
                continue
            for key in keys[: hit + 1] if hit >= 0 else keys if hit == -1 and self.width_proven else keys[:1]:
                counts[key] = counts.get(key, 0) + 1
        return counts

    def key_behind(self, product: str) -> list[str]:
        """cam50: keys a list named behind its lead whose reply drew no firm value, not covered since. A shopper who reads
        only the first key never heard them, so the next message about the product leads with the first of them."""
        covered = self.key_asked(product)
        out: list[str] = []
        for name, keys, hit in self.key_msgs:
            if name == product and hit in (-1, -2):
                out += [k for k in keys[1:] if k not in covered and k not in out]
        return out

    def settled_words(self, product: str) -> list[set[str]]:
        """Keys the request or the shopper already settled for this product: stated and heard requirements, firm replies
        and, when single questions were asked, properties the shopper does not mind."""
        demands = self.product_demands(product) if self.bundle_names else self.demands()
        out = [self.key_words(p) for p, _ in demands] + [self.key_words(k) for k in self.key_firm.get(product, set())]
        out += [self.key_words(p["prop"]) for p in self.pointers if p.get("prop")]
        out += [self.key_words(str(f)) for f in self.free]
        return [w for w in out if w]

    def question_keys(self, product: str, first: list[str]) -> list[str]:
        """The keys of the next question about one product: the planner's picks for it (story hints first), then the keys
        most carried by its opened sheets with values that differ, never one already settled, the least asked first."""
        settled = self.settled_words(product)
        asked = self.key_asked(product)
        behind = [k for k in self.key_behind(product) if not any(self.related_keys(self.key_words(k), s) for s in settled)]
        asked.update({k: 1 for k in behind})        # cam50: a key behind a shrug may lead the next message, never stand behind again
        pool = [(fid, label, n, d) for fid, label, n, d in self.key_pool(product)]
        hinted = [p for p, _ in self.hints()]
        if self.bundle_names:                        # the story's hints go to the product whose sheets carry them
            hinted = [h for h in hinted if any(self.related_keys(self.key_words(h), self.key_words(label)) for _, label, _, _ in pool)]
        chosen: list[str] = []
        taken: list[set[str]] = []

        def fresh(label: str) -> bool:
            words = self.key_words(label)
            return (bool(words) and not asked.get(field_id(label)) and not any(self.related_keys(words, s) for s in settled)
                    and not any(words <= t or t <= words for t in taken))

        def take(label: str) -> None:
            chosen.append(label)
            taken.append(self.key_words(label))

        for label in [l for l in list(first) + hinted if l]:
            if fresh(label):
                take(field_id(label))
        floor = 2 if len(self.product_sheets(product)) >= 4 else 1
        for fid, label, n, d in sorted(pool, key=lambda t: (t[2] < floor, t[3] < 2, -t[2])):
            if fresh(label):
                take(label)
        width = max(1, self.key_width)
        # cam50: rotation. The lead goes to the first key that stood behind a shrugged lead (a shopper who reads the first
        # key alone never heard it); the rest of the list goes to keys never named, so a shopper who reads lists whole
        # hears new keys all the same. With no new keys left, the keys behind fill the list.
        lead = behind[:1]
        keys = lead + [k for k in chosen if not any(self.related_keys(self.key_words(k), self.key_words(l)) for l in lead)]
        keys += [k for k in behind[1:] if k not in keys]
        return keys[:width]

    def key_question(self, product: str, keys: list[str], plain: bool = False) -> tuple[str, list[str]]:
        """cam50: one short question about ONE product in the store's own property names, and the keys it kept: trailing
        keys are dropped while the text is longer than QUESTION_CHARS. A plain re-ask after a puzzled reply uses the
        fewest words."""
        name = single_name(product)
        keys = [re.sub(r"[_\s]+", " ", str(k)).strip() for k in keys if str(k).strip()]

        def worded(names: list[str]) -> str:
            listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
            if plain:
                return f"Which {listed} do you want for the {name}?"
            if len(names) == 1:
                return f"What {names[0]} would you like for the {name}?"
            return f"For the {name}, what {listed} would you like?"

        while len(keys) > 1 and len(worded(keys)) > QUESTION_CHARS:
            keys = keys[:-1]
        return worded(keys or ["preference"]), keys

    def status_tool(self) -> str | None:
        """A tool that reports the episode's allowances: no required argument, no role, a status or allowance cue, not final."""
        used = set(self.roles.values())
        for tool in self.tools:
            function = tool["function"]
            name = function["name"]
            if name in used or name not in self.schemas or (self.schemas[name].get("required") or []):
                continue
            text = name.replace("_", " ") + " " + str(function.get("description") or "")
            if re.search(r"(?i)status|usage|allowance|remain|quota|how many|balance|\bleft\b|meter", text) and not self.tool_shape(name)["final"]:
                return name
        return None

    def probe_allowance(self) -> None:
        """No message cap in the request: the store's allowance report, when it has one, is read once (one step); a count
        named for what is left or available becomes the cap. A count above QUESTION_CAP + 1 is no message allowance."""
        if self.probed or self.message_budget is not None or self.steps_left() < 14:
            return
        self.probed = True
        tool = self.status_tool()
        if not tool:
            return
        observation = self.agent_action(tool, {}, "Reading the store's allowance report before asking the shopper.", "allowance")
        found: list[tuple[str, int]] = []
        stack = [("", observation, 0)]
        while stack:
            key, value, depth = stack.pop()
            if isinstance(value, dict) and depth < 4:
                stack += [(f"{key} {k}", v, depth + 1) for k, v in value.items()]
            elif isinstance(value, int) and not isinstance(value, bool):
                found.append((key.lower(), value))
        left = [(k, v) for k, v in found if re.search(r"avail|remain|left|unused|unspent", k) and not re.search(r"step|move|turn|time", k)]
        named = [(k, v) for k, v in left if re.search(r"messag|question|ask|chat|contact|allowance|quota|note", k)]
        pick = named or left
        if len({v for _, v in pick}) > 1:
            pick = [min(pick, key=lambda kv: kv[1])]   # counts that disagree (what is left beside what was granted): the least
        if pick and 0 <= pick[0][1] <= QUESTION_CAP + 1:
            self.message_budget = self.messages_used + pick[0][1]
            self.trace.append({"kind": "allowance", "messages": self.message_budget})

    def product_order(self, force: bool) -> list[str]:
        """The products to ask about next, one question each: those without a firm reply first, then the least asked; a
        product with too few opened sheets waits until the asking mark (unless forced)."""
        names = self.bundle_names or [str(self.plan.get("product") or "item")]
        late = force or self.step >= self.max_steps * ASK_MARK or self.time_left() < TASK_SECONDS * ASK_TIME
        ready = [n for n in names if late or not self.bundle_names or len(self.product_sheets(n)) >= KEY_READY]
        asked = {n: sum(1 for p, _, _ in self.key_msgs if p == n) for n in names}
        return sorted(ready, key=lambda n: (bool(self.key_firm.get(n)), asked[n], names.index(n)))

    def planner_picks(self, menu: str, count: int, urgent: bool) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
        """The planner's properties per product (the main product for an unnamed one) and its step-0 items."""
        # one property per message makes the order of the picks decisive, so the planner reads on the judge model while the
        # spend is under the soft line (judge() moves it to the light model past it)
        result = self.judge(self.planning_prompt(count, menu), 2000, [JUDGE_MODEL, READ_MODEL])
        planned = [q for q in (result or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("property") or q.get("question") or "").strip()]
        names = self.bundle_names or [str(self.plan.get("product") or "item")]
        picks: dict[str, list[str]] = {n: [] for n in names}
        firsts = []
        for index, item in enumerate(planned):
            if urgent and not firsts and str(item.get("question") or "").strip() and (item.get("first") is True or index == 0):
                firsts.append(item)                   # what must be learned first goes out in the planner's own words
                continue
            words = self.name_words(str(item.get("product") or ""))
            name = max(names, key=lambda n: (len(self.name_words(n) & words), -names.index(n))) if self.bundle_names else names[0]
            prop = re.sub(r"\s*\(.*?\)\s*", " ", str(item.get("property") or "")).strip(" ?.").lower()
            if prop and not re.search(NOT_A_PROPERTY, prop) and len(prop.split()) <= 4:
                picks[name].append(prop)
        return picks, firsts

    def hear_keys(self, product: str, keys: list[str], before: tuple[list[str], list[str], int], text: str) -> int:
        """After one question: which listed key drew a firm value (its index), or -1 for a reply without one (a shrug),
        -2 for a reply that spoke to the first key alone, -3 for a crowding complaint or a puzzled reply (nothing taken
        in). The kind of reply is kept in reply_kind for the width that follows."""
        revealed, free, pointers = before
        fresh = [r for r in self.revealed if r not in revealed] + [p["text"] for p in self.pointers[pointers:]]
        if self.current is not None:
            fresh += [r for r in self.current if r not in self.revealed and r not in revealed and r not in self.stated]
        hit = -1
        own = {stem(w) for w in word_list(product)}
        bare = lambda label: (self.key_words(label) - own) or self.key_words(label)   # a need named with its product's words
        for need in fresh:
            words = bare(str(need).partition(":")[0])
            index = next((i for i, k in enumerate(keys) if words and self.related_keys(bare(k), words)), -1)
            if index >= 0:
                self.key_firm.setdefault(product, set()).add(keys[index])
                hit = index if hit < 0 else min(hit, index)
            if words:
                self.key_firm.setdefault(product, set()).add(" ".join(sorted(words)))
        line = lowered(self.heard_now + " " + self.last_reply)   # (a line heard before is not heard anew, so the reply itself counts too)
        spoken = {stem(w) for w in word_list(line)} - own
        named = [i for i, k in enumerate(keys) if self.key_words(k) and self.key_words(k) <= spoken]
        if hit > 0 or (not fresh and any(i > 0 for i in named)):
            self.width_proven = True                  # a reply spoke to a key behind the first: this shopper reads lists whole
        if fresh or hit >= 0:
            self.reply_kind = "firm"
            if len(keys) > 1:                         # a firm value for one key says nothing sure about the others it named
                self.free = [f for f in self.free if f in free or (self.key_words(str(f)) and self.key_words(str(f)) <= spoken)]
            return hit
        crowded = bool(self.flags.get("one_at_a_time")) or bool(CROWD_CUE.search(line))
        crowd = len(keys) > 1 and crowded              # (a complaint about a one-key message asks for it again, like a puzzled reply)
        puzzled = not crowd and (bool(self.flags.get("unclear")) or crowded or bool(PUZZLE_CUE.search(line)))
        if crowd or puzzled:
            self.reply_kind = "crowd" if crowd else "puzzled"
            self.free = [f for f in self.free if f in free]
            return -3
        # a shrug: the reader's "does not matter" holds for the lead key, for a key the reply names, and for every key once
        # lists are shown to be read whole; when the reader heard a shrug but named another key, the lead key is marked free
        lead = self.key_words(keys[0]) if keys else set()
        shrugged = any(f not in free for f in self.free)
        kept = []
        for f in self.free:
            words = self.key_words(str(f))
            if f in free or self.width_proven or len(keys) < 2 or (words and (self.related_keys(words, lead) or words <= spoken)):
                kept.append(f)
        self.free = kept
        if shrugged and lead and not any(self.related_keys(self.key_words(str(f)), lead) for f in self.free):
            self.free.append(keys[0])
        self.reply_kind = "first" if named == [0] and len(keys) > 1 else "shrug"
        return -2 if self.reply_kind == "first" else -1

    def ask_planned(self, force: bool = False, spare: int = 0) -> bool:
        """The questions, sent by the agent itself once enough listings were opened (or at once when something must be
        learned first). Each question names up to key_width of ONE product's keys (the planner's picks and the keys its
        opened sheets carry most), one question per product in turn, until the message cap is spent; every answer is read
        and kept with its question, and the number of keys per question follows how the shopper answers (adapt_width).
        With spare > 0 (cam50: the order is about to go out) up to that many more questions go out, whatever the pace."""
        tool = self.roles.get("message")
        if not tool:
            self.planner_on = False
            return False
        self.forced_ask = self.forced_ask or force
        menu, fact_sets = self.property_menu()
        urgent = bool(self.plan.get("ask_first")) and not self.asked
        if len(self.sheets) < 4 and self.step < min(10, self.max_steps // 3) and not urgent and not force:
            return False
        urgent = urgent and not force
        never = self.messages_used == 0 and force and self.steps_left() > 8 and self.time_left() > 45
        if not never and (self.steps_left() <= 10 or self.time_left() < WRAP_UP_SECONDS + 40 or (self.commit_time() and not spare)):
            self.planner_on = False                 # with a fit known and few steps left, a question is a step the finish needs
            return False                            # (but nothing is ever ordered with no question asked while one can go)
        if self.message_budget is None and not urgent:
            self.probe_allowance()
            if self.finished or self.dead:
                return False
        limit = min(self.message_limit(), self.unstated_questions() if self.message_budget is None else QUESTION_CAP) - self.held_messages()
        if spare:
            limit = self.messages_used + spare
        if self.messages_used >= limit:
            self.planner_on = False
            return False
        order = self.product_order(force)
        if not order and not urgent:
            return False
        slots = max(limit - self.messages_used, len(self.bundle_names))
        picks, firsts = self.planner_picks(menu, min(10, max(3, max(1, self.key_width) * slots + 1)), urgent)
        sent = False
        thin = urgent and fact_sets < 4
        for item in firsts[:1] if urgent else []:
            text = self.phrase(item, True)
            if self.send_question(tool, str(item.get("property") or "what must be learned first"), text, "", []):
                sent = True
        rounds = 0
        while not thin and self.messages_used < limit - self.unready_reserve(force) and not self.finished and not self.dead and self.steps_left() > 4 and rounds < 2 * QUESTION_CAP:
            rounds += 1
            plain = False
            if self.requeue:                          # cam50: a complained or puzzled message's keys go out again first
                product, keys, plain = self.requeue.pop(0)
                settled = self.settled_words(product)
                keys = [k for k in keys if field_id(k) not in self.key_firm.get(product, set())
                        and not any(self.related_keys(self.key_words(k), s) for s in settled)]
                if not keys:
                    continue
            else:
                order = self.product_order(force)
                if not order:
                    break
                product = order[0]
                if self.key_firm.get(product) and self.unready_reserve(force):
                    break                             # a product still unasked waits for its sheets: no second question before it
                keys = self.question_keys(product, picks.get(product) or [])
                if not keys:
                    if all(not self.question_keys(n, picks.get(n) or []) for n in order):
                        break
                    self.key_msgs.append((product, [], -1))   # nothing left to ask about this product: the next one goes first
                    continue
            for flag in ("unclear", "muddled", "one_at_a_time"):
                self.flags.pop(flag, None)            # the flags below are this reply's own
            text, keys = self.key_question(product, keys[: max(1, self.key_width)], plain)
            self.reply_kind = ""
            if not self.send_question(tool, ", ".join(keys) + (" (again, simpler)" if plain else ""), text, product, keys):
                break
            sent = True
            self.adapt_width(product, keys, plain)
            if spare and self.messages_used >= limit:
                break
        waiting = bool(self.bundle_names) and len(self.product_order(force)) < len(self.bundle_names)
        self.planner_on = self.messages_used < limit and (thin or waiting)
        return sent

    def spare_at_buy(self) -> int:
        """cam50: messages still free when the cart is filled or the order goes out: the cap less those used and those held
        for finish duties. The message held for asking before a swap is free too once the store's change after the first
        cart was seen and no line in the cart is gone (no swap is pending)."""
        held = self.held_messages()
        swap = any(d["when"] == "replace" for d in self.duties) and not self.swap_asked
        if swap and self.cart_used and self.changes > 0 and self.cart_ids and not any(i in self.gone for i in self.cart_ids):
            held -= 1
        cap = min(self.message_limit(), self.unstated_questions() if self.message_budget is None else QUESTION_CAP)
        return cap - held - self.messages_used

    def buy_question(self, moment: str) -> bool:
        """cam50: the cart or the order is about to go out while messages are still free: they carry more planned
        questions (once per moment), so no allowed message is left unspent. True when a question went out."""
        if moment in self.buy_asks or not self.roles.get("message") or self.finished or self.dead:
            return False
        spare = self.spare_at_buy()
        if spare <= 0 or self.steps_left() <= max(10, self.loop_floor() + 3) or self.time_left() < WRAP_UP_SECONDS + 40:
            return False
        names = self.bundle_names or [str(self.plan.get("product") or "item")]
        if not self.requeue and not any(self.question_keys(n, []) for n in names):
            return False                              # nothing left to ask: no planner call is spent
        self.buy_asks.add(moment)
        used = self.messages_used
        self.ask_planned(force=True, spare=spare)
        self.trace.append({"kind": "buy_question", "moment": moment, "spare": spare, "sent": self.messages_used - used})
        return self.messages_used > used

    def adapt_width(self, product: str, keys: list[str], plain: bool) -> None:
        """cam50: the width of later questions after one reply. A crowding complaint ("one at a time", "split them up")
        sets one key per message for the rest of the episode and re-sends the complained keys one by one, first first. A
        puzzled reply ("say that again", "simpler") halves the width and re-sends the message once in plainer words (at
        most SIMPLER_ASKS times per episode); a plain re-send that puzzles again counts as asked, so the next message moves
        on. A shrug leaves the width as it is: its lead key is free and the next message leads with a key never led."""
        kind = self.reply_kind
        before = self.key_width
        if kind == "crowd":
            self.key_width = 1
            self.width_locked = True
            self.requeue = [(product, [k], False) for k in keys] + [(p, ks, pl) for p, ks, pl in self.requeue if p != product or not set(ks) & set(keys)]
        elif kind == "puzzled":
            if not self.width_locked and len(keys) > 1:   # the width that puzzled is the message's own; a one-key message keeps it
                self.key_width = max(1, (min(self.key_width, len(keys)) + 1) // 2)
            if keys and not plain and self.simpler_sent < SIMPLER_ASKS and field_id(keys[0]) not in self.simpler_keys:
                self.simpler_sent += 1
                self.simpler_keys.add(field_id(keys[0]))
                self.requeue.insert(0, (product, keys[: max(1, self.key_width)], True))
            elif self.key_msgs and self.key_msgs[-1][2] == -3:
                name, ids, _ = self.key_msgs[-1]
                self.key_msgs[-1] = (name, ids, -1)   # re-sent once already (or no re-send left): counted as asked, the next message moves on
        self.trace.append({"kind": "question_width", "reply": kind or "none", "keys": len(keys), "width": [before, self.key_width],
                           "locked": self.width_locked, "requeued": len(self.requeue)})

    def second_look(self, text: str) -> bool:
        """One question about a single key that stood behind the first in a list which drew no firm value: a shopper who
        reads only the first key never heard it, so it may still go out while messages remain."""
        if text.count("?") != 1 or self.width_proven:
            return False
        words = {stem(w) for w in word_list(text)}
        for _, keys, hit in self.key_msgs:
            if hit == -1 and len(keys) > 1:
                behind = [k for k in keys[1:] if self.key_words(k) and self.key_words(k) <= words]
                if len(behind) == 1 and not any(self.key_words(k) <= words for k in keys[:1]):
                    return True
        return False

    def novel_question(self, text: str) -> bool:
        """The model's own question names a property no question asked yet (and is one question): the agent ran out of
        keys of its own, so this one may go out while messages remain."""
        if text.count("?") != 1:
            return False
        words = {stem(w) for w in word_list(text)} - {stem(w) for w in word_list(str(self.plan.get("product") or ""))} - set(SMALL_WORDS)
        asked = set().union(*[self.key_words(k) for _, keys, _ in self.key_msgs for k in keys]) if self.key_msgs else set()
        asked |= set().union(*[self.key_words(a[2:].split(": asked", 1)[0]) for a in self.asked]) if self.asked else set()
        fresh = {w for w in words - asked if len(w) >= 4 and not re.search(r"(?i)would|like|which|what|prefer|want|need", w)}
        return bool(fresh) and not self.width_proven

    def unready_reserve(self, force: bool) -> int:
        """Messages kept for bundle products not yet asked whose sheets are still too few: each gets its own question."""
        ready = set(self.product_order(force))
        return sum(1 for n in self.bundle_names if n not in ready and not any(p == n for p, _, _ in self.key_msgs))

    def send_question(self, tool: str, label: str, text: str, product: str, keys: list[str]) -> bool:
        """One question sent alone and read; True when it reached the store."""
        calls, labels = self.question_calls([{"property": label, "question": text}], tool)
        if not calls:
            return False
        before = (list(self.revealed), list(self.free), len(self.pointers))
        used = self.messages_used
        self.last_keys = list(keys)
        self.messages.append({"role": "assistant", "content": "I will ask the shopper about " + label + ".", "tool_calls": calls})
        replies = self.run_calls(calls, planned=True)
        for call_id, reply in replies:
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply})
        if self.messages_used == used:
            return False
        if self.message_budget is None and self.messages_used == 1 and len(self.revealed) == len(before[0]) and len(self.pointers) == before[2]:
            self.blind_retry = True                   # no cap known and the one message drew nothing firm: one more may go
        answer = (replies[0][1] if replies else "").split(chr(10) + "[status]")[0][:400]
        self.asked.append(f"- {label}: asked \"{text}\" -> {answer}")
        self.last_reply = answer.split(chr(10) + "[")[0]   # the store's own text of the reply, not the agent's notes under it
        if keys:
            hit = self.hear_keys(product, [field_id(k) for k in keys], before, text)
            self.key_msgs.append((product, [field_id(k) for k in keys], hit))
        self.last_keys = []
        return True

    def phrase(self, item: dict[str, Any], first: bool) -> str:
        """One question in the fixed form "What <property> would you like for the <product>?"; an item that must be learned
        first (a missing budget, a choice between two wishes) keeps the planner's own words."""
        own = str(item.get("question") or "").strip()
        if own and (first or item.get("first") is True):
            return own
        prop = re.sub(r"^(?:the|a|an|your)\s+", "", re.sub(r"[\s?.!:]+$", "", str(item.get("property") or "")).strip(), flags=re.I)
        product = str(item.get("product") or "").strip() or (self.bundle_names[0] if self.bundle_names else str(self.plan.get("product") or "item"))
        product = re.sub(r"^(?:the|a|an|your|some)\s+", "", re.sub(r"[\s?.!]+$", "", product), flags=re.I)
        return f"What {prop} would you like for the {product}?"

    def send_reply(self, text: str) -> bool:
        tool = self.roles.get("message")
        limit = self.message_limit() - self.held_messages()
        if not tool or self.messages_used >= limit or self.steps_left() <= 4 or self.time_left() < WRAP_UP_SECONDS:
            return False
        args = self.message_args(text)
        if not args:
            return False
        args, problems = shape_args(self.schemas[tool], args, "args")
        if problems:
            return False
        call = {"id": f"reply_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "I will answer the shopper's question.", "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        for index, (call_id, reply_text) in enumerate(replies):
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text})
        return True

    def quantity_warning(self, name: str, args: dict[str, Any]) -> str:
        spec = self.basket_spec(name)
        if spec and isinstance(args.get(spec[0]), list):
            if not spec[2]:
                return ""
            for line in args[spec[0]]:          # a whole basket: each line at its own product's quantity
                if isinstance(line, dict) and isinstance(line.get(spec[1]), str) and isinstance(line.get(spec[2]), int):
                    want = self.line_quantity(line[spec[1]])
                    if line[spec[2]] != want and (self.bundle_names and self.product_of(line[spec[1]]) or not self.bundle_names):
                        return f"the request needs {want} of {line[spec[1]]}, but this basket asks for {line[spec[2]]}."
            return ""
        wanted = self.plan.get("quantity")
        if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
            return ""
        properties = self.schemas[name].get("properties") or {}
        signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if self.removing_call(name, args):
            return ""
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                value = args.get(key, spec.get("default", 1))
                if isinstance(value, int) and value != wanted:
                    return f"the request needs a quantity of {wanted} of one listing, but this call asks for {value}."
        return ""

    def review_before_decline(self) -> str:
        """The one review pass a decline meets: the request and the shopper's answers are put back in front of the model,
        with the requirements in force, and one more round of searching and reading is asked for before the decline may
        go out (a decline scores only when nothing in the store fits)."""
        request = str(self.policy.get("query") or "")
        answers = "\n".join(self.asked[-4:])[-1000:]
        held = [i for i in self.cart_ids if i not in self.gone]
        if held:
            return (f"One review before a decline: the cart holds {', '.join(held)}, which the store has not reported gone. Re-read the request and the "
                    "shopper's answers; if the held line still meets every requirement in force, order it; if it fails one, replace it with a listing "
                    "that meets everything (search once more if none is known) and order that; decline only if no listing fits. Send the decline "
                    "again to confirm it.\nTHE REQUEST: " + request[:700] + ("\nTHE SHOPPER SAID:\n" + answers if answers else "")
                    + "\nREQUIREMENTS IN FORCE: " + self.ledger()[:600])
        return ("One review before a decline. Re-read the request and the shopper's answers, then make one more round: a search in words you "
                "have not tried yet (the category's own label, one requirement value, no filters you are unsure of) and open the details of the "
                "best candidates you have not read; read the store's own text of each result, not only the agent's notes. If a listing meets "
                "everything, put it in the cart and order it; only if none does, call the decline tool again.\nTHE REQUEST: " + request[:700]
                + ("\nTHE SHOPPER SAID:\n" + answers if answers else "") + "\nREQUIREMENTS IN FORCE: " + self.ledger()[:600])

    def audit_decline(self) -> str:
        if self.audited or self.time_left() < WRAP_UP_SECONDS + 30:
            return ""
        self.audited = True
        blocks: list[str] = []
        total = 0
        room = 30000 if self.spend_level() == 0 else 14000    # past the soft line the audit reads a shorter prompt
        for ident in reversed(list(self.sheets)):
            text = _compact(self.sheets[ident])
            if total + len(text) > room:
                continue
            blocks.append(text)
            total += len(text)
        for record in reversed(self.records):
            text = _compact(_brief(record, 160 if self.parse_trusted() else RAW_CUT))
            if total + len(text) > room + 6000:
                continue
            blocks.append(text)
            total += len(text)
        if not blocks:
            return ""
        prompt = AUDIT_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(blocks)
        # the light reader looks first; names it gives are checked again anyway, and only the judge may let a decline stand
        result = self.judge(prompt, 1200, [READ_MODEL])
        fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        if not fits:
            result = self.judge(prompt, 1200)
        self.trace.append({"kind": "audit", "result": result})
        fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        if not fits:
            return ""
        return "An independent check thinks these listings may meet every requirement: " + ", ".join(fits) + (f" ({result.get('reason')})" if result.get("reason") else "") + ". Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."

    def neutral_args(self, name: str, text: str, kind: str | None = None) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        properties = schema.get("properties") or {}
        args: dict[str, Any] = dict(self.finish_args.get(kind) or {}) if kind else {}
        for key in schema.get("required") or []:
            if key in args:
                continue
            spec = properties.get(key) or {}
            if spec.get("type") == "string" or "string" in (spec.get("type") or []):
                args[key] = text[: spec.get("maxLength", len(text))]
            elif "enum" in spec and spec["enum"]:
                args[key] = spec["enum"][0]
            elif spec.get("type") == "boolean":
                args[key] = kind != "decline"           # a yes/no finishing switch: yes buys, no ends without buying
            elif spec.get("type") in ("integer", "number"):
                args[key] = max(1, spec.get("minimum", 1))
        for key, spec in properties.items():
            if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec and len(properties) == 1:
                args[key] = text[: spec.get("maxLength", len(text))]
        if kind == "order" and self.cart_ids and text == "Order the cart.":
            text = (self.basket_text() + " Each line was read on its own listing; " + ", ".join(self.cart_ids) + ".")[:900]
        for key, spec in properties.items():   # an optional reason slot gets the plain account of the finish
            if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec and re.search(r"(?i)justif|reason|explan|why|summar|remark|note", key):
                args[key] = text[: spec.get("maxLength", len(text))]
        return args

    def unmet(self) -> list[str]:
        """What could not be met, from the store's own facts: the requirements most opened listings fail (by their sheets)
        and the price limit when opened listings exceed it; the independent checks' findings when the facts show nothing.
        Named without any listing (ids, titles or prices), so a decline that must not suggest substitutes may carry it.
        Decline text must never break a finish, so any fault here reads as 'nothing to name'."""
        try:
            return self.unmet_reasons()
        except Exception:
            return []

    def unmet_reasons(self) -> list[str]:
        demands = self.demands()
        kind, group = set(label_words(self.plan.get("product") or "")), self.main_group()
        fails: dict[str, int] = {}
        dear = 0
        for ident, sheet in self.sheets.items():
            own = _shelf(sheet) or _shelf(self.rows.get(ident) or {})
            if group and own and own != group:
                continue
            if not self.affordable(sheet):
                dear += 1
            flat = self.flat_sheet(sheet)
            for prop, value in demands:
                if read_sheet(flat, prop, value, kind)[0] == "no":
                    fails[f"{prop} {value}"] = fails.get(f"{prop} {value}", 0) + 1
        reasons = [text for text, _ in sorted(fails.items(), key=lambda t: -t[1])]
        if dear and self.budget_now is not None:
            reasons.append(f"price at most {self.budget_now}")
        if not reasons:
            for verdict in self.verdicts.values():
                found = re.search(r"requirement: (.+?)\. Unless", verdict or "")
                if found:
                    reasons += [p.strip() for p in found.group(1).split(";") if p.strip()]
        if not reasons and self.reason_owed():
            # nothing singled out, yet the shopper asked to be told: name the requirements in force as a whole
            reasons = [f"{prop} {value}" for prop, value in demands]
            if self.budget_now is not None:
                reasons.append(f"price at most {self.budget_now}")
        return list(dict.fromkeys(reasons))[:3]

    def decline_words(self) -> str:
        """The agent's own decline text: what was not met first, so that a short field still carries the reason."""
        reasons = self.unmet()
        if reasons:
            return "No listing met: " + "; ".join(reasons) + ". So I am not ordering anything."
        return "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."

    def reason_owed(self) -> bool:
        """Whether the request asks the decline to say what was missing (a duty at the decline or the finish)."""
        return any(d["when"] in ("decline", "finish") and re.search(r"(?i)reason|which requirement|what was (?:missing|wrong|the problem)|what went wrong|why|explain|could ?n[o']t be met|was the problem", d["do"])
                   for d in self.duties)

    def with_reason(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        """When the request asks the decline to name what was missing and the model's words do not, the agent's own
        reason-bearing text takes their place in the free-text fields."""
        reasons = self.unmet()
        if not reasons or not self.reason_owed():
            return args
        words = {w.lower() for r in reasons for w in re.findall(r"[A-Za-z0-9]+", r) if len(w) > 2} | {"price", "budget", "cost", "limit"}
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        out = dict(args)
        for key, value in args.items():
            spec = properties.get(key) or {}
            if isinstance(value, str) and "enum" not in spec and not any(w in words for w in re.findall(r"[A-Za-z0-9]+", value.lower())):
                text = self.decline_words()
                out[key] = text[: spec.get("maxLength", len(text))]
        for duty in self.duties:
            if duty["when"] in ("decline", "finish") and not duty["done"] and self.reason_owed() and re.search(r"(?i)reason|requirement|missing|wrong|problem|why|explain", duty["do"]):
                duty["done"] = True
        return out

    def clean_decline(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        if self.plan.get("decline_no_names") is False:
            return args
        dirty = False
        for key, value in args.items():
            if isinstance(value, str):
                for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value):
                    if token and token in self.seen_ids:
                        dirty = True
        if self.plan.get("decline_no_names") or dirty:
            neutral = self.decline_words()
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

    def label_of(self, ident: str) -> str:
        record = self.sheets.get(ident) or self.rows.get(ident) or {}
        title = next((v for k, v in record.items() if isinstance(v, str) and name_key(k) and v.strip()), "") or next((v for k, v in record.items() if isinstance(v, str) and titled(k) and v.strip()), "")
        price = self.latest_price(ident)
        return (title[:80] or ident) + (f" ({price:g} {self.plan.get('currency') or ''})".replace(" )", ")") if price is not None else "")

    def basket_text(self) -> str:
        """The cart in plain words with the total, written by the agent itself."""
        total, parts = 0.0, []
        for ident in self.cart_ids:
            price, quantity = self.latest_price(ident), self.cart_qty.get(ident) or self.line_quantity(ident)
            total += (price or 0.0) * quantity
            parts.append(f"{quantity} x {self.label_of(ident)}")
        if not parts:
            return ""
        return "Here is what I am about to order: " + "; ".join(parts) + f". Total: {total:g} {self.plan.get('currency') or ''}".rstrip() + "."

    def say(self, text: str) -> tuple[list[str], str] | None:
        """Send one message by the agent's own hand (one step) and read the answer; None when it could not go out."""
        tool = self.roles.get("message")
        args = self.message_args(text[:600]) if tool else None
        if not args:
            return None
        args, problems = shape_args(self.schemas[tool], args, "args")
        if problems:
            return None
        reply = self.env_call([{"name": tool, "args": args}])
        self.trace.append({"kind": "agent_message", "text": text, "reply": reply})
        if reply is None:
            return None
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        observation = self.book_result(tool, args, reply.get("observation"), reply, self.roles.get("cart"), tool, finishing)
        heard, _ = self.shopper_lines(tool, args, observation, tool)
        return heard, (self.interpret(heard, text) if heard else "")

    def carry_duties(self, kind: str, light: bool = False) -> tuple[str, bool]:
        """Right before an order or a decline: what the request said to do or tell at this point and is still undone, carried
        out in one message (worded by the light reader, or by the agent itself when spending or time is short). Returns the
        note for the model and whether the finish must wait (the shopper's answer changed something)."""
        due = self.duties_due(kind)
        if not due:
            return "", False
        if kind == "order" and self.roles.get("cart") and not self.cart_ids:
            return "", False                        # an order on an empty cart is refused; its duties wait for the real order
        talking = [d for d in due if d["tell"]]
        tool = self.roles.get("message")
        room = self.message_budget is None or self.messages_used < self.message_budget
        for duty in due:
            if not duty["tell"] or not tool or not room:
                duty["done"] = True            # nothing to say, or no way left to say it: as done as it can be
        if not talking or not tool or not room or self.steps_left() < 2 or self.time_left() < 6:
            return "", False
        text = ""
        if not light and self.spend_level() < 2 and self.time_left() > WRAP_UP_SECONDS / 2:
            action = "place the order" if kind == "order" else "decline (end without buying anything)"
            prompt = WRAP_PROMPT.replace("ACTION", action) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
            prompt += "\n\nTO DO NOW:\n" + "\n".join(f"- {d['do']}" for d in talking)
            prompt += "\n\nMESSAGES ALREADY SENT:\n" + ("\n".join(f"- {t}" for t in self.sent_texts[-6:]) or "(none)")
            if kind == "order":
                prompt += "\n\nTHE BASKET:\n" + (self.basket_text() or "(empty)")
            elif self.unmet():
                prompt += "\n\nWHAT COULD NOT BE MET (name it when the shopper asked to be told what was missing):\n" + "; ".join(self.unmet())
            result = self.judge(prompt, 600, [READ_MODEL, JUDGE_MODEL])
            if result is not None:
                text = str(result.get("message") or "").strip()
                if not text:
                    return "", False
        if not text:
            text = self.basket_text() if kind == "order" else self.decline_words()
        version = self.ledger_version
        exchange = self.say(text) if text else None
        if exchange is None:
            return "", False                      # the message did not go out: the duties stay owed for the next attempt
        for duty in talking:
            duty["done"] = True
        heard, meaning = exchange
        note = f"Before finishing, the shopper was told: {text}"
        if heard:
            note += " Their answer: " + " / ".join(heard)[:300] + (f" ({meaning})" if meaning else "")
        return note, self.ledger_version != version and not self.finished

    def final_check(self, kind: str) -> str:
        """Once, right before the first finishing call: one light look (the light reader only, no store step) at what the
        request and the shopper demanded that the cart or the record does not yet show. A step still undone is put on the
        duty list, so it is carried out before the finish; a condition a cart line may fail holds an order once, with the
        list, so the model can fix the cart or send the order again. A decline is never held: it is still right then.
        A check must never break a finish, so any fault inside it reads as 'nothing found'."""
        try:
            return self.final_look(kind)
        except Exception as exc:
            self.trace.append({"kind": "final_check", "action": kind, "error": f"{type(exc).__name__}: {exc}"[:200]})
            return ""

    def final_look(self, kind: str) -> str:
        if self.steps_left() < 3 or self.spend_level() == 2 or self.time_left() < WRAP_UP_SECONDS / 2:
            return ""
        mark = f"{kind}|{_compact(sorted(self.cart_ids))}|{self.ledger_version}"
        if self.final_checked and not (self.long_world() and self.spend_level() == 0 and mark not in self.final_marks and len(self.final_marks) < LONG_FINAL_CHECKS and self.room_for_extras()):
            return ""                               # once per episode; in a long world once more for a finishing call with a new cart or new requirements, while steps allow
        self.final_checked = True
        self.final_marks.add(mark)
        action = "place the order" if kind == "order" else "decline (end without buying anything)"
        prompt = FINAL_PROMPT.replace("ACTION", action) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        lines = [f"[{ident}] {text}" for ident in self.cart_ids for text in self.item_records(ident)[-2:]]
        prompt += "\n\nTHE CART:\n" + ("\n".join(lines)[:6000] if lines else "(empty)")
        prompt += "\n\nSTEPS RECORDED FROM THE REQUEST AND THE SHOPPER:\n" + ("\n".join(f"- {d['do']} (at the {d['when']}; {'done' if d['done'] else 'not done'})" for d in self.duties) or "(none)")
        prompt += "\n\nMESSAGES SENT TO THE SHOPPER:\n" + ("\n".join(f"- {t}" for t in self.sent_texts[-6:]) or "(none)")
        prompt += "\n\nWHAT THE SHOPPER SAID (oldest first):\n" + ("\n".join(self.said[-10:]) or "(nothing)")
        result = self.judge(prompt, 500, [READ_MODEL])
        self.trace.append({"kind": "final_check", "action": kind, "result": result})
        if not isinstance(result, dict):
            return ""
        # a step counts only when it is anchored in the request's or the shopper's own words, never in the reader's
        grounding = {w for w in word_list(str(self.policy.get("query") or "") + " " + " ".join(self.said)) if len(w) >= 4}
        for step in [s for s in result.get("undone_steps") or [] if isinstance(s, str) and s.strip()][:2]:
            if any(w in grounding for w in word_list(step) if len(w) >= 4):
                self.add_duty(step.strip()[:160], kind, True)
        conditions = [str(c).strip() for c in result.get("unmet_conditions") or [] if isinstance(c, (str, int)) and str(c).strip()][:4]
        if self.shed_words:                         # what describes a pointed-at listing (its shop, its attributes) is never a condition on the item
            conditions = [c for c in conditions if not any(stem(w) in self.shed_words for w in word_list(c.split(":")[-1]))]
        if kind != "order" or not conditions or not self.cart_ids:
            return ""
        return ("A final check before the order finds requirements the cart line may not meet: " + "; ".join(conditions)
                + ". Check the item's own records against them and fix the cart if it fails; if you are certain the cart is right, send the same call again.")

    def ask_swap(self, target: str) -> str:
        """When the request says to check before replacing an item: ask once per new listing, and return the exchange."""
        if target in self.swap_asked or not any(d["when"] == "replace" for d in self.duties):
            return ""
        old = self.cart_ids[-1] if self.cart_ids else next(iter(self.gone), "")
        if not old or old == target or not self.roles.get("message"):
            return ""
        if self.messages_used >= self.message_limit() or self.steps_left() <= 4:
            return ""
        self.swap_asked.add(target)
        text = f"{self.label_of(old)} can't be bought as planned. May I replace it with {self.label_of(target)}?"
        exchange = self.say(text)
        if exchange is None:
            return ""
        heard, meaning = exchange
        return (f"The request says to check with the shopper before replacing an item, so they were asked: \"{text}\" Their answer: "
                + (" / ".join(heard)[:300] or "(none)") + (f" ({meaning})" if meaning else "") + ".")

    def change_review(self) -> str:
        """After a new budget or requirement, a refusal or a gone item: one light look at the cart and the opened listings."""
        reason, self.change_reason = self.change_reason, ""
        if self.reviews_left <= 0 or self.spend_level() == 2 or self.steps_left() <= 3 or self.time_left() < WRAP_UP_SECONDS + 20:
            return ""
        blocks: list[str] = []
        for ident in self.cart_ids:
            blocks += [f"[in the cart] {text}" for text in self.item_records(ident)[-2:]]
        total = sum(len(b) for b in blocks)
        for ident in reversed(list(self.sheets)):
            text = _compact(self.sheets[ident])[:1500]
            if ident in self.cart_ids or total + len(text) > 14000:
                continue
            blocks.append(text)
            total += len(text)
        if not blocks:
            return ""
        self.reviews_left -= 1
        prompt = REVIEW_PROMPT + "\n\nWHAT CHANGED:\n" + reason + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS NOW:\n" + self.ledger()
        if self.plan.get("priority") == "cheapest":
            prompt += "\nThe cheapest fitting listing is wanted."
        prompt += "\n\nTHE CART: " + (", ".join(self.cart_ids) or "empty") + ("; refused as gone: " + ", ".join(sorted(self.gone)) if self.gone else "")
        prompt += "\n\nRECORDS:\n" + "\n".join(blocks)
        result = self.judge(prompt, 900, [READ_MODEL, JUDGE_MODEL])
        self.trace.append({"kind": "change_review", "reason": reason, "result": result})
        if not result:
            return ""
        why = str(result.get("why") or "").strip()
        instead = [str(x) for x in result.get("take_instead") or [] if isinstance(x, (str, int)) and str(x) not in self.gone][:3]
        if result.get("cart_still_right") is True and not (self.cart_ids and self.cart_ids[-1] in self.gone):
            return f"the cart still looks right after this change ({why})." if why else "the cart still looks right after this change."
        return (f"after this change the cart no longer looks right ({why})" + (f"; opened listings that meet everything now: {', '.join(instead)}" if instead
                else "; no opened listing clearly meets everything now, so search again") + ". Fix the cart before ordering.")

    def together_note(self) -> str:
        """Several products bought together: once, the cart must hold one line per product and they must go together."""
        if not self.companions or self.together_checked:
            return ""
        self.together_checked = True
        needed = 1 + len(self.companions)
        if self.cart_ids and len(self.cart_ids) < needed:
            return (f"The request needs {needed} products bought together ({self.plan.get('product')}; " + "; ".join(self.companions)
                    + f"), but the cart holds {len(self.cart_ids)}: add the missing ones, each checked to go with the others, then order.")
        if self.spend_level() == 2 or len(self.cart_ids) < 2:
            return ""
        records = [f"[{ident}] {text}" for ident in self.cart_ids for text in self.item_records(ident)[-2:]]
        prompt = TOGETHER_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "") + "\n\nRECORDS OF THE PRODUCTS IN THE CART:\n" + "\n".join(records)
        result = self.judge(prompt, 700)
        self.trace.append({"kind": "together", "result": result})
        if not result or result.get("works_together") is not False:
            return ""
        faults = [str(p) for p in (result.get("problems") or []) + [f"missing: {m}" for m in result.get("missing") or []]][:4]
        return "A check of the products together says they do not go together as required: " + "; ".join(faults) + ". Fix the cart, then order."

    # ---------- several products in one request ----------
    @staticmethod
    def split_products(plan: dict[str, Any]) -> list[str]:
        """The products a request for several products names: the setup's item list, else the combined product name split
        at its joins, else the product and its companions. Empty for a request for one product."""
        names = [str(i.get("product") or i.get("name") or "").strip() if isinstance(i, dict) else str(i).strip() for i in plan.get("items") or []]
        companions = [str(c) for c in plan.get("companions") or [] if isinstance(c, str) and c.strip()]
        if len([n for n in names if n]) < 2 and companions:
            whole = str(plan.get("product") or "").strip()
            names = [p for p in re.split(r",\s*|\s+and\s+|\s*[+&]\s*", whole) if p.strip()]
            if len(names) < 2:
                names = [whole] + [c.split(":")[0] for c in companions]
        out: list[str] = []
        for name in names:
            if name.strip() and lowered(name.strip()) not in [lowered(o) for o in out]:
                out.append(name.strip())
        return out[:6] if len(out) >= 2 else []

    def name_words(self, name: str) -> set[str]:
        return set(label_words(name)) - set(SMALL_WORDS)

    def product_of(self, ident: str) -> str:
        """The product of the request a listing belongs to, by its shelf (weighed most) and its title words ("" when unclear)."""
        record = dict(self.rows.get(ident) or {})
        record.update(self.flat_sheet(self.sheets.get(ident) or {}))
        shelf = set(label_words(_shelf(record)))
        title = set(label_words(" ".join(str(v) for k, v in record.items() if isinstance(v, str) and (re.search(r"title|heading|name", str(k), re.I) or titled(k)))))
        scores = sorted(((3 * sum(any(kin(w, h) for h in shelf) for w in self.name_words(n)) + sum(any(kin(w, h) for h in title) for w in self.name_words(n)), n)
                         for n in self.bundle_names), key=lambda t: -t[0])
        if not scores or scores[0][0] == 0 or (len(scores) > 1 and scores[1][0] == scores[0][0]):
            return ""
        return scores[0][1]

    def product_demands(self, name: str) -> list[tuple[str, str]]:
        """The requirements in force that bind one product: those whose property names it (most of its words), and the
        untagged ones for the first product. A value that names another product ("the same colour as the mug") is a
        relation between lines, left to the checks that read the cart as a whole."""
        out = []
        for prop, value in self.demands():
            words = set(label_words(prop))
            scores = {n: len(self.name_words(n) & words) for n in self.bundle_names}
            top = max(scores.values())
            heads = {n: [w for w in label_words(n) if w not in SMALL_WORDS][-1:] for n in self.bundle_names}
            if any(set(h) & set(label_words(value)) for h in heads.values()):
                continue                              # the value names a product (its head word): a relation between lines
            if (top and scores.get(name) == top) or (not top and name == self.bundle_names[0]):
                out.append((prop, value))
        return out

    def own_demands(self, ident: str) -> list[tuple[str, str]]:
        """The requirements that bind one listing: its product's own in a request for several products, else all."""
        name = self.product_of(ident) if self.bundle_names else ""
        if not name:
            return self.demands()
        out = []
        for prop, value in self.product_demands(name):
            bare = re.sub(r"^\s*" + re.escape(name.lower()) + r"\s+", "", prop.lower()) if prop.lower().startswith(name.lower() + " ") else prop
            out.append((bare, value))                # "laptop memory module memory type" is the module's "memory type"
        return out

    def product_fits(self, name: str) -> list[tuple[str, float]]:
        """Opened listings of one product whose own sheets state its every requirement, in stock, affordable, not gone; in
        the order the request's rule prefers."""
        kind, demands = self.name_words(name), self.product_demands(name)
        found = []
        for ident, sheet in self.sheets.items():
            price = self.latest_price(ident)
            if ident in self.gone or price is None or self.product_of(ident) != name or not self.affordable(sheet) or self.known_bad(ident):
                continue
            if all(read_sheet(self.flat_sheet(sheet), prop, value, kind)[0] == "yes" for prop, value in demands):
                found.append((ident, price))
        return self.ranked_by_rule(self.code_first(found))

    def known_bad(self, ident: str) -> bool:
        """A line that fails a known requirement: an independent check under the requirements now in force named a problem,
        or its fitment lines name only other codes than the one in play."""
        if self.bundle_names and self.product_of(ident) == self.bundle_names[0] and not any(len(re.findall(r"\d", t)) >= 3 for t in CODE_TOKEN.findall(str(self.policy.get("query") or ""))):
            return bool(self.verdicts.get(f"{ident}|{self.ledger_version}"))   # the main product's own model code sets the fitment; it is not held to it
        return bool(self.verdicts.get(f"{ident}|{self.ledger_version}")) or (bool(self.wanted_codes()) and self.code_fit(ident) is False)

    def bundle_unread(self, name: str) -> list[str]:
        """Rows of one product shown in results, never opened, in stock and affordable: cheapest first."""
        opened = {v for entry in self.trace for action in (entry.get("actions") or [entry.get("action") or {}])
                  if isinstance(action, dict) and action.get("name") in (self.roles.get("details"), self.roles.get("view"))
                  for v in (action.get("args") or {}).values() if isinstance(v, str)}
        rows = [(_price(r) or 0.0, i) for i, r in self.rows.items() if i not in self.sheets and i not in opened and i not in self.gone and self.affordable(r) and self.product_of(i) == name]
        return [i for _, i in sorted(rows)]

    def bundle_picks(self) -> dict[str, str]:
        """One line per product: the held line while it still fits and the rule prefers no other, else the rule's first fit;
        the cheapest fits instead when the preferred ones together pass the budget. "" for a product without a fit."""
        picks: dict[str, str] = {}
        for name in self.bundle_names:
            fits = self.product_fits(name)
            held = [i for i in self.cart_ids if i in dict(fits)]
            pick = fits[0][0] if fits else ""
            if held and (self.plan.get("priority") not in ("cheapest", "best") or self.ranked_by_rule([f for f in fits if f[0] in held or f[0] == pick])[0][0] in held):
                pick = held[0]
            picks[name] = pick
        cap = self.budget_now
        if isinstance(cap, (int, float)) and not isinstance(cap, bool) and all(picks.values()):
            total = lambda chosen: sum((self.latest_price(i) or 0.0) * self.line_quantity(i) for i in chosen.values())
            if total(picks) > cap + 1e-9:
                picks = {n: min(self.product_fits(n), key=lambda t: (t[1], t[0]))[0] for n in self.bundle_names}
                if total(picks) > cap + 1e-9:
                    picks = {n: "" for n in picks}
        return picks

    def bundle_exhausted(self, name: str) -> bool:
        """A product with no fit left to find: nothing read fits, no unread row of it is shown, and it was searched for."""
        return not self.product_fits(name) and not self.bundle_unread(name) and (name in self.bundle_searched or self.searches_done >= len(self.bundle_names))

    def bundle_status(self) -> str:
        """One line per product of a bundle: its cart line, or what is missing for it."""
        if not self.bundle_names:
            return ""
        picks, lines = self.bundle_picks(), []
        for name in self.bundle_names:
            held = [i for i in self.cart_ids if self.product_of(i) == name]
            if held:
                lines.append(f"\n[bundle] {name}: cart line {held[0]} held; keep it (the order is refused while any product is missing)")
            elif picks.get(name):
                lines.append(f"\n[bundle] {name}: missing from the cart; {picks[name]} states its every requirement: put it in")
            else:
                lines.append(f"\n[bundle] {name}: missing, no line yet and no verified fit read: search for it and open its listings")
        return "".join(lines)

    def bundle_search(self, name: str) -> bool:
        """One search for a product of a bundle by its own name (the words slot, a text mode when the tool has modes)."""
        tool = self.roles.get("search")
        props = (self.schemas.get(tool or "") or {}).get("properties") or {}
        args: dict[str, Any] = {}
        for key, spec in props.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("enum"):
                text = next((v for v in spec["enum"] if re.search(r"(?i)text|keyword|word|free|query", str(v))), None)
                if text is not None or key in ((self.schemas.get(tool or "") or {}).get("required") or []):
                    args[key] = text if text is not None else spec["enum"][0]
            elif spec.get("type") == "string" and re.search(r"(?i)text|query|keyword|term|words|search|\bq\b|phrase|string", key.replace("_", " ")) and not re.search(GROUP_LIKE, key, re.I):
                args.setdefault(key, name)
        if name not in args.values():
            return False
        self.bundle_searched.add(name)
        self.bundle_reads += 1
        return self.agent_action(tool, args, f"Looking for the {name}, a product the request needs.", "bundle") is not None

    def bundle_look(self) -> bool:
        """The agent reads on for a bundle itself: a product without a fit gets its cheapest unread rows opened (or one search
        when none is shown); under a best rule every unread row of a product is read before its pick. True when a step went."""
        if self.bundle_reads >= BUNDLE_READS * len(self.bundle_names) or self.steps_left() < self.finish_reserve(empty_cart=True) + len(self.bundle_names) + 3:
            return False
        for name in self.bundle_names:
            unread = self.bundle_unread(name)
            fits = self.product_fits(name)
            if fits and self.plan.get("priority") == "cheapest" and not self.measure_words():
                floor = min(p for _, p in fits)
                unread = [i for i in unread if (_price(self.rows.get(i) or {}) or 0.0) < floor - 1e-9]   # only rows cheaper than the cheapest fit can change the pick
                if not unread:
                    if self.bundle_page(name):      # a later page of a short list may hold a cheaper fit
                        return True
                    continue
            elif fits and not (self.measure_words() and unread):
                continue
            if unread:
                calls = self.open_calls(unread[:1])
                if calls:
                    self.bundle_reads += 1
                    function = calls[0]["function"]
                    return self.agent_action(function["name"], _parse_json_object(function["arguments"]) or {}, f"Opening {unread[0]}, a listing for the {name}.", "bundle") is not None
            elif not self.product_fits(name) and name not in self.bundle_searched:
                return self.bundle_search(name)
            elif not self.product_fits(name) and self.bundle_page(name):
                return True
        return False

    def page_slot_of(self, tool: str) -> str | None:
        props = (self.schemas.get(tool or "") or {}).get("properties") or {}
        return next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") in ("integer", "number") and re.search(r"(?i)page|row|start|offset|next", k)), None)

    def note_list(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """A result that lists rows and names its next page (a number under a 'next'/'more' key): kept per call shape, so
        the agent can turn that list's pages itself."""
        slot = self.page_slot_of(name)
        if not slot or not isinstance(observation, dict) or name in (self.roles.get("cart"), self.roles.get("order"), self.roles.get("decline")):
            return
        _, rows = self.records_in(observation)
        nxt = next((v for k, v in observation.items() if re.search(r"(?i)next|more|token|cursor", str(k)) and isinstance(v, int) and not isinstance(v, bool)), None)
        if not rows:
            return
        key = name + _compact({k: v for k, v in args.items() if k != slot})
        ids = [i for i in (self.id_in(r) for r in rows) if i]
        old = self.list_pages.pop(key, {"ids": []})
        self.list_pages[key] = {"tool": name, "args": dict(args), "slot": slot, "next": nxt, "ids": old["ids"] + ids}

    def bundle_page(self, name: str) -> bool:
        """A product of a bundle with no fit and no unread row: the next page of the newest search that showed its rows
        (pages are short, and the search does not know model codes), a few pages per product at most."""
        tools = {self.roles.get("search")}
        if self.bundle_paged.get(name, 0) >= BUNDLE_PAGES:
            return False
        for key, listed in reversed(list(self.list_pages.items())):
            if listed["next"] is None or not any(self.product_of(i) == name for i in listed["ids"]):
                continue
            args = dict(listed["args"], **{listed["slot"]: listed["next"]})
            args, problems = shape_args(self.schemas[listed["tool"]], args, "args")
            if problems:
                continue
            listed["next"] = None                   # turned once; the new page's result records its own next
            self.bundle_paged[name] = self.bundle_paged.get(name, 0) + 1
            self.bundle_reads += 1
            return self.agent_action(listed["tool"], args, f"No {name} read so far fits: turning to the next page of that list.", "bundle") is not None
        for key, cover in reversed(list(self.page_cover.items())):
            if not cover["seen"] or len(cover["seen"]) >= cover["total"] or not any(self.product_of(i) == name for i in cover["seen"]):
                continue
            args = dict(cover["args"])
            tool = self.roles.get("search")
            props = (self.schemas.get(tool or "") or {}).get("properties") or {}
            slot = next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") in ("integer", "number") and re.search(r"(?i)page|row|from|start|offset|token|next", k)), None)
            if not tool or tool not in tools or not slot:
                continue
            paged = re.search(r"(?i)page", slot) and not re.search(r"(?i)row|start|offset|from", slot)
            args[slot] = (max(1, cover["page"]) + 1) if paged else len(cover["seen"])
            self.bundle_paged[name] = self.bundle_paged.get(name, 0) + 1
            self.bundle_reads += 1
            return self.agent_action(tool, args, f"No {name} read so far fits: turning to the next page of that search.", "bundle") is not None
        return False

    def resolve_gone(self) -> None:
        """A refusal said a line is gone without naming it: each held line is opened again and the ones now shown out of
        stock are marked gone."""
        self.bundle_unsure = False
        tool = self.roles.get("view") or self.roles.get("details")
        schema = self.schemas.get(tool or "") or {}
        key = next((k for k, v in (schema.get("properties") or {}).items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v), None)
        held = list(self.cart_ids)
        for ident in held:
            if not key or self.steps_left() < self.finish_reserve(swap=False) + 2:
                return
            observation = self.agent_action(tool, {key: ident}, f"The store said a line can no longer be bought: re-reading {ident}.", "bundle")
            shown = [r for r in Episode.dicts_in(observation, 8) if self.id_in(r) == ident or r is observation]
            if any(_in_stock(r) is False for r in shown) or _in_stock(self.sheets.get(ident) or {}) is False:
                self.gone.add(ident)
                return
        if held and not any(i in self.gone for i in held):
            self.gone.add(held[0])                  # no page tells: the first line carted is the one a market change hits

    def bundle_missing(self) -> list[str]:
        """Products without a held line that is not gone ([] also when lines of no clear product can cover the rest)."""
        missing = [n for n in self.bundle_names if not any(self.product_of(i) == n for i in self.cart_ids if i not in self.gone)]
        unknown = [i for i in self.cart_ids if i not in self.gone and not self.product_of(i)]
        return missing if len(missing) > len(unknown) else []

    def bundle_hold(self, name: str, args: dict[str, Any] | None) -> str:
        """The model's order, decline or removal in a bundle: a removal of a product's verified line is held; an order of a
        cart missing a product, or a decline while every product has a fit, makes the agent complete the cart and order
        itself; a decline stands only when some product truly has no fit. "" lets the call run."""
        args = args or {}
        picks = self.bundle_picks()
        target = self.removal_target(name, args)
        if target:
            if target in picks.values() and target not in self.gone and not self.known_bad(target) and not self.model_veto(target):
                return f"Not run (no step used): {target} is the verified line for the {self.product_of(target)}; the order needs one line per product."
            return ""
        kind = self.intent(name, args)
        missing = self.bundle_missing()
        if kind == "order" and not missing:
            return ""
        if kind == "decline" and any(self.bundle_exhausted(n) for n in self.bundle_names):
            fallback = self.fallback_picks()        # a decline stands only when some product has no buyable listing not proven wrong
            if fallback and self.steps_left() >= 3 and not self.fallback_tried:
                self.fallback_tried = True
                note = self.place_basket(fallback, force=True, why="is the nearest listing found for")
                if note:
                    return "Not run (handled by the agent): " + note
            return ""
        if all(picks.values()):
            note = self.fill_bundle(force=True)
            if note:
                return "Not run (handled by the agent): " + note
        if kind == "decline" and self.steps_left() <= 2:
            return ""
        return ("Not run (no step used): the request needs one line per product and these are missing: " + ", ".join(missing or [n for n, i in picks.items() if not i])
                + ". Find and open a listing for each that states its every requirement; decline only when one of them has none.")

    def fill_bundle(self, force: bool = False) -> str:
        """Bundles only: once the questions are done (or the order is due, or `force`), the agent itself puts one verified
        line per product in the cart (taking out lines of no product, gone lines and lines the rule replaces) and sends the
        order; a partial cart is never ordered. Returns a note ("" when nothing was done)."""
        cart, order = self.roles.get("cart"), self.roles.get("order")
        if not self.bundle_names or not cart or not order or self.finished or self.dead or self.time_left() < FORCE_SECONDS / 2:
            return ""
        if self.bundle_unsure:
            self.resolve_gone()
        if not force and not (self.questions_done() or self.step >= self.pace_targets()[1]):
            return ""
        if not force and any(not p["done"] for p in self.pointers):
            return ""
        if not force and self.bundle_look():
            return "looked"
        picks = self.bundle_picks()
        if not all(picks.values()):
            return ""
        return self.place_basket(picks, force)

    def place_basket(self, picks: dict[str, str], force: bool = False, why: str = "states every requirement of") -> str:
        """Puts exactly one line per product in the cart (one whole-basket call when the store sets the basket at once,
        else removals and adds line by line) and sends the order. Returns a note ("" when nothing was done)."""
        cart = self.roles.get("cart")
        if self.basket_spec(cart):
            wanted = list(dict.fromkeys(picks.values()))
            if set(wanted) != set(self.cart_ids) or any((self.cart_qty.get(i) or 1) != self.line_quantity(i) for i in wanted):
                if self.steps_left() < 2:
                    return ""
                self.cart_qty = {}
                self.agent_action(cart, self.basket_args(wanted), "The basket gets one line per product: " + ", ".join(f"{n}: {i}" for n, i in picks.items()) + ".", "bundle")
                if self.finished or set(self.cart_ids) != set(wanted):
                    return ""
            self.cart_changed_at = self.step
            self.agent_placed.update(picks.values())
            self.order_checked = self.order_checked or force
            return self.order_now("The cart holds one line per product (" + ", ".join(f"{n}: {i}" for n, i in picks.items()) + ").", force=force)
        need = len([i for i in self.cart_ids if i not in picks.values()]) + len([i for i in picks.values() if i not in self.cart_ids])
        if self.steps_left() < need + 1:
            return ""
        if any(i in self.bundle_removed for i in self.cart_ids if i not in picks.values()):
            return ""                               # a line taken out once and put back by the model stays: no add/remove ping-pong
        for ident in [i for i in self.cart_ids if i not in picks.values()]:
            self.bundle_removed.add(ident)
            removal = self.remove_args(ident)
            if removal is None or self.agent_action(removal[0], removal[1], f"{ident} is not the line the bundle needs: taking it out.", "bundle") is None:
                return ""
            self.cart_ids = [i for i in self.cart_ids if i != ident]
        for name, ident in picks.items():
            if ident not in self.cart_ids:
                self.agent_action(cart, self.cart_args(ident), f"{ident} {why} the {name}: putting it in the cart.", "bundle")
                if ident not in self.cart_ids or self.finished:
                    return ""
        self.cart_changed_at = self.step
        self.agent_placed.update(picks.values())
        self.order_checked = self.order_checked or force
        return self.order_now("The cart holds one verified line per product (" + ", ".join(f"{n}: {i}" for n, i in picks.items()) + ").", force=force)

    def code_finish_modes(self) -> None:
        """Without a plan (the setup reading failed): a finishing tool that both buys and ends without buying is found by
        its mode argument, whose values name both endings, so a finish in code can still order or decline."""
        if self.finish_args and self.roles.get("decline"):
            return
        buy_sign, stop_sign = self.ROLE_CUES["order"] + r"|commit", self.ROLE_CUES["decline"] + r"|^none$|^stop$|cancel"
        for tool in self.tools:
            name = tool["function"]["name"]
            for key, spec in ((self.schemas.get(name) or {}).get("properties") or {}).items():
                values = [str(v) for v in (spec.get("enum") or [])] if isinstance(spec, dict) else []
                stop = next((v for v in values if re.search(stop_sign, v.replace("_", " ").lower())), None)
                buy = next((v for v in values if v != stop and re.search(buy_sign, v.replace("_", " ").lower())), None)
                if len(values) == 2 and (stop or buy) and not (stop and buy):
                    stop = stop or next(v for v in values if v != buy)   # two modes on a finishing tool: the one not buying ends without a purchase
                    buy = buy or next(v for v in values if v != stop)
                if stop and buy:
                    self.finish_args = {"order": {key: buy}, "decline": {key: stop}}
                    self.roles.setdefault("order", name)
                    self.roles.setdefault("decline", name)
                    return
        for name in (self.roles.get("order"), self.roles.get("decline")):
            shape = self.tool_shape(name) if name else None
            flag = next((k for k in shape["flags"] if k in shape["required"]), None) if shape and shape["final"] else None
            if flag:                                  # one finishing tool with a required yes/no switch
                self.finish_args = {"order": {flag: True}, "decline": {flag: False}}
                self.roles.setdefault("order", name)
                self.roles.setdefault("decline", name)
                return

    def last_ditch(self) -> None:
        """The finish with no time for anything else: the cart as it stands is ordered when it holds lines that are not
        known to be gone or unfit, else the episode is declined."""
        order, decline = self.roles.get("order"), self.roles.get("decline")
        good = [i for i in self.cart_ids if i not in self.gone]   # cam47: a held line the store did not call gone is ordered
        if order and good and len(good) == len(self.cart_ids):
            action = {"name": order, "args": self.neutral_args(order, "Order the cart.", "order")}
        elif decline:
            self.empty_for_decline()
            action = {"name": decline, "args": self.neutral_args(decline, self.decline_words(), "decline")}
        else:
            return
        reply = self.env_call([action], final=True)
        self.trace.append({"kind": "last_ditch", "action": action, "reply": reply})
        if reply and not reply.get("finished") and not self.finished and decline and self.steps_left() > 0 and self.intent(action["name"], action["args"]) == "order":
            self.empty_for_decline()
            self.env_call([{"name": decline, "args": self.neutral_args(decline, self.decline_words(), "decline")}], final=True)

    def known_unfit(self, ident: str) -> bool:
        """A listing whose own sheet reads 'no' on a requirement in force, or that is over the budget."""
        sheet = self.sheets.get(ident)
        if not sheet or self.bundle_names:
            return False
        if not self.affordable(sheet):
            return True
        kind = set(label_words(self.plan.get("product") or ""))
        return any(read_sheet(self.flat_sheet(sheet), prop, value, kind)[0] == "no" for prop, value in self.demands())

    def force_finish(self) -> None:
        """The finish in code, tried again (at most twice more) when the store answers a finishing call with the shopper's
        words instead of ending the episode: those words are taken in first, then the cart and the ending are decided anew."""
        for _ in range(3):
            self.final_reply = None
            self.finish_once()
            reply = self.final_reply
            if self.finished or self.dead or self.steps_left() <= 0 or not isinstance(reply, dict) or reply.get("finished"):
                return
            if self.time_left() + (FINAL_SECONDS - TASK_SECONDS) < 6:
                return
            observation = reply.get("observation")
            try:
                heard, _ = self.shopper_lines("", {}, observation, self.roles.get("message"))
                heard += [t for t in self.notice_texts(observation) if t not in heard]
                if heard:
                    self.hear_in(heard)
                    if self.time_left() > FORCE_SECONDS / 2:
                        self.interpret(heard, None)
            except Exception as exc:                # taking the words in must never stop the finish
                self.trace.append({"kind": "crash", "error": f"refused finish: {type(exc).__name__}: {exc}"[:300]})
            self.trace.append({"kind": "finish_again", "step": self.step})

    def finish_once(self) -> None:
        if self.finished or self.dead or self.steps_left() <= 0:
            return
        if self.time_left() < FORCE_SECONDS / 2:
            self.code_finish_modes()
            self.last_ditch()                       # the clock is nearly out: the finish goes out now, with nothing else first
            return
        self.code_finish_modes()
        order = self.roles.get("order")
        decline = self.roles.get("decline")
        cart = self.roles.get("cart")
        if self.bundle_names:
            for _ in range(FINISH_READS):           # steps left over: the agent reads and pages on for products without a fit first
                if (self.finished or self.dead or all(self.bundle_picks().values()) or self.time_left() < FORCE_SECONDS / 2
                        or self.steps_left() < self.finish_reserve(empty_cart=True) + len(self.bundle_names) + 3):
                    break
                self.bundle_reads = min(self.bundle_reads, BUNDLE_READS * len(self.bundle_names) - 1)
                if not self.bundle_look():
                    break
            if self.finished or self.dead or self.steps_left() <= 0:
                return
            self.fill_bundle(force=True)            # one verified line per product and the order
            if self.finished or self.dead or self.steps_left() <= 0:
                return
            if self.bundle_missing() or any(i in self.gone for i in self.cart_ids) or not self.cart_ids:
                picks = self.fallback_picks()       # no decline while each product has a buyable listing not proven wrong
                if picks and cart and order:
                    self.place_basket(picks, force=True, why="is the nearest listing found for")
                    if self.finished or self.dead or self.steps_left() <= 0:
                        return
            if (not self.cart_ids or all(i in self.gone for i in self.cart_ids)) and decline:
                self.empty_for_decline()
                self.env_call([{"name": decline, "args": self.neutral_args(decline, self.decline_words(), "decline")}], final=True)
                return
        if order and cart and not self.companions and self.steps_left() >= 5 and self.time_left() > 12:
            self.last_reads()
        if order and cart and self.time_left() > 8:
            self.drop_gone_lines()
        # any fresh line may be refused at the order (a first add sets off a market change, a later pick may be gone), so a
        # fresh cart needs room for the add, the order and a decline after it, plus a message owed before ordering
        room = 3 + int(any(d["tell"] for d in self.duties_due("order")))
        fits = (self.fits_by_facts() or self.fits_by_facts(strict=False)) if order and cart and not self.cart_ids and self.steps_left() >= room and self.time_left() > 8 else []
        fits = [f for f in fits if f[0] not in self.gone]      # a listing refused as gone cannot be ordered
        if fits:
            best = self.forced_pick(fits)
            reply = self.env_call([{"name": cart, "args": self.cart_args(best)}])
            self.trace.append({"kind": "forced_cart", "item": best, "reply": reply})
            if reply is not None and not (isinstance(reply.get("observation"), dict) and any(re.search(r"fault|error|issue", str(k)) for k in reply["observation"])):
                self.cart_used = True
                self.cart_ids = [best]
        if (order and cart and not self.cart_ids and not self.bundle_names and not self.companions and self.raw_results
                and (not self.rows and self.blind_results or not self.parse_trusted() or not self.model_declined)
                and self.steps_left() >= room and self.time_left() > 20):
            self.blind_pick()                       # the code found no fit: the model picks from the store's own text
        if order and cart and not self.cart_ids and not self.cart_used and not self.bundle_names and not self.companions and self.steps_left() >= 3:
            self.hold_first(by_code=True)           # never a decline with nothing ever held: the first add brings the shopper's changes
            if self.finished or self.dead or self.steps_left() <= 0:
                return
        if order and cart and self.cart_ids and self.steps_left() >= 4 and self.time_left() > 10:
            self.enforce_cheapest()                 # a cheaper fit already read is ordered instead
            self.enforce_best()                     # under a best rule, the opened fit the rule ranks first is ordered instead
        # cam47: ordering is the default ending. A held cart is never declined because the agent's own parse reads a line as
        # unfit; only a line the agent itself held as a probe (hold_first) and the model, reading the store's text, finds
        # unfit with nothing else fitting leads to a decline; a basket is declined as partial only when the parse is trusted
        # and the cart holds fewer lines than the request has products (a count, not the parse's guess of what each line is)
        self.cart_ids_unfit = False
        if order and decline and self.bundle_names and self.cart_ids and self.bundle_missing() and self.parse_trusted() and len(set(self.cart_ids)) < len(self.bundle_names):
            self.trace.append({"kind": "partial_bundle_declined", "missing": self.bundle_missing(), "cart": list(self.cart_ids)})
            self.cart_ids_unfit = True              # an order missing a product the request needs is never right
        if (order and decline and cart and self.cart_ids == [self.hold_unfit] and self.hold_unfit and not self.bundle_names
                and self.hold_unfit not in {i for i, _ in self.fits_by_facts(strict=False)}):
            self.cart_ids_unfit = self.probe_verdict() == "decline"
        if order and decline and self.cart_ids_unfit:
            self.empty_for_decline()
            kind, action = "decline", {"name": decline, "args": self.neutral_args(decline, self.decline_words(), "decline")}
        elif order and (self.cart_ids if cart else self.cart_used) and not (self.cart_ids and all(i in self.gone for i in self.cart_ids)):
            kind, action = "order", {"name": order, "args": self.neutral_args(order, "Order the cart.", "order")}
        elif decline:
            self.empty_for_decline()
            kind, action = "decline", {"name": decline, "args": self.neutral_args(decline, self.decline_words(), "decline")}
        else:
            return
        if self.steps_left() >= (3 if kind == "order" and decline else 2) and self.time_left() > 6:
            self.final_check(kind)                  # steps a light look still finds undone are carried out below
            self.carry_duties(kind, light=True)
        reply = self.env_call([action], final=True)
        self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        # an order and a decline may be one tool in two modes: the repairs below are for a refused ORDER only (a refused
        # decline goes back to force_finish, which takes the shopper's words in and decides the ending anew)
        if reply and not reply.get("finished") and kind == "order" and self.steps_left() > 0:
            reply = self.repair_in_code(action, reply)
        said = _compact((reply or {}).get("observation"))
        if (reply and not reply.get("finished") and kind == "order" and self.steps_left() > 0
                and self.flags.get("price_moved") and not self.flags.get("unavailable") and not any(i in self.gone for i in self.cart_ids)):
            reply = self.env_call([action])          # a changed price alone leaves the pick acceptable: order once more
            self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        if reply and not reply.get("finished") and kind == "order" and decline and self.steps_left() > 0:
            self.empty_for_decline()
            action = {"name": decline, "args": self.neutral_args(decline, self.decline_words(), "decline")}
            reply = self.env_call([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})

    def store_pick(self, held: str = "") -> tuple[bool, str | None] | None:
        """The model reads the store's own text of the latest results (not the agent's parse) with the request, the
        requirements in force and the shopper's answers, and names the one listing that meets everything, or none.
        Returns (read, pick): None when no reading could be had (no model, no money, no text); pick is None when the
        model found nothing that fits. Once per episode."""
        if self.blind_tried or self.llm_dead or self.spent() >= self.spend_caps()[2] or self.time_left() < 12:
            return None
        texts = [_raw_text(r) for r in self.raw_results[-8:]]
        raw = "\n---\n".join(t[:3500] for t in texts)
        if len(raw) < 80:
            return None
        self.blind_tried = True
        prompt = (BLIND_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "") + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
                  + ("\n\nTHE SHOPPER'S ANSWERS:\n" + "\n".join(self.asked)[-1500:] if self.asked else "")
                  + (f"\n\nTHE CART HOLDS: {held} (name it as the pick if it meets everything)" if held else "")
                  + "\n\nTHE STORE'S RECENT RESULTS (its own text, oldest first):\n" + raw[-10000:])
        result = self.judge(prompt, 600, [JUDGE_MODEL, READ_MODEL] if self.spend_level() == 0 and self.time_left() > 40 else [READ_MODEL])
        if not isinstance(result, dict):
            return None
        pick = result.get("pick")
        self.trace.append({"kind": "blind_pick", "pick": pick, "why": str(result.get("why") or "")[:200]})
        if not isinstance(pick, str) or not pick.strip():
            return True, None
        pick = pick.strip()
        if not _ID_HINT.match(pick) or pick not in raw + held or pick in self.gone:
            return True, None
        return True, pick

    def blind_pick(self) -> None:
        """The finish with an empty cart and no fit the code could read: the model's own pick from the store's text goes
        into the cart, so the order goes out instead of a decline made without a look."""
        found = self.store_pick()
        if not found or found[1] is None:
            return
        pick = found[1]
        cart = self.roles.get("cart")
        reply = self.env_call([{"name": cart, "args": self.cart_args(pick)}])
        self.trace.append({"kind": "blind_cart", "item": pick, "reply": reply})
        if reply is not None and not (isinstance(reply.get("observation"), dict) and any(re.search(r"fault|error|issue", str(k)) for k in reply["observation"])):
            self.cart_used = True
            self.cart_ids = [pick]

    def probe_verdict(self) -> str:
        """The cart holds only the line the agent put in as a probe, which its parse reads as unfit: the model reads the
        store's text and decides. "order" when it names the held line, or when no reading could be had and the parse is
        not trusted; "swap" when it named another listing and the swap was made; "decline" when it found nothing that
        fits (or, with no reading, the trusted parse says the line fails)."""
        held = self.hold_unfit
        found = self.store_pick(held)
        if found is None:
            return "decline" if self.parse_trusted() else "order"
        pick = found[1]
        if pick is None:
            return "decline"
        if pick == held:
            return "order"
        cart, removal = self.roles.get("cart"), self.remove_args(held)
        if not cart or removal is None or self.steps_left() < 4 or self.time_left() < 10:
            return "order"
        if self.env_call([{"name": removal[0], "args": removal[1]}]) is None:
            return "order"
        self.cart_ids = [i for i in self.cart_ids if i != held]
        reply = self.env_call([{"name": cart, "args": self.cart_args(pick)}])
        self.trace.append({"kind": "probe_swap", "out": held, "in": pick, "reply": reply})
        if reply is not None:
            self.cart_ids = [pick]
        return "swap"

    def proven_wrong(self, ident: str) -> bool:
        """A listing shown to fail: gone or out of stock, over the limit, its own facts contradicting a requirement in
        force, an independent check failing it, or fitment lines naming only other codes."""
        key = self.closeness(ident)
        return key is None or key[0] or key[1] > 0 or self.known_bad(ident)

    def states_own(self, ident: str) -> bool:
        """A listing whose own facts state every requirement of its product, leaving out fitment requirements (those are
        judged by the codes): an unread row or a sheet silent on a requirement is not evidence enough to order."""
        demands = [(p, v) for p, v in self.own_demands(ident) if not (self.wanted_codes() and re.search(r"(?i)compat|\bfits?\b|\bwork", p + " " + v))]
        if not demands:
            return self.code_fit(ident) is not False
        if ident not in self.sheets:
            return False
        own = self.product_of(ident)
        kind = self.name_words(own) if own else set(label_words(self.plan.get("product") or ""))
        flat = self.flat_sheet(self.sheets[ident])
        return all(read_sheet(flat, p, v, kind)[0] == "yes" for p, v in demands)

    def fallback_picks(self) -> dict[str, str]:
        """For a finish without a verified line for every product: per product its verified fit, else its held line, else
        the nearest buyable listing of it not proven wrong (a fitment match first, then a read sheet, then the price);
        within the budget when the cheapest choices allow. {} when some product has no such listing at all."""
        verified = self.bundle_picks()
        picks: dict[str, str] = {}
        pools: dict[str, list[str]] = {}
        for name in self.bundle_names:
            head = [stem(w) for w in label_words(name) if w not in SMALL_WORDS][-1:]
            def named(i: str) -> bool:              # the product's head noun on the listing's shelf or in its name
                record = dict(self.rows.get(i) or {}, **{k: v for k, v in (self.sheets.get(i) or {}).items() if isinstance(v, str)})
                words = {stem(w) for k, v in record.items() if isinstance(v, str) and (grouped(k) or name_key(k)) for w in label_words(v)}
                return bool(head) and any(kin(head[0], w) for w in words)
            pool = [i for i in list(self.sheets) + [r for r in self.rows if r not in self.sheets]
                    if self.product_of(i) == name and not self.proven_wrong(i) and named(i) and self.states_own(i)]
            pool.sort(key=lambda i: (self.code_fit(i) is not True, i not in self.sheets, self.closeness(i)[3], self.latest_price(i) or 0.0, i))
            held = [i for i in self.cart_ids if i in pool]
            pools[name] = pool
            pick = verified.get(name) or (held[0] if held else (pool[0] if pool else ""))
            if not pick:
                return {}
            picks[name] = pick
        cap = self.budget_now
        total = lambda chosen: sum((self.latest_price(i) or 0.0) * self.line_quantity(i) for i in chosen.values())
        if isinstance(cap, (int, float)) and not isinstance(cap, bool) and total(picks) > cap + 1e-9:
            cheap = {n: (verified.get(n) or min(pools[n] or [picks[n]], key=lambda i: (self.latest_price(i) or 0.0, i))) for n in picks}
            picks = cheap if total(cheap) <= total(picks) else picks
        return picks

    def drop_gone_lines(self) -> None:
        """Sold-out lines beside buyable ones are taken out by the agent itself before an order (never on bundles, whose
        gone line needs a replacement first)."""
        stale = [i for i in self.cart_ids if i in self.gone]
        if not stale or len(self.cart_ids) <= len(stale) or self.bundle_names or self.companions or self.steps_left() < 2 + len(stale):
            return
        for ident in stale:
            removal = self.remove_args(ident)
            if removal is None or self.agent_action(removal[0], removal[1], f"{ident} can no longer be bought: taking it out before the order.", "drop_gone") is None:
                return
            self.cart_ids = [i for i in self.cart_ids if i != ident]

    def trim_bundle_lines(self) -> None:
        """A bundle's order never carries more lines than the request has products: per product the line kept is its
        verified pick, else its newest line; a second line of a product comes out (a line of no known product stays)."""
        if not self.bundle_names or len(set(self.cart_ids)) <= len(self.bundle_names) or self.finished or self.dead:
            return
        picks = self.bundle_picks()
        keep: dict[str, str] = {}
        for ident in reversed(self.cart_ids):
            name = self.product_of(ident)
            if name and picks.get(name) in self.cart_ids:
                keep[name] = picks[name]
            elif name:
                keep.setdefault(name, ident)
        kept = list(dict.fromkeys(keep.values()))
        # only a second line of a product known by its shelf and title comes out; a line of no known product stays
        extra = [i for i in self.cart_ids if i not in kept and self.product_of(i)]
        cart = self.roles.get("cart")
        if not extra or not kept or not cart or self.steps_left() < len(extra) + 2:
            return
        if self.basket_spec(cart):
            left = [i for i in self.cart_ids if i not in extra]
            if any(not self.cart_qty.get(i) for i in left):
                return                                # a held quantity unknown: the basket is not rewritten on a guess
            self.agent_action(cart, self.basket_args(left), "The order carries one line per product: setting the basket to " + ", ".join(left) + ".", "one_per_product")
            return
        for ident in extra:
            removal = self.remove_args(ident)
            if removal is None or self.agent_action(removal[0], removal[1], f"{ident} is a line beyond one per product: taking it out before the order.", "one_per_product") is None:
                return
            self.cart_ids = [i for i in self.cart_ids if i != ident]

    def last_reads(self) -> None:
        """The forced finish with no buyable fit known while steps remain: the nearest unread listings are opened (two at
        most) and a cart holding only lines that can no longer be bought is emptied, so the finish below can still order a
        fit instead of declining with candidates unread."""
        if any(f[0] not in self.gone for f in self.fits_by_facts() or []) or (self.cart_ids and not all(i in self.gone for i in self.cart_ids)):
            return
        unread = [i for i in self.nearest() if i not in self.sheets and i not in self.gone][:2]
        for ident in unread:
            if self.steps_left() < 5 or self.time_left() < 12:
                break
            calls = self.open_calls([ident])
            if not calls:
                break
            function = calls[0]["function"]
            if self.agent_action(function["name"], _parse_json_object(function["arguments"]) or {}, f"Opening {ident}: no buyable fit is known yet.", "last_read") is None:
                break
            if any(f[0] not in self.gone for f in self.fits_by_facts() or []):
                break
        if self.cart_ids and any(f[0] not in self.gone for f in self.fits_by_facts() or []) and self.steps_left() >= 3 + len(self.cart_ids):
            for ident in list(self.cart_ids):
                removal = self.remove_args(ident)
                if removal is None or self.agent_action(removal[0], removal[1], f"{ident} can no longer be bought: taking it out.", "last_read") is None:
                    return
                self.cart_ids = [i for i in self.cart_ids if i != ident]

    def repair_in_code(self, action: dict[str, Any], reply: dict[str, Any]) -> dict[str, Any] | None:
        """A forced order was refused and the model is not asked again (spend, time or a failed model): the refusal is read
        by code (a budget in figures, a line reported gone), then the held line is ordered again while it is still buyable
        within the limit, or the cheapest opened listing whose own sheet states every requirement takes its place; with
        neither, the caller declines. Returns the last reply."""
        observation = reply.get("observation")
        lines, _ = self.shopper_lines(action["name"], {}, observation, self.roles.get("message"))
        if isinstance(observation, list):
            _sentences([v for v in observation if isinstance(v, str)], lines)
        elif isinstance(observation, str):
            lines.append(observation)
        lines = list(dict.fromkeys(lines + self.notice_texts(observation)))
        budget_before = self.budget_now
        self.hear_in(lines)
        said = _compact(observation)
        if lines and not self.llm_dead and self.spent() < self.spend_caps()[2]:
            self.interpret(lines, None)              # one light reading of the refusal: gone, price moved, a new budget
        if len(self.cart_ids) == 1 and self.cart_ids[0] not in self.gone and not self.flags.get("price_moved") and self.budget_now == budget_before:
            self.reread_lines()                      # no reading, or one that named nothing: the line's own page says whether it is gone
        self.trace.append({"kind": "code_repair", "heard": lines[:3], "budget": self.budget_now, "gone": sorted(self.gone)[:6]})
        if len(self.cart_ids) != 1 or self.bundle_names or self.companions:
            return reply
        held = self.cart_ids[0]
        verified = dict(self.fits_by_facts())
        moved = self.budget_moved_at == self.step or bool(self.flags.get("price_moved"))
        if (moved and held in verified and held not in self.gone and self.code_ok(held)
                and self.affordable(self.sheets.get(held) or self.rows.get(held) or {})):
            reply = self.env_call([action])
            self.trace.append({"kind": "code_repair", "action": action, "reply": reply})
            return reply
        if held not in self.gone and self.budget_now == budget_before:
            return reply                            # a refusal the code cannot read: no swap on a guess (the caller declines)
        cart = self.roles.get("cart")
        fits = [f for f in self.fits_by_facts()
                if f[0] != held and self.code_ok(f[0]) and f[0] not in self.gone and self.affordable(self.sheets.get(f[0]) or self.rows.get(f[0]) or {})]
        removal = self.remove_args(held)
        if not fits or not cart or not removal or self.steps_left() < 4 or self.time_left() < 6:
            return reply
        best = self.forced_pick(fits)
        out = self.env_call([{"name": removal[0], "args": removal[1]}])
        if out is None or self.finished:
            return out
        self.cart_ids = []
        out = self.env_call([{"name": cart, "args": self.cart_args(best)}])
        if out is None or self.finished:
            return out
        self.cart_ids = [best]
        reply = self.env_call([action])
        self.trace.append({"kind": "code_repair", "swap": [held, best], "reply": reply})
        return reply

    def note_observation(self, name: str, observation: Any) -> str:
        notes = []
        if self.last_blind:
            notes.append("The agent could not read listing records from this result, so its checks cannot help here: read the text yourself (listing ids, prices, stock and facts are written in it), open the likely listings and decide from what they say.")
        if isinstance(observation, dict):
            _collect_ids(observation, self.seen_ids, 0, self.id_keys())
            faulted = [k for k, v in observation.items() if v and re.search(r"(?i)error|fault|problem|issue|hiccup|fail|trouble", str(k))]
            if name == self.roles.get("message") and status_code_reply(observation):
                faulted = faulted or ["code"]
            if faulted and name == self.roles.get("message"):
                # a refused message: the cap is discovered (stated or not, nothing more can be sent)
                self.message_budget = self.messages_used if self.message_budget is None else min(self.message_budget, self.messages_used)
                notes.append("That budget is used up; do not call this tool again.")
            if "refusal" in observation:
                self.refusals += 1
                self.refusal_at = self.step
                self.order_checked = False
                self.change_reason = "the order was refused: " + _compact(observation.get("refusal"))[:300]
                self.changes += 1
                notes.append("THE ORDER WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (remove any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again.")
                left = self.live_fits() if self.plan.get("priority") == "cheapest" else []
                if left:
                    notes.append("Already read and fitting by their own sheets, cheapest first: " + ", ".join(f"{i} ({p:g})" for i, p in left[:3]) + ".")
            elif len(observation) >= 2 and name != self.roles.get("message"):
                lines: list[str] = []
                for key, value in observation.items():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        _sentences(value, lines)
                if lines:
                    notes.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(lines[:4]))
        elif isinstance(observation, list):
            _collect_ids(observation, self.seen_ids, 0, self.id_keys())
        return " ".join(notes)

    def run_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        replies: list[tuple[str, str]] = []
        wrap_note = ""
        changes_before, refusals_before = self.changes, self.refusals
        allowed = {t["function"]["name"] for t in self.tools_for_turn()}
        runnable: list[tuple[str, str, dict[str, Any]]] = []
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        message_tool = self.roles.get("message")
        cart_tool = self.roles.get("cart")
        exploring = {self.roles.get("search"), self.roles.get("details"), self.roles.get("view")} - {None}
        committing = finishing | ({cart_tool} if cart_tool else set())
        planned_messages = 0
        held_removal = False
        for call in calls:
            call_id = call["id"]
            function = call.get("function") or {}
            name = function.get("name")
            raw = function.get("arguments")
            name, raw = self.as_virtual(name, raw)
            if name not in self.schemas:
                replies.append((call_id, f"Not run (no step used): '{name}' is not one of the tools. Use one of: {', '.join(sorted(allowed))}."))
                continue
            if self.bundle_names and not planned and (name in finishing or self.removal_target(name, _parse_json_object(raw) if isinstance(raw, str) else raw or {})):
                held = self.bundle_hold(name, _parse_json_object(raw) if isinstance(raw, str) else raw or {})
                if held:
                    replies.append((call_id, held))
                    return replies if name in finishing else replies + [(c["id"], "Not run (no step used).") for c in calls if c is not call]
            if not planned and name not in allowed:
                if name == self.roles.get("decline") and self.fits_open() and len(calls) == 1:
                    gate = self.decline_gate()          # a decline the agent withheld from the offer: the gate answers it
                    replies.append((call_id, gate or "Not run: the episode is over."))
                    return replies
                if name == message_tool and self.planner_on:
                    reason = "questions to the shopper are planned and sent for you after you have opened a few listings' details"
                else:
                    reason = "too few steps are left for this; only these tools can be used now: " + ", ".join(sorted(allowed))
                replies.append((call_id, f"Not run (no step used): {reason}."))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _parse_json_object(raw or "{}") if (raw or "").strip() else {}
                if args is None:
                    replies.append((call_id, "Not run (no step used): the arguments were not a valid JSON object."))
                    continue
            args, problems = shape_args(self.schemas[name], args, "args")
            if problems:
                replies.append((call_id, "Not run (no step used): invalid arguments: " + "; ".join(problems[:6])))
                continue
            if not planned:
                args, deaf = self.search_guard(name, args)
                if deaf:
                    replies.append((call_id, "Not run (no step used): " + deaf))
                    continue
            if name in finishing and len(calls) > 1:
                replies.append((call_id, "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls."))
                continue
            if name == message_tool and self.messages_used + planned_messages >= self.message_limit() - (0 if planned else self.held_messages()):
                kept = "" if self.messages_used + planned_messages >= self.message_limit() else " (the last one is kept for what the request says to tell the shopper at the finish, which is sent for you)"
                replies.append((call_id, "Not run (no step used): the message budget is used up" + kept + ". Do not message the shopper again."))
                continue
            signature = name + _compact(args)
            if not planned and name in exploring and not any(c.get("function", {}).get("name") in committing for c in calls):
                hold = self.explore_hold()
                if hold:
                    replies.append((call_id, "Not run (no step used): " + hold))
                    continue
            if name not in finishing and name != message_tool and name != cart_tool and signature in self.recent:
                replies.append((call_id, "Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else."))
                continue
            if not planned:
                kept = self.discard_hold(name, args)
                if kept:
                    held_removal = True
                    replies.append((call_id, "Not run (no step used). " + kept))
                    continue
            if held_removal and name == cart_tool and not self.removing_call(name, args):
                replies.append((call_id, "Not run (no step used): it came with a removal that was held, and adding beside the held line would leave two lines in the cart."))
                continue
            if name == cart_tool and self.steps_left() > 3:
                properties = self.schemas[name].get("properties") or {}
                if not self.removing_call(name, args):
                    target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)
                    if target and target in self.gone and self.steps_left() > 2:
                        self.gone_held.add(target)          # every time: the store said this line is gone, so it is never put back on a guess
                        replies.append((call_id, f"Not run (no step used): {target} was reported sold out by the store, so putting it back would be refused at the order. Choose another listing that states every requirement" + (": " + ", ".join(f"{i} ({p:g})" for i, p in self.live_fits()[:3] if i not in self.gone) if any(i not in self.gone for i, _ in self.live_fits()) else "") + "."))
                        continue
                    swapping = self.gone if self.companions else (self.cart_ids or self.gone)
                    if target and self.cart_used and target not in self.cart_ids and swapping:
                        exchange = self.ask_swap(target)
                        if exchange:
                            replies.append((call_id, "Not run (this call). " + exchange + " If they agree, send the same call again; otherwise follow what they said."))
                            continue
                    fallback = self.fallback_note(target) if target else ""
                    if fallback:
                        replies.append((call_id, "Not run (no step used). " + fallback))
                        continue
                    if self.finished or self.dead:
                        replies.append((call_id, "Not run: the episode is over."))
                        continue
                    if target and target not in self.shelf_held and self.wider_shelf(target):
                        self.shelf_held.add(target)
                        replies.append((call_id, "Not run (no step used). " + self.wider_shelf(target)))
                        continue
                    if target and self.wanted_codes() and not self.code_ok(target) and target not in self.fit_told and self.steps_left() > 4:
                        self.fit_told.add(target)
                        covering = [i for i in sorted(set(self.sheets) | set(self.pages)) if self.code_fit(i) and i not in self.gone and i != target]
                        why = ("its page names only other codes than " + ", ".join(self.wanted_codes()) if self.code_fit(target) is False
                               else "its page was never read, and the fitment lines of this product are on the pages: open it with " + str(self.page_tool()) + " first")
                        replies.append((call_id, f"Not run (no step used): {target} is not shown to fit the requested code: {why}."
                                        + (" Pages that cover the code: " + ", ".join(covering[:5]) + "." if covering else "")))
                        continue
                    if target and f"{target}|{self.ledger_version}" not in self.cart_checked:
                        self.cart_checked.add(f"{target}|{self.ledger_version}")
                        verdict = self.rule_check(target, True) or self.verify_item(target, "ADD TO THE CART")
                        if verdict:
                            replies.append((call_id, "Not run (no step used). " + verdict + " If you are certain it fits, send the same call again."))
                            continue
            if name == cart_tool and not self.basket_spec(name) and not self.removing_call(name, args):
                again = next((v for v in args.values() if isinstance(v, str) and v in self.cart_ids and self.mode_units.get(v, 0) >= self.line_quantity(v)), None)
                units = [v for k, v in args.items() if isinstance(v, int) and not isinstance(v, bool)]
                if again and (not units or units[0] == self.line_quantity(again)):     # a no-op where the tool sets the count, a double where it adds
                    replies.append((call_id, f"Not run (no step used): {again} is already in the cart at the needed quantity; adding it again would at best change nothing and may double the units."))
                    continue
            if name == cart_tool:
                warning = self.quantity_warning(name, args)
                if warning and signature not in self.qty_warned:
                    self.qty_warned.add(signature)
                    replies.append((call_id, "Not run (no step used): " + warning + " If you are sure, send the same call again."))
                    continue
            said_text = " ".join(str(v) for v in args.values() if isinstance(v, str)) if name == message_tool else ""
            if (name == message_tool and not planned and self.asked and not any(d["tell"] and not d["done"] for d in self.duties) and not self.novel_question(said_text)
                    and not self.second_look(said_text)):
                replies.append((call_id, "Not run (no step used): the questions to the shopper are done, one property each; another message would only re-explain, acknowledge or sum up, and reveals nothing. Go on with the task."))
                continue
            if name == message_tool and not planned and said_text.count("?") > 1 and signature not in self.told_once:
                self.told_once.add(signature)
                replies.append((call_id, "Not run (no step used): this message asks two things; the shopper answers only the first. Send one question about one property."))
                continue
            if (name == message_tool and not planned and "?" not in said_text and signature not in self.told_once
                    and not any(d["tell"] for d in self.duties)):
                self.told_once.add(signature)
                replies.append((call_id, "Not run (no step used): a message to the shopper must ask one question about one property; a statement or a confirmation only spends the message budget. If a rule in the request requires this message, send the same call again."))
                continue
            if name == message_tool:
                planned_messages += 1
            if not planned and self.removal_target(name, args):
                self.model_discarded.add(self.removal_target(name, args))   # the model's own choice: the agent never carts this for it again
            runnable.append((call_id, name, args))
        if runnable and runnable[0][1] in finishing:
            call_id, name, args = runnable[0]
            kind = self.intent(name, args)
            pick = self.cart_ids[-1] if self.cart_ids else ""
            fresh_note = ""
            cart_state = _compact(sorted(self.cart_ids)) + f"|{self.ledger_version}"
            soft_room = self.order_holds.get(cart_state, 0) < 1   # the order is held for a soft reason at most once per cart state
            if kind == "order" and self.cart_used and not self.cart_ids and self.cart_changed_at not in self.empty_order_warned:
                self.empty_order_warned.add(self.cart_changed_at)
                replies.append((call_id, "Not run (no step used): the cart is empty, so this order would be refused. Put the item that meets every requirement in first, then order; if none does, decline."))
                return replies
            if kind == "order":
                self.drop_gone_lines()
                self.trim_bundle_lines()
                if self.finished or self.dead:
                    replies.append((call_id, "Not run: the episode is over."))
                    return replies
                pick = self.cart_ids[-1] if self.cart_ids else ""
                cart_state = _compact(sorted(self.cart_ids)) + f"|{self.ledger_version}"
            if (kind == "order" and len(set(self.cart_ids)) > 1 and not self.bundle_names and not self.companions
                    and self.steps_left() > 2 and cart_state not in self.multi_held):
                self.multi_held.add(cart_state)
                replies.append((call_id, f"Not run (no step used): the request is for one product, but the cart holds {len(set(self.cart_ids))} listings ({', '.join(self.cart_ids)}). Take out every line but the one that meets every requirement best, then order."))
                return replies
            if kind == "order" and len(set(self.cart_ids)) > 1 and not self.bundle_names and not self.companions and self.steps_left() > len(self.cart_ids):
                # held once already: the agent keeps the line the request's rule ranks first among those that fit and takes out the rest
                fitting = [(i, p) for i, p in self.fits_open() if i in self.cart_ids and i not in self.gone]
                ranked = self.ranked_by_rule(fitting) if self.measure_words() else fitting
                keep = (ranked[0][0] if ranked and self.plan.get("priority") in ("cheapest", "best") else self.preferred(fitting) if fitting else "") or next((i for i in reversed(self.cart_ids) if i not in self.gone), "")
                for ident in [i for i in self.cart_ids if i != keep]:
                    removal = self.remove_args(ident)
                    if removal is None or self.agent_action(removal[0], removal[1], f"One product is wanted: keeping {keep}, taking {ident} out.", "one_line") is None:
                        break
                    self.cart_ids = [i for i in self.cart_ids if i != ident]
                pick = self.cart_ids[-1] if self.cart_ids else ""
            if kind == "order" and any(i in self.gone for i in self.cart_ids) and self.steps_left() > 2:
                pick = next(i for i in self.cart_ids if i in self.gone)
                replies.append((call_id, f"Not run (no step used): {pick} was reported sold out, so this order would be refused again. Take it out of the cart, put the next item that meets every requirement in (check its own properties first), then order."))
                return replies
            if kind == "order" and pick and self.first_add_at == self.step and not self.add_looked and self.steps_left() >= self.finish_reserve() + 6 and self.time_left() > WRAP_UP_SECONDS / 2:
                looked = self.post_add_look()
                if looked or self.finished or self.dead:
                    replies.append((call_id, looked or "Not run: the episode is over."))
                    return replies
                pick = self.cart_ids[-1] if self.cart_ids else ""
            if kind in ("order", "decline") and self.steps_left() > 2:
                early = self.early_finish_hold(kind)
                if early:
                    replies.append((call_id, "Not run yet (no step used). " + early))
                    return replies
            if kind == "order" and pick and self.steps_left() > 2 and soft_room:
                stale = self.long_order_hold(pick)
                if stale:
                    self.order_holds[cart_state] = self.order_holds.get(cart_state, 0) + 1
                    replies.append((call_id, "Not run yet (no step used). " + stale))
                    return replies
            if kind == "order" and pick and pick not in self.shelf_held and self.steps_left() > 2 and soft_room:
                shelf = self.wider_shelf(pick)
                if shelf:
                    self.shelf_held.add(pick)
                    self.order_holds[cart_state] = self.order_holds.get(cart_state, 0) + 1
                    replies.append((call_id, "Not run (no step used). " + shelf))
                    return replies
            if kind == "order" and not self.order_checked and self.steps_left() > 2 and self.time_left() > WRAP_UP_SECONDS:
                note = (self.together_note() or self.order_note(self.cart_ids[-1] if self.cart_ids else "")) if (soft_room or self.companions) else ""
                if note:
                    self.order_holds[cart_state] = self.order_holds.get(cart_state, 0) + 1
                    replies.append((call_id, "Not run yet (no step used). " + note))
                    return replies
                self.order_checked = True
                verdict = self.verify_cart()
                if verdict:
                    replies.append((call_id, "Not run yet (no step used). " + verdict))
                    return replies
                fresh = self.fresh_look(pick)
                if self.finished or self.dead:
                    replies.append((call_id, "Not run: the episode is over."))
                    return replies
                if pick in self.gone:
                    replies.append((call_id, f"Not run yet (one step used to re-read the pick). {fresh} {pick} is sold out: take it out, put the next fit in, then order."))
                    return replies
                if pick not in self.agent_placed and soft_room:   # a line the agent put in was already checked by code, and a cart state is held for the checklist once: no second checklist turn
                    self.order_holds[cart_state] = self.order_holds.get(cart_state, 0) + 1
                    replies.append((call_id, "Not run yet (" + ("one step used to re-read the pick" if fresh else "no step used") + "). " + (fresh + " " if fresh else "")
                                    + "Pre-order check: (1) the cart holds exactly one line, the chosen item at the required quantity, and nothing else; (2) its stock and price as last shown (the agent re-reads them for you after a change in the store) are in stock and price times quantity within the current budget; (3) its own properties meet every stated requirement, every need the shopper revealed and every change announced; (4) if the cheapest or best was asked for, no better fitting item exists among everything you could find; (5) every procedural rule in the request has been followed. If anything is unchecked, check it now; otherwise call the order tool again."))
                    return replies
                fresh_note = fresh
            if kind == "decline" and not self.companions:
                floor = self.bundle_floor() if self.bundle_names else self.view_floor()
                if floor or self.finished or self.dead:
                    replies.append((call_id, floor or "Not run: the episode is over."))
                    return replies
            if kind == "decline" and not self.bundle_names and not self.companions:
                held = self.hold_first()
                if held or self.finished or self.dead:
                    replies.append((call_id, held or "Not run: the episode is over."))
                    return replies
            if kind == "decline" and self.steps_left() > 3:
                paced = self.decline_pacing()
                if paced or self.finished or self.dead:
                    replies.append((call_id, paced or "Not run: the episode is over."))
                    return replies
            if kind == "decline":
                gate = self.decline_gate()
                if gate or self.finished or self.dead:
                    replies.append((call_id, gate or "Not run: the episode is over."))
                    return replies
                note = self.decline_note()
                if note:
                    replies.append((call_id, "Not run yet (no step used). " + note))
                    return replies
                if self.decline_checked and not self.search_guarded and self.steps_left() > 12 and self.searches_done < 8 and self.time_left() > WRAP_UP_SECONDS + 30:
                    self.search_guarded = True
                    replies.append((call_id, f"Not run yet (no step used). {self.steps_left()} steps remain and only {self.searches_done} searches were made. Search again in other ways first: the requested value in the sellers' words or as a fact value, the shelf or category listing, further pages; open the details of candidates you have not read and re-read the ones you opened. Decline only if each candidate still fails."))
                    return replies
                if self.decline_checked and self.audit_version != self.ledger_version and self.steps_left() > 4 and self.time_left() > WRAP_UP_SECONDS:
                    # the requirements moved since the last audit (a note on a finishing call, a later reply): every listing
                    # seen is checked again under the requirements now in force before a second decline goes out
                    self.audited = False
                    self.audit_version = self.ledger_version
                    audit = self.audit_decline()
                    if audit:
                        replies.append((call_id, "Not run yet (no step used). The requirements changed since the last check. " + audit))
                        return replies
                if not self.decline_checked and self.steps_left() > 3 and self.time_left() > FORCE_SECONDS + 25:
                    # every decline meets one review (late in the episode without the audit's judgement, which needs time)
                    self.decline_checked = True
                    self.audit_version = self.ledger_version
                    audit = self.audit_decline() if self.steps_left() > 6 else ""
                    if audit:
                        replies.append((call_id, "Not run yet (no step used). " + audit))
                        return replies
                    replies.append((call_id, "Not run yet (no step used). " + self.review_before_decline()))
                    return replies
                self.final_check("decline")
                args = self.with_reason(self.clean_decline(args, name), name)
                runnable[0] = (call_id, name, args)
                self.model_declined = True
            swap_note = ""
            if kind == "order":
                swap_note = (fresh_note + " " + self.enforce_cheapest()).strip()
                if self.finished or self.dead:
                    replies.append((call_id, "Not run: the episode is over."))
                    return replies
                hold = self.final_check("order")
                if hold:
                    replies.append((call_id, "Not run yet (no step used). " + hold))
                    return replies
            wrap_note, held = self.carry_duties(kind or "order")
            if swap_note:
                wrap_note = (swap_note + " " + wrap_note).strip()
            if held and self.steps_left() > self.finish_reserve() + 1:
                replies.append((call_id, "Not run yet (this call). " + wrap_note + " Apply what they said before finishing."))
                return replies
        if runnable and runnable[0][1] not in finishing and finishing:
            # the last two steps are kept for the finish (and one repair of a refused order): a reply with many reading calls
            # must not run the episode into its step limit with nothing ordered or declined
            keep = self.steps_left() - self.loop_floor()
            if keep < len(runnable):
                for call_id, name, _ in runnable[max(0, keep):]:
                    replies.append((call_id, "Not run (no step used): the last steps are kept for the finish; order (or decline) now."))
                runnable = runnable[: max(0, keep)]
            # past the order mark a reply opens at most two sheets: the four-to-six-sheet batches there push the episode
            # into the forced finish with nothing ordered
            readers = {self.roles.get("details"), self.roles.get("view")} - {None}
            if self.step >= self.pace_targets()[1] and sum(1 for _, n, _ in runnable if n in readers) > LATE_READS:
                kept_calls, opened = [], 0
                for call_id, name, args in runnable:
                    if name in readers:
                        opened += 1
                        if opened > LATE_READS:
                            replies.append((call_id, f"Not run (no step used): past step {self.pace_targets()[1]} at most {LATE_READS} sheets are opened per reply; order the fit you know (or decline) now."))
                            continue
                    kept_calls.append((call_id, name, args))
                runnable = kept_calls
        for start in range(0, len(runnable), 16):
            chunk = runnable[start : start + 16]
            if self.finished or self.dead or self.steps_left() <= 0:
                for call_id, _, _ in chunk:
                    replies.append((call_id, "Not run: the episode is over or no steps are left."))
                continue
            chunk = chunk[: self.steps_left()]
            actions = [{"name": name, "args": args} for _, name, args in chunk]
            before = self.step
            reply = self.env_call(actions)
            self.trace.append({"kind": "env", "step": before + 1, "actions": actions, "reply": reply})
            if reply is None:
                for call_id, _, _ in chunk:
                    replies.append((call_id, "The call failed (network or server). Try again."))
                continue
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            raws = reply.get("raw_observations") if isinstance(reply.get("raw_observations"), list) else [reply.get("raw_observation")]
            for index, (call_id, name, args) in enumerate(chunk):
                if index >= len(observations):
                    replies.append((call_id, "Not run: the episode finished before this call."))
                    continue
                observation = self.book_result(name, args, observations[index], reply, cart_tool, message_tool, finishing)
                note = self.note_observation(name, observation)
                heard, question = self.shopper_lines(name, args, observation, message_tool)
                if name != message_tool:
                    heard += [t for t in self.notice_texts(observation) if t not in heard]
                if heard:
                    read_by_code = self.hear_in(heard)
                    if read_by_code:
                        note = (note + " " if note else "") + "Read by the agent: " + "; ".join(read_by_code) + "."
                    meaning = self.interpret(heard, question)
                    if meaning:
                        note = (note + " " if note else "") + meaning
                raw = raws[index] if index < len(raws) else None
                text = self.result_text(observation if raw is None else raw, observation, name)
                if index == len(observations) - 1 and reply.get("finished"):
                    text += " [the episode is finished]"
                if note:
                    text += "\n[note] " + note
                replies.append((call_id, text))
        if wrap_note:
            replies = [(i, f"[{wrap_note}] " + t) if runnable and i == runnable[0][0] else (i, t) for i, t in replies]
        order = {c["id"]: i for i, c in enumerate(calls)}
        replies.sort(key=lambda r: order.get(r[0], 0))
        if not planned and replies and not self.finished and not self.dead and self.refusals > refusals_before:
            again = self.after_refusal()
            if again:
                call_id, text = replies[-1]
                replies[-1] = (call_id, text + "\n[order re-sent by the agent] " + again)
        if not planned and replies and not self.finished and not self.dead and self.changes > changes_before:
            reranked = self.rerank_after_note()
            if reranked:
                call_id, text = replies[-1]
                replies[-1] = (call_id, text + "\n[cart re-ranked by the agent] " + reranked)
        if not planned and replies and not self.finished and not self.dead and self.changes > changes_before:
            repaired = self.recover()
            if repaired:
                self.repaired_by_code = True
                ordered = self.order_now(repaired)   # the repaired cart goes straight to the order path, with every check and duty
                call_id, text = replies[-1]
                replies[-1] = (call_id, text + "\n[cart repaired by the agent] " + repaired + (" " + ordered if ordered else ""))
        if self.repaired_by_code:
            self.repaired_by_code = False
            self.change_reason = ""                   # the agent acted on the change itself: no review turn is owed
        if self.change_reason and replies and not self.finished:
            review = self.change_review()
            if review:
                call_id, text = replies[-1]
                replies[-1] = (call_id, text + "\n[change review] " + review)
        return replies

    def reading_tools(self) -> set[str]:
        return ({self.roles.get("details"), self.roles.get("view"), self.roles.get("compare")} | self.fact_tools) - {None, ""}

    def note_reading(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """After a search or a reading call: the listings it opened are marked viewed, and a result in which the code
        found no listing record at all (an encoding it cannot read) is marked blind, so the model reads its text itself."""
        others = {self.roles.get(r) for r in ("search", "cart", "remove", "message", "order", "decline", "cart_view", "stock")} - {None, ""}
        reading = name in self.reading_tools() or (name not in others and any(
            isinstance(one, str) and _ID_HINT.match(one.strip()) and re.search(r"\d", one)
            for value in args.values() for one in (value if isinstance(value, list) else [value])))
        if reading:
            for value in args.values():
                for one in value if isinstance(value, list) else [value]:
                    if isinstance(one, str) and _ID_HINT.match(one.strip()):
                        self.viewed.add(one.strip())
        self.last_blind = False
        if name != self.roles.get("search") and not reading:
            return
        single, rows = self.records_in(normal(observation)) if isinstance(observation, (dict, list)) else (None, None)
        found = bool(single is not None and self.id_in(single)) or any(self.id_in(r) for r in rows or [])
        found = found or (reading and any(i in self.sheets for i in self.viewed))
        if not found and len(_compact(observation)) >= 240:
            self.blind_results += 1
            self.last_blind = True

    def view_floor(self) -> str:
        """A single product is never declined before a couple of listings were opened: the agent opens the most promising
        unread rows itself, or, with no row it could read, asks the model to open listings named in the results. At most
        twice per episode, never into the last steps. Returns the reply for the model ("" when the decline may go on)."""
        need = 2 - len(self.viewed)
        if need <= 0 or self.bundle_names or self.floor_rounds >= 2 or self.finished or self.dead or not (self.roles.get("details") or self.roles.get("view")):
            return ""
        if self.steps_left() < need + 4 or self.time_left() < WRAP_UP_SECONDS or not (self.rows or self.blind_results):
            return ""                             # nothing shown that could be opened: the decline goes on to its gates
        self.floor_rounds += 1
        pool = [i for i in self.sweep_pool() if i not in self.viewed] or [i for i in self.rows if i not in self.viewed and i not in self.gone]
        calls = self.open_calls(pool[:need]) if pool else []
        if calls:
            self.own_rounds(calls[:need], "Before any decline at least two listings are opened and checked: " + ", ".join(pool[:need]) + ".")
            fits = [f"{i} ({p:g})" for i, p in self.fits_open()[:3]]
            return ("Not run (the decline was not sent): the agent opened " + ", ".join(pool[:need]) + " first, since a decline needs at least two listings read. "
                    + ("These fit every requirement in force by their own sheets: " + ", ".join(fits) + ". Put the best in the cart and order it." if fits
                       else "Check them against every requirement; order one that fits, or decline again if none does."))
        return ("Not run (no step used): no listing has been opened yet, and a decline scores 0 whenever one fits. Open at least two listings from the results above "
                "with the reading tool (their ids are written in the result text), check each against every requirement, and order one that fits; decline only if each fails.")

    def bundle_floor(self) -> str:
        """Bundles: no decline before every product was searched and its best candidate opened; the agent does both
        itself, one product at a time while the steps allow. Returns the reply ("" when the decline may go on)."""
        if (not self.bundle_names or self.floor_rounds >= 2 or self.finished or self.dead or self.time_left() < WRAP_UP_SECONDS
                or not (self.roles.get("details") or self.roles.get("view"))):
            return ""
        todo = [n for n in self.bundle_names if not any(self.product_of(i) == n for i in self.viewed)]
        if not todo or self.steps_left() < 2 * len(todo) + 4:
            return ""
        self.floor_rounds += 1
        done = []
        for name in todo:
            if self.finished or self.dead or self.steps_left() < 6:
                break
            if not self.bundle_unread(name) and name not in self.bundle_searched:
                self.bundle_search(name)
            unread = [i for i in self.bundle_unread(name) if i not in self.viewed]
            calls = self.open_calls(unread[:1]) if unread else []
            if calls:
                self.own_rounds(calls[:1], f"Before any decline every product's best candidate is opened: {unread[0]} for the {name}.")
                done.append(f"{unread[0]} for the {name}")
        missing = self.bundle_missing()
        return ("Not run (the decline was not sent): before a bundle is declined every product is searched and its best candidate opened"
                + ("; the agent opened " + ", ".join(done) if done else "") + ". "
                + ("Still without a verified line: " + ", ".join(missing) + ". " if missing else "")
                + "Order the full cart when each product has a fit; decline only if one product truly has none.")

    def book_result(self, name: str, args: dict[str, Any], observation: Any, reply: dict[str, Any], cart_tool: str | None,
                    message_tool: str | None, finishing: set[str]) -> Any:
        """The bookkeeping after one call ran: counts, refusal reading, cart lines, records. Returns the observation as
        the model should read it (an order that did not end the episode is marked refused)."""
        if name == message_tool:
            self.messages_used += 1
            self.sent_texts.append(" ".join(str(v) for v in args.values() if isinstance(v, str))[:400])
        if self.intent(name, args) == "decline" and not reply.get("finished"):
            self.decline_refused_at = self.step         # the decline came back with the shopper's words: the episode goes on
        if self.intent(name, args) == "order":
            self.order_attempts += 1
            if not reply.get("finished") and not (isinstance(observation, dict) and "refusal" in observation):
                if isinstance(observation, dict) and any(re.search(r"(?i)token|confirm", str(k)) for k in observation):
                    observation = dict(observation, note="the checkout waits for a further step: take it now with the tool that finishes the checkout; nothing is bought until the episode ends")
                else:
                    self.refusals += 1
                    self.order_checked = False
                    observation = {"refusal": observation} if not isinstance(observation, dict) else dict(observation, refusal=observation.get("refusal") or "not completed")
            if not reply.get("finished") and self.cart_ids:
                self.refused_text = _compact(observation)   # the reader says what the refusal means; which line it hits comes from it or from re-reads
        if name == cart_tool:
            self.cart_used = True
            self.order_checked = False
        if name in (cart_tool, message_tool) or name in finishing:
            self.recent = []
        else:
            self.recent = (self.recent + [name + _compact(args)])[-2:]
        self.records.append(observation)
        self.last_target = next((v for v in args.values() if isinstance(v, str) and _ID_HINT.match(v)), "") if name in (self.roles.get("details"), self.roles.get("view")) else ""
        self.remember(name, observation)
        self.note_reading(name, args, observation)
        self.note_page(name, observation)
        self.note_list(name, args, observation)
        for ident in self.ids_shown(observation):
            self.seen_at[ident] = self.step
        if name == self.roles.get("search"):
            self.note_search(args, observation)
            if self.wanted_codes() and any(isinstance(v, str) and any(code_family(t) in {code_family(w) for w in self.wanted_codes()} for t in CODE_TOKEN.findall(v)) for v in args.values()):
                if self.match_count(observation)[0] == 0 or not self.match_count(observation)[1]:
                    self.code_misses += 1          # the store's search does not know the code

        remover = self.roles.get("remove")
        if name == cart_tool or (remover and name == remover):
            properties = self.schemas[name].get("properties") or {}
            signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
            removing = self.removing_call(name, args)
            spec = self.basket_spec(name)
            if spec and isinstance(args.get(spec[0]), list):
                lines = [l for l in args[spec[0]] if isinstance(l, dict) and isinstance(l.get(spec[1]), str)]
                self.cart_ids = list(dict.fromkeys(l[spec[1]] for l in lines))
                self.cart_qty = {l[spec[1]]: int(l[spec[2]]) for l in lines if spec[2] and isinstance(l.get(spec[2]), (int, float)) and not isinstance(l.get(spec[2]), bool)}
                args = {}
            for key, value in list(args.items()):
                if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    for one in dict.fromkeys(value):    # a list of listing ids: each one goes in (or out)
                        self.cart_state[one] = signature
                        self.cart_ids = [i for i in self.cart_ids if i != one] + ([] if removing else [one])
                    break
            for key, value in args.items():
                spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
                if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip():
                    self.cart_state[value] = signature
                    if not removing:
                        self.cart_ids = [i for i in self.cart_ids if i != value] + [value]
                        units = [v for k, v in args.items() if isinstance(v, int) and not isinstance(v, bool) and "enum" not in (properties.get(k) or {})]
                        if units:                          # the most units this line was put in with: a removal takes them all
                            self.mode_units[value] = max(self.mode_units.get(value, 0), units[0])
                    else:
                        self.cart_ids = [i for i in self.cart_ids if i != value]
                        self.mode_units.pop(value, None)
                    break
            self.cart_used = True
            self.cart_changed_at = self.step
            if self.first_add_at < 0 and self.cart_ids:
                self.first_add_at = self.step
        if name in (cart_tool, remover, self.roles.get("cart_view")) and name and isinstance(observation, dict) and not reply.get("finished"):
            _, lines = self.records_in(normal(observation))
            held = [self.id_in(r) or next((v for v in r.values() if isinstance(v, str) and v in self.seen_ids), None) for r in lines or []]
            if lines and all(held):
                self.cart_ids = list(dict.fromkeys(h for h in held if h))   # the cart's own lines, as the store lists them
        if name == self.roles.get("cart_view"):
            self.last_cart_view = observation
        return observation

    @staticmethod
    def shopper_lines(name: str, args: dict[str, Any], observation: Any, message_tool: str | None) -> tuple[list[str], str | None]:
        """The shopper's sentences in a result (the answer to a message, or a change carried by another result) and the
        question that drew them."""
        heard: list[str] = []
        if name == message_tool:
            if status_code_reply(observation):
                heard = []                          # a status code (a cap or an unknown-call code), not the shopper's words
            elif isinstance(observation, str):
                heard = [observation]
            elif isinstance(observation, dict) and not any(re.search(r"(?i)error|fault|problem|issue|hiccup|fail|trouble", str(k)) for k in observation):
                _sentences(list(v for v in observation.values() if isinstance(v, (str, list))), heard)
            question = " ".join(str(v) for v in args.values() if isinstance(v, str))
        else:
            question = None
            if isinstance(observation, dict):
                for value in observation.values():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        _sentences(value, heard)
                    elif isinstance(value, str) and len(value.split()) >= 4 and ("refusal" in observation):
                        heard.append(value)
        return heard, question

    def opening(self) -> list[dict[str, Any]]:
        """The first two messages: the playbook, then the request, the opening observation, the plan and the status."""
        opening = self.unpacked(self.policy.get("opening"), None)
        parts = ["SHOPPER REQUEST:\n" + str(self.policy.get("query") or "")]
        if opening is not None:
            parts.append("OPENING OBSERVATION:\n" + _raw_text(self.policy.get("opening")))
            self.records.append(opening)
            _collect_ids(opening, self.seen_ids, 0, self.id_keys())
        parts += [x for x in (self.brief(), self.status()) if x]
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]

    def model_calls(self, message: dict[str, Any], counter: int) -> tuple[list[dict[str, Any]], int]:
        """The reply's tool calls in one shape (an id each, arguments as a JSON string)."""
        calls = []
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            counter += 1
            function = call.get("function") or {}
            arguments = function.get("arguments")
            calls.append({"id": call.get("id") or f"call_{counter}", "type": "function",
                          "function": {"name": function.get("name"), "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {})}})
        return calls, counter

    def turn(self, state: dict[str, int]) -> bool:
        """One model turn: returns False when the loop should stop and the agent finish by itself."""
        limits = [("time", self.time_left() < FORCE_SECONDS), ("steps", self.steps_left() <= self.loop_floor()), ("spend", self.spent() >= self.spend_caps()[2]),
                  ("model", self.llm_dead), ("turns", self.tool_turns >= self.max_steps + TURN_SLACK)]
        fired = [name for name, hit in limits if hit]
        if fired:
            self.stop_reason = ",".join(fired)     # which limit ended the model's turns (kept in the end-of-episode record)
            return False
        if self.pending_reply:
            text, self.pending_reply = self.pending_reply, ""
            if self.send_reply(text):
                return True
        if self.planner_on and self.ask_planned():
            return True
        if any(not p["done"] for p in self.pointers):
            self.resolve_pointers()
            if self.finished or self.dead or self.steps_left() <= self.loop_floor():
                return False
        if (self.plan.get("priority") == "cheapest" and self.steps_left() >= PAGE_FROM_STEPS
                and self.max_steps >= 2 * PAGE_FROM_STEPS and self.order_attempts == 0 and self.walk_room()):
            self.page_out()                         # a short worded list the model left half-read is paged to its end before the model goes on (only when the episode is long enough to afford it)
        if self.read_fitment() and (self.finished or self.dead or self.steps_left() <= self.loop_floor()):
            return False
        self.sweep_reads()                          # the most promising unread rows of the newest search are read by the agent
        if self.finished or self.dead or self.steps_left() <= self.loop_floor():
            return False
        self.commit_cart()                          # a fit known and the order due: the agent fills the cart, the model orders
        if self.bundle_names and not self.finished and not self.dead and self.steps_left() > self.loop_floor() and self.parse_trusted():
            note = self.fill_bundle()               # several products: the agent reads on, fills one line per product and orders
            if note and note != "looked" and not self.finished:
                self.messages.append({"role": "user", "content": "[bundle, done by the agent] " + note})
        if self.finished or self.dead or self.steps_left() <= self.loop_floor():
            return False
        models = [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL] if self.spend_level() == 0 else [MAIN_MODEL, MAIN_MODEL]
        self.fold_history()
        self.tool_turns += 1
        message = self.infer(self.turn_messages(), self.tools_for_turn(), models, 2500)
        if message is None:
            state["failures"] += 1
            return state["failures"] < 2 and not self.llm_dead
        calls, state["counter"] = self.model_calls(message, state["counter"])
        content = message.get("content") or ""
        content = content if isinstance(content, str) else _compact(content)
        self.last_said = content
        committing = {self.roles.get("cart"), self.roles.get("order"), self.roles.get("decline")} - {None}
        unasked = self.messages_used == 0 and bool(self.roles.get("message")) and self.message_limit() - self.held_messages() > 0 and not self.forced_ask
        buying = any(c["function"]["name"] in committing - {self.roles.get("decline")} for c in calls)
        if ((self.planner_on or unasked) and any(c["function"]["name"] in committing for c in calls) and self.ask_planned(force=True)
                or buying and self.buy_question("order" if any(c["function"]["name"] == self.roles.get("order") for c in calls) else "cart")):
            names = ", ".join(dict.fromkeys(c["function"]["name"] for c in calls))
            self.messages.append({"role": "user", "content": f"Your call(s) to {names} were not run: the planned questions went first. Send them again now if you still want them."})
            return True
        if self.plan.get("priority") == "cheapest" and self.committed_to(calls) and self.descend(self.committed_to(calls)):
            return True
        if not calls:
            state["idle"] += 1
            self.messages += [{"role": "assistant", "content": content or "(no tool call)"},
                              {"role": "user", "content": "Every reply must call at least one of the supplied tools. Continue."}]
            return state["idle"] < 3
        state["idle"] = 0
        self.trace.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in calls]})
        self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        replies = self.run_calls(calls)
        for index, (call_id, text) in enumerate(replies):
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        return True

    def fold_history(self) -> None:
        """Past HISTORY_CHARS of conversation (a third of that past the hard spend line) the oldest long tool results are
        folded to their head and the listing ids they named, oldest first, until the conversation is back under 60% of the
        line. Done in one pass, so the prefix stays byte-identical (and cached) until the next fold."""
        limit = HISTORY_CHARS if self.spend_level() < 2 else HISTORY_CHARS // 3
        size = sum(len(m["content"]) for m in self.messages if isinstance(m.get("content"), str))
        if size <= limit:
            return
        known = set(self.rows) | set(self.sheets) | set(self.cart_ids)
        last = len(self.messages) - 6
        for index, message in enumerate(self.messages):
            if size <= 0.6 * limit or index >= last:
                break
            content = message.get("content")
            if message.get("role") != "tool" or not isinstance(content, str) or len(content) <= 900 or content.startswith("[folded]"):
                continue
            named = [i for i in known if i in content][:24]
            folded = "[folded] " + content[:600] + " ..." + (" listings named: " + ", ".join(named) if named else "")
            self.messages[index] = {**message, "content": folded}
            size -= len(content) - len(folded)
        self.trace.append({"kind": "fold", "chars": size})

    def turn_messages(self) -> list[dict[str, Any]]:
        """The conversation as sent: the stored messages unchanged (so every turn's prefix is byte-identical to the last
        and is read from the cache) with the current status line added to the newest message only."""
        if not self.messages:
            return self.messages
        last = dict(self.messages[-1])
        if last.get("role") in ("tool", "user") and isinstance(last.get("content"), str):
            last["content"] = last["content"] + "\n" + self.status()
        return self.messages[:-1] + [last]

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.setup()
        self.trace.append({"kind": "plan", "plan": self.plan, "roles": self.roles})
        self.messages = self.opening()
        if self.message_budget is None and self.roles.get("message"):
            self.probe_allowance()                  # no cap in the request: the store's allowance report is read before any message
        state = {"idle": 0, "failures": 0, "counter": 0}
        for _ in range(MAX_TURNS):
            if self.finished or self.dead or self.steps_left() <= 0 or not self.turn(state):
                break
        if not self.finished and not self.dead:
            self.force_finish()                     # the one exit: time, steps, silence or failures all end here
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "spent": round(self.spent(), 5), "tool_turns": self.tool_turns, "model_given_up": self.llm_dead, "stopped_by": getattr(self, "stop_reason", ""), "seconds": round(time.monotonic() - self.started, 1)})
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
