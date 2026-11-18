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
# --- race spend governor (champ21) ------------------------------------------------------------------------------------
# A race run is about 90 episodes on one miner key capped at $5; the shopper simulator and the judge bill the same key,
# which leaves about $0.045 a task for the agent itself. The lines below compare the figure the proxy reports (no scaling):
#   soft  0.030: the heavy model gives way to the light one for every check; the window keeps fewer full old results.
#   hard  0.046: the backup model (dearer than the main one) is no longer tried; questions and searches go on.
#   stop  0.060: no model call at all; the code finishes from what it read (its own reads, the cart, the order).
# Money never removes a search or a question from the model: past each line the episode moves to cheaper calls instead.
BILL_SCALE = 1.0            # the proxy's own cost figure is used as reported
HARD_FLOOR = 0.046          # the hard line of a step cap up to HARD_FROM ...
HARD_FROM = 40
HARD_PER_STEP = 0.0001      # ... grows this much per step above it ...
HARD_ROOF = 0.05            # ... up to this
SOFT_SHARE = 0.65           # the soft line as a share of the hard line
STOP_OVER = 0.014           # the stop line sits this far above the hard line (about $0.06)
HEAVY_AFTER_SOFT = 2        # heavy-model calls an episode may still make once past the soft line
TOOL_TURN_SLACK = 6         # model tool turns per episode at most: the step cap plus this
LLM_FAIL_LIMIT = 4          # consecutive failed model calls (429, 5xx, network, empty) after which the episode stops calling
LEAN_TABLE_ROWS = 8         # ... and the listings table under the newest result this many rows
NOTE_CALLS_LATE = 2         # note-reader calls an episode may still make once past the soft line
LONG_WORLD = 40             # a step cap of at least this many steps makes a long world: more rounds of care
LONG_LOOKS = 3              # in a long world the look before the finish runs again for each new cart state, up to this many times
LONG_RECOVERIES = 2         # in a long world the code may recover twice from the same announced change
PRE_ADD_READS = 3           # in a long world, before the first add with fewer than three sheets read, the code opens up to this many
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
OBS_CHARS = 7000
BLIND_ASKS = ("material", "intended use")   # what a question names when no listing's sheet could be read
MAX_TURNS = 80              # model turns per episode at most; the cap of one episode is its step cap plus TURN_SLACK, at least MIN_TURNS
MIN_TURNS = 20
TURN_SLACK = 8
HISTORY_CHARS = 40000       # the model sees the whole episode while the transcript stays under this many characters ...
HISTORY_KEEP = 26000        # ... past it, the oldest tool results are cut to their head until the rest fits in this many
OLD_HEAD_CHARS = 700        # ... each such old result keeps this much (plus the listing ids it showed)
RAW_FIELD_CHARS = 600       # the model reads the store's own result: a text under a key is cut to this many characters ...
RAW_RECORD_CHARS = 1500     # ... a bare text in a list (often a whole record written as one string) to this many ...
RAW_RESULT_CHARS = 9000     # ... and one result to this many
RETRY_WAITS = (1.2, 3.0, 6.0)   # seconds before the 2nd, 3rd and 4th try of a call that met 429/5xx/no answer (plus a jitter)
RETRY_STATUSES = (None, 408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529)
DIGEST_SHEETS = 6           # listings whose full property sheet travels with the digest of the earlier turns
DIGEST_SHEET_CHARS = 900   # ... each at most this long
RECORD_CHARS = 4500         # a listing record kept on the shelf (what the checks read) is at most this long
PLACED_HOLDS = 2            # removals of the code's own cart line held before the next one goes through
WALK_STEPS = 4              # store steps the code may spend per episode walking the price limit down or paging a search
WALK_PAGES = 2              # the no-text walk runs only when the list under the limit fits in this many result pages
KEYWORD_LIST = 100          # the model's own keyword search is paged to the end when it reports no more hits than this
WALK_FROM = 14              # the walk starts only while at least this many steps remain
EARLY_PAGE_FROM = 12        # on a cheapest request the code pages a short keyword search to its end while this many steps remain
REFILL_STEPS = 4            # store steps the code may spend after a stockout looking for another fit at the cheapest fitting price
BLIND_TOTAL = 30            # a category search of this many hits or more that returns the same count for different words ignores them
PACE_FIT = 0.4              # share of the step cap by which the first verified fit should be known
PACE_COMMIT = 0.5           # share of the step cap by which the first order attempt should have gone out
BUNDLE_BY = 0.7             # share of the step cap by which a request for several products must have a line per product, or the held fits are ordered
INTERVIEW_FROM = 3          # sheets opened before the code-owned interview starts (or a fifth of the cap in steps)
INTERVIEW_RESERVE = 6       # steps kept free of questions
SURVEY_SHEETS = 6           # sheets of a product the code wants open before its first question (the property menu is read off them)
SURVEY_READS = 4            # ... the code opens at most this many itself for that, once per product ...
SURVEY_SPARE = 12           # ... and only while this many steps stay free beyond the finish reserve and the questions left
SURVEY_PER_PRODUCT = 4      # ... plus this many for each further product of the request
RANK_MODEL_FROM = 2         # candidate properties from which a cheap model call orders the code's shortlist by the story
ZERO_SEARCHES = 2           # free-text searches with no result after which the code falls back to the category and price filter
ZERO_HOLDS = 3              # how many later wordy searches are held (no step) after that fallback
AUDIT_CHECKS = 3            # before a decline goes out, the opened candidates the code has judged itself, at most this many
POINTER_STEPS = 9           # store steps the code may spend per episode following a need that points at another listing's page
POINTER_READS = 3           # sheets opened per round while following such a pointer
REFUSAL_BLOCK = 3           # steps after a refused finish during which a decline is held while an opened sheet fits
DECLINE_BLOCKS = 2          # declines held with a fitting sheet opened before the code carts and orders that sheet itself
COMMIT_STEPS = 12           # with this many steps left and a verified fit known, the episode only commits
COMMIT_SHARE = 0.5          # once the questions are done, from this share of the step cap the code itself puts a known fit in an empty cart
CODE_MOVES_FROM = 24        # the code's own cart moves (the commit, the recovery after a stockout) need a step cap of at least this
LATE_READS = 2              # past the pace schedule's commit mark one reply may open at most this many sheets or views
# --- question width (champ22): the width adapts to the shopper within one episode ---------------------------------------
# Shoppers differ from race to race: one reads a whole list of properties and names the one it holds a need about, another
# refuses any message about more than one subject ("one at a time", "split them up"). So the first code question names
# several planned properties of ONE product (never two products in one message), and the replies decide what follows:
#   a crowding complaint     -> one property a message for the rest of the episode; the complained message is put again,
#                               one property at a time, starting with its first property
#   a reply it did not follow -> the width is halved and the first half is put once more in plainer words
#   a wave-off of the list    -> every property it named is settled (free); the next message names the next planned ones
KEYS_PER_MESSAGE = 3        # properties a code question names at most (fewer when fewer are planned); race 18: width 3 revealed 12-13 %, width 5 only 4 %
FIRST_WIDTH = 3             # ... the episode's first question at most this many
KEYS_AFTER_FIRST = 1        # a shopper who answers only the first property named, or who complained of crowding: one a message
WIDE_WORDS = 4              # a property label longer than this many words stays out of a list (a list of long labels reads as wordy)
CROWD_REDOS = 3             # questions the code puts again after crowding complaints, per episode
PLAIN_RETRIES = 1           # a question the shopper did not follow is put once more in plainer words (once an episode, and only
                            # while two messages are left), then counted as asked
LATE_ASKS = 1               # questions put with the message kept for a decline duty when the episode is about to order instead
ASKS_AFTER_NEED = 1         # questions a product may still get once a firm need of it was heard (long worlds only)
UNSTATED_MESSAGES = 4       # questions the code may put when the request states no message limit (one property each)
NOTE_CALLS = 8              # cheap model reads of free-text notes per episode at most
FIT_PER_SEARCH = 3          # under a fitment code, pages the code opens per new search (the cheapest unread rows within the limit)
FIT_PER_ROUND = 6           # ... and per round at most
FIT_MIN, FIT_MAX = 6, 30    # pages the code opens for the codes per episode: FIT_SHARE of the step cap, within these bounds
FIT_SHARE = 0.35
BUNDLE_PAGE_FROM = 50       # several products and a step cap of at least this: the code reads short result lists to their end ...
BUNDLE_PAGES = 6            # ... at most this many pages per episode ...
BUNDLE_LIST_MAX = 40        # ... of lists reporting at most this many rows ...
CODE_BUNDLE_READS = 3       # past the money: the code opens this many rows a round for a product without a known fit ...
CODE_BUNDLE_TRIES = 4       # ... for at most this many rounds per product
BUNDLE_PAGE_READS = 2       # ... opening the cheapest unread rows within the limit of each new page
BROAD_PER_SEARCH = 4        # sheets the code itself opens after one search at most (the cheapest unread rows of the kind within the limit) ...
BROAD_SHARE = 0.3           # ... and per episode at most this share of the step cap
BROAD_SPARE = 6             # steps beyond the finish reserve and the questions still open that stay untouched by them
BROAD_FROM = 32             # ... only in a world whose step cap is at least this (a short world keeps its steps for the model's own reads)
BROAD_PER_STEP = 3          # one code read per this many spare steps
LONG_CAP = 30               # a step cap of at least this many steps: declines wait until LONG_DECLINE_BY of it while candidates are unread
LONG_DECLINE_BY = 0.7
REFUSAL_ROOM = 2            # steps kept, until a first finish was refused, for repairing a refused checkout (race 18: a dropped
                            # requirement came as a checkout refusal in about 25 of 90 tasks, and champ23 often sent its first
                            # finish on the last step)

SYSTEM_PROMPT = """You are a careful shopping assistant working for one shopper inside a simulated store, acting only through the supplied tools. The task scores 1 or 0: 1 only when the finish is fully right (the ordered listing meets every stated requirement, every need the shopper reveals when asked and every change announced while you work; quantity and budget are right; the cheapest or best choice is taken when the request asks for it; every rule in the request is followed). When nothing in the store can meet everything, the right finish is a decline, worded the way the request says.

READING RESULTS
- Every tool result is shown as the store wrote it (long texts cut). Notes marked [code read] or [note] and the [status] block are the code's own reading of the results: ids, prices, stock and the property sheets it could parse. When a result's own text shows listings, ids or values the code's reading lacks, the result's text is right: use the ids and values exactly as the store wrote them.

STEPS AND TOOLS
- Every tool call costs one step, failed calls included; thinking is free. Independent read-only calls (searches, sheets) may share one reply and run in order; a message, a cart change, an order or a decline travels alone.
- Stores differ: read each tool's schema and use its exact argument names and enum values; learn from the first results which tool shows prices, stock and the property sheet. Find the store's category labels early and stay in the one matching the request's wording as a whole ("A or B" means the combined label "A & B", not the narrower "B"). Results may be capped, paged or sorted by id: compare the reported total with the rows shown and narrow down (words, category, price ceiling, pages) until the relevant set is fully seen.

THE SHOPPER
- The shopper holds needs they did not write down, revealed only when a message names that property, and a reply settles at most one property: a message naming several gets a muddled answer or none. The code plans and sends the questions for you (one property of one product each, the likeliest hidden need first, each chosen after the answer before it was read) once you have opened the details of two or three listings of each requested product, so do that in your first moves; the answers appear in the conversation with their meaning. If you must ask yourself: ONE property per message, in the form "Which <property> would you like for the <product>?", the property named as the listings name it, the product by its plain noun, no example values, one question mark. Never ask what the request already settles and never exceed the message budget; a question the shopper did not follow may be put once more in shorter, plainer words, after which the property counts as free.
- "No preference" or "anything is fine" frees a property; a stated preference or limit is a hard requirement from then on, exactly as worded. Insistence idioms ("X is non-negotiable") mean X is wanted, never excluded.
- Requirements are only what the request states and what the shopper reveals or changes. Never add one from the story (an age, a size, who will use it); if such a detail may matter, it is a question, not a condition. Something merely wished ("ideally", "would be nice") only breaks ties between full fits and never outranks a lowest-price rule. What only the shopper can tell you (a missing price limit, which of two wishes matters more) is asked before choosing.
- Messages are for questions only (and for answering what the shopper asks). Never report findings, choices or a decline by message; the finishing tool carries that. Procedural rules in the request (what to check, what to tell, how to decline) are graded like conditions: follow each literally.

CHANGES
- The limit may move up or down, conditions may be dropped, replaced or added, listings may change price or sell out, with a notice hidden in a tool result (an extra field of sentences) or riding on an order attempt. Each change binds from that moment; under a lowest-price rule choose again over the new acceptable set. A price change alone never disqualifies a listing while price times quantity stays within the limit.
- An order can be REFUSED: the reason sits in some field of the result even when the inner payload looks successful. Only an order that ends the episode buys anything; while the episode runs, nothing is bought. After a refusal apply every change it states, take what no longer qualifies out of the cart (a sold-out line), put the next verified fit in, re-check its stock and price, and order again.

JUDGING A LISTING (this decides the score)
- A requirement is met only when the listing's own property for it states the value. Keys vary in spelling (spaces, dots, underscores, prefixes) and one property may sit under several keys; any key naming it counts. Values match by meaning: one element of a comma list is enough, singular equals plural, close synonyms count, case is ignored; a word that merely starts the same way, or a broader or different class, does not count (girls is not women, plain unisex is not female). When several keys name the property (a plain 'style' beside 'skirt style', 'shape' beside 'clock_shape', any spelling or namespace), the listing meets it when ANY of them states the value; the others do not cancel it. The code marks fits with these rules.
- Title or description count only when the listing has no key for that property at all; when the property says otherwise, the listing fails. A related property does not substitute (a lining's material is not the material). Numbers compare after unit conversion, edges included; a range counts only when all of it meets the limit. Respect exclusions and closed lists.
- The requested kind is met by the listing's category, never by a property repeating the category's name. It must be in stock when the request wants things on hand, and price times quantity within the budget (equal allowed). When a listing offers options or variants with their own ids, prices or stock, each option is its own candidate and the cart line names the chosen option as the cart tool's schema asks.

CHEAPEST OR BEST
- Under a lowest-price rule the order is the cheapest listing among ALL that meet everything: one fit seen is not enough. Use price ceilings and category filters, lower the ceiling until every remaining row is visible, and check candidates in rising price order; the first that fully fits, every cheaper one checked and rejected, is the answer, the next fits are backups. Under a "best by some measure" rule compare every fit on that measure and take the extreme; a listing stating no value for the measure cannot win.

FINISHING
- Keep a ranked list of verified fits with a backup, since the first pick may sell out at checkout. Before each order the cart holds exactly the lines the request needs (one per requested product, at the asked quantity) and nothing else, and the pick's stock and price were seen after the last announced change (the code re-reads for you when a record went stale). Get the first order in while at least seven steps remain so a refusal can be repaired; from then on only cart, view and finishing tools stay offered. Decide once: every cart swap costs a step.
- Ordering is the normal ending. Declining is rare and a wrong decline scores 0: decline only after searching in several ways and opening the closest candidates, and only when every candidate fails a stated condition, a revealed need, the limit or stock by its own records. Doubt, a missing nice-to-have or an assumed requirement is no reason. Before a decline goes out the code asks for one audit round (re-read the request and the shopper's answers, one more search or view); decline after it only if still nothing fits. A cart still held when time runs out is ordered. When the request forbids substitutes, the decline remark names nothing: no ids, titles or prices.
- Every reply holds at least one tool call, with a short line of working notes before it: the requirements now, the verified fits with price and stock, the next move."""

SETUP_PROMPT = """Before a shopping assistant starts, you read the shopper's request and the store's tool list and write the brief it works from. Answer with ONE compact JSON object (no indentation) and nothing else, with these keys:

"product": the kind of thing wanted, in the request's own words (several kinds when several different things are requested).
"quantity": the number of units of one listing to order (an integer; 1 when the request does not ask for several). Something naturally sold as one item, such as a pair of shoes or a set, is quantity 1.
"items": when the request asks for several DIFFERENT products, one object per product: {"product": a short name for it in the request's words, "quantity": the units of that product (integer)}; an empty list when only one kind of product is requested.
"requirements": firm stated conditions on the item, a list of strings, each "property: value" with the shopper's exact value, limit, exclusion or closed list. Not the price, stock, quantity or wishes, and not the item kind itself: a season, audience or occasion that is part of the category's name belongs to the category, not here. A value stated with should be, has to, got to, must or only is a requirement however casually it is said, also when the shopper says they will check it themselves; so is a plain statement of a property value ("they're for <use>", "it's <value>") when the shopper says beside it that they will check or double-check it (the use or occasion then becomes a requirement such as "occasion: <use>"). When a value is described by pointing at another listing ("the value shown on the page for the <product> from <name> that comes with <details>, or else on the listing that page points to"; "anything other than the value shown on the page for ..."), write ONE requirement "property: <that whole description word for word>"; the other listing's name and details are not requirements of their own.
"price_cap": the highest total price as a number, or null when none is stated.
"currency": the price unit the request uses, or null.
"in_stock_only": true when only things on hand or in stock count.
"priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only when it explicitly asks for the best by some measure; otherwise "none". Remarks that money is tight or that every penny counts are not a request for the cheapest; they only explain the price cap.
"priority_detail": for "best", what "best" is measured by; otherwise "".
"message_budget": how many messages the shopper allows (stated in the request or in a tool description), as an integer, or null.
"step_budget": how many steps or moves the request allows, as an integer, or null.
"rules": a list of every procedural rule the request states for the assistant (how many listings to look at, what to check or tell, how to decline, what not to do, what to include), quoted word for word. Remarks about what the shopper will do themselves are story, not rules.
"soft_wants": wishes phrased as ideally, preferably, would be nice, if possible; never in "requirements".
"hint_properties": property names the story suggests the shopper cares about without stating a value (a list, maybe empty).
"ask_first": a list of things the agent can only learn from the shopper and should ask about before choosing: for example the budget when no price limit is stated, which of two wishes matters more when the request says both may not be possible (then also put both wishes in soft_wants, not in requirements), or anything the request says to check with the shopper. Empty when none.
Every sentence of the request lands somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or story. When unsure whether a sentence is a rule, put it in rules.
"duties": every step the request asks the assistant to take besides finding and buying the item, a list of objects {"do": the step in the request's own words, "when": "before_order", "before_decline", "before_finish" (either way), "before_replace" (before putting a different item in place of one chosen earlier) or "anytime", "say": true when the step is carried out by a message to the shopper (telling them the total or the basket, asking them to confirm, checking with them before a swap, a closing word), false otherwise, "ask": true when the step is really an invitation to put questions to the shopper (to learn what they need or what is worth asking) so that the usual questions fulfil it, false otherwise}. Empty when the request asks for nothing of the kind. What to say when declining or ordering is NOT a duty when the finishing tool itself carries a message or reason (see its arguments in the tool list): list a step here only when it needs a message of its own or an action besides the finish. A step listed here may also appear in "rules".
"bundle": true when the request asks for several products that have to work with or fit each other (parts of one setup, a matching set); false otherwise.
"decline_no_names": true when the request's rule for declining forbids offering, naming or suggesting substitutes or alternatives; false when it asks you to name something; null when the request gives no rule.
"roles": an object naming the tool for each role, using the exact tool names from the list, or null when no tool fits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "message" is the tool that sends a message to the shopper. "cart" is the tool that adds to or removes from the cart. "order" is the tool that buys or checks out. "decline" is the tool that ends the task without buying.
"cart_modes": when the cart tool takes a mode argument, {"add": the exact enum value that puts a line in, "remove": the exact enum value that takes a line out, "view": the value that only lists the cart, or null}; else null.
"""

INTERPRET_PROMPT = """Translate what a shopper just said to a shopping assistant into changes of the shopping conditions. Return ONE JSON object and nothing else, with these keys:
"meaning": one short sentence saying what the shopper means.
"needs": a list of new firm requirements on the product, each written as "property: constraint" with the exact value, limit, exclusion or closed list the shopper gave.
"no_preference": a list of properties the shopper said they do not care about.
"dropped": a list of earlier requirements the shopper no longer needs.
"new_budget": the new maximum total price as a number, or null.
"new_quantity": when the shopper changes how many units they want, {"product": the product it is for (or "" when only one kind is requested), "quantity": the new number as an integer}; otherwise null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), else false.
"requirements_now": the complete list of firm requirements on the product as they stand after everything the shopper has said so far, built from the requirements so far (given below) and what the shopper has explicitly said, each as "property: constraint". Never add a requirement from the story or from your own reading of the request: only values the request states as requirements or the shopper states in a reply count. Keep every earlier requirement unless the shopper dropped or replaced it; for a replaced value keep only the newest one; add every newly revealed need. Do not include price, stock, quantity or the product category.
"duties": new steps the shopper now asks the assistant to take before it finishes (for example telling them the total first, or checking with them before swapping an item), each as {"do": the step in the shopper's words, "when": "before_order", "before_decline", "before_finish", "before_replace" or "anytime", "say": true when it is done by a message to the shopper, "ask": true when asking the shopper questions fulfils it}; otherwise [].
"unclear": true when the shopper says they did not follow or understand the assistant's message, or asks for it to be explained, simplified, reworded or repeated; otherwise false.
"crowded": true when the shopper objects that the message asked about too much at once and wants one subject per message (asks to split it up, to go one at a time, to be given only the first thing); then "unclear" is false; otherwise false.
"reply": when the shopper asks the assistant something or asks for a confirmation, a short, plain reply to send back that answers it and names the change in substance (for example confirming the new value); otherwise "". Always "" when "unclear" is true: a request to explain the assistant's own question is not a question to answer.
When the request asks for several different products, start every requirement in "needs" and "requirements_now" with the short name of the product it applies to, as "<product> <property>: constraint" (the shopper often says "<property> of the <product kind>").
When the shopper describes the wanted value by pointing at another listing ("the value shown on the page for the <product> from <name> that comes with <details>, or else on the listing that page points to", "anything other than the value shown on the page for ..."), copy that whole description word for word as the constraint of the property asked about, as one need; the other listing's name and details are NOT requirements of their own and must not be listed separately.
Insistence idioms ("X isn't up for trade", "X is non-negotiable", "it has to be X") mean X is firmly wanted, never excluded. One line can carry several changes: report every one. A requirement replaced by another is both dropped (the old one) and a new need (the new one). When the shopper says which of two wishes matters more, the preferred wish is a need. When the shopper states a budget, it is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and charitably. Folksy, figurative or roundabout phrasing that names a value means the shopper firmly wants that value; treat a value as excluded only when the shopper clearly says to avoid it. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit. A new budget replaces the old one whether it is higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference, or anything works, put the property asked about in no_preference; such a reply applies only to the property the assistant asked about, never drops a stated requirement, a price limit or a rule, and when the assistant's message did not ask about a property it changes nothing."""

VERIFY_PROMPT = """Judge one listing for a shopping assistant before it commits to it. You see only the listing's own store records, the shopper's request and the conditions in force; judge from the records alone, and treat what they never state as unknown rather than satisfied. The task scores 1 only if the ordered item meets every requirement below; use these reading rules:
- A requirement is met only when the item's own property for it states the value. Property keys vary in spelling (spaces, dots, underscores, prefixes) and the same property may appear under several keys; any key naming that property counts. Values match by meaning: one element of a comma list is enough, singular/plural and close synonyms count, case does not matter. A merely similar-looking word or a broader/different class does not count. When several keys name that property (a plain 'style' beside 'skirt style', 'shape' beside 'clock.shape', in any spelling or namespace), the requirement is met when ANY of them states the value; the others do not cancel it and that is not mixed evidence.
- Two kinds named together ("pants or jeans") are met only by the store's combined category ("Pants & Jeans") when one exists.
- The title or description counts only when the item has no property key for that requirement. If its own property says something else, it fails even when the title says the value.
- A value under a different, related property does not count.
- Numbers: compare after converting units; limits include the edge; a range counts only when the whole range meets the limit.
- Exclusions and closed lists must be respected.
- The item must be the requested kind of product; when listings carry a category label, it must be the category whose label matches the request's wording as a whole, not a narrower or neighbouring one.
- When every key naming the property states a different value, the requirement fails, even if the title suggests otherwise.
- Price times quantity must be within the budget (equal is fine). If the records show it out of stock, it fails.
- A mere wish ("ideally", "would be nice") listed as not required is not a requirement.
- Judge only the requirements listed below; never add one of your own from the story (an age group, a gender, a size, who will use it).
- When the request asks for several different products, judge the item only against the requirements of its own kind of product: a requirement that starts with another product's name does not apply to it, and the quantity is the one for its kind.
When records disagree (for example on price or stock), the newest record counts.
Return ONE JSON object and nothing else: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement that is not met or not shown]}. Use fits=false only for a requirement that the records clearly fail, or that no record addresses at all; do not fail an item for a missing record of stock or price."""

PLAN_PROMPT = """Decide which questions a shopping assistant should still put to the shopper. The shopper holds hidden needs: each is the value of one property of the item, and it surfaces only when a message asks about exactly that property; a message is answered about the first thing it asks, and a property the shopper does not care about earns a shrug. Choose the QUESTION_COUNT properties most likely to be real needs, best first:
0. first, anything listed below as MUST BE LEARNED FROM THE SHOPPER that has not been asked yet; ask it plainly. For the budget, ask how much they want to spend at most. For a choice between two wishes, name both values with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?";
1. then properties that the listings' details below actually carry, preferring (a) properties that most listings carry and whose values differ between them, of any kind (what it is made of, how it looks, how it is built or works, who it is for, technical specifications); a property with the same value in every listing seen is a poor question; and (b) a property the story in the request points to (who will use it, how it must look, where or how it will be used). Avoid logistics (warranty, condition, years) and values that only list everything on offer (all sizes or variants in one string), and never ask about the brand, maker, seller or shop. A size, colour or other option that a listing, or one of its options, has as its own single value is a normal property; so is how many pieces come in one pack.
Never pick a property the request already settles (its stated requirements, the price, stock, quantity), never repeat a property already asked for the same product (a question the shopper did not follow counts as asked: move on, unless that question did not name the product), never spend two questions on near-duplicate properties (two kinds of style, two spellings of one property), and avoid properties no listing carries. Use the property names as the listings name them.
When the request asks for several different products, each question is about ONE of them, and the questions are spread over the products: prefer a product that has had fewer questions so far, and take its properties from the listings of that product's kind (the menu below is grouped by the listings' labels when they carry one).
Write each question as "Which <property> would you like the <product> to have?", with the property's name exactly as the listings name it (spaces instead of underscores or dots) and the product named by its plain noun from the request ("the rug", "the curtains"; never with adjectives, materials, colours or other property values). No example values, no other words, no second question. For something that must be learned first (rule 0), ask it plainly in one short sentence.
"Who it is for" means the listings' property that records the wearer or user (a gender, age group or audience): when the story names the person the item is for and the listings carry such a property with differing values, ask it before properties the story does not touch.
Return ONE JSON object and nothing else: {"questions": [{"product": "...", "property": "...", "question": "..."}, ...]}."""

RANK_PROMPT = """Decide which questions a shopping assistant should still put to the shopper, by putting the code's shortlist of properties in order. The shopper holds hidden needs: each is the value of one property of the product, and it surfaces only when a message asks about exactly that property. The assistant's code has already chosen the shortlist below from the listings it opened (properties most listings carry and whose values differ); your only job is to rank that shortlist: which of these properties would this shopper, given the story in the request, most likely have a firm need about? Put first a property the story points to (who it is for, where or how it is used, what the shopper worries about), then properties a buyer of this product usually decides by (who it is for, such as a gender or age group, when the listings differ on it; how it looks; what it is made of; how it is built or works), then the rest. Logistics (warranty, condition, delivery) come last: a shopper almost never holds a firm need about them; the number of pieces in a pack is not logistics but a property like the others. Choose the QUESTION_COUNT properties most likely to be real needs, best first. Use only properties from the shortlist, spelled as listed; never add one, never repeat one already asked.
Return ONE JSON object and nothing else: {"questions": [{"property": "..."}, ...]}."""

AUDIT_PROMPT = """A shopping assistant is about to decline, saying nothing in the store meets every condition. Check the listing records below against the conditions in force. A condition is met when the listing's own property for it states the value (any key spelling, a member of a comma list, singular/plural, close synonyms; not a broader or different class); the title or description counts only when the listing has no key for that property; numbers compare after unit conversion, edges included; the price must be within the limit and the item not shown out of stock; only the listed conditions count, never one assumed from the story; with several kinds of product, judge each listing against its own kind's conditions. A wrong decline scores 0, so name every listing that may meet everything, and leave one out only when its own records clearly fail a listed condition; a listing whose records contradict nothing and whose sheet was never opened is a possible fit.
Return ONE JSON object and nothing else: {"possible_fits": [ids of listings that appear to meet every requirement, best first, at most 5], "reason": "one sentence"}."""

NOTE_PROMPT = """A shopping assistant works for a shopper in a simulated store. The free-text lines below turned up inside the store's tool results (they are not listing records): the shopper or the store may be announcing a change. Read each line by its meaning, whatever its wording or story, and say what it changes. Return ONE JSON object and nothing else: {"lines": [one object per line, in order: {"new_budget": the new total amount the shopper wants to spend, as the number written in the line (in the request's price unit), when the line proposes, asks for or announces a spending amount, higher or lower; else null, "stockout_of": when the line says a listing can no longer be bought (no piece left, finished, empty, exhausted, unavailable, cannot be ordered), the id of that listing if one is named, else "held"; else "", "price_moved_of": when the line says a listing's price or charge changed (higher, lower, a sale, altered), the id if named, else "held"; else "", "added_need": a list of new firm requirements on the product the line states ("property: value"), "dropped_need": a list of requirements the line drops}]}"""

_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

_NUMBER_WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()
_COUNT = r"(\d{1,3}|" + "|".join(_NUMBER_WORDS) + r")"
_LIMIT_UNITS = r"(messages?|questions?|repl(?:y|ies)|texts?|steps?|moves?|actions?|tool calls?)"
_LIMIT_BEFORE = re.compile(r"\b(?:only|just|at most|no more than|not more than|up to|a maximum of|maximum of|max(?:imum)?|limit(?:ed)? (?:of|to)|cap(?:ped)? at)\s+"
                           + _COUNT + r"\s+(more\s+|further\s+|additional\s+|other\s+)?" + _LIMIT_UNITS + r"\b", re.I)
# the cap in other words, read only when the brief names none: "I'll only answer two messages", "you can send me 2 messages"
_LIMIT_VERB = re.compile(r"\b(?:(?:can|may) (?:only )?(?:send|ask|write)(?: me| us)?|(?:only|just) (?:answer|read|reply to|take|accept|get))\s+"
                         + _COUNT + r"\s+(more\s+|further\s+|additional\s+|other\s+)?" + _LIMIT_UNITS + r"\b", re.I)
_LIMIT_LEFT = re.compile(r"\b" + _COUNT + r"\s+(more\s+)?" + _LIMIT_UNITS + r"\s+(?:left|remaining|remain|to go)\b", re.I)


def limits_in(text: str, loose: bool = False) -> list[tuple[str, int, bool]]:
    """Message and step limits a sentence states, as (kind, count, counted from now); `loose` reads the verb forms only."""
    found: list[tuple[str, int, bool]] = []
    pairs = ([(m, bool(m.group(2))) for m in _LIMIT_VERB.finditer(text)] if loose else
             [(m, bool(m.group(2))) for m in _LIMIT_BEFORE.finditer(text)] + [(m, True) for m in _LIMIT_LEFT.finditer(text)])
    for match, relative in pairs:
        word = match.group(1).lower()
        count = int(word) if word.isdigit() else _NUMBER_WORDS.index(word)
        unit = match.group(3).lower()
        kind = "step" if re.match(r"step|move|action|tool|call", unit) else "message"
        found.append((kind, count, relative))
    return found


# ----------------------------------------------------------------------------- shopper lines read by the code

# a line about what may be spent: a spending word, a sign that the amount is being changed, and the figure itself.
# No unit lists: money is recognised by the request's own price unit (carried in the brief) or a currency sign.
_SPEND_CUE = re.compile(r"\b(?:spend\w*|budget|ceiling|limit)\b", re.I)
_SPEND_SHIFT = re.compile(r"\b(?:now|instead|only|actually)\b", re.I)
_FIGURE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?(?!\w|\.\d)")
_CURRENCY_SIGN = r"[$€£¥₱¢]"


def money_pattern(currency: str) -> str:
    """A pattern for the request's own price unit (a sign, or the word the brief recorded, singular or plural) beside
    a figure: the only way the code tells money from any other number."""
    unit = str(currency or "").strip()
    out = _CURRENCY_SIGN
    if unit and not re.fullmatch(_CURRENCY_SIGN, unit):
        out += r"|(?<![A-Za-z])" + re.escape(unit.rstrip("s").rstrip(".")) + r"s?\.?(?![A-Za-z])"
    return "(?:" + out + ")"


def unit_of(text: str, figure: Any) -> str:
    """The price unit the request writes beside its own limit (the sign before the figure, or the short token after
    it), or ''. The code learns the unit from the request instead of knowing any."""
    if not isinstance(figure, (int, float)) or isinstance(figure, bool):
        return ""
    for form in {f"{figure:g}", f"{figure:,.0f}", f"{figure:.2f}"}:
        for match in re.finditer(r"(?<![\w.])" + re.escape(form) + r"(?!\w|\.\d)", str(text or "")):
            before = text[: match.start()]
            after = text[match.end():]
            sign = re.search(_CURRENCY_SIGN + r"\s*$", before)
            if sign:
                return sign.group(0).strip()
            tail = re.match(r"\s*(" + _CURRENCY_SIGN + r"|[A-Za-z]{1,5}\b\.?)", after)
            if tail and unit_after_figures(tail.group(1), text):
                return tail.group(1)
    return ""


def step_words(text: str) -> set[str]:
    """The words that tell one step from another: four letters or more, and not part of the frame every step is
    written in (the words of our own step-checking instructions: shopper, before, finishing, ...)."""
    return {w for w in re.findall(r"[a-z]{4,}", str(text or "").lower()) if w not in _STEP_FRAME}


def unit_after_figures(token: str, text: str) -> bool:
    """Whether a short word read right after a figure is the request's price unit: a unit only ever follows figures,
    while an ordinary word of the sentence ("50 for", "20 and") also turns up without one."""
    word = token.rstrip(".")
    if not word.isalpha():
        return True
    uses = list(re.finditer(r"(?<![\w])" + re.escape(word) + r"\b", str(text or ""), re.I))
    return all(re.search(r"\d\s*$", text[: m.start()]) for m in uses)


def budget_in(text: str, currency: str = "", around: float | None = None) -> float | None:
    """A new total price limit a shopper line states, or None. The line must talk about the amount (a spending word
    or a sign that it is being changed), its figure must carry the request's own price unit (or no unit at all when
    the request itself used none), it must be of the same order as the current limit, and a sentence about a listing's
    price, or about a count of steps or messages, is not one."""
    best: tuple[int, float] | None = None
    money = money_pattern(currency)
    for sentence in re.split(r"(?<=[.!?;])\s+|\n", str(text or "")):
        cue = _SPEND_CUE.search(sentence) is not None
        shift = _SPEND_SHIFT.search(sentence) is not None
        if not (cue or shift) or limits_in(sentence):
            continue
        if re.search(r"\b(?:price|cost)\w*\b", sentence, re.I) and not cue:
            continue                         # a listing's price moving, not the shopper's limit
        for match in _FIGURE.finditer(sentence):
            before, after = sentence[: match.start()], sentence[match.end():]
            unit = re.search(money + r"\s*$", before, re.I) is not None or re.match(r"\s*" + money, after, re.I) is not None
            if not unit:
                if re.match(r"\s*(?:[A-Za-z]{1,2}\b|%|[x×]\s*\d)", after):
                    continue                 # a measurement (a figure with a short unit of its own), not money
                if currency or not cue or (around is not None and not shift):
                    continue                 # a bare figure counts only when the request itself named no unit, beside a spending word (and a sign of change once a limit is known)
            try:
                figure = float(match.group(1).replace(",", "") + ("." + match.group(2) if match.group(2) else ""))
            except ValueError:
                continue
            if figure <= 0 or (around is not None and around > 0 and not 0.2 * around <= figure <= 5 * around):
                continue
            score = (2 if unit else 0) + (1 if shift else 0) + (1 if cue else 0)
            if best is None or score > best[0]:
                best = (score, figure)
    return best[1] if best else None


# a need described by pointing at another listing's page: "the value shown on the page for the <kind> from <who> that
# comes with <details>, or else on the listing that page points to"; "anything other than the value shown on ..."
_POINTER = re.compile(r"\bon the (?:page|listing) (?:for|of) (?:the |a |an )?(?P<kind>[^,.;]+?) (?:from|by) (?P<who>[A-Z][\w&'’\-]*(?: [A-Z][\w&'’\-]*){0,2})"
                      r"(?: (?:that|which) (?:comes with|has|carries|lists) (?P<attrs>[^.;]+?))?"
                      r"(?=,? or (?:else )?(?:on |to )?(?:the )?(?:listing|page)|[,.;!?]|$| (?:so|please|even|no matter)\b)", re.I)
_POINTER_A = _POINTER
# the same pointer written with the other listing's name in the possessive and its details after ", with":
# "on the page of the <kind> from <Who>'s listing, with <a>, <b> and <c> (or wherever that page leads)"
_POINTER_B = re.compile(r"\bon the (?:page|listing) (?:for|of) (?:the |a |an )?(?P<kind>[^,.;]+?) (?:from|by) (?P<who>[A-Z][\w&\-]*(?: [A-Z][\w&\-]*){0,2})['’]s (?:listing|page)"
                        r"(?:,? with (?P<attrs>.+?))?(?=\s*\(or\b|[;!?]|\.(?!\d)|$)")


# the possessive name first and the kind after it: "on the page for <Who>'s listing of the <kind> - the one with <a> - or ..."
_POINTER_C = re.compile(r"\bon the (?:page|listing) (?:for|of) (?P<who>[A-Z][\w&\-]*(?: [A-Z][\w&\-]*){0,2})['’]s (?:listing|page)"
                        r"(?: (?:of|for) (?:the |a |an )?(?P<kind>[^,.;—–()]+?))?"
                        r"(?:\s*(?:[—–,(]|\s-)\s*(?:the one|one|that one) (?:with|that has|carrying) (?P<attrs>[^—–.;()]+?))?(?=\s*[—–()]|\s-\s|,? or\b|[,.;!?]|$)")


# the value read off a page the reader lands on: "... the same as whatever you land on from the page of the <kind> you'll find
# at <Who> with <a>, <b> and <c>, following each listing it names ..." (the name is case-sensitive, the rest is not)
_POINTER_D = re.compile(r"\bwhatever you (?:land on|end up (?:on|with|at)|arrive at|reach|find|get) (?:from|on|via) the (?:page|listing) (?:of|for) "
                        r"(?:the |a |an )?(?P<kind>[^,.;]+?) (?:you(?:'|’)?ll|you will|you'd|you would|you can) (?:find|see|spot|locate) (?:at|from|by|under) "
                        r"(?P<who>(?-i:[A-Z][\w&'’\-]*(?: [A-Z][\w&'’\-]*){0,2}))(?: (?:with|that has|carrying) (?P<attrs>[^;]+?))?"
                        r"(?=,? (?:and )?follow\w*\b|[;!?]|\.(?!\d)|$)", re.I)


# the seller named after the kind as the one who stocks it, the details after ", with", the page possibly sending on:
# "... the one the page of the <kind> that <Who> carries, with <a> and <b> points to or states ..."
_POINTER_E = re.compile(r"\b(?:the )?(?:page|listing) (?:of|for) (?:the |a |an )?(?P<kind>[^,.;]+?) (?:that|which) "
                        r"(?P<who>(?-i:[A-Z][\w&'’\-]*(?: [A-Z][\w&'’\-]*){0,2})) (?:carries|stocks|sells|offers|lists|has on (?:sale|offer))"
                        r"(?:,? with (?P<attrs>.+?))?(?=,? (?:either )?(?:points|refers|leads|links|sends) (?:you )?(?:to|on)\b|,? (?:states|shows|gives)\b|[;!?]|\.(?!\d)|$)", re.I)


class _EitherPointer:
    """Any pointer form: the landing form and the possessive ones first (their name stops before "'s listing"), the
    stocking form before the plain one (both name the seller after the kind)."""

    @staticmethod
    def search(text: str) -> Any:
        return (_POINTER_D.search(text) or _POINTER_B.search(text) or _POINTER_C.search(text) or _POINTER_E.search(text)
                or _POINTER_A.search(text))


_POINTER = _EitherPointer()
_CHAIN = re.compile(r"\b(?:listing|page)\b[^.;]{0,30}\b(?:points|refers|links|leads|send\w*|directs?|takes you)\b", re.I)
_EXCLUDE = re.compile(r"\b(?:other than|except|anything but|avoid\w*|not|never|nothing like|apart from|different from|distinct from|unlike)\b|n't\b", re.I)
# a bound written before the pointer: "not under the one ..." is at least that value, "not over the one ..." at most
_FLOOR_BEFORE = re.compile(r"\b(?:not (?:under|below|less than|lower than|smaller than)|no (?:less|lower|smaller) than|at least)\s+(?:the one\s+)?\Z", re.I)
_ROOF_BEFORE = re.compile(r"\b(?:not (?:over|above|more than|higher than|bigger than|larger than)|no (?:more|higher|bigger|larger) than|at most)\s+(?:the one\s+)?\Z", re.I)
_PAIRED = re.compile(r"\b(?:alongside|pair\w*|partner\w*|goes with|compatible|match\w*)\b", re.I)


def pointer_in(prop: str, text: str) -> dict[str, Any] | None:
    """The pointer a need's wording holds, as a plan for the code: the property to settle, how the found value is to be
    used (same, paired, exclude), the other listing's kind, name and details, and whether its page may point further."""
    match = _POINTER.search(str(text or ""))
    if not match or not str(prop or "").strip():
        return None
    before = re.split(r"[;!?—–]|\.(?!\d)", text[: match.start()])[-1][-80:]
    after = text[match.end(): match.end() + 80]
    lead = re.sub(r"\s*\bthe one\s*\Z", " ", before, flags=re.I)
    if _FLOOR_BEFORE.search(lead + " ") or _ROOF_BEFORE.search(lead + " "):
        relation = "min" if _FLOOR_BEFORE.search(lead + " ") else "max"
    else:
        relation = "exclude" if _EXCLUDE.search(before) else ("paired" if _PAIRED.search(before) or re.search(r"\bgo(?:es)? (?:well )?with\s*$", before, re.I) else "same")
    bound: dict[str, Any] = {}
    near = re.search(r"\b(?:inside|within)\s+(\d+(?:\.\d+)?)\s*([^\d\s,.;]*)\s+of\s*$", before, re.I)
    if relation == "same" and near:
        relation, bound = "near", {"by": float(near.group(1)), "unit": near.group(2)}
    elif relation == "same" and re.search(r"^[^.;]{0,60}?\bor (?:higher|more|above|greater|bigger|larger)\b", after, re.I):
        relation = "min"
    elif relation == "same" and re.search(r"^[^.;]{0,60}?\bor (?:lower|less|below|smaller)\b", after, re.I):
        relation = "max"
    attrs = [a.strip(" ,") for a in re.split(r",\s*|\s+and\s+|\s+plus\s+|\s+with\s+", match.group("attrs") or "") if a.strip(" ,")]
    attrs = [re.sub(r"\s*\((?:or|and)\b.*$", "", a).strip(" ,") for a in attrs]
    attrs = [a for a in attrs if len(words(a)) >= 1]
    kind = re.sub(r"^(?:the|a|an)\s+", "", (match.group("kind") or "").strip(), flags=re.I)
    # "... the page ... points to or states": the value is the one the page sends on to, else the one it states itself
    follow = re.search(r"^\W{0,3}(?:either )?(?:points|refers|leads|links|sends) (?:you )?(?:to|on)\b[^.;]{0,24}\b(?:states|shows|gives|says|lists)\b", after, re.I) is not None
    chain = (_CHAIN.search(text[match.end() - 3:]) is not None or re.search(r"^\W{0,3}(?:and )?follow\w*\b[^.;]{0,30}\b(?:listing|page|link)", after, re.I) is not None
             or re.search(r"^\W{0,3}(?:either )?(?:points|refers|leads|links|sends) (?:you )?(?:to|on)\b", after, re.I) is not None)
    return {"prop": flat(prop), "relation": relation, "bound": bound, "kind": kind, "who": match.group("who").strip(), "attrs": attrs,
            "chain": chain, "follow": follow, "legs": [match.group("who").strip()], "searched": 0,
            "opened": [], "page": "", "value": "", "done": False, "failed": False, "steps": 0, "source": str(text)}


# ----------------------------------------------------------------------------- fitment codes (model / part codes)

# a code is a run of capitals followed by digits, optionally closed by a run of x standing for any digit ("SIR21xx")
# a fitment code: two to five capitals, then digits (a hyphen or a space may stand between, "BEK-1210", "BEX 35xx"), with an
# optional x wildcard; a one-digit series ("TA4") counts where a list, a tag or a sentence end follows it
_CODE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,5})(?:[- ](?=\d{2}))?(\d{2,6}|\d(?=[-,.;\]]|\s|$))([xX]{0,4})(?![A-Za-z0-9])")
_CODE_EXCEPT = re.compile(r"\b(?:except|excluding|other than)\b", re.I)


def codes_in(text: str) -> list[str]:
    """Every fitment-shaped code in a text, in order, without repeats."""
    out: list[str] = []
    for m in _CODE.finditer(str(text or "")):
        code = m.group(1) + m.group(2) + m.group(3).lower()
        if code not in out:
            out.append(code)
    return out


def code_prefix(code: str) -> str:
    m = re.match(r"[A-Z]+", code or "")
    return m.group(0) if m else ""


def code_covers(listed: str, wanted: str) -> bool:
    """Whether a code written on a page covers the wanted code: equal, or equal digit for digit where either writes x."""
    if code_prefix(listed) != code_prefix(wanted):
        return False
    a, b = listed[len(code_prefix(listed)):], wanted[len(code_prefix(wanted)):]
    return len(a) == len(b) and all(x == y or x == "x" or y == "x" for x, y in zip(a, b))


def code_claims(text: str) -> list[tuple[list[str], list[str], list[str]]]:
    """What a page's free text says about codes, one claim per sentence: (subjects, listed, excluded). A sentence of the
    form "<subject code> ...: <codes>" speaks for the listing that carries the subject code; one with no code before its
    colon speaks for the page's own listing (subjects empty). Codes after an exception word are excluded."""
    out = []
    for sentence in re.split(r"(?<=[.;!?])\s+|\n", str(text or "")):
        if not _CODE.search(sentence):
            continue
        head, colon, tail = sentence.partition(":")
        subjects = codes_in(head) if colon and codes_in(tail) else []
        body = tail if subjects else sentence
        cut = _CODE_EXCEPT.search(body)
        listed = codes_in(body[: cut.start()] if cut else body)
        excluded = codes_in(body[cut.end():]) if cut else []
        out.append((subjects, [c for c in listed if c not in subjects], excluded))
    return out


_RULE_HEAD = re.compile(r"^.*?\b(?:rules?|pairings?)\b[^:]{0,60}?(?::|\bis that\b|\bare\b|\bis\b)\s*:?\s*", re.I)
_PAIR_LINK = re.compile(r"^(.+?)\s+(?:goes\s+(?:together\s+)?with|go\s+(?:together\s+)?with|pairs?\s+with|is\s+paired\s+with|matches|with)\s+(.+)$", re.I)
_SELF_PAIR = re.compile(r"^(?:its|that|this|the)\b[^.;,]{0,40}?\b(?:goes|go|pairs?|is paired)\s+(?:together\s+)?with\s+([^.;,]{1,40}?)\s*$", re.I)


def listed_pairs(text: str) -> dict[str, str]:
    """The value pairs of a rule written as a list after its lead-in ("Our pack size rule is that 10 goes with 25, 20 with
    12, and 6 with 4"; "The store's part type pairings are: a goes with b, ..."), and a page's own value paired in one line
    ("Its socket goes together with am5", kept under '*')."""
    out: dict[str, str] = {}
    for sentence in re.split(r"(?<=[.!?;])\s+", str(text or "")):
        body = sentence.strip().rstrip(".!?; ")
        single = _SELF_PAIR.match(body)
        if single:
            out.setdefault("*", single.group(1).strip(" .,:;"))
            continue
        head = _RULE_HEAD.match(body)
        if not head:
            continue
        for piece in re.split(r",\s*(?:and\s+)?|\s+and\s+(?=[^,]*\bwith\b)", body[head.end():]):
            found = _PAIR_LINK.match(piece.strip(" .,:;"))
            if not found:
                continue
            left, right = found.group(1).strip(" .,:;"), found.group(2).strip(" .,:;")
            if left and right and len(words(left)) <= 5 and len(words(right)) <= 5:
                out.setdefault(flat(left), right)
    return out


def pairing_rule(text: str) -> dict[str, str]:
    """A rule on a listing's page that pairs values ("pairs A with B, C with D, and E with F"): each value's partner."""
    out: dict[str, str] = dict(listed_pairs(text))
    if out:
        return out
    for sentence in re.split(r"(?<=[.!?;])\s+", str(text or "")):
        head = re.search(r"\b(?:pair\w*|partner\w*|match\w*|goes)\b", sentence, re.I)
        if not head or " with " not in sentence[head.end():]:
            continue
        body = sentence[head.end():]
        for piece in re.split(r",\s*(?:and\s+)?|\s+and\s+(?=[^,]*\bwith\b)", body):
            left, sep, right = piece.partition(" with ")
            left = re.sub(r"^(?:up|together|along|also)\s+", "", left.strip(" .,:;"), flags=re.I)
            right = right.strip(" .,:;")
            if sep and left and right and len(words(left)) <= 4 and len(words(right)) <= 4:
                out[flat(left)] = right
    return out


# a remark that sends one property's value elsewhere: "For <prop>, check / refer to / look at / use <the other listing>",
# "<Prop> is to be (should be) taken from <the other listing>", "Take <prop> from <the other listing>"
_SEND_ON = (re.compile(r"\bfor (?:the |its |this item's )?[^,.;:]{1,40}?,\s*(?:check|refer to|look at|look to|use|see|consult)\s+"
                       r"(?:(?:a different|another|the other) product(?: instead of this one)?\s*[:,]\s*)?(?P<desc>(?:[^.;]|\.(?=\d))+)", re.I),
            re.compile(r"\btaken from\s+(?:(?:a different|another) product\s*[:,]?\s*)?(?P<desc>(?:[^.;]|\.(?=\d))+)", re.I),
            re.compile(r"\btake [^.;]{1,40}? from (?P<desc>(?:[^.;]|\.(?=\d))+)", re.I))
_SEND_STOP = re.compile(r",?\s+(?:not (?:from|to|at)|rather than|instead|since|as (?:this|it|that)|because|which has)\b", re.I)


def sent_on_to(text: str, known: str) -> str:
    """The seller a remark of the sending form names (one capitalised word: after 'from', a possessive, or the first word
    after the article), or ''. A decoy line ("a version from X exists elsewhere") is not of this form."""
    for pattern in _SEND_ON:
        for match in pattern.finditer(str(text or "")):
            desc = _SEND_STOP.split(match.group("desc"))[0]
            name = ""
            found = re.search(r"\bfrom\s+((?-i:[A-Z][\w&\-]*))", desc)
            if found:
                name = found.group(1)
            else:
                found = re.search(r"((?-i:[A-Z][\w&\-]*))['’]s\b", desc)
                if found:
                    name = found.group(1)
                else:
                    found = re.match(r"\s*(?:the |a |an )?((?-i:[A-Z][\w&\-]*))", desc)
                    name = found.group(1) if found else ""
            if len(name) >= 2 and flat(name) != flat(known) and not _FIRST_PERSON.fullmatch(name) and name.lower() not in ("this", "the", "another"):
                return name
    return ""


def referral_in(text: str, known: str, kind: str = "") -> str:
    """Another seller or brand a page's remark sends the reader to ("a version with X exists from Qibelsari", "use the
    Qibelsari Sports Watches product"), or ''. Words of the product kind that follow the name are not part of it."""
    sent = sent_on_to(text, known)
    if sent:
        return sent
    # a line that only says another version is sold elsewhere sends nobody anywhere
    text = " ".join(x for x in re.split(r"(?<=[.!?;])\s+", str(text or "")) if not re.search(r"\belsewhere\b", x, re.I))
    name_part = r"([A-Z][\w&'’\-]*(?: [A-Z][\w&'’\-]*)?)"
    patterns = (r"\b(?:exists?|available|offered|sold|taken)\b[^.;]{0,80}?\b(?:from|at|by|under)\s+" + name_part,
                r"\b(?:use|see|try|check)\s+(?:the\s+)?" + name_part + r"\b",
                r"\b(?:from|see|use|on)\s+(?:a |an |the )?(?:different|another|other|separate)\s+(?:product|listing|item|page)\W+(?:the\s+)?" + name_part)
    named = r"((?-i:[A-Z][\w&\-]*(?: [A-Z][\w&\-]*)?))(?:['’]s)?"
    # "use the <Kind> product from <Who>", "the <Kind> model by <Who>": the name after the kind word decides
    after_kind = re.compile(r"\b(?:product|model|listing|item|piece|page|version)\s+(?:from|by|of)\s+" + named, re.I)
    # "take the figure from the <Who> monitor", "look to <Who>'s soundbar", "refer to another product: <Who>'s jumpsuit"
    sent_to = re.compile(r"\b(?:tak(?:e|en|ing)|read|look\w*|refer\w*|draw\w*|source\w*)\b[^.;:]{0,40}?\b(?:from|to|at)\s+"
                         r"(?:(?:another|a different|a separate|other) (?:product|listing|item|page)\s*[:,]?\s*)?(?:the |a |an )?" + named, re.I)
    for match in after_kind.finditer(str(text or "")):
        name = match.group(1).strip()
        if len(name) >= 2 and flat(name) != flat(known) and not _FIRST_PERSON.fullmatch(name):
            return name
    for match in re.finditer("|".join(patterns), str(text or "")):
        parts = (match.group(3) or match.group(1) or match.group(2) or "").strip().split()
        while len(parts) > 1 and (parts[-1].lower() in words(kind) or parts[-1].lower().rstrip("s") in ("product", "listing", "item", "page")):
            parts.pop()
        name = " ".join(parts)
        # a name is capitalised wherever it appears; a word the remark also writes in lower case is an ordinary word
        ordinary = re.search(r"\b" + re.escape(name.lower()) + r"\b", str(text or "")) is not None
        name = re.sub(r"['’]s$", "", name)
        if len(name) >= 2 and flat(name) != flat(known) and not ordinary and not _FIRST_PERSON.fullmatch(name):
            return name
    for match in sent_to.finditer(str(text or "")):
        parts = match.group(1).strip().split()
        while len(parts) > 1 and (parts[-1].lower() in words(kind) or parts[-1].lower().rstrip("s") in ("product", "listing", "item", "page")):
            parts.pop()
        name = " ".join(parts)
        if len(name) >= 2 and flat(name) != flat(known) and not _FIRST_PERSON.fullmatch(name) and name.lower() not in ("this", "the", "another"):
            return name
    return ""


def referral_details(text: str, name: str) -> list[str]:
    """The details a page's remark gives for the listing it sends the reader to, in the brackets after its name ("use the
    Qibelsari product (blue, 100 m water resistance)"): one phrase per detail, or none."""
    if not name:
        return []
    for sentence in re.split(r"(?<=[.!?;])\s+", str(text or "")):
        if flat(name) not in flat(sentence):
            continue
        match = re.search(re.escape(name) + r"\b[^()]{0,60}?\(([^()]{1,120})\)", sentence)
        if match:
            return [a.strip(" ,") for a in re.split(r",\s*|\s+and\s+", match.group(1)) if a.strip(" ,")]
        # "... the Lukorazen sneakers in mesh material, plain pattern, high-top sneaker height, not from this item."
        match = re.search(re.escape(name) + r"\b[^.;,]{0,40}?\b(?:in|with)\s+((?:[^.;]|\.(?=\d)){3,160}?)(?:,?\s+(?:not|rather than|instead of|since|as this|as it|because)\b|[.;](?!\d)|$)", sentence)
        if match:
            return [re.sub(r"^(?:a|an|the)\s+", "", a.strip(" ,"), flags=re.I) for a in re.split(r",\s*|\s+and\s+", match.group(1)) if a.strip(" ,")]
    return []


def loose_strings(value: Any, out: list[str], depth: int = 0, keys: tuple[str, ...] = (), listing: Any = None) -> None:
    """Short texts anywhere in a result, except inside listing records (a notice can sit at any depth)."""
    if depth > 6 or len(out) > 60:
        return
    if isinstance(value, str):
        if 3 <= len(value) <= 500:
            out.append(value)
    elif isinstance(value, list):
        for item in value[:40]:
            loose_strings(item, out, depth + 1, keys, listing)
    elif isinstance(value, dict):
        if listing(value) if listing is not None else (field(value, keys + ID_KEYS) is not None and field(value, PRICE_KEYS) is not None):
            return
        for item in value.values():
            loose_strings(item, out, depth + 1, keys, listing)


DUTY_PROMPT = """A shopping assistant is about to WHAT. The shopper's request (or the shopper later) asked it to do the steps below by messaging the shopper first. Write ONE short, friendly message that carries out every one of them, using only the facts given (the basket lines and the total). It must not ask about anything else, and when the assistant is declining it must not suggest or name any listing unless a step asks for that. Return ONE JSON object and nothing else: {"message": "..."}"""

CHECK_PROMPT = """Last look before a shopping assistant finishes: it is about to WHAT. Read everything below once and say what is still missing. Return ONE JSON object and nothing else: {"unmet_conditions": ["..."], "undone_steps": ["..."], "ok": true or false}.
- unmet_conditions: conditions in force (stated, revealed by the shopper or changed) that the cart line's own records do NOT show met. A condition is met when the listing's own property states the value (any key spelling, a member of a comma list, singular or plural, a close synonym); what the records never mention counts as unmet only when the request plainly requires it; price times quantity must be within the budget and the item must not be shown out of stock. Name each as "property: value" with a few words of why. Leave the list empty when the assistant is declining.
- undone_steps: things the request (or the shopper later) told the assistant to do before finishing that the exchanges so far do not show done: for example telling the shopper the basket and total, asking for a go-ahead and waiting for it, a closing remark. Leave out steps already marked done below, steps that only ask the assistant to put questions, and anything the request does not actually ask for. Quote each step in the request's own words.
- ok: true when both lists are empty."""

_STEP_FRAME = frozenset(re.findall(r"[a-z]{4,}", CHECK_PROMPT.split("- undone_steps:")[1].split("- ok:")[0].lower()))

REVIEW_PROMPT = """Something changed while a shopping assistant was shopping. Decide whether its cart still holds the right purchase under the conditions as they stand now. An item is right only if its own records meet every current requirement, the price times quantity is within the current budget, and it is not sold out. Return ONE JSON object and nothing else: {"still_right": true or false, "why": "one short sentence", "next": "one short instruction for the assistant (what to fix, or that it can order)"}"""

BUNDLE_PROMPT = """A shopping assistant is about to order several products that the shopper needs to work together. From the listings' own records, decide whether the cart's products fit and work with each other (sizes, connectors, standards, models or styles the records state). Treat what the records never state as unknown, not as a conflict. Return ONE JSON object and nothing else: {"work_together": true or false, "problem": "the clash the records show, or empty"}"""


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


_PROPERTY_KEYS = re.compile(r"fact|propert|trait|spec|attribut|feature|detail|characteristic", re.I)
def _line_of(args: dict[str, Any], ident: str) -> str:
    """The cart line (as text) that names this listing, when the cart call passes its lines as a list."""
    for value in args.values():
        if isinstance(value, list):
            for line in value:
                if isinstance(line, dict) and ident in line.values():
                    return _compact(line)
    return ""


def _has_text(value: Any, depth: int = 0) -> bool:
    """Whether a result holds any word (not only ids, numbers and flags)."""
    if depth > 4:
        return False
    if isinstance(value, str):
        return any(c.isalpha() for c in value) and not (_ID_HINT.match(value) and any(c.isdigit() for c in value))
    if isinstance(value, dict):
        return any(_has_text(v, depth + 1) for v in value.values())
    if isinstance(value, list):
        return any(_has_text(v, depth + 1) for v in value)
    return False


def _pending_step(observation: dict[str, Any]) -> bool:
    """An order result that waits for another step (a token to confirm with) and carries no shopper sentence."""
    lines: list[str] = []
    _sentences([v for v in observation.values() if isinstance(v, (list, str))], lines)
    keys = list(observation)
    for value in observation.values():
        if isinstance(value, dict):
            keys += list(value)
    return any(_PENDING_KEYS.search(str(k)) for k in keys) and not any(len(line.split()) >= 8 for line in lines)


_WRAPPERS = re.compile(r"^(?:data|result|payload|observation|response|body)$", re.I)


def _said_lines(node: Any) -> list[str] | None:
    """The sentences a list carries when it is a list of notes: plain strings, or event records of one sentence each."""
    if not isinstance(node, list) or not node:
        return None
    texts = []
    for entry in node:
        if isinstance(entry, str):
            texts.append(entry)
            continue
        if not isinstance(entry, dict) or len(entry) > 3 or any(isinstance(x, (dict, list)) for x in entry.values()):
            return None
        long = [x for x in entry.values() if isinstance(x, str) and x.count(" ") >= 3]
        if len(long) != 1 or any(isinstance(x, (int, float)) and not isinstance(x, bool) for x in entry.values()):
            return None
        texts.append(long[0])
    texts = [t for t in texts if t.count(" ") >= 3]
    return texts or None


def unpack_reply(observation: Any) -> Any:
    """A store reply in the form the bookkeeping reads: a lone wrapper key around the payload is dropped, and notes the
    store tucked inside the payload or wrote as event records are moved to one top-level list beside it (or stand alone
    when nothing else came back). Replies that already have that form pass untouched."""
    if isinstance(observation, dict) and len(observation) == 2:
        if sum(isinstance(v, list) and bool(v) and all(isinstance(x, str) for x in v) for v in observation.values()) == 1:
            return observation
    found: list[tuple[str, list[str]]] = []
    queue: list[tuple[Any, dict | None, Any]] = [(observation, None, None)]
    while queue:
        node, parent, key = queue.pop()
        if parent is not None and len(found) < 6:
            said = _said_lines(node)
            if said:
                found.append((str(key), said))
                del parent[key]
                continue
        if isinstance(node, dict) and (parent is None or len(queue) < 40):
            queue.extend((v, node, k) for k, v in list(node.items()) if isinstance(v, (dict, list)))
    for _ in range(3):
        if isinstance(observation, dict) and len(observation) == 1 and _WRAPPERS.match(str(next(iter(observation)))):
            inner = next(iter(observation.values()))
            if isinstance(inner, (dict, str)) or inner is None:
                observation = inner
                continue
        break
    if not found:
        return observation
    lines = [t for _, said in found for t in said]
    label = "refusal" if any(k.lower() == "refusal" for k, _ in found) else "notes"
    if observation in (None, "", {}):
        return lines
    return {"payload": observation, label: lines}


# --- records the store writes as text ---------------------------------------------------------------------------------
# A world may write each listing as ONE string ("ref: p-1 | title: ... | price: 120 | facts: colour=red; size=M"), as JSON
# inside a string, as "Key: value" lines, or wrapped ({"value": "<record>"}). Every observation is read into plain objects
# before any bookkeeping touches it. A text becomes a record only when its own shape says so: at least three labelled
# fields, cut only where a label follows the separator (a title may hold the separator itself). Anything else stays the
# text it was, so the model still reads it as written.

_LABEL_AT = re.compile(r"([A-Za-z_][\w .\-/()]{0,40}?)\s*:(?!//)[ \t]?(.*)\Z", re.S)
_EQUALS_AT = re.compile(r"([A-Za-z_][\w .\-/()]{0,40}?)\s*=(?!=)[ \t]?(.*)\Z", re.S)
_PAIR_AT = re.compile(r"([^=;|\n:]{1,120}?)\s*=(?!=)\s*(.*)\Z", re.S)
_ROW_CUTS = (r"[ \t]+[|¦•·][ \t]+", r"[ \t]*\|[ \t]*", r"[ \t]+;[ \t]+")
_GROUP_CUTS = ((r"\s*;\s*", _PAIR_AT, 1), (r"\s*,\s*", _PAIR_AT, 2), (r"\s*;\s*", _LABEL_AT, 2), (r"\s*,\s*", _LABEL_AT, 2))
_NUMBER_TEXT = re.compile(r"-?(?:0|[1-9]\d{0,15})(?:\.\d{1,9})?")


def labelled_pieces(text: str, cut: str, head: re.Pattern, widest: int) -> list[list[str]]:
    """The labelled fields of a text cut at `cut`. A piece that does not open with a fresh label is glued back to the field
    before it, with the separator it was cut at. [] when the text does not open with a label."""
    parts = re.split("(" + cut + ")", text.strip())
    fields: list[list[str]] = []
    labels: set[str] = set()
    for at in range(0, len(parts), 2):
        hit = head.match(parts[at])
        label = hit.group(1).strip() if hit else ""
        if label and len(label) <= widest and label.lower() not in labels and ". " not in label and (head is _PAIR_AT or len(label.split()) <= 4):
            labels.add(label.lower())
            fields.append([label, hit.group(2)])
        elif fields:
            fields[-1][1] += parts[at - 1] + parts[at]
        else:
            return []
    return fields


def text_value(text: str, depth: int = 0) -> Any:
    """One field's text as a value: a bare number, a yes/no flag, an empty marker, JSON, a group of k=v (or k: v) pairs
    (a property sheet), else the text itself."""
    value = text.strip()
    low = value.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    if _NUMBER_TEXT.fullmatch(value):
        return float(value) if "." in value else int(value)
    if depth < 4 and value[:1] in "[{" and value[-1:] in "]}":
        try:
            return read_records(json.loads(value), 0, depth + 1)
        except ValueError:
            pass
    if depth < 4 and ("=" in value or ":" in value):
        for cut, head, least in _GROUP_CUTS:
            pairs = labelled_pieces(value, cut, head, 120 if head is _PAIR_AT else 40)
            if len(pairs) < least or (len(pairs) == 1 and (len(pairs[0][0].split()) > 4 or not pairs[0][1].strip())):
                continue
            if any(len(inner.split()) > 30 for _, inner in pairs):
                continue                    # prose that happens to hold a colon or an equals sign, not a group of values
            return {label: text_value(inner, 9) for label, inner in pairs}
    return text


def text_lines_record(body: str, depth: int) -> dict[str, Any] | None:
    """A record written as 'Key: value' lines; an indented or bulleted run of 'k: v' lines under an empty key is its group."""
    record: dict[str, Any] = {}
    last = ""
    group: dict[str, Any] | None = None
    for line in body.split("\n"):
        if not line.strip():
            continue
        content = line.strip().lstrip("-*•").strip()
        inset = line[: len(line) - len(line.lstrip())] != "" or line.strip()[:1] in "-*•"
        hit = _LABEL_AT.match(content)
        label = hit.group(1).strip() if hit else ""
        if hit and inset and group is not None and len(label) <= 120:
            group[label] = text_value(hit.group(2), 9)
        elif hit and not inset and label and label not in record and len(label.split()) <= 4:
            rest = hit.group(2).strip()
            group = {} if not rest else None
            record[label] = group if group is not None else rest
            last = label
        elif last and isinstance(record.get(last), str):
            record[last] += " " + content
        elif not record:
            return None
    if sum(1 for v in record.values() if v not in ("", {})) < 3:
        return None
    return {k: (text_value(v, depth + 1) if isinstance(v, str) else v) for k, v in record.items()}


def text_record(text: str, depth: int = 0, field: bool = False) -> Any:
    """A text read as records when its shape says so: JSON, several records (one per line or per block), one record of
    labelled fields cut at a pipe-like separator, or 'Key: value' lines. Otherwise the text, unchanged. A field of an
    object the store already wrote as JSON is read only as JSON or a pipe-cut record (its lines are its own prose)."""
    body = text.strip()
    if depth > 4 or len(body) < 8:
        return text
    if body[:1] in "[{" and body[-1:] in "]}":
        try:
            return read_records(json.loads(body), 0, depth + 1)
        except ValueError:
            pass
    if "\n" in body and not field:
        blocks = [b for b in re.split(r"\n[ \t]*\n|\n[ \t]*-{3,}[ \t]*\n", body) if b.strip()]
        if len(blocks) < 2:
            blocks = [b for b in body.split("\n") if b.strip()]
        if len(blocks) >= 2:
            read = [text_record(b, depth + 1) for b in blocks]
            found = sum(1 for r in read if isinstance(r, dict))
            if found >= 2 and 2 * found >= len(read):
                return read
    for cut in _ROW_CUTS:
        if not re.search(cut, body):
            continue
        for head in (_LABEL_AT, _EQUALS_AT):
            fields = labelled_pieces(body, cut, head, 40)
            if len(fields) >= 3 and all(len(label.split()) <= 4 for label, _ in fields):
                return {label: text_value(value, depth + 1) for label, value in fields}
    if "\n" in body and not field:
        lined = text_lines_record(body, depth)
        if lined is not None:
            return lined
    return text


def read_reply(value: Any) -> Any:
    """read_records that never fails an episode: a result it cannot read stays exactly as the store wrote it."""
    try:
        return read_records(value)
    except (ValueError, TypeError, RecursionError, AttributeError, IndexError):
        return value


def read_records(value: Any, depth: int = 0, level: int = 0) -> Any:
    """An observation with every record the store wrote as text read into objects (see text_record). A lone key around a
    record read from text is dropped, so a wrapped page reads like a bare one."""
    if depth > 8:
        return value
    if isinstance(value, str):
        return text_record(value, level)
    if isinstance(value, list):
        return [read_records(v, depth + 1, level) for v in value]
    if not isinstance(value, dict):
        return value
    out = {k: (text_record(v, level, len(value) > 1) if isinstance(v, str) else read_records(v, depth + 1, level)) for k, v in value.items()}
    if len(out) == 1:
        (only, inner), = out.items()
        if isinstance(inner, dict) and isinstance(value[only], str) and len(inner) >= 3 and not any(c.isdigit() for c in str(only)):
            return inner
    return out


def _listed_ids(args: dict[str, Any]) -> list[str] | None:
    """Listing ids in a cart call that passes its lines as a list (None when the call has no such list)."""
    def id_like(key: Any, value: Any) -> bool:
        return isinstance(value, str) and bool(_ID_HINT.match(value)) and any(c.isdigit() for c in value) and bool(_IDENT_KEYS.search(str(key)))

    if any(id_like(k, v) for k, v in args.items()):
        return None             # the call names its listing directly; a list beside it holds choices, not lines
    plain = [x for v in args.values() if isinstance(v, list) for x in v if isinstance(x, str) and _ID_HINT.match(x)]
    if plain:
        return list(dict.fromkeys(plain))     # a bare list of listing ids
    lists = [v for v in args.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
    # one id per line (the first id-like value, normally the listing; an option id that follows it is shown with the listing)
    out = []
    for lines in lists:
        for line in lines:
            ident = next((v for k, v in line.items() if id_like(k, v)), None)
            if ident is None:
                # a line whose listing field has a name none of the usual words cover (race 18: "ChosenItem", which left
                # champ23 reading every hold as an empty cart): the line's one id-shaped text value is its listing
                shaped = [v for v in line.values() if isinstance(v, str) and _ID_HINT.match(v) and any(c.isdigit() for c in v)]
                ident = shaped[0] if len(shaped) == 1 else None
            if ident:
                out.append(ident)
    return out or None


# a refusal that says the item can no longer be bought (as against one that reports a changed price)
_DOUBT = re.compile(r"\b(?:conflict\w*|mixed|ambiguous|inconsistent|contradict\w*|not a\b|isn't a\b|is not a\b|a set\b|sets?\b|bundle\w*|combo|kit)\b", re.I)
# "I'm no longer fussed about X" is the shopper dropping a wish, not a listing gone: a first-person "no longer" is no stockout
_SELF_NO_LONGER = re.compile(r"\b(?:i|i'm|im|i am|we|we're|we are)\s+(?:\w+\s+)?no longer\Z", re.I)
_GONE = re.compile(r"sold out|out of stock|unavailable|no longer|zero left|none left|all gone|is gone|it's gone|now gone|"
                   r"gone from stock|dried up|ran dry|run dry|cleared out|cleared from|bought the last|can't be ordered|cannot be ordered|"
                   r"grayed out|greyed out|hit zero|no units|none remaining|no .{0,12}remaining|been sold|been taken|snapped up|"
                   r"in stock any ?more|not in stock|isn't in stock|no stock|stock.{0,20}\bzero\b|last one", re.I)
# a line that is about the price (a changed price alone does not make an item unacceptable)
_PRICE_WORDS = re.compile(r"price|cost|charg|priced|bumped|gone up|went up|come down|dropped|cents|dearer|cheaper", re.I)


# listing metadata rather than properties of the product: never offered to the question planner. Words that are
# always metadata match anywhere in the name; words that also end real property names ("energy rating", "scent notes",
# "mounting options") only when they are the whole name
_LISTING_META = re.compile(r"\b(brand|maker|manufacturer|seller|vendor|merchant|headline|heading|sku|url|title|made by|sold by|"
                           r"blurb|tagline|caption|teaser|slogan)\b|"
                           r"^(make|shop|shop name|store|store name|outlet|company|category|department|shelf|description|"
                           r"(overview|description|summary)( text)?|notes?|image|images|rating|reviews?|review count|link|"
                           r"options?|variants?|id|name|price|amount|cost|stock|on hand|in stock)$", re.I)
# an error that says a tool's allowance is spent (not a bad argument or an over-budget cart)
_USED_UP = re.compile(r"budget_exceeded|quota|allowance|used up|no more\b.{0,40}\b(may|can|allowed|left)|limit reached|"
                      r"reached the limit|maximum (number|reached)|already used", re.I)
_IDENT_KEYS = re.compile(r"id|ref|sku|code|number|key|listing|product", re.I)
_OPTION_KEYS = re.compile(r"option|variant|choice|config|version|edition|sku|model", re.I)
_KIND_KEYS = re.compile(r"categor|department|^dept$|shelf|aisle|section|^kind$|product.?(type|family|group|kind)|"
                        r"item.?(type|kind)|offer.?kind|^family$|^group$", re.I)
# a key that marks a pending step of a checkout (a token to confirm with), not a refusal
_FIRST_PERSON = re.compile(r"\b(I|I'm|I'd|I've|I'll|my|me|mine)\b")
# a shopper line that wants an answer, and a request rule that wants the shopper told something
_ASKS = re.compile(r"\?|\b(confirm|let me know|tell me|acknowledge|reply|answer|get back to me)\b", re.I)
_TELL_RULE = re.compile(r"\b(tell|inform|let (me|them|the shopper) know|update|confirm|report|mention|explain|message)\b", re.I)
_REASON_RULE = re.compile(r"\b(reason|why|which (requirement|condition|need|one)|what (was|is|went|got) (missing|wrong|the problem)|what was lacking|explain|was the problem|couldn'?t be met|could not be met)\b", re.I)
_FINISH_STEP = re.compile(r"final|confirm|complete|checkout|pay|place|submit|commit|token", re.I)
_PENDING_KEYS = re.compile(r"token|pending|await|confirm|next.?step|checkout.?id|session", re.I)
_DECLINE_WORDS = re.compile(r"refus|declin|abandon|give.?up|no.?(buy|purchase|order)|reject|cancel|quit|skip", re.I)
_GONE_TOO = re.compile(r"\bnothing (?:is )?left\b|\blast (?:one|of (?:those|them|these|it)) (?:got|was|were|has been) (?:bought|sold|taken)", re.I)
# a shopper who refuses a message for naming too much at once, read from three shapes of the same complaint:
#   "one" + a few words + a sequencing word ("one at a time", "one by one", "one subject per turn", "one item and then the next")
_ONE_BY_ONE = re.compile(r"\bone\b(?:\s+\w+){0,2}?\s+(?:at a time|by one\b|per\b|a (?:message|turn)\b|(?:and )?then\b|and wait\b)|\bone at a time\b", re.I)
#   a request to cut the message up or to take only its first part ("split them up", "single file", "just the first one")
_SPLIT_UP = re.compile(r"\bsplit\b(?:\s+\w+){0,2}?\s+up\b|\bsingle file\b|\bin sequence\b|\bseparately\b|\b(?:just|only) the (?:first|opening)\b"
                       r"|\b(?:give|ask|send|hand) me (?:just )?the first\b", re.I)
#   a remark on the amount ("too many at once", "a lot at once", "smaller bites")
_TOO_MUCH = re.compile(r"\btoo many (?:questions|things|at once|balls|topics)\b|\b(?:a lot|too much|so much) at once\b|\bsmaller (?:bites|pieces|chunks|steps)\b", re.I)


def crowded_reply(text: str) -> bool:
    """Whether a shopper's line objects that the message asked about too much at once."""
    text = str(text or "")
    return bool(_ONE_BY_ONE.search(text) or _SPLIT_UP.search(text) or _TOO_MUCH.search(text))


_EMPTY_FAULT = re.compile(r"no.?items|empty.?(?:cart|basket)|(?:cart|basket) is empty|nothing (?:in|to order)", re.I)
_CLAIM_WORDS = re.compile(r"compat|fit|with|for\b|part|also", re.I)
_NO_BUY = re.compile(r"\b(?:pass|none|nothing|abstain|without|walk)\b", re.I)
_BUY_WORDS = re.compile(r"order|buy|purchase|check.?out|place|confirm", re.I)
_OUT_WORDS = re.compile(r"remov|discard|drop|delet|take|clear|out|minus|withdraw|cancel|unkeep|release", re.I)
_ID_HINT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")


def _compact(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > OBS_CHARS:
        text = text[:OBS_CHARS] + "...(cut)"
    return text


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """The answer object in a model's text (our reader: fenced, embedded or corrected objects)."""
    return find_json(text)

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
    """Arguments fitted to a tool's parameter schema (our fitter: undeclared keys dropped, values shaped to their types,
    bounds and enums), with the problems that still keep the call from the store."""
    function = {"parameters": schema if isinstance(schema, dict) else {}}
    fitted = shape_args(function, value if isinstance(value, dict) else {})
    return fitted, [f"{path}: {problem}" for problem in lacking(function, fitted)]

def _sentences(value: Any, out: list[str], depth: int = 0) -> None:
    """Shopper sentences anywhere in a value, appended to out (our sentence finder)."""
    for line in sentences(value, depth):
        if line not in out:
            out.append(line)

def _collect_ids(value: Any, out: set[str], depth: int = 0, keys: tuple[str, ...] = ()) -> None:
    """Listing ids anywhere in a value: under a key that says 'id' or under a key the store's schema or rows taught us."""
    if depth > 6:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and ("id" in key.lower() or flat(key) in keys) and _ID_HINT.match(item):
                out.add(item)
            else:
                _collect_ids(item, out, depth + 1, keys)
    elif isinstance(value, list):
        for item in value:
            _collect_ids(item, out, depth + 1, keys)


TRIM_CHARS = 180

def flat(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def words(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", flat(re.sub(r"[_.\-]+", " ", str(text))))


def trim(value: Any, depth: int = 0, chars: int = TRIM_CHARS) -> Any:
    if depth > 7:
        return None
    if isinstance(value, str):
        return value if len(value) <= chars else value[:chars] + "…"
    if isinstance(value, dict):
        return {k: trim(v, depth + 1, chars) for k, v in list(value.items())[:80]}
    if isinstance(value, list):
        return [trim(v, depth + 1, chars) for v in value[:60]]
    return value


def render(value: Any, limit: int = OBS_CHARS) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "…(cut)"


def find_json(text: Any) -> dict | None:
    """The first complete JSON object in a model answer (fenced, bare or wrapped in prose)."""
    if isinstance(text, list):
        text = " ".join(str(p.get("text") or "") for p in text if isinstance(p, dict))
    if not isinstance(text, str):
        return None
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        depth = 0
        for i in range(start, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                try:
                    parsed = json.loads(text[start:i + 1])
                except ValueError:
                    break
                if isinstance(parsed, dict):
                    return parsed
                break
    return None


def sentences(value: Any, depth: int = 0) -> list[str]:
    """Lines of at least four words anywhere in a value (lists of strings, strings under note-like keys, bare strings)."""
    out: list[str] = []
    if depth > 4:
        return out
    if isinstance(value, str):
        if len(value.split()) >= 4:
            out.append(value[:500])
    elif isinstance(value, list):
        for v in value[:10]:
            out += sentences(v, depth + 1)
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, (list, dict)) or (isinstance(v, str) and re.search(r"note|news|message|refus|shopper|customer|update|change|say|complain|object", flat(k))):
                out += sentences(v, depth + 1)
    return out[:8]


def idlike(key: Any, value: Any) -> bool:
    return (isinstance(value, str) and 3 <= len(value) <= 64 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-:.]*", value) is not None
            and any(c.isdigit() for c in value) and re.search(r"\bid\b|_id|sku|listing|product|item|target|article|code|pid|ref|key", flat(re.sub(r"[_\-]+", " ", str(key)))) is not None)


GONE_WORDS = r"sold out|out of stock|unavailab|no longer|zero left|none left|nothing left|all gone|is gone|it's gone|now gone|gone from|dried up|ran dry|run dry|cleared out|bought the last|can't be ordered|cannot be ordered|grayed out|greyed out|hit zero|no units|none remaining|been sold|snapped up|not in stock|isn't in stock|no stock|last one|just sold"
PRICE_WORDS = r"price|cost|charg|priced|bumped|gone up|went up|come down|dropped|cents|dearer|cheaper|ceiling|limit|budget|coupon"
OUT_WORDS = r"remov|delet|drop|discard|unbag|releas|clear|take out|\btake\b|erase|cancel|decreas|minus|subtract"
IN_WORDS = r"\badd|\bput|include|insert|\bhold|reserve|keep|\bset\b|stash"
META_KEYS = r"brand|maker|manufactur|seller|vendor|merchant|company|shop|store|title|name|heading|sku|url|link|image|rating|review|description|overview|summary|blurb|^id$|_id$|price|amount|cost|stock|on hand|in stock|availab|categor|department|shelf|aisle|section|collection|options?$|variants?$"
ENDS = r"\bfinal\b|finish|\bends?\b|end the|terminat|no more actions|close out|closes? the|complet(?:e|es|ing) the (?:task|session|episode|purchase|order)|last action|nothing can follow"
PENDING_KEYS = r"token|pending|await|confirm|next.?step|checkout.?id|session"
USED_UP = r"budget_exceeded|quota|allowance|used up|exhaust|no more\b.{0,40}\b(may|can|allowed|left)|limit reached|reached the limit|maximum (number|reached)|already used"
QUIT_WORDS = r"refus|declin|abandon|give.?up|no.?(buy|purchase|order)|reject|cancel|quit|skip|walk"

KIND_SIGNS = {
    "quit": r"declin|abstain|refus|give up|giving up|walk away|abandon|cancel|quit|empty.handed|nothing (?:fits|suits|found|suitable|available|matches|matching|appropriate|works)|no purchase|without (?:a |an |any )?(?:purchase|order|buying|ordering)|not (?:buy|order|purchas)|no order|skip (?:the )?(?:order|purchase)|unable|cannot|can't|no.match|not found|unsuitable|none (?:fit|match|suit)|stop shopping|report that nothing",
    "buy": r"purchase|order|\bbuy|check ?out|settle|\bpay\b|place|confirm|complete|commit|finali",
    "talk": r"messag|\bask|\bsay\b|chat",
    "cart": r"cart|basket|\bbag\b|trolley|\bhold\b|reserve|stash",
    "stock": r"stock|availab|inventory|on.hand|units left|how many units",
    "seek": r"search|find|look ?up|browse|query|catalog|filter|\blist\b|discover|explore|hunt|locate",
    "peek": r"detail|view|\bspec|sheet",
}
TEXTY = r"text|message|content|question|say|body|words|note|msg|utterance|prompt|remark|reason|comment"
IDDY = r"\bid\b|_id|sku|listing|product|item|target|article|code|pid|ref"
QUERYISH = r"query|\bq\b|text|keyword|term|search|word|filter|page|cap|price|brand|category|department|maker|ceiling|budget|collection|aisle|section|shelf"


FINAL_MORE = r"afterwards|session (?:ends|clos|is over)|nothing (?:can|may) (?:be done|follow)|no further"
NEGATED = r"without|nothing (?!can|may|else)|empty.handed|no purchase|not (?:buy|order)|give.?up"
HOLDER = r"tote|crate|hamper|cart|basket|bag|trolley"
PERSON = r"shopper|customer|buyer|person|requester|client|owner"
ID_PARAM = IDDY + r"|key\b|handle"
ALLOWANCE = r"allowance|quota|remaining|usage|budget|left"


def sort_tool(function: dict) -> tuple[str, bool]:
    """(kind, ends the episode) for one tool, read off the shape of its parameters first (no argument, one free text, one
    listing pointer, a pointer with a count or a mode, a list of pointers, several search fields) and its description's
    meaning second; the word scoring of the name and description decides only what the shape leaves open."""
    words_kind, ends = word_sort(function)
    desc = flat(re.sub(r"([a-z])([A-Z])", r"\1 \2", f"{function.get('name') or ''} {function.get('description') or ''}").replace("_", " ").replace("-", " "))
    props = ((function.get("parameters") or {}).get("properties") or {})
    final = ends or re.search(FINAL_MORE, desc) is not None
    negated = re.search(NEGATED, desc) is not None
    holder = re.search(HOLDER, desc) is not None
    kinds = {k: (r.get("type") if isinstance(r, dict) else None) for k, r in props.items()}
    enums = [k for k, r in props.items() if isinstance(r, dict) and r.get("enum")]
    strings = [k for k, t in kinds.items() if (t == "string" or (isinstance(t, list) and "string" in t)) and k not in enums]
    counts = [k for k, t in kinds.items() if t in ("integer", "number")]
    arrays = [k for k, t in kinds.items() if t == "array"]
    pointer = lambda k: re.search(ID_PARAM, flat(re.sub(r"([a-z])([A-Z])", r"\1 \2", k).replace("_", " "))) is not None
    if not props:
        if final and negated:
            return "quit", True
        if re.search(KIND_SIGNS["buy"], desc) and not negated:
            return "buy", final
        if re.search(ALLOWANCE, desc) and re.search(KIND_SIGNS["talk"] + "|note", desc):
            return "usage", False
        if holder:
            return "cartview", False
        return words_kind, ends
    line_lists = [k for k in arrays if isinstance(props[k].get("items"), dict) and (props[k]["items"].get("properties") or {})]
    if line_lists and not final:
        return "cart", False                  # a list of line objects: the whole basket set in one call
    if final and (negated or words_kind == "quit") and not arrays and len(strings) <= 1 and not counts:
        return "quit", True
    if final and not negated and not strings and not enums:
        return "buy", True
    if len(props) == 1 and len(strings) == 1 and not pointer(strings[0]) and (re.search(PERSON, desc) or words_kind == "talk"):
        return "talk", False
    if holder and (enums or counts or arrays) and (strings or arrays) and len(strings) <= 1:
        if not enums and re.search(OUT_WORDS + "|out\\b", desc) and not re.search(IN_WORDS + "|into", desc):
            return "uncart", False
        return "cart", False
    if len(props) == 1 and len(strings) == 1 and pointer(strings[0]):
        if holder and not re.search(KIND_SIGNS["peek"] + "|record|entry|attribute|propert|characteristic", desc):
            return ("uncart" if re.search(OUT_WORDS + "|out\\b", desc) else "cart"), False
        return ("stock" if words_kind == "stock" else "peek"), False
    if arrays and not holder and words_kind not in ("buy", "quit"):
        return "compare", False
    if len(props) >= 2 and strings and not holder and words_kind not in ("buy", "quit", "talk"):
        return "seek", False
    return words_kind, ends


def word_sort(function: dict) -> tuple[str, bool]:
    """(kind, ends the episode) for one tool, from its name, description and parameters: every kind scores, the top wins."""
    name = str(function.get("name") or "")
    desc = flat(re.sub(r"[_\-]+", " ", f"{name} {function.get('description') or ''}"))
    name_words = flat(re.sub(r"[_\-]+", " ", name))
    params = [flat(re.sub(r"[_\-]+", " ", p)) for p in ((function.get("parameters") or {}).get("properties") or {})]
    ends = re.search(ENDS, desc) is not None
    texty = any(re.search(TEXTY, p) for p in params)
    iddy = any(re.search(IDDY, p) for p in params)
    queryish = any(re.search(QUERYISH, p) for p in params)
    score = {kind: 2.0 * len(re.findall(sign, desc)) + 3.0 * len(re.findall(sign, name_words)) for kind, sign in KIND_SIGNS.items()}
    if re.search(KIND_SIGNS["cart"], desc):
        score["peek"] -= 4
        score["talk"] -= 3
    if re.search(KIND_SIGNS["buy"], name_words) and not re.search(r"add|put|insert|remov|delet|drop|discard|view|show|review|list|contents|status|empty|clear", name_words):
        score["cart"] -= 4
    # the shape of the parameters speaks louder than the words: one free-text slot and nothing to point at is a line
    # to the shopper; one pointer and nothing to search with is a page to open
    lone_text = texty and not iddy and len(params) == 1
    lone_pointer = iddy and not queryish and len(params) == 1
    score["talk"] += (6 if lone_text else 3) if texty else -4
    score["buy"] -= 4 if lone_text else 0
    score["peek"] += (4 if lone_pointer else 2 if iddy else -3) + (-2 if queryish else 0)
    score["seek"] += 2 if queryish else 0
    if re.search(KIND_SIGNS["seek"], desc):
        score["buy"] -= 2
        score["stock"] -= 2
    if not params and score["cart"] > 0 and re.search(r"view|show|contents|review|status|list|your|what", desc):
        return "cartview", ends
    if ends:
        return ("quit" if score["quit"] > 0 else "buy"), ends
    if score["quit"] > 0:
        return "quit", ends
    best = max(score, key=lambda k: (score[k], k))
    kind = best if score[best] > 0 else "misc"
    if kind == "cart":
        if not params and re.search(r"view|show|contents|review|status|list|what", desc):
            return "cartview", ends
        if re.search(OUT_WORDS, name_words) and not re.search(IN_WORDS, name_words):
            return "uncart", ends
    return kind, ends


def mode_pair(values: list[Any], hint: Any = None) -> tuple[str | None, str | None]:
    """Which value of a cart tool's mode enum puts a line in and which takes it out. The brief's reading of the schema
    first; then the plain cues; then the shape of the pair: a value that is the other with a negating prefix or a longer
    form of it takes out ('lodge' / 'unlodge'), and in a two-value mode the one left over is the other half."""
    vals = [v for v in values if isinstance(v, str)]
    add = out = None
    if isinstance(hint, dict):
        add = hint.get("add") if hint.get("add") in vals else None
        out = hint.get("remove") if hint.get("remove") in vals else None
    add = add or next((v for v in vals if re.search(IN_WORDS, v, re.I) and not _OUT_WORDS.search(v)), None)
    out = out or next((v for v in vals if v != add and _OUT_WORDS.search(v)), None)
    if add is None or out is None:
        for a in vals:
            for b in vals:
                if a != b and len(b) > len(a) and flat(a) in flat(b) and (add in (None, a)) and (out in (None, b)):
                    add, out = a, b
    if len(vals) >= 2:
        # stores list the putting value first and the taking-out one next: the order decides what the words did not
        rest = [v for v in vals if v not in (add, out)]
        if add is None and rest:
            add = rest.pop(0)
        if out is None and rest:
            out = rest.pop(0)
    return add, out


def arg_value(value: Any, rule: dict) -> Any:
    """One argument shaped to its schema: numbers from strings, integers from whole floats, enums by case, bounds applied."""
    kinds = rule.get("type")
    kinds = [kinds] if isinstance(kinds, str) else (kinds or [])
    if isinstance(value, str) and "integer" in kinds and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
        value = int(float(value))
    elif isinstance(value, str) and "number" in kinds and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
        value = float(value)
    elif isinstance(value, float) and "integer" in kinds and value.is_integer():
        value = int(value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and kinds == ["string"]:
        value = str(value)
    elif isinstance(value, str) and "boolean" in kinds and value.strip().lower() in ("true", "false"):
        value = value.strip().lower() == "true"
    if isinstance(rule.get("enum"), list) and isinstance(value, str) and value not in rule["enum"]:
        match = next((e for e in rule["enum"] if isinstance(e, str) and e.lower() == value.strip().lower()), None)
        value = match if match is not None else value
    if isinstance(value, str) and isinstance(rule.get("maxLength"), int) and len(value) > rule["maxLength"]:
        value = value[: rule["maxLength"]]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(rule.get("minimum"), (int, float)) and value < rule["minimum"]:
            value = rule["minimum"]
        if isinstance(rule.get("maximum"), (int, float)) and value > rule["maximum"]:
            value = rule["maximum"]
    return value


def shape_args(function: dict, args: dict[str, Any]) -> dict[str, Any]:
    """Arguments fitted to the tool's schema: undeclared ones dropped, values shaped, defaults for missing optionals left to the store."""
    params = function.get("parameters") or {}
    props = params.get("properties") or {}
    if not isinstance(args, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in args.items():
        if key not in props:
            continue
        rule = props[key] if isinstance(props[key], dict) else {}
        if value is None:
            continue
        out[key] = arg_value(value, rule)
    return out


def lacking(function: dict, args: dict[str, Any]) -> list[str]:
    """Required arguments missing or of the wrong type, enum values outside the list."""
    params = function.get("parameters") or {}
    props = params.get("properties") or {}
    problems = []
    for key in params.get("required") or []:
        if key not in args:
            problems.append(f"{key} is required")
    for key, value in args.items():
        rule = props.get(key) if isinstance(props.get(key), dict) else {}
        kinds = rule.get("type")
        kinds = [kinds] if isinstance(kinds, str) else (kinds or [])
        ok = not kinds or any(
            (k == "string" and isinstance(value, str)) or (k == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (k == "number" and isinstance(value, (int, float)) and not isinstance(value, bool)) or (k == "boolean" and isinstance(value, bool))
            or (k == "object" and isinstance(value, dict)) or (k == "array" and isinstance(value, list)) or (k == "null" and value is None) for k in kinds)
        if not ok:
            problems.append(f"{key} must be {'/'.join(kinds)}")
        elif isinstance(rule.get("enum"), list) and value not in rule["enum"]:
            problems.append(f"{key} must be one of {rule['enum']}")
    return problems


# ----------------------------------------------------------------------------- the shelf: what the store has shown

ID_KEYS = ("item_pid", "id", "product_id", "sku", "listing_id", "article_id", "pid", "listing_ref", "item_id", "target_id", "product", "ref")
PRICE_KEYS = ("price", "unit_price", "cost", "amount", "list_price", "price_php", "current_price", "sale_price")
STOCK_KEYS = ("in_stock", "stock", "available", "availability", "on_hand", "instock", "units", "stock_level", "quantity_available")
TITLE_KEYS = ("title", "name", "product_name", "heading", "label")
GROUP_KEYS = ("category", "department", "aisle", "section", "collection", "shelf", "dept", "kind", "product_type", "category_path")
SHEET_KEYS = r"trait|attribute|propert|spec|fact|detail|feature|characteristic"
WHO_KEYS = r"seller|vendor|brand|maker"
# single-word names of a price, stock, title or group field: a key carrying one of these words never holds a listing's id
# (a learnt id stem such as 'item' must not pick 'item_list_price')
_NOT_ID_WORDS = frozenset(n for n in PRICE_KEYS + STOCK_KEYS + TITLE_KEYS + GROUP_KEYS if "_" not in n)


def key_words(key: Any) -> list[str]:
    """The words of a record key: 'cost_amount', 'costAmount' and 'Cost Amount' all give ['cost', 'amount']."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


def field_key(record: dict, names: tuple[str, ...]) -> Any:
    """The key of a record that holds one of the usual names: the exact name first, then a key the name is one word of
    ('cost_amount' for 'cost', 'item_name' for 'name'), the earliest name and the shortest key winning; None when none."""
    low = {flat(k): k for k in record}
    for n in names:
        if n in record:
            return n
        if n in low:
            return low[n]
    for n in names:
        if "_" in n:
            continue
        found = sorted((len(key_words(k)), str(k), k) for k in record if n in key_words(k) and len(key_words(k)) <= 3 and record[k] is not None)
        if found:
            return found[0][2]
    return None


def field(record: dict, names: tuple[str, ...]) -> Any:
    """A record's value under one of the usual names (see field_key)."""
    key = field_key(record, names)
    return None if key is None else record[key]




def stock_key(record: dict) -> Any:
    """The key of a record's stock flag: a usual stock name, else a yes/no field named like 'orderable' or 'purchasable'."""
    key = field_key(record, STOCK_KEYS)
    if key is None:
        key = next((k for k, v in record.items() if isinstance(v, bool) and re.search(r"able$|avail", flat(k))), None)
    return key


def group_key(record: dict) -> Any:
    """The key of a record's shelf or category: a usual name, else a key with 'group' among its words."""
    key = field_key(record, GROUP_KEYS)
    if key is None:
        key = field_key(record, ("group",))
    return key


def as_price(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and len(value.split()) <= 6:
        # an amount written with its unit in any words ("95¢", "95 cents of a dollar", "$0.95", "1,200 usd-cents"): the
        # one figure of a short text
        found = _FIGURE.findall(value)
        if len(found) == 1:
            whole, part = found[0]
            return float(whole.replace(",", "") + ("." + part if part else ""))
    return None


def price_residue(value: Any) -> str:
    """What a written amount says besides its figure (its unit words), to tell a price column from other texts."""
    return flat(_FIGURE.sub("#", value)) if isinstance(value, str) else ""


def last_page(observation: Any) -> bool:
    """Whether a search result says it is the last page: a key about the next page or more results that holds nothing."""
    if not isinstance(observation, dict):
        return False
    for key, value in observation.items():
        if re.search(r"next|continu|cursor|has more|more (?:results|pages|rows|hits)", flat(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).replace("_", " "))):
            if value is None or value is False or value == "" or value == 0:
                return True
    return False


def as_stock(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str) and value.strip():
        return re.search(r"out|sold|unavail|none|\bno\b|^0$|gone", value.lower()) is None
    return None


def pairs_to_sheet(value: Any) -> dict[str, Any] | None:
    """A property sheet delivered as a list of small label/value records."""
    if not (isinstance(value, list) and len(value) >= 3 and all(isinstance(x, dict) and 2 <= len(x) <= 4 for x in value[:60])):
        return None
    out: dict[str, Any] = {}
    for pair in value[:60]:
        label = next((v for k, v in pair.items() if isinstance(v, str) and re.search(r"label|name|key|attribute|property|field|spec|trait", k.lower())), None)
        worth = next((v for k, v in pair.items() if re.search(r"value|val\b|content|setting|text|worth", k.lower()) and isinstance(v, (str, int, float, bool))), None)
        if label is None or worth is None:
            return None
        out[label] = worth
    return out if len(out) >= 3 else None


def spend_lines(max_steps: int) -> tuple[float, float, float]:
    """The soft, hard and stop spending lines of one episode, by its step cap (the race spend governor above: a short task
    keeps a small purse, a long bundle the turns it needs to finish; about ninety race episodes share one key)."""
    hard = min(HARD_ROOF, HARD_FLOOR + HARD_PER_STEP * max(0, max_steps - HARD_FROM))
    return round(hard * SOFT_SHARE, 4), round(hard, 4), round(hard + STOP_OVER, 4)


class Shelf:
    """Every listing the store has shown: price, stock, title, group, the property sheet once opened, the raw records."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.sheets: dict[str, dict[str, Any]] = {}
        self.records: dict[str, list[str]] = {}
        self.ids: set[str] = set()
        self.id_keys: list[str] = []          # key names that hold a listing's id here, learnt from the tools and the rows
        self.kind_words: set[str] = set()     # words of the product kind the shopper asked for, once the plan is read
        self.sheets_seen: dict[str, int] = {}  # listing -> the step its sheet was last delivered
        self.now = 0                          # the step of the result being absorbed
        self.layout: dict[str, list[str]] = {}  # field role -> record keys that hold it in this store, learnt from the shape of the rows
        self.scale: float | None = None       # the request's limit, the yardstick for which number of a row is its price
        self.factor = 1.0                     # what one written listing amount is worth in the request's own price unit
        self.factor_set = False               # decided once per world, from the first price column beside the limit
        self.factor_doubt: tuple[float, str, float] | None = None  # (limit, a written amount, other factor) when two scales both fit

    def written_factors(self, values: list[Any]) -> list[float]:
        """The scales (1, 100 or 1/100) that put the median of a column of written amounts ("$0.95") within 0.05x-20x of
        the request's limit. Only amounts written with a unit ("$0.95", "95 cents") can be on another scale than the limit; bare
        numbers stay."""
        figures = [as_price(v) for v in values]
        if not self.scale or any(f is None for f in figures) or not figures:
            return [1.0]
        if not all(isinstance(v, str) and re.search(r"[^\d\s.,]", v) for v in values):
            return [1.0]
        middle = sorted(figures)[len(figures) // 2]
        return [f for f in (1.0, 100.0, 0.01) if 0.05 * self.scale <= middle * f <= 20 * self.scale] or [1.0]

    def decide_factor(self, values: list[Any]) -> None:
        if self.factor_set or not self.scale:
            return
        fits = self.written_factors(values)
        self.factor_set = True
        self.factor = fits[0] if 1.0 not in fits else 1.0
        if len(fits) > 1:
            other = next(f for f in fits if f != self.factor)
            self.factor_doubt = (float(self.scale), str(values[len(values) // 2]), other)

    def money(self, value: Any) -> float | None:
        """A listing amount in the request's price unit."""
        price = as_price(value)
        if price is not None and self.factor != 1.0 and isinstance(value, str) and re.search(r"[^\d\s.,]", value):
            price = round(price * self.factor, 6)
        return price

    def learn_keys(self, names: list[str]) -> None:
        """Key names the store's own tools use for a listing (the argument a details or cart tool takes) and their
        stems, so rows keyed the same way are read as listings whatever the name."""
        for name in names:
            low = flat(re.sub(r"[_\-]+", " ", str(name)))
            for cand in (flat(name), low, re.sub(r"\b(id|ids|identifier|number|code|key)\b", "", low).strip(), low.replace(" ", "_")):
                if cand and len(cand) >= 2 and cand not in self.id_keys:
                    self.id_keys.append(cand)

    def learn_layout(self, rows: list[dict]) -> None:
        """Which key of this store's listing rows holds the id, the price, the stock, the title and the group, read off
        the shape of a batch of similar rows rather than their names: the id is the one id-shaped string unique on every
        row, the price the varied amount nearest the request's limit, the stock a yes/no or a small count, the title the
        short varied phrase, the group a phrase the rows share. Learnt once per store; names only break ties."""
        rows = [r for r in rows if isinstance(r, dict)][:40]
        if len(rows) < 2:
            return
        shared = [k for k in rows[0] if all(k in r for r in rows)]
        if len(shared) < 2:
            return
        n = len(rows)
        column = {k: [r[k] for r in rows] for k in shared}
        taken: set[str] = set()

        def keep(role: str, key: Any) -> None:
            if key is None:
                return
            taken.add(key)
            known = self.layout.setdefault(role, [])
            if key not in known:
                known.insert(0, key)

        ids = [k for k, vs in column.items() if all(isinstance(v, str) and _ID_HINT.match(v.strip()) and any(c.isdigit() for c in v) for v in vs)
               and len(set(vs)) == n]
        if ids:
            keep("id", sorted(ids, key=lambda k: (flat(k) not in self.id_keys and not any(w in self.id_keys for w in key_words(k)), shared.index(k)))[0])
        amounts = []
        for k, vs in column.items():
            if k in taken or any(isinstance(v, bool) for v in vs):
                continue
            figures = [as_price(v) if not isinstance(v, (dict, list)) else None for v in vs]
            if any(f is None for f in figures) or len({price_residue(v) for v in vs}) > 1:
                continue             # every row's amount, written the same way (a title with a number in it is not one)
            unit_written = all(isinstance(v, str) and (re.search(_CURRENCY_SIGN, v) or re.match(r"#\s*[A-Za-z]", price_residue(v))) for v in vs)
            fits = ([1.0] if not unit_written else self.written_factors(vs)) if not self.factor_set else [self.factor if unit_written else 1.0]
            figures = [f * (fits[0] if 1.0 not in fits else 1.0) for f in figures]
            score = 0.0
            if any(abs(f - round(f)) > 1e-9 for f in figures):
                score += 2
            if len(set(figures)) >= max(2, 0.6 * n):
                score += 2
            if any(isinstance(v, str) for v in vs):
                score += 1               # an amount written with its unit ("$12.50")
            ordered = sorted(figures)
            middle = ordered[len(ordered) // 2]
            if self.scale:
                score += 4 if self.scale / 40 <= middle <= self.scale * 4 else -2
            if max(figures) <= 5 and (not self.scale or self.scale > 25):
                score -= 4               # a rating
            if all(abs(f - round(f)) < 1e-9 for f in figures) and min(figures) >= 0 and max(figures) <= 60 and (not self.scale or self.scale > 300):
                score -= 3               # a count of units
            if field_key({k: 1}, PRICE_KEYS) is not None:
                score += 1.5
            amounts.append((score, -shared.index(k), k))
        if amounts:
            keep("price", max(amounts)[2])
            self.decide_factor(column[max(amounts)[2]])
        flags = [k for k, vs in column.items() if k not in taken and all(isinstance(v, bool) for v in vs)]
        if flags:
            keep("stock", sorted(flags, key=lambda k: (stock_key({k: True}) is None, shared.index(k)))[0])
        texts = []
        for k, vs in column.items():
            if k in taken or not all(isinstance(v, str) for v in vs):
                continue
            sizes = [len(v.split()) for v in vs]
            mean = sum(sizes) / n
            if 2 <= mean <= 30 and len(set(vs)) == n:
                overlap = sum(1 for v in vs if set(words(v)) & self.kind_words)
                texts.append((-overlap, mean, shared.index(k), k))
        if texts:
            keep("title", min(texts)[3])
        if n >= 3:
            groups = []
            for k, vs in column.items():
                if k in taken:
                    continue
                labels = [v[-1] if isinstance(v, list) and v and isinstance(v[-1], str) else v for v in vs]
                if not all(isinstance(v, str) and v.strip() and (" " in v.strip() or not any(c.isdigit() for c in v)) for v in labels):
                    continue
                distinct = len(set(labels))
                if distinct < n and distinct <= max(1, n // 2):
                    groups.append((group_key({k: 1}) is None, not any(isinstance(v, list) for v in vs), distinct, shared.index(k), k))
            if groups:
                keep("group", min(groups)[4])

    def role_key(self, record: dict, role: str) -> Any:
        """The key of one record that holds a field role: a key learnt from this store's rows first, the usual names after."""
        for key in self.layout.get(role) or []:
            if key in record and record[key] is not None:
                return key
        if role == "price":
            key = field_key(record, PRICE_KEYS)
            if key is None:
                # a record the rows never showed (a cart line, a page): the number that equals the listing's known price
                ident = self.id_field(record)
                known = (self.rows.get(str(ident)) or {}).get("price") if ident is not None else None
                if known is not None:
                    key = next((k for k, v in record.items() if not isinstance(v, (bool, dict, list)) and self.money(v) == known), None)
            return key
        if role == "stock":
            return stock_key(record)
        if role == "title":
            return field_key(record, TITLE_KEYS)
        if role == "group":
            return group_key(record)
        return None

    def role_value(self, record: dict, role: str) -> Any:
        key = self.role_key(record, role)
        return None if key is None else record.get(key)

    def is_listing(self, record: dict) -> bool:
        """A record that is one listing: it carries an id and a price (any key this store uses for them)."""
        ident = self.id_field(record)
        return ident is not None and (self.role_value(record, "price") is not None or str(ident).strip() in self.ids)

    def infer_key(self, rows: list[dict]) -> None:
        """When a batch of similar records carries no known id key, the one key whose values are present in every record,
        unique across them and shaped like identifiers names the listing; it is remembered for the rest of the episode."""
        rows = [r for r in rows if isinstance(r, dict)]
        if len(rows) < 3 or any(self.id_field(r) is not None for r in rows[:3]):
            return
        known = {flat(k) for k in PRICE_KEYS + STOCK_KEYS + TITLE_KEYS + GROUP_KEYS} | {flat(k) for role, keys in self.layout.items() if role != "id" for k in keys}
        for cand in rows[0]:
            if flat(cand) in known or not isinstance(rows[0].get(cand), str):
                continue
            values = [r.get(cand) for r in rows]
            if not all(isinstance(v, str) and _ID_HINT.match(v) and any(c.isdigit() for c in v) for v in values):
                continue
            if len(set(values)) == len(values) and all(self.role_value(r, "price") is not None or self.role_value(r, "title") is not None for r in rows):
                self.learn_keys([cand])
                return

    def id_key(self, record: dict) -> Any:
        """The key of a record's listing id (learnt keys first, the usual names after); never a price, stock, title or
        group field, nor a yes/no flag."""
        for key in self.layout.get("id") or []:
            if isinstance(record.get(key), (str, int)) and not isinstance(record.get(key), bool):
                return key
        others = {k for role, keys in self.layout.items() if role != "id" for k in keys}
        plain = {k: v for k, v in record.items() if not isinstance(v, bool) and k not in others and not (set(key_words(k)) & _NOT_ID_WORDS)}
        return field_key(plain, tuple(self.id_keys) + ID_KEYS)

    def id_field(self, record: dict) -> Any:
        """A record's listing id under any key the store uses for it (see id_key)."""
        key = self.id_key(record)
        return None if key is None else record[key]

    def absorb(self, observation: Any, step: int, kind: str = "") -> None:
        self.now = step
        stack: list[tuple[Any, int]] = [(observation, 0)]
        while stack:
            value, depth = stack.pop()
            if depth > 5:
                continue
            if isinstance(value, list):
                if len(value) >= 2 and all(isinstance(v, dict) for v in value[:3]):
                    self.learn_layout(value)
                if len(value) >= 3 and all(isinstance(v, dict) for v in value[:3]):
                    self.infer_key(value[:12])
                stack.extend((v, depth + 1) for v in value[:80])
                continue
            if not isinstance(value, dict):
                continue
            stack.extend((v, depth + 1) for v in value.values())
            key = self.id_field(value)
            if isinstance(key, (int, float)) and not isinstance(key, bool):
                key = str(key)
            if not isinstance(key, str) or not key.strip() or " " in key.strip() or len(key) > 64:
                continue
            key = key.strip()
            self.ids.add(key)
            row = self.rows.setdefault(key, {"price": None, "stock": None, "title": "", "group": "", "sheet": False, "step": 0, "stock_step": -1, "price_step": -1})
            price = self.money(self.role_value(value, "price"))
            if price is not None:
                row["price"], row["price_step"] = price, step
            skey = self.role_key(value, "stock")
            stock_raw = None if skey is None else value[skey]
            stock = as_stock(stock_raw) if not isinstance(stock_raw, (dict, list)) else None
            if stock is not None:
                row["stock"], row["stock_step"] = stock, step
            title = self.role_value(value, "title")
            if isinstance(title, str) and title.strip():
                row["title"] = " ".join(title.split())[:80]
            gkey = self.role_key(value, "group")
            group = None if gkey is None else value[gkey]
            if isinstance(group, list) and group:
                group = group[-1]
            if isinstance(group, str) and group.strip():
                row["group"] = group.strip()[:60]
            roles_here = {self.id_key(value), self.role_key(value, "price"), skey, self.role_key(value, "title"), gkey}
            for k, v in value.items():
                # who offers the listing (its seller, shop, brand or maker: any short name-like text of the record that is
                # not its id, price, stock, title or group) and the free text its page carries, for the needs that point
                # at another listing's page
                claim = isinstance(v, str) and k not in roles_here and ":" in v and bool(codes_in(v))   # "Compatible with: VIR2110."
                if (isinstance(v, str) and v.strip() and len(v) < 40 and k not in roles_here and len(v.split()) <= 4 and not claim
                        and not (any(c.isdigit() for c in v) and _ID_HINT.match(v.strip()))
                        and not re.search(r"remark|note|descri|text", flat(k))):
                    who = row.setdefault("who", [])
                    if v.strip()[:60] not in who:
                        who.append(v.strip()[:60])
                        del who[:-4]
                elif isinstance(v, str) and v.strip() and (len(v) >= 40 or claim or re.search(r"descri|print|remark|blurb", flat(k))) and not re.search(r"title|name|url|link", flat(k)):
                    notes = row.setdefault("notes", [])
                    if v[:700] not in notes:
                        notes.append(v[:700])
                        del notes[:-4]
            # a nested record of plain values is the property sheet, however deep a chain of one-key wrappers holds it
            # and whatever the store calls it: the one such record beside the listing's id, else the one whose key says
            # so, else the largest
            nested: dict[Any, Any] = {}
            for k, v in value.items():
                inner, hops = v, 0
                while isinstance(inner, dict) and len(inner) == 1 and isinstance(next(iter(inner.values())), (dict, list)) and hops < 3:
                    inner, hops = next(iter(inner.values())), hops + 1
                if isinstance(inner, dict) and len(inner) >= 2 and all(isinstance(x, (str, int, float, bool, type(None))) for x in inner.values()):
                    nested[k] = inner
                elif (isinstance(inner, dict) and re.search(SHEET_KEYS, flat(k)) and any(isinstance(x, dict) for x in inner.values())
                      and all(isinstance(x, (str, int, float, bool, type(None), dict)) for x in inner.values())):
                    # a named sheet that groups some of its properties in sub-records ("physical": {...}): one level is
                    # flattened, a sub-record's plain values joining the sheet under their own labels
                    merged = {a: b for a, b in inner.items() if not isinstance(b, dict)}
                    for sub in (x for x in inner.values() if isinstance(x, dict)):
                        for a, b in sub.items():
                            if isinstance(b, (str, int, float, bool)) and a not in merged:
                                merged[a] = b
                    if len(merged) >= 2:
                        nested[k] = merged
                elif pairs_to_sheet(inner) is not None:
                    nested[k] = pairs_to_sheet(inner)
            named = [k for k in nested if re.search(SHEET_KEYS, flat(k))]
            price_key = self.role_key(value, "price")
            # the record's own price written as an object (an amount and its unit) is the price, never a property sheet
            def figure_object(inner: dict) -> bool:
                # a figure with its unit (an amount beside a short code), not a property sheet
                vals = list(inner.values())
                return len(vals) <= 3 and any(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vals) and all(
                    isinstance(x, (int, float)) or (isinstance(x, str) and len(x.strip()) <= 4) for x in vals if x is not None)
            plain = [k for k in nested if not re.search(META_KEYS, flat(re.sub(r"[_\-]+", " ", k))) and k != price_key and not figure_object(nested[k])]
            chosen = named or (plain if len(plain) == 1 else sorted(plain, key=lambda k: -len(nested[k]))[:1] if plain and max(len(nested[k]) for k in plain) >= 3 else [])
            for k in chosen:
                self.take_sheet(key, nested[k])
            if not row["sheet"] and kind in ("peek", "stock", "view"):
                known = set(ID_KEYS + PRICE_KEYS + STOCK_KEYS + TITLE_KEYS + GROUP_KEYS) | set(self.id_keys)
                # the keys this record uses for its id, price, stock, title and group are listing meta, never properties;
                # nor is a sentence (a description or a remark is not a property value)
                own = {self.id_key(value), self.role_key(value, "price"), skey, self.role_key(value, "title"), gkey} - {None}
                loose = {str(a): b for a, b in value.items() if isinstance(b, (str, int, float)) and not isinstance(b, bool) and flat(a) not in known
                         and a not in own and not (isinstance(b, str) and len(b.split()) >= 6)
                         and not re.search(META_KEYS, flat(re.sub(r"[_\-]+", " ", a)))}
                if len(loose) >= 3:
                    self.take_sheet(key, loose)
            if any(re.search(r"descri", flat(k)) for k in value):
                row["paged"] = True          # the listing's own page (its description) has been read
            kept = self.records.setdefault(key, [])
            kept.append(render(trim(value, chars=260), RECORD_CHARS))
            del kept[:-3]
            row["step"] = step

    def take_sheet(self, key: str, sheet: dict[str, Any]) -> None:
        """Labels that differ only in case, spaces, hyphens or underscores are one field, so their values are kept
        together; a dotted namespace ('clothing.material') stays apart from the plain label."""
        clean: dict[str, Any] = {}
        for a, b in list(sheet.items())[:60]:
            if b is None:
                continue
            label = re.sub(r"[_\-\s]+", " ", str(a)).strip().lower()
            if label in clean and flat(clean[label]) != flat(b) and flat(b) not in flat(clean[label]):
                clean[label] = f"{clean[label]}; {b}"
            else:
                clean[label] = b
        if clean:
            self.rows.setdefault(key, {"price": None, "stock": None, "title": "", "group": "", "sheet": False, "step": 0, "stock_step": -1, "price_step": -1})
            self.rows[key]["sheet"] = True
            self.sheets[key] = {**self.sheets.get(key, {}), **clean}
            self.sheets_seen[key] = self.now

    def says(self, key: str, prop: str, value: str) -> str:
        """How a listing's own sheet answers one condition: 'yes', 'no' or 'silent'. A label whose words cover the
        property's words decides; without one the title decides, else the sheet is silent."""
        pw = [w for w in words(prop) if w not in ("type", "kind", "the", "of")]
        vw = [w for w in words(value) if len(w) > 1]
        if not pw or not vw:
            return "silent"
        named = self.named(key, pw)
        labelled = [v for _, v in named]
        texts = labelled or ([self.rows.get(key, {}).get("title") or ""] if self.rows.get(key, {}).get("title") else [])
        if not texts:
            return "silent"

        def covers(text: str, need: list[str]) -> bool:
            tw = words(text)
            return all(any(h == w or (len(w) >= 4 and h.startswith(w)) or (len(h) >= 4 and w.startswith(h)) for h in tw) for w in need)

        # a value that repeats the product's own kind ("trench coat" as the style of a coat) is carried by its other words
        kind = self.kind_words
        core = [w for w in vw if not any(w == k or (len(w) >= 4 and k.startswith(w)) or (len(k) >= 4 and w.startswith(k)) for k in kind)]

        def holds(text: str) -> bool:
            return covers(text, vw) or (bool(core) and len(core) < len(vw) and covers(text, core))

        both = re.match(r"^(?P<pos>.+?)[,;]?\s+(?:but\s+|and\s+)?(?:not|never|nor)\s+(?P<neg>.+)$", flat(value))
        if both and [w for w in words(both.group("pos")) if len(w) > 1] and not re.match(r"^(?:not|no|without)\b", flat(value)):
            # a value with its own exclusions ("baggy, not slim or straight"): the first part must hold, no excluded one may
            pos = [w for w in words(both.group("pos")) if len(w) > 1]
            alternatives = [[w for w in words(part) if len(w) > 1] for part in re.split(r",|/|\bor\b|\bnor\b", both.group("neg"))]
            if any(alt and covers(t, alt) for alt in alternatives for t in texts):
                return "no"
            if any(covers(t, pos) for t in texts):
                return "yes"
            return "no" if labelled else "silent"

        negative = re.match(r"^(?:not|no|without|anything but|except|excluding|avoid)\s+(.+)$", flat(value))
        if negative:
            inner = [w for w in words(negative.group(1)) if len(w) > 1]
            hit = any(all(any(h == w or (len(w) >= 4 and h.startswith(w)) for h in words(t)) for w in inner) for t in texts)
            return "no" if hit else "yes"
        if any(holds(t) for t in labelled):
            # any label that names the property and states the value carries it ("skirt style: wrap skirts" beside "style:
            # casual", "clock_shape: rectangle, square" beside "shape: rectangle"): stores keep several taxonomies side by side
            return "yes"
        if labelled and self.claims_clash(named, holds):
            return "no"
        direct = [v for k, v in named if self.is_direct(k, pw)]
        if direct and not any(holds(t) for t in direct):
            return "no"                      # the label that is the property itself speaks for it; a longer one cannot overrule it
        if any(holds(t) for t in texts):
            return "yes"
        return "no" if labelled else "silent"

    def named(self, key: str, pw: list[str]) -> list[tuple[str, str]]:
        """The labels of a listing's sheet that name one property: those whose words cover the property's words, and,
        when the property is the kind of the listing itself ("type of top" on a top), the bare generic labels ("type",
        "kind") too, since such a label states what kind of item the listing is."""
        sheet = self.sheets.get(key) or {}
        out = [(k, str(v)) for k, v in sheet.items() if all(any(h == w or (len(w) >= 4 and h.startswith(w)) for h in words(k)) for w in pw)]
        kind = set(words((self.rows.get(key) or {}).get("group") or "")) | self.kind_words
        if pw and all(any(g == w or (len(w) >= 3 and g.startswith(w)) or (len(g) >= 3 and w.startswith(g)) for g in kind) for w in pw):
            out += [(k, str(v)) for k, v in sheet.items() if words(k) and all(h in ("type", "kind") for h in words(k)) and (k, str(v)) not in out]
        return out

    @staticmethod
    def is_direct(label: str, pw: list[str]) -> bool:
        """Whether a label is the property itself ("type", "top type" for the type of a top) rather than a longer one
        ("muslim top style")."""
        return not [h for h in words(label) if h not in ("type", "kind", "the", "of") and not any(h == w or (len(w) >= 4 and h.startswith(w)) for w in pw)]

    def direct_states(self, key: str, prop: str, value: str) -> bool:
        """Whether a label that is the property itself holds the value (the strongest statement a sheet can make)."""
        pw = [w for w in words(prop) if w not in ("type", "kind", "the", "of")]
        vw = [w for w in words(value) if len(w) > 1]
        if not pw or not vw or re.match(r"^(?:not|no|without|anything but|except|excluding|avoid)\s", flat(value)):
            return False
        direct = [v for k, v in self.named(key, pw) if self.is_direct(k, pw)]
        return bool(direct) and all(all(any(h == w or (len(w) >= 4 and h.startswith(w)) or (len(h) >= 4 and w.startswith(h)) for h in words(t)) for w in vw) for t in direct)

    def offered_by(self, key: str, who: str) -> bool:
        """Whether a listing's seller, shop, brand or maker field names `who` (any of them, by word)."""
        wanted = [w for w in words(who) if len(w) >= 2]
        if not wanted:
            return False
        names = (self.rows.get(key) or {}).get("who") or []
        if any(all(w in words(n) for w in wanted) for n in names):
            return True
        # a remark may run the seller's name into the product's ("the Kokarika Fruit Juice"): its first word, when it is a
        # name of its own (five letters or more) and the record's name is that one word, counts
        return len(wanted) > 1 and len(wanted[0]) >= 5 and any(words(n) == wanted[:1] for n in names)

    def label_value(self, key: str, prop: str) -> str:
        """The value a listing's own sheet gives for one property (the label whose words cover the property's words; the
        shortest such label first), or ''."""
        pw = [w for w in words(prop) if w not in ("type", "kind", "the", "of")]
        if not pw:
            return ""
        named = [(len(words(k)), k, v) for k, v in self.named(key, pw)]
        for _, _, v in sorted(named):
            text = str(v).strip()
            if text:
                return text.split(";")[0].strip()[:80]
        return ""

    def phrase_fits(self, key: str, phrase: str) -> bool:
        """Whether a listing's sheet states a detail given as one phrase ("water resistance 100m", "flavor latte"): some
        label and its value together cover every word of the phrase, or the value alone does when the phrase is short."""
        norm = lambda t: re.sub(r"(\d)\s+(?=[a-z]\b|[a-z]{1,3}\b)", r"\1", flat(t))
        want = set(words(norm(phrase)))
        if not want:
            return True
        sheet = self.sheets.get(key) or {}
        for k, v in sheet.items():
            have = set(words(k)) | set(words(norm(v)))
            if want <= have and set(words(norm(v))) & want:
                return True
        title = (self.rows.get(key) or {}).get("title") or ""
        return not sheet and want <= set(words(norm(title)))

    def page_text(self, key: str) -> str:
        """The free text a listing's page carries (remarks, blurbs), as one string."""
        return " ".join((self.rows.get(key) or {}).get("notes") or [])

    def evidence_lines(self, key: str, conditions: list[tuple[str, str]]) -> list[str]:
        """For each condition, what the listing's own sheet says: the labels naming the property with their values (keys that
        differ only in spacing, underscores or case were merged into one field) and the verdict of says()."""
        title = (self.rows.get(key) or {}).get("title") or ""
        out = []
        for prop, value in conditions:
            pw = [w for w in words(prop) if w not in ("type", "kind", "the", "of")]
            named = self.named(key, pw) if pw else []
            verdict = self.says(key, prop, value)
            word = {"yes": "states it", "no": "contradicts it"}.get(verdict, "silent")
            if verdict == "yes" and self.direct_states(key, prop, value):
                word = "states it under the property's own label"
            if named:
                out.append(f"{prop}: {value} -> " + "; ".join(f"'{k}': {v[:60]}" for k, v in named[:3]) + f" ({word})")
            else:
                out.append(f"{prop}: {value} -> no label for it" + (f"; title: {title[:50]}" if title else "") + f" ({word})")
        return out

    @staticmethod
    def claims_clash(named: list[tuple[str, str]], holds: Any) -> bool:
        """A namespaced label and the plain one ('clothing.material' beside 'material') are separate claims about the
        same property: when one holds the value and the other does not, the sheet contradicts itself. Labels for other
        properties ('heel height' beside 'heel type') are not read against each other."""
        cores: dict[str, dict[str, bool]] = {}
        for k, v in named:
            core = str(k).rsplit(".", 1)[-1].strip()
            cores.setdefault(core, {})[str(k)] = holds(str(v))
        return any(len(fields) > 1 and any(fields.values()) and not all(fields.values()) for fields in cores.values())

    def settles(self, key: str, prop: str, value: str) -> bool:
        """Whether the listing's own labels settle one condition beyond doubt: at least one label names the property,
        and every label naming it holds the value (or, for an exclusion, none holds the excluded value). Figures and
        bounds are left to the checker."""
        if re.search(r"\d", value) or self.says(key, prop, value) != "yes":
            return False
        pw = [w for w in words(prop) if w not in ("type", "kind", "the", "of")]
        named = self.named(key, pw)
        # a label that is the property itself ("heel type" for heel) speaks for it; a longer one ("heel height") only
        # when no such label exists
        direct = [v for k, v in named if self.is_direct(k, pw)]
        labelled = direct or [v for _, v in named]
        if not labelled:
            return False
        negative = re.match(r"^(?:not|no|without|anything but|except|excluding|avoid)\s+(.+)$", flat(value))
        inner = [w for w in words(negative.group(1) if negative else value) if len(w) > 1]

        def holds(text: str) -> bool:
            tw = words(text)
            return all(any(h == w or (len(w) >= 4 and h.startswith(w)) or (len(h) >= 4 and w.startswith(h)) for h in tw) for w in inner)

        if negative:
            return not any(holds(t) for t in labelled)
        return all(holds(t) for t in labelled)

    def fits(self, conditions: list[tuple[str, str]], cap: float | None, quantity: int, group: str = "", skip: set[str] | None = None) -> list[str]:
        """Opened listings whose sheet states every condition, available, within the limit, in the group; cheapest first."""
        if not conditions:
            return []
        out = []
        for key in self.sheets:
            row = self.rows.get(key) or {}
            if skip and key in skip or row.get("stock") is False or row.get("price") is None:
                continue
            if cap is not None and row["price"] * quantity > cap + 1e-9:
                continue
            if group and row.get("group") and flat(row["group"]) != flat(group):
                continue
            if all(self.says(key, p, v) == "yes" for p, v in conditions):
                out.append((row["price"], not all(self.direct_states(key, p, v) for p, v in conditions), key))
        return [k for _, _, k in sorted(out)]

    def unopened_cheaper(self, pick: str, cap: float | None, quantity: int) -> list[tuple[str, float]]:
        """Rows priced below the pick, in its group, not shown out of stock, within the limit, never opened."""
        held = self.rows.get(pick)
        if not held or held.get("price") is None:
            return []
        found = []
        for key, row in self.rows.items():
            if key == pick or row.get("sheet") or row.get("price") is None or row["price"] >= held["price"] or row.get("stock") is False:
                continue
            if cap is not None and row["price"] * quantity > cap + 1e-9:
                continue
            if held.get("group") and row.get("group") and flat(row["group"]) != flat(held["group"]):
                continue
            found.append((row["price"], key))
        return [(k, p) for p, k in sorted(found)[:5]]

    def menu(self) -> tuple[str, int]:
        """The sheet properties seen so far: how many sheets carry each, how many different values, samples. Properties
        whose values differ between the sheets come first, since only those can change the choice."""
        count: dict[str, int] = {}
        values: dict[str, list[str]] = {}
        for sheet in self.sheets.values():
            for label, value in sheet.items():
                text = flat(value)[:40]
                if not text or re.search(META_KEYS, label) or (re.fullmatch(r"[a-z0-9_\-]*\d[a-z0-9_\-]*", text) and ("-" in text or "_" in text)):
                    continue
                count[label] = count.get(label, 0) + 1
                bucket = values.setdefault(label, [])
                if text not in bucket:
                    bucket.append(text)
        ranked = sorted(count, key=lambda k: (len(values[k]) < 2, -count[k], k))[:30]
        lines = [f"- {k} (on {count[k]} sheets, {len(values[k])} different values): " + " | ".join(values[k][:5]) for k in ranked]
        return "\n".join(lines), len(self.sheets)

    def splits(self, conditions: list[tuple[str, str]], request: str, settled: list[str]) -> tuple[str, int]:
        """The open questions the opened sheets pose: among opened listings of the most-read category that no condition in
        force rules out, the properties whose values differ, that no condition or settled property covers and whose
        values the request does not already name; ranked by how many of those listings the answer could set aside."""
        tally: dict[str, int] = {}
        for key in self.sheets:
            label = flat((self.rows.get(key) or {}).get("group") or "")
            tally[label] = tally.get(label, 0) + 1
        home = max(tally, key=lambda g: (tally[g], g != "")) if tally else ""
        pool = [k for k in self.sheets if flat((self.rows.get(k) or {}).get("group") or "") == home
                and (self.rows.get(k) or {}).get("stock") is not False
                and all(self.says(k, p, v) != "no" for p, v in conditions)]
        if len(pool) < 2:
            return "", len(pool)
        taken = [set(words(p)) - {"type", "kind", "the", "of"} for p, _ in conditions] + [set(words(t)) for t in settled]
        asked_words = set(words(request))
        options: list[tuple[int, int, str, str]] = []
        for label in {l for k in pool for l in self.sheets[k]}:
            if re.search(META_KEYS, label):
                continue
            lw = set(words(label))
            if any(t and (t <= lw or lw <= t) for t in taken):
                continue
            seen: dict[str, int] = {}
            for k in pool:
                text = flat(self.sheets[k].get(label))[:40]
                if text:
                    seen[text] = seen.get(text, 0) + 1
            if len(seen) < 2 or sum(len(re.split(r"[,;|/]", v)) >= 3 for v in seen) * 2 > len(seen):
                continue
            if any(set(words(v)) and set(words(v)) <= asked_words for v in seen if len(v) > 2):
                continue
            carriers = sum(seen.values())
            options.append((carriers - max(seen.values()), carriers, label, " | ".join(f"{v} x{n}" for v, n in sorted(seen.items(), key=lambda vn: -vn[1])[:5])))
        options.sort(key=lambda o: (-o[0], -o[1], o[2]))
        lines = [f"- {label}: an answer could set aside {cut} of the {carriers} candidates carrying it ({values})" for cut, carriers, label, values in options[:12]]
        return "\n".join(lines), len(pool)

    def group_counts(self) -> str:
        counts: dict[str, int] = {}
        for row in self.rows.values():
            if row.get("group"):
                counts[row["group"]] = counts.get(row["group"], 0) + 1
        return ", ".join(f"{g} (x{n})" for g, n in sorted(counts.items(), key=lambda kv: -kv[1])[:8])

    def table(self, cap: float | None, limit: int = 20) -> str:
        rows = sorted(((k, r) for k, r in self.rows.items() if r.get("price") is not None), key=lambda kr: (kr[1]["price"], kr[0]))
        if not rows:
            return ""
        lines = []
        for key, row in rows[:limit]:
            stock = {True: "in stock", False: "OUT OF STOCK"}.get(row["stock"], "stock ?")
            over = " over limit" if cap is not None and row["price"] > cap else ""
            lines.append(f"{key} | {row['price']:g}{over} | {stock} | {'sheet seen' if row['sheet'] else 'no sheet yet'} | {row['group']} | {row['title']}")
        more = f" ({len(rows) - limit} dearer not listed)" if len(rows) > limit else ""
        return f"LISTINGS SEEN SO FAR, cheapest first{more} (id | price | stock as last reported | property sheet | category | title):\n" + "\n".join(lines)


# ----------------------------------------------------------------------------- the clerk: one episode

FRESH_RESULTS = 6
NEUTRAL_DECLINE = "Sorry, nothing in the store meets every condition, so I am not placing an order."


class Episode:
    def __init__(self, problem_data: dict[str, Any]):
        self.started = time.monotonic()
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.tools = [t for t in self.policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}} for t in self.tools}
        self.schemas_full = {t["function"]["name"]: t["function"] for t in self.tools}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.cost_soft, self.cost_hard, self.cost_stop = spend_lines(self.max_steps)
        self.turn_cap = max(MIN_TURNS, min(MAX_TURNS, self.max_steps + TURN_SLACK))
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.llm_down = False                # the key is spent (402) or the model kept failing: no more model calls this episode
        self.llm_fails = 0                   # consecutive failed model calls
        self.model_calls = 0                 # model calls sent this episode (failed ones included)
        self.tool_turns = 0                  # model tool turns this episode (capped at the step cap plus TOOL_TURN_SLACK)
        self.heavy_late = 0                  # heavy-model calls made past the soft line
        self.messages_used = 0
        self.message_budget: int | None = None
        self.order_attempts = 0
        self.refusals = 0
        self.search_usage: dict[str, list[Any]] = {}  # the values each search argument carried so far
        self.decline_reads: set[str] = set()
        self.decline_paged = False
        self.cheaper_named = 0               # orders of a named listing held because a verified cheaper fit exists
        self.sheet_tools: set[str] = set()   # reading tools seen returning a property sheet
        self.bare_tools: set[str] = set()    # reading tools seen returning a listing without one
        self.multi_held: set[str] = set()    # cart states whose order was held for holding several lines of one product
        self.listing_step = -1               # the step of the last basket listing (its lines are not news)   # rows the code opened itself before letting a decline go out
        self.refusal_step = -1               # the step whose finishing call was turned down
        self.order_checked = False
        self.decline_checks = 0
        self.opened: set[str] = set()
        self.broad_spent = 0                  # sheets the code opened itself after searches
        self.compare_sheets: bool | None = None  # whether the side-by-side tool delivers property sheets (None: not tried)
        self.broad_raw: dict[str, tuple[str, int, Any]] = {}  # listing -> (tool, step, record) of such a read the model has not opened itself
        self.cheaper_checks = 0
        self.early_bounced = False
        self.refused_at = -1        # step of the last refused order
        self.searched_at = -1       # step of the last search
        self.research_checks = 0
        self.cart_used = False
        self.seen_ids: set[str] = set()
        self.stated: list[str] = []
        self.revealed: list[str] = []
        self.free: list[str] = []
        self.asked_props: list[str] = []
        self.dropped: list[str] = []
        self.budget_now: Any = None
        self.heard: set[str] = set()
        self.cart_ids: list[str] = []
        self.gone: set[str] = set()          # listings a refusal or a shopper line reported sold out
        self.gone_checks = 0
        self.shelf_held: set[str] = set()
        self.records: list[Any] = []
        self.call_counter = 0
        self.shelf = Shelf()                 # rows, sheets and records of every listing seen
        self.last_cart_view: Any = None
        self.cart_changed_at = 0
        self.cart_view_at = -1
        self.cart_state: dict[str, str] = {}
        self.add_signature: str | None = None
        self.view_mode: tuple[str, str] | None = None   # (mode argument, value) of a cart tool that also lists the basket
        self.asked: list[str] = []
        self.questions_sent: list[str] = []
        self.planner_on = True
        self.sets_at_last_ask = -1
        self.last_unclear = False
        self.plan_misses = 0
        self.capped: set[str] = set()
        self.cart_lines: dict[str, str] = {}
        self.plan_retry_at = 0
        self.audited = False
        self.audit_seen: tuple[str, str] = ("", "")
        self.qty_warned: set[str] = set()
        self.pending_reply = ""
        self.current: list[str] | None = None
        self.verdicts: dict[str, str] = {}
        self.ledger_version = 0
        self.cart_checked: set[str] = set()
        self.said: list[str] = []
        self.recent: list[str] = []
        self.roles: dict[str, str] = {}
        self.plan: dict[str, Any] = {}
        self.cost = 0.0
        self.duties: list[dict[str, Any]] = []      # steps the request or the shopper asked for besides the purchase
        self.duty_sending = False
        self.swap_asked: set[str] = set()
        self.review_reason = ""
        self.reviews = 0
        self.bundle_checked: set[str] = set()
        self.trace: list[dict[str, Any]] = []
        self.walked: set[str] = set()          # pick prices the price-limit walk has already started from
        self.walk_held: set[str] = set()       # cart targets held once because the walk found a cheaper fit
        self.walk_log: list[str] = []
        self.walking = False
        self.walk_used = 0                     # store steps the walk has spent this episode (hard cap WALK_STEPS)
        self.passed: set[str] = set()          # listings an item check judged to fit
        self.passed_at: dict[str, int] = {}    # listing -> the ledger version its item check passed under
        self.pace_held = False                 # the one step-free hold of exploration once the order is overdue
        self.empty_held = False                # the one step-free hold of an order sent with nothing in the cart
        self.checked_once = False              # the one look over conditions and steps right before the first finish
        self.check_held = False                # the one step-free hold that look may put on an order
        self.dearer_ok: set[str] = set()       # dearer picks after a stockout the model confirmed by sending again
        self.searches: list[dict[str, Any]] = []   # the model's own searches: arguments, reported total, rows shown
        self.text_tries: dict[str, dict[str, str]] = {}   # search mode/category/limit -> free text -> result signature
        self.text_blind: set[str] = set()      # search modes whose free text does not change the result
        self.blind_refused: set[str] = set()   # mode|text refused once because the mode ignores the words
        self.text_totals: dict[str, dict[str, int]] = {}   # search mode|category -> free text -> reported total
        self.add_args: dict[str, Any] | None = None   # the first add to the cart, the shape the code's own cart calls copy
        self.keep_held: set[str] = set()       # cart lines whose removal was held once (cheapest requests)
        self.swap_done = False                 # the one cart swap the code makes to the cheapest fitting listing
        self.swap_note = ""                    # its note, handed to the model with the order's result
        self.vetoed: set[str] = set()          # candidates the model ruled out by naming a condition their sheet contradicts
        self.early_paged = False               # the one early paging of a short keyword search on a cheapest request
        self.model_removed: set[str] = set()   # cart lines the model itself took out (the code never commits to one of these)
        self.agent_placed: set[str] = set()    # cart lines the code put in (the commit, a recovery, the cheapest swap)
        self.placed_holds: dict[str, int] = {} # how often taking each such line out was held
        self.code_move = False                 # True while a cart call of the code's own is booked
        self.order_holds: dict[str, int] = {}  # cart state -> advisory holds already put on an order in that state
        self.refill_used = 0                   # store steps spent after stockouts looking for a fit at the cheapest fitting price
        self.refilled_at = -2                  # the announced change the code already looked for replacement candidates after
        self.gone_held: set[str] = set()       # sold-out listings whose re-add the code refused once
        self.last_content = ""                 # the model's words in its latest turn
        self.market_at = -1                    # step of the last change the store or the shopper announced (refusal, sold-out, price, budget)
        self.committed = ""                    # the sheet-fit the code itself put in the cart at the commit mark
        self.recovered_at = -2                 # the announced change the code already recovered from by swapping the cart
        self.recoveries: dict[int, int] = {}   # announced change (its step) -> how often the code recovered from it
        self.looked: set[str] = set()          # cart states (and finish kinds) the look before the finish already covered
        self.checked_version = -1              # the requirements version the item check at the order last ran under
        self.ask_held = False                  # the one order hold that sends the planned questions first
        self.pre_read = False                  # the code already opened extra sheets before the first add
        self.freshened: set[str] = set()       # picks whose stock and price the code re-read itself right before an order
        self.order_targets: list[str] = []     # the listings the latest order call named itself (a store without a cart)
        self.order_args: dict[str, Any] = {}   # the arguments of the order now being held or sent
        self.pointers: list[dict[str, Any]] = []   # needs that point at another listing's page, with how far the code got
        self.pointer_steps = 0                 # store steps spent following them
        self.zero_searches = 0                 # the model's free-text searches that returned nothing
        self.zero_words: list[set[str]] = []   # the words of those searches
        self.zero_holds = 0                    # wordy searches held after the fallback
        self.zero_fallback = False             # the one category-and-limit search the code runs after too many empty searches
        self.budget_changed_at = -1            # step at which a shopper line last changed the limit (read by the code)
        self.repaired_at = -1                  # the refused finish the code already repaired after
        self.decline_blocks = 0                # declines held because an opened sheet still fits
        self.page_calls = 0                    # store calls that open a listing's page or compare listings (any reader)
        self.search_calls = 0                  # store searches (the model's and the code's)
        self.unseen_holds = 0                  # declines held because the products' candidates were never opened
        self.heard_budget_lines: set[str] = set()   # shopper lines already read for a new limit
        self.unfolded = 0                      # answers to questions the hearing turned into nothing (the code's picture may miss a need)
        self.refusal_text = ""                 # the words of the last refused finish
        # --- the code-owned interview, the bundle tracker, the market notes and the read cache ---
        self.interview_log: list[dict[str, Any]] = []   # every question the code sent: product, property, outcome
        self.confused: list[str] = []          # properties whose question earned a confused reply: counted as "no preference", never re-asked
        self.needs_of: dict[str, int] = {}     # product -> firm needs revealed for it so far
        self.asks_of: dict[str, int] = {}      # product -> messages the code put about it
        self.group_size = KEYS_PER_MESSAGE     # properties the next code question names at most (the width; it only narrows)
        self.plain_next: tuple[str, list[str]] | None = None   # (product, properties) to put once more in plainer words after a confused reply
        self.crowded = False                   # a crowding complaint was heard: one property a message for the rest of the episode
        self.last_crowded = False              # the latest reply to a question objected to its width
        self.last_lines: list[str] = []        # the shopper's own lines in reply to the latest question
        self.redo_queue: list[tuple[str, str]] = []   # (product, property) of a complained message, put again one at a time, first first
        self.crowd_redos = 0                   # questions put again after crowding complaints
        self.crowd_tried: dict[str, int] = {}  # product|property -> times put again after a complaint (twice settles it)
        self.unclear_lines: set[str] = set()   # shopper lines the hearing read as not following the question (a repeat says the same)
        self.ending_freed = False              # the message kept for a decline duty was given to one more question before an order
        self.late_asks = 0                     # questions put with that message
        self.width_log: list[int] = []         # properties each code question named, in order (the trace shows the adaptation)
        self.free_of: dict[str, list[str]] = {}   # product -> properties waved off in reply to a code question about it
        self.fit_mark: Any = None              # the shelf state empty_cart_fit last judged, and its answer
        self.fit_known = False
        self.plain_tries: dict[str, int] = {}  # product|property -> plainer re-asks already sent
        self.asks_at_need: dict[str, int] = {} # product -> questions it had got when its first firm need was heard
        self.group_form = ""                   # "" unknown, "any" (a reply answered a property listed after the first), "first"
        self.parked: list[str] = []            # properties of a message the shopper did not follow while the form was unknown
        self.last_group: list[str] = []        # the properties the latest code question named, in order
        self.allowance_read = False            # the store's no-argument allowance report was read once for the message cap
        self.audit_mark = -1                   # the state of the conditions at which the code last judged the candidates before a decline
        self.lines_for: dict[str, str] = {}    # product -> the cart line held for it (several products)
        self.bundle_done = False               # the one code-made completion of a cart for several products
        self.budget_rose_at = -1               # step of the last rise of the limit the note reader saw
        self.reranked_at = -1                  # the rise the code already re-ranked the cart after
        self.relaxed_at = -1                   # the step a note or reply last dropped a requirement
        self.relax_done = -1                   # the relaxation the code already re-ranked after
        self.probe_holds = 0                   # declines turned into a hold of the closest candidate
        self.probe_line = ""                   # the line the code held before a decline
        self.probe_at = -1
        self.add_unread = False                # a line went into the cart and no other call has gone to the store since
        self.budget_before_rise: float | None = None   # the limit before the latest rise
        self.probe_order_holds = 0
        self.peeks = 0                         # cart views the code made after an add, before the order
        self.market_notes: list[str] = []      # what the note reader found in observations (budget, gone, price moved)
        self.cache_hits = 0                    # re-opens of an unchanged sheet answered from the shelf without a step
        self.opened_by: dict[str, set[str]] = {}   # listing -> the reading tools that opened it (the read cache serves the same tool only)
        self.story_cache: dict[str, list[str]] = {}   # product|sheets|questions -> the shortlist in story order (one model call per state)
        self.surveyed: set[str] = set()      # products whose sheets the code widened before their first question
        self.probe_handed = False            # the forced finish handed a probe that drew a change back to the model once
        self.rescued: set[str] = set()       # cart states the code ordered instead of a decline (once per state)
        self.look_vetoed: dict[str, list[str]] = {}   # lines the last look read as stating another value: the properties it named
        self.hold_log: list[str] = []          # the holds the code put on the model's calls (the digest shows the latest)
        self.repeat_held: set[str] = set()     # model questions held once because their property was already put to the shopper
        self.dead_unknown = -1                 # step of a stockout report that named no line while several lines were held
        self.bundle_recovered_at = -1          # the stockout the code already recovered the bundle from
        self.price_moved_at = -1               # step at which a price was reported moved (the lines are re-read before the order)
        self.repriced_at = -1                  # the price move the code already re-read the lines after
        self.bundle_swapped: set[str] = set()  # cart states the per-product cheapest swap already ran on
        # --- the note reader, fitment codes and decline pacing ---
        self.note_reads: dict[str, dict[str, Any]] = {}   # free-text line -> what the cheap note read found in it
        self.note_calls = 0
        self.note_calls_late = 0               # note reads made past the soft line
        self.note_gone: set[str] = set()       # lines read as a stockout
        self.note_price: set[str] = set()      # lines read as a price move
        self.fit_searched = 0                  # searches the fitment round already took candidates from
        self.fit_spent = 0                     # pages the code opened for the codes
        self.fit_asked: set[str] = set()       # listings whose page the code already asked for
        self.fit_held: set[str] = set()        # adds and carts held once for a line whose page lists other codes
        self.budget_holds = 0                  # declines held right after a new limit
        self.long_holds = 0                    # declines held early in a long episode
        self.stock_refilled: set[int] = set()  # announced changes after which the recovery opened unread rows itself
        self.messages: list[dict[str, Any]] = []
        self.paged: dict[tuple[Any, str], dict[str, Any]] = {}   # result lists seen, by call shape (for the code's own paging)
        self.bundle_paged = 0
        self.code_reranked = False
        self.handed_back = False               # a refusal of the code's own finish was handed back to the model once
        self.churn: dict[str, int] = {}        # removals of a held line per (line, market state, ledger version)
        self.code_bundle_tries: dict[str, int] = {}   # rounds of the code's own search per product of several
        self.empty_fault = -1                  # step of an order the store refused for an empty basket
        self.empty_repaired = -1
        self.main_busy = False                 # guards main_pick against re-entry through the fitment checks
        self.basket_counts: dict[str, int] = {}  # units per line of a whole-basket cart, as last set
        # --- champ21: the store's own text for the model, the code's reading as a note, and how far that reading is trusted ---
        self.raw_last: list[Any] = []          # the observations of the latest envelope as the store wrote them
        self.raw_seen = ""                     # the store's own text of every result so far (bounded), for id checks
        self.parse_good = 0                    # results whose listings the code's reader found (about as many as the text shows)
        self.parse_low = 0                     # results whose text shows more listings than the reader found
        self.raw_misses = 0                    # ids the model used that the store wrote but the code's reader never found
        self.compact_upto = 0                  # transcript entries before this index are shown cut (only past HISTORY_CHARS)
        self.cut_cache: dict[int, str] = {}    # transcript entry -> its cut form, made once
        self.audit_pass = False                # the one audit round asked of the model before its first decline

    def time_left(self) -> float:
        return TASK_SECONDS - (time.monotonic() - self.started)

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def long_world(self) -> bool:
        """A step cap of LONG_WORLD or more: the episode can afford a second round of care after a change."""
        return self.max_steps >= LONG_WORLD

    def mark_change(self) -> None:
        """Remember the step of a change the store or the shopper announced: a record of a listing seen before it may be stale."""
        self.market_at = self.step

    def past_soft(self) -> bool:
        return self.cost > self.cost_soft

    def infer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        """One model answer, trying the models in order. Never raises and never waits on a dead key: a 402 (or a key the
        proxy refuses) ends every model call of the episode at once, and LLM_FAIL_LIMIT failures in a row (429, 5xx, network,
        an empty completion) do too; the episode then finishes in code."""
        for model in models:
            if self.llm_down:
                return None
            if model in (SETUP_MODEL, JUDGE_MODEL) and model != MAIN_MODEL and self.past_soft():
                if self.heavy_late >= HEAVY_AFTER_SOFT:
                    continue                 # past the soft line the heavy model is called at most HEAVY_AFTER_SOFT more times
                self.heavy_late += 1
            left = self.time_left()
            if left < FORCE_SECONDS - 10:
                return None
            _llm.timeout = int(max(8, min(75, left - FORCE_SECONDS + 15)))
            sent = messages
            if model.startswith("anthropic/") and len(messages) >= 2:
                # the playbook, the opening and every transcript entry before the fresh status are byte-stable from turn to turn:
                # breakpoints after the opening and after the last entry let the next turn read them from the cache
                sent = [dict(m) for m in messages]
                for index in sorted({0, 1} | ({len(sent) - 2} if len(sent) >= 4 and tools else set())):
                    if isinstance(sent[index].get("content"), str) and sent[index]["content"]:
                        sent[index]["content"] = [{"type": "text", "text": sent[index]["content"], "cache_control": {"type": "ephemeral"}}]
            body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": max_tokens}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            status: Any = None
            error: Any = None
            for attempt in range(len(RETRY_WAITS) + 1):
                if attempt:
                    # a passing fault (429, 5xx, no answer) is tried again after a short jittered wait, never into the
                    # time kept for the finish
                    wait = self.backoff(attempt)
                    if self.time_left() - wait < FORCE_SECONDS + 25:
                        break
                    time.sleep(wait)
                    _llm.timeout = int(max(8, min(75, self.time_left() - FORCE_SECONDS + 15)))
                self.model_calls += 1
                try:
                    result = _llm.post_verbose("/inference/chat/completions", body)
                    data, error = result.data, result.error
                except Exception as exc:          # a transport fault is a failed call, never a crash
                    data, error = None, {"kind": "exception", "status": None, "body": f"{type(exc).__name__}: {exc}"}
                if isinstance(data, dict) and data.get("choices"):
                    self.charge(data.get("usage"))
                    message = (data["choices"][0] or {}).get("message") or {}
                    if message.get("content") or message.get("tool_calls"):
                        self.llm_fails = 0
                        return message
                    status = "empty"
                    break                         # an empty completion goes to the next model, not again to this one
                status = error.get("status") if isinstance(error, dict) else None
                self.trace.append({"kind": "inference_failure", "model": model, "status": status, "attempt": attempt, "error": str(error)[:300] if error else "empty"})
                if status not in RETRY_STATUSES:
                    break
            if status in (401, 402):
                self.llm_down = True             # the key is spent or refused: every later call would fail the same way
                self.trace.append({"kind": "llm_down", "status": status, "calls": self.model_calls})
                return None
            self.llm_fails += 1
            if self.llm_fails >= LLM_FAIL_LIMIT:
                self.llm_down = True
                self.trace.append({"kind": "llm_down", "status": status, "calls": self.model_calls})
                return None
        return None

    def charge(self, usage: Any) -> None:
        """Adds one answer's price to the episode's spend: the proxy's own cost figure, or (a provider that reports none)
        an estimate from the token counts at a deliberately high rate."""
        usage = usage if isinstance(usage, dict) else {}
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            self.cost += float(cost) * BILL_SCALE
            return
        tokens_in = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), (int, float)) else 3000
        tokens_out = usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), (int, float)) else 300
        self.cost += (float(tokens_in) * 2.0 + float(tokens_out) * 10.0) / 1e6

    def env_call(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        """One envelope (one action, or a group run as consecutive steps). A reply that never comes is sent again with the
        same step (the store replays it); a drifted counter resumes where the store expects it; a spent key ends the run."""
        envelope: dict[str, Any] = {**self.binding, "step": self.step + 1}
        envelope["action" if len(actions) == 1 else "actions"] = actions[0] if len(actions) == 1 else actions
        passing = 0
        for _ in range(6):
            room = self.time_left()
            if room < 2:
                return None
            _env.timeout = int(max(3, min(95, room - 1)))
            outcome = _env.post_verbose("/environment/call", envelope)
            answer = outcome.data
            if isinstance(answer, dict):
                kept = answer.get("observations") if isinstance(answer.get("observations"), list) else [answer.get("observation")]
                self.raw_last = [self.copied(o) for o in kept]
                # the shopper's own words stay as written; every other result has its text-written records read first
                talk = [a.get("name") == self.roles.get("message") for a in actions]
                answer = {k: ([unpack_reply(o if i < len(talk) and talk[i] else read_reply(o)) for i, o in enumerate(v)] if k == "observations" and isinstance(v, list)
                              else unpack_reply(v if talk and talk[0] and len(talk) == 1 else read_reply(v)) if k == "observation" else v) for k, v in answer.items()}
            if isinstance(answer, dict) and ("observation" in answer or "observations" in answer):
                self.step = answer["step"] if isinstance(answer.get("step"), int) else self.step + len(actions)
                self.finished = self.finished or bool(answer.get("finished"))
                seen = answer.get("observations") if isinstance(answer.get("observations"), list) else [answer.get("observation")]
                for action, observation in zip(actions, seen):
                    self.shelf.absorb(observation, self.step, sort_tool(self.schemas_full.get(action.get("name"), {}))[0])
                    self.note_list(action, observation)
                    if (action.get("name") == self.roles.get("order") and not self.finished and isinstance(observation, dict)
                            and _EMPTY_FAULT.search(_compact(observation))):
                        # the store found the basket empty: whatever the code believed was held is not there
                        self.cart_ids = []
                        self.empty_fault = self.step
                if self.shelf.factor_doubt is not None:
                    self.check_scale()
                return answer
            trouble = outcome.error if isinstance(outcome.error, dict) else {}
            text = str(trouble.get("body") or "").lower()
            self.trace.append({"kind": "env_failure", "status": trouble.get("status"), "body": text[:300]})
            if trouble.get("status") == 402 or (trouble.get("status") == 409 and "finished" in text):
                self.dead = trouble.get("status") == 402
                self.finished = self.finished or trouble.get("status") == 409
                return None
            drift = re.search(r"expected (\d+)", text) if trouble.get("status") == 409 else None
            if drift:
                self.step = int(drift.group(1)) - 1
                envelope["step"] = self.step + 1
                continue
            if trouble.get("status") in RETRY_STATUSES and trouble.get("status") != 409 and passing < len(RETRY_WAITS):
                # a passing fault (429, 5xx, no answer): the same envelope again after a short jittered wait while time allows
                wait = self.backoff(passing + 1)
                passing += 1
                if self.time_left() - wait < 4:
                    return None
                time.sleep(wait)
                continue
            return None
        return None

    @staticmethod
    def copied(value: Any) -> Any:
        """A deep copy of a store result (the bookkeeping below rewrites its own copy in place)."""
        try:
            return json.loads(json.dumps(value))
        except (TypeError, ValueError):
            return value

    def backoff(self, attempt: int) -> float:
        """The wait before the next try of a call that met a passing fault: RETRY_WAITS plus a jitter drawn from the call
        count (never from the clock), so concurrent episodes spread their retries."""
        base = RETRY_WAITS[min(max(1, attempt), len(RETRY_WAITS)) - 1]
        jitter = ((self.model_calls * 37 + self.step * 13 + attempt * 7) % 10) / 10.0
        return base + 0.6 * jitter

    def check_scale(self) -> None:
        """Two price scales both put the store's written amounts near the request's limit: one cheap model call reads one
        amount in the request's own unit, once per world, and the shelf moves to the scale it names."""
        limit, sample, other = self.shelf.factor_doubt or (0.0, "", 1.0)
        self.shelf.factor_doubt = None
        figure = as_price(sample)
        if not figure or self.past_soft():
            return
        query = str(self.policy.get("query") or "")[:600]
        got = self.judge("A shopper wrote this request to a store:\n" + query + "\n\nThe request's spending limit was read as "
                         f"{limit:g}. The store writes one listing's price as {sample!r}. Write that listing price as a number in the "
                         "same unit the shopper's limit uses (for example cents or dollars), so the two can be compared. Reply "
                         'with JSON only: {"price_in_request_unit": <number>}', 60, [READ_MODEL, MAIN_MODEL])
        value = (got or {}).get("price_in_request_unit")
        if isinstance(value, str):
            value = as_price(value)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            return
        if abs(value - figure * other) <= 0.02 * figure * other and abs(value - figure * self.shelf.factor) > 0.02 * figure * self.shelf.factor:
            change = other / self.shelf.factor
            self.shelf.factor = other
            for row in self.shelf.rows.values():
                if isinstance(row.get("price"), (int, float)):
                    row["price"] = round(row["price"] * change, 6)
            self.trace.append({"kind": "price_scale", "factor": other, "sample": sample})

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
        # the full model writes the brief: cheaper readings drop or reword stated conditions, and every later step
        # searches and checks against this brief; the cheaper model only steps in when that reading does not hold up
        message = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, BACKUP_MODEL], 2000)
        plan = _parse_json_object((message or {}).get("content") or "") or {}
        if not self.brief_holds(plan):
            message = self.infer([{"role": "user", "content": content}], None, [MAIN_MODEL], 2200)
            plan = _parse_json_object((message or {}).get("content") or "") or plan
        if not isinstance(plan.get("roles"), dict) and self.time_left() > 200:
            # without the reading every later step works blind: try once more
            message = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, MAIN_MODEL], 2000)
            plan = _parse_json_object((message or {}).get("content") or "") or plan
        # a reading that drifted in its key names still counts: the product, priority and limit under a near name
        if isinstance(plan, dict):
            if not plan.get("product") and isinstance(plan.get("item"), str):
                plan["product"] = plan["item"]
            if plan.get("priority") not in ("cheapest", "best") and plan.get("preference") in ("cheapest", "best"):
                plan["priority"] = plan["preference"]
            if not plan.get("hint_properties"):
                near = [h.get("property") if isinstance(h, dict) else h for h in (plan.get("hints") or plan.get("hinted_attributes") or [])]
                plan["hint_properties"] = [h for h in near if isinstance(h, str) and h.strip()][:4]
        self.plan = plan
        self.shelf.kind_words = {w for w in words(plan.get("product") or "") if len(w) >= 3}
        roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        if not self.roles:
            # the reading failed: without roles every guard that depends on them is off, so guess from the tool names
            self.roles = self.guess_roles()
            self.trace.append({"kind": "guessed_roles", "roles": self.roles})
        guessed = self.guess_roles()
        taken = set(self.roles.values())
        declined = {k for k, v in roles.items() if v is None}      # roles the reading said no tool fills
        for role, tool in guessed.items():
            # a role the reading left out (or named a tool this store does not have) is filled from the schemas
            shared_end = role in ("order", "decline") and guessed.get("order") == guessed.get("decline") == tool and tool in (self.roles.get("order"), self.roles.get("decline"))
            if role not in self.roles and role not in declined and (tool not in taken or shared_end):
                self.roles[role] = tool
                taken.add(tool)
        self.settle_cart_role()
        self.split_cart_role()
        self.find_view_mode()
        if self.roles.get("cart_view") and self.roles.get("cart_view") == self.roles.get("cart"):
            del self.roles["cart_view"]          # one tool both moves lines and lists them: the listing goes by its mode
        self.split_finish_roles()
        self.trace.append({"kind": "roles", "roles": dict(self.roles), "view_mode": self.view_mode})
        self.shelf.learn_keys(self.id_arguments())
        budget = plan.get("message_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget >= 0:
            self.message_budget = int(budget)
        self.stated = [str(r) for r in plan.get("requirements") or [] if r]
        self.take_pointers(self.stated)
        self.budget_now = plan.get("price_cap")
        query = str(self.policy.get("query") or "")
        if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) and self.budget_now > 0:
            self.shelf.scale = float(self.budget_now)
        if self.budget_now is None:
            # the brief named no limit: a figure beside a spending word in the request itself is read by the code
            self.budget_now = budget_in(query, str(plan.get("currency") or ""))
        if not plan.get("currency") and self.budget_now is not None:
            plan["currency"] = unit_of(query, self.budget_now)   # the price unit the request writes beside its limit
        if plan and self.budget_now is None and self.roles.get("message"):
            ask = [str(a) for a in plan.get("ask_first") or [] if a]
            plan["ask_first"] = ["the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"] + ask
        steps = plan.get("step_budget")
        if isinstance(steps, (int, float)) and not isinstance(steps, bool) and 0 < steps < self.max_steps:
            self.max_steps = int(steps)
        for duty in plan.get("duties") or []:
            self.add_duty(duty)
        for kind, count, _ in limits_in(str(self.policy.get("query") or "")):
            self.tighten(kind, count, False)
        if self.message_budget is None:
            for kind, count, _ in limits_in(str(self.policy.get("query") or ""), loose=True):
                if kind == "message" and count >= 1:
                    self.tighten(kind, count, False)

    def add_duty(self, duty: Any) -> None:
        if isinstance(duty, str):
            duty = {"do": duty, "when": "before_finish", "say": bool(_TELL_RULE.search(duty))}
        if not isinstance(duty, dict) or not str(duty.get("do") or "").strip():
            return
        text = " ".join(str(duty["do"]).split())[:240]
        if any(d["do"].lower() == text.lower() for d in self.duties):
            return
        when = str(duty.get("when") or "before_finish")
        if when not in ("before_order", "before_decline", "before_finish", "before_replace", "anytime"):
            when = "before_finish"
        say = duty.get("say") is True
        if say and when == "before_decline" and self.decline_speaks():
            # what to say when nothing fits travels in the decline call itself: no separate message to keep back for it
            say = False
        asks = duty.get("ask") is True and bool(self.roles.get("message"))
        # an invitation to question the shopper is met by the ordinary questions: it holds no finishing message
        self.duties.append({"do": text, "when": when, "say": say and not asks, "done": asks})

    def decline_speaks(self) -> bool:
        """Whether the decline tool takes free text of its own (a remark, note, message or reason)."""
        props = (self.schemas.get(self.roles.get("decline") or "") or {}).get("properties") or {}
        return any(isinstance(v, dict) and v.get("type") == "string" and re.search(r"remark|note|message|reason|text|explan", k, re.I)
                   for k, v in props.items())

    def tighten(self, kind: str, count: int, relative: bool) -> None:
        """A message or step limit from the request or a later notice; limits only ever get tighter."""
        if kind == "message":
            cap = self.messages_used + count if relative else count
            if self.message_budget is None or cap < self.message_budget:
                self.message_budget = max(self.messages_used, cap)
                self.trace.append({"kind": "limit", "messages": self.message_budget})
        else:
            cap = self.step + count if relative else count
            if cap < self.max_steps:
                self.max_steps = max(self.step + 1, cap)
                self.trace.append({"kind": "limit", "steps": self.max_steps})

    def open_duties(self, moments: set[str], say: bool | None = None) -> list[dict[str, Any]]:
        return [d for d in self.duties if not d["done"] and d["when"] in moments and (say is None or d["say"] is say)]

    def finish_said(self) -> int:
        """One message kept for the finish while a step that is said to the shopper there is still owed (a step owed only at a
        decline stops holding it once the episode gave that message to one more question before an order)."""
        # a remark owed only at a decline holds no message back: declines are rare (about 2 of 90 race tasks) while the
        # second question decides many passes (race 18: 18 episodes sent one message or none, most failed for a need not heard)
        moments = {"before_order", "before_finish", "anytime"}
        return 1 if self.roles.get("message") and self.open_duties(moments, True) else 0

    def ending_message_free(self) -> bool:
        """Whether the one message held back is held only for a step owed at a decline, while every other message went to
        questions: an order does not need it, so it may carry one more planned question (once an episode). Never while a
        step owed at the order or at any finish, or the question owed before a swap, needs a message."""
        if self.ending_freed or self.late_asks >= LATE_ASKS or self.message_budget is None or not self.roles.get("message"):
            return False
        if self.swap_reserve() or self.open_duties({"before_order", "before_finish", "anytime"}, True):
            return False
        if not self.open_duties({"before_decline"}, True) or self.roles.get("message") in self.capped:
            return False
        return self.messages_open() >= 1 and self.messages_used >= self.message_budget - self.held_back()

    def swap_reserve(self) -> int:
        """One message kept for the question a step asks before a swap, until that question has been put."""
        return 1 if self.roles.get("message") and not self.swap_asked and self.open_duties({"before_replace"}) else 0

    def held_back(self) -> int:
        """Messages kept out of the questions: for the finish, or for the question owed before a swap (never more than one)."""
        return max(self.finish_said(), self.swap_reserve())

    def wants_reason(self, moments: set[str]) -> bool:
        """Whether a step owed at these moments asks to be told the reason, the requirement that failed."""
        return any(_REASON_RULE.search(d["do"]) for d in self.open_duties(moments))

    def decline_reason(self) -> str:
        """What the opened listings show was missing, from their own sheets: each condition no affordable listing in stock
        states, and the price limit when the listings that state every condition all cost more. Empty when nothing is known."""
        conditions = self.conditions()
        budget, quantity = self.walk_scope()
        rows = [(k, self.shelf.rows.get(k) or {}) for k in self.shelf.sheets if k not in self.gone]
        live = [(k, r) for k, r in rows if r.get("stock") is not False]
        affordable = [(k, r) for k, r in live if budget is None or r.get("price") is None or r["price"] * quantity <= budget + 1e-9]
        missing = []
        for prop, value in conditions:
            answers = [self.shelf.says(k, prop, value) for k, _ in affordable]
            if answers and "yes" not in answers and "no" in answers:
                missing.append(f"{prop}: {value}")
        if not missing and budget is not None and conditions:
            stating = [k for k, r in live if all(self.shelf.says(k, p, v) == "yes" for p, v in conditions)]
            if stating and all((self.shelf.rows[k].get("price") or 0) * quantity > budget + 1e-9 for k in stating):
                missing.append(f"price limit {budget:g}")
        return "; ".join(missing[:3])

    def reasoned(self, text: str, name: str) -> str:
        """A decline text that names what was missing when a step asks for it and the text does not already."""
        reason = self.decline_reason()
        if not reason or re.search(re.escape(reason.split(":")[0].split()[0]), text, re.I):
            return text
        return f"Unmet: {reason}. {text}"

    def messages_open(self) -> int:
        if self.message_budget is None:
            return 99
        return self.message_budget - self.messages_used

    def finish_reserve(self) -> int:
        """Steps the model loop leaves to the forced finish: the order (or decline) and one repair, plus the step of the
        message the forced finish sends first when a step said to the shopper is still owed and a message remains."""
        room = REFUSAL_ROOM if not self.refusals and self.max_steps >= 20 else 0
        return 2 + room + (1 if self.finish_said() and self.messages_open() >= 1 else 0)

    def empty_cart_fit(self) -> bool:
        """Whether an opened listing meets every known condition while the cart is empty (cached per state of the shelf)."""
        mark = (self.ledger_version, len(self.shelf.sheets), len(self.gone), self.budget_now)
        if self.fit_mark != mark:
            self.fit_mark = mark
            try:
                self.fit_known = bool([k for k in self.live_fits() if k not in self.gone])
            except Exception:
                self.fit_known = False
        return self.fit_known

    def pace_marks(self) -> tuple[int, int]:
        """The step by which a first verified fit should be known, and the step by which the order should be out."""
        cap = max(1, self.max_steps)
        return max(4, round(cap * PACE_FIT)), max(6, round(cap * PACE_COMMIT))

    def verified_now(self) -> list[str]:
        """Listings an item check passed under the current requirements and that are not reported gone, cheapest first."""
        fits = [k for k, v in self.passed_at.items() if v == self.ledger_version and k not in self.gone]
        return sorted(fits, key=lambda k: (self.shelf.rows.get(k) or {}).get("price") or 1e12)

    def commit_phase(self) -> bool:
        """True once a verified fit is known and only COMMIT_STEPS steps remain: no more questions or walking."""
        return self.order_attempts == 0 and self.steps_left() <= COMMIT_STEPS and bool(self.verified_now())

    def order_overdue(self) -> bool:
        """True when a verified fit is known, no order has gone out, the request has no lowest-price rule and the commit
        mark of the pace schedule has passed (one product only)."""
        if self.order_attempts or self.items() or self.plan.get("priority") == "cheapest":
            return False
        return self.step >= self.pace_marks()[1] and bool(self.verified_now())

    def pace_note(self) -> str:
        """One status line on the pace: where the episode stands against the schedule and what that asks for now."""
        if self.order_attempts or self.finished:
            return ""
        fit_by, commit_by = self.pace_marks()
        if self.committed and self.committed in self.cart_ids:
            return (f"[pace] the code put {self.committed} in the cart at step {self.commit_mark()}: ORDER IT NOW, or name the condition "
                    "its own sheet contradicts and fix the cart.")
        fits = self.verified_now()
        if fits:
            named = ", ".join(f"{k} ({(self.shelf.rows.get(k) or {}).get('price', '?')})" for k in fits[:3])
            if self.commit_phase():
                return (f"[pace] COMMIT NOW: {self.steps_left()} steps left and a verified fit is known ({named}). Put the chosen fit in the "
                        "cart and order; no more searching, reading or questions.")
            if self.items():
                return f"[pace] verified so far: {named}; the order should be out by step {commit_by}."
            if self.plan.get("priority") == "cheapest":
                state = "BEHIND" if self.step >= commit_by else "on pace"
                return (f"[pace] {state}: verified fit {named}; with the lowest-price rule, check only the cheaper candidates still unread, "
                        f"then order by step {commit_by}.")
            if self.step >= commit_by:
                return (f"[pace] BEHIND: a verified fit is known ({named}) and the order should have gone out by step {commit_by}. "
                        "Put it in the cart and order now; more browsing costs the steps a refusal needs.")
            return (f"[pace] on pace: a verified fit is known ({named}). Unless a stated requirement is still unchecked, put it in the cart "
                    f"and order now (order by step {commit_by}); keep the next fit as a backup.")
        if self.step >= fit_by:
            return (f"[pace] BEHIND: no verified fit yet, and one should have been known by step {fit_by}. Open the details of the two "
                    f"most promising listings and put the best one in the cart; order by step {commit_by}.")
        return f"[pace] on pace: first verified fit by step {fit_by}, order by step {commit_by}."

    def brief_holds(self, plan: dict[str, Any]) -> bool:
        """Whether a brief is complete enough to work from: a product, the list-shaped keys as lists, a known priority,
        and roles that name only real tools, searching and ordering among them."""
        if not isinstance(plan, dict) or not str(plan.get("product") or "").strip():
            return False
        if any(not isinstance(plan.get(k, []), list) for k in ("requirements", "rules", "ask_first", "items")):
            return False
        if plan.get("priority") not in ("cheapest", "best", "none"):
            return False
        cap = plan.get("price_cap")
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, float)) or cap <= 0):
            return False
        roles = plan.get("roles")
        if not isinstance(roles, dict):
            return False
        named = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        if len(named) < len([v for v in roles.values() if v]):
            return False
        return bool(named.get("search")) and bool(named.get("order"))

    def id_arguments(self) -> list[str]:
        """The argument names by which this store's details, view, stock and cart tools name one listing: the required
        string arguments that read like identifiers (a single required string argument counts even without such a name)."""
        out: list[str] = []
        for role in ("details", "view", "stock", "cart", "cart_remove"):
            function = self.schemas_full.get(self.roles.get(role) or "") or {}
            params = function.get("parameters") or {}
            props = params.get("properties") or {}
            required = [p for p in (params.get("required") or []) if isinstance(p, str)] or list(props)
            strings = [p for p in required if isinstance(props.get(p), dict) and "string" in str(props[p].get("type")) and "enum" not in props[p]]
            iddy = [p for p in strings if re.search(IDDY, flat(re.sub(r"[_\-]+", " ", p)))]
            picks = iddy or (strings if len(strings) == 1 and role in ("details", "view", "stock") else [])
            out.extend(p for p in picks if p not in out)
        return out

    def split_finish_roles(self) -> None:
        """One tool read as both finishes although it has no argument to tell a purchase from a refusal: when the store
        also has a finishing tool without required arguments that the guess reads as the refusal, that one declines (else
        an empty-cart 'decline' would go out as an order the store refuses, again and again)."""
        order, decline = self.roles.get("order"), self.roles.get("decline")
        if not order or order != decline:
            return
        params = (self.schemas_full.get(order) or {}).get("parameters") or {}
        if any(isinstance(r, dict) and (r.get("type") == "boolean" or isinstance(r.get("enum"), list)) for r in (params.get("properties") or {}).values()):
            return                               # a switch or a mode tells the two apart: one tool really serves both
        other = self.guess_roles().get("decline")
        if not other or other == order or other not in self.schemas:
            return
        if (((self.schemas_full.get(other) or {}).get("parameters") or {}).get("required") or []):
            return
        self.roles["decline"] = other
        self.trace.append({"kind": "roles_split", "order": order, "decline": other})

    def settle_cart_role(self) -> None:
        """A tool that only takes lines out is never the cart: when the reading named one, the tool that puts lines in
        takes the cart role and the remover becomes cart_remove."""
        cart = self.roles.get("cart")
        function = self.schemas_full.get(cart or "")
        if not function or sort_tool(function)[0] != "uncart":
            return
        adders = [str(f.get("name")) for f in (t.get("function") or {} for t in self.tools)
                  if f.get("name") != cart and sort_tool(f)[0] == "cart"]
        if adders:
            self.roles["cart"] = adders[0]
            self.roles.setdefault("cart_remove", cart)
            self.trace.append({"kind": "cart_role_fixed", "cart": adders[0], "remover": cart})

    def split_cart_role(self) -> None:
        """A cart in two tools of the same shape (one puts a line in, one takes it out) and no remover named: the one whose
        description speaks of taking out is the remover."""
        cart = self.roles.get("cart")
        if not cart or self.roles.get("cart_remove"):
            return
        shape = lambda f: sorted(((f.get("parameters") or {}).get("properties") or {}).keys())
        mine = self.schemas_full.get(cart) or {}
        if any(isinstance(v, dict) and v.get("enum") for v in ((mine.get("parameters") or {}).get("properties") or {}).values()):
            return
        assigned = set(self.roles.values())
        twins = [f for f in (t["function"] for t in self.tools) if f.get("name") != cart and f.get("name") not in assigned and shape(f) == shape(mine) and shape(f)]
        if len(twins) != 1:
            return
        twin = twins[0]
        says_out = lambda f: bool(_OUT_WORDS.search(flat(f"{f.get('name')} {f.get('description') or ''}".replace("_", " "))))
        if says_out(mine) and not says_out(twin):
            self.roles["cart"], self.roles["cart_remove"] = str(twin["name"]), cart
        else:
            self.roles["cart_remove"] = str(twin["name"])
        self.trace.append({"kind": "cart_split", "cart": self.roles["cart"], "remover": self.roles["cart_remove"]})

    def find_view_mode(self) -> None:
        """A cart tool whose mode enum carries a value besides putting in and taking out (and no separate cart view): that
        value lists the basket."""
        self.view_mode = None
        cart = self.roles.get("cart")
        props = ((self.schemas_full.get(cart or "") or {}).get("parameters") or {}).get("properties") or {}
        for key, rule in props.items():
            values = [v for v in (rule.get("enum") or []) if isinstance(v, str)] if isinstance(rule, dict) else []
            if len(values) < 3:
                continue
            hint = self.plan.get("cart_modes") if isinstance(self.plan.get("cart_modes"), dict) else {}
            if hint.get("view") in values:
                self.view_mode = (key, hint["view"])
                return
            add, out = mode_pair(values, hint)
            rest = [v for v in values if v not in (add, out)]
            if rest:
                self.view_mode = (key, next((v for v in rest if re.search(r"view|show|list|look|content", v, re.I)), rest[-1]))
                return

    def viewing(self, name: str, args: dict[str, Any]) -> bool:
        """A cart-tool call in its basket-listing mode (no line moves)."""
        return bool(self.view_mode) and name == self.roles.get("cart") and args.get(self.view_mode[0]) == self.view_mode[1]

    def sheet_tool(self) -> str | None:
        """The reading tool that returns a listing's property sheet: one seen doing so, else the details tool unless it was
        seen returning none while the view tool exists."""
        details, view = self.roles.get("details"), self.roles.get("view")
        for tool in (details, view, self.roles.get("compare")):
            if tool and tool in self.sheet_tools:
                return tool
        if details and details in self.bare_tools and view and view not in self.bare_tools:
            return view
        return details or view

    def view_call(self) -> tuple[str, dict[str, Any]] | None:
        """The call that lists the basket: the cart view tool, else the cart tool in its listing mode."""
        view = self.roles.get("cart_view")
        if view and view in self.schemas_full:
            return view, {}
        if self.view_mode and self.roles.get("cart"):
            return self.roles["cart"], {self.view_mode[0]: self.view_mode[1]}
        # no way to list the basket (a whole-basket hold): the held line's own page is read again instead; a change the
        # hold set off rides on that reply
        tool = self.sheet_tool()
        held = next((k for k in reversed(self.cart_ids) if k not in self.gone), "")
        args = self.id_args(tool, held) if tool and held else None
        return (tool, args) if args else None

    def guess_roles(self) -> dict[str, str]:
        """Roles read off each tool's name, description and parameters by our weighted scoring (every kind scores, the top
        wins), so a failed reading of the request still leaves every guard working."""
        to_role = {"seek": "search", "peek": "details", "talk": "message", "cart": "cart", "cartview": "cart_view",
                   "buy": "order", "quit": "decline", "stock": "stock", "uncart": "cart_remove"}
        to_role.update(compare="compare", usage="usage")
        guessed: dict[str, str] = {}
        for tool in self.tools:
            function = tool.get("function") or {}
            kind, _ = sort_tool(function)
            role = to_role.get(kind)
            if role and role not in guessed:
                guessed[role] = str(function.get("name"))
        if "order" not in guessed:
            # one finishing tool whose mode argument chooses between buying and not buying serves both roles
            both = next((str(f.get("name")) for f in (t.get("function") or {} for t in self.tools) if sort_tool(f)[1]
                         and any(isinstance(r, dict) and r.get("enum") for r in ((f.get("parameters") or {}).get("properties") or {}).values())), None)
            if both:
                guessed["order"] = both
                guessed.setdefault("decline", both)
        if "order" not in guessed or "decline" not in guessed:
            # a finish ruled by a yes/no switch, or the store's only finishing tool, serves both finishes
            enders = [f for f in (t.get("function") or {} for t in self.tools) if sort_tool(f)[1]]
            switched = [f for f in enders if any(isinstance(r, dict) and r.get("type") == "boolean" for r in ((f.get("parameters") or {}).get("properties") or {}).values())]
            pick = (switched or (enders if len(enders) == 1 else []))[:1]
            for f in pick:
                guessed.setdefault("order", str(f.get("name")))
                guessed.setdefault("decline", str(f.get("name")))
        if "details" not in guessed and "view" not in guessed and "compare" in guessed:
            guessed["details"] = guessed["compare"]   # the store's only reader takes a list of listings
        if "details" in guessed and "view" not in guessed:
            others = [str((t.get("function") or {}).get("name")) for t in self.tools if sort_tool(t.get("function") or {})[0] == "peek" and (t.get("function") or {}).get("name") != guessed["details"]]
            if others:
                guessed["view"] = others[0]
        return guessed

    def items(self) -> list[tuple[str, int]]:
        # the different products of a request that asks for several; empty for a one-product request
        out = []
        raw = self.plan.get("items")
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, str) and item.strip():
                item = {"product": item, "quantity": 1}
            if isinstance(item, dict) and str(item.get("product") or "").strip():
                quantity = item.get("quantity")
                if isinstance(quantity, str) and quantity.strip().isdigit():
                    quantity = int(quantity.strip())
                quantity = int(quantity) if isinstance(quantity, (int, float)) and not isinstance(quantity, bool) and quantity >= 1 else 1
                out.append((str(item["product"]).strip(), quantity))
        return out if len(out) >= 2 else []

    def brief(self) -> str:
        plan = self.plan
        if not plan:
            return ""
        lines = ["A first reading of the request (check it against the request itself):"]
        if plan.get("product"):
            lines.append(f"- product: {plan.get('product')}")
        items = self.items()
        if items:
            for name, quantity in items:
                lines.append(f"- item to buy: {name}, quantity {quantity} (one cart line per item)")
        else:
            lines.append(f"- quantity: {plan.get('quantity') or 1}")
        for requirement in plan.get("requirements") or []:
            lines.append(f"- stated requirement: {requirement}")
        if plan.get("price_cap") is not None:
            lines.append(f"- price cap (total): {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip())
        if plan.get("in_stock_only"):
            lines.append("- only items in stock / on hand")
        if plan.get("priority") in ("cheapest", "best"):
            detail = f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else ""
            lines.append(f"- choose the {plan.get('priority')}{detail} among all fitting items")
        else:
            lines.append("- no cheapest/best rule was found: any item that meets every requirement within the budget is right, so do not spend steps hunting for the lowest price (unless the request itself says otherwise)")
        for want in plan.get("soft_wants") or []:
            lines.append(f"- nice to have, NOT required (only a tie-breaker): {want}")
        for item in plan.get("ask_first") or []:
            lines.append(f"- must be learned from the shopper first: {item}")
        for rule in plan.get("rules") or []:
            lines.append(f"- rule: {rule}")
        for duty in self.duties:
            lines.append(f"- step to take ({duty['when'].replace('_', ' ')}): {duty['do']}" + (" (the message for it is sent for you right before the finish)" if duty["say"] and duty["when"] != "before_replace" else ""))
        if self.plan.get("bundle"):
            lines.append("- the products must work with each other: check that the chosen listings fit together")
        if self.message_budget is not None:
            lines.append(f"- messages to the shopper allowed: {self.message_budget}")
        lines.append(f"- steps allowed: {self.max_steps}")
        hints = [h for h in plan.get("hint_properties") or [] if isinstance(h, str)]
        if hints:
            lines.append("- properties the story may hint at (ask about these first, one per message): " + ", ".join(hints))
        return "\n".join(lines)

    def status(self) -> str:
        left = self.steps_left()
        parts = [f"[status] steps used {self.step} of {self.max_steps}, {left} left"]
        if self.message_budget is not None:
            parts.append(f"messages sent {self.messages_used} of {self.message_budget}")
        else:
            parts.append(f"messages sent {self.messages_used}")
        parts.append(f"about {int(max(0, self.time_left() - FORCE_SECONDS))} s left")
        text = "; ".join(parts) + "."
        pace = self.pace_note()
        if pace:
            text += "\n" + pace
        text += "\n[requirements now] " + self.ledger()
        pointed = self.pointer_note()
        if pointed:
            text += "\n" + pointed
        bundle = self.bundle_note()
        if bundle:
            text += "\n" + bundle
        fitted = self.fit_note()
        if fitted:
            text += "\n" + fitted
        if self.market_notes and self.step - self.market_at <= 6:
            text += "\n[market] the store or the shopper said: " + " | ".join(self.market_notes[-2:]) + " Apply it to the cart before ordering."
        if self.budget_changed_at >= 0 and self.step - self.budget_changed_at <= 6:
            text += f"\n[budget changed] the shopper's line at step {self.budget_changed_at} set the total limit to {self.budget_now}: judge every price against it now."
        owed = [d["do"] for d in self.duties if not d["done"]]
        if owed:
            text += "\n[still to do before finishing] " + "; ".join(owed)
        if self.refusals and not self.finished and self.step - self.refused_at >= 2:
            # a refused order is often misread as a placed one; until the episode ends nothing has been bought
            text += (f" NOTHING HAS BEEN BOUGHT YET: the order attempt {self.step - self.refused_at} steps ago did not end the episode."
                     " If the cart holds fitting items within the current budget, call the order tool now; if not, fix the cart first, then order"
                     " (decline only if nothing in the store fits any more).")
        table = self.shelf.table(self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None, LEAN_TABLE_ROWS if self.past_soft() else 12)
        if table:
            text += "\n" + table
        if left <= 2:
            text += " LAST STEPS: order now if the cart holds a verified fit, otherwise decline."
        elif left <= 5 or self.time_left() < WRAP_UP_SECONDS:
            text += " FINISH NOW: fix the cart to the verified line(s) the request needs, re-check stock, then order (or decline if nothing fits)."
        elif self.order_attempts == 0 and left <= 14:
            text += " Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when 7 steps remain, and a refused order needs about 4 steps to repair."
        return text

    def ledger(self) -> str:
        if self.current is not None:
            items = list(self.current)
        else:
            items = [f"{r} (stated)" for r in self.stated if r not in self.dropped]
            items += [f"{r} (revealed by the shopper)" for r in self.revealed if r not in self.dropped]
        text = "; ".join(items) if items else "see the request"
        if self.budget_now is not None:
            text += f"; total price at most {self.budget_now} {self.plan.get('currency') or ''}".rstrip()
        items = self.items()
        if items:
            text += "; items and quantities: " + ", ".join(f"{name} x{quantity}" for name, quantity in items)
        else:
            text += f"; quantity {self.plan.get('quantity') or 1}"
        if self.dropped:
            text += "; no longer required: " + "; ".join(self.dropped)
        if self.free:
            text += "; shopper does not care about: " + ", ".join(self.free)
        return text

    def judge(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None) -> dict[str, Any] | None:
        models = models or [JUDGE_MODEL, MAIN_MODEL]
        if self.cost > self.cost_soft and models[0] == JUDGE_MODEL:
            # an expensive episode keeps its checks but makes them on the cheaper model, so one task cannot eat the key
            models = [READ_MODEL, JUDGE_MODEL]
        message = self.infer([{"role": "user", "content": prompt}], None, models, max_tokens)
        return _parse_json_object((message or {}).get("content") or "")

    def interpret(self, lines: list[str], question: str | None) -> str:
        """The shopper's new lines read by the hearing model and folded into the requirements; the gist for the transcript."""
        fresh = [line for line in lines if line not in self.heard]
        if question is not None:
            # the width of the questions is read by the code from every line (a repeated complaint is still a complaint)
            self.last_lines = [str(line) for line in lines]
            self.last_crowded = any(crowded_reply(line) for line in lines)
            if not fresh and any(line in self.unclear_lines for line in lines):
                self.last_unclear = True     # the same "say that again" line as before: not followed again
            if self.last_crowded:
                self.narrow_for_crowd()
        if not fresh:
            return ""
        self.heard.update(fresh)
        coded = self.code_hearing(fresh, question)
        if self.time_left() < WRAP_UP_SECONDS:
            return coded
        result = self.judge(self.hearing_prompt(fresh, question), 1500, [READ_MODEL, JUDGE_MODEL])
        if not result:
            if question and not coded:
                self.unfolded += 1
            return coded
        notes = self.fold_changes(result)
        if coded:
            notes.append(coded)
        if question and not coded and not any(result.get(k) for k in ("needs", "no_preference", "dropped", "new_budget", "new_quantity", "unavailable", "unclear")):
            self.unfolded += 1
        if notes:
            self.ledger_version += 1
        follow = self.after_hearing(result, question, fresh)
        if follow:
            notes.append(follow)
        return "Meaning: " + str(result.get("meaning") or "") + (" (" + "; ".join(notes) + ")" if notes else "")

    def hearing_prompt(self, fresh: list[str], question: str | None) -> str:
        """The hearing model's input; the exchange is remembered for the next reading."""
        parts = [INTERPRET_PROMPT, "THE SHOPPER'S REQUEST (for context):\n" + str(self.policy.get("query") or ""), "REQUIREMENTS SO FAR:\n" + self.ledger()]
        if self.said:
            parts.append("EARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(self.said[-12:]))
        if question:
            parts.append("THE ASSISTANT ASKED:\n" + question)
            if len(self.last_group) > 1:
                parts.append("THAT QUESTION NAMED SEVERAL PROPERTIES: " + "; ".join(self.last_group) + ". The reply may give a firm value for any of "
                             "them, or for none; write each need with the property's name exactly as listed here, and put a property in "
                             "no_preference only when the reply waves it off (a reply that waves off the question as a whole frees all of them).")
        parts.append("THE SHOPPER NOW SAID:\n" + "\n".join(fresh))
        self.said.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        return "\n\n".join(parts)

    def fold_changes(self, result: dict[str, Any]) -> list[str]:
        """Requirements, free properties, drops, budget, quantity and availability from one hearing, with notes."""
        notes: list[str] = []
        # a reading that drifted in its key names still counts (new needs, free properties, drops, a limit, a sold-out line)
        for want, near in (("needs", ("adds", "new_needs")), ("no_preference", ("frees", "released")), ("dropped", ("drops", "withdrawn")),
                           ("new_budget", ("limit", "budget")), ("unavailable", ("sold_out", "gone"))):
            if not result.get(want):
                found = next((result[k] for k in near if result.get(k)), None)
                if found is not None:
                    result[want] = found
        now = result.get("requirements_now")
        if isinstance(now, list) and all(isinstance(r, str) for r in now):
            self.current = [r for r in now if r.strip()]
            self.groom_current()
        if [v for v in result.get("dropped") or [] if isinstance(v, str) and v and v not in self.dropped]:
            self.relaxed_at = self.step
        for key, store, label in (("needs", self.revealed, "new requirement"), ("no_preference", self.free, None), ("dropped", self.dropped, "dropped")):
            for value in result.get(key) or []:
                if isinstance(value, str) and value and value not in store:
                    store.append(value)
                    if label:
                        notes.append(f"{label}: {value}")
        pointed = self.take_pointers([str(n) for n in result.get("needs") or [] if isinstance(n, str)])
        for item in pointed:
            notes.append(f"the need '{item.partition(':')[0].strip()}' points at another listing's page: the code will open that page and settle the value")
        budget = result.get("new_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget_now:
            self.set_budget(float(budget), "the shopper's reply", "hearing")
            notes.append(f"new budget: {budget}")
        quantity_note = self.fold_quantity(result.get("new_quantity"))
        if quantity_note:
            notes.append(quantity_note)
        if result.get("unavailable"):
            notes.append("an item became unavailable; remove it from the cart and do not order it")
            self.mark_change()
        if result.get("price_moved") and self.cart_ids:
            self.price_moved_at = self.step
            notes.append("a price moved: the held lines are re-read and judged again before the order")
            self.mark_change()
        for duty in result.get("duties") or []:
            before = len(self.duties)
            self.add_duty(duty)
            if len(self.duties) > before:
                notes.append("new step to take: " + self.duties[-1]["do"])
        if any(not n.startswith("new step") for n in notes):
            self.review_reason = "; ".join(notes)
        return notes

    def fold_quantity(self, change: Any) -> str:
        """A new quantity: the plan's for one kind of product, the named product's line for several."""
        if isinstance(change, int) and not isinstance(change, bool):
            change = {"product": "", "quantity": change}
        if not (isinstance(change, dict) and isinstance(change.get("quantity"), int) and not isinstance(change.get("quantity"), bool) and change["quantity"] >= 1):
            return ""
        wanted, product = change["quantity"], str(change.get("product") or "").strip().lower()
        if not self.items():
            self.plan["quantity"] = wanted       # one kind of product: the plan's quantity is the one that counts
            return f"new quantity: {wanted}"
        items = self.plan.get("items") if isinstance(self.plan.get("items"), list) else []
        target = next((i for i in items if isinstance(i, dict) and product and (product in str(i.get("product") or "").lower() or str(i.get("product") or "").lower() in product)), None)
        if target is None:
            return ""
        target["quantity"] = wanted
        return f"new quantity: {target.get('product')} x{wanted}"

    def after_hearing(self, result: dict[str, Any], question: str | None, fresh: list[str]) -> str:
        """A question the shopper did not follow (with several products: one that named none of them), or a reply owed."""
        if result.get("crowded") is True and question is not None and not self.last_crowded:
            self.last_crowded = True
            self.narrow_for_crowd()
        unclear = result.get("unclear") is True and question is not None and not self.last_crowded
        self.last_unclear = unclear
        if self.last_crowded:
            # a complaint about the width is not confusion: the code's interview puts the same properties again, one a message
            return "the shopper wants one subject per message: from now on every question names ONE property of ONE product"
        if unclear:
            self.unclear_lines.update(fresh)
            named = True
            if self.items():
                asked_words = {w.rstrip("s") for w in str(question).lower().replace("?", " ").split()}
                named = any(w.rstrip("s") in asked_words for product, _ in self.items() for w in product.lower().split()
                            if len(w) >= 3 and w not in ("and", "the", "for", "set", "with", "pair", "pack", "of"))
            if not named:
                # with several products, a question that does not say which one is ambiguous: the property may still matter
                return "the shopper did not follow that question because it did not say which product it is about: if messages remain, ask about that property once more, naming the product"
            # the code's interview puts a one-property question once more in plainer words; a second miss counts it as free
            return "the shopper did not follow that question: put it once more in short plain words (one property only), and after a second miss treat the property as free"
        # a line that merely states a change ("could we work with N instead?") gets no acknowledgement: messages are for
        # questions only, and the change is applied by the ledger
        return ""

    def code_hearing(self, fresh: list[str], question: str | None) -> str:
        """What the code itself reads off the shopper's new lines before any model: a need that points at another
        listing's page (the property is the one the question asked about)."""
        if any(_POINTER.search(line) for line in fresh) and not any(_POINTER.search(str(n)) for n in self.revealed):
            prop = ""
            line = next(l for l in fresh if _POINTER.search(l))
            group = list(self.last_group) if question and len(self.last_group) > 1 else []
            if group:
                # a question that named several properties: the one the reply itself names (before the pointer), else the first
                head = line[: _POINTER.search(line).start()]
                at = self.key_of(head, group)
                prop = group[at if at >= 0 else max(0, self.key_of(line, group))]
            elif question:
                found = re.search(r"\b(?:which|what)\s+(.+?)\s+(?:would|do|should)\b", question, re.I)
                prop = found.group(1).strip() if found else ""
            prop = prop or (self.asked_props[-1] if self.asked_props else "")
            if prop:
                need = f"{prop}: {line.strip()}"
                self.revealed.append(need)
                self.ledger_version += 1
                if self.take_pointers([need]):
                    return f"the reply points at another listing's page for the value of {prop}: the code will open that page and settle it"
                self.revealed.remove(need)
        return ""

    def item_records(self, item: str) -> list[str]:
        """The latest records of one listing as the shelf ledger kept them (its search row, details and views)."""
        return list(self.shelf.records.get(item) or [])[-3:]

    def verify_cart(self) -> str:
        if not self.cart_ids:
            return ""
        # a request for several different products puts one line per product in the cart: check each line
        for item in (self.cart_ids[-3:] if self.items() else self.cart_ids[-1:]):
            verdict = self.verify_item(item, "ORDER", self.cart_lines.get(item, ""))
            if verdict:
                return verdict
        return ""

    def verify_item(self, item: str, purpose: str, line: str = "") -> str:
        if self.time_left() < WRAP_UP_SECONDS + 20 or self.cost >= self.cost_hard:
            return ""
        key = f"{item}|{line}|{self.ledger_version}"
        if key in self.verdicts:
            return self.verdicts[key]
        targets = self.fit_targets()
        if targets and self.fit_state(item, targets) == "fails":
            verdict = (f"{item}'s own page lists fitment codes, but not {', '.join(targets[:3])}, so it does not fit. Choose a listing whose page lists the code.")
            self.verdicts[key] = verdict
            return verdict
        records = self.item_records(item)
        if not records:
            return ""
        prompt = VERIFY_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + self.ledger()
        prompt += f"\n\nTHE ITEM TO {purpose}: {item}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n" + "\n".join(records)
        if line:
            prompt += "\n\nTHE CART LINE FOR IT (when it names an option or variant, judge that option: its own values, price and stock count): " + line
        if self.last_cart_view is not None:
            prompt += "\n\nLATEST CART VIEW:\n" + _compact(self.last_cart_view)[:2000]
        labels = self.row_labels()
        if labels:
            prompt += "\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the item's category label must be the one that matches the request's wording as a whole, not a neighbouring one):\n" + labels
        settled = self.labels_settle(item)
        plain = settled or self.labels_settle(item, loose=True)
        result = self.judge(prompt, 1200, [READ_MODEL] if plain else None)
        if plain and (not result or (result.get("fits") is not True and not self.past_soft())):
            # the cheap check doubts the listing (or gives no clear answer): the full check has the last word (past the soft
            # line only when the cheap check gave no answer at all: the doubt then stands without a second paid look)
            result = self.judge(prompt, 1200, [JUDGE_MODEL, MAIN_MODEL]) or result
        if result and result.get("fits") is True:
            self.passed.add(item)
            self.passed_at[item] = self.ledger_version
        self.trace.append({"kind": "verify", "item": item, "result": result, "settled": settled, "plain": plain})
        if not result or result.get("fits") is not False:
            self.verdicts[key] = ""
            return ""
        problems = [str(p) for p in result.get("problems") or []][:5]
        settled_props = {flat(p["prop"]): p.get("value") for p in self.pointers if p.get("done")}
        if settled_props:
            # a property the code settled by following a pointer, which the item's own sheet meets by the code's reading,
            # is not the check's to doubt; nor is the seller of the page the pointer named
            def stands(problem: str) -> bool:
                prop = flat(problem.partition(":")[0])
                if prop in settled_props and self.shelf.says(item, prop, str(settled_props[prop])) == "yes":
                    return False
                return not (re.search(WHO_KEYS, prop) and not any(re.search(WHO_KEYS, flat(c)) for c, _ in self.conditions()))
            problems = [p for p in problems if stands(p)]
            if not problems:
                self.trace.append({"kind": "verify", "item": item, "overruled": "pointer settled by the code"})
                self.passed.add(item)
                self.passed_at[item] = self.ledger_version
                self.verdicts[key] = ""
                return ""
        verdict = f"An independent check says {item} does NOT meet every requirement: " + "; ".join(problems) + ". Unless you are certain the check is wrong, choose another fully verified fit."
        self.verdicts[key] = verdict
        return verdict

    def labels_settle(self, item: str, loose: bool = False) -> bool:
        """Whether the listing's own records leave nothing for judgement: one product kind, its store label naming the
        requested kind, a known price within the limit, not shown sold out, and every condition in force settled by
        its labels. Such a listing is checked on the cheaper model first. Loose: every condition only has to read
        'yes' on its sheet (figures and secondary labels included), which still sends a clear fit to the cheaper check."""
        if self.items():
            return False
        row = self.shelf.rows.get(item) or {}
        if row.get("price") is None or row.get("stock") is False or not row.get("group"):
            return False
        cap = self.budget_now
        quantity = self.plan.get("quantity") or 1
        if isinstance(cap, (int, float)) and isinstance(quantity, int) and row["price"] * quantity > cap + 1e-9:
            return False
        stem = lambda w: w[:-2] if w.endswith("es") and len(w) > 4 else (w[:-1] if w.endswith("s") and len(w) > 3 else w)
        kind = {stem(w) for w in words(self.plan.get("product") or "") if len(w) >= 3}
        label = {stem(w) for w in words(row["group"]) if len(w) >= 3}
        if not kind or not (kind & label):
            return False
        if loose:
            conditions = self.conditions()
            return bool((self.shelf.rows.get(item) or {}).get("sheet")) and bool(conditions) and all(self.shelf.says(item, prop, value) == "yes" for prop, value in conditions)
        return all(self.shelf.settles(item, prop, value) for prop, value in self.conditions())

    def tools_for_turn(self) -> list[dict[str, Any]]:
        left = self.steps_left()
        order, decline, cart = self.roles.get("order"), self.roles.get("decline"), self.roles.get("cart")
        names: set[str] | None = None
        if left <= 1:
            names = {order, decline}
        elif left <= 3:
            names = {order, decline, cart}
        elif left <= 7 and (self.order_attempts == 0 or self.refusals > 0):
            names = {order, decline, cart, self.roles.get("cart_view"), self.roles.get("view")}
        # money never narrows the tools: past the lines the episode makes cheaper calls, it never stops searching
        tools = self.tools
        if names is not None and order and cart:
            # the tool that takes lines out stays with the cart tool: a dead line must be removable in the last steps
            names |= {self.roles.get("cart_remove")} - {None}
            # a tool no role covers that finishes a checkout (a confirm step after the order tool) stays available
            assigned = set(self.roles.values())
            names |= {t["function"]["name"] for t in self.tools if t["function"]["name"] not in assigned
                      and _FINISH_STEP.search(t["function"]["name"])}
            narrowed = [t for t in self.tools if t["function"]["name"] in names]
            if narrowed:
                tools = narrowed
        if self.planner_on and self.roles.get("message"):
            tools = [t for t in tools if t["function"]["name"] != self.roles.get("message")]
        if decline and order and self.refusals and self.step - self.refused_at <= REFUSAL_BLOCK and len(tools) > 1 and self.live_fits():
            # right after a refused finish a decline is never the answer while an opened sheet fits: the tool is not offered
            tools = [t for t in tools if t["function"]["name"] != decline] or tools
        if self.capped:
            kept = [t for t in tools if t["function"]["name"] not in self.capped]
            tools = kept or tools
        return tools

    def property_menu(self) -> tuple[str, int]:
        """The sheet properties seen so far, those whose values differ first, and how many sheets were read (the shelf's menu)."""
        return self.shelf.menu()

    def row_labels(self) -> str:
        """The store categories seen with how many listings carry each (the shelf ledger's count)."""
        return self.shelf.group_counts()

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

    def ask_planned(self, force: bool = False) -> bool:
        tool = self.roles.get("message")
        limit = self.message_budget - self.held_back() if self.message_budget is not None else UNSTATED_MESSAGES
        if not tool or self.messages_used >= limit:
            self.planner_on = False
            return False
        menu, fact_sets = self.property_menu()
        split, pool = self.shelf.splits(self.conditions(), str(self.policy.get("query") or ""), self.free + self.asked_props)
        # once two candidates of one kind have been read, the open questions come from their differences, one at a
        # time, each chosen after the answer before it has narrowed the candidates
        stepwise = bool(split) and not self.items()
        urgent = bool(self.plan.get("ask_first")) and not self.asked
        if fact_sets < 4 and self.step < 8 and not urgent and not force:
            return False
        if force:
            urgent = False
        # a later planning round (questions left over, or one at a time for several products) waits for more listings
        # (or a commit attempt), so it is chosen knowing the earlier answers and more of the store
        if self.asked and not force and not urgent and not self.last_unclear and not stepwise and fact_sets < self.sets_at_last_ask + 2:
            return False
        if not force and fact_sets < self.plan_retry_at:
            return False
        if self.steps_left() <= (6 if force else 10) or self.time_left() < WRAP_UP_SECONDS + 40 or self.commit_phase():
            self.planner_on = False
            return False
        # one product: plan every remaining question at once, so the questions do not eat into the steps needed for
        # the search. Several products: one at a time, so each is chosen knowing which products were already asked
        # about and what more of the store shows
        count = 1          # one question a round: each is chosen after the answer before it was read
        prompt = PLAN_PROMPT.replace("QUESTION_COUNT", str(count)) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        items = self.items()
        if items:
            prompt += "\n\nTHE REQUEST ASKS FOR SEVERAL DIFFERENT PRODUCTS (name the one each question is about): " + "; ".join(name for name, _ in items)
        if self.asked:
            prompt += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.asked)
        if self.plan.get("ask_first"):
            prompt += "\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in self.plan.get("ask_first") or [])
        prompt += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n" + (menu or "(none seen yet)")
        if stepwise:
            prompt += (f"\n\nOPEN QUESTIONS AMONG THE {pool} CANDIDATES STILL IN PLAY (opened listings of the requested kind that no "
                       "requirement rules out; properties the request or earlier answers already settle are left out), the property "
                       "whose answer could set aside the most candidates first:\n" + split +
                       "\nAfter anything under rule 0, take the property from the top of this list unless the story clearly points "
                       "to another one on it.")
        result = self.judge(prompt, 2000, [READ_MODEL, JUDGE_MODEL])
        planned = [q for q in (result or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("question") or "").strip()]
        if urgent and fact_sets < 4:
            planned = planned[:1]
        # never send a question that has been sent before in the same words
        planned = [q for q in planned if str(q.get("question")).strip().lower() not in self.questions_sent]
        planned = planned[:count]
        calls = []
        labels = []
        for index, item in enumerate(planned):
            args = self.message_args(str(item.get("question")).strip())
            if not args:
                continue
            args, problems = _check(args, self.schemas[tool], "args")
            if problems:
                continue
            calls.append({"id": f"ask_{self.messages_used + index + 1}_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}})
            product = str(item.get("product") or "").strip()
            labels.append(((product + " " if product and items else "") + str(item.get("property")), str(item.get("question")).strip()))
            self.asked_props.append(str(item.get("property") or ""))
        if not calls:
            # a planning call that gave nothing usable is retried once, after another listing has been opened
            self.plan_misses += 1
            self.plan_retry_at = fact_sets + 1
            if self.plan_misses >= 2:
                self.planner_on = False
            return False
        self.messages.append({"role": "assistant", "content": "I will ask the shopper about " + ", then ".join(l[0] for l in labels) + ".", "tool_calls": calls})
        self.last_unclear = False      # used up by this round; the new answers set it again
        replies = self.run_calls(calls, planned=True)
        for index, (call_id, text) in enumerate(replies):
            if index == len(replies) - 1:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            if index < len(labels):
                self.asked.append(f"- {labels[index][0]}: asked \"{labels[index][1]}\" -> {text.split(chr(10) + '[status]')[0][:400]}")
        self.sets_at_last_ask = fact_sets
        self.planner_on = self.messages_used < limit
        return True

    def send_reply(self, text: str) -> bool:
        tool = self.roles.get("message")
        limit = self.message_budget - self.held_back() if self.message_budget is not None else UNSTATED_MESSAGES
        if not tool or self.messages_used >= limit or self.steps_left() <= 4 or self.time_left() < WRAP_UP_SECONDS:
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

    def quantity_warning(self, name: str, args: dict[str, Any]) -> str:
        wanted = self.plan.get("quantity")
        if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
            return ""
        properties = self.schemas[name].get("properties") or {}
        signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if self.add_signature is not None and signature != self.add_signature:
            return ""
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                value = args.get(key, spec.get("default", 1))
                if self.items():
                    allowed = {q for _, q in self.items()}
                    if isinstance(value, int) and value not in allowed:
                        return f"the request needs these quantities: {', '.join(f'{n} x{q}' for n, q in self.items())}, but this call asks for {value}."
                    return ""
                if isinstance(value, int) and value != wanted:
                    return f"the request needs a quantity of {wanted} of one listing, but this call asks for {value}."
        return ""

    def unchecked_cheaper(self) -> list[tuple[str, Any]]:
        """For a cheapest request: rows priced below the carted pick, in its category, not shown out of stock, within the
        budget, never opened (the shelf ledger's view; at most five, cheapest first)."""
        if self.plan.get("priority") != "cheapest" or not self.cart_ids:
            return []
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        quantity = self.plan.get("quantity") if isinstance(self.plan.get("quantity"), int) and self.plan.get("quantity") > 0 else 1
        return [(k, p) for k, p in self.shelf.unopened_cheaper(self.cart_ids[-1], budget, quantity) if k not in self.opened and k not in self.cart_ids][:5]


    # ------------------------------------------------------------------ the price-limit walk (cheapest requests)

    def search_keys(self) -> dict[str, Any]:
        """Which arguments of the search tool carry the free text, the category, the price limit and the page, and its
        enum arguments (a mode, a sort order), read off the schema."""
        function = self.schemas_full.get(self.roles.get("search") or "") or {}
        params = function.get("parameters") or {}
        props = params.get("properties") or {}

        def first(pattern: str, kinds: tuple[str, ...]) -> str | None:
            for key, rule in props.items():
                rule = rule if isinstance(rule, dict) else {}
                kind = rule.get("type")
                kind = kind if isinstance(kind, list) else [kind]
                if "enum" not in rule and any(k in kinds for k in kind) and re.search(pattern, flat(re.sub(r"[_\-]+", " ", key))):
                    return key
            return None

        enums = [(k, r) for k, r in props.items() if isinstance(r, dict) and isinstance(r.get("enum"), list)]
        sort_key, sort_value = None, None
        for key, rule in enums:
            if re.search(r"sort|order|rank", flat(re.sub(r"[_\-]+", " ", key))):
                values = [v for v in rule["enum"] if isinstance(v, str)]
                cheap = next((v for v in values if re.search(r"price.{0,6}(asc|low|up)|(^|[^a-z])(low|cheap)", v.lower())), None)
                if cheap is not None:
                    sort_key, sort_value = key, cheap
        # what the search calls so far put in each argument teaches the roles a name does not tell: a several-word value
        # is the free text, a value that is a category label of the rows is the category, a figure near the limit the cap
        used = self.search_usage
        labels = {flat(r.get("group")) for r in self.shelf.rows.values() if r.get("group")}
        learnt_text = next((k for k, vs in used.items() if k in props and any(isinstance(v, str) and len(v.split()) >= 2 and flat(v) not in labels for v in vs)), None)
        learnt_group = next((k for k, vs in used.items() if k in props and any(isinstance(v, str) and flat(v) in labels for v in vs)), None)
        scale = self.shelf.scale
        learnt_cap = next((k for k, vs in used.items() if k in props and scale and any(isinstance(v, (int, float)) and not isinstance(v, bool) and 0.2 * scale <= v <= 3 * scale for v in vs)), None)
        group = first(r"categor|department|aisle|section|shelf|collection|dept", ("string",)) or learnt_group
        text = first(r"query|\bq\b|text|keyword|term|search|word", ("string",)) or (learnt_text if learnt_text != group else None)
        if text is None:
            # no telling name: the one free string argument that is not the category is the search text
            free = [k for k, r in props.items() if isinstance(r, dict) and "enum" not in r and r.get("type") == "string" and k != group]
            text = free[0] if len(free) == 1 else None
        cap = first(r"cap|ceiling|max|budget|limit|under|below|upto|price", ("number", "integer")) or learnt_cap
        page = first(r"page|offset", ("integer", "number"))
        if page is None:
            # no telling name: the one whole-number argument besides the cap moves through the pages
            whole = [k for k, r in props.items() if isinstance(r, dict) and r.get("type") == "integer" and "enum" not in r and k != cap]
            page = whole[0] if len(whole) == 1 else None
        if cap is None:
            figures = [k for k, r in props.items() if isinstance(r, dict) and r.get("type") in ("integer", "number") and "enum" not in r and k != page]
            cap = figures[0] if len(figures) == 1 and (props[figures[0]].get("type") == "number" or page is not None) else None
        return {
            "function": function,
            "required": set(params.get("required") or []),
            "cap": cap,
            "text": text,
            "group": group,
            "page": page,
            "enums": enums,
            "sort": (sort_key, sort_value),
        }

    def walk_args(self, below: float, group: str, page: int = 1) -> dict[str, Any] | None:
        """One search for the walk: the category alone (no free text), the price limit just under `below`, cheapest first
        when the store can sort, a filtering mode when the store has modes."""
        keys = self.search_keys()
        function, cap_key, text_key, group_key = keys["function"], keys["cap"], keys["text"], keys["group"]
        if not function or not cap_key or not group_key or not group:
            return None
        whole = "integer" in str(((function.get("parameters") or {}).get("properties") or {}).get(cap_key, {}).get("type"))
        limit: Any = (int(below) - 1 if float(below).is_integer() else int(below)) if whole else round(below - 0.01, 2)
        if limit <= 0:
            return None
        args: dict[str, Any] = {group_key: group, cap_key: limit}
        if text_key and text_key in keys["required"]:
            noun = " ".join(words(self.plan.get("product") or "")[-1:])
            if not noun:
                return None
            args[text_key] = noun
        for key, rule in keys["enums"]:
            if key == keys["sort"][0]:
                args[key] = keys["sort"][1]
                continue
            values = [v for v in rule["enum"] if isinstance(v, str)]
            if re.search(r"sort|order|rank", flat(re.sub(r"[_\-]+", " ", key))):
                continue
            mode = next((v for v in values if re.search(r"field|filter|facet|attrib|struct|browse|categor|shelf", v.lower())), None)
            if mode is not None:
                args[key] = mode
            elif key in keys["required"]:
                return None
        if page > 1 and keys["page"]:
            args[keys["page"]] = page
        args = shape_args(function, args)
        return None if lacking(function, args) else args

    def walk_search(self, args: dict[str, Any]) -> tuple[Any, bool]:
        """Run one walk search as its own store step; (observation, True) when it went through."""
        name = self.roles["search"]
        reply = self.env_call([{"name": name, "args": args}])
        self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": name, "args": args}], "reply": reply, "walk": True})
        if reply is None:
            return None, False
        observation = reply.get("observation")
        self.walking = True
        try:
            self.book_call(name, args, observation)
        finally:
            self.walking = False
        self.walk_used += 1
        return observation, True

    def walk_reads(self, keys: list[str]) -> int:
        """Open the sheets of a few listings for the walk, one per envelope; the steps used."""
        tool = self.sheet_tool()
        function = self.schemas_full.get(tool or "") or {}
        props = (function.get("parameters") or {}).get("properties") or {}
        id_key = next((k for k in (function.get("parameters") or {}).get("required") or [] if isinstance(props.get(k), dict) and "string" in str(props[k].get("type"))), None)
        id_key = id_key or next((k for k in props if re.search(IDDY, flat(re.sub(r"[_\-]+", " ", k)))), None)
        if not tool or not id_key or not keys:
            return 0
        used = 0
        for key in keys:
            actions = [{"name": tool, "args": shape_args(function, {id_key: key})}]
            reply = self.env_call(actions)
            self.trace.append({"kind": "env", "step": self.step, "actions": actions, "reply": reply, "walk": True})
            if reply is None:
                break
            used += 1
            self.walk_used += 1
            self.book_call(tool, actions[0]["args"], reply.get("observation"))
            if self.finished or self.dead:
                break
        return used

    def descend(self, pick: str) -> tuple[str, str]:
        """The price-limit walk, guarded: a fault inside it never ends the episode (the model simply goes on)."""
        try:
            return self.walk_down(pick)
        except Exception as exc:
            self.walking = False
            self.trace.append({"kind": "walk_fault", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return pick, ""

    def walk_scope(self) -> tuple[Any, int]:
        """The budget and quantity the walk judges prices against."""
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        quantity = self.plan.get("quantity") if isinstance(self.plan.get("quantity"), int) and self.plan.get("quantity") > 0 else 1
        return budget, quantity

    def page_size(self) -> int:
        """How many rows one result page carries, from the searches seen so far."""
        sizes = [len(s["shown"]) for s in self.searches if s.get("total") is not None and s["total"] > len(s["shown"]) and s["shown"]]
        return max(sizes) if sizes else 10

    def known_list(self, group: str, below: float) -> int | None:
        """How long the category list under the limit is, when a search already told: the smallest total reported by a
        search of this category with no words (or in a mode that ignores words) and a limit at or above `below`."""
        keys = self.search_keys()
        totals = []
        for s in self.searches:
            args, total = s["args"], s.get("total")
            if total is None or not keys["group"] or flat(str(args.get(keys["group"]) or "")) != flat(group):
                continue
            text = str(args.get(keys["text"]) or "").strip() if keys["text"] else ""
            if text and self.search_mode(args, keys) not in self.text_blind:
                continue
            cap = args.get(keys["cap"]) if keys["cap"] else None
            if isinstance(cap, (int, float)) and not isinstance(cap, bool) and cap < below - 1e-9:
                continue
            totals.append(total)
        return min(totals) if totals else None

    def keyword_search(self, below: float, group: str, any_group: bool = False, any_text: bool = False) -> dict[str, Any] | None:
        """The model's own most recent search with a requirement word in its text whose list is short enough to page to
        the end and not fully shown yet; None when there is none or the tool cannot page."""
        keys = self.search_keys()
        if not keys["page"] or not keys["text"]:
            return None
        wanted: set[str] = set(words(self.plan.get("product") or ""))
        for item in list(self.stated) + list(self.revealed):
            wanted |= set(words(str(item)))
        for s in reversed(self.searches):
            args, total = s["args"], s.get("total")
            text = str(args.get(keys["text"]) or "")
            if total is None or total > KEYWORD_LIST or total <= len(s["shown"]) or not text.strip():
                continue
            if args.get(keys["page"]) not in (None, 1) or self.search_mode(args, keys) in self.text_blind:
                continue
            if not any_text and self.shelf.kind_words and not any(w in wanted for w in words(text)):
                continue
            if not any_group and keys["group"] and args.get(keys["group"]) and flat(str(args[keys["group"]])) != flat(group):
                continue
            cap = args.get(keys["cap"]) if keys["cap"] else None
            if isinstance(cap, (int, float)) and not isinstance(cap, bool) and cap < below - 1e-9:
                continue
            return s
        return None

    def walk_down(self, pick: str) -> tuple[str, str]:
        """For a cheapest request, before any stockout, the code looks below the pick's price for a cheaper listing whose
        own sheet states every condition, within WALK_STEPS store steps for the whole episode. When the category list
        under the limit is short (or the store sorts by price) it searches the category alone under the current best
        price and opens the cheapest rows, again under each cheaper fit, until a search under the limit shows nothing in
        stock. Otherwise it pages the model's own keyword search to the end and opens the cheapest rows it shows.
        Returns the cheapest fit it knows and a note for the model ('' when the walk does not run)."""
        row = self.shelf.rows.get(pick) or {}
        price = row.get("price")
        conds = self.conditions()
        group = row.get("group") or ""
        if (self.plan.get("priority") != "cheapest" or self.items() or price is None or not conds or not group
                or f"{price:g}" in self.walked or self.cost >= self.cost_hard
                or not self.roles.get("search") or not (self.roles.get("details") or self.roles.get("view"))
                or self.steps_left() < WALK_FROM or self.time_left() < WRAP_UP_SECONDS + 45):
            return pick, ""
        reserve = 6 + self.held_back()
        allowance = min(WALK_STEPS - self.walk_used, self.steps_left() - reserve)
        if allowance < 2 or self.walk_args(price, group) is None:
            return pick, ""
        budget, quantity = self.walk_scope()
        best, floor, used, proven, path = pick, float(price), 0, False, [f"{pick} at {price:g}"]
        self.walked.add(f"{floor:g}")

        def same_group(r: dict[str, Any]) -> bool:
            return not r.get("group") or flat(r["group"]) == flat(group)

        def under_rows(shown: list[str], limit: float) -> list[str]:
            return [k for k in dict.fromkeys(shown) if (self.shelf.rows.get(k) or {}).get("price") is not None
                    and self.shelf.rows[k]["price"] < limit - 1e-9 and self.shelf.rows[k].get("stock") is not False
                    and k not in self.gone and same_group(self.shelf.rows[k])
                    and (budget is None or self.shelf.rows[k]["price"] * quantity <= budget + 1e-9)]

        def read_cheapest(under: list[str], limit: float) -> tuple[str, bool]:
            """Open the cheapest unread rows under the limit until one fits or the steps run out; (fit, all read)."""
            nonlocal used
            unread = sorted((k for k in under if not self.shelf.rows[k].get("sheet")), key=lambda k: (self.shelf.rows[k]["price"], k))
            while True:
                fits = [k for k in self.shelf.fits(conds, budget, quantity, group, self.gone) if self.shelf.rows[k]["price"] < limit - 1e-9]
                if fits:
                    return fits[0], not unread
                if not unread or used >= allowance or self.time_left() < WRAP_UP_SECONDS + 30:
                    return "", not unread
                take = unread[: min(2, allowance - used)]
                unread = unread[len(take):]
                got = self.walk_reads(take)
                used += got
                if not got or self.finished or self.dead:
                    return "", False

        sorted_store = self.search_keys()["sort"][0] is not None
        listed_total = self.known_list(group, floor)
        small = sorted_store or (listed_total is not None and listed_total <= WALK_PAGES * self.page_size())
        mode = "walk" if small or (listed_total is None and self.keyword_search(floor, group) is None) else "pages"
        if mode == "walk":
            while used < allowance and not proven:
                self.walked.add(f"{floor:g}")
                args = self.walk_args(floor, group)
                if args is None or self.time_left() < WRAP_UP_SECONDS + 30:
                    break
                observation, ok = self.walk_search(args)
                if not ok or self.finished or self.dead:
                    return best, ""
                used += 1
                listed = [k for k, r in self.shelf.rows.items() if r.get("step") == self.step]
                total = self.match_count(observation)
                complete = total is None or len(set(listed)) >= total
                if not sorted_store and total is not None and total > WALK_PAGES * max(len(listed), 1) and listed_total is None:
                    # a long unsorted list: the cheapest rows are anywhere in it, so page the keyword search instead
                    mode = "pages"
                    break
                under = under_rows(listed, floor)
                if not under:
                    proven = complete
                    break
                found, all_read = read_cheapest(under, floor)
                if not found:
                    proven = complete and all_read
                    break
                best, floor = found, float(self.shelf.rows[found]["price"])
                path.append(f"{found} at {floor:g}")
        if mode == "pages" and used < allowance:
            source = self.keyword_search(floor, group)
            if source is not None:
                keys = self.search_keys()
                shown = list(source["shown"])
                total, page, ok = source["total"], 2, True
                while len(set(shown)) < total and used < allowance and self.time_left() >= WRAP_UP_SECONDS + 30:
                    args = shape_args(keys["function"], {**source["args"], keys["page"]: page})
                    if lacking(keys["function"], args):
                        break
                    observation, ok = self.walk_search(args)
                    if not ok or self.finished or self.dead:
                        return best, ""
                    used += 1
                    listed = [k for k, r in self.shelf.rows.items() if r.get("step") == self.step]
                    if not listed:
                        break
                    shown += listed
                    page += 1
                paged = len(set(shown)) >= total
                found, all_read = read_cheapest(under_rows(shown, floor), floor)
                if found:
                    best, floor = found, float(self.shelf.rows[found]["price"])
                    path.append(f"{found} at {floor:g}")
                proven = paged and all_read and not found
        self.walked.add(f"{floor:g}")
        self.trace.append({"kind": "walk", "pick": pick, "best": best, "steps": used, "proven": proven, "mode": mode, "path": path})
        head = f"[price check run by the code: {used} store step(s); {' -> '.join(path)}] "
        if best != pick:
            tail = (f"Nothing else in stock under {floor:g} showed, so {best} is the cheapest listing whose own sheet states every condition."
                    if proven else f"{best} is the cheapest listing found whose own sheet states every condition (the check stopped before proving nothing cheaper exists).")
            note = head + tail
        elif proven:
            note = head + f"Nothing in stock under {floor:g} states every condition, so {pick} is the cheapest fit."
        else:
            note = head + f"No cheaper listing that states every condition was found below {pick} ({floor:g})."
        self.walk_log.append(note)
        return best, note

    def match_count(self, observation: Any) -> int | None:
        """How many listings a search says matched (a count field next to the rows), when it says so."""
        if not isinstance(observation, dict):
            return None
        named = lambda key: re.search(r"match|total|count|hits|found|results|number", flat(re.sub(r"[_\-]+", " ", str(key)))) is not None
        rows = max((len(v) for v in observation.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), default=0)
        if rows:
            # the count is a whole number beside the rows and at least as large as the rows shown (a page cursor is smaller)
            counts = [(not named(k), -v, k) for k, v in observation.items() if isinstance(v, int) and not isinstance(v, bool) and v >= rows]
            if counts:
                return observation[min(counts)[2]]
        for key, value in observation.items():
            if isinstance(value, int) and not isinstance(value, bool) and named(key):
                return value
        if len(observation) <= 3:
            inner = [v for v in observation.values() if isinstance(v, dict)]
            if len(inner) == 1:
                return self.match_count(inner[0])      # the result inside a note wrapper
        return None

    def fitting_rows(self) -> list[tuple[float, str]]:
        """Listings whose own sheets state every condition (or that an item check passed), not reported gone, not shown
        out of stock and within the limit; cheapest first."""
        budget, quantity = self.walk_scope()
        pool = set(self.passed)
        if self.conditions() and not self.items():
            pool |= set(self.shelf.fits(self.conditions(), budget, quantity, "", self.gone))
        ranked = []
        for key in pool:
            row = self.shelf.rows.get(key) or {}
            if key in self.gone or row.get("price") is None or row.get("stock") is False:
                continue
            if budget is not None and row["price"] * quantity > budget + 1e-9:
                continue
            ranked.append((row["price"], key))
        return sorted(ranked)

    def verified_cheaper(self, target: str) -> str:
        """The cheapest listing an item check passed that is cheaper than the target and still on offer; '' when none."""
        price = (self.shelf.rows.get(target) or {}).get("price")
        if price is None:
            return ""
        for p, key in self.fitting_rows():
            if key in self.passed and key != target and p < price - 1e-9:
                return key
        return ""

    def fallback_note(self) -> str:
        ranked = self.fitting_rows()[:3]
        if not ranked:
            return ""
        listed = ", ".join(f"{k} ({p:g}{', checked' if k in self.passed else ''})" for p, k in ranked)
        return (f"Cheapest listings whose own records state every condition and are not reported gone: {listed}. Put the cheapest "
                "in the cart in place of the gone line unless a reason its own records state rules it out.")

    def blind_hold(self, args: dict[str, Any]) -> str | None:
        """A search in a mode the store has shown to ignore the words is refused once (no step): the model is pointed to
        the keyword mode and its pages, or a lower limit."""
        keys = self.search_keys()
        text_key = keys["text"]
        if not text_key or not isinstance(args.get(text_key), str) or not args[text_key].strip() or text_key in keys["required"]:
            return None
        mode = self.search_mode(args, keys)
        if mode not in self.text_blind:
            return None
        mark = mode + "|" + flat(args[text_key])
        if mark in self.blind_refused:
            return None
        self.blind_refused.add(mark)
        self.trace.append({"kind": "hygiene", "blind_refused": args[text_key], "mode": mode})
        filtered = bool(keys["group"] and args.get(keys["group"]))
        route = ("Send the same words again WITHOUT the category filter in this mode, or use the keyword mode and read its next pages"
                 if filtered else "Use the keyword mode with the requirement words and read its next pages")
        return ("Not run (no step used): in this search mode the store ignores the words, so this call would return the same long category "
                f"list as before. {route}; a lower price limit also narrows the list. If you still want the list, send the same call again.")

    def zero_hold(self, args: dict[str, Any]) -> str | None:
        """Two kinds of wordy search are held without a step: one made only of words that describe a page a pointer
        named (a seller, its details), and, after ZERO_SEARCHES empty searches, one that repeats the words of an empty
        search or adds brand-like words no row or condition carries (at most ZERO_HOLDS times)."""
        keys = self.search_keys()
        text_key = keys["text"]
        if not text_key or not isinstance(args.get(text_key), str) or not args[text_key].strip():
            return None
        asked = set(words(args[text_key]))
        describe = self.pointer_words()
        if asked and describe and asked <= describe and not any(not p["done"] and not p["failed"] for p in self.pointers):
            self.trace.append({"kind": "hold", "pointer_words": args[text_key]})
            return ("Not run (no step used): those words name the page the shopper pointed at, not the product. The code has read that page "
                    "(see the requirements): search for the product with the settled value and the other requirement words.")
        if self.zero_searches < ZERO_SEARCHES or self.zero_holds >= ZERO_HOLDS:
            return None
        known = set(words(self.plan.get("product") or ""))
        for prop, value in self.conditions():
            known |= set(words(prop)) | set(words(value))
        for row in self.shelf.rows.values():
            known |= set(words(row.get("group") or "")) | set(words(row.get("title") or ""))
        strange = [w for w in asked if w not in known and len(w) >= 3]
        repeated = any(asked <= z or (z and z <= asked and len(asked) - len(z) <= 1) for z in self.zero_words)
        empty_words = set().union(*self.zero_words) if self.zero_words else set()
        if (asked - empty_words) & known:
            return None                      # it adds a product, condition or category word no empty search carried: it runs
        if not repeated or not strange and len(self.zero_words) < 2:
            return None
        self.zero_holds += 1
        self.trace.append({"kind": "hold", "zero_search": args[text_key], "strange": strange[:4]})
        return (f"Not run (no step used): {self.zero_searches} searches returned nothing already, and this one "
                + (f"uses words no listing seen carries ({', '.join(strange[:4])})" if strange else "repeats the words of an empty search")
                + ". Use the category filter with the price limit (no free text, or only the product noun), open the cheapest unread rows within the limit "
                "and judge them by their sheets; a brand or page name is never a requirement unless the shopper named it as one.")

    def search_hygiene(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A model search tidied by the code: words that neither the request, the conditions, the product, a category name
        seen nor the title of a fitting listing contain are taken out (never all of them)."""
        keys = self.search_keys()
        text_key = keys["text"]
        if not text_key or not isinstance(args.get(text_key), str):
            return args
        known = set(words(str(self.policy.get("query") or ""))) | set(words(self.plan.get("product") or ""))
        for item in list(self.stated) + list(self.revealed):
            known |= set(words(str(item)))
        for row in self.shelf.rows.values():
            known |= set(words(row.get("group") or ""))
        for _, key in self.fitting_rows():
            known |= set(words((self.shelf.rows.get(key) or {}).get("title") or ""))
        # the names and details of a page a pointer named describe that page, not the product
        known -= self.pointer_words() - set(words(self.plan.get("product") or ""))
        stem = lambda w: w[:5] if len(w) >= 5 else w
        stems = {stem(w) for w in known}
        tokens = str(args[text_key]).split()
        # a fitment code never goes into a search (the pages carry it, not the titles), unless a title seen carries it
        titles = " ".join((r.get("title") or "") for r in self.shelf.rows.values())
        coded = [t for t in tokens if any(code_covers(c, f) for c in codes_in(t.upper()) for f in self.fit_targets()) and not any(c in titles.upper() for c in codes_in(t.upper()))]
        if coded and len(coded) < len(tokens):
            tokens = [t for t in tokens if t not in coded]
            args = {**args, text_key: " ".join(tokens)}
            self.trace.append({"kind": "hygiene", "code_out": coded})
        elif coded:
            product = " ".join(words(self.plan.get("product") or ""))[:60]
            if product:
                args = {**args, text_key: product}
                tokens = product.split()
        kept = [t for t in tokens if not words(t) or all(w in known or stem(w) in stems for w in words(t))]
        if kept and len(kept) < len(tokens):
            self.trace.append({"kind": "hygiene", "from": args[text_key], "to": " ".join(kept)})
            return {**args, text_key: " ".join(kept)}
        return args

    def search_mode(self, args: dict[str, Any], keys: dict[str, Any] | None = None) -> str:
        keys = keys or self.search_keys()
        return _compact({k: args.get(k) for k, _ in keys["enums"] if k in args})

    def note_search(self, args: dict[str, Any], observation: Any) -> None:
        """Remember the model's search (its total and the rows shown) and watch whether free text changes a category
        search's result: two different texts under the same mode, category and limit that return the same count and the
        same first rows mark the mode as one that ignores the text."""
        keys = self.search_keys()
        text_key, group_key = keys["text"], keys["group"]
        total = self.match_count(observation)
        shown = [k for k, r in self.shelf.rows.items() if r.get("step") == self.step]
        self.searches.append({"args": dict(args), "total": total, "shown": shown, "step": self.step, "last": last_page(observation)})
        del self.searches[:-40]
        if text_key and isinstance(args.get(text_key), str) and args[text_key].strip() and (total == 0 or (total is None and not shown)):
            self.zero_searches += 1
            self.zero_words.append(set(words(args[text_key])))
            self.trace.append({"kind": "zero_search", "text": args[text_key], "count": self.zero_searches})
        if not text_key or not group_key or not args.get(group_key) or not isinstance(args.get(text_key), str) or not args[text_key].strip():
            return
        mode = self.search_mode(args, keys)
        broad = mode + "|" + flat(str(args[group_key]))
        counted = self.text_totals.setdefault(broad, {})
        text = flat(args[text_key])
        if total is not None and total >= BLIND_TOTAL and any(t != text and n == total for t, n in counted.items()) and mode not in self.text_blind:
            self.text_blind.add(mode)
            self.trace.append({"kind": "hygiene", "blind_mode": mode, "slot": broad, "by": "totals"})
        if total is not None:
            counted[text] = total
        if args.get(keys["page"] or "") not in (None, 1) or not isinstance(observation, dict):
            return
        firsts: list[str] = []
        stack = [observation]
        while stack and len(firsts) < 4:
            value = stack.pop(0)
            if isinstance(value, dict):
                key = self.shelf.id_field(value)
                if isinstance(key, str) and key.strip() and " " not in key.strip():
                    firsts.append(key)
                stack.extend(v for v in value.values() if isinstance(v, (dict, list)))
            elif isinstance(value, list):
                stack.extend(value[:20])
        if total is None or not firsts:
            return
        mode = self.search_mode(args, keys)
        slot = mode + "|" + flat(str(args[group_key])) + "|" + str(args.get(keys["cap"] or ""))
        sign = f"{total}|{','.join(firsts)}"
        tries = self.text_tries.setdefault(slot, {})
        text = flat(args[text_key])
        if any(t != text and s == sign for t, s in tries.items()) and total > 3:
            self.text_blind.add(mode)
            self.trace.append({"kind": "hygiene", "blind_mode": mode, "slot": slot})
        tries[text] = sign
    def conditions(self) -> list[tuple[str, str]]:
        """The requirements in force as (property, value) pairs for the shelf's sheet reading."""
        out = []
        for item in list(self.stated) + list(self.revealed):
            if item in self.dropped or self.is_dropped(item):
                continue
            prop, sep, value = str(item).partition(":")
            if _POINTER.search(value):
                continue                     # a need that points at another page is settled by the code first (see the pointers)
            value = re.sub(r"\s*\(([^)]*)\)", lambda m: (" or " + m.group(1)) if 0 < len(m.group(1).split()) <= 3 and not re.search(r"\bi\b|check|^\s*(?:no|not)\b", m.group(1).lower()) else "", value).strip(" .").lower()
            if sep and prop.strip() and value and not re.search(r"price|budget|stock|categor|quantity|deliver", prop.lower()):
                out.append((prop.strip().lower(), value))
        return out

    def is_dropped(self, item: str) -> bool:
        """Whether a requirement was given up later: a dropped entry names the same property (by its words). A requirement
        the shopper states after the drop stands."""
        prop = set(words(str(item).partition(":")[0])) - {"type", "kind", "the", "of"}
        if not prop:
            return False
        for gone in self.dropped:
            gp = set(words(str(gone).partition(":")[0] if ":" in str(gone) else str(gone))) - {"type", "kind", "the", "of", "requirement", "need"}
            if gp and (gp <= prop or prop <= gp):
                return True
        return False

    def drop_needs(self, values: list[Any], line: str) -> list[str]:
        """Requirements a note or a reply gives up: each lands in the dropped list once; the ledger moves and the code
        re-ranks the opened sheets."""
        out = []
        for value in values or []:
            text = str(value).strip() if isinstance(value, (str, int, float)) else ""
            if not text or text in self.dropped:
                continue
            self.dropped.append(text)
            out.append(text)
        if out:
            self.relaxed_at = self.step
            self.ledger_version += 1
            self.mark_change()
            self.trace.append({"kind": "relaxed", "dropped": out, "line": line[:160]})
        return out

    # ------------------------------------------------------------------ needs that point at another listing's page

    def take_pointers(self, items: list[str]) -> list[str]:
        """Needs among `items` ("property: constraint") whose constraint points at another listing's page become pointers
        the code follows; the stated conditions that merely describe that other listing (its name, its details) are set
        aside, since they are not requirements of the product. The pointer sources are returned."""
        found: list[str] = []
        for item in list(items):
            prop, sep, value = str(item).partition(":")
            if not sep or not prop.strip():
                continue
            if any(p["source"] == item or (flat(p["prop"]) == flat(prop) and not p["done"]) for p in self.pointers):
                found.append(item)
                continue
            pointer = pointer_in(prop.strip(), value.strip())
            if pointer:
                pointer["source"] = item
                self.pointers.append(pointer)
                found.append(item)
                self.trace.append({"kind": "pointer", "prop": pointer["prop"], "relation": pointer["relation"], "who": pointer["who"], "attrs": pointer["attrs"]})
        if found:
            self.unmix(found)
        return found

    def pointer_words(self) -> set[str]:
        """Words that describe the pages the pointers name (names, kinds, details), never the product wanted."""
        out: set[str] = set()
        for p in self.pointers:
            for text in [p["who"]] + list(p.get("legs") or []) + list(p["attrs"]):
                out |= set(words(text))
        return out

    def unmix(self, sources: list[str]) -> None:
        """Stated or revealed conditions whose words all describe a pointed-at page (its name, its details) are no
        requirements of the product: they are dropped from the ledger."""
        describe = self.pointer_words()
        if not describe:
            return
        for store in (self.stated, self.revealed):
            for item in list(store):
                if item in sources:
                    continue
                prop, sep, value = str(item).partition(":")
                if not sep:
                    continue
                vw = set(words(value))
                if vw and vw <= describe and item not in self.dropped:
                    store.remove(item)
                    self.trace.append({"kind": "pointer", "unmixed": item})
        if self.current is not None:
            kept = []
            for item in self.current:
                prop, sep, value = str(item).partition(":")
                vw = set(words(value)) if sep else set()
                if sep and vw and vw <= describe and not _POINTER.search(value):
                    continue
                kept.append(item)
            self.current = kept

    def groom_current(self) -> None:
        """The hearing's own requirement list, cleaned by what the code knows about pointers: an item whose value only
        describes a pointed-at page (its seller, its details) is no requirement of the product; an item still written as a
        pointer takes the settled value once the code has read it."""
        if self.current is None or not self.pointers:
            return
        describe = self.pointer_words()
        sources = {flat(str(p["source"])) for p in self.pointers if p.get("source")}
        stated = {flat(str(x)) for x in self.stated}       # a requirement the request itself states is never groomed away
        props = [set(words(p["prop"])) for p in self.pointers]
        kept: list[str] = []
        for item in self.current:
            prop, sep, value = str(item).partition(":")
            vw = set(words(value)) if sep else set()
            pw = set(words(prop))
            from_pointer = flat(str(item)) in sources or any(pp and (pp <= pw or pw <= pp) for pp in props)
            if sep and vw and vw <= describe and not _POINTER.search(value) and (from_pointer or flat(str(item)) not in stated):
                continue                     # a pointer's own wording (the page's seller or details), not a requirement of the product
            if sep and _POINTER.search(value):
                pw = set(words(prop))
                done = next((p for p in self.pointers if p["done"] and (set(words(p["prop"])) <= pw or pw <= set(words(p["prop"])))), None)
                if done:
                    item = f"{prop.strip()}: {done['value']}"
            if item not in kept:
                kept.append(item)
        self.current = kept

    def open_pointers(self) -> list[dict[str, Any]]:
        return [p for p in self.pointers if not p["done"] and not p["failed"]]

    def pointer_note(self) -> str:
        """The status line on pointers: what the code is settling, or what it settled."""
        lines = []
        for p in self.pointers:
            if p["done"]:
                lines.append(f"[pointer settled by the code] the page the shopper pointed at ({p['page']}) gives {p['prop']} = {p.get('seen') or '?'}; "
                             f"so the requirement is '{p['prop']}: {p['value']}'. Search with that value, not with the words {p['who']} or {', '.join(p['attrs'][:2])}.")
            elif p["failed"]:
                lines.append(f"[pointer] the shopper's need for {p['prop']} refers to the value on the page of the {p['kind']} from {p['who']}"
                             + (" that comes with " + " and ".join(p["attrs"]) if p["attrs"] else "") + "; the code could not find that page. "
                             f"Judge {p['prop']} from any listing of theirs you can open; {p['who']} is NOT a requirement of the product.")
            else:
                lines.append(f"[pointer: being settled by the code] the need for {p['prop']} is the value on the page of the {p['kind']} from {p['who']}"
                             + (" that comes with " + " and ".join(p["attrs"]) if p["attrs"] else "") + f" ({p['relation']}). Do not search for {p['who']} yourself and do not treat "
                             f"{p['who']} or those details as requirements; carry on with the other conditions.")
        return "\n".join(lines)

    def settle_pointer(self, pointer: dict[str, Any], page: str, seen: str, value: str) -> None:
        """The pointer becomes a plain requirement on the ledger (stated or revealed, wherever its source sat)."""
        pointer.update({"done": True, "page": page, "seen": seen, "value": value})
        prop = pointer["prop"]
        if self.items():
            # with several products the settled need names its product, or it would bind every product
            names = self.product_names()
            source_prop = set(words(str(pointer["source"]).partition(":")[0])) | set(words(pointer.get("kind") or ""))
            if not any(self.product_words(n) and self.product_words(n) <= set(words(prop)) for n in names):
                owner = max(names, key=lambda n: len(self.product_words(n) & source_prop), default="")
                if owner and self.product_words(owner) & source_prop:
                    prop = f"{owner} {prop}"
        requirement = f"{prop}: {value}"
        placed = False
        names = [flat(n) for n in [pointer["who"]] + list(pointer.get("legs") or []) if n]

        def restates(item: Any) -> bool:
            # another wording of the same pointer (the reader's paraphrase): same property, naming the page or its seller
            prop, _, rest = str(item).partition(":")
            return flat(prop) == flat(pointer["prop"]) and (bool(re.search(r"\b(?:page|listing)\b", rest, re.I)) or any(n and n in flat(rest) for n in names))

        for store in (self.stated, self.revealed):
            kept = []
            for item in store:
                if item == pointer["source"] or restates(item) or (_POINTER.search(str(item)) and flat(str(item).partition(":")[0]) == flat(pointer["prop"])):
                    if not placed:
                        kept.append(requirement)
                        placed = True
                    continue
                kept.append(item)
            store[:] = kept
        if not placed:
            self.revealed.append(requirement)
        if self.current is not None:
            self.current = [requirement if (_POINTER.search(str(c)) and flat(str(c).partition(":")[0]) == flat(pointer["prop"])) or restates(c) else c for c in self.current]
            self.current = [c for i, c in enumerate(self.current) if c not in self.current[:i]]
            if requirement not in self.current:
                self.current.append(requirement)
        self.ledger_version += 1
        self.mark_change()
        self.trace.append({"kind": "pointer", "settled": requirement, "page": page, "seen": seen})

    @staticmethod
    def bounded(pointer: dict[str, Any], seen: str) -> str:
        """The requirement a value read off the pointed-at page makes under the pointer's relation: the value itself, at
        least or at most it, or the range within the stated distance of it (the figure of the page's value, its unit kept)."""
        relation = pointer.get("relation")
        figure = re.search(r"(-?\d+(?:\.\d+)?)\s*(.*)$", str(seen).strip())
        if relation == "min":
            return f"at least {seen}"
        if relation == "max":
            return f"at most {seen}"
        if relation == "near" and figure:
            by = float((pointer.get("bound") or {}).get("by") or 0)
            centre, unit = float(figure.group(1)), figure.group(2).strip() or str((pointer.get("bound") or {}).get("unit") or "")
            low, high = round(centre - by, 4), round(centre + by, 4)
            return f"between {low:g} and {high:g} {unit}".strip()
        return seen

    def leg_attrs(self, pointer: dict[str, Any], who: str) -> list[str]:
        """The details that pick out the page of `who`: the shopper's own for the first leg, the sending page's for a
        later one (an onward leg without details is picked out by the property's presence alone)."""
        if who == pointer["legs"][0]:
            return list(pointer["attrs"])
        return list((pointer.get("leg_details") or {}).get(who) or [])

    def pointer_rows(self, pointer: dict[str, Any], who: str) -> list[str]:
        """Listings seen that `who` offers, those whose sheets state the pointer's details first, then unopened ones."""
        rows = [k for k in self.shelf.rows if self.shelf.offered_by(k, who)]
        attrs = self.leg_attrs(pointer, who)
        def rank(k: str) -> tuple[int, int, float]:
            sheet = bool((self.shelf.rows.get(k) or {}).get("sheet"))
            hits = sum(self.shelf.phrase_fits(k, a) for a in attrs) if sheet else 0
            title = set(words((self.shelf.rows.get(k) or {}).get("title") or ""))
            title_hits = sum(bool(set(words(a)) & title) for a in attrs)
            return (-hits, -title_hits, (self.shelf.rows.get(k) or {}).get("price") or 0)
        return sorted(rows, key=rank)

    def pointer_page(self, pointer: dict[str, Any], who: str) -> str:
        """The opened listing of `who` whose sheet states every detail the pointer names (the page it means), or ''."""
        attrs = self.leg_attrs(pointer, who)
        for key in self.pointer_rows(pointer, who):
            row = self.shelf.rows.get(key) or {}
            if row.get("sheet") and all(self.shelf.phrase_fits(key, a) for a in attrs) and (attrs or self.shelf.label_value(key, pointer["prop"])):
                return key
        if who != pointer["legs"][0] and attrs:
            # an onward leg's details come from a free-text remark and may not read cleanly: the opened page of that seller
            # that carries the property and states the most of them stands for it
            best, at = 0, ""
            for key in self.pointer_rows(pointer, who):
                row = self.shelf.rows.get(key) or {}
                if row.get("sheet") and self.shelf.label_value(key, pointer["prop"]):
                    hits = sum(1 for a in attrs if self.shelf.phrase_fits(key, a))
                    if hits > best or not at:
                        best, at = hits, key
            return at
        return ""

    def sends_elsewhere(self, page: str, pointer: dict[str, Any]) -> str:
        """The seller a page's own remark sends this pointer's property to ("Gender should be taken from a different
        product, the Lukorazen sneakers ..."), or ''."""
        prop_words = set(words(pointer["prop"]))
        for sentence in re.split(r"(?<=[.!?;])\s+", self.shelf.page_text(page)):
            if prop_words and prop_words <= set(words(sentence)):
                onward = referral_in(sentence, pointer["legs"][-1], pointer["kind"])
                if onward:
                    return onward
        return ""

    def code_search(self, args: dict[str, Any]) -> bool:
        """One search of the code's own (a store step), booked like the model's."""
        name = self.roles.get("search")
        if not name:
            return False
        reply = self.env_call([{"name": name, "args": args}])
        self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": name, "args": args}], "reply": reply, "pointer": True})
        if reply is None:
            return False
        self.walking = True
        try:
            self.book_call(name, args, reply.get("observation"))
        finally:
            self.walking = False
        self.read_notices(name, reply.get("observation"))
        self.note_observation(name, reply.get("observation"))
        return True

    def code_reads(self, keys: list[str]) -> int:
        """The code opens a few sheets itself, one envelope each; the steps used."""
        tool = self.sheet_tool()
        used = 0
        for key in keys:
            args = self.id_args(tool, key) if tool else None
            if not args or self.finished or self.dead or self.steps_left() < self.finish_reserve() + 2:
                break
            reply = self.env_call([{"name": tool, "args": args}])
            self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": args}], "reply": reply, "pointer": True})
            if reply is None:
                break
            used += 1
            self.opened.add(key)
            self.book_call(tool, args, reply.get("observation"))
            self.read_notices(tool, reply.get("observation"))
            self.note_observation(tool, reply.get("observation"))
        return used

    def text_search_args(self, text: str) -> dict[str, Any] | None:
        """A plain keyword search (no category, no limit) in the store's keyword mode, when the tool allows one."""
        keys = self.search_keys()
        function, text_key = keys["function"], keys["text"]
        if function and not text_key:
            # the only free string the tool takes carries the words, even when the searches so far put a shelf label in it
            props = (function.get("parameters") or {}).get("properties") or {}
            free = [k for k, r in props.items() if isinstance(r, dict) and "enum" not in r and r.get("type") == "string"]
            text_key = free[0] if len(free) == 1 else None
        if not function or not text_key or not text.strip():
            return None
        args: dict[str, Any] = {text_key: text.strip()[:80]}
        for key, rule in keys["enums"]:
            if key == keys["sort"][0] or re.search(r"sort|order|rank", flat(re.sub(r"[_\-]+", " ", key))):
                continue
            values = [v for v in rule["enum"] if isinstance(v, str)]
            mode = next((v for v in values if re.search(r"text|keyword|typed|free|word|query|search", v.lower())), None)
            if mode is not None:
                args[key] = mode
            elif key in keys["required"]:
                return None
        args = shape_args(function, args)
        return None if lacking(function, args) else args

    def resolve_pointers(self) -> bool:
        """One round of following the first open pointer, by the code: search for the named seller's listings of that
        kind (twice at most), open up to POINTER_READS of their sheets, and when the page that states the details is
        found read the property's value there. 'same' keeps that value, 'exclude' forbids it, 'paired' looks on the page
        (or, when it may point further, on the page of the seller it names) for the rule that pairs values and takes
        the partner. The settled value becomes a plain requirement. True when store steps were spent."""
        pointer = next(iter(self.open_pointers()), None)
        if (pointer is None or self.order_attempts or self.pointer_steps >= POINTER_STEPS or not self.roles.get("search")
                or not (self.roles.get("details") or self.roles.get("view")) or self.steps_left() < self.finish_reserve() + 6
                or self.time_left() < WRAP_UP_SECONDS + 40 or self.cost >= self.cost_hard):
            return False
        try:
            return self.pointer_round(pointer)
        except Exception as exc:
            self.walking = False
            pointer["failed"] = True
            self.trace.append({"kind": "pointer", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return False

    def pointer_round(self, pointer: dict[str, Any]) -> bool:
        who = pointer["legs"][-1]
        before = self.step
        page = self.pointer_page(pointer, who)
        if not page:
            unread = [k for k in self.pointer_rows(pointer, who) if not (self.shelf.rows.get(k) or {}).get("sheet") and k not in pointer["opened"]]
            if not unread and pointer["searched"] < 2:
                noun = " ".join(words(pointer["kind"])[-2:])
                text = f"{who} {noun}".strip() if pointer["searched"] == 0 else who
                args = self.text_search_args(text)
                pointer["searched"] += 1
                if args and self.code_search(args):
                    self.pointer_steps += 1
                    unread = [k for k in self.pointer_rows(pointer, who) if not (self.shelf.rows.get(k) or {}).get("sheet") and k not in pointer["opened"]]
                    if not unread and pointer["searched"] == 1 and self.pointer_steps < POINTER_STEPS:
                        # the rows of the seller may be hidden among others: the second search names the seller alone
                        args = self.text_search_args(who)
                        pointer["searched"] += 1
                        if args and self.code_search(args):
                            self.pointer_steps += 1
                            unread = [k for k in self.pointer_rows(pointer, who) if not (self.shelf.rows.get(k) or {}).get("sheet") and k not in pointer["opened"]]
            if unread and not self.finished and not self.dead:
                take = unread[: min(POINTER_READS, POINTER_STEPS - self.pointer_steps)]
                if take:
                    got = self.code_reads(take)
                    self.pointer_steps += got
                    pointer["opened"] += take[:got]
                page = self.pointer_page(pointer, who)
        if page:
            seen = self.shelf.label_value(page, pointer["prop"])
            elsewhere = self.sends_elsewhere(page, pointer)
            if elsewhere and elsewhere not in pointer["legs"] and len(pointer["legs"]) < 3:
                # the page says this property is to be taken from another product: that page decides, not this one
                pointer["legs"].append(elsewhere)
                pointer.setdefault("leg_details", {})[elsewhere] = referral_details(self.shelf.page_text(page), elsewhere)
                pointer["searched"] = 0
                self.trace.append({"kind": "pointer", "onward": elsewhere, "from": page, "redirect": True, "details": pointer["leg_details"][elsewhere]})
                seen = ""
                page = ""
            if seen:
                if pointer["relation"] == "exclude":
                    self.settle_pointer(pointer, page, seen, f"not {seen}")
                elif pointer["relation"] == "paired":
                    rule = pairing_rule(self.shelf.page_text(page))
                    partner = next((v for k, v in rule.items() if set(words(k)) == set(words(seen))), "")
                    if not partner:
                        # the page writes its own value shorter or longer than the rule does ("10" against "10 hangers"):
                        # one rule entry whose words hold the value's (or the other way round) decides
                        near = [v for k, v in rule.items() if k != "*" and set(words(seen)) and (set(words(k)) <= set(words(seen)) or set(words(seen)) <= set(words(k)))]
                        partner = near[0] if len(near) == 1 else rule.get("*", "")
                    if partner:
                        self.settle_pointer(pointer, page, seen, partner)
                    else:
                        onward = referral_in(self.shelf.page_text(page), who, pointer["kind"]) if pointer["chain"] else ""
                        if onward and onward not in pointer["legs"] and len(pointer["legs"]) < 3:
                            pointer["legs"].append(onward)
                            pointer.setdefault("leg_details", {})[onward] = referral_details(self.shelf.page_text(page), onward)
                            pointer["searched"] = 0
                            pointer["seen_first"] = seen
                            self.trace.append({"kind": "pointer", "onward": onward, "from": page})
                        else:
                            # no rule anywhere: the value itself is the best reading of "a type that goes with it" (the
                            # page's own, or for "the one the page points to or states" the page it sent on to)
                            self.settle_pointer(pointer, page, seen, seen if pointer.get("follow") else (pointer.get("seen_first") or seen))
                else:
                    self.settle_pointer(pointer, page, seen, self.bounded(pointer, seen))
            elif not page:
                pass
            elif pointer["chain"]:
                onward = referral_in(self.shelf.page_text(page), who, pointer["kind"])
                if onward and onward not in pointer["legs"] and len(pointer["legs"]) < 3:
                    pointer["legs"].append(onward)
                    pointer.setdefault("leg_details", {})[onward] = referral_details(self.shelf.page_text(page), onward)
                    pointer["searched"] = 0
                    self.trace.append({"kind": "pointer", "onward": onward, "from": page, "details": pointer["leg_details"][onward]})
                else:
                    pointer["failed"] = True
            else:
                pointer["failed"] = True
        elif pointer["searched"] >= 2 and not [k for k in self.pointer_rows(pointer, who) if not (self.shelf.rows.get(k) or {}).get("sheet") and k not in pointer["opened"]]:
            pointer["failed"] = True
        if self.pointer_steps >= POINTER_STEPS and not pointer["done"]:
            pointer["failed"] = True
        spent = self.step - before
        if spent or pointer["done"] or pointer["failed"]:
            self.messages.append({"role": "user", "content": f"[note] The code followed the shopper's pointer for {pointer['prop']} ({spent} store step(s); the store's replies are booked). "
                                                             + self.pointer_note() + "\n" + self.status()})
        return spent > 0

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:
        """Whether this finish call ends without buying. A store may use one tool for both finishes, told apart by an
        argument value (for example a mode of "refuse" against "commit")."""
        if name != self.roles.get("decline"):
            return False
        if self.roles.get("order") != name:
            return True
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        switches = [k for k, r in properties.items() if isinstance(r, dict) and (r.get("type") == "boolean" or isinstance(r.get("enum"), list))]
        if not switches:
            return not self.cart_ids           # one ending for both: with nothing held it ends without buying
        for key, value in args.items():
            spec = properties.get(key) or {}
            if spec.get("type") == "boolean" and value is False:
                return True
            if not isinstance(value, str) or not isinstance(spec.get("enum"), list):
                continue
            if _DECLINE_WORDS.search(value) or _NO_BUY.search(value):
                return True
            if not _BUY_WORDS.search(value) and any(isinstance(e, str) and _BUY_WORDS.search(e) for e in spec["enum"]):
                return True          # the mode beside the buying one ("pass" against "order")
        return False

    def audit_decline(self) -> str:
        if self.time_left() < WRAP_UP_SECONDS + 30 or self.cost > self.cost_hard:
            return ""
        self.audited = True
        blocks: list[str] = []
        total = 0
        for record in reversed(self.records):
            text = _compact(_brief(record, 160))
            if total + len(text) > 16000:
                continue
            blocks.append(text)
            total += len(text)
        if not blocks:
            return ""
        prompt = AUDIT_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(blocks)
        if prompt == self.audit_seen[0]:
            # nothing new since the last look: the same check gives the same answer, so it is not paid for twice
            return self.audit_seen[1]
        # the cheaper model looks first; when it names possible fits they are verified one by one anyway, so only its
        # "nothing fits" (which lets the decline stand) is confirmed by the full model
        result = self.judge(prompt, 1200, [READ_MODEL])
        fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        if not fits and not (self.past_soft() and result):
            result = self.judge(prompt, 1200)        # past the soft line a clear cheap "nothing fits" is not paid for twice
        self.trace.append({"kind": "audit", "result": result})
        fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        if not fits:
            self.audit_seen = (prompt, "")
            return ""
        text = "An independent check thinks these listings may meet every requirement: " + ", ".join(fits) + (f" ({result.get('reason')})" if result.get("reason") else "") + ". Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."
        if self.order_attempts:
            text += " Nothing has been bought yet: an order that goes through ends the episode, and this one is still running, so declining now would leave the shopper with nothing."
        self.audit_seen = (prompt, text)
        return text

    def neutral_args(self, name: str, text: str, declining: bool = False) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        properties = schema.get("properties") or {}
        args: dict[str, Any] = {}
        for key in schema.get("required") or []:
            spec = properties.get(key) or {}
            if "enum" in spec and spec["enum"]:
                # a finish tool shared by both finishes: pick the mode that matches the intent
                fitting = [v for v in spec["enum"] if isinstance(v, str) and bool(_DECLINE_WORDS.search(v) or _NO_BUY.search(v)) == declining]
                if not fitting and declining:
                    # no value reads as a decline: the one that does not read as buying (never the ordering value)
                    fitting = [v for v in spec["enum"] if isinstance(v, str) and not _BUY_WORDS.search(v)]
                args[key] = (fitting or spec["enum"])[0]
            elif spec.get("type") == "string" or "string" in (spec.get("type") or []):
                args[key] = text[: spec.get("maxLength", len(text))]
            elif spec.get("type") == "boolean":
                args[key] = not declining         # the switch says yes to buying, no to ending without a purchase
            elif spec.get("type") in ("integer", "number"):
                args[key] = max(1, spec.get("minimum", 1))
        for key, spec in properties.items():
            if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec and len(properties) == 1:
                args[key] = text[: spec.get("maxLength", len(text))]
        return args

    def name_missing(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        """When a step asks to be told what was missing, the decline's own text names it (the reason first, so a short
        field keeps it)."""
        if not self.wants_reason({"before_decline", "before_finish", "anytime"}):
            return args
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        out = dict(args)
        for key, value in args.items():
            spec = properties.get(key) or {}
            if isinstance(value, str) and "enum" not in spec and spec.get("type") == "string":
                out[key] = self.reasoned(value, name)[: spec.get("maxLength", 10_000)]
        return out

    def clean_decline(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        if self.plan.get("decline_no_names") is False:
            return args
        dirty = False
        for key, value in args.items():
            if isinstance(value, str):
                for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value):
                    if token and (token in self.seen_ids or token in self.shelf.ids):
                        dirty = True
        if self.plan.get("decline_no_names") or dirty:
            neutral = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
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

    def narrower_shelf(self, target: str) -> str:
        """A request naming two kinds together ("pants or jeans") means the store's combined category ("Pants & Jeans"):
        a pick from a category carrying fewer of those words is held once."""
        if target in self.shelf_held or self.items():
            return ""
        stem = lambda w: w[:-2] if w.endswith("es") and len(w) > 4 else (w[:-1] if w.endswith("s") and len(w) > 3 else w)
        kind = {stem(w) for w in re.findall(r"[a-z]+", str(self.plan.get("product") or "").lower()) if len(w) >= 3 and w not in ("and", "the", "for", "with", "pair", "set")}
        if len(kind) < 2:
            return ""
        labels: dict[str, str] = {}
        def visit(node: Any) -> None:
            if isinstance(node, dict):
                ident = self.shelf.id_field(node)
                ident = ident if isinstance(ident, str) else None
                label = next((v for k, v in node.items() if isinstance(v, str) and re.search(r"categor|department|shelf", k.lower())), None)
                if ident and label:
                    labels[ident] = label
                for v in node.values():
                    visit(v)
            elif isinstance(node, list):
                for v in node:
                    visit(v)
        visit(self.records)
        share = lambda g: len(kind & {stem(w) for w in re.findall(r"[a-z]+", g.lower())})
        mine = labels.get(target)
        if not mine or not labels:
            return ""
        top = max(set(labels.values()), key=share)
        if share(top) < 2 or share(mine) >= share(top):
            return ""
        self.shelf_held.add(target)
        return (f"{target} is in the store category {mine!r}, but the request asks for {self.plan.get('product')!r} and the store has the category {top!r}, which carries every word of that. "
                "Two kinds named together mean the store's combined category: choose from it.")

    def order_known_fit(self) -> None:
        """Puts the best fit the code knows (by the request's rule, never a gone line) in the empty cart, without a model call."""
        fits = self.rank_by_rule([k for k in self.live_fits() if k not in self.gone])
        if not fits and not self.items() and not self.open_pointers():
            close = self.closest_line("")
            fits = [close] if close else []
        found = self.add_call_args(fits[0]) if fits else None
        if not found:
            return
        tool, in_args = found
        self.agent_placed.add(fits[0])
        reply = self.env_call([{"name": tool, "args": in_args}])
        self.trace.append({"kind": "forced_finish", "action": {"name": tool, "args": in_args}, "reply": reply, "known_fit": fits[0]})
        if reply is None or self.finished or self.dead:
            return
        self.code_booking(tool, in_args, reply.get("observation"))
        said = _compact(reply.get("observation"))
        if (isinstance(reply.get("observation"), dict) and reply["observation"].get("error")) or self.gone_said(said):
            self.gone.add(fits[0])
            refused = isinstance(reply.get("observation"), dict) and reply["observation"].get("error")
            out = self.remove_args(fits[0]) if not refused and fits[0] in self.cart_ids and self.steps_left() >= 3 else None
            if out:
                # the add went through but the line is reported gone: it comes out of the store's cart too
                gone_reply = self.env_call([{"name": out[0], "args": out[1]}])
                if gone_reply is not None:
                    self.code_booking(out[0], out[1], gone_reply.get("observation"))
            self.cart_ids = [i for i in self.cart_ids if i != fits[0]]

    def force_finish(self) -> None:
        if self.finished or self.dead or self.steps_left() <= 0 or self.time_left() < 2:
            return
        self.stock_check()
        order = self.roles.get("order")
        decline = self.roles.get("decline")
        try:
            self.replace_duty()
        except Exception as exc:
            self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
        if self.finished or self.dead or self.steps_left() <= 0:
            return
        if order and self.roles.get("cart") and self.cart_ids and (set(self.cart_ids) & self.gone) and self.steps_left() >= 3:
            # a forced finish never declines over a dead line: the dead line comes out and, when it was the only line and a fit
            # is known by the request's rule, that fit goes in; then the cart is ordered
            dead_lines = [i for i in self.cart_ids if i in self.gone]
            spare = self.steps_left() - 2
            for dead in dead_lines[:spare]:
                found = self.remove_args(dead)
                if not found:
                    break
                tool, out_args = found
                reply = self.env_call([{"name": tool, "args": out_args}])
                self.trace.append({"kind": "forced_finish", "action": {"name": tool, "args": out_args}, "reply": reply})
                if reply is None or self.finished or self.dead:
                    return
                self.code_booking(tool, out_args, reply.get("observation"))
            if not self.cart_ids and self.steps_left() >= 2 and not self.items():
                fits = self.rank_by_rule([k for k in self.live_fits() if k not in self.gone])
                if not fits and self.steps_left() >= 4:
                    self.unread_rows_read()
                    fits = self.rank_by_rule([k for k in self.live_fits() if k not in self.gone])
                found = self.add_call_args(fits[0]) if fits else None
                if found:
                    tool, in_args = found
                    self.agent_placed.add(fits[0])
                    reply = self.env_call([{"name": tool, "args": in_args}])
                    self.trace.append({"kind": "forced_finish", "action": {"name": tool, "args": in_args}, "reply": reply, "swap_in": fits[0]})
                    if reply is None or self.finished or self.dead:
                        return
                    self.code_booking(tool, in_args, reply.get("observation"))
                    said = _compact(reply.get("observation"))
                    if (isinstance(reply.get("observation"), dict) and reply["observation"].get("error")) or self.gone_said(said):
                        self.gone.add(fits[0])
                        self.cart_ids = [i for i in self.cart_ids if i != fits[0]]
        try:
            self.swap_failing_line()
        except Exception as exc:
            self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
        if self.finished or self.dead or self.steps_left() <= 0:
            return
        if (order and self.roles.get("cart") and not self.cart_ids and not self.items() and not self.probe_line
                and (self.steps_left() >= 3 or (self.steps_left() == 2 and self.empty_cart_fit()))):
            # a forced finish with an empty cart never declines over a fit: the unread rows within the limit are opened first
            # (when steps allow), and the best verified fit goes in. With only the two last steps left, an opened listing that
            # meets every known condition still goes in and is ordered (the add, then the order) instead of a decline: in
            # race 17, 35 of champ21's 37 declines were wrong, the ones Cairn passed at the last step of a short world
            try:
                if self.items():
                    # several products and an empty cart: each product's known fit goes in, only when every product has one
                    missing = self.bundle_state()[2]
                    room = 2 if self.basket_form() else len(missing) + 2
                    known = [p for p in missing if self.product_fit(p) or self.closest_line(p)]
                    added = self.fill_bundle(loose=True) if missing and known and self.steps_left() >= room else []
                    if added:
                        self.trace.append({"kind": "forced_finish", "filled": added})
                elif not [k for k in self.live_fits() if k not in self.gone] and self.steps_left() >= 5:
                    self.unread_rows_read()
                if not (self.finished or self.dead) and self.steps_left() >= 2:
                    self.order_known_fit()
            except Exception as exc:
                self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if self.finished or self.dead or self.steps_left() <= 0:
                return
        if order and self.roles.get("cart") and self.steps_left() >= 2 and (self.llm_down or self.cost >= self.cost_stop):
            # the money or the model ran out before the cart was settled: the code fills it from what it verified (a product of
            # several without a line gets its known fit; an empty single-product cart gets the best known fit), never a model call
            try:
                if self.items():
                    added = self.fill_bundle(loose=True)
                    if added:
                        self.trace.append({"kind": "forced_finish", "filled": added})
                elif not self.cart_ids:
                    self.order_known_fit()
            except Exception as exc:
                self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if self.finished or self.dead or self.steps_left() <= 0:
                return
        ordering = bool(order and self.cart_used and self.cart_ids and not (set(self.cart_ids) & self.gone))
        if ordering and self.probe_stands():
            # the cart holds only the probe the code put in before a decline, and no note changed anything since: the
            # verdict that nothing fits stands
            self.trace.append({"kind": "forced_finish", "probe_declined": self.probe_line})
            ordering = False
        if ordering and self.items() and self.steps_left() >= 3:
            # the code's own finish never orders a bundle with a product missing while it knows a line for that product
            try:
                if self.bundle_state()[2]:
                    added = self.fill_bundle(loose=True)
                    if added:
                        self.trace.append({"kind": "forced_finish", "filled": added})
                if self.bundle_state()[2] and self.steps_left() >= 6:
                    # still a product without a line: dearer held lines may leave it no room within the limit; under a
                    # lowest-price rule the cheaper fits go in first, then the product is filled again
                    note = self.code_powered(self.bundle_cheapest)
                    if note:
                        self.trace.append({"kind": "forced_finish", "cheaper": note[:200]})
                    added = self.fill_bundle(loose=True) if not (self.finished or self.dead) else []
                    if added:
                        self.trace.append({"kind": "forced_finish", "filled": added})
            except Exception as exc:
                self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if self.finished or self.dead or self.steps_left() <= 0:
                return
        partial = False
        if ordering and self.items() and (self.llm_down or self.cost >= self.cost_stop) and self.parse_sure():
            # out of money or model with a product of several still without a line: a partial order is never right, the
            # grounded decline is (and is right when the store holds no fit for that product); under an untrusted reading
            # the "missing" product may simply be one the code could not place, so the held cart is ordered
            try:
                missing = self.bundle_state()[2]
            except Exception:
                missing = []
            if missing:
                self.trace.append({"kind": "forced_finish", "partial_declined": list(missing)})
                ordering, partial = False, True
        look_against, look_line = False, ""
        try:
            self.last_look(not ordering)     # when money and steps allow, the same look as before a model's finish
            last = self.trace[-1] if self.trace else {}
            unmet = [str(c) for c in ((last.get("result") or {}).get("unmet_conditions") or []) if isinstance(c, (str, int))] if last.get("kind") == "last_look" and isinstance(last.get("result"), dict) else []
            # the look read the one held line's own record as stating ANOTHER value for a condition ("not met (the sheet says
            # 45 cm)"): when that line is only the code's guess (no opened sheet states every condition), it is no finish
            look_against = bool(ordering and not self.items() and len(self.cart_ids) == 1 and self.cart_ids[0] not in self.live_fits()
                                and any(re.search(r"\b(?:says|states|shows|reads|lists)\b", c) and re.search(r"not met|differ|instead|rather than", c, re.I) for c in unmet))
            if look_against:
                look_line = self.cart_ids[0]
                self.look_vetoed[look_line] = [flat(c.split(":")[0]) for c in unmet if ":" in c and flat(c.split(":")[0])]
                self.trace.append({"kind": "forced_finish", "look_against": look_line, "unmet": unmet[:3]})
                ordering = False
        except Exception as exc:
            self.trace.append({"kind": "last_look", "error": f"{type(exc).__name__}: {exc}"[:300]})
        if ordering and self.plan.get("priority") == "cheapest":
            try:
                note = self.enforce_cheapest()
                if note:
                    self.trace.append({"kind": "swap", "forced": True})
            except Exception as exc:
                self.trace.append({"kind": "swap", "error": f"{type(exc).__name__}: {exc}"[:300]})
            ordering = bool(order and self.cart_used and self.cart_ids and not (set(self.cart_ids) & self.gone)) and not partial
        owed = self.open_duties({"before_finish", "anytime", "before_order" if ordering else "before_decline"}, True)
        if owed and self.steps_left() >= 2 and self.messages_open() >= 1 and self.roles.get("message"):
            sent = self.say_now(self.duty_message(owed, not ordering), "Before finishing, I tell the shopper what the request asks.")
            if sent is not None:
                for duty in owed:
                    duty["done"] = True
            if self.finished or self.dead or self.steps_left() <= 0:
                return
        if not ordering and order and self.roles.get("cart") and not self.cart_ids and self.steps_left() >= 2 and not partial:
            # the code's own finish: the best-verified line of each product (else the closest opened one) is held first
            try:
                if self.items():
                    self.fill_bundle(loose=True)
                else:
                    self.order_known_fit()
            except Exception as exc:
                self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if self.finished or self.dead or self.steps_left() <= 0:
                return
            ordering = bool(self.cart_ids and not (set(self.cart_ids) & self.gone)) and not self.probe_stands()
        if (not ordering and order and decline and self.roles.get("cart") and not self.cart_ids and not self.items() and not self.probe_holds
                and self.steps_left() >= 4 and self.time_left() > FORCE_SECONDS):
            # never a decline from an episode that never held: the closest candidate goes in first, and the note the
            # hold sets off decides (a raise or a dropped requirement turns it into an order)
            try:
                held = self.code_powered(self.probe_hold)
            except Exception as exc:
                held = None
                self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if self.finished or self.dead or self.steps_left() <= 0:
                return
            if held is not None:
                fits = self.rank_by_rule([k for k in self.live_fits() if k not in self.gone])
                if fits and fits[0] not in self.cart_ids and self.steps_left() >= 2:
                    found = self.add_call_args(fits[0])
                    if found:
                        reply = self.env_call([{"name": found[0], "args": found[1]}])
                        if reply is not None and not self.finished:
                            self.code_booking(found[0], found[1], reply.get("observation"))
                ordering = bool(self.cart_ids and not (set(self.cart_ids) & self.gone)) and not self.probe_stands()
                if ordering and self.probe_back(held):
                    return
        if ordering and self.add_unread and self.steps_left() >= 6 and self.time_left() > 5:
            # the hold's note rides on the next call: the held line is read once more before the order goes out
            mark = (self.budget_now, self.relaxed_at, len(self.gone), self.price_moved_at)
            try:
                self.peek_after_add()
            except Exception as exc:
                self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if self.finished or self.dead or self.steps_left() <= 0:
                return
            if mark != (self.budget_now, self.relaxed_at, len(self.gone), self.price_moved_at) and not self.items():
                try:
                    if self.rerank_cart("A note arrived after the hold.") or self.finished:
                        return
                except Exception as exc:
                    self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
            ordering = bool(self.cart_ids and not (set(self.cart_ids) & self.gone))
        if not ordering and not partial and decline and not self.items() and self.time_left() > FORCE_SECONDS + 10:
            # the code's own decline gets the same last look as a model's: an opened candidate that passes the item check
            # against every condition heard is ordered instead
            try:
                self.decline_audit()
            except Exception as exc:
                self.trace.append({"kind": "decline_audit", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if self.finished or self.dead or self.steps_left() <= 0:
                return
            ordering = (bool(self.cart_ids and not (set(self.cart_ids) & self.gone)) and not self.probe_stands()
                        and not (look_line and self.cart_ids == [look_line]))
        if not ordering and not partial and order and self.cart_used and not look_against:
            try:
                if self.held_plausible():
                    self.trace.append({"kind": "forced_finish", "held_rescue": list(self.cart_ids)})
                    ordering = True
            except Exception as exc:
                self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
        if ordering:
            action = {"name": order, "args": self.neutral_args(order, "Order the cart.")}
        elif decline:
            action = {"name": decline, "args": self.name_missing(self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything.", declining=True), decline)}
        else:
            return
        reply = self.env_call([action])
        self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        if reply and not self.finished and self.empty_fault == self.step and self.empty_repair():
            return
        seen = (reply or {}).get("observation")
        refused = seen if isinstance(seen, dict) else {}
        lines: list[str] = []
        parts = [v for v in refused.values() if isinstance(v, (list, str))]
        parts += [x for v in refused.values() if isinstance(v, dict) for x in v.values() if isinstance(x, (list, str))]
        if isinstance(seen, (list, str)):
            parts.append(seen)        # a refusal delivered as the whole result
        _sentences(parts, lines)
        if (reply and not reply.get("finished") and lines and not self.handed_back and not self.llm_down and self.cost < self.cost_stop
                and self.steps_left() > self.finish_reserve() + 1 and self.time_left() > WRAP_UP_SECONDS):
            # the code's own finish drew a note: the model reads it (as on any refusal) and acts on it for a few turns
            self.handed_back = True
            try:
                self.read_notices(action["name"], seen)
                note = self.note_observation(action["name"], seen, refused=True)
            except Exception as exc:
                note = ""
                self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
            self.refused_at = self.step
            self.messages.append({"role": "user", "content": "[note] The order the code sent was refused with this note from the store or the shopper: "
                                  + " | ".join(lines[:3]) + (" " + note if note else "") + "\nApply it, fix the cart and order again (decline only if nothing fits).\n" + self.status()})
            return
        if (reply and not reply.get("finished") and lines and order and decline and self.is_decline(action["name"], action.get("args") or {})
                and self.steps_left() > 0 and not self.finished and not self.dead):
            # a decline refused with a note (race 18: a dropped requirement often arrives as the refusal of the first finish,
            # a decline included): the note is applied and the cart judged again before the finish goes out once more
            try:
                self.read_notices(action["name"], seen)
                if self.steps_left() >= 2:
                    self.swap_failing_line()
                if not (self.finished or self.dead) and not self.cart_ids and not self.items() and self.steps_left() >= 2 and self.roles.get("cart"):
                    fits = self.rank_by_rule([k for k in self.live_fits() if k not in self.gone])
                    found = self.add_call_args(fits[0]) if fits else None
                    if found:
                        self.agent_placed.add(fits[0])
                        put = self.env_call([{"name": found[0], "args": found[1]}])
                        self.trace.append({"kind": "forced_finish", "action": {"name": found[0], "args": found[1]}, "reply": put, "after_refused_decline": fits[0]})
                        if put is not None and not (self.finished or self.dead):
                            self.code_booking(found[0], found[1], put.get("observation"))
            except Exception as exc:
                self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if self.finished or self.dead or self.steps_left() <= 0:
                return
            try:
                buy = bool(self.cart_ids) and not (set(self.cart_ids) & self.gone) and (self.held_plausible() or bool(set(self.cart_ids) & set(self.live_fits())))
            except Exception:
                buy = False
            again = {"name": order, "args": self.neutral_args(order, "Order the cart.")} if buy else action
            reply = self.env_call([again])
            self.trace.append({"kind": "forced_finish", "action": again, "reply": reply, "after_refused_decline": True})
            return
        if reply and not reply.get("finished") and action["name"] == order and self.gone_said(" ".join(lines) or _compact(seen)) and self.cart_ids:
            self.report_gone(" ".join(lines) or _compact(seen), self.cart_ids)
            if len(self.cart_ids) > 1 and self.steps_left() >= 3:
                try:
                    self.code_powered(self.bundle_recover)      # the dead line is found by re-reading, swapped, and the order goes out again
                    if not (self.finished or self.dead) and self.steps_left() >= 2:
                        self.code_powered(self.code_bundle_round)
                        missing = self.bundle_state()[2]
                        if missing:
                            self.fill_bundle(loose=True)
                        if not self.bundle_state()[2] and self.cart_ids and not (set(self.cart_ids) & self.gone) and self.steps_left() >= 1:
                            again = self.env_call([action])
                            self.trace.append({"kind": "forced_finish", "action": action, "reply": again, "after_recovery": True})
                except Exception as exc:
                    self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
                if self.finished or self.dead:
                    return
        if reply and not reply.get("finished") and action["name"] == order:
            mark = (self.budget_now, self.price_moved_at, len(self.gone), self.relaxed_at)
            try:
                self.read_notices(order, seen)       # a new limit in the refusal moves the ledger's budget
            except Exception as exc:
                self.trace.append({"kind": "forced_finish", "error": f"{type(exc).__name__}: {exc}"[:300]})
            unread = mark == (self.budget_now, self.price_moved_at, len(self.gone), self.relaxed_at) and not self.gone_said(" ".join(lines) or _compact(seen))
            if unread and self.cart_ids and self.steps_left() == 1 and not self.finished and not self.dead:
                # one step left after a refusal the code cannot read: a decline never passes where the held line still
                # stands, so the same cart goes out once more (the note may have been a raise or a sale)
                reply = self.env_call([action])
                self.trace.append({"kind": "forced_finish", "action": action, "reply": reply, "unread_refusal": True})
                return
            if (unread and self.cart_ids and self.steps_left() >= 2 and not self.finished and not self.dead):
                # a refusal the code cannot read: the held lines' own records say what changed (a stock flag turned off)
                self.mark_change()
                held = list(self.cart_ids)
                self.reread_lines(held, keep=2)
                dead = [k for k in held if k in self.gone or (self.shelf.rows.get(k) or {}).get("stock") is False]
                if dead and not (self.finished or self.dead):
                    self.gone.update(dead)
                    self.trace.append({"kind": "forced_finish", "dead_by_state": dead})
                    if not self.items():
                        fits = self.rank_by_rule([k for k in self.live_fits() if k not in self.gone])
                        if fits and self.steps_left() >= 3:
                            self.code_order(fits[0], f"The order was refused and {dead[0]} now reads as out of stock; {fits[0]} meets every condition.")
                            if self.finished or self.dead:
                                return
                    if decline and self.steps_left() > 0 and not self.finished:
                        action = {"name": decline, "args": self.name_missing(self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything.", declining=True), decline)}
                        reply = self.env_call([action])
                        self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
                    return
        if (reply and not reply.get("finished") and lines and order and action["name"] == order and self.cart_used
                and not self.gone_said(" ".join(lines)) and self.steps_left() > 0):
            # refused for a changed price or a new limit: the held line is ordered once more while it fits the limit; when it
            # does not, the cheapest opened fit within the limit takes its place (two steps) before the order goes out again
            budget, quantity = self.walk_scope()
            held = self.cart_ids[-1] if self.cart_ids else ""
            price = (self.shelf.rows.get(held) or {}).get("price")
            swapped = ""
            if held and price is not None and budget is not None and price * quantity > budget + 1e-9 and self.steps_left() >= 4:
                better = [k for k in self.live_fits() if k != held]
                out_args = self.remove_args(held) if better else None
                in_args = self.add_call_args(better[0]) if better else None
                if out_args and in_args:
                    for cart_tool, cart_args in (out_args, in_args):
                        swap = self.env_call([{"name": cart_tool, "args": cart_args}])
                        self.trace.append({"kind": "forced_finish", "action": {"name": cart_tool, "args": cart_args}, "reply": swap})
                        if swap is None or self.finished or self.dead:
                            return
                        self.code_booking(cart_tool, cart_args, swap.get("observation"))
                    swapped = better[0]
            if swapped or price is None or budget is None or price * quantity <= budget + 1e-9 or _PRICE_WORDS.search(" ".join(lines)):
                reply = self.env_call([action])
                self.trace.append({"kind": "forced_finish", "action": action, "reply": reply, "swapped": swapped})
        if reply and not reply.get("finished") and isinstance(reply.get("observation"), (dict, list, str)) and (lines or action["name"] == order) and decline and self.steps_left() > 0:
            action = {"name": decline, "args": self.name_missing(self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything.", declining=True), decline)}
            reply = self.env_call([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})

    def has_line(self, observation: Any) -> bool:
        """Whether a result carries a sentence (three words or more) outside any listing record."""
        told: list[str] = []
        loose_strings(observation, told, keys=tuple(self.shelf.id_keys), listing=self.shelf.is_listing)
        return any(len(t.split()) >= 3 for t in told)

    def note_observation(self, name: str, observation: Any, refused: bool = False) -> str:
        notes = []
        if isinstance(observation, dict):
            _collect_ids(observation, self.seen_ids, keys=tuple(self.shelf.id_keys) + tuple(flat(k) for k in self.shelf.layout.get("id") or []))
            problem = " ".join(str(observation.get(k) or "") for k in ("error", "detail", "message", "reason") if isinstance(observation.get(k), str))
            core = {self.roles.get(r) for r in ("order", "decline", "cart", "cart_view")} - {None}
            if observation.get("error") and _USED_UP.search(problem) and name not in core:
                if name == self.roles.get("message"):
                    self.message_budget = self.messages_used if self.message_budget is None else min(self.message_budget, self.messages_used)
                self.capped.add(name)
                notes.append("That tool's allowance is used up; it is no longer offered. Do not try it again: read prices and stock from search results or the cart view instead, and decide with what you have.")
            if refused or "refusal" in observation:
                self.refusal_step = self.step
                self.refusals += 1
                self.refused_at = self.step
                self.order_checked = False
                self.mark_change()
                notes.append("THE ORDER WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (remove any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again.")
                refusal = observation.get("refusal")
                if refusal is None:
                    told: list[str] = []
                    loose_strings(observation, told, keys=tuple(self.shelf.id_keys), listing=self.shelf.is_listing)
                    refusal = [t for t in told if len(t.split()) >= 3] or observation
                said = refusal if isinstance(refusal, str) else (" ".join(refusal) if isinstance(refusal, list) and all(isinstance(t, str) for t in refusal) else _compact(refusal))
                self.refusal_text = said[:600]
                if self.gone_said(said) and (self.cart_ids or self.order_targets):
                    self.report_gone(said, self.cart_ids or self.order_targets)
                    backup = self.fallback_note()
                    if backup:
                        notes.append(backup)
                self.review_reason = "the order was refused: " + said[:300]
                if _PRICE_WORDS.search(said) and not self.gone_said(said):
                    notes.append("This refusal does not say the item is gone: when it only reports a changed price (or another change the item still meets), the item is still available. Re-read its current price; if price times quantity is still within the budget and it still meets every requirement, order it again right away. Never decline because a price changed.")
            elif len(observation) >= 2 and name != self.roles.get("message"):
                lines: list[str] = []
                for key, value in observation.items():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        _sentences(value, lines)
                if lines:
                    if self.gone_said(" ".join(lines)) and self.cart_ids:
                        self.report_gone(" ".join(lines), self.cart_ids)
                        backup = self.fallback_note()
                        if backup:
                            notes.append(backup)
                    notes.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(lines[:4]))
        elif isinstance(observation, list):
            _collect_ids(observation, self.seen_ids, keys=tuple(self.shelf.id_keys) + tuple(flat(k) for k in self.shelf.layout.get("id") or []))
            said = " ".join(v for v in observation if isinstance(v, str))
            if self.gone_said(said) and (self.cart_ids or (self.order_targets and name == self.roles.get("order"))):
                self.report_gone(said, self.cart_ids or self.order_targets)
        elif isinstance(observation, str) and self.gone_said(observation) and name != self.roles.get("message") and (self.cart_ids or (self.order_targets and name == self.roles.get("order"))):
            self.report_gone(observation, self.cart_ids or self.order_targets)
        return " ".join(notes)

    def report_gone(self, text: str, lines: list[str]) -> None:
        """A report that a line sold out: the line the text names is gone; with one line held, that line; with several lines
        and no name, which line died stays unknown until the lines are re-read (the bundle recovery does that)."""
        low = (text or "").lower()
        named = [k for k in lines if k and k.lower() in low]
        if named:
            self.gone.update(named)
        elif len(lines) <= 1:
            self.gone.update(lines)
        else:
            self.dead_unknown = self.step
        self.mark_change()

    # ------------------------------------------------------------------ the note reader: free-text lines read by meaning

    def note_lines(self, name: str, texts: list[str]) -> list[str]:
        """Free-text lines of a result worth a read: sentences of five words or more outside any listing record, not yet
        read, never the shopper's answer to our own message (the hearing reads those)."""
        if name == self.roles.get("message"):
            return []
        out = []
        for text in texts:
            line = " ".join(str(text).split())
            if len(re.findall(r"[A-Za-z]{2,}", line)) < 5 or line in self.note_reads or line in out:
                continue
            out.append(line)
        return out[:6]

    def read_note(self, lines: list[str]) -> None:
        """One cheap model read of new free-text lines: a new limit, a sold-out line, a moved price, an added or dropped
        need. Each line is read once; what the read finds lands in note_reads, keyed by the line."""
        if not lines:
            return
        for line in lines:
            self.note_reads[line] = {}
        if (self.note_calls >= NOTE_CALLS or self.time_left() < WRAP_UP_SECONDS or self.cost >= self.cost_stop
                or (self.past_soft() and self.note_calls_late >= NOTE_CALLS_LATE)):
            return
        self.note_calls += 1
        self.note_calls_late += int(self.past_soft())
        held = [f"{k} | {(self.shelf.rows.get(k) or {}).get('title', '')[:60]} | price {(self.shelf.rows.get(k) or {}).get('price', '?')}" for k in (self.cart_ids or self.order_targets)]
        prompt = (NOTE_PROMPT + "\n\nPRICE UNIT OF THE REQUEST: " + str(self.plan.get("currency") or "(none stated)")
                  + f"\nTOTAL LIMIT NOW: {self.budget_now}"
                  + "\nLINES HELD IN THE CART (id | title | price):\n" + ("\n".join(held) if held else "(none)")
                  + "\n\nTHE LINES:\n" + "\n".join(f"{i + 1}. {l}" for i, l in enumerate(lines)))
        message = self.infer([{"role": "user", "content": prompt}], None, [READ_MODEL], 400)
        result = _parse_json_object((message or {}).get("content") or "") or {}
        reads = result.get("lines") if isinstance(result.get("lines"), list) else [result]
        for index, line in enumerate(lines):
            got = reads[index] if index < len(reads) and isinstance(reads[index], dict) else (reads[0] if len(reads) == 1 and isinstance(reads[0], dict) else {})
            self.note_reads[line] = got
        self.trace.append({"kind": "note_read", "lines": [l[:120] for l in lines], "result": result})

    def notes_settled(self, heard: list[str]) -> bool:
        """Whether every heard line was read by the note reader, which found a change in it and no need added or dropped."""
        reads = [self.note_reads.get(" ".join(str(h).split())) for h in heard]
        return bool(reads) and all(r and any(r.get(k) for k in ("new_budget", "stockout_of", "price_moved_of")) and not r.get("added_need") and not r.get("dropped_need") for r in reads)

    def note_gist(self, heard: list[str]) -> str:
        parts = []
        for h in heard:
            r = self.note_reads.get(" ".join(str(h).split())) or {}
            if self.note_budget(" ".join(str(h).split())) is not None:
                parts.append(f"the shopper's total limit is now {self.budget_now}")
            if r.get("stockout_of"):
                parts.append("a held line can no longer be bought: take it out and put the next verified fit in")
            if r.get("price_moved_of"):
                parts.append("a price moved: the held line stays acceptable while price times quantity is within the limit; re-read it and order again")
        return "Meaning: " + "; ".join(parts) if parts else ""

    def note_budget(self, line: str) -> float | None:
        """The new limit the note reader found in a line, when the figure really stands in the line."""
        figure = (self.note_reads.get(line) or {}).get("new_budget")
        if isinstance(figure, str):
            figure = as_price(figure)
        if not isinstance(figure, (int, float)) or isinstance(figure, bool) or figure <= 0:
            return None
        written = {float(m.group(1).replace(",", "") + ("." + m.group(2) if m.group(2) else "")) for m in _FIGURE.finditer(line)}
        return float(figure) if any(abs(w - figure) < 1e-6 for w in written) else None

    def budget_figure(self, line: str) -> float | None:
        """The structural reading of a limit line: exactly one figure carrying the request's own price unit, no listing
        named, and no word about a listing's price unless a spending word stands beside it."""
        currency = str(self.plan.get("currency") or "")
        if not currency or any(k and k in line for k in list(self.shelf.ids)[:400]):
            return None
        money = money_pattern(currency)
        hits = []
        for match in _FIGURE.finditer(line):
            before, after = line[: match.start()], line[match.end():]
            if re.search(money + r"\s*$", before, re.I) or re.match(r"\s*" + money, after, re.I):
                hits.append(float(match.group(1).replace(",", "") + ("." + match.group(2) if match.group(2) else "")))
        if len(hits) != 1 or limits_in(line):
            return None
        if re.search(r"\b(?:price|cost|charge)\w*\b", line, re.I) and not _SPEND_CUE.search(line):
            return None
        return hits[0] if hits[0] > 0 else None

    def set_budget(self, figure: float, text: str, name: str) -> None:
        """A new total limit applied at once: the ledger's budget, its version and the change mark move."""
        if self.budget_now is not None and abs(float(self.budget_now) - figure) < 1e-9:
            return
        self.trace.append({"kind": "budget", "from": self.budget_now, "to": figure, "line": text[:200], "tool": name})
        if isinstance(self.budget_now, (int, float)) and figure > float(self.budget_now):
            self.budget_rose_at = self.step
            self.budget_before_rise = float(self.budget_now)
        self.budget_now = figure
        self.budget_changed_at = self.step
        self.ledger_version += 1
        self.mark_change()
        self.review_reason = self.review_reason or f"the shopper set a new total limit: {figure:g} ({text[:160]})"

    def gone_said(self, text: str) -> bool:
        """Whether a text reports a line sold out: by the cue the code knows, or because it carries a line the note reader
        read as a stockout."""
        if not text:
            return False
        cue = next((m for m in _GONE.finditer(text) if not _SELF_NO_LONGER.search(text[max(0, m.start() - 16): m.end()])), None)
        if cue or _GONE_TOO.search(text):
            return True
        low = flat(text)
        return any(flat(line)[:48] in low for line in self.note_gone)

    def apply_note(self, line: str, name: str) -> None:
        """What one read line changes besides the limit: a stockout (the named or the only held line is gone; with several
        lines and no name the lines are re-read) and a price move (the held lines are re-read before the order)."""
        got = self.note_reads.get(line) or {}
        dropped = got.get("dropped_need")
        if isinstance(dropped, str):
            dropped = [dropped]
        if isinstance(dropped, list) and dropped:
            self.drop_needs(dropped, line)
        held = list(self.cart_ids or self.order_targets)
        sold = got.get("stockout_of")
        if sold and held and line not in self.note_gone:
            self.note_gone.add(line)
            named = [k for k in held if isinstance(sold, str) and k in sold]
            self.report_gone(" ".join(named) if named else line, held)
            self.market_notes.append("a line sold out: " + line[:120])
            self.review_reason = self.review_reason or "a store notice: " + line[:300]
        moved = got.get("price_moved_of")
        if moved and held and not sold and line not in self.note_price:
            self.note_price.add(line)
            self.price_moved_at = self.step
            self.market_notes.append("a price moved: " + line[:120])
            self.review_reason = self.review_reason or "a store notice: " + line[:300]
            self.mark_change()

    # ------------------------------------------------------------------ fitment codes: the request's model code on the pages

    def fit_targets(self) -> list[str]:
        """The model or part codes the purchase must fit: those the request and the shopper name, and the model number a
        held line's own page gives for itself (its companions must fit it)."""
        found: list[str] = []
        for text in [str(self.policy.get("query") or "")] + [str(x) for x in list(self.stated) + list(self.revealed)]:
            for code in codes_in(text):
                if code not in found:
                    found.append(code)
        if found or len(self.items()) > 1:
            sources = list(self.cart_ids)
            if len(self.items()) > 1 and not any(self.own_code(k) for k in sources):
                main = self.main_pick()
                sources += [main] if main else []
            for key in sources:
                for sentence in re.split(r"(?<=[.;!?])\s+|\n", self.shelf.page_text(key)):
                    own = codes_in(sentence)
                    if len(own) == 1 and ":" not in sentence and not _CODE_EXCEPT.search(sentence) and own[0] not in found and "x" not in own[0]:
                        found.append(own[0])
                code = self.own_code(key)
                if code and code not in found:
                    found.append(code)
        return found

    def own_code(self, key: str) -> str:
        """The model code a listing's page gives for the listing itself: a sentence of the form "<label>: <one code>" whose
        short label (one or two words) makes no compatibility claim ("Model: VIT-4510."). '' when none."""
        for sentence in re.split(r"(?<=[.;!?])\s+|\n", self.shelf.page_text(key)):
            head, colon, tail = sentence.partition(":")
            if not colon or codes_in(head) or len(words(head)) > 2 or _CLAIM_WORDS.search(head):
                continue
            listed = codes_in(tail.split(".")[0] if "." in tail else tail)
            if (len(listed) == 1 and "x" not in listed[0] and not _CODE_EXCEPT.search(tail)
                    and len(re.sub(r"\D", "", listed[0])) >= 3 and len(words(tail)) <= 3):
                return listed[0]             # a model number (three digits or more) standing alone after its label
        return ""

    def main_pick(self) -> str:
        """With several products and no held line naming its own model code: the product whose opened pages give their own
        model codes is the one the others must fit; when its listings that state every one of its conditions (in stock,
        within the limit) share one code, the cheapest of them. '' otherwise."""
        if self.main_busy:
            return ""
        self.main_busy = True
        try:
            budget, _ = self.walk_scope()
            product = self.main_product()
            if not product:
                return ""
            keys = [k for k in self.shelf.sheets if self.own_code(k) and self.product_of(k) == product]
            held = self.lines_for.get(product)
            if held and held in keys and held not in self.gone:
                return held
            conds = self.product_conditions(product)
            quantity = dict(self.items()).get(product, 1)
            pool = []
            for key in keys:
                row = self.shelf.rows.get(key) or {}
                if key in self.gone or row.get("stock") is False or row.get("price") is None:
                    continue
                if budget is not None and row["price"] * quantity > budget + 1e-9:
                    continue
                if conds and not all(self.shelf.says(key, p, v) == "yes" for p, v in conds):
                    continue
                pool.append((row["price"], key))
            if not pool:
                return ""
            ranked = self.rank_by_rule([k for _, k in sorted(pool)])
            return ranked[0] if ranked else min(pool)[1]
        finally:
            self.main_busy = False

    def main_product(self) -> str:
        """The product of several whose opened pages give their own model codes (the one the others must fit), or ''."""
        owners = {self.product_of(k) for k in self.shelf.sheets if self.own_code(k)} - {""}
        return next(iter(owners)) if len(owners) == 1 else ""

    def fit_state(self, key: str, targets: list[str] | None = None) -> str:
        """How a listing's own page answers the codes: 'fits' (it lists a wanted code, by wildcard too, not excepted, or
        another page says its part code also fits one), 'fails' (it lists codes of the same family but none wanted),
        'silent' (it names no such code) or 'unread' (its page was never opened)."""
        targets = self.fit_targets() if targets is None else targets
        if not targets:
            return "silent"
        families = {code_prefix(t) for t in targets}
        spoke = False

        def wanted(listed: list[str], excluded: list[str]) -> bool:
            return any(any(code_covers(c, t) for c in listed) and not any(code_covers(e, t) for e in excluded) for t in targets)

        own = codes_in((self.shelf.rows.get(key) or {}).get("title") or "")
        for other, row in self.shelf.rows.items():
            if other == key or not own or not (row.get("paged") or (row.get("sheet") and code_claims(self.shelf.page_text(other)))):
                continue
            for subjects, listed, excluded in code_claims(self.shelf.page_text(other)):
                if subjects and set(subjects) & set(own) and wanted(listed, excluded):
                    return "fits"
        if not (self.shelf.rows.get(key) or {}).get("paged") and not code_claims(self.shelf.page_text(key)):
            return "unread"                  # a record whose text already carries the codes counts as read
        for subjects, listed, excluded in code_claims(self.shelf.page_text(key)):
            if subjects and not set(subjects) & set(own):
                continue
            if not any(code_prefix(c) in families for c in listed + excluded):
                continue
            spoke = True
            if wanted(listed, excluded):
                return "fits"
        return "fails" if spoke else "silent"

    def fit_groups(self, targets: list[str]) -> set[str]:
        """Store categories whose opened pages speak about the wanted codes' families: there a listing must list the code."""
        families = {code_prefix(t) for t in targets}
        out = set()
        for key, row in self.shelf.rows.items():
            if row.get("paged") and row.get("group"):
                if any(not s and any(code_prefix(c) in families for c in l + e) for s, l, e in code_claims(self.shelf.page_text(key))):
                    out.add(flat(row["group"]))
        return out

    def fit_blocked(self, key: str) -> bool:
        """A listing the code must not commit to under the codes: its page lists other codes of the family, or it is silent
        (or unread) in a category whose pages list them."""
        targets = self.fit_targets()
        if not targets:
            return False
        state = self.fit_state(key, targets)
        if state == "fits":
            return False
        if state == "fails":
            return True
        group = flat((self.shelf.rows.get(key) or {}).get("group") or "")
        coded = self.fit_groups(targets)
        return bool(coded) and (group in coded or (state == "unread" and not group))

    def fit_note(self) -> str:
        targets = self.fit_targets()
        if not targets:
            return ""
        fits = [k for k, r in self.shelf.rows.items() if r.get("paged") and self.fit_state(k, targets) == "fits" and k not in self.gone]
        fails = [k for k, r in self.shelf.rows.items() if r.get("paged") and self.fit_state(k, targets) == "fails"]
        text = (f"[fitment] the purchase must fit {', '.join(targets[:4])}: a listing qualifies only when its own page (the description, not the sheet) "
                "lists that code (a code written with x stands for any digit; an 'except' list rules codes out; a page saying another part's code "
                "'also fits' a code extends that part). Never search for the code itself: search by the product's words and open the pages.")
        if fits:
            text += " Pages that list it: " + ", ".join(f"{k} ({(self.shelf.rows.get(k) or {}).get('price', '?')}, {(self.shelf.rows.get(k) or {}).get('group', '')})" for k in fits[:8]) + "."
        if fails:
            text += " Pages read that do not: " + ", ".join(fails[:8]) + "."
        return text

    def fit_tool(self) -> str | None:
        return self.roles.get("view") or self.roles.get("details")

    def fit_candidates(self, limit: int) -> list[str]:
        """Unread listings worth opening for the codes: from each search since the last round, the cheapest rows within the
        limit that are on offer, taken in turns so every product gets its share."""
        budget, quantity = self.walk_scope()
        lists = []
        for search in self.searches[self.fit_searched:]:
            rows = []
            for key in search.get("shown") or []:
                row = self.shelf.rows.get(key) or {}
                if row.get("paged") or key in self.gone or key in self.fit_asked or row.get("stock") is False or row.get("price") is None:
                    continue
                if budget is not None and row["price"] * quantity > budget + 1e-9:
                    continue
                rows.append((row["price"], key))
            lists.append([k for _, k in sorted(rows)][:FIT_PER_SEARCH])
        picked: list[str] = []
        for turn in range(FIT_PER_SEARCH):
            for keys in lists:
                if turn < len(keys) and keys[turn] not in picked:
                    picked.append(keys[turn])
        return picked[:limit]

    def fit_reads(self, keys: list[str], why: str) -> int:
        """The code opens listings' pages itself, in one envelope (a step each). Steps spent."""
        tool = self.fit_tool()
        if not tool or not keys:
            return 0
        actions = []
        for key in keys:
            args = self.id_args(tool, key)
            if args:
                actions.append({"name": tool, "args": args})
                self.fit_asked.add(key)
        actions = actions[: max(0, self.steps_left() - self.finish_reserve() - 2)]
        if not actions:
            return 0
        reply = self.env_call(actions)
        self.trace.append({"kind": "env", "step": self.step, "actions": actions, "reply": reply, "fitment": why})
        if reply is None:
            return 0
        observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
        for action, observation in zip(actions, observations):
            self.book_call(action["name"], action["args"], observation)
            self.read_notices(action["name"], observation)
            self.note_observation(action["name"], observation)
        self.fit_spent += len(actions)
        return len(actions)

    def fitment_round(self) -> bool:
        """With a code to fit, after each new search the code opens the pages of the cheapest unread rows it showed (a few per
        search, in turns over the products), so the choice rests on the pages that carry the codes. True when steps were spent."""
        targets = self.fit_targets()
        if (not targets or not self.fit_tool() or self.finished or self.dead or self.fit_searched >= len(self.searches)
                or self.fit_spent >= self.fit_cap() or self.steps_left() < self.finish_reserve() + 8 or self.cost >= self.cost_hard
                or self.time_left() < WRAP_UP_SECONDS + 40):
            return False
        keys = self.fit_candidates(min(FIT_PER_ROUND, self.fit_cap() - self.fit_spent))
        self.fit_searched = len(self.searches)
        if not keys:
            return False
        spent = self.fit_reads(keys, "round")
        if spent:
            fits = [k for k in keys if self.fit_state(k, targets) == "fits"]
            self.messages.append({"role": "user", "content": f"[fitment] the code opened {spent} page(s) ({', '.join(keys[:spent])}) to read their fitment codes"
                                  + (f"; these list {targets[0]}: {', '.join(fits)}" if fits else "; none of them lists the code")
                                  + ". Their records are in the store's replies; judge only from what the pages say.\n" + self.status()})
        return spent > 0

    def fit_cap(self) -> int:
        return max(FIT_MIN, min(FIT_MAX, int(self.max_steps * FIT_SHARE)))

    def fit_pre_reads(self, runnable: list[tuple[str, str, dict[str, Any]]], replies: list[tuple[str, str]]) -> list[tuple[str, str, dict[str, Any]]]:
        """Before an add under a code: the target's page is opened first when it was never read; an add whose page then
        lists other codes of the family (or none, where the category's pages list them) is held with the fits named."""
        targets = self.fit_targets()
        cart = self.roles.get("cart")
        if not targets or not cart or not self.fit_tool():
            return runnable
        adds = []
        for call_id, name, args in runnable:
            if name != cart:
                continue
            target = next((v.strip() for v in args.values() if isinstance(v, str) and v.strip() in self.shelf.ids), "") or ((_listed_ids(args) or [""])[-1])
            if target and target not in self.cart_ids:
                adds.append((call_id, target))
        unread = [t for _, t in adds if not (self.shelf.rows.get(t) or {}).get("paged") and t not in self.fit_asked]
        if unread and self.steps_left() >= self.finish_reserve() + 3 + len(unread):
            self.fit_reads(unread[:3], "before add")
            if self.finished or self.dead:
                return []
        out = []
        held_ids = set()
        for call_id, target in adds:
            if self.fit_state(target, targets) == "fails" and f"add|{target}" not in self.fit_held:
                self.fit_held.add(f"add|{target}")
                held_ids.add(call_id)
                others = [k for k, r in self.shelf.rows.items() if r.get("paged") and k != target and self.fit_state(k, targets) == "fits" and k not in self.gone]
                replies.append((call_id, f"Not run (no step used): {target}'s own page lists fitment codes, but not {', '.join(targets[:3])}, so it does not fit. "
                                + (f"Pages that list it: {', '.join(others[:5])}. " if others else "Open the pages of other candidates of this product. ")
                                + "If you are certain it fits, send the same call again."))
        for item in runnable:
            if item[0] not in held_ids:
                out.append(item)
        return out

    def fit_order_hold(self) -> str | None:
        """Before an order under a code: held lines whose page was never read are opened; a line whose page lists other
        codes of the family holds the order once per cart, with the pages that list the code named."""
        targets = self.fit_targets()
        if not targets or not self.cart_ids or not self.fit_tool():
            return None
        unread = [k for k in self.cart_ids if not (self.shelf.rows.get(k) or {}).get("paged") and k not in self.fit_asked]
        if unread and self.steps_left() >= self.finish_reserve() + 2 + len(unread):
            self.fit_reads(unread[:4], "before order")
            if self.finished or self.dead:
                return None
        bad = [k for k in self.cart_ids if self.fit_state(k, targets) == "fails"]
        signature = "order|" + ",".join(sorted(self.cart_ids))
        if not bad or signature in self.fit_held or self.steps_left() < 4:
            return None
        self.fit_held.add(signature)
        others = [k for k, r in self.shelf.rows.items() if r.get("paged") and k not in self.cart_ids and self.fit_state(k, targets) == "fits" and k not in self.gone]
        return (f"Not run (no step used): the page of {', '.join(bad)} lists fitment codes but not {', '.join(targets[:3])}, so that line does not fit. "
                + ("Pages that list the code: " + ", ".join(f"{k} ({(self.shelf.rows.get(k) or {}).get('price', '?')})" for k in others[:6]) + ". " if others else "")
                + "Swap the line for a listing whose page lists the code, then order. If you are certain the line fits, send the order again.")

    # ------------------------------------------------------------------ pacing of declines (after a limit note, long caps)

    def pacing_decline_hold(self) -> str | None:
        """A decline right after a new limit, or early in a long episode, is held: after a limit note the held line is
        re-sent when it fits the new limit, else the cheapest verified fit goes in and is ordered; in a long episode the
        decline waits until LONG_DECLINE_BY of the cap while unread candidates remain."""
        if self.budget_changed_at >= 0 and self.step - self.budget_changed_at <= 3 and self.budget_holds < 2 and self.steps_left() >= 3:
            self.budget_holds += 1
            budget, quantity = self.walk_scope()
            if self.items():
                good, bad, missing = self.bundle_state()
                total = self.bundle_total()
                if self.cart_ids and not bad and not missing and (budget is None or (total is not None and total <= budget + 1e-9)):
                    return self.code_resend(f"The shopper's new limit is {budget}; the held lines total {total} and still fit, so the order goes out again.")
            else:
                line = self.cart_ids[-1] if self.cart_ids else ""
                row = self.shelf.rows.get(line) or {}
                if line and line not in self.gone and row.get("price") is not None and row.get("stock") is not False and (budget is None or row["price"] * quantity <= budget + 1e-9) and self.roles.get("cart"):
                    self.code_order(line, f"The shopper's new limit is {budget}; the held line {line} ({row['price']:g} x {quantity}) still fits it, so it is ordered again instead of declining.")
                    if self.finished:
                        return "Not run: the code re-sent the order for the held line (see above); the episode is finished."
                    return "Not run (no step used): the held line still fits the new limit, so the code re-sent its order; the result is above."
                fits = self.rank_by_rule([k for k in self.live_fits() if k != line])
                if fits and self.roles.get("cart"):
                    self.code_order(fits[0], f"The shopper's new limit is {budget}; {fits[0]} is the best verified fit within it, so it is ordered instead of declining.")
                    if self.finished:
                        return "Not run: the code ordered the best fit within the new limit (see above); the episode is finished."
                    return "Not run (no step used): the code ordered the best verified fit within the new limit; the result is above."
            return (f"Not run (no step used): the shopper just set a new limit ({budget}). A new limit is not a reason to decline: put the cheapest listing that meets "
                    "every condition within it in the cart (keep the held line when it still fits) and order it.")
        # in a mid-length task (below 50 steps) the wait applies only while the model holds no line of its own: a decline
        # right after its own hold is how the store's note on that hold is drawn
        own_line = any(k not in self.agent_placed for k in self.cart_ids)
        if (self.max_steps >= LONG_CAP and (self.max_steps >= 50 or not own_line) and self.step < LONG_DECLINE_BY * self.max_steps
                and self.long_holds < 3 and self.steps_left() > 8):
            unread = [k for s in self.searches for k in s.get("shown") or [] if not (self.shelf.rows.get(k) or {}).get("paged") and not (self.shelf.rows.get(k) or {}).get("sheet") and k not in self.gone]
            if unread or len(self.searches) < max(2, len(self.product_names())):
                self.long_holds += 1
                # the code does the reading itself (steps, no model turns): the cheapest unread candidates are opened now
                try:
                    did = self.code_bundle_round() if self.items() else self.code_single_round()
                except Exception as exc:
                    did = False
                    self.trace.append({"kind": "hold", "error": f"{type(exc).__name__}: {exc}"[:300]})
                if did and not self.finished:
                    fits = self.rank_by_rule([k for k in self.live_fits() if k not in self.gone]) if not self.items() else []
                    return (f"Not run (no step used): step {self.step} of {self.max_steps} is early for a decline in this long task, so the code opened more "
                            "candidates within the limit (their records are in the store's replies above)"
                            + (f"; these meet every condition by their own sheets: {', '.join(fits[:3])}. Put the best in the cart and order it." if fits else
                               ". Judge them; decline only when every candidate fails a condition by its own records."))
                return (f"Not run (no step used): step {self.step} of {self.max_steps} is early for a decline in this long task. Search each requested product by its "
                        "category words, open the pages of the unread candidates within the limit and complete the cart; decline only when every candidate of a product fails "
                        "a condition by its own records.")
        return None

    def bundle_budget_resend(self) -> bool:
        """After a refused order of several lines that carried a new limit: when every product holds a fitting line and the
        lines total within the new limit, the code sends the order again (once per refusal)."""
        if (not self.items() or not self.refusals or self.refused_at <= self.repaired_at or self.budget_changed_at < self.refused_at
                or self.finished or not self.roles.get("order") or self.steps_left() < 2 or self.cost >= self.cost_stop):
            return False
        self.repaired_at = self.refused_at
        good, bad, missing = self.bundle_state()
        budget, _ = self.walk_scope()
        total = self.bundle_total()
        if not self.cart_ids or bad or missing or (set(self.cart_ids) & self.gone) or total is None or (budget is not None and total > budget + 1e-9):
            return False
        self.code_resend(f"The refused order carried a new limit ({budget}); the held lines total {total:g} and every one still fits, so the order goes out again.")
        return True

    def code_resend(self, why: str) -> str:
        order = self.roles.get("order")
        if not order:
            return ""
        self.messages.append({"role": "user", "content": "[note] " + why})
        call = {"id": f"resend_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(self.neutral_args(order, "Order the cart."))}}
        self.messages.append({"role": "assistant", "content": "I order the held lines again.", "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        for call_id, reply_text in replies or [(call["id"], "Not sent.")]:
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text + "\n" + self.status()})
        if self.finished:
            return "Not run: the code re-sent the order for the held lines (see above); the episode is finished."
        return "Not run (no step used): the code re-sent the order for the held lines; its result is above."

    # ------------------------------------------------------------------ one model turn: holds, the store, bookkeeping

    def call_hold(self, call: dict[str, Any], together: int, allowed: set[str], planned: bool, queued_messages: int) -> tuple[str | None, str, dict[str, Any]]:
        """The no-step hold for one proposed call, or None with the call's name and fitted arguments when it may go out."""
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        message_tool, cart_tool = self.roles.get("message"), self.roles.get("cart")
        function = call.get("function") or {}
        name, raw = function.get("name"), function.get("arguments")
        if name not in self.schemas:
            return f"Not run (no step used): '{name}' is not one of the tools. Use one of: {', '.join(sorted(allowed))}.", name, {}
        if not planned and name not in allowed and name == message_tool and self.planner_on and self.interview(force=True):
            return ("Not run as written (no step used): the code put its own question to the shopper instead; the exchange is above. "
                    "Use the answer; the next question goes out the same way."), name, {}
        if not planned and name not in allowed:
            reason = ("questions to the shopper are planned and sent for you after you have opened a few listings' details" if name == message_tool and self.planner_on
                      else "too few steps are left for this; only these tools can be used now: " + ", ".join(sorted(allowed)))
            return f"Not run (no step used): {reason}.", name, {}
        args = raw if isinstance(raw, dict) else (_parse_json_object(raw or "{}") if (raw or "").strip() else {})
        if args is None:
            return "Not run (no step used): the arguments were not a valid JSON object.", name, {}
        args, problems = _check(args, self.schemas[name], "args")
        if problems:
            return "Not run (no step used): invalid arguments: " + "; ".join(problems[:6]), name, args
        if name in finishing and together > 1:
            return "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls.", name, args
        if name == message_tool:
            text = " ".join(str(v) for v in args.values() if isinstance(v, str))
            if not planned and "?" not in text and not any(_TELL_RULE.search(str(rule)) for rule in self.plan.get("rules") or []):
                return "Not run (no step used): messages to the shopper are only for questions, and this one asks nothing. Do not report findings, choices or plans to the shopper; the finishing tool carries that.", name, args
            if text.strip().lower() in self.questions_sent:
                return "Not run (no step used): this exact message was already sent, and its answer is above. Ask about something else, or carry on.", name, args
            repeat = self.asked_already(text)
            if repeat and not planned and text.strip().lower() not in self.repeat_held:
                self.repeat_held.add(text.strip().lower())
                return (f"Not run (no step used): the shopper was already asked about the {repeat} (see the exchanges above); a reply that did not follow the "
                        "question counts as no preference. Ask about a different property, or carry on.", name, args)
            if not planned and (text.count("?") > 1 or re.search(r"\?\s*\S|\b(?:and|or)\s+(?:what|which|how|would)\b|,\s*(?:and\s+)?(?:what|which)\b", text, re.I)):
                return "Not run (no step used): this message puts several questions. Ask ONE short question about ONE property of one product, with one question mark.", name, args
            if not planned and text.count(",") >= 2 and len(text.split()) > 9:
                return "Not run (no step used): this question lists several properties; the shopper answers at most one per message. Ask about ONE property (the likeliest need), with one question mark.", name, args
            if not planned and "?" not in text and len(text.split()) <= 4 and not re.search(r"\d", text):
                return "Not run (no step used): an acknowledgement wastes a message. Messages are only for questions.", name, args
            if self.message_budget is not None and self.messages_used + queued_messages >= self.message_budget - self.held_back():
                return "Not run (no step used): the message budget is used up (one message is kept for the steps owed at the finish). Do not message the shopper again.", name, args
        if name == self.roles.get("search") and self.parse_sure():
            # the code holds or tidies a model search only while its own reading of the store is trusted
            blind = self.blind_hold(args)
            if blind:
                return blind, name, args
            wordy = self.zero_hold(args)
            if wordy:
                return wordy, name, args
            args = self.search_hygiene(name, args)
        args = self.resolve_titles(name, args)
        cached = self.read_hold(name, args)
        if cached:
            return cached, name, args
        signature = name + _compact(args)
        if name not in finishing and name not in (message_tool, cart_tool) and signature in self.recent:
            return "Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else.", name, args
        if name == cart_tool and not self.viewing(name, args):
            held = self.cart_hold(name, args, signature)
            if held:
                return held, name, args
        if name == self.roles.get("cart_remove") and name != cart_tool and self.parse_sure():
            held = self.churn_hold(name, args)
            if held:
                return held, name, args
        return None, name, args

    def cart_hold(self, name: str, args: dict[str, Any], signature: str) -> str | None:
        """Holds on an add to the cart: a swap the shopper wants to hear about first, a narrower store category, a pick its
        records fail (checked once per ledger version, one product only) and a quantity other than the plan's. The holds
        that rest on the code's own reading of the store (churn, the code's placed or kept lines, the price walk) apply
        only while that reading is trusted."""
        sure = self.parse_sure()
        churn = self.churn_hold(name, args) if sure else None
        if churn:
            return churn
        sold = self.gone_hold(name, args)
        if sold:
            return sold
        placed = self.placed_hold(name, args) if sure else None
        if placed:
            return placed
        kept = self.keep_hold(name, args) if sure else None
        if kept:
            return kept
        swap = self.swap_hold(name, args)
        if swap:
            return swap
        walk = self.walk_hold(name, args) if sure else None
        if walk:
            return walk
        if self.steps_left() > 3 and not self.items():      # several products: each line is checked once, at the order
            properties = self.schemas[name].get("properties") or {}
            mode = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
            if self.add_signature is None or mode == self.add_signature:
                target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)
                if target is None and _listed_ids(args):
                    target = _listed_ids(args)[-1]
                if target and target in self.cart_ids and mode == self.add_signature and len(self.cart_ids) == 1:
                    # invariant: the one line already holds the asked quantity; a second add would double it
                    return (f"Not run (no step used): {target} is already the one line in the cart at the asked quantity; adding it again would double it. "
                            "Order it now, or take it out first if you want another listing.")
                shelf = self.narrower_shelf(target) if target else ""
                if shelf:
                    return "Not run (no step used). " + shelf + " If you are certain, send the same call again."
                if target and f"{target}|{self.ledger_version}" not in self.cart_checked:
                    self.cart_checked.add(f"{target}|{self.ledger_version}")
                    verdict = self.verify_item(target, "ADD TO THE CART", _line_of(args, target))
                    if verdict:
                        return "Not run (no step used). " + verdict + " If you are certain it fits, send the same call again."
        warning = self.quantity_warning(name, args)
        if warning and signature not in self.qty_warned:
            self.qty_warned.add(signature)
            return "Not run (no step used): " + warning + " If you are sure, send the same call again."
        return None

    def churn_hold(self, name: str, args: dict[str, Any]) -> str | None:
        """A cart call that takes out a line the model has already taken out and put back twice while nothing changed is
        held: the cart is complete as it stands and the order is the next move."""
        remover = name == self.roles.get("cart_remove") and name != self.roles.get("cart")
        if (name != self.roles.get("cart") and not remover) or self.basket_form():
            return None
        properties = self.schemas[name].get("properties") or {}
        mode = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() in self.cart_ids), None)
        if not target or (not remover and (self.add_signature is None or mode == self.add_signature)):
            return None
        mark = f"{target}|{self.ledger_version}|{len(self.gone)}|{self.budget_now}"
        self.churn[mark] = self.churn.get(mark, 0) + 1
        if self.churn[mark] <= 1 or target in self.gone:
            return None
        return (f"Not run (no step used): {target} was taken out and put back {self.churn[mark] - 1} times with nothing changed in between. "
                "The cart is complete as it stands: order it now (or name the condition a held line fails before changing it).")

    def gone_hold(self, name: str, args: dict[str, Any]) -> str | None:
        """An add of a listing the store already reported sold out never runs: it could only lead to a refused order. Every
        such attempt is answered without a step."""
        properties = self.schemas[name].get("properties") or {}
        mode = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if self.add_signature is not None and mode != self.add_signature:
            return None                      # taking a line out
        target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)
        if target is None and _listed_ids(args):
            target = _listed_ids(args)[-1]
        if not target or target not in self.gone:
            return None
        self.gone_held.add(target)
        self.trace.append({"kind": "hold", "gone_add": target})
        return (f"Not run (no step used): {target} was reported sold out, so putting it in the cart would only lead to another refused order. "
                + (self.fallback_note() or "Choose another listing that meets every requirement (search again or read the next result pages).")
                + " Pick a listing that is still on offer, or decline if nothing else meets every requirement.")

    def placed_hold(self, name: str, args: dict[str, Any]) -> str | None:
        """Taking out a line the code itself put in (the commit, a recovery, the cheapest swap) that still meets every condition
        by its own sheet (or an item check) is held every time, without a step, unless the model's words name a condition the
        line's own sheet contradicts."""
        if self.items() or self.add_signature is None or _listed_ids(args) is not None or not self.agent_placed:
            return None
        properties = self.schemas[name].get("properties") or {}
        mode = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if mode == self.add_signature:
            return None                      # an add
        target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)
        if not target or target not in self.agent_placed or target not in self.cart_ids or target in self.gone:
            return None
        row = self.shelf.rows.get(target) or {}
        budget, quantity = self.walk_scope()
        if row.get("stock") is False or row.get("price") is None or (budget is not None and row["price"] * quantity > budget + 1e-9):
            return None
        conds = self.conditions()
        if not conds or not (target in self.passed or all(self.shelf.says(target, p, v) == "yes" for p, v in conds)):
            return None
        if self.model_vetoes(target, conds):
            self.vetoed.add(target)
            return None
        self.placed_holds[target] = self.placed_holds.get(target, 0) + 1
        if self.placed_holds[target] > PLACED_HOLDS:
            return None                      # the model insists: the line comes out rather than the turns going to a loop
        self.trace.append({"kind": "hold", "placed_remove": target})
        lines = self.shelf.evidence_lines(target, conds)
        return (f"Not run (no step used): the code put {target} ({row['price']:g}) in the cart because its own sheet states every condition, "
                "and taking it out would only cost steps. Its sheet: " + "; ".join(lines) + ". Keys that differ only in spacing, underscores or "
                "case are one field and a list value that contains the required value states it; that is not a conflict. Order it now. It can "
                "be taken out only by naming the condition whose own value on this sheet contradicts the requirement.")

    def keep_hold(self, name: str, args: dict[str, Any]) -> str | None:
        """On a cheapest request, taking out a cart line whose own sheet states every condition (or that an item check passed)
        waits once, without a step, with the sheet's evidence; sending the call again takes it out."""
        if self.plan.get("priority") != "cheapest" or self.items() or self.add_signature is None or _listed_ids(args) is not None:
            return None
        properties = self.schemas[name].get("properties") or {}
        mode = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if mode == self.add_signature:
            return None                      # an add
        target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)
        if not target or target not in self.cart_ids or target in self.keep_held or target in self.gone:
            return None
        row = self.shelf.rows.get(target) or {}
        budget, quantity = self.walk_scope()
        if row.get("stock") is False or row.get("price") is None or (budget is not None and row["price"] * quantity > budget + 1e-9):
            return None
        conds = self.conditions()
        if not conds or not (target in self.passed or all(self.shelf.says(target, p, v) == "yes" for p, v in conds)):
            return None
        self.keep_held.add(target)
        lines = self.shelf.evidence_lines(target, conds)
        return (f"Not run (no step used): {target} ({row['price']:g}) is in the cart and its own sheet states every condition, so taking it out "
                f"would drop the cheapest fit. Its sheet: " + "; ".join(lines) + ". Keys that differ only in spacing, underscores or case are one field "
                "and a list value that contains the required value states it; that is not a conflict. Keep it and order. Only if one condition's own "
                "value on this sheet contradicts the requirement, say which and send the same call again.")

    def cart_shape(self) -> dict[str, Any] | None:
        """How the cart tool names one listing at a time: the id argument, the one enum argument that picks the mode, the add
        value (from the first add) and the value that takes a line out; None when the tool takes the cart as a list, has no
        single mode or no value for taking out."""
        name = self.roles.get("cart")
        if not name or not self.add_args:
            return None
        function = self.schemas_full.get(name) or {}
        props = (function.get("parameters") or {}).get("properties") or {}
        if any(isinstance(v, dict) and v.get("type") == "array" for v in props.values()):
            return None
        modes = [k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get("enum"), list)]
        if len(modes) != 1:
            return None
        mode_key = modes[0]
        strings = [k for k, v in props.items() if isinstance(v, dict) and "enum" not in v and "string" in str(v.get("type"))]
        id_key = (next((k for k in strings if flat(k) in self.shelf.id_keys), None)
                  or next((k for k in strings if re.search(IDDY, flat(re.sub(r"[_\-]+", " ", k)))), None) or (strings[0] if strings else None))
        add = self.add_args.get(mode_key)
        if not id_key or not isinstance(add, str):
            return None
        pair = mode_pair(props[mode_key]["enum"], self.plan.get("cart_modes"))
        out = pair[1] if pair[1] != add else next((v for v in props[mode_key]["enum"] if isinstance(v, str) and v != add and _OUT_WORDS.search(v)), None)
        if out is None:
            return None
        template = {k: v for k, v in self.add_args.items() if k not in (mode_key, id_key)}
        return {"function": function, "mode_key": mode_key, "id_key": id_key, "add": add, "out": out, "template": template}

    def cart_args(self, adding: bool, target: str) -> dict[str, Any] | None:
        shape = self.cart_shape()
        if not shape:
            return None
        args = shape_args(shape["function"], {**shape["template"], shape["mode_key"]: shape["add"] if adding else shape["out"], shape["id_key"]: target})
        return None if lacking(shape["function"], args) else args

    def set_aside(self, key: str) -> bool:
        """A listing the code must not commit to on its own: one the model itself took out of the cart, or one the model's
        latest words named in the same sentence as a doubt about its sheet (conflicting or mixed labels, a set or bundle
        rather than the product). The model's reading of a sheet stands where the code's label rules see nothing."""
        if key in self.model_removed or self.fit_blocked(key):
            return True
        said = self.last_content or ""
        if not said or key.lower() not in said.lower():
            return False
        for sentence in re.split(r"[.;\n]+", said):
            if key.lower() in sentence.lower() and _DOUBT.search(sentence):
                return True
        return False

    def stray_figures(self, key: str, conds: list[tuple[str, str]]) -> int:
        """How many other fields of the listing's sheet carry a figure of the same shape (digit-run length) as a figure in a
        condition's value: '1080p' or '1920x1080' beside a '3440 x 1440' resolution. A tie-breaker only; 0 when the
        conditions carry no figures."""
        sheet = self.shelf.sheets.get(key) or {}
        count = 0
        for prop, value in conds:
            shapes = {len(m) for m in re.findall(r"\d+", str(value))}
            if not shapes:
                continue
            pw = [w for w in words(prop) if w not in ("type", "kind", "the", "of")]
            for k, v in sheet.items():
                if pw and all(any(h == w or (len(w) >= 4 and h.startswith(w)) for h in words(k)) for w in pw):
                    continue                                   # the field naming the property itself
                runs = {len(m) for m in re.findall(r"\d+", str(v))}
                if runs & shapes:
                    count += 1
        return count

    def model_vetoes(self, key: str, conds: list[tuple[str, str]]) -> bool:
        """The model's latest words rule a candidate out only by naming a condition's property that the candidate's own
        sheet contradicts (a 'label conflict' with nothing contradicted is no veto)."""
        said = set(words(self.last_content))
        if not said:
            return False
        for prop, value in conds:
            if self.shelf.says(key, prop, value) == "no" and any(w in said for w in words(prop) if len(w) >= 3):
                return True
        return False

    def enforce_cheapest(self) -> str:
        """Right before an order on a cheapest request, once per episode: when a listing that states every condition (or an
        item check passed), still on offer and within the limit, is cheaper than the one cart line, the code itself takes
        that line out and puts the cheaper one in (two store steps), then the order goes ahead. The model can stop it only
        by naming a condition the cheaper listing's own sheet contradicts. Never with several products or several lines."""
        if (self.swap_done or self.plan.get("priority") != "cheapest" or self.items() or len(self.cart_ids) != 1 or not self.roles.get("cart")
                or self.unfolded or (self.roles.get("message") and self.messages_used == 0 and self.messages_open() > 0)):
            return ""                        # (an unread reply, or a shopper never asked, may hold a need the cheaper line fails)
        current = self.cart_ids[0]
        price = (self.shelf.rows.get(current) or {}).get("price")
        conds = self.conditions()
        if price is None or not conds or self.steps_left() < self.finish_reserve() + 2 or self.time_left() < WRAP_UP_SECONDS + 10:
            return ""
        best = ""
        for p, key in self.fitting_rows():
            if key == current or key in self.gone or key in self.vetoed or p >= price - 1e-9 or self.set_aside(key):
                continue
            if self.model_vetoes(key, conds):
                self.vetoed.add(key)
                continue
            best = key
            break
        if not best:
            return ""
        out_call, in_call = self.remove_args(current), self.add_call_args(best)
        if not out_call or not in_call:
            return ""
        out_args, in_args = out_call[1], in_call[1]
        self.swap_done = True
        self.agent_placed.add(best)
        cheaper = self.shelf.rows[best]["price"]
        for tool, args in (out_call, in_call):
            reply = self.env_call([{"name": tool, "args": args}])
            self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": args}], "reply": reply, "swap": True})
            if reply is None or self.finished or self.dead:
                return ""
            observation = reply.get("observation")
            self.code_booking(tool, args, observation)
            self.read_notices(tool, observation)
            if args is in_args:
                said = _compact(observation)
                if (isinstance(observation, dict) and observation.get("error")) or self.gone_said(said):
                    self.gone.add(best)
                    taken = self.remove_args(best) if not (isinstance(observation, dict) and observation.get("error")) else None
                    if taken and self.steps_left() >= self.finish_reserve() + 2:
                        # the add went through before the store said the line is gone: it comes out again
                        gone_reply = self.env_call([{"name": taken[0], "args": taken[1]}])
                        if gone_reply is not None:
                            self.code_booking(taken[0], taken[1], gone_reply.get("observation"))
                    self.cart_ids = [i for i in self.cart_ids if i != best]
                    back_call = self.add_call_args(current)
                    back = back_call[1] if back_call else None
                    tool = back_call[0] if back_call else tool
                    if back and self.steps_left() >= self.finish_reserve() + 1 and current not in self.gone:
                        again = self.env_call([{"name": tool, "args": back}])
                        self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": back}], "reply": again, "swap": True})
                        if again is not None:
                            self.code_booking(tool, back, again.get("observation"))
                    note = f"[cheapest rule applied by the code] {best} ({cheaper:g}) could not be put in the cart ({said[:160]}); the cart holds {current} ({price:g}) again."
                    self.trace.append({"kind": "swap", "from": current, "to": best, "failed": True})
                    return note
        note = (f"[cheapest rule applied by the code: 2 store steps] The cart now holds {best} ({cheaper:g}) instead of {current} ({price:g}): "
                f"it is cheaper and its own sheet states every condition: " + "; ".join(self.shelf.evidence_lines(best, conds)) +
                ". Keys that differ only in spacing, underscores or case are one field, and a list value that contains the required value states it. "
                "The order goes ahead with it; if it is refused, fix the cart and order again.")
        self.trace.append({"kind": "swap", "from": current, "to": best, "note": note})
        return note

    def basket_form(self) -> dict[str, Any] | None:
        """How a cart tool that takes the whole basket as one list names its lines: the list argument, and inside each line
        the id field and the count field (from the schema's line object, else from the model's own earlier call). None when
        the cart tool names one listing at a time."""
        tool = self.roles.get("cart")
        function = self.schemas_full.get(tool) or {} if tool else {}
        props = (function.get("parameters") or {}).get("properties") or {}
        lists = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array"]
        if len(lists) != 1:
            return None
        key = lists[0]
        line = props[key].get("items") if isinstance(props[key].get("items"), dict) else {}
        fields = line.get("properties") if isinstance(line.get("properties"), dict) else {}
        form: dict[str, Any] = {"tool": tool, "function": function, "key": key, "plain": False, "id": None, "count": None,
                                "template": {k: v for k, v in (self.add_args or {}).items() if k != key and k in props}}
        if not fields and line.get("type") == "string":
            remover = self.roles.get("cart_remove")
            if remover and remover != tool:
                return None                    # ids added on top of the basket, with their own tool to take lines out
            form["plain"] = True
            return form
        if not fields:
            seen = (self.add_args or {}).get(key)
            sample = seen[0] if isinstance(seen, list) and seen and isinstance(seen[0], dict) else {}
            fields = {k: {"type": "integer" if isinstance(v, int) and not isinstance(v, bool) else "string"} for k, v in sample.items()}
        strings = [k for k, v in fields.items() if isinstance(v, dict) and "string" in str(v.get("type")) and "enum" not in v]
        form["id"] = (next((k for k in strings if flat(k) in self.shelf.id_keys), None)
                      or next((k for k in strings if re.search(IDDY, flat(re.sub(r"[_\-]+", " ", k)))), None) or (strings[0] if strings else None))
        counts = [k for k, v in fields.items() if isinstance(v, dict) and v.get("type") in ("integer", "number")]
        form["count"] = counts[0] if counts else None
        return form if form["id"] else None

    def merge_basket(self, args: dict[str, Any]) -> dict[str, Any]:
        """A whole-basket cart call from the model that puts a new line in while leaving out the held lines of OTHER products
        (race 18: champ23 re-held one line at a time and each hold emptied the rest of the bundle): the held lines of the
        products the call does not cover are kept in it, with their counts. A call that only drops or recounts lines, or that
        replaces a line of the same product, is the model's own choice and goes as sent."""
        form = self.basket_form()
        if not form or not self.items() or not self.cart_ids or not isinstance(args.get(form["key"]), list):
            return args
        listed = args[form["key"]]
        if form["plain"]:
            ids = [x.strip() for x in listed if isinstance(x, str)]
        else:
            ids = [str(line.get(form["id"])).strip() for line in listed if isinstance(line, dict) and isinstance(line.get(form["id"]), str)]
        if not ids or not [i for i in ids if i not in self.cart_ids]:
            return args
        covered = {self.product_of(i) for i in ids} - {""}
        keep = []
        for key in self.cart_ids:
            owner = self.product_of(key, self.cart_lines.get(key) or "")
            if key in ids or key in self.gone or not owner or owner in covered or owner in {self.product_of(k) for k in keep}:
                continue
            keep.append(key)
        if not keep:
            return args
        if form["plain"]:
            extra: list[Any] = list(keep)
        else:
            extra = [{form["id"]: k, **({form["count"]: max(1, int(self.basket_counts.get(k, self.line_count(k))))} if form["count"] else {})} for k in keep]
        merged = dict(args)
        merged[form["key"]] = list(listed) + extra
        self.trace.append({"kind": "merge_basket", "kept": keep, "step": self.step})
        return merged

    def line_count(self, key: str) -> int:
        """How many units a line of this listing should carry: its product's asked quantity (several products), else the
        request's quantity."""
        if self.items():
            product = self.product_of(key)
            for name, quantity in self.items():
                if name == product:
                    return quantity
            return self.basket_counts.get(key, 1)
        return self.walk_scope()[1]

    def basket_call(self, lines: list[tuple[str, int]]) -> tuple[str, dict[str, Any]] | None:
        """The cart call that sets the whole basket to these lines (a whole-basket cart tool only)."""
        form = self.basket_form()
        if not form:
            return None
        if form["plain"]:
            listed: list[Any] = [key for key, _ in lines]
        else:
            listed = [{form["id"]: key, **({form["count"]: max(1, int(count))} if form["count"] else {})} for key, count in lines]
        args = shape_args(form["function"], {**form["template"], form["key"]: listed})
        args[form["key"]] = listed
        return None if lacking(form["function"], args) else (form["tool"], args)

    def basket_with(self, target: str) -> tuple[str, dict[str, Any]] | None:
        """The whole basket with this listing put in: with one product it replaces the held line; with several it replaces the
        line of the same product and keeps the others with their counts."""
        if not self.items():
            return self.basket_call([(target, self.line_count(target))])
        product = self.product_of(target)
        keep = [(k, self.basket_counts.get(k, self.line_count(k))) for k in self.cart_ids
                if k != target and not (product and self.product_of(k) == product)]
        return self.basket_call(keep + [(target, self.line_count(target))])

    def add_call_args(self, target: str) -> tuple[str, dict[str, Any]] | None:
        """The cart tool and the arguments that put one listing in: the shape of the first add when there was one, else read
        from the tool's schema (the enum value that adds, the id argument, the asked quantity)."""
        tool = self.roles.get("cart")
        if not tool:
            return None
        if self.basket_form():
            return self.basket_with(target)
        args = self.cart_args(True, target)
        if args:
            return tool, args
        function = self.schemas_full.get(tool) or {}
        props = (function.get("parameters") or {}).get("properties") or {}
        id_lists = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array" and (v.get("items") or {}).get("type") == "string"]
        if id_lists:
            units = 1 if any(isinstance(v, dict) and v.get("type") in ("integer", "number") for v in props.values()) else max(1, int(self.walk_scope()[1] or 1))
            args = shape_args(function, {id_lists[0]: [target] * units})
            return None if lacking(function, args) else (tool, args)
        if any(isinstance(v, dict) and v.get("type") == "array" for v in props.values()):
            return None
        strings = [k for k, v in props.items() if isinstance(v, dict) and "enum" not in v and "string" in str(v.get("type"))]
        id_key = (next((k for k in strings if flat(k) in self.shelf.id_keys), None)
                  or next((k for k in strings if re.search(IDDY, flat(re.sub(r"[_\-]+", " ", k)))), None) or (strings[0] if strings else None))
        if not id_key:
            return None
        raw: dict[str, Any] = {id_key: target}
        for k, v in props.items():
            if isinstance(v, dict) and isinstance(v.get("enum"), list):
                adding = mode_pair(v["enum"], self.plan.get("cart_modes"))[0]
                if adding is None:
                    return None
                raw[k] = adding
            elif isinstance(v, dict) and v.get("type") in ("integer", "number"):
                raw[k] = self.walk_scope()[1]    # the one figure a cart tool takes is how many units
        args = shape_args(function, raw)
        return None if lacking(function, args) else (tool, args)

    def commit_mark(self) -> int:
        """The step from which the code itself puts a known fit in an empty cart: half the cap once the questions are done,
        else the pace schedule's commit mark."""
        cap = max(1, self.max_steps)
        return max(6, round(cap * COMMIT_SHARE)) if not self.planner_on else self.pace_marks()[1]

    def paced_fit(self) -> str:
        """The listing the code would commit to: one an item check passed under the current conditions, else one whose own
        sheet states every condition; in stock, within the limit, not gone, failed by no check, seen since the last change."""
        conds = self.conditions()
        if not conds:
            return ""
        budget, quantity = self.walk_scope()
        pool = self.shelf.fits(conds, budget, quantity, "", self.gone)
        pool = sorted(pool, key=lambda k: (k not in self.passed, self.stray_figures(k, conds) > 0, (self.shelf.rows.get(k) or {}).get("price") or 1e12))
        for key in pool:
            row = self.shelf.rows.get(key) or {}
            if key in self.gone or key in self.vetoed or self.set_aside(key) or row.get("stock") is False or row.get("price") is None:
                continue
            if budget is not None and row["price"] * quantity > budget + 1e-9:
                continue
            if any(k.startswith(key + "|") and k.endswith(f"|{self.ledger_version}") and v for k, v in self.verdicts.items()):
                continue
            if self.market_at >= 0 and row.get("step", 0) <= self.market_at:
                continue
            if self.model_vetoes(key, conds):
                self.vetoed.add(key)
                continue
            return key
        return ""

    def code_booking(self, tool: str, args: dict[str, Any], observation: Any) -> None:
        """Books a cart call of the code's own, so that the line it takes out is not counted as one the model removed."""
        self.code_move = True
        try:
            self.book_call(tool, args, observation)
        finally:
            self.code_move = False

    def questions_open(self) -> bool:
        """True while something the shopper must be asked is still unasked: an ask-first item before any question went out, or
        a property the story hints at that no question named yet, while messages remain."""
        if self.plan.get("ask_first") and not self.asked:
            return True
        limit = self.message_budget - self.held_back() if self.message_budget is not None else UNSTATED_MESSAGES
        if self.messages_used >= limit or not self.roles.get("message"):
            return False
        asked = " ".join(self.asked_props + [a.lower() for a in self.asked]).lower()
        for hint in self.plan.get("hint_properties") or []:
            if isinstance(hint, str) and hint.strip() and not any(w in asked for w in words(hint) if len(w) >= 3):
                return True
        return False

    def commit_fit(self) -> bool:
        """At the commit mark, with the questions done, no order out and nothing in the cart, the code itself puts the known fit
        in the cart (one step) and asks the model to order it, or to name the condition its own sheet contradicts. Once per
        episode; not under a lowest-price rule (the swap at the order covers that), not with several products, not while a
        change is still to be reviewed, not while an ask-first item or a hinted property is still unasked, and never with a
        line the model itself took out. True when a turn was spent."""
        if (self.committed or self.plan.get("priority") == "cheapest" or self.items() or self.order_attempts or self.cart_ids
                or self.max_steps < CODE_MOVES_FROM or self.planner_on or self.review_reason or self.step < self.commit_mark() or not self.roles.get("cart")
                or self.steps_left() < self.finish_reserve() + 3 or self.cost >= self.cost_stop or self.time_left() < WRAP_UP_SECONDS + 20
                or self.questions_open()):
            return False
        try:
            fit = self.paced_fit()
            if not fit or fit in self.model_removed:
                return False
            found = self.add_call_args(fit)
            if not found:
                return False
            tool, args = found
            self.committed = fit
            self.agent_placed.add(fit)
            reply = self.env_call([{"name": tool, "args": args}])
            self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": args}], "reply": reply, "commit": True})
            if reply is None or self.finished or self.dead:
                return True
            observation = reply.get("observation")
            self.code_booking(tool, args, observation)
            self.read_notices(tool, observation)
            note = self.note_observation(tool, observation)
            row = self.shelf.rows.get(fit) or {}
            said = _compact(observation)
            if (isinstance(observation, dict) and observation.get("error")) or fit in self.gone or self.gone_said(said):
                self.gone.add(fit)
                self.cart_ids = [i for i in self.cart_ids if i != fit]
                text = f"[note] The code tried to put {fit} in the cart (1 store step), but the store refused: {said[:200]}. Choose the next fit."
            else:
                lines = self.shelf.evidence_lines(fit, self.conditions())
                text = (f"[note] Step {self.step} of {self.max_steps}: the questions are done and {fit} ({row.get('price', '?')}) meets every condition by its "
                        f"own sheet, so the code put it in the cart (1 store step). Its sheet: " + "; ".join(lines) + ". ORDER IT NOW with the order tool. "
                        "Only if one condition's own value on this sheet contradicts the requirement, say which condition and fix the cart instead."
                        + (" " + note if note else ""))
            self.trace.append({"kind": "commit", "fit": fit, "note": text[:300]})
            self.messages.append({"role": "user", "content": text + "\n" + self.status()})
            return True
        except Exception as exc:
            self.trace.append({"kind": "commit", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return bool(self.committed)

    def questions_done(self) -> bool:
        """Whether nothing more is to be asked: no planned or hinted question open, and the messages spent or the planner done."""
        limit = self.message_budget - self.held_back() if self.message_budget is not None else UNSTATED_MESSAGES
        return not self.questions_open() and (self.messages_used >= limit or not self.planner_on or not self.roles.get("message"))

    def recoveries_spent(self) -> bool:
        """Whether the code has recovered from the current announced change as often as the world allows."""
        return self.recoveries.get(self.market_at, 0) >= (LONG_RECOVERIES if self.long_world() else 1)

    def recover(self) -> bool:
        """After the one cart line was reported sold out, the code itself replaces it with the best listing still on offer (two
        steps: the gone line out, the fit in) and sends the order through the usual holds, instead of leaving the model to
        re-read and re-cart. Once per announced change; not with several products, not when the request wants the shopper
        asked before a swap, never toward a listing the model's words rule out by a condition its sheet contradicts, never
        toward one the model took out or called conflicting, mixed or a set. Under a priority other than lowest price the
        code swaps and orders only when an independent check passed the candidate under the current conditions; otherwise
        it only clears the sold-out line and lists the candidates with their evidence, and the model chooses and orders."""
        if (len(self.cart_ids) != 1 or self.cart_ids[0] not in self.gone or self.items() or self.recoveries_spent() or self.max_steps < CODE_MOVES_FROM
                or self.open_duties({"before_replace"}) or not self.roles.get("cart") or not self.roles.get("order")
                or self.steps_left() < self.finish_reserve() + 3 or self.cost >= self.cost_stop or self.time_left() < WRAP_UP_SECONDS + 20):
            return False
        conds = self.conditions()
        if not conds:
            return False
        self.recovered_at = self.market_at
        self.recoveries[self.market_at] = self.recoveries.get(self.market_at, 0) + 1
        gone = self.cart_ids[0]
        budget, quantity = self.walk_scope()
        cheapest_rule = self.plan.get("priority") == "cheapest"
        pool = self.shelf.fits(conds, budget, quantity, "", self.gone)      # sheet-stated fits, cheapest first
        checked = [k for k in self.passed if self.passed_at.get(k) == self.ledger_version]
        direct = lambda k: not all(self.shelf.direct_states(k, p, v) for p, v in conds)
        if not cheapest_rule:
            pool = sorted(dict.fromkeys(pool + checked), key=lambda k: (k not in checked, direct(k), self.stray_figures(k, conds) > 0, (self.shelf.rows.get(k) or {}).get("price") or 1e12))
            if self.plan.get("priority") == "best":
                # the request's rule ranks the replacements: the best by the named measure, not the first fit
                pool = self.rank_by_rule(pool)
        else:
            pool = sorted(pool, key=lambda k: ((self.shelf.rows.get(k) or {}).get("price") or 1e12, direct(k), self.stray_figures(k, conds) > 0))
        best = ""
        listed = []
        for key in pool:
            row = self.shelf.rows.get(key) or {}
            if key in self.gone or key in self.vetoed or key == gone or self.set_aside(key) or row.get("stock") is False or row.get("price") is None:
                continue
            if budget is not None and row["price"] * quantity > budget + 1e-9:
                continue
            if self.model_vetoes(key, conds):
                self.vetoed.add(key)
                continue
            listed.append(key)
            if not best and (cheapest_rule or key in checked):
                best = key
        if not best and not listed and self.market_at not in self.stock_refilled:
            # nothing opened fits any more: the cheapest unread rows within the limit are opened first (a step each), then the
            # recovery looks again
            self.stock_refilled.add(self.market_at)
            unread = sorted((row["price"], key) for key, row in self.shelf.rows.items() if not row.get("sheet") and key not in self.gone and key != gone
                            and row.get("stock") is not False and row.get("price") is not None and (budget is None or row["price"] * quantity <= budget + 1e-9))
            keys = [k for _, k in unread][:3]
            at = self.market_at
            if keys and self.steps_left() >= self.finish_reserve() + 4 + len(keys) and self.code_reads(keys):
                # the reads may announce a new change (a new key); only this change's count is given back
                self.recoveries[at] = max(0, self.recoveries.get(at, 0) - 1)
                self.stock_refilled.add(self.market_at)     # one refill per stockout, even when the reads re-announced it
                if self.finished or self.dead:
                    return True
                return self.recover()
        if not best and listed and self.long_world() and self.cost < self.cost_hard and self.questions_done():
            # a long world affords the check the swap needs: the best-placed sheet fits are checked by the code, and the first
            # that passes is swapped in and ordered without the turn going back to the model
            for key in listed[:2]:
                if self.verify_item(key, "REPLACE THE SOLD-OUT LINE WITH") == "" and self.passed_at.get(key) == self.ledger_version:
                    best = key
                    break
                if self.cost >= self.cost_hard:
                    break
        out_call = self.remove_args(gone)
        out_tool, out_args = out_call if out_call else (self.roles.get("cart"), None)
        if not best:
            if not listed or not out_args:
                return False
            # no candidate an independent check approved: the code only clears the sold-out line and shows what fits
            self.recovered_at = self.market_at
            reply = self.env_call([{"name": out_tool, "args": out_args}])
            self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": out_tool, "args": out_args}], "reply": reply, "recover": True})
            if reply is None or self.finished or self.dead:
                return True
            self.code_booking(out_tool, out_args, reply.get("observation"))
            self.read_notices(out_tool, reply.get("observation"))
            shown = []
            for key in listed[:4]:
                price = (self.shelf.rows.get(key) or {}).get("price")
                shown.append(f"{key} ({price:g}): " + "; ".join(self.shelf.evidence_lines(key, conds)) if isinstance(price, (int, float))
                             else f"{key}: " + "; ".join(self.shelf.evidence_lines(key, conds)))
            self.messages.append({"role": "user", "content": (
                f"[recovery by the code: 1 store step] {gone} was reported sold out, so the code took it out of the cart. No listing has passed an "
                f"independent check under the current conditions, so the choice is yours. Listings whose own sheets state every condition, cheapest first: "
                + " | ".join(shown) + ". Put the one you judge right in the cart and order it; if none fits the shopper's words, decline.\n" + self.status())})
            self.trace.append({"kind": "recover", "gone": gone, "cleared": True, "shown": listed[:4]})
            return True
        found = self.add_call_args(best)
        if not found or not out_args:
            return False
        tool, in_args = found
        self.agent_placed.add(best)
        try:
            for tool, args in ((out_tool, out_args), (tool, in_args)):
                reply = self.env_call([{"name": tool, "args": args}])
                self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": args}], "reply": reply, "recover": True})
                if reply is None or self.finished or self.dead:
                    return True
                observation = reply.get("observation")
                self.code_booking(tool, args, observation)
                self.read_notices(tool, observation)
                if args is in_args:
                    said = _compact(observation)
                    if (isinstance(observation, dict) and observation.get("error")) or self.gone_said(said):
                        self.gone.add(best)
                        self.cart_ids = [i for i in self.cart_ids if i != best]
                        self.messages.append({"role": "user", "content": f"[note] The code took the sold-out {gone} out of the cart (1 store step), but {best} "
                                                                         f"could not be put in ({said[:160]}). Put the next fit in the cart, then order.\n" + self.status()})
                        return True
            row = self.shelf.rows.get(best) or {}
            lines = self.shelf.evidence_lines(best, conds)
            self.messages.append({"role": "user", "content": (
                f"[recovery by the code: 2 store steps] {gone} was reported sold out, so the cart now holds {best} ({row.get('price', '?')}"
                + (", passed by an independent check" if best in self.passed else "") + ") instead; its own sheet: " + "; ".join(lines)
                + ". The order goes out now through the usual checks; if it is held or refused, fix the cart and order again.")})
            order = self.roles["order"]
            call = {"id": f"recover_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(self.neutral_args(order, "Order the cart."))}}
            self.messages.append({"role": "assistant", "content": f"I order {best} now.", "tool_calls": [call]})
            if not self.shelf.unopened_cheaper(best, budget, quantity):
                self.research_checks = max(1, self.research_checks)    # every cheaper row was opened: nothing to search again for
            replies = self.run_calls([call], planned=True)
            for call_id, reply_text in replies or [(call["id"], "Not sent.")]:
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text + "\n" + self.status()})
            self.trace.append({"kind": "recover", "gone": gone, "to": best, "ordered": self.finished, "reply": [r[:200] for _, r in replies or []]})
        except Exception as exc:
            self.trace.append({"kind": "recover", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return True

    def refill_candidates(self) -> bool:
        """After a stockout or a refused order on a cheapest request, when no listing still on offer fits at the cheapest
        price a fit was known at, the code looks for one itself before any dearer fallback is proposed: it pages the model's
        own short keyword search to its end (at most two pages) and opens the cheapest unread rows at or under that price
        (at most two), within REFILL_STEPS for the episode and the finish reserve. Once per announced change; True when a
        turn was spent on it."""
        if (self.refilled_at == self.market_at or not (self.gone or self.refusals) or self.plan.get("priority") != "cheapest" or self.items()
                or self.refill_used >= REFILL_STEPS or self.cost >= self.cost_hard or not self.roles.get("search") or not self.conditions()
                or self.steps_left() < self.finish_reserve() + 3 or self.time_left() < WRAP_UP_SECONDS + 45):
            return False
        self.refilled_at = self.market_at
        conds = self.conditions()
        budget, quantity = self.walk_scope()
        known = set(self.passed) | set(self.shelf.fits(conds, budget, quantity, "", set())) | set(self.gone)
        prices = [self.shelf.rows[k]["price"] for k in known if (self.shelf.rows.get(k) or {}).get("price") is not None]
        if not prices:
            return False
        floor = float(min(prices))
        if any(p <= floor + 1e-9 for p, _ in self.fitting_rows()):
            return False                     # a fit at that price is still on offer
        allowance = min(REFILL_STEPS - self.refill_used, self.steps_left() - self.finish_reserve() - 1)
        if allowance < 1:
            return False
        keys = self.search_keys()
        used, fresh = 0, []
        try:
            source = self.keyword_search(floor, "", any_group=True) if keys["page"] else None
            if source is not None:
                shown, total, page = list(source["shown"]), source["total"], 2
                while len(set(shown)) < total and used < min(2, allowance) and self.time_left() >= WRAP_UP_SECONDS + 30:
                    args = shape_args(keys["function"], {**source["args"], keys["page"]: page})
                    if lacking(keys["function"], args):
                        break
                    observation, ok = self.walk_search(args)
                    if not ok or self.finished or self.dead:
                        break
                    used += 1
                    self.walk_used -= 1          # these steps count against the refill allowance, not the walk's
                    self.refill_used += 1
                    listed = [k for k, r in self.shelf.rows.items() if r.get("step") == self.step]
                    if not listed:
                        break
                    shown += listed
                    fresh += listed
                    page += 1
                source["shown"] = shown
            if not self.finished and not self.dead:
                unread = sorted((k for k, r in self.shelf.rows.items() if r.get("price") is not None and r["price"] <= floor + 1e-9
                                 and not r.get("sheet") and r.get("stock") is not False and k not in self.gone
                                 and (budget is None or r["price"] * quantity <= budget + 1e-9)), key=lambda k: (self.shelf.rows[k]["price"], k))
                room = min(2, allowance - used)
                if unread and room > 0:
                    got = self.walk_reads(unread[:room])
                    used += got
                    self.walk_used -= got
                    self.refill_used += got
        except Exception as exc:
            self.walking = False
            self.trace.append({"kind": "refill_fault", "error": f"{type(exc).__name__}: {exc}"[:300]})
        if not used:
            return False
        found = [(p, k) for p, k in self.fitting_rows() if p <= floor + 1e-9]
        named = ", ".join(f"{k} ({p:g}" + (", checked" if k in self.passed else "") + ")" for p, k in found[:4])
        source_text = f"{len(set(fresh))} more rows from the next page(s) of your search" if fresh else "the cheapest unread rows opened"
        note = (f"[note] After the stockout the code looked for another listing at the cheapest fitting price ({floor:g}): {used} store step(s), {source_text}. "
                + (f"Still on offer and stating every condition at that price: {named}. Put the cheapest of them in the cart in place of the gone line, then order."
                   if found else f"Nothing else in stock at {floor:g} states every condition. " + (self.fallback_note() or "The next fits, if any, are dearer; search again or decline if nothing fits.")))
        self.trace.append({"kind": "refill", "floor": floor, "steps": used, "found": [k for _, k in found[:4]]})
        self.messages.append({"role": "user", "content": note + "\n" + self.status()})
        return True

    def page_early(self, force: bool = False) -> bool:
        """On a cheapest request, once per episode while steps are plentiful: the model's own keyword search with a short list
        (not fully shown) is paged to its end by the code, within the walk's step allowance, and the cheapest new rows are
        named to the model. True when a turn was spent on it."""
        if not force and (self.early_paged or self.plan.get("priority") != "cheapest" or self.items() or self.cart_ids
                or self.steps_left() < EARLY_PAGE_FROM or self.walk_used >= WALK_STEPS or self.cost >= self.cost_hard
                or self.time_left() < WRAP_UP_SECONDS + 45 or not self.roles.get("search") or not self.conditions()):
            return False
        if force and (self.items() or not self.roles.get("search") or self.time_left() < WRAP_UP_SECONDS + 20):
            return False
        source = self.keyword_search(0.0 if force else float("inf"), "", any_group=True, any_text=force)
        if source is None:
            return False
        self.early_paged = True
        keys = self.search_keys()
        allowance = min(WALK_STEPS - self.walk_used, self.steps_left() - 6 - self.held_back()) if not force else min(2, self.steps_left() - self.finish_reserve() - 3)
        if allowance < 1:
            return False
        shown, total, page, used, fresh = list(source["shown"]), source["total"], 2, 0, []
        capped = False                       # the pages ended before the reported total: the store shows only part of the list
        try:
            last = bool(source.get("last"))
            while len(set(shown)) < total and used < allowance and self.time_left() >= WRAP_UP_SECONDS + 30 and not last:
                args = shape_args(keys["function"], {**source["args"], keys["page"]: page})
                if lacking(keys["function"], args):
                    break
                observation, ok = self.walk_search(args)
                if not ok or self.finished or self.dead:
                    break
                used += 1
                listed = [k for k, r in self.shelf.rows.items() if r.get("step") == self.step and k not in shown]
                if not listed:
                    last = True              # a page past the end repeats the last one
                    break
                shown += listed
                fresh += listed
                page += 1
                last = last_page(observation)
            capped = last and len(set(shown)) < total
            if capped and keys["cap"] and used < allowance + 1 and self.steps_left() >= EARLY_PAGE_FROM - 2:
                # the hidden part of the list may hold the cheapest rows: the same search under a price limit at the lower
                # quartile of the prices shown brings the cheap end of the whole list up
                budget, quantity = self.walk_scope()
                prices = sorted(r["price"] for k in set(shown) for r in [self.shelf.rows.get(k) or {}]
                                if r.get("price") is not None and (budget is None or r["price"] * quantity <= budget + 1e-9))
                if len(prices) >= 3:
                    whole = "integer" in str(((keys["function"].get("parameters") or {}).get("properties") or {}).get(keys["cap"], {}).get("type"))
                    middle = prices[len(prices) // 4] / (self.shelf.factor or 1.0)
                    limit = int(middle) if whole else round(middle, 2)
                    args = shape_args(keys["function"], {k: v for k, v in source["args"].items() if k != keys["page"]} | {keys["cap"]: limit})
                    if not lacking(keys["function"], args):
                        observation, ok = self.walk_search(args)
                        if ok and not (self.finished or self.dead):
                            used += 1
                            listed = [k for k, r in self.shelf.rows.items() if r.get("step") == self.step and k not in shown]
                            shown += listed
                            fresh += listed
                            self.trace.append({"kind": "capped_list", "total": total, "seen": len(set(shown)), "limit": limit, "new": len(listed)})
        except Exception as exc:
            self.walking = False
            self.trace.append({"kind": "walk_fault", "error": f"{type(exc).__name__}: {exc}"[:300]})
        source["shown"] = shown
        if not used:
            return False
        budget, quantity = self.walk_scope()
        rows = [(r["price"], k) for k in dict.fromkeys(fresh) for r in [self.shelf.rows.get(k) or {}]
                if r.get("price") is not None and r.get("stock") is not False and (budget is None or r["price"] * quantity <= budget + 1e-9)]
        named = ", ".join(f"{k} ({p:g}" + (f", {self.shelf.rows[k]['title'][:40]}" if self.shelf.rows[k].get("title") else "") + ")" for p, k in sorted(rows)[:6])
        text = str(source["args"].get(keys["text"]) or "")
        note = (f"[note] The code read the next page(s) of your search \"{text}\" to the end ({used} store step(s), {len(set(fresh))} more rows). "
                + (f"Cheapest of them within the limit: {named}. Open the cheapest candidates' details before choosing: the request asks for the cheapest fit."
                   if named else "None of them is in stock within the limit."))
        self.trace.append({"kind": "early_pages", "text": text, "steps": used, "rows": len(set(fresh))})
        self.messages.append({"role": "user", "content": note + "\n" + self.status()})
        return True

    def swap_hold(self, name: str, args: dict[str, Any]) -> str | None:
        """When a step says to check with the shopper before swapping, a new listing that takes the place of an earlier
        choice waits until the shopper has been asked (a code-written question, one per listing)."""
        if not self.open_duties({"before_replace"}) or not (self.cart_ids or self.gone or self.refusals):
            return None
        properties = self.schemas[name].get("properties") or {}
        mode = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if self.add_signature is not None and mode != self.add_signature:
            return None                      # taking a line out is not a swap
        target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)
        if target is None and _listed_ids(args):
            target = _listed_ids(args)[-1]
        if not target or target in self.cart_ids or target in self.swap_asked:
            return None
        if self.messages_open() < 1 or self.steps_left() <= 4:
            return None
        self.swap_asked.add(target)
        earlier = [i for i in self.cart_ids if i != target] or sorted(self.gone)
        row = self.shelf.rows.get(target) or {}
        old = (self.shelf.rows.get(earlier[-1]) or {}).get("title") or earlier[-1] if earlier else "the earlier choice"
        price = f" at {row['price']:g}" if isinstance(row.get("price"), (int, float)) else ""
        question = f"{old} can no longer be the pick. May I replace it with {row.get('title') or target} ({target}){price}?"
        answer = self.say_now(question, "I will check with the shopper before swapping.")
        if answer is None:
            return None
        return (f"Not run yet (no step used): as asked, the shopper was asked first: \"{question}\" and answered: {answer[:300]}. "
                "If they agreed, send the same cart call again; if they did not, choose another listing.")

    def say_now(self, text: str, intro: str) -> str | None:
        """One code-written message to the shopper, sent at once (it may use a step kept for the finish); the shopper's
        answer, or None when it could not be sent."""
        tool = self.roles.get("message")
        args = self.message_args(text) if tool else None
        if not args:
            return None
        args, problems = _check(args, self.schemas[tool], "args")
        if problems:
            return None
        call = {"id": f"duty_{self.step}_{self.messages_used}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        replies: list[tuple[str, str]] = []
        self.duty_sending = True
        try:
            self.send_chunk([(call["id"], tool, args)], replies)
        finally:
            self.duty_sending = False
        self.messages.append({"role": "assistant", "content": intro, "tool_calls": [call]})
        for call_id, reply_text in replies or [(call["id"], "Not sent.")]:
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text})
        return replies[0][1] if replies else None

    def basket_lines(self) -> tuple[list[str], float | None]:
        lines, total = [], 0.0
        for item in self.cart_ids:
            row = self.shelf.rows.get(item) or {}
            quantity = self.plan.get("quantity") if not self.items() else 1
            quantity = quantity if isinstance(quantity, int) and not isinstance(quantity, bool) and quantity >= 1 else 1
            price = row.get("price")
            if isinstance(price, (int, float)):
                total += price * quantity
                lines.append(f"{row.get('title') or item} ({item}): {quantity} x {price:g}")
            else:
                total = float("nan")
                lines.append(f"{row.get('title') or item} ({item}): {quantity}")
        return lines, (None if total != total or not lines else round(total, 2))

    def duty_message(self, owed: list[dict[str, Any]], declining: bool) -> str:
        lines, total = self.basket_lines()
        facts = ("BASKET:\n" + "\n".join(lines) + (f"\nTOTAL: {total:g} {self.plan.get('currency') or ''}".rstrip() if total is not None else "")) if lines and not declining else "NOTHING IS BEING BOUGHT."
        if declining and self.decline_reason():
            facts += "\nWHAT THE OPENED LISTINGS COULD NOT MEET: " + self.decline_reason()
        if self.cost < self.cost_stop + STOP_OVER and not self.llm_down and self.time_left() > FORCE_SECONDS:
            # one cheap call writes it, even past the hard line: a step the request asks for decides the pass
            prompt = DUTY_PROMPT.replace("WHAT", "decline (buy nothing)" if declining else "place the order")
            prompt += "\n\nSTEPS TO CARRY OUT IN THE MESSAGE:\n" + "\n".join("- " + d["do"] for d in owed) + "\n\n" + facts
            prompt += "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
            result = self.judge(prompt, 500, [READ_MODEL, JUDGE_MODEL])
            text = str((result or {}).get("message") or "").strip()
            if text:
                return text[:900]
        # short of money or time: the message is written by the code
        if declining:
            reason = self.decline_reason()
            return ("I could not find anything that meets every one of your requirements" + (f" (not met: {reason})" if reason else "")
                    + ", so I am not placing an order.")
        text = "Here is what I am ordering: " + "; ".join(lines) + (f". Total: {total:g}" if total is not None else "") + "."
        asked = [d["do"].strip().rstrip(".") for d in owed if d.get("do") and not re.search(r"\b(?:tell|total|basket|list|confirm)\b", d["do"], re.I)]
        if asked:
            text += " As you asked: " + "; ".join(asked)[:300] + "."
        if any(re.search(r"confirm|okay|ok\b|approve|check with|go ahead|permission", d["do"], re.I) for d in owed):
            text += " Please confirm and I will place it."
        return text

    def settle_duties(self, declining: bool) -> str | None:
        """Right before an order or a decline: the steps owed by message are carried out in one message. A hold when the
        shopper's answer changes something; None lets the finish go ahead."""
        owed = self.open_duties({"before_finish", "anytime", "before_decline" if declining else "before_order"}, True)
        if not owed:
            return None
        if not declining and self.roles.get("cart") and not self.cart_ids:
            return None                      # nothing is in the cart yet: the message about the order waits for the real one
        if self.messages_open() < 1 or self.steps_left() < 2 or not self.roles.get("message"):
            return None
        marks = (self.ledger_version, self.budget_now, len(self.gone), self.plan.get("quantity"))
        text = self.duty_message(owed, declining)
        answer = self.say_now(text, "Before finishing, I will do what the request asks me to tell the shopper.")
        if answer is None:
            return None
        for duty in owed:
            duty["done"] = True              # a step counts as done only once its message has actually gone out
        if (self.ledger_version, self.budget_now, len(self.gone), self.plan.get("quantity")) != marks or re.search(r"\b(no|don't|do not|wait|stop|hold on|instead|rather)\b", answer[:400], re.I):
            return (f"Not run yet (no step used): before finishing, the shopper was told: \"{text[:300]}\" and answered: {answer[:400]}. "
                    "Their answer changes something: apply it, then finish.")
        return None

    def known_step(self, text: str) -> bool:
        """Whether a step the look names is already on the ledger, by shared words (so a reworded step is not sent twice)."""
        words = step_words(text)
        if not words:
            return True
        for duty in self.duties:
            theirs = step_words(duty["do"])
            if theirs and len(words & theirs) * 2 >= min(len(words), len(theirs)):
                return True
        return False

    def last_look(self, declining: bool) -> str | None:
        """Once per episode, right before the first finish: one cheap look over the request, the conditions in force, the
        cart lines' records, the steps on the ledger and the exchanges so far. A step it finds undone joins the ledger (the
        finish then sends it); a condition the cart does not show met holds an order once, without a step. A decline is
        never held by it."""
        if self.steps_left() < 3 or self.cost >= self.cost_hard or self.time_left() < WRAP_UP_SECONDS + 20:
            return None
        mark = ("decline" if declining else "order") + "@" + self.cart_signature()
        if self.checked_once and (not self.long_world() or self.past_soft() or mark in self.looked or len(self.looked) >= LONG_LOOKS):
            return None
        if self.checked_once:
            self.check_held = False          # a new cart state in a long world may be held once more, without a step
        self.checked_once = True
        self.looked.add(mark)
        prompt = CHECK_PROMPT.replace("WHAT", "decline (buy nothing)" if declining else "place the order")
        prompt += "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCONDITIONS IN FORCE:\n" + self.ledger()
        prompt += "\n\nTHE FINISH ABOUT TO HAPPEN: " + ("a decline, nothing is bought" if declining else "an order of the cart below")
        if not declining:
            for item in self.cart_ids[-3:]:
                prompt += f"\n\nCART LINE {item}" + (f" ({self.cart_lines[item]})" if self.cart_lines.get(item) else "") + ":\n" + "\n".join(self.item_records(item))
        if self.duties:
            prompt += "\n\nSTEPS ON THE LEDGER:\n" + "\n".join(f"- {d['do']} [{'done' if d['done'] else 'open: the finish sends it' if d['say'] else 'open'}]" for d in self.duties)
        prompt += f"\n\nEXCHANGES WITH THE SHOPPER SO FAR ({self.messages_used} message(s) sent):\n" + ("\n".join(self.said[-12:]) or "(none)")
        result = self.judge(prompt, 600, [READ_MODEL])
        self.trace.append({"kind": "last_look", "declining": declining, "result": result})
        if not result:
            return None
        steps = [str(s).strip() for s in result.get("undone_steps") or [] if isinstance(s, (str, int)) and str(s).strip()][:3]
        if self.roles.get("message") and self.messages_open() >= 1:
            for step in steps:
                if not self.known_step(step):
                    self.add_duty({"do": step, "when": "before_finish", "say": True})
        unmet = [str(c).strip() for c in result.get("unmet_conditions") or [] if isinstance(c, (str, int)) and str(c).strip()][:4]
        describe = self.pointer_words()
        if describe:
            # a condition made only of the words that describe a pointed-at page (its seller, its details) is no requirement
            unmet = [c for c in unmet if not (set(words(c.split(" not ")[0].split("(")[0])) - {"not", "met"} <= describe | {"brand", "seller", "maker", "shop"} )]
        if declining or not unmet or self.check_held or not self.cart_ids:
            return None
        self.check_held = True
        return ("Not run yet (no step used): a last look at the cart line's own records against the conditions in force finds these not shown met: "
                + "; ".join(unmet) + ". Check them from the item's records (or the details tool if unopened); if a condition really fails, put a listing "
                "that meets everything in the cart instead and order; if the records do show it met, send the same call again.")

    def bundle_hold(self) -> str | None:
        if not self.plan.get("bundle") or len(self.cart_ids) < 2 or self.cost >= self.cost_hard or self.steps_left() <= 3:
            return None
        signature = "|".join(sorted(self.cart_ids))
        if signature in self.bundle_checked:
            return None
        self.bundle_checked.add(signature)
        prompt = BUNDLE_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        for item in self.cart_ids:
            prompt += f"\n\nCART LINE {item}:\n" + "\n".join(self.item_records(item))
        result = self.judge(prompt, 500, [READ_MODEL])
        if result and result.get("work_together") is False and not self.past_soft():
            result = self.judge(prompt, 600) or result      # a clash is confirmed by the full model before it holds the order
        if not result or result.get("work_together") is not False:
            return None
        return "Not run yet (no step used): the products in the cart do not seem to work together: " + str(result.get("problem") or "") + ". Replace the one that clashes with a listing that fits the others, then order."

    def change_review(self) -> None:
        """After a change (new budget or requirement, a refusal, a sold-out or repriced line): one cheap look at whether
        the cart still holds the right purchase, added to the transcript."""
        reason, self.review_reason = self.review_reason, ""
        if not reason or self.reviews >= 3 or not self.cart_ids or self.cost >= self.cost_hard or self.steps_left() <= 4 or self.time_left() < WRAP_UP_SECONDS + 20:
            return
        self.reviews += 1
        prompt = REVIEW_PROMPT + "\n\nWHAT CHANGED:\n" + reason + "\n\nCONDITIONS NOW:\n" + self.ledger()
        if self.gone:
            prompt += "\n\nREPORTED SOLD OUT: " + ", ".join(sorted(self.gone))
        for item in self.cart_ids[-3:]:
            prompt += f"\n\nCART LINE {item}:\n" + "\n".join(self.item_records(item))
        result = self.judge(prompt, 500, [READ_MODEL])
        self.trace.append({"kind": "review", "reason": reason[:200], "result": result})
        if not result:
            return
        verdict = "still right" if result.get("still_right") is True else "NO LONGER RIGHT"
        self.messages.append({"role": "user", "content": f"[review after a change] The cart is {verdict}: {result.get('why') or ''} Next: {result.get('next') or ''}\n" + self.status()})

    def walk_hold(self, name: str, args: dict[str, Any]) -> str | None:
        """On a cheapest request, an add to the cart first lets the code look below the target's price; when it finds a
        cheaper listing whose sheet states every condition, the add waits once (sending it again overrides). After a
        stockout, only a DEARER add is held, and only when a cheaper listing an item check passed is still on offer."""
        if self.plan.get("priority") != "cheapest" or self.items():
            return None
        properties = self.schemas[name].get("properties") or {}
        mode = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if self.add_signature is not None and mode != self.add_signature:
            return None                      # taking a line out
        target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)
        if target is None and _listed_ids(args):
            target = _listed_ids(args)[-1]
        if not target or target in self.walk_held:
            return None
        if (self.gone or self.refusals) and target not in self.dearer_ok:
            backup = self.verified_cheaper(target)
            if backup:
                mine, theirs = self.shelf.rows[target]["price"], self.shelf.rows[backup]["price"]
                self.dearer_ok.add(target)
                return (f"Not run (no step used): {backup} ({theirs:g}) is cheaper than {target} ({mine:g}) and an independent check already found it "
                        f"to fit; the request asks for the cheapest. Put {backup} in the cart instead. If {target} is right for a reason its own "
                        "records state, say that reason and send the same call again.")
            return None
        if target in self.dearer_ok:
            return None
        best, note = self.descend(target)
        if best != target and note:
            self.walk_held.add(target)
            return "Not run (no step used). " + note + f" Put {best} in the cart instead of {target}; if you are certain {target} is right, send the same call again."
        return None

    def cart_signature(self) -> str:
        """The cart state an advisory order hold is counted against: the lines, the requirements' version and the last change."""
        return "|".join(sorted(self.cart_ids)) + f"#{self.ledger_version}#{self.market_at}"

    def order_hold(self) -> str | None:
        """The holds on an order: the safety holds (a gone line, the bundle, a narrower category, a failed item check, a stale
        pick) every time; the advisory ones (a cheaper fit the walk found, a search after a refusal, unopened cheaper rows) at most
        once per cart state, after which the code opens the cheapest unopened row itself if it can and lets the order go."""
        kind, text = self.order_rule()
        if not text:
            return None
        if kind in ("walk", "research", "cheaper"):
            state = self.cart_signature()
            count = self.order_holds.get(state, 0)
            self.order_holds[state] = count + 1
            if count >= 1:
                self.trace.append({"kind": "hold", "order_advisory_waived": kind, "state": state})
                if kind == "cheaper":
                    self.open_cheapest_unread()
                return None
        return text

    def pre_add_reads(self, target: str) -> None:
        """In a long world, when the model puts its first line in the cart with fewer than three sheets read, the code first opens
        the sheets the choice should rest on: the target itself when unopened, then the cheapest unopened rows of its kind within
        the limit, up to PRE_ADD_READS in all, once per episode. The add then goes out in the same turn."""
        if self.pre_read or not self.long_world() or self.cart_used or self.items():
            return
        self.pre_read = True
        tool = self.sheet_tool()
        if not tool or self.property_menu()[1] >= 3 or self.steps_left() < 16 or self.cost >= self.cost_hard or self.time_left() < WRAP_UP_SECONDS + 60:
            return
        budget, quantity = self.walk_scope()
        held = self.shelf.rows.get(target) or {}
        picks = [target] if target in self.shelf.rows and not held.get("sheet") and target not in self.opened else []
        found = []
        for key, row in self.shelf.rows.items():
            if key == target or row.get("sheet") or key in self.opened or key in self.gone or row.get("price") is None or row.get("stock") is False:
                continue
            if budget is not None and row["price"] * quantity > budget + 1e-9:
                continue
            if held.get("group") and row.get("group") and flat(row["group"]) != flat(held["group"]):
                continue
            found.append((row["price"], key))
        picks += [k for _, k in sorted(found)]
        picks = picks[:PRE_ADD_READS]
        for key in picks:
            if self.steps_left() < 14 or self.finished or self.dead:
                break
            args = self.id_args(tool, key)
            if not args:
                continue
            reply = self.env_call([{"name": tool, "args": args}])
            self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": args}], "reply": reply, "pre_add_read": True})
            if reply is None or self.finished or self.dead:
                return
            observation = reply.get("observation")
            self.book_call(tool, args, observation)
            self.read_notices(tool, observation)
            self.note_observation(tool, observation)
        if picks:
            self.messages.append({"role": "user", "content": f"[note] Before the first add the code opened {len(picks)} more listing sheet(s) ({', '.join(picks)}; "
                                                             f"{len(picks)} store steps), so the choice and the questions rest on more than one sheet. Their records are in the store's replies below."})

    def open_cheapest_unread(self) -> None:
        """The code opens the cheapest unopened cheaper row itself (one step) when the steps allow, so the cheapest swap at the
        order can judge it; nothing is asked of the model."""
        cheaper = self.unchecked_cheaper()
        tool = self.sheet_tool()
        if not cheaper or not tool or self.steps_left() < self.finish_reserve() + 2 or self.time_left() < WRAP_UP_SECONDS + 10:
            return
        key = cheaper[0][0]
        args = self.id_args(tool, key)
        if not args:
            return
        reply = self.env_call([{"name": tool, "args": args}])
        self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": args}], "reply": reply, "cheapest_read": True})
        if reply is None or self.finished or self.dead:
            return
        observation = reply.get("observation")
        self.book_call(tool, args, observation)
        self.read_notices(tool, observation)
        self.note_observation(tool, observation)

    def order_rule(self) -> tuple[str, str | None]:
        """The first hold an order meets, with its kind; ("", None) lets it go out."""
        left, spare = self.steps_left(), self.time_left()
        named = set(_listed_ids(self.order_args) or []) | {v.strip() for v in self.order_args.values() if isinstance(v, str) and v.strip() in self.shelf.ids}
        shown_out = {k for k in set(self.cart_ids) | named if (self.shelf.rows.get(k) or {}).get("stock") is False}
        self.gone.update(shown_out)              # a line whose own record now reads out of stock is gone, whatever any note said
        gone = (set(self.cart_ids) | named) & self.gone
        if gone and self.gone_checks < 4 and left > 2:
            self.gone_checks += 1
            return "gone", ("Not run (no step used): " + ", ".join(sorted(gone)) + " was reported sold out, so this order would be refused again. Order the next listing "
                            "that meets every requirement instead (when the store has a cart: take the gone line out and put the new one in first)."
                            + (" If nothing else in the store meets every requirement, decline instead." if self.gone_checks >= 2 else ""))
        if not self.items() and len(set(self.cart_ids)) > 1 and left > 2 and self.cart_signature() not in self.multi_held:
            # one product requested, several lines held: the order would buy them all
            self.multi_held.add(self.cart_signature())
            return "lines", ("Not run (no step used): the cart holds " + ", ".join(dict.fromkeys(self.cart_ids)) + ", but the request wants one product. "
                             "Take out every line but the one pick that meets every condition, then order.")
        bundle = self.bundle_hold()
        if bundle:
            return "bundle", bundle
        shelf = self.narrower_shelf(self.cart_ids[-1]) if self.cart_ids else ""
        if shelf:
            return "shelf", "Not run (no step used). " + shelf
        cheapest = self.plan.get("priority") == "cheapest"
        if cheapest and self.cart_ids and spare > WRAP_UP_SECONDS + 45:
            pick = self.cart_ids[-1]
            best, note = self.descend(pick)
            if best != pick and note and pick not in self.walk_held:
                self.walk_held.add(pick)
                return "walk", "Not run yet (no step used). " + note + f" Swap {pick} for {best} in the cart (check its stock and price), then order; if you are certain {pick} is right, order again."
        # After a refused order a backup picked earlier may no longer be the cheapest fit: search once more first.
        # (a refusal that only moved a price or the limit leaves the cheapest fit where it was: the held line is ordered again)
        moved_only = (self.price_moved_at >= self.refused_at or self.budget_changed_at >= self.refused_at or _PRICE_WORDS.search(self.refusal_text or "") is not None) and not self.gone_said(self.refusal_text or "")
        if cheapest and self.refused_at > self.searched_at and self.research_checks < 1 and left > COMMIT_STEPS and spare > WRAP_UP_SECONDS + 20 and not moved_only:
            self.research_checks += 1
            try:
                # the code itself reads the rest of the model's search and opens the cheapest unread rows within the limit
                if self.steps_left() >= self.finish_reserve() + 5:
                    self.page_early(force=True)
                if not (self.finished or self.dead):
                    opened = self.unread_rows_read()
                    fits = self.rank_by_rule([k for k in self.live_fits() if k not in self.gone])
                    if opened and fits:
                        return "research", (f"Not run yet (no step used). The last order was refused; the code read on and opened {', '.join(opened)}. "
                                            f"The cheapest listing that now states every condition within the limit is {fits[0]} "
                                            f"({(self.shelf.rows.get(fits[0]) or {}).get('price', '?')}): order that one (fix the cart first when the store has one).")
            except Exception as exc:
                self.trace.append({"kind": "research", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return "research", "Not run yet (no step used). The last order was refused, and the request asks for the cheapest fitting item. A backup chosen earlier is right only if nothing cheaper fits now, so search again before ordering: the stated requirement words and the category, with a price limit just above the refused item's price if the search offers one, and the next pages. Open the details of cheaper candidates, put the cheapest one that meets every requirement in the cart, then order."
        if (self.plan.get("priority") == "cheapest" and not self.items() and named and not self.cart_ids and self.cheaper_named < 2 and left > 1
                and not self.unfolded):
            # a store without a cart: the order names its listing; a verified fit that costs less is ordered instead
            fits = self.rank_by_rule([k for k in self.live_fits() if k not in self.gone])
            target = next(iter(named))
            price = (self.shelf.rows.get(target) or {}).get("price")
            if fits and fits[0] != target and price is not None and ((self.shelf.rows.get(fits[0]) or {}).get("price") or 1e18) < price - 1e-9:
                self.cheaper_named += 1
                return "named_cheaper", (f"Not run (no step used): {fits[0]} ({self.shelf.rows[fits[0]]['price']:g}) states every condition by its own sheet and costs less "
                                   f"than {target} ({price:g}); the request asks for the cheapest fit, so order {fits[0]} instead.")
        if self.cheaper_checks < 2 and left > COMMIT_STEPS and spare > WRAP_UP_SECONDS + 20:
            cheaper = self.unchecked_cheaper()
            if cheaper:
                self.cheaper_checks += 1
                return "cheaper", "Not run yet (no step used). The request asks for the cheapest fitting item, but these cheaper listings that you have seen in results were never opened with the details tool: " + ", ".join(f"{i} ({p})" for i, p in cheaper) + ". Open their details (several in one reply), and if one meets every requirement, put the cheapest such one in the cart instead; if none does, order again."
        if (not self.ask_held and (self.questions_open() if self.long_world() else self.messages_used == 0) and self.messages_open() >= 1
                and left > (6 if self.long_world() else 4) and spare > WRAP_UP_SECONDS + 40 and self.cost < self.cost_hard):
            # the questions the plan still owes (or, with messages allowed and none sent, the first one) go out before the
            # order, by the code: a hidden need surfaces only when asked
            self.ask_held = True
            if self.interview(force=True):
                return "ask", ("Not run yet (no step used): the questions the shopper had not been asked went out first (see the exchanges above). "
                               "Apply their answers to the pick (swap it if a revealed need says so), then order again.")
        if (self.ending_message_free() and left > (6 if self.long_world() else 4) and spare > WRAP_UP_SECONDS + 40
                and self.cost < self.cost_stop and (self.cart_ids or not self.roles.get("cart"))):
            # the message kept for a step owed only at a decline is not needed by an order: it carries one more planned
            # question (the product most likely to hide a need first) before the order goes out
            self.late_asks += 1
            self.ending_freed = True
            asked_before = len(self.interview_log)
            if self.interview(force=True) and len(self.interview_log) > asked_before:
                self.trace.append({"kind": "late_ask", "properties": [q["property"] for q in self.interview_log[asked_before:]], "step": self.step})
                return "late_ask", ("Not run yet (no step used): an order needs no message of its own, so the message kept for the ending carried one more "
                                    "planned question first (see the exchange above). Apply its answer to the pick (swap it if a revealed need says so), then order again.")
            self.ending_freed = bool(len(self.interview_log) > asked_before)
        recheck = self.long_world() and self.checked_version != self.ledger_version and self.cart_ids
        if (not self.order_checked or recheck) and left > 2 and spare > WRAP_UP_SECONDS:
            self.order_checked = True
            self.checked_version = self.ledger_version
            verdict = self.verify_cart()
            if verdict:
                return "verdict", "Not run yet (no step used). " + verdict
            stale = self.freshen_pick()
            if stale:
                return "stale", stale
        return "", None

    def id_args(self, tool: str, target: str) -> dict[str, Any] | None:
        """Arguments that point a reading tool (details, view, stock) at one listing."""
        function = self.schemas_full.get(tool or "") or {}
        props = (function.get("parameters") or {}).get("properties") or {}
        id_key = next((k for k in (function.get("parameters") or {}).get("required") or [] if isinstance(props.get(k), dict) and "string" in str(props[k].get("type"))), None)
        id_key = id_key or next((k for k in props if re.search(IDDY, flat(re.sub(r"[_\-]+", " ", k)))), None)
        listed = next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array" and (v.get("items") or {}).get("type") in ("string", None)), None)
        if listed and (not id_key or (props.get(id_key) or {}).get("type") == "array"):
            id_key, target = listed, [target]  # a tool that takes a list of listings: a list of one
        if not id_key:
            return None
        args = shape_args(function, {id_key: target})
        return None if lacking(function, args) else args

    def freshen_pick(self) -> str | None:
        """Before an order the pick's stock and price must have been seen after the last announced change. When they were
        not, the code re-reads them itself (one step) rather than asking the model to; a pick found gone holds the order."""
        if not self.cart_ids or self.market_at < 0:
            return None
        pick = self.cart_ids[-1]
        row = self.shelf.rows.get(pick) or {}
        if pick in self.freshened or pick in self.gone or row.get("step", 0) >= self.market_at:
            return None
        tool = self.roles.get("view") or self.roles.get("details") or self.roles.get("stock")
        if not tool or self.steps_left() < self.finish_reserve() + 2 or self.time_left() < WRAP_UP_SECONDS + 10:
            return None
        args = self.id_args(tool, pick)
        if not args:
            return None
        self.freshened.add(pick)
        reply = self.env_call([{"name": tool, "args": args}])
        self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": args}], "reply": reply, "fresh": True})
        if reply is None or self.finished or self.dead:
            return None
        observation = reply.get("observation")
        self.book_call(tool, args, observation)
        self.read_notices(tool, observation)
        self.note_observation(tool, observation)
        row = self.shelf.rows.get(pick) or {}
        if pick in self.gone or row.get("stock") is False:
            return (f"Not run yet (the order used no step; 1 step re-read {pick}): it is now out of stock, so the order would be refused. "
                    + (self.fallback_note() or "Put the next listing that meets every requirement in the cart, then order."))
        return None

    def decline_hold(self) -> str | None:
        """A decline is checked twice: it is rarely the right finish, so each attempt first gets an independent look for
        listings that meet the requirements."""
        if self.decline_checks >= 2 or self.steps_left() <= 4 or self.time_left() <= WRAP_UP_SECONDS + 30:
            return None
        self.decline_checks += 1
        audit = self.audit_decline()
        if audit:
            return "Not run yet (no step used). " + audit
        if self.decline_checks == 1:
            return "Not run yet (no step used). Before declining, be sure: you searched in several ways (different words, filters, pages), you checked every candidate's own properties, and each one fails a stated requirement, a revealed need, the budget or stock. Remember that close synonyms and any key spelling count, and that a requirement you only assumed from the story does not count. If something does fit, buy it instead; if nothing fits, call the decline tool again."
        return None

    def book_cart(self, name: str, args: dict[str, Any]) -> None:
        """Cart lines after a cart call: the first call's enum mode is the add mode, another mode takes lines out; a cart
        tool that takes the whole basket as a list replaces the lines."""
        properties = self.schemas[name].get("properties") or {}
        mode = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if self.add_signature is None:
            modes = [(k, v) for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]]
            hint = self.plan.get("cart_modes") if isinstance(self.plan.get("cart_modes"), dict) else None
            takes_out = any(v == mode_pair(properties[k]["enum"], hint)[1] for k, v in modes)
            if not takes_out:
                self.add_signature = mode
                self.add_args = dict(args)
        adding = mode == self.add_signature
        listed = _listed_ids(args)
        form = self.basket_form()
        if form and isinstance(args.get(form["key"]), list):
            if form["plain"]:
                listed = [str(x) for x in args[form["key"]] if isinstance(x, str)]
            else:
                # the line's id field as the cart schema names it, whatever its spelling; the generic reading only as a fallback
                by_form = [str(line.get(form["id"])).strip() for line in args[form["key"]]
                           if isinstance(line, dict) and form["id"] and isinstance(line.get(form["id"]), str) and str(line.get(form["id"])).strip()]
                listed = list(dict.fromkeys(by_form)) or listed or []
            adding = True
            counts = {}
            for line in args[form["key"]]:
                if isinstance(line, dict):
                    ident = line.get(form["id"]) if form["id"] else None
                    count = line.get(form["count"]) if form["count"] else None
                    if isinstance(ident, str):
                        counts[ident] = int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) and count >= 1 else 1
            self.basket_counts = counts
        if listed is not None:
            if adding:
                dropped = [i for i in self.cart_ids if i not in listed]
                self.cart_ids = listed
                self.cart_lines = {i: _line_of(args, i) for i in listed}
            else:
                dropped = [i for i in self.cart_ids if i in listed]
                self.cart_ids = [i for i in self.cart_ids if i not in listed]     # the same tool used to take lines out
            if not self.code_move:
                self.model_removed.update(dropped)
        else:
            for key, value in args.items():
                spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
                if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip():
                    self.cart_state[value] = mode
                    if not adding and not self.code_move and value in self.cart_ids:
                        self.model_removed.add(value)
                    self.cart_ids = [i for i in self.cart_ids if i != value] + ([value] if adding else [])
        self.cart_changed_at = self.step
        self.track_lines()

    def book_call(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Counters and memories after one call went to the store."""
        roles = self.roles
        message_tool, cart_tool = roles.get("message"), roles.get("cart")
        if name == message_tool:
            self.messages_used += 1
            if self.message_refused(observation):
                # the store turned the message down for its allowance (or as unknown): no further message is sent
                self.message_budget = self.messages_used - 1 if self.message_budget is None else min(self.message_budget, self.messages_used - 1)
                self.capped.add(name)
                self.planner_on = False
                self.trace.append({"kind": "message_refused", "used": self.messages_used, "reply": _compact(observation)[:200]})
            sent_text = " ".join(str(v) for v in args.values() if isinstance(v, str)).strip().lower()
            if sent_text and not (isinstance(observation, dict) and observation.get("error")):
                self.questions_sent.append(sent_text)      # a message that failed may be sent again
        if name in {roles.get("details"), roles.get("view"), roles.get("compare")} - {None} and not (isinstance(observation, dict) and observation.get("error")):
            self.page_calls += 1
        elif name == roles.get("search"):
            self.search_calls += 1
        if name == roles.get("order"):
            self.order_attempts += 1
            self.order_targets = list(dict.fromkeys((_listed_ids(args) or []) + [v.strip() for v in args.values() if isinstance(v, str) and v.strip() in self.shelf.ids]))
        listing_mode = self.viewing(name, args)
        if name == cart_tool and not listing_mode:
            self.cart_used = True
            self.order_checked = False
        remover = roles.get("cart_remove")
        if remover and name == remover and remover != cart_tool:
            # a separate remove tool: the lines it names leave the cart bookkeeping too
            taken = [v.strip() for v in args.values() if isinstance(v, str) and v.strip() in self.cart_ids] + [i for i in (_listed_ids(args) or []) if i in self.cart_ids]
            if taken:
                self.cart_ids = [i for i in self.cart_ids if i not in taken]
                if not self.code_move:
                    self.model_removed.update(taken)
                self.cart_changed_at = self.step
                self.order_checked = False
                self.track_lines()
        resets = name in (cart_tool, message_tool, roles.get("order"), roles.get("decline"))
        self.recent = [] if resets else (self.recent + [name + _compact(args)])[-2:]
        self.records.append(observation)
        if name == cart_tool and not listing_mode:
            before, held_before = len(self.cart_ids), set(self.cart_ids)
            self.book_cart(name, args)
            # a note the add triggers arrives on the NEXT call: the order must not be that call
            self.add_unread = len(self.cart_ids) > before or (self.add_unread and len(self.cart_ids) >= before)
            if self.basket_form() and self.cart_ids and set(self.cart_ids) != held_before:
                self.add_unread = True       # a whole-basket hold that changed the lines counts as an add
        elif name != roles.get("order"):
            self.add_unread = False
        if name == roles.get("search"):
            self.searched_at = self.step
            for k, v in args.items():
                if isinstance(v, (str, int, float)) and not isinstance(v, bool):
                    self.search_usage.setdefault(k, []).append(v)
            if not self.walking:
                self.note_search(args, observation)
        if name in (roles.get("details"), roles.get("view"), roles.get("compare")) and name not in self.sheet_tools:
            batch = _listed_ids(args) or []
            if any(self.shelf.sheets_seen.get(i) == self.step for i in batch):
                self.sheet_tools.add(name)
        if name in (roles.get("details"), roles.get("view")):
            # which reading tool actually returns the property sheet is learnt from what it returned
            target = next((v.strip() for v in args.values() if isinstance(v, str) and v.strip() in self.shelf.ids), "")
            if target:
                (self.sheet_tools if target in self.shelf.sheets and self.shelf.sheets_seen.get(target) == self.step else self.bare_tools).add(name)
            self.opened.update(v for v in args.values() if isinstance(v, str))
            for v in args.values():
                if isinstance(v, str) and v.strip() in self.shelf.ids:
                    self.opened_by.setdefault(v.strip(), set()).add(name)
        if name == roles.get("cart_view") or listing_mode:
            self.last_cart_view = observation
            self.listing_step = self.step

    @staticmethod
    def message_refused(observation: Any) -> bool:
        """Whether a message's result is a fault that reports the allowance spent or the call unknown (an error-like key
        whose text says so), rather than a reply of the shopper."""
        if not isinstance(observation, dict):
            return False
        nodes = [observation] + [v for v in observation.values() if isinstance(v, dict)]
        for node in nodes:
            for key, value in node.items():
                if isinstance(value, str) and re.search(r"error|trouble|fault|problem|fail|issue", str(key), re.I) \
                        and (re.search(r"cap|exhaust|exceed|unknown|allowance|quota", value, re.I) or _USED_UP.search(value)):
                    return True
        return False

    def order_outcome(self, name: str, observation: Any, finished: bool) -> str:
        """What an order call that did not end the episode means: a pending checkout step, or a refusal in any wording."""
        if name != self.roles.get("order") or name == self.roles.get("decline") or finished:
            return ""
        refusal = isinstance(observation, dict) and ("refusal" in observation or self.refusal_step == self.step)
        if isinstance(observation, dict) and _pending_step(observation) and not refusal:
            return "The order is waiting for a further step (it gives a token or asks for a confirmation): take that step now with the tool that finishes the checkout. Nothing is bought until the episode ends."
        if refusal:
            return ""
        # the order did not end the episode: refused (in any wording), failed, or waiting for another step
        self.refusals += 1
        self.refused_at = self.step
        self.refusal_text = _compact(observation)[:600]
        self.order_checked = False
        self.mark_change()
        return "THE ORDER CALL DID NOT END THE EPISODE, SO NOTHING HAS BEEN BOUGHT YET, whatever its result says (an order that goes through ends the episode). If its result asks for a further step (for example a confirmation, with a token or code it gives), take that step now. If it was refused or failed, read why, apply every change it states, fix the cart, re-check stock and price of the pick, and order again; a price change alone does not make an item unacceptable while price times quantity stays within the budget."

    def shopper_lines(self, name: str, args: dict[str, Any], observation: Any) -> tuple[list[str], str | None]:
        """The shopper's words in a result, and the question they answer when the call was a message."""
        heard: list[str] = []
        if name == self.roles.get("message"):
            if isinstance(observation, str):
                heard = [observation]
            elif isinstance(observation, (dict, list)) and not (isinstance(observation, dict) and "error" in observation):
                if isinstance(observation, dict):
                    _sentences(list(v for v in observation.values() if isinstance(v, (str, list))), heard)
                if not heard and _has_text(observation):
                    heard = [_compact(observation)[:600]]     # a short or structured answer ("Navy.") is still an answer
            return heard, " ".join(str(v) for v in args.values() if isinstance(v, str))
        if isinstance(observation, dict):
            wrapper = len(observation) <= 3 and any(isinstance(v, (dict, list)) for v in observation.values())
            for value in observation.values():
                if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    _sentences(value, heard)
                elif isinstance(value, str) and len(value.split()) >= 4 and ("refusal" in observation or self.refusal_step == self.step):
                    heard.append(value)
                elif isinstance(value, str) and wrapper and len(value.split()) >= 6 and (_FIRST_PERSON.search(value) or name == self.roles.get("order")):
                    heard.append(value)     # a line wrapped around the real result, as a plain string
        return heard, None

    def finishes_checkout(self, name: str) -> bool:
        """A tool no ordinary role covers that completes a pending checkout (a confirm step after the order tool)."""
        ordinary = {self.roles.get(r) for r in ("search", "details", "view", "message", "cart", "cart_view", "stock")} - {None}
        return name not in ordinary and bool(_FINISH_STEP.search(name or ""))

    def send_chunk(self, chunk: list[tuple[str, str, dict[str, Any]]], replies: list[tuple[str, str]]) -> None:
        """Up to sixteen calls in one envelope; each result is booked, read for news and turned into the model's reply."""
        if self.finished or self.dead or self.steps_left() <= 0:
            replies.extend((call_id, "Not run: the episode is over or no steps are left.") for call_id, _, _ in chunk)
            return
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        if not any(name in finishing for _, name, _ in chunk) and not self.duty_sending:
            room = max(0, self.steps_left() - 2) + sum(1 for _, name, _ in chunk if self.finishes_checkout(name))
            replies.extend((call_id, "Not run (no step used): the last steps are kept for the finish (the order or the decline).") for call_id, _, _ in chunk[room:])
            chunk = chunk[:room]
            if not chunk:
                return
        chunk = chunk[: self.steps_left()]
        actions = [{"name": name, "args": args} for _, name, args in chunk]
        before = self.step
        reply = self.env_call(actions)
        raws = list(self.raw_last)
        self.trace.append({"kind": "env", "step": before + 1, "actions": actions, "reply": reply})
        if reply is None:
            replies.extend((call_id, "The call failed (network or server). Try again.") for call_id, _, _ in chunk)
            return
        observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
        for index, (call_id, name, args) in enumerate(chunk):
            if index >= len(observations):
                replies.append((call_id, "Not run: the episode finished before this call."))
                continue
            observation = observations[index]
            self.book_call(name, args, observation)
            self.read_notices(name, observation)
            ended = bool(reply.get("finished")) and index == len(observations) - 1
            # a finishing call (order or decline) that came back while the episode goes on was turned down, whatever its
            # keys are called; a result that asks for a further checkout step is not a refusal
            refused = name in finishing and not ended and isinstance(observation, dict) and not _pending_step(observation) and self.has_line(observation)
            notes = [self.note_observation(name, observation, refused), self.order_outcome(name, observation, bool(reply.get("finished")))]
            if self.swap_note and name == self.roles.get("order"):
                notes.insert(0, self.swap_note)
                self.swap_note = ""
            heard, question = self.shopper_lines(name, args, observation)
            if heard and name != self.roles.get("message") and self.notes_settled(heard):
                self.heard.update(heard)       # the note reader already read these lines and found no need in them
                notes.append(self.note_gist(heard))
            elif heard:
                notes.append(self.interpret(heard, question))
            note = " ".join(n for n in notes if n)
            raw = raws[index] if index < len(raws) else observation
            text = self.model_text(name, raw, observation) + (" [the episode is finished]" if index == len(observations) - 1 and reply.get("finished") else "")
            replies.append((call_id, text + ("\n[note] " + note if note else "")))
        searched = [(i, observations[i]) for i, (_, name, _) in enumerate(chunk) if name == self.roles.get("search") and i < len(observations)]
        if searched and not self.finished and not self.dead and not self.walking:
            try:
                read = self.broad_reads([o for _, o in searched])
            except Exception as exc:
                read = ""
                self.trace.append({"kind": "broad_read", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if read:
                at = next(j for j in range(len(replies) - 1, -1, -1) if replies[j][0] == chunk[searched[-1][0]][0])
                replies[at] = (replies[at][0], replies[at][1] + "\n[note] " + read)

    def raw_view(self, value: Any, depth: int = 0, field: bool = False) -> Any:
        """The store's own result for the model: a text under a key is cut to RAW_FIELD_CHARS, a bare text (often a whole
        record written as one string) to RAW_RECORD_CHARS, a list to its first 30 entries."""
        if depth > 8:
            return value
        if isinstance(value, str):
            most = RAW_FIELD_CHARS if field and "\n" not in value and " | " not in value else RAW_RECORD_CHARS
            return value if len(value) <= most else value[:most] + "…"
        if isinstance(value, dict):
            return {k: self.raw_view(v, depth + 1, True) for k, v in value.items()}
        if isinstance(value, list):
            shown = [self.raw_view(v, depth + 1) for v in value[:30]]
            return shown + ([f"…{len(value) - 30} more entries not shown"] if len(value) > 30 else [])
        return value

    @staticmethod
    def entries_in(value: Any, depth: int = 0) -> int:
        """How many records a result seems to hold by its own shape: the longest list of objects or texts in it."""
        if depth > 5:
            return 0
        if isinstance(value, list):
            own = sum(1 for v in value if isinstance(v, (dict, str)) and len(_compact(v)) >= 40)
            return max([own] + [Episode.entries_in(v, depth + 1) for v in value[:40]])
        if isinstance(value, dict):
            return max([Episode.entries_in(v, depth + 1) for v in value.values()] + [0])
        return 0

    def model_text(self, name: str, raw: Any, parsed: Any) -> str:
        """What the model reads of one result: the store's own text (long texts cut), then, for a result whose records the
        code had to read out of text, the code's reading of it as a separate note. A reading that found fewer listings
        than the text shows is said to be partial (and the code's own moves stop leaning on it)."""
        text = render(self.raw_view(raw), RAW_RESULT_CHARS)
        if len(self.raw_seen) < 400000:
            self.raw_seen += " " + _compact(raw)
        readers = {self.roles.get("search"), self.roles.get("details"), self.roles.get("view"), self.roles.get("compare"), self.roles.get("stock")} - {None}
        if name not in readers:
            return text
        found: set[str] = set()
        _collect_ids(parsed, found, keys=tuple(self.shelf.id_keys))
        found = {k for k in found if k in self.shelf.rows}
        shown = self.entries_in(raw)
        changed = _compact(raw) != _compact(parsed)
        listing = name == self.roles.get("search")
        partial = listing and ((shown >= 2 and len(found) * 2 < shown) or (not found and len(_compact(raw)) > 400))
        if partial:
            self.parse_low += 1
        elif found and listing:
            self.parse_good += 1
        if not changed and not partial:
            return text
        parts = []
        for key in sorted(found)[:24]:
            row = self.shelf.rows.get(key) or {}
            price = row.get("price")
            stock = {True: "in stock", False: "out of stock"}.get(row.get("stock"), "stock ?")
            parts.append(f"{key} ({price:g}, {stock})" if isinstance(price, (int, float)) else f"{key} (price ?, {stock})")
        note = ("[code read] the code read these listings out of the text above: " + "; ".join(parts)) if parts else "[code read] the code could read no listing record out of the text above."
        if partial:
            note += " Its reading is partial (the text shows more entries): read the text yourself and use its ids and values as written."
        return text + "\n" + note

    def parse_sure(self) -> bool:
        """Whether the code's own reading of this store's results is trusted enough for the code to act on it (cart moves,
        holds on searches, removals, code-made declines): sheets were read, the reader found about as many listings as the
        results show, and the model never had to use an id the reader missed."""
        if self.blind() or self.raw_misses > 0:
            return False
        return self.parse_low <= max(1, self.parse_good // 4)

    def broad_room(self) -> int:
        """How many sheets the code may open itself after a search: none past the commit mark, the soft spending line or
        the time reserve; else one per BROAD_PER_STEP spare steps (beyond the finish reserve, the questions still open and
        BROAD_SPARE), at most BROAD_PER_SEARCH, within BROAD_SHARE of the cap per episode."""
        if not self.sheet_tool() or self.cost >= 0.8 * self.cost_soft or self.time_left() < WRAP_UP_SECONDS + 60:
            return 0
        if self.items() or self.max_steps < BROAD_FROM:
            return 0                         # a short world, or several products: their pages are opened per product by the bundle and fitment code
        if any(k not in self.gone for k in self.cart_ids):
            return 0                         # a line is held: the choice is made, the steps go to the order
        if self.step >= self.pace_marks()[1] and not (self.refusals and self.step - self.refused_at <= 6):
            return 0                         # past the commit mark only the search right after a refused order is widened
        questions = min(self.messages_open(), UNSTATED_MESSAGES) if self.roles.get("message") else 0
        spare = self.steps_left() - self.finish_reserve() - questions - BROAD_SPARE
        total = int(BROAD_SHARE * self.max_steps) - self.broad_spent
        return max(0, min(BROAD_PER_SEARCH, spare // BROAD_PER_STEP, total))

    def compare_reads(self, keys: list[str]) -> list[str]:
        """Several sheets in one step: a side-by-side tool that takes a list of listings and says it shows their properties
        is tried once; it stays in use only when it really delivered sheets. The listings whose sheets it delivered."""
        tool = self.roles.get("compare")
        function = self.schemas_full.get(tool or "") or {}
        if not function or self.compare_sheets is False or len(keys) < 2:
            return []
        desc = flat(function.get("description") or "")
        if self.compare_sheets is None and not re.search(r"spec|propert|attribut|detail|sheet|feature|characteristic|technical", desc):
            return []
        params = function.get("parameters") or {}
        props = params.get("properties") or {}
        arrays = [k for k, r in props.items() if isinstance(r, dict) and r.get("type") == "array"]
        if len(arrays) != 1 or any(k != arrays[0] for k in params.get("required") or []):
            return []
        most = props[arrays[0]].get("maxItems")
        if not isinstance(most, int):
            found = re.search(r"\bup to (\d+|two|three|four|five|six)\b", desc)
            # no stated maximum: two at a time (race 18's side-by-side tool refuses three or more as invalid arguments)
            most = (int(found.group(1)) if found.group(1).isdigit() else _NUMBER_WORDS.index(found.group(1))) if found else 2
        chosen = keys[: max(0, most)]
        if len(chosen) < 2 or self.steps_left() <= self.finish_reserve() + BROAD_SPARE:
            return []
        action = {"name": tool, "args": {arrays[0]: chosen}}
        reply = self.env_call([action])
        self.trace.append({"kind": "env", "step": self.step, "actions": [action], "reply": reply, "broad_read": "compare"})
        if reply is None:
            return []
        observation = reply.get("observation")
        self.book_call(tool, action["args"], observation)
        self.read_notices(tool, observation)
        self.note_observation(tool, observation)
        self.broad_spent += 1
        got = [k for k in chosen if self.shelf.sheets_seen.get(k) == self.step]
        self.compare_sheets = bool(got)
        return got

    def broad_reads(self, observations: list[Any]) -> str:
        """After a search, the code opens the sheets of the cheapest unread rows it showed that are of the requested kind,
        in stock and within the limit (one envelope, a step each), so the choice rests on many sheets rather than the few
        the model opens; a row already opened is never read again. Returns the note for the model."""
        room = self.broad_room()
        if room <= 0:
            return ""
        shown: list[str] = []
        stack: list[Any] = list(observations)
        while stack and len(shown) < 80:
            value = stack.pop(0)
            if isinstance(value, dict):
                key = self.shelf.id_field(value)
                if isinstance(key, (str, int)) and not isinstance(key, bool) and str(key).strip() in self.shelf.rows:
                    shown.append(str(key).strip())
                stack.extend(v for v in value.values() if isinstance(v, (dict, list)))
            elif isinstance(value, list):
                stack.extend(value[:60])
        budget, quantity = self.walk_scope()
        pool = []
        for key in dict.fromkeys(shown):
            row = self.shelf.rows.get(key) or {}
            if row.get("sheet") or key in self.opened or key in self.gone or row.get("stock") is False or row.get("price") is None:
                continue
            if budget is not None and row["price"] * quantity > budget + 1e-9:
                continue
            if not self.of_kind(row):
                continue
            pool.append((row["price"], len(pool), key))
        picks = [k for _, _, k in sorted(pool)[:room]]
        if not picks:
            return ""
        compared = self.compare_reads(picks)
        picks = [k for k in picks if k not in compared]
        tool = self.sheet_tool()
        actions = [{"name": tool, "args": a} for a in (self.id_args(tool, k) for k in picks) if a]
        if not actions:
            return ""
        done = []
        for key, action in zip(picks, actions):
            # one envelope per sheet: a store that takes no batched envelope still serves them
            if self.finished or self.dead or self.steps_left() <= self.finish_reserve() + BROAD_SPARE:
                break
            reply = self.env_call([action])
            self.trace.append({"kind": "env", "step": self.step, "actions": [action], "reply": reply, "broad_read": True})
            if reply is None:
                break
            observation = reply.get("observation")
            self.book_call(action["name"], action["args"], observation)
            self.read_notices(action["name"], observation)
            self.note_observation(action["name"], observation)
            self.broad_spent += 1
            self.broad_raw[key] = (action["name"], self.step, observation)
            done.append(key)
        done = compared + done
        if not done:
            return ""
        lines = [self.sheet_text(k, 600) for k in done]
        return (f"The code also opened {len(lines)} sheet(s) of the cheapest unread rows of this result ({len(lines)} store steps); "
                "do not open them again:\n" + "\n".join(lines))

    def read_notices(self, name: str, observation: Any) -> None:
        """Message and step limits a store notice or a shopper line states, wherever they sit in a result; a price or
        stock notice about a cart line calls for a review."""
        texts: list[str] = []
        loose_strings(observation, texts, keys=tuple(self.shelf.id_keys), listing=self.shelf.is_listing)
        for text in texts:
            for kind, count, _ in limits_in(text):
                self.tighten(kind, count, True)
        try:
            readers = {self.roles.get("search"), self.roles.get("details"), self.roles.get("view"), self.roles.get("stock")} - {None}
            if name in readers:
                # a reading tool's own result is listing text; a note rides on it only as a list of sentences beside it
                wrapped = [x for v in observation.values() if isinstance(v, list) for x in v if isinstance(x, str)] if isinstance(observation, dict) else []
                if isinstance(observation, dict) and not self.shelf.is_listing(observation):
                    # a sentence standing beside the rows (not inside a listing record) is a note too, whatever its key
                    wrapped += [v for v in observation.values() if isinstance(v, str) and len(v.split()) >= 5]
                    wrapped += [x for v in observation.values() if isinstance(v, dict) and not self.shelf.is_listing(v)
                                for x in v.values() if isinstance(x, str) and len(x.split()) >= 5]
                fresh = self.note_lines(name, wrapped)
            else:
                fresh = self.note_lines(name, texts)
            self.read_note(fresh)
            for line in fresh:
                self.apply_note(line, name)
        except Exception as exc:
            self.trace.append({"kind": "note_read", "error": f"{type(exc).__name__}: {exc}"[:300]})
        self.hear_budget(texts, name)
        self.read_market(texts, name)
        if name not in (self.roles.get("search"), self.roles.get("details"), self.roles.get("view")) and self.cart_ids and self.listing_step != self.step:
            joined = " ".join(texts)
            # a price word followed by a direction of change or by a figure ("now 700") is a price move
            moved = re.search(r"\b(?:price|cost|priced)s?\b.{0,40}(?:\b(?:up|down|r[oia]s|chang|drop)|creas|\d)|now costs|new price", joined, re.I)
            if (self.gone_said(joined) or moved) and not self.review_reason:
                self.review_reason = "a store notice: " + joined[:300]
                self.mark_change()
            if moved and self.items():
                self.price_moved_at = self.step

    def hear_budget(self, texts: list[str], name: str) -> None:
        """A shopper line in any result that sets a new total limit is applied by the code at once: the ledger's budget,
        its version and the change mark move, whatever the hearing model later says. Lines inside listing records never
        reach here (loose_strings skips them); the request's own figures are not re-read."""
        if name in (self.roles.get("details"), self.roles.get("view"), self.roles.get("message")):
            return                           # listing pages carry no shopper line (a note can ride on a search or a cart view)
        reader = name not in (self.roles.get("search"), self.roles.get("cart_view"))
        for text in texts:
            if text in self.heard_budget_lines:
                continue
            line = " ".join(str(text).split())
            figure = self.note_budget(line)
            if figure is None and reader:
                figure = budget_in(text, str(self.plan.get("currency") or ""), self.budget_now if isinstance(self.budget_now, (int, float)) else None)
            if figure is None and line in self.note_reads and not self.note_reads[line] and (reader or _FIRST_PERSON.search(line)):
                figure = self.budget_figure(line)       # the note read gave nothing: one figure in the request's price unit
            around = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
            scaled = self.shelf.factor
            if scaled != 1.0 and around and re.search(r"\d", line):
                # the store writes amounts on another scale than the request ("$1.20" beside a limit of 90 cents): a new
                # limit written the store's way is read on its scale
                if figure is None and reader:
                    figure = budget_in(text, str(self.plan.get("currency") or ""), around / scaled)
                    figure = None if figure is None else round(figure * scaled, 6)
                elif figure is not None and not 0.05 * around <= figure <= 20 * around and 0.05 * around <= figure * scaled <= 20 * around:
                    figure = round(figure * scaled, 6)
            if figure is None:
                continue
            if around and not 0.05 * around <= figure <= 20 * around:
                continue
            self.heard_budget_lines.add(text)
            self.set_budget(figure, text, name)

    def live_fits(self) -> list[str]:
        """Opened listings that meet everything known by their own sheets (or passed an item check under the current
        conditions), in stock, within the current limit, not gone, not ruled out by the model; cheapest first. With no
        condition on the ledger, every opened listing of the requested kind within the limit counts."""
        if self.items() or self.open_pointers() or any(p["failed"] and not p["done"] for p in self.pointers):
            return []                        # a need the code could not settle is judged by the model alone
        budget, quantity = self.walk_scope()
        conds = self.conditions()
        pool = set(self.shelf.fits(conds, budget, quantity, "", self.gone)) if conds else set()
        pool |= {k for k, v in self.passed_at.items() if v == self.ledger_version}
        if not conds:
            stem = lambda w: w[:-2] if w.endswith("es") and len(w) > 4 else (w[:-1] if w.endswith("s") and len(w) > 3 else w)
            kind = {stem(w) for w in words(self.plan.get("product") or "") if len(w) >= 3}
            for key in self.shelf.sheets:
                label = {stem(w) for w in words((self.shelf.rows.get(key) or {}).get("group") or "") if len(w) >= 3}
                if kind and label and kind & label:
                    pool.add(key)
        out = []
        for key in pool:
            row = self.shelf.rows.get(key) or {}
            if key in self.gone or key in self.vetoed or self.set_aside(key) or row.get("stock") is False or row.get("price") is None:
                continue
            if budget is not None and row["price"] * quantity > budget + 1e-9:
                continue
            if any(k.startswith(key + "|") and k.endswith(f"|{self.ledger_version}") and v for k, v in self.verdicts.items()):
                continue
            if conds and self.model_vetoes(key, conds):
                self.vetoed.add(key)
                continue
            out.append((key not in self.passed, row["price"], key))
        return [k for _, _, k in sorted(out)]

    def code_order(self, target: str, why: str) -> list[tuple[str, str]]:
        """The code puts `target` in the cart (taking out any other line first) and sends the order through the usual
        holds, telling the model why. The replies of the calls it made."""
        tool, order = self.roles.get("cart"), self.roles.get("order")
        replies: list[tuple[str, str]] = []
        if not order:
            return replies
        if tool and target not in self.cart_ids:
            others = [i for i in self.cart_ids if i != target] if not self.basket_form() else []   # a whole-basket hold replaces them
            if self.steps_left() < len(others) + 2:
                return replies               # the lines out, the line in and the order must all fit in the steps left
            for other in others:
                out_call = self.remove_args(other)
                if not out_call or self.steps_left() < 2:
                    return replies
                out_tool, out_args = out_call
                reply = self.env_call([{"name": out_tool, "args": out_args}])
                self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": out_tool, "args": out_args}], "reply": reply, "code_order": True})
                if reply is None or self.finished or self.dead:
                    return replies
                self.code_booking(out_tool, out_args, reply.get("observation"))
                self.read_notices(out_tool, reply.get("observation"))
            found = self.add_call_args(target)
            if not found or self.steps_left() < 2:
                return replies
            tool, in_args = found
            self.agent_placed.add(target)
            reply = self.env_call([{"name": tool, "args": in_args}])
            self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": in_args}], "reply": reply, "code_order": True})
            if reply is None or self.finished or self.dead:
                return replies
            observation = reply.get("observation")
            self.code_booking(tool, in_args, observation)
            self.read_notices(tool, observation)
            self.note_observation(tool, observation)
            said = _compact(observation)
            if (isinstance(observation, dict) and observation.get("error")) or target in self.gone or self.gone_said(said):
                self.gone.add(target)
                self.cart_ids = [i for i in self.cart_ids if i != target]
                self.messages.append({"role": "user", "content": f"[note] {why} The code tried to put {target} in the cart, but the store refused ({said[:160]}). Choose the next fit.\n" + self.status()})
                return replies
        elif tool and len(self.cart_ids) > 1:
            return replies
        row = self.shelf.rows.get(target) or {}
        self.messages.append({"role": "user", "content": f"[note] {why} The cart holds {target} ({row.get('price', '?')}); its own sheet: "
                                                         + "; ".join(self.shelf.evidence_lines(target, self.conditions())) + ". The order goes out now through the usual checks."})
        call = {"id": f"code_order_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(self.neutral_args(order, "Order the cart."))}}
        self.messages.append({"role": "assistant", "content": f"I order {target} now.", "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        for call_id, reply_text in replies or [(call["id"], "Not sent.")]:
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text + "\n" + self.status()})
        self.trace.append({"kind": "code_order", "target": target, "why": why[:120], "ordered": self.finished})
        return replies

    def repair_refusal(self) -> bool:
        """After a refused finish that did not report the line sold out (a new limit, a moved price), the code repairs
        it instead of the model: the held line is re-read when stale and, if it still fits the current limit, ordered
        again; if it is now over the limit, the cheapest opened listing that fits within it takes its place and is
        ordered; with nothing opened that fits, the model is told to look once more before any decline. Once per
        refusal, one product only. True when a turn was spent."""
        if (not self.refusals or self.refused_at <= self.repaired_at or self.finished or not self.roles.get("order")
                or self.steps_left() < 2 or self.time_left() < WRAP_UP_SECONDS + 10 or self.cost >= self.cost_stop):
            return False
        if self.items():
            # several products: a refusal the code cannot read sends every held line's own record to be read again (its
            # stock flag and price say what changed); the bundle recovery takes it from there
            self.repaired_at = self.refused_at
            moved = self.budget_changed_at >= self.refused_at or self.price_moved_at >= self.refused_at or self.relaxed_at >= self.refused_at
            held = [k for k in self.cart_ids if k not in self.gone]
            if held and not moved and not self.gone_said(self.refusal_text) and self.steps_left() >= self.finish_reserve() + len(held) + 2:
                for k in held:
                    self.freshened.discard(k)
                self.reread_lines(held)
                dead = [k for k in held if k in self.gone or (self.shelf.rows.get(k) or {}).get("stock") is False]
                if dead and not (self.finished or self.dead):
                    self.gone.update(dead)
                    self.mark_change()
                    self.trace.append({"kind": "repair", "lines": dead, "dead_by_state": True})
                    return True
            return False
        self.repaired_at = self.refused_at
        line = self.cart_ids[-1] if self.cart_ids else (self.order_targets[-1] if self.order_targets else "")
        if not line or line in self.gone or (set(self.cart_ids) & self.gone):
            return False                     # a sold-out line is the recovery's business
        moved = (self.budget_changed_at >= self.refused_at or self.price_moved_at >= self.refused_at or self.relaxed_at >= self.refused_at
                 or (_PRICE_WORDS.search(self.refusal_text) is not None))
        if not moved and not self.gone_said(self.refusal_text) and self.steps_left() >= self.finish_reserve() + 3:
            # a refusal that names no figure, no price and no stockout the code can read: the held line's own record tells
            # (its stock flag), so it is re-read; a dead line found this way is the recovery's business
            self.freshened.discard(line)
            self.reread_lines([line])
            if self.finished or self.dead:
                return True
            if line in self.gone:
                self.mark_change()
                self.trace.append({"kind": "repair", "line": line, "dead_by_state": True})
                self.messages.append({"role": "user", "content": f"[note] The refused order's line {line} reads as no longer in stock on its own record: take it out and put the next verified fit in, then order.\n" + self.status()})
                return True
        if not moved or self.gone_said(self.refusal_text):
            return False                     # a refusal the code does not read as a limit or price change is the model's to handle
        try:
            self.freshened.discard(line)
            stale = self.freshen_pick()      # one step when the line's record predates the change
            if self.finished or self.dead:
                return True
            if stale:
                # found gone on the re-read: the recovery takes over from here
                self.gone.add(line)
                self.mark_change()
                self.messages.append({"role": "user", "content": "[note] " + stale + "\n" + self.status()})
                return True
            budget, quantity = self.walk_scope()
            row = self.shelf.rows.get(line) or {}
            conds = self.conditions()
            fits_now = (row.get("price") is not None and row.get("stock") is not False
                        and (budget is None or row["price"] * quantity <= budget + 1e-9)
                        and (line in self.passed or not conds or all(self.shelf.says(line, p, v) != "no" for p, v in conds)))
            if fits_now and self.roles.get("cart") and line in self.cart_ids and self.relaxed_at >= self.refused_at:
                # a requirement was given up: the wider set is ranked again by the request's rule before the re-send
                self.relax_done = self.relaxed_at
                wider = self.rank_by_rule([k for k in self.live_fits() if k not in self.model_removed] + [line])
                best = wider[0] if wider else line
                if best != line:
                    self.code_order(best, f"The refused order carried a change (a requirement was dropped); by the request's rule {best} is now the best fit, so it replaces {line}.")
                    return True
            if (fits_now and self.roles.get("cart") and line in self.cart_ids and self.plan.get("priority") == "best"
                    and self.budget_changed_at >= self.refused_at and self.steps_left() >= self.finish_reserve() + 5):
                # a new limit under a 'best' rule: the best-placed unread rows within it are opened, then the opened fits are
                # ranked again with the held line; the held line is re-sent only when it is still the best
                unread = sorted((-(r["price"] or 0), k) for k, r in self.shelf.rows.items()
                                if k not in self.shelf.sheets and k not in self.gone and r.get("stock") is not False and r.get("price") is not None
                                and (budget is None or r["price"] * quantity <= budget + 1e-9) and self.of_kind(r))
                if unread:
                    self.code_reads([k for _, k in unread[:2]])
                    if self.finished or self.dead:
                        return True
                ranked = self.rank_by_rule([k for k in self.live_fits() if k not in self.model_removed] + [line])
                best = ranked[0] if ranked else line
                if best != line and self.measure_value(best) is not None and (self.measure_value(line) is None or self.measure_value(best) > self.measure_value(line)):
                    self.code_order(best, f"The refused order carried a new limit ({budget}); by the request's rule {best} is the best opened fit within it, so it replaces {line}.")
                    return True
            if fits_now and self.roles.get("cart") and line in self.cart_ids:
                self.code_order(line, f"The refused order carried a change (limit now {budget}); the held line still fits it (price {row['price']:g} x {quantity}), so it is ordered again.")
                return True
            if fits_now and not self.roles.get("cart"):
                return False                 # a store without a cart: the model re-sends the order itself
            sure = self.unfolded == 0 and (self.messages_used > 0 or not self.roles.get("message") or self.messages_open() <= 0)
            better = [k for k in self.live_fits() if k != line] if sure else []
            if not sure:
                self.messages.append({"role": "user", "content": (f"[note] The refused order carried a change: the limit is now {budget}, and the held line {line} no longer fits it. "
                                                                 "The shopper may still hold an unasked need, so the code makes no swap: ask the planned question(s) if messages remain, "
                                                                 "then put the cheapest listing that meets everything within the limit in the cart and order it.\n" + self.status())})
                self.trace.append({"kind": "repair", "line": line, "budget": budget, "handed_back": True})
                return False
            if better and self.roles.get("cart"):
                best = (self.rank_by_rule(better) or better)[0]
                self.code_order(best, f"The refused order carried a change (limit now {budget}); the held line {line} no longer fits, so {best} ({(self.shelf.rows.get(best) or {}).get('price', '?')}) takes its place.")
                return True
            self.messages.append({"role": "user", "content": (f"[note] The refused order carried a change: the limit is now {budget}. No opened listing states every condition within it. "
                                                             f"Search once more with the limit {budget} (the category filter and the requirement words) and open the cheapest unread rows within it; "
                                                             "decline only when every candidate within the limit fails a condition by its own records.\n" + self.status())})
            self.trace.append({"kind": "repair", "line": line, "budget": budget, "none": True})
        except Exception as exc:
            self.trace.append({"kind": "repair", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return False

    def zero_fallback_round(self) -> bool:
        """After ZERO_SEARCHES empty free-text searches, once: the code lists the most-seen category under the limit
        itself and opens the two cheapest unread rows within it, so the choice can rest on rows that exist."""
        if (self.zero_fallback or self.zero_searches < ZERO_SEARCHES or self.order_attempts or self.items() or not self.roles.get("search")
                or self.steps_left() < self.finish_reserve() + 5 or self.time_left() < WRAP_UP_SECONDS + 40 or self.cost >= self.cost_hard):
            return False
        self.zero_fallback = True
        try:
            budget, quantity = self.walk_scope()
            counts: dict[str, int] = {}
            for row in self.shelf.rows.values():
                if row.get("group"):
                    counts[row["group"]] = counts.get(row["group"], 0) + 1
            group = max(counts, key=lambda g: counts[g]) if counts else ""
            spent = 0
            if group and budget is not None:
                args = self.walk_args(float(budget) + (1 if float(budget).is_integer() else 0.01), group)
                if args and self.code_search(args):
                    spent += 1
            unread = sorted((k for k, r in self.shelf.rows.items() if r.get("price") is not None and not r.get("sheet") and k not in self.opened
                             and r.get("stock") is not False and k not in self.gone and (budget is None or r["price"] * quantity <= budget + 1e-9)
                             and (not group or not r.get("group") or flat(r["group"]) == flat(group))), key=lambda k: (self.shelf.rows[k]["price"], k))
            if unread:
                spent += self.code_reads(unread[:2])
            if not spent:
                return False
            fits = self.live_fits()
            named = ", ".join(f"{k} ({(self.shelf.rows.get(k) or {}).get('price', '?')})" for k in fits[:3])
            self.messages.append({"role": "user", "content": (f"[note] {self.zero_searches} of your searches returned nothing, so the code listed the category {group or 'seen'} under the limit and opened the cheapest unread rows ({spent} store step(s)). "
                                                             + (f"Listings whose own sheets state every condition within the limit: {named}. Put the best in the cart and order." if named else "Judge the opened rows by their sheets; search with the requirement words only (no brand or page names) if more is needed.")
                                                             + "\n" + self.status())})
            self.trace.append({"kind": "zero_fallback", "group": group, "steps": spent, "fits": fits[:3]})
            return True
        except Exception as exc:
            self.walking = False
            self.trace.append({"kind": "zero_fallback", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return False

    def decline_gate(self) -> str | None:
        """A decline never goes out while an opened listing meets everything known within the limit: the first such
        attempts are held with the fits named; after DECLINE_BLOCKS holds (or within REFUSAL_BLOCK steps of a refused
        finish, or with few steps left) the code carts the best fit and orders it itself. With nothing opened that
        fits, the usual audit applies."""
        try:
            rescued = self.held_rescue()
        except Exception as exc:
            rescued = None
            self.trace.append({"kind": "hold", "rescue_error": f"{type(exc).__name__}: {exc}"[:300]})
        if rescued or self.finished or self.dead:
            return rescued or "Not run: the episode is over."
        paced = self.pacing_decline_hold()
        if paced:
            self.trace.append({"kind": "hold", "decline_paced": paced[:80]})
            return paced
        unseen = self.unseen_decline_hold()
        if unseen:
            return unseen
        bundle = self.bundle_decline_gate()
        if bundle:
            return bundle
        if not self.parse_sure():
            # the code's reading of this store is not trusted: no code-made order over the model's verdict, only the one
            # audit round (the model re-reads and looks once more), then its decline stands
            return self.audit_pass_hold()
        fits = self.live_fits()
        if fits and self.relaxed_at >= 0 and self.step - self.relaxed_at <= 3 and not self.items() and self.roles.get("cart") and self.steps_left() >= 3:
            # a requirement was just dropped and an opened listing meets the rest: no decline, the rule's best is ordered
            pick = self.rank_by_rule(fits)[0]
            self.relax_done = max(self.relax_done, self.relaxed_at)
            self.trace.append({"kind": "hold", "decline_after_relax": pick})
            self.code_order(pick, f"The shopper dropped a requirement; {pick} meets every remaining condition within the limit by its own sheet.")
            if self.finished:
                return "Not run: the code ordered " + pick + " instead (see above); the episode is finished."
            return "Not run (no step used): after the dropped requirement the code ordered " + pick + "; the result is above."
        if not fits:
            pointed = self.pointers_before_decline()
            if pointed:
                return pointed
            probe = self.probe_hold()
            if probe:
                return probe
            read = self.unread_before_decline()
            if read:
                return read
            held = self.decline_hold()
            return held or self.decline_audit() or self.audit_pass_hold()
        if (self.roles.get("message") and self.messages_open() - self.held_back() >= 1 and self.planner_on and self.steps_left() > 6
                and self.time_left() > WRAP_UP_SECONDS + 40 and self.cost < self.cost_hard and not self.ask_held):
            # the shopper may still hold a need no sheet was judged against: the planned questions go out before any verdict
            self.ask_held = True
            if self.interview(force=True):
                return ("Not run yet (no step used): before any decline, the questions the shopper had not been asked went out (see the exchanges above). "
                        "Apply their answers; order a listing that meets everything, or decline only if every candidate fails a condition by its own records.")
        self.decline_blocks += 1
        named = ", ".join(f"{k} ({(self.shelf.rows.get(k) or {}).get('price', '?')}" + (", passed an independent check" if k in self.passed else "") + ")" for k in fits[:3])
        soon = self.refusals and self.step - self.refused_at <= REFUSAL_BLOCK
        # the code overrules a decline only when its picture of the needs is complete: every answer was folded into the
        # ledger, and the shopper was asked at all when messages were allowed
        sure = self.unfolded == 0 and (self.messages_used > 0 or not self.roles.get("message") or self.messages_open() <= 0)
        if not sure and self.decline_blocks > 1:
            return self.decline_hold()
        if sure and (self.decline_blocks >= DECLINE_BLOCKS or soon or self.steps_left() <= 6) and self.roles.get("cart") and self.roles.get("order") and self.steps_left() >= 3:
            pick = (self.rank_by_rule(fits) or fits)[0]
            self.trace.append({"kind": "hold", "decline_overruled": pick, "blocks": self.decline_blocks})
            replies = self.code_order(pick, f"A decline was proposed, but {pick} meets every condition within the limit by its own sheet, and a wrong decline scores 0.")
            if self.finished:
                return "Not run: the code ordered " + pick + " instead (see above); the episode is finished."
            return ("Not run (no step used): instead of declining, the code put " + pick + " in the cart and sent the order; its result is above. "
                    "If it was held or refused, fix the cart and order again; decline only if every listed fit fails a condition by its own records.")
        self.trace.append({"kind": "hold", "decline_blocked": fits[:3], "blocks": self.decline_blocks})
        return ("Not run (no step used): a decline is wrong while a listing meets every condition. These opened listings state every condition within the limit by their "
                f"own sheets and are in stock: {named}. Put the best of them in the cart and order it. A decline can go out only if you name, for each of them, "
                "the condition its own sheet contradicts; otherwise the code orders the first one itself.")

    def audit_pass_hold(self) -> str | None:
        """Once per episode, before the first decline goes out: the model is asked for one audit round (re-read the request
        and the shopper's answers, one more search or view round, then order what fits); a later decline is not held by it."""
        if (self.audit_pass or self.finished or self.dead or self.steps_left() < self.finish_reserve() + 3
                or self.time_left() < WRAP_UP_SECONDS + 25 or self.llm_down or self.cost >= self.cost_stop):
            return None
        self.audit_pass = True
        answers = "\n".join("- " + line[:300] for line in self.said[-6:]) or "- (no exchange with the shopper)"
        tried = "; ".join(_compact(q.get("args"))[:90] for q in self.searches[-6:]) or "(none recorded)"
        self.trace.append({"kind": "hold", "audit_pass": self.step})
        return ("Not run yet (no step used): one audit round comes before any decline, and a wrong decline scores 0.\n"
                "1. Re-read the request and the shopper's answers below; list each condition as it really stands (a story detail or a "
                "wish is not a condition; a property the shopper waved off is free).\n"
                "2. Make one more round: a search in words not tried yet (or the category with the price limit), and open the closest "
                "candidates you have not opened; judge each by its own text.\n"
                "3. If one meets every condition within the limit and in stock, put it in the cart and order it. Decline only if after "
                "this round every candidate still fails a condition by its own records.\n"
                "REQUEST: " + str(self.policy.get("query") or "")[:1500] + "\nSHOPPER'S ANSWERS:\n" + answers
                + "\nSEARCHES TRIED: " + tried + "\nREQUIREMENTS NOW: " + self.ledger())

    def blind(self) -> bool:
        """The code could read no listing's property sheet in this world (a record encoding it does not know) although
        pages were opened or a quarter of the cap is spent: the model then reads the raw results itself, and the questions
        go out without the sheets' labels."""
        return not self.shelf.sheets and (self.page_calls >= 1 or self.step >= max(6, self.max_steps // 4))

    def unseen_decline_hold(self) -> str | None:
        """No decline before the candidates were looked at: every requested product searched for and the page of its best
        candidate opened. A product no opened page belongs to (when the code reads no sheet at all: fewer pages opened than
        products) holds the decline, twice per episode at most, and the model is told what to open."""
        readers = [r for r in (self.roles.get("view"), self.roles.get("details"), self.roles.get("compare")) if r]
        if (not readers or self.unseen_holds >= 2 or self.finished or self.dead or self.steps_left() < self.finish_reserve() + 3
                or self.time_left() < WRAP_UP_SECONDS + 20 or self.cost >= self.cost_stop):
            return None
        products = [p for p in self.product_names() if p]
        opened = [k for k in self.shelf.rows if k in self.shelf.sheets or (self.shelf.rows.get(k) or {}).get("paged")]
        if not opened:
            missing = products if self.page_calls < max(1, len(products)) else []
            missing = missing or ([] if self.page_calls else ["the requested product"])
        elif len(products) > 1:
            owners = [self.product_of(k) for k in opened]
            missing = [p for p in products if p not in owners]
            missing = missing[: max(0, len(missing) - owners.count(""))]      # a page the code cannot place may be any product's
        else:
            missing = []
        if not missing:
            return None
        self.unseen_holds += 1
        named = ", ".join(missing[:4])
        searched = "" if self.search_calls >= max(1, len(products)) else " Search for it first when no result shows it."
        self.trace.append({"kind": "hold", "decline_unseen": missing[:4], "pages": self.page_calls, "blind": not opened})
        if not opened:
            return ("Not run (no step used): no decline before the candidates were looked at, and the code could not read this store's records itself. "
                    f"Read the results above as written: for {named}, take the listing that best matches the request within the limit, open its page with "
                    f"{readers[0]} and judge it by its own text; put what fits in the cart and order it.{searched} Decline only once the best candidate of "
                    "each product was opened and fails a stated condition by its own page.")
        return (f"Not run (no step used): no decline before the candidates were looked at: no listing of {named} was opened yet. Open the best "
                f"candidate within the limit with {readers[0]} and judge it by its own page; order what fits.{searched} Decline only if it fails a "
                "stated condition by its own records.")

    def pointers_before_decline(self) -> str | None:
        """A decline never goes out while a need that points at another listing's page is unsettled and the steps allow
        following it: the code follows it now (a round), then the decline is judged again."""
        if self.items() or not self.open_pointers() or self.finished or self.dead:
            return None
        before = self.step
        saved, self.order_attempts = self.order_attempts, 0      # the pointer rounds wait for no order here
        try:
            moved = self.resolve_pointers()
        finally:
            self.order_attempts = saved
        if not moved and self.step == before:
            return None
        fits = [k for k in self.live_fits() if k not in self.gone]
        if fits and self.roles.get("cart") and self.roles.get("order") and self.steps_left() >= 3:
            pick = self.rank_by_rule(fits)[0]
            self.code_order(pick, f"Before the decline the code followed the shopper's pointer; {pick} meets every condition by its own sheet.")
            if self.finished:
                return "Not run: the code ordered " + pick + " instead (see above); the episode is finished."
            return "Not run (no step used): after following the pointer the code ordered " + pick + "; the result is above."
        return ("Not run (no step used): before any decline the code followed the shopper's pointer (see the note above). "
                "Judge the candidates against the settled value; decline only if every one fails a condition by its own records.")

    def decline_audit(self) -> str | None:
        """The last look before a decline of one product goes out: every opened listing of the requested kind, on offer,
        within the limit and contradicting no condition by its own sheet, is judged by the item check against the
        conditions as heard (cheapest by the request's rule first, AUDIT_CHECKS at most, once per state of the
        conditions); the first that passes is carted and ordered instead."""
        if (self.items() or self.finished or self.dead or not self.roles.get("order") or self.steps_left() < 3
                or self.time_left() < WRAP_UP_SECONDS + 15 or self.cost >= self.cost_hard or self.audit_mark == self.ledger_version):
            return None
        self.audit_mark = self.ledger_version
        budget, quantity = self.walk_scope()
        conds = self.conditions()
        pool = []
        for key in self.shelf.sheets:
            row = self.shelf.rows.get(key) or {}
            if key in self.gone or row.get("stock") is False or row.get("price") is None or not self.of_kind(row):
                continue
            if budget is not None and row["price"] * quantity > budget + 1e-9:
                continue
            if any(self.shelf.says(key, p, v) == "no" for p, v in conds):
                continue
            pool.append(key)
        checked = []
        for key in self.rank_by_rule(pool)[:AUDIT_CHECKS]:
            verdict = self.verify_item(key, "ORDER")
            checked.append(key)
            if not verdict and self.passed_at.get(key) == self.ledger_version:
                self.trace.append({"kind": "decline_audit", "order": key, "checked": checked})
                self.code_order(key, f"Before the decline the code checked the opened candidates against every condition heard; {key} passed the check.")
                if self.finished:
                    return "Not run: the code ordered " + key + " instead (see above); the episode is finished."
                return "Not run (no step used): before the decline the code found " + key + " meeting every condition and ordered it; the result is above."
        self.trace.append({"kind": "decline_audit", "order": None, "checked": checked})
        return None

    def unread_rows_read(self, most: int = 2) -> list[str]:
        """The code opens the cheapest unopened rows of the requested kind within the limit, not shown out of stock (each
        row once per episode); the rows it opened."""
        budget, quantity = self.walk_scope()
        wanted = {w for _, v in self.conditions() for w in words(v) if len(w) >= 3}
        # rows whose title already names a required value first, then the cheapest
        rows = sorted((not (wanted & set(words(row.get("title") or ""))), row["price"], key) for key, row in self.shelf.rows.items()
                      if key not in self.shelf.sheets and key not in self.gone and key not in self.decline_reads and row.get("stock") is not False
                      and row.get("price") is not None and (budget is None or row["price"] * quantity <= budget + 1e-9) and self.of_kind(row))
        room = self.steps_left() - 3
        keys = [k for _, _, k in rows][: max(0, min(most, room))]
        if not keys:
            return []
        self.decline_reads.update(keys)
        return keys if self.code_reads(keys) else []

    def unread_before_decline(self) -> str | None:
        """Invariant: no decline while an affordable listing of the requested kind, not shown out of stock, was never
        opened. The code opens the cheapest two such rows itself (once per episode for each), then orders a fit they
        reveal, or lets the decline be judged again."""
        if self.items() or self.finished or self.dead or self.time_left() < WRAP_UP_SECONDS + 10:
            return None
        keys = self.unread_rows_read()
        if not keys and not self.decline_paged and self.steps_left() >= self.finish_reserve() + 4:
            # the model's search reported more matches than it showed: the rest is paged by the code before any decline
            self.decline_paged = True
            if self.page_early(force=True) and not (self.finished or self.dead):
                keys = self.unread_rows_read()
        if not keys or self.finished or self.dead:
            return None
        fits = [k for k in self.live_fits() if k not in self.model_removed]
        self.trace.append({"kind": "decline_reads", "opened": keys, "fits": fits[:3]})
        if fits and self.roles.get("order") and self.steps_left() >= 3:
            pick = self.rank_by_rule(fits)[0]
            self.code_order(pick, f"Before the decline the code opened the unread rows within the limit; {pick} meets every condition by its own sheet.")
            if self.finished:
                return "Not run: the code ordered " + pick + " instead (see above); the episode is finished."
            return "Not run (no step used): before the decline the code opened unread listings and ordered " + pick + "; the result is above."
        return ("Not run (no step used): before any decline the code opened the unread listings within the limit (" + ", ".join(keys)
                + "); their sheets are above in the listings table. Decline only if each fails a condition by its own records.")

    # ------------------------------------------------------------------ the code-owned interview (several properties of one product per message)

    def product_names(self) -> list[str]:
        """The products a question can be about: each requested product, or the one product of the request."""
        items = self.items()
        if items:
            return [name for name, _ in items]
        product = str(self.plan.get("product") or "").strip()
        return [product] if product else [""]

    @staticmethod
    def product_words(product: str) -> set[str]:
        return {w for w in words(product) if len(w) >= 3 and w not in ("and", "the", "for", "set", "with", "pair", "pack", "of")}

    def product_of(self, key: str, extra: str = "") -> str:
        """The requested product a listing belongs to: the one whose name's words its store label and title carry most, words
        no other product's name shares counting first ('laptop screen protector' is the protector, not the laptop sleeve).
        '' when no product's words appear."""
        products = self.product_names()
        row = self.shelf.rows.get(key) or {}
        have = set(words(row.get("group") or "")) | set(words(row.get("title") or "")) | set(words(extra))
        if not have:
            return ""
        named = {p: self.product_words(p) for p in products}
        shared = {w for p, pw in named.items() for w in pw if any(w in named[q] for q in products if q != p)}
        label = set(words(row.get("group") or ""))
        hit = lambda w, pool=have: any(h == w or (len(w) >= 4 and h.startswith(w)) or (len(h) >= 4 and w.startswith(h)) for h in pool)
        scored = []
        for index, product in enumerate(products):
            pw = named[product]
            own = sum(1 for w in pw if w not in shared and hit(w))
            total = sum(1 for w in pw if hit(w))
            # the store's own category label speaks first: a cable whose title says "charger" stays a cable
            in_label = sum(1 for w in pw if hit(w, label)) if label else 0
            scored.append((-in_label, -own, -total, index, product))
        scored.sort()
        return scored[0][4] if scored and scored[0][2] < 0 else ""

    def sheets_of(self, product: str) -> list[str]:
        """Opened sheets that belong to one product: with several products, those whose store label or title name that product
        (all sheets when none names any product); with one product, every sheet."""
        keys = sorted(self.shelf.sheets)
        if not self.items():
            return keys
        owners = {key: self.product_of(key) for key in keys}
        mine = [key for key in keys if owners[key] == product]
        return mine or ([] if any(owners.values()) else keys)

    def product_conditions(self, product: str) -> list[tuple[str, str]]:
        """The conditions in force that bear on one product: with several products, those whose property starts with the
        product's words plus those naming no product; with one product, all of them."""
        conds = self.conditions()
        if not self.items():
            return conds
        names = [self.product_words(n) for n in self.product_names()]
        mine = self.product_words(product)
        out = []
        for prop, value in conds:
            pw = set(words(prop))
            # the product whose name shares the most words with the property owns it ("laptop memory module form
            # factor" is the memory module's, not the laptop sleeve's)
            best = max((len(n & pw) for n in names if n), default=0)
            owners = [n for n in names if n and best and len(n & pw) == best]
            if not owners or (mine and mine in owners):
                prop_clean = " ".join(w for w in prop.split() if flat(w) not in mine) if mine & pw else prop
                out.append((prop_clean.strip() or prop, value))
        return out

    def asked_for(self, product: str) -> list[str]:
        """Properties already put to the shopper about one product (confused replies included; a property parked because
        the shopper did not follow a long question is not asked yet)."""
        return [q["property"] for q in self.interview_log if q["product"] == product and q["outcome"] not in ("parked", "soft")]

    def property_candidates(self, product: str) -> list[str]:
        """The properties worth a question about one product, best first, from the code alone: the story's hints first, then
        labels most of its opened sheets carry whose values differ between them; never a property a condition settles, one
        already asked (or met with confusion), one the shopper does not care about, a store detail, or a list of variants."""
        sheets = self.sheets_of(product)
        taken = [set(words(p)) - {"type", "kind", "the", "of"} for p, _ in self.product_conditions(product)]
        settled = [set(words(t)) for t in self.asked_for(product)]
        shared = [set(words(t)) for t in self.free + self.confused + self.asked_props]
        exact: list[set[str]] = []
        if len(self.product_names()) > 1:
            # with several products a list about one product settles nothing of another (a "form factor" put about the
            # computer leaves the memory module's form factor open): only the product's own questions count, and the
            # properties waved off in reply to them or named with the product's words
            mine = self.product_words(product)
            own = list(self.free_of.get(product, [])) + [f for f in self.free if mine & set(words(f))]
            settled += [set(words(t)) - mine for t in own]
        else:
            settled += shared
        hints = [flat(h) for h in self.plan.get("hint_properties") or [] if isinstance(h, str)]
        asked_cores = {self.label_core(t) for t in self.asked_for(product)} - {""}
        count: dict[str, int] = {}
        values: dict[str, set[str]] = {}
        for key in sheets:
            for label, value in (self.shelf.sheets.get(key) or {}).items():
                text = flat(value)[:40]
                if not text or re.search(META_KEYS, label) or len(re.split(r"[,;|/]", str(value))) >= 3:
                    continue
                count[label] = count.get(label, 0) + 1
                values.setdefault(label, set()).add(text)
        pool = set(count) | {h for h in hints if h}
        ranked: list[tuple[tuple, str]] = []
        for label in pool:
            lw = set(words(label))
            if not lw or any(t and (t <= lw or lw <= t) for t in taken) or any(s and (s <= lw or lw <= s) for s in settled) or lw in exact:
                continue
            if self.label_core(label) in asked_cores:
                continue                 # the same property in another spelling ("protein source", "proteinsource") was asked
            hinted = any(set(words(h)) and (set(words(h)) <= lw or lw <= set(words(h))) for h in hints)
            differs = len(values.get(label, ())) >= 2
            ranked.append(((not hinted, not differs, -count.get(label, 0), len(lw), label), label))
        return [label for _, label in sorted(ranked)]

    def story_order(self, product: str, candidates: list[str]) -> list[str]:
        """The code's shortlist put in story order by one cheap model call: the model may only reorder the shortlist (a
        property it names outside it is ignored), and the code's order stands when the call gives nothing."""
        if len(candidates) < RANK_MODEL_FROM or self.cost >= self.cost_hard or self.time_left() < WRAP_UP_SECONDS + 30:
            return candidates
        mark = f"{product}|{len(self.shelf.sheets)}|{len(self.interview_log)}"
        if self.story_cache.get(mark) is not None:
            cached = [c for c in self.story_cache[mark] if c in candidates]
            return cached + [c for c in candidates if c not in cached]
        left = self.messages_open() - self.held_back() if self.message_budget is not None else UNSTATED_MESSAGES
        count = max(1, min(len(candidates), max(1, left) * self.width()))
        prompt = RANK_PROMPT.replace("QUESTION_COUNT", str(count)) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        if self.items():
            prompt += f"\n\nTHE QUESTIONS ARE ABOUT THIS PRODUCT OF THE REQUEST: {product}"
        if self.interview_log:
            prompt += "\n\nALREADY ASKED (never again):\n" + "\n".join(f"- {q['product']} {q['property']}: {q['outcome']}" for q in self.interview_log)
        prompt += "\n\nPROPERTIES TO CHOOSE FROM (only these; the listings carry them):\n" + "\n".join(f"- {c}" for c in candidates[:18])
        result = self.judge(prompt, 800, [READ_MODEL, JUDGE_MODEL])
        picked: list[str] = []
        for item in (result or {}).get("questions") or []:
            name = flat(item.get("property") if isinstance(item, dict) else item)
            match = next((c for c in candidates if flat(c) == name or (set(words(c)) and set(words(c)) <= set(words(name)))), None)
            if match and match not in picked:
                picked.append(match)
        self.trace.append({"kind": "story_order", "product": product, "candidates": candidates[:18], "picked": picked})
        self.story_cache[mark] = picked + [c for c in candidates if c not in picked]
        return list(self.story_cache[mark])

    def next_question(self, skip: tuple[str, ...] = (), cap: int = 0) -> tuple[str, list[str]] | None:
        """The product and the properties the next message names. Products without a firm need come first, then the one
        asked least; every message left is used while some product still has unasked properties. The properties are the
        product's best candidates (story and most-carried, differing labels first), as many as the current group size;
        properties parked after a reply the shopper did not follow go last. With `skip` (the partner of a shared message)
        the products named there are passed over, nothing waits, and the list holds at most `cap` properties."""
        products = self.product_names()
        if self.plain_next and not skip:
            return self.plain_next[0], list(self.plain_next[1])
        while self.redo_queue and not skip:
            # a complained message is put again one property at a time, in its own order (its first property first)
            product, prop = self.redo_queue[0]
            if prop in self.asked_for(product) or any(set(words(prop)) and set(words(prop)) <= set(words(f))
                                                      for f in self.free_of.get(product, [])):
                self.redo_queue.pop(0)
                continue
            return product, [prop]
        # a product whose firm need was heard is done (a long world may put ASKS_AFTER_NEED more questions about it)
        settled = [p for p in products if self.needs_of.get(p, 0) > 0 and (not self.long_world()
                   or self.asks_of.get(p, 0) - self.asks_at_need.get(p, 0) >= ASKS_AFTER_NEED)]
        open_ = [p for p in products if p not in settled]
        # every message the request allows is used: when each product already gave a firm need, the next message asks
        # them again about properties not named yet (a second need may sit there; an unused message reveals nothing)
        products = open_ or ([p for p in settled if self.property_candidates(p)] if not skip and self.message_budget is not None else [])
        if not products:
            return None
        blind = self.blind()
        unread = [] if blind else [p for p in products if self.items() and not self.sheets_of(p)]
        unasked = [p for p in products if not self.asks_of.get(p)]
        if skip:
            products = [p for p in products if p not in skip and p not in unread and not self.asks_of.get(p)]
        elif (unread and self.step < self.max_steps // (3 if not self.messages_used else 2) and self.time_left() > WRAP_UP_SECONDS + 90
              and self.messages_open() - self.held_back() <= len(unasked) + (1 if any(p in unasked for p in unread) else 0)):
            # the questions left are as few as the products still waiting: a product whose listings are not open yet gets
            # its question once they are, so the messages spread over the products instead of all going to the first
            return "", []
        order = sorted(products, key=lambda p: (self.needs_of.get(p, 0) > 0, self.asks_of.get(p, 0), products.index(p)))
        for product in order:
            if product in unread:
                continue
            candidates = self.property_candidates(product)
            if not candidates and blind:
                # no sheet to take labels from: the question names directly what a hidden need is most often about
                gone_by = self.free_of.get(product, []) if len(self.product_names()) > 1 else self.confused + self.free + self.asked_props
                done = [set(words(t)) for t in self.asked_for(product) + list(gone_by)]
                candidates = [c for c in BLIND_ASKS if not any(d and d & set(words(c)) for d in done)]
            if not candidates:
                continue
            generic = blind and not self.property_candidates(product)
            ordered = candidates if generic else self.story_order(product, candidates)
            if ordered and generic:
                # no sheet gave the store's own property names: a generic word goes alone (a shopper hears one plain word
                # more loosely than a list of them)
                return product, [ordered[0]]
            if ordered:
                # the best property leads (a shopper who reads only the first name answers about it); the rest of the list
                # takes properties never named before, then those named after the lead of an earlier question
                soft = {q["property"] for q in self.interview_log if q["product"] == product and q["outcome"] in ("soft", "parked")}
                if self.group_form != "first":
                    # the rotation: the next message moves on to planned properties no message named yet, and a property an
                    # earlier list named without an answer comes back only behind them (so one the shopper does not mind
                    # cannot keep hiding the ones behind it)
                    ordered = [c for c in ordered if c not in soft] + [c for c in ordered if c in soft]
                rest = [c for c in ordered[1:] if c not in soft] + [c for c in ordered[1:] if c in soft]
                # a list keeps to short labels in the store's own words (a long label makes the list read as wordy), each
                # property once (two spellings of one label, or a label with a bracketed note, name the same property)
                group, cores = [ordered[0]], {self.label_core(ordered[0])}
                for c in rest:
                    core = self.label_core(c)
                    if len(group) < (cap or self.width()) and core and core not in cores and len(re.sub(r"[_.]+", " ", c).split()) <= WIDE_WORDS:
                        group.append(c)
                        cores.add(core)
                return product, group
        return ("", []) if unread else None

    @staticmethod
    def label_core(label: str) -> str:
        """A property label reduced to its letters and digits, without a bracketed note: the same property in two spellings."""
        return re.sub(r"[^a-z0-9]+", "", re.sub(r"\s*\([^)]*\)", "", str(label).lower()))

    def width(self) -> int:
        """Properties the next code question names at most: the current width, and one once the shopper complained of crowding;
        the episode's first question names at most FIRST_WIDTH (a shopper who minds long lists minds them most at the start,
        and a reply to it shows whether a longer list is taken)."""
        if self.crowded:
            return KEYS_AFTER_FIRST
        if not self.width_log:
            return max(1, min(self.group_size, FIRST_WIDTH))
        return max(1, self.group_size)

    def narrow_for_crowd(self) -> None:
        """A crowding complaint: one property a message for the rest of the episode."""
        if not self.crowded:
            self.trace.append({"kind": "width", "from": self.group_size, "to": KEYS_AFTER_FIRST, "why": "crowding complaint", "step": self.step})
        self.crowded = True
        self.group_size = KEYS_AFTER_FIRST

    def asked_already(self, text: str) -> str:
        """The property (and product) an earlier question of this episode already put to the shopper that `text` asks about
        again, or ''. With several products the product must be named too."""
        low = " " + flat(text) + " "
        for entry in self.interview_log:
            prop = flat(entry["property"])
            if not prop or f" {prop} " not in low or entry["outcome"] == "parked":
                continue
            product = entry["product"]
            if self.items() and product and not any(f" {w} " in low or f" {w}s " in low for w in self.product_words(product)):
                continue
            return f"{product} {entry['property']}".strip()
        return ""

    def question_text(self, product: str, props: list[str]) -> str:
        """The neutral template of every question: the product named, its properties listed in one sentence with a single
        question mark, no example values."""
        labels = [re.sub(r"[_.]+", " ", p).strip() for p in props if p]
        named = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " or " + labels[-1]
        if product and self.items() and len(labels) == 1:
            return f"For the {product}, which {named} would you like?"
        if product:
            return f"What {named} would you like for the {product}?" if len(labels) > 1 else f"Which {named} would you like for the {product}?"
        return f"What {named} would you like?" if len(labels) > 1 else f"Which {named} would you like?"

    def plain_text(self, product: str, props: list[str]) -> str:
        """The plainer second wording of a question the shopper did not follow: shorter, fewer properties (half the list, or
        the property alone), joined by "or"."""
        labels = [re.sub(r"[_.]+", " ", p).strip() for p in props if p] or [""]
        label = " or ".join(labels)
        if product and self.items():
            return f"What {label} do you need for the {product}?"
        return f"What {label} do you need?"

    def interview_ready(self, force: bool) -> bool:
        """Whether a question may go out now: a message tool and messages left, enough sheets opened (or a fifth of the cap
        spent), steps and money in reserve, and not in the commit phase."""
        tool = self.roles.get("message")
        limit = self.message_budget - self.held_back() if self.message_budget is not None else UNSTATED_MESSAGES
        if not tool or self.messages_used >= limit or self.cost >= self.cost_stop + STOP_OVER or tool in self.capped:
            return False                     # a question is cheap and a hidden need decides the pass: asked well past the hard line
        late = self.time_left() < WRAP_UP_SECONDS + (40 if self.messages_used else 15)
        if self.steps_left() <= INTERVIEW_RESERVE or late or (self.commit_phase() and self.messages_used):
            return False
        if force:
            return True
        return (len(self.shelf.sheets) >= INTERVIEW_FROM or self.step >= max(6, self.max_steps // 5)
                or (len(self.shelf.sheets) >= 1 and self.max_steps >= 20 and self.step >= self.max_steps // 4))

    def read_allowance(self) -> bool:
        """With no stated message cap and a store tool that reports the allowance (no arguments, its description speaking
        of an allowance), the code reads it once before the first question: the count left (or the cap less the count used)
        becomes the message cap. True when a step was spent."""
        tool = self.roles.get("usage")
        if self.allowance_read or not tool or self.steps_left() <= INTERVIEW_RESERVE + 3 or self.max_steps < 20:
            return False
        self.allowance_read = True
        reply = self.env_call([{"name": tool, "args": {}}])
        self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": {}}], "reply": reply, "allowance": True})
        if reply is None or self.finished or self.dead:
            return True
        observation = reply.get("observation")
        self.code_booking(tool, {}, observation)
        numbers: list[tuple[str, int]] = []
        def walk(node: Any, path: str, depth: int = 0) -> None:
            if depth > 3:
                return
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{path} {k}", depth + 1)
            elif isinstance(node, (int, float)) and not isinstance(node, bool) and 0 <= node <= 50:
                numbers.append((flat(re.sub(r"([a-z])([A-Z])", r"\1 \2", path).replace("_", " ")), int(node)))
        walk(observation, "")
        talk = [(k, v) for k, v in numbers if re.search(r"messag|question|ask|chat|talk", k)] or numbers
        left = next((v for k, v in talk if re.search(r"availab|remain|left", k) and not re.search(r"drawn|used|spent|consum", k)), None)
        used = next((v for k, v in talk if re.search(r"drawn|used|spent|consum", k)), None)
        cap = next((v for k, v in talk if re.search(r"\bcap|max|limit|total", k) and not re.search(r"drawn|used|spent|consum|availab|remain|left", k)), None)
        found = self.messages_used + left if left is not None else (cap if cap is not None else None)
        if found is not None:
            self.message_budget = max(self.messages_used, found)
            self.trace.append({"kind": "limit", "messages": self.message_budget, "from": "allowance report", "used": used})
        self.messages.append({"role": "user", "content": "[note] The code read the store's allowance report (1 store step): " + _compact(observation)[:300] + "\n" + self.status()})
        return True

    def key_of(self, text: str, keys: list[str]) -> int:
        """The index of the listed property a need or a reply line names (the most shared words, the longer label on a tie;
        a label is named when all its words appear), or -1."""
        have = {w.rstrip("s") for w in words(text)}
        best, at = 0, -1
        for index, key in enumerate(keys):
            kw = {w.rstrip("s") for w in words(key) if w not in ("the", "of", "and")}
            if kw and kw <= have and len(kw) > best:
                best, at = len(kw), index
        return at

    def sort_group(self, entries: list[dict[str, Any]], fresh: list[str], answer: str, before_free: int) -> None:
        """The outcome of each property one question named, from the needs the hearing read: a property a firm need names is
        settled; a reply that names only the first property (and nothing firm on the others) shows a shopper who answers the
        first property alone, so the rest are parked for later, shorter questions; a firm need on a later property shows a
        shopper who reads the whole list."""
        keys = [e["property"] for e in entries]
        firm: dict[int, list[str]] = {}
        for need in fresh:
            index = self.key_of(need.partition(":")[0], keys)
            if index < 0:
                index = self.key_of(need, keys)
            firm.setdefault(index if index >= 0 else 0, []).append(need)
        if any(i > 0 for i in firm):
            self.group_form = "any"
            for q in self.interview_log:
                if q["outcome"] == "soft":
                    q["outcome"] = "no need stated"
                    self.asked_props.append(q["property"])
        named = [i for i, k in enumerate(keys) if self.key_of(answer, [k]) == 0]
        # the properties the hearing read as waved off ("any of those" frees the whole list, "that one" only the lead)
        freed = {self.key_of(f, keys) for f in self.free[before_free:]} - {-1}
        # a shrug that names the first property alone (a firm value on it says nothing about the rest)
        first_only = len(keys) > 1 and not firm and named == [0] and self.group_form != "any"
        if first_only:
            self.group_form = "first"
            if self.group_size > KEYS_AFTER_FIRST:
                self.trace.append({"kind": "width", "from": self.group_size, "to": KEYS_AFTER_FIRST, "why": "answers the first property only", "step": self.step})
            self.group_size = min(self.group_size, KEYS_AFTER_FIRST)
        # one need heard from a list: a shopper who answers one property a message stops at it, so the properties named after
        # it are not settled by the silence (a second need may sit there); those before it were passed over, so they are
        top = max(firm) if len(firm) == 1 and len(fresh) == 1 else len(keys)
        for index, entry in enumerate(entries):
            if index in firm:
                entry["outcome"] = "a firm need: " + "; ".join(firm[index])
            elif first_only and index > 0:
                entry["outcome"] = "parked"
                if entry["property"] not in self.parked:
                    self.parked.append(entry["property"])
            elif index > top and index not in named and index not in freed:
                entry["outcome"] = "soft"
            elif index > 0 and self.group_form != "any" and index not in named and index not in freed:
                # named after the lead, and neither waved off by the reply nor spoken of, while no reply has shown that the
                # whole list is read: not settled yet; it may lead a later question (the rotation)
                entry["outcome"] = "soft"
            elif len(self.free) > before_free:
                entry["outcome"] = "no preference"
            else:
                entry["outcome"] = "no need stated"

    def interview(self, force: bool = False) -> bool:
        """The code puts one question to the shopper: the product and the properties it chose, in the fixed template. The
        reply is read by the usual hearing and sorted per property (sort_group). A reply the shopper did not follow counts
        as 'no preference' for every property it named, and while no reply has shown that the whole list is read it also
        shortens the next questions. The interview goes on while messages are left and some product has unasked
        properties. True when a turn was spent."""
        if not self.interview_ready(force):
            if not self.roles.get("message") or self.messages_open() - self.held_back() <= 0:
                self.planner_on = False
            return False
        if self.message_budget is None and self.read_allowance():
            return True
        if self.plan.get("ask_first") and not self.asked:
            return self.ask_planned(force=True)       # what the request says to learn first keeps its own, plainly worded question
        choice = self.next_question()
        if choice is not None and choice[0] == "" and not choice[1]:
            return False                 # a product's listings are not open yet: the question waits, the planner stays on
        if not choice:
            # nothing to ask yet: with few sheets open the planner waits for more (it stays on), else it is done
            self.planner_on = len(self.shelf.sheets) < INTERVIEW_FROM + 2 and self.steps_left() > INTERVIEW_RESERVE + 4
            return False
        product, props = choice
        plain = bool(self.plain_next) and (product, props) == (self.plain_next[0], list(self.plain_next[1]))
        redo = not plain and bool(self.redo_queue) and (product, props) == (self.redo_queue[0][0], [self.redo_queue[0][1]])
        if not force and not plain and not redo and not self.asks_of.get(product) and self.survey(product):
            return True                  # the menu was thin: more of the product's sheets are open now, the question comes next turn
        partner = self.partner_for(product) if not plain and not redo and self.items() else None
        if partner == "survey":
            return True                  # the partner product's sheets were widened first
        if partner:
            # two products in one message (only when products outnumber the messages left): with two products one short
            # property each; with three or more (a muddled shopper hears any position alike, race 18) the lead keeps two
            props = props[: (2 if len(self.product_names()) >= 3 else 1)]
            partner = (partner[0], list(partner[1])[:1])
        self.plain_next = None
        tool = self.roles.get("message")
        args = None
        while props:
            text = (self.pair_text(product, props, partner[0], partner[1]) if partner else
                    self.plain_text(product, props) if plain else self.question_text(product, props))
            args = self.message_args(text)
            if not args:
                self.planner_on = False
                return False
            args, problems = _check(args, self.schemas[tool], "args")
            if not problems:
                break
            if partner:
                partner = None           # too long for the message field as a pair: the first product alone
                continue
            props = props[:-1] if len(props) > 1 else []     # too long for the message field: one property fewer
        if not props or args is None:
            self.planner_on = False
            return False
        if redo:
            self.redo_queue.pop(0)
            self.crowd_redos += 1
            self.crowd_tried[f"{product}|{props[0]}"] = self.crowd_tried.get(f"{product}|{props[0]}", 0) + 1
        self.width_log.append(len(props))
        entries = [{"product": product, "property": prop, "outcome": "pending", "step": self.step} for prop in props]
        if partner:
            entries += [{"product": partner[0], "property": prop, "outcome": "pending", "step": self.step} for prop in partner[1]]
            self.asks_of[partner[0]] = self.asks_of.get(partner[0], 0) + 1
        self.interview_log.extend(entries)
        self.asks_of[product] = self.asks_of.get(product, 0) + 1
        self.parked = [p for p in self.parked if p not in props and not (partner and p in partner[1])]
        self.last_group = list(props) + (list(partner[1]) if partner else [])
        before_needs, before_free = len(self.revealed), len(self.free)
        call = {"id": f"ask_{len(self.interview_log)}_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": f"I ask the shopper about the {product + ' ' if product else ''}{', '.join(props)}.", "tool_calls": [call]})
        self.last_unclear = False
        self.last_crowded = False
        self.last_lines = []
        used_before = self.messages_used
        replies = self.run_calls([call], planned=True)
        for index, (call_id, reply_text) in enumerate(replies):
            if index == len(replies) - 1:
                reply_text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text})
        answer = (replies[0][1].split(chr(10) + "[status]")[0] if replies else "")[:300]
        if self.messages_used == used_before or tool in self.capped:
            # the message never reached the shopper (held, or the allowance is spent): nothing was asked
            for entry in entries:
                entry["outcome"] = "parked"
            self.last_group = []
            self.planner_on = False if self.messages_used == used_before else (self.messages_open() - self.held_back() > 0 and tool not in self.capped)
            return bool(self.messages_used != used_before)
        fresh = self.revealed[before_needs:]
        self.free_of.setdefault(product, []).extend(f for f in self.free[before_free:] if f not in self.free_of.get(product, []))
        if fresh:
            owner = self.need_owner(fresh, product)
            if not self.needs_of.get(owner):
                self.asks_at_need[owner] = self.asks_of.get(owner, 0)
            self.needs_of[owner] = self.needs_of.get(owner, 0) + 1
        key = f"{product}|{props[0]}"
        if partner and not fresh and (self.last_crowded or self.last_unclear):
            # a shared message the shopper refused or did not follow: no re-ask of the pair; a crowding complaint narrows the
            # width, and every property it named counts as asked (each product may still get a question of its own)
            again: list[tuple[str, str]] = []
            if self.last_crowded:
                self.narrow_for_crowd()
                if self.crowd_redos < CROWD_REDOS and self.messages_open() - self.held_back() >= 1:
                    # each product's lead property is put again on its own (one property a message from now on)
                    again = [(product, props[0]), (partner[0], partner[1][0])]
                    self.redo_queue = again + [q for q in self.redo_queue if q not in again]
                    self.crowd_redos += 1
            for entry in entries:
                if (entry["product"], entry["property"]) in again:
                    entry["outcome"] = "parked"
                    continue
                entry["outcome"] = "the shopper did not follow the question: counted as no preference"
                self.confused.append(entry["property"])
                self.asked_props.append(entry["property"])
            self.asked.append(f"- {product}, {partner[0]}: asked \"{text}\" -> {answer} (not followed)")
            self.trace.append({"kind": "interview", "product": product, "partner": partner[0], "properties": [e["property"] for e in entries],
                               "outcomes": ["not followed"], "size": self.width(), "step": self.step})
            self.last_group = []
            self.planner_on = (self.next_question() is not None) and self.messages_open() - self.held_back() > 0
            return True
        if self.last_crowded and not fresh:
            # the shopper refused the message for asking too much at once: the width is one from now on (narrow_for_crowd),
            # and what the message asked is put again one property at a time, beginning with its first property (a complaint
            # about a one-property message is noise: that property is put again too); a property put again twice counts as asked
            self.narrow_for_crowd()
            again = [p for p in props if self.crowd_tried.get(f"{product}|{p}", 0) < 2]
            room = self.messages_open() - self.held_back() >= 1 and self.steps_left() > INTERVIEW_RESERVE + 1
            if again and room and self.crowd_redos < CROWD_REDOS:
                queued = [(product, p) for p in again]
                self.redo_queue = queued + [q for q in self.redo_queue if q not in queued]
                for entry in entries:
                    entry["outcome"] = "parked" if entry["property"] in again else "complained twice: counted as asked"
                self.asked.append(f"- {product + ' ' if product else ''}{', '.join(props)}: asked \"{text}\" -> {answer} (too much at once: put again one property a message)")
                self.trace.append({"kind": "interview", "product": product, "properties": props, "outcomes": ["crowded, re-asked one at a time"],
                                   "size": self.width(), "requeued": again, "step": self.step})
                self.asked_props.extend(e["property"] for e in entries if e["outcome"] != "parked")
                self.last_group = []
                self.planner_on = True
                return True
            for entry in entries:
                entry["outcome"] = "the shopper refused the message as too crowded: counted as asked"
            self.asked_props.extend(props)
            self.asked.append(f"- {product + ' ' if product else ''}{', '.join(props)}: asked \"{text}\" -> {answer}")
            self.trace.append({"kind": "interview", "product": product, "properties": props, "outcomes": ["crowded, not put again"], "size": self.width(), "step": self.step})
            self.last_group = []
            self.planner_on = (self.next_question() is not None) and self.messages_open() - self.held_back() > 0
            return True
        if (self.last_unclear and not fresh and sum(self.plain_tries.values()) < PLAIN_RETRIES
                and self.messages_open() - self.held_back() >= 2 and self.steps_left() > INTERVIEW_RESERVE + 1):
            # a question the shopper did not follow: the width is halved and the first half is put once more, in plainer
            # words, before the interview moves on (the second half goes back to the planned properties)
            half = max(KEYS_AFTER_FIRST, len(props) // 2)
            self.group_size = min(self.group_size, half)
            self.plain_tries[key] = self.plain_tries.get(key, 0) + 1
            self.plain_next = (product, props[:half])
            for entry in entries:
                entry["outcome"] = "parked"
            self.parked.extend(p for p in props[half:] if p not in self.parked)
            self.asked.append(f"- {product + ' ' if product else ''}{', '.join(props)}: asked \"{text}\" -> {answer} (not followed: put again in plainer words)")
            self.trace.append({"kind": "interview", "product": product, "properties": props, "outcomes": ["not followed, re-asked plainly"],
                               "size": self.group_size, "plain": props[:half], "step": self.step})
            if self.group_size < len(props):
                self.trace.append({"kind": "width", "from": len(props), "to": self.group_size, "why": "not followed", "step": self.step})
            self.last_group = []
            self.planner_on = True
            return True
        if self.last_unclear and not fresh:
            if len(props) > 1 and sum(self.plain_tries.values()) >= PLAIN_RETRIES:
                # a list the shopper did not follow after the plainer re-ask was spent: the next questions name half as many
                # properties (a first miss with no message to spare on a re-ask keeps the width: the next message covers more)
                self.group_size = min(self.group_size, max(KEYS_AFTER_FIRST, len(props) // 2))
                self.trace.append({"kind": "width", "from": len(props), "to": self.group_size, "why": "not followed again", "step": self.step})
            for entry in entries:
                entry["outcome"] = "the shopper did not follow the question: counted as no preference"
            self.confused.extend(props)
        else:
            # which properties the reply speaks of is read from the shopper's own words, not from the hearing's paraphrase
            self.sort_group(entries, fresh, " ".join(self.last_lines) or answer, before_free)
            if self.unfolded and not fresh:
                for entry in entries:
                    if entry["outcome"] == "no need stated":
                        entry["outcome"] = "no need read"
        self.asked_props.extend(e["property"] for e in entries if e["outcome"] not in ("parked", "soft"))
        self.asked.append(f"- {product + ' ' if product else ''}{', '.join(props)}: asked \"{text}\" -> {answer}")
        self.trace.append({"kind": "interview", "product": product, "properties": props, "outcomes": [e["outcome"][:60] for e in entries],
                           "form": self.group_form, "size": self.group_size, "step": self.step})
        self.last_group = []
        self.planner_on = (self.next_question() is not None or len(self.shelf.sheets) < INTERVIEW_FROM + 2) and self.messages_open() - self.held_back() > 0
        return True

    def partner_for(self, product: str) -> Any:
        """With several products and fewer messages left than products still waiting for a question, one message asks
        about two products, part of the width each ("What a or b would you like for the X, and what c for the Y?"), so no
        product goes without a question. The partner's sheets are widened first like any product's. None when the message
        stays with one product, 'survey' when store steps went to the partner's sheets, else (partner, its properties)."""
        if (self.crowded or self.group_form == "first" or self.width() < 2 or self.redo_queue or self.plain_next or self.message_budget is None
                or self.blind()):
            return None                  # (a generic word with no sheet behind it is heard best alone)
        waiting = [p for p in self.product_names() if p != product and not self.asks_of.get(p) and not self.needs_of.get(p)]
        left = self.messages_open() - self.held_back()
        if left < 1 or len(waiting) + 1 <= left:
            return None
        for round_ in range(2):
            found = self.next_question(skip=(product,), cap=max(1, self.width() // 2))
            if not found or not found[0] or not found[1] or found[0] == product:
                return None
            if round_ == 0 and found[0] not in self.surveyed and self.survey(found[0]):
                return "survey"
            if found[0] in self.surveyed or round_ == 1:
                return found
        return None

    def pair_text(self, first: str, props: list[str], second: str, more: list[str]) -> str:
        """One message about two products, each with its own short list, in the neutral template."""
        def listed(names: list[str]) -> str:
            labels = [re.sub(r"[_.]+", " ", p).strip() for p in names if p]
            return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " or " + labels[-1]
        return f"What {listed(props)} would you like for the {first}, and what {listed(more)} for the {second}?"

    def survey(self, product: str) -> bool:
        """Before the first question about a product, the code widens the property menu: a hidden need is about a property
        of the product's own listings, and the labels a handful of sheets carry miss it far more often than those of six.
        With fewer than SURVEY_SHEETS of the product's sheets open, it opens the cheapest unread rows of that product that
        the searches showed (in stock, within the limit first), at most SURVEY_READS, once per product, only while the
        steps after them still cover the questions left, SURVEY_SPARE more and the finish. True when store steps were spent."""
        if product in self.surveyed or not self.parse_sure() or not self.sheet_tool():
            return False
        self.surveyed.add(product)
        have = len(self.sheets_of(product)) if self.shelf.sheets else 0
        if have >= SURVEY_SHEETS or self.time_left() < WRAP_UP_SECONDS + 60 or self.cost >= self.cost_soft:
            return False
        questions = max(0, min(self.messages_open() - self.held_back(), UNSTATED_MESSAGES)) if self.roles.get("message") else 0
        # every further product of a bundle still needs its own search, reads and add: more steps stay free for each
        room = self.steps_left() - self.finish_reserve() - questions - SURVEY_SPARE - SURVEY_PER_PRODUCT * max(0, len(self.product_names()) - 1)
        take = max(0, min(SURVEY_SHEETS - have, SURVEY_READS, room))
        if take <= 0:
            return False
        budget, quantity = self.walk_scope()
        several = bool(self.items())
        pool = []
        for key, row in self.shelf.rows.items():
            if row.get("sheet") or key in self.shelf.sheets or key in self.opened or key in self.gone or key in self.model_removed:
                continue
            if row.get("stock") is False or row.get("price") is None:
                continue
            if several and self.product_of(key) != product:
                continue
            if not several and not self.of_kind(row):
                continue
            over = not several and budget is not None and row["price"] * quantity > budget + 1e-9
            pool.append((over, row["price"], key))
        picks = [k for _, _, k in sorted(pool)[:take]]
        if not picks:
            return False
        before = self.step
        got = self.code_reads(picks)
        self.trace.append({"kind": "survey", "product": product, "had": have, "opened": picks[:got], "step": self.step})
        if got:
            lines = [self.sheet_text(k, 500) for k in picks[:got]]
            self.messages.append({"role": "user", "content": f"[note] Before asking the shopper about the {product or 'product'}, the code opened {got} more sheet(s) of it "
                                  f"({self.step - before} store steps; do not open them again):\n" + "\n".join(lines) + "\n" + self.status()})
        return got > 0

    def need_owner(self, needs: list[str], asked: str) -> str:
        """Which product a revealed need belongs to: the one its words name, else the product asked about."""
        for need in needs:
            pw = set(words(need.partition(":")[0]))
            ranked = sorted(((len(self.product_words(p) & pw), p) for p in self.product_names()), key=lambda t: -t[0])
            if ranked and ranked[0][0] > 0:
                return ranked[0][1]
        return asked

    # ------------------------------------------------------------------ the note reader and the re-ranking after a change

    def read_market(self, texts: list[str], name: str) -> None:
        """One cheap pass over the free text of any observation: a line that reports a line gone or a price moved is noted
        (the hearing does the rest); the step of a rise of the limit is kept for the re-ranking."""
        if name in (self.roles.get("search"), self.roles.get("details"), self.roles.get("view"), self.roles.get("cart_view")) or self.listing_step == self.step:
            return
        for text in texts:
            if text in self.market_notes:
                continue
            if self.gone_said(text) and self.cart_ids:
                self.market_notes.append(text[:200])
            elif re.search(_PRICE_WORDS, text) and re.search(r"\d", text) and self.cart_ids and not budget_in(text, str(self.plan.get("currency") or "")):
                self.market_notes.append(text[:200])
                self.market_at = self.step
        del self.market_notes[:-4]

    def measure_value(self, key: str) -> float | None:
        """The figure a listing's own sheet gives for the measure a 'best' rule names, signed so that higher is better."""
        detail = str(self.plan.get("priority_detail") or "")
        if not detail:
            return None
        bare = re.sub(r"\s*\([^)]*\)", "", detail).strip() or detail
        product = set(words(self.plan.get("product") or ""))
        trimmed = " ".join(w for w in words(bare) if w not in product and w not in ("by", "highest", "best", "most", "largest", "longest"))
        text = ""
        for probe in dict.fromkeys([bare, trimmed, detail]):
            text = self.shelf.label_value(key, probe) if probe else ""
            if text:
                break
        found = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        if not found:
            return None
        value = float(found.group(0))
        direction = " ".join(w for w in words(bare) if w not in product)      # the product's own name ("fairy lights") sets no direction
        return -value if re.search(r"\b(?:low|least|less|light|small|short|thin)\w*", direction, re.I) else value

    def rank_by_rule(self, keys: list[str]) -> list[str]:
        """Candidates in the order the request's rule wants: the cheapest first under a lowest-price rule; the best by the
        named measure (checked listings first, price as the tie-breaker) under a 'best' rule; otherwise checked first, then
        cheapest. Deterministic: the id breaks every tie."""
        rule = self.plan.get("priority")
        price = lambda k: (self.shelf.rows.get(k) or {}).get("price") or 1e12
        if rule == "cheapest":
            return sorted(dict.fromkeys(keys), key=lambda k: (price(k), k))
        if rule == "best":
            def best_key(k: str) -> tuple:
                value = self.measure_value(k)
                return (value is None, -(value or 0.0), k not in self.passed, price(k), k)
            return sorted(dict.fromkeys(keys), key=best_key)
        return sorted(dict.fromkeys(keys), key=lambda k: (k not in self.passed, price(k), k))

    def rerank_after_rise(self) -> bool:
        """After the limit rose, under a 'best' rule, the cart line is compared with the opened fits the new limit lets in: when
        a better one by the measure is known (and an item check passed it, or its sheet states every condition), the code swaps
        it in and orders. Once per rise; one product only. Under a lowest-price rule the held line stays."""
        if (self.budget_rose_at <= self.reranked_at or self.plan.get("priority") != "best" or self.items() or len(self.cart_ids) != 1
                or not self.roles.get("cart") or not self.roles.get("order") or self.steps_left() < self.finish_reserve() + 3
                or self.cost >= self.cost_stop or self.time_left() < WRAP_UP_SECONDS + 20 or self.finished):
            return False
        self.reranked_at = self.budget_rose_at
        held = self.cart_ids[0]
        fits = [k for k in self.live_fits() if k != held and k not in self.model_removed]
        if not fits:
            return False
        best = self.rank_by_rule(fits + [held])[0]
        if best == held or self.measure_value(best) is None or (self.measure_value(held) is not None and self.measure_value(best) <= self.measure_value(held)):
            return False
        self.trace.append({"kind": "rerank", "after_rise": self.budget_rose_at, "from": held, "to": best})
        self.code_order(best, f"The limit rose to {self.budget_now}, and by the request's rule ({self.plan.get('priority_detail')}) {best} is now the better fit within it, so it replaces {held}.")
        return True

    def peek_after_add(self) -> str:
        """One cart view right after an add, before the order: a dropped requirement or a new limit the add triggered
        arrives on it. Returns what it carried ('' when nothing changed or no view was made)."""
        found = self.view_call()
        if not (self.add_unread and found and self.peeks < 3 and self.steps_left() >= 6) or self.finished or self.dead:
            return ""
        view = found[0]
        function = self.schemas_full.get(view) or {}
        args = shape_args(function, found[1])
        if lacking(function, args):
            return ""
        self.add_unread = False
        self.peeks += 1
        mark = (self.market_at, len(self.dropped), self.budget_now, len(self.gone), self.price_moved_at, len(self.revealed))
        reply = self.env_call([{"name": view, "args": args}])
        self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": view, "args": args}], "reply": reply, "peek": True})
        if reply is None or self.finished or self.dead:
            return ""
        observation = reply.get("observation")
        self.book_call(view, args, observation)
        self.read_notices(view, observation)
        note = self.note_observation(view, observation)
        heard, _ = self.shopper_lines(view, args, observation)
        meaning = self.interpret(heard, None) if heard else ""
        if mark == (self.market_at, len(self.dropped), self.budget_now, len(self.gone), self.price_moved_at, len(self.revealed)) and not heard:
            return ""
        self.trace.append({"kind": "peek_note", "heard": heard[:3]})
        return ("The cart view after the add carried a note: " + " | ".join(heard[:3]) + (" " + note if note else "") + (" " + meaning if meaning else "")).strip()

    def probe_back(self, note: str) -> bool:
        """The forced finish's probe drew a change (a new limit, a dropped requirement) while the probe itself still fails a
        condition by its own sheet and nothing opened fits: rather than ordering a line known to fail, the model gets the
        change and a few more turns to find the fit (once per episode; run() gives those turns). True when handed back."""
        line = self.probe_line
        if (self.probe_handed or not line or self.cart_ids != [line] or self.items() or self.llm_down or self.cost >= self.cost_stop
                or self.steps_left() < self.finish_reserve() + 5 or self.time_left() < WRAP_UP_SECONDS + 20):
            return False
        budget, quantity = self.walk_scope()
        row = self.shelf.rows.get(line) or {}
        misses = [f"{p}: {v}" for p, v in self.conditions() if self.shelf.says(line, p, v) == "no"]
        if budget is not None and row.get("price") is not None and row["price"] * quantity > budget + 1e-9:
            misses.append(f"price {row['price']:g} x {quantity} over the limit {budget}")
        if not misses or line in self.live_fits():
            return False
        self.probe_handed = True
        self.refused_at = self.step
        self.trace.append({"kind": "forced_finish", "probe_back": line, "misses": misses[:3]})
        self.messages.append({"role": "user", "content": "[note] " + (note or "") + f"\nThe held {line} still fails: " + "; ".join(misses[:3])
                              + ". Search again under the conditions as they stand now, open the candidates, put the best fit in place of it and order."
                              + "\n" + self.status()})
        return True

    def probe_stands(self) -> bool:
        """The cart holds only the probe line, neither a dropped requirement nor a new limit came since it went in, and the
        probe's own records count against it (a contradicted condition, over the limit, gone, or no condition stated at all).
        A probe whose records contradict nothing and state most conditions is ordered: in race 18 champ23 held the right
        listing as its probe and declined over it (a wrong decline scores 0; true no-fit tasks are about 2 of 90)."""
        return (bool(self.probe_line) and self.cart_ids == [self.probe_line] and not self.items()
                and max(self.relaxed_at, self.budget_changed_at, self.price_moved_at) < self.probe_at
                and not self.line_plausible(self.probe_line, self.conditions(), *self.walk_scope()))

    def line_plausible(self, key: str, conds: list[tuple[str, str]], budget: float | None = None, quantity: int = 1) -> bool:
        """Whether a held line is a fair finish by its own records: on offer, not gone, within the limit, failed by no item
        check or fitment code, no condition contradicted, and at least half of the conditions stated outright."""
        row = self.shelf.rows.get(key) or {}
        if key in self.gone or row.get("stock") is False or row.get("price") is None:
            return False
        if budget is not None and row["price"] * max(1, quantity or 1) > budget + 1e-9:
            return False
        if any(k.startswith(key + "|") and k.endswith(f"|{self.ledger_version}") and v for k, v in self.verdicts.items()):
            return False
        if self.fit_targets() and self.fit_state(key) == "fails":
            return False
        vetoed = self.look_vetoed.get(key)
        if vetoed is not None and (not vetoed or any(flat(p) in vetoed or any(flat(p) in v or v in flat(p) for v in vetoed) for p, _ in conds)):
            return False                     # the last look read another value for a condition still in force
        said = [self.shelf.says(key, p, v) for p, v in conds]
        if "no" in said:
            return False
        return said.count("yes") * 2 >= len(said)

    def held_plausible(self) -> bool:
        """Whether the cart as held is a fair finish: one product with its one line plausible, or several products each with
        its own plausible line, nothing gone, the total within the limit; never while a pointed need is still unsettled."""
        if not self.cart_ids or (set(self.cart_ids) & self.gone) or not self.roles.get("order") or not self.parse_sure():
            return False
        if self.open_pointers() or any(p["failed"] and not p["done"] for p in self.pointers):
            return False
        budget, quantity = self.walk_scope()
        if not self.items():
            return len(self.cart_ids) == 1 and self.line_plausible(self.cart_ids[0], self.conditions(), budget, quantity)
        good, bad, missing = self.bundle_state()
        if bad or missing or len(set(self.cart_ids)) < len(self.product_names()) or len(set(self.lines_for.values())) < len(self.product_names()):
            return False
        total = self.bundle_total()
        if budget is not None and (total is None or total > budget + 1e-9):
            return False
        return all(self.line_plausible(k, self.product_conditions(p)) for p, k in self.lines_for.items())

    def swap_failing_line(self) -> None:
        """The code's own finish with one product: when the one held line is not a fair finish by its own records (a
        condition contradicted, over the limit) while an opened listing meets every condition now (often after a requirement
        was dropped at a refused checkout), the fit takes its place before the order: one call with a whole-basket cart tool,
        else the line out and the fit in."""
        order = self.roles.get("order")
        if (not order or not self.roles.get("cart") or self.items() or len(self.cart_ids) != 1 or not self.parse_sure()
                or self.open_pointers()):
            return
        held = self.cart_ids[0]
        budget, quantity = self.walk_scope()
        if held in self.gone or self.line_plausible(held, self.conditions(), budget, quantity):
            return
        fits = self.rank_by_rule([k for k in self.live_fits() if k not in self.gone and k != held])
        if not fits:
            return
        whole = bool(self.basket_form())
        if self.steps_left() < (2 if whole else 3):
            return
        if not whole:
            out_call = self.remove_args(held)
            if not out_call:
                return
            reply = self.env_call([{"name": out_call[0], "args": out_call[1]}])
            self.trace.append({"kind": "forced_finish", "action": {"name": out_call[0], "args": out_call[1]}, "reply": reply, "swap_out": held})
            if reply is None or self.finished or self.dead:
                return
            self.code_booking(out_call[0], out_call[1], reply.get("observation"))
        found = self.add_call_args(fits[0])
        if not found or self.steps_left() < 2:
            return
        self.agent_placed.add(fits[0])
        reply = self.env_call([{"name": found[0], "args": found[1]}])
        self.trace.append({"kind": "forced_finish", "action": {"name": found[0], "args": found[1]}, "reply": reply, "swap_in": fits[0], "for": held})
        if reply is None or self.finished or self.dead:
            return
        self.code_booking(found[0], found[1], reply.get("observation"))
        said = _compact(reply.get("observation"))
        if (isinstance(reply.get("observation"), dict) and reply["observation"].get("error")) or self.gone_said(said):
            self.gone.add(fits[0])
            self.cart_ids = [i for i in self.cart_ids if i != fits[0]]

    def held_rescue(self) -> str | None:
        """A decline proposed while the cart holds a fair finish (held_plausible): the code orders the held cart instead,
        once per cart state; a second decline over the same cart goes through the usual gate."""
        sig = "|".join(sorted(self.cart_ids))
        if not sig or sig in self.rescued or self.finished or self.dead or self.steps_left() < 1 or not self.held_plausible():
            return None
        self.rescued.add(sig)
        order = self.roles["order"]
        self.trace.append({"kind": "hold", "decline_rescued": list(self.cart_ids), "step": self.step})
        lines = "; ".join(f"{k}: " + "; ".join(self.shelf.evidence_lines(k, self.product_conditions(self.product_of(k)) if self.items() else self.conditions()))[:300]
                          for k in self.cart_ids[:4])
        self.messages.append({"role": "user", "content": "[note] A decline was proposed, but the cart already holds lines whose own records contradict no "
                                                         "condition, within the limit and in stock: " + lines + ". A wrong decline scores 0, so the code orders the held cart."})
        call = {"id": f"rescue_order_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(self.neutral_args(order, "Order the cart."))}}
        self.messages.append({"role": "assistant", "content": "I order the held cart now.", "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        for call_id, reply_text in replies or [(call["id"], "Not sent.")]:
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text + "\n" + self.status()})
        if self.finished:
            return "Not run: the code ordered the held cart instead (see above); the episode is finished."
        return ("Not run (no step used): instead of declining, the code sent the order for the held cart; its result is above. If it was held "
                "or refused, apply what it says and order again; decline only if a held line's own record contradicts a condition.")

    def probe_order_hold(self) -> str | None:
        """An order of the line the code only held as a probe, while its own sheet still contradicts a condition (or it is over
        the limit) and no opened listing fits: held twice, with the advice to take it out and decline."""
        line = self.probe_line
        if not line or line not in self.cart_ids or self.items() or self.probe_order_holds >= 2 or self.steps_left() < 3:
            return None
        if line in self.live_fits() and not self.probe_stands():
            return None
        budget, quantity = self.walk_scope()
        row = self.shelf.rows.get(line) or {}
        misses = [f"{p}: {v}" for p, v in self.conditions() if self.shelf.says(line, p, v) == "no"]
        if budget is not None and row.get("price") is not None and row["price"] * quantity > budget + 1e-9:
            misses.append(f"price {row['price']:g} x {quantity} over the limit {budget}")
        if not misses and not self.probe_stands():
            return None
        self.probe_order_holds += 1
        self.trace.append({"kind": "hold", "probe_order": line, "misses": misses[:3]})
        fits = self.live_fits()
        return (f"Not run (no step used): {line} was put in the cart only to hear the store's notes; its own record still contradicts: " + "; ".join(misses[:3]) + ". "
                + (f"These opened listings fit: {', '.join(fits[:3])}: put the best by the request's rule in its place and order it." if fits else
                   "No opened listing meets every condition: take it out of the cart and decline (or open another candidate that could fit)."))

    def rerank_cart(self, why: str) -> bool:
        """After a note changed the conditions or the limit (one product): the opened sheets are ranked again by the request's
        rule; when the best fit is not the held line the code swaps it in and orders. True when the code ordered."""
        if (self.items() or not self.roles.get("cart") or not self.roles.get("order") or self.finished or self.open_pointers()
                or self.steps_left() < 3 or self.time_left() < WRAP_UP_SECONDS):
            return False
        sure = self.unfolded == 0 and (self.messages_used > 0 or not self.roles.get("message") or self.messages_open() <= 0)
        if (sure and self.budget_rose_at >= 0 and self.step - self.budget_rose_at <= 3 and self.budget_before_rise is not None
                and (self.plan.get("priority") != "cheapest" or not (set(self.cart_ids) & set(self.live_fits())))):
            # the limit rose: rows the old limit kept out were never opened; the code opens the best-placed two of them
            budget, quantity = self.walk_scope()
            low = self.budget_before_rise
            newly = [(-(row["price"] or 0), k) for k, row in self.shelf.rows.items()
                     if k not in self.shelf.sheets and k not in self.gone and row.get("stock") is not False and row.get("price") is not None
                     and row["price"] * quantity > low + 1e-9 and (budget is None or row["price"] * quantity <= budget + 1e-9) and self.of_kind(row)]
            if newly:
                self.code_reads([k for _, k in sorted(newly)[:2]])
                if self.finished or self.dead:
                    return True
        fits = [k for k in self.live_fits() if k not in self.model_removed]
        if not fits or not sure:
            return False
        held = self.cart_ids[0] if len(self.cart_ids) == 1 else ""
        best = self.rank_by_rule(fits + ([held] if held in fits else []))[0]
        if held and best == held:
            return False
        self.trace.append({"kind": "rerank_cart", "from": held, "to": best, "why": why[:80]})
        self.code_order(best, why + f" By the request's rule {best} is now the best opened fit within the limit.")
        return True

    def relax_reorder(self) -> bool:
        """After the shopper dropped a requirement (one product): the opened sheets are ranked again under the narrower
        conditions and the request's rule; when the best fit differs from the held line (or nothing is held and a probe
        hold was made, or the held line is the probe and now fits) the code carts it and orders. Once per relaxation."""
        if (self.relaxed_at <= self.relax_done or self.items() or not self.roles.get("cart") or not self.roles.get("order") or self.finished
                or self.steps_left() < self.finish_reserve() + 3 or self.cost >= self.cost_stop or self.time_left() < WRAP_UP_SECONDS + 20):
            return False
        self.relax_done = self.relaxed_at
        if self.open_pointers():
            return False
        fits = [k for k in self.live_fits() if k not in self.model_removed]
        if not fits:
            self.messages.append({"role": "user", "content": "[note] The shopper dropped a requirement. No opened listing yet meets the remaining conditions within the limit by its own sheet: "
                                                             "search once more without the dropped requirement and open the best candidates by the request's rule; do not decline yet.\n" + self.status()})
            return False
        held = self.cart_ids[0] if len(self.cart_ids) == 1 else ""
        best = self.rank_by_rule(fits + ([held] if held in fits else []))[0]
        if held and best == held and held != self.probe_line:
            return False                     # the held line is still the rule's best: the usual order path sends it
        self.trace.append({"kind": "relax_reorder", "from": held, "to": best})
        self.code_order(best, f"The shopper dropped a requirement; under the remaining conditions and the request's rule {best} is the best opened fit within the limit.")
        return True

    def of_kind(self, row: dict) -> bool:
        """Whether a row's group or title shares a word with the requested product (a row with neither counts)."""
        stem = lambda w: w[:-2] if w.endswith("es") and len(w) > 4 else (w[:-1] if w.endswith("s") and len(w) > 3 else w)
        kind = {stem(w) for w in words(self.plan.get("product") or "") if len(w) >= 3}
        label = {stem(w) for w in words(row.get("group") or "") if len(w) >= 3}
        title = {stem(w) for w in words(row.get("title") or "") if len(w) >= 3}
        return not kind or bool(kind & (label | title)) or not (label or title)

    def closest_candidate(self) -> str:
        """The opened listing nearest to the request when none meets everything: in stock, the fewest conditions its own
        sheet contradicts, then the fewest it leaves open, within the limit if any is; the request's rule breaks ties. With
        nothing opened, the best row of the requested kind within the limit."""
        budget, quantity = self.walk_scope()
        conds = self.conditions()
        of_kind = self.of_kind
        opened, rows = [], []
        for key, row in self.shelf.rows.items():
            if key in self.gone or key in self.model_removed or row.get("stock") is False or row.get("price") is None or not of_kind(row):
                continue
            over = budget is not None and row["price"] * quantity > budget + 1e-9
            if key in self.shelf.sheets or row.get("paged"):
                said = [self.shelf.says(key, p, v) for p, v in conds]
                opened.append((over, sum(1 for x in said if x == "no"), sum(1 for x in said if x != "yes"), key))
            elif not over:
                rows.append(key)
        if opened:
            top = min(opened)[:3]
            tier = [k for o, n, u, k in opened if (o, n, u) == top]
            if not top[0]:
                return self.rank_by_rule(tier)[0]
        if rows:
            return sorted(rows, key=lambda k: ((self.shelf.rows.get(k) or {}).get("price") or 1e12, k))[0]
        return self.rank_by_rule(tier)[0] if opened else ""

    def probe_hold(self) -> str | None:
        """Before a decline with an empty cart (one product), the code puts the closest candidate in the cart and lets
        the store answer: a dropped requirement or a new limit usually arrives on the first add. The decline is held
        once; after the notes it may go out only when nothing fits."""
        if (self.probe_holds >= 1 or self.items() or self.cart_ids or not self.roles.get("cart") or self.steps_left() < 4
                or self.time_left() < WRAP_UP_SECONDS or self.cost >= self.cost_stop):
            return None
        target = self.closest_candidate()
        found = self.add_call_args(target) if target else None
        if not found:
            return None
        self.probe_holds += 1
        tool, in_args = found
        self.agent_placed.add(target)
        reply = self.env_call([{"name": tool, "args": in_args}])
        self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": in_args}], "reply": reply, "probe_hold": True})
        if reply is None or self.finished or self.dead:
            return None
        observation = reply.get("observation")
        self.code_booking(tool, in_args, observation)
        self.read_notices(tool, observation)
        note = self.note_observation(tool, observation)
        heard, _ = self.shopper_lines(tool, in_args, observation)
        meaning = self.interpret(heard, None) if heard else ""
        self.probe_line, self.probe_at = target, self.step
        peeked = self.peek_after_add()
        if self.finished or self.dead:
            return "Not run: the episode is over."
        fits = [k for k in self.live_fits() if k not in self.model_removed]
        if not fits and peeked and self.rerank_cart("After the hold, " + peeked):
            if self.finished:
                return "Not run: after the hold the code re-ranked and ordered (see above); the episode is finished."
            return "Not run (no step used): after the hold the code re-ranked and ordered; the result is above."
        if fits and self.steps_left() >= 2:
            pick = self.rank_by_rule(fits)[0]
            self.relax_done = max(self.relax_done, self.relaxed_at)
            self.trace.append({"kind": "probe_order", "target": pick, "held": target})
            self.code_order(pick, "After the hold, " + (peeked or "the store's reply") + f" left {pick} meeting every condition within the limit.")
            if self.finished:
                return "Not run: the code held the closest candidate, read the notes and ordered " + pick + " (see above); the episode is finished."
            return "Not run (no step used): the code held the closest candidate, read the notes and ordered " + pick + "; the result is above."
        if peeked:
            meaning = (meaning + " " + peeked).strip()
        row = self.shelf.rows.get(target) or {}
        misses = [f"{p}: {v}" for p, v in self.conditions() if self.shelf.says(target, p, v) != "yes"]
        self.trace.append({"kind": "probe_hold", "target": target, "misses": misses[:4], "heard": heard[:3]})
        return ("Not run (no step used): before any decline the code put the closest candidate " + target + f" ({row.get('price', '?')}) in the cart; "
                + ("it does not state: " + "; ".join(misses[:4]) + ". " if misses else "")
                + "The store's reply: " + _compact(observation)[:500] + (" " + note if note else "") + (" " + meaning if meaning else "")
                + " Apply every change it carries (a dropped requirement, a new limit). If a listing now meets every remaining condition within the limit, "
                "put the best one by the request's rule in the cart and order it; decline only if every candidate still fails a condition by its own records.")

    # ------------------------------------------------------------------ the bundle tracker (one cart line per requested product)

    def track_lines(self) -> None:
        """Which cart line stands for which requested product: by the words the product's name shares with the line's title or
        store label, else the first product without a line (in the request's order)."""
        if not self.items():
            return
        products = self.product_names()
        self.lines_for = {p: k for p, k in self.lines_for.items() if k in self.cart_ids}
        for key in self.cart_ids:
            if key in self.lines_for.values():
                continue
            owner = self.product_of(key, self.cart_lines.get(key) or "")
            if not owner:
                free = [p for p in products if p not in self.lines_for]
                if not free:
                    continue
                owner = free[0]
            self.lines_for[owner] = key

    def line_fits(self, key: str, product: str) -> bool:
        """Whether a held line still fits its product: on offer, not gone, within the limit on its own, and its own sheet
        contradicts no condition of that product (what the sheet never states is left to the checks)."""
        row = self.shelf.rows.get(key) or {}
        if key in self.gone or row.get("stock") is False or row.get("price") is None:
            return False
        if any(k.startswith(key + "|") and k.endswith(f"|{self.ledger_version}") and v for k, v in self.verdicts.items()):
            return False
        if self.fit_targets() and self.fit_state(key) == "fails":
            return False
        return all(self.shelf.says(key, p, v) != "no" for p, v in self.product_conditions(product))

    def bundle_total(self) -> float | None:
        items = dict(self.items())
        total = 0.0
        for product, key in self.lines_for.items():
            price = (self.shelf.rows.get(key) or {}).get("price")
            if price is None:
                return None
            total += price * items.get(product, 1)
        return total

    def bundle_state(self) -> tuple[list[str], list[str], list[str]]:
        """Products with a fitting line, products whose line no longer fits, products without a line."""
        self.track_lines()
        good, bad, missing = [], [], []
        for product in self.product_names():
            key = self.lines_for.get(product)
            if not key:
                missing.append(product)
            elif self.line_fits(key, product):
                good.append(product)
            else:
                bad.append(product)
        return good, bad, missing

    def bundle_note(self) -> str:
        if not self.items():
            return ""
        good, bad, missing = self.bundle_state()
        by = max(6, round(self.max_steps * BUNDLE_BY))
        parts = [f"[bundle] one cart line per product, all in by step {by}."
                 + (" The cart tool sets the WHOLE basket: every call lists every line you keep, with its count." if self.basket_form() else "")]
        if good:
            parts.append("holding a fit for: " + ", ".join(f"{p} ({self.lines_for[p]})" for p in good) + ".")
        if bad:
            parts.append("held but no longer fitting (swap it): " + ", ".join(f"{p} ({self.lines_for[p]})" for p in bad) + ".")
        if missing:
            parts.append("no line yet for: " + ", ".join(missing) + ".")
        budget, _ = self.walk_scope()
        total = self.bundle_total()
        if total is not None and budget is not None:
            parts.append(f"cart total {total:g} of {budget:g}.")
        return " ".join(parts)

    def product_fit(self, product: str, below: float | None = None, leaving: str = "", loose: bool = False) -> str:
        """An opened listing of one product, not in the cart, whose own sheet states every condition of that product and that
        is on offer within what the limit leaves; the request's rule ranks the candidates. `below`: only listings cheaper than
        this price; `leaving`: a held line about to be swapped out, whose price the limit no longer has to cover."""
        budget, _ = self.walk_scope()
        quantity = dict(self.items()).get(product, 1)
        total = self.bundle_total() or 0.0
        if leaving and (self.shelf.rows.get(leaving) or {}).get("price") is not None:
            total -= self.shelf.rows[leaving]["price"] * quantity
        conds = self.product_conditions(product)
        pool = []
        for key in self.sheets_of(product):
            row = self.shelf.rows.get(key) or {}
            if key in self.cart_ids or key in self.gone or (key in self.model_removed and not loose) or row.get("stock") is False or row.get("price") is None:
                continue                     # the code's own finish (loose) may take back a line the model took out
            if product != self.main_product() and self.fit_blocked(key):
                continue
            if below is not None and row["price"] >= below - 1e-9:
                continue
            if budget is not None and total + row["price"] * quantity > budget + 1e-9:
                continue
            if conds and not all(self.shelf.says(key, p, v) == "yes" for p, v in conds):
                continue
            pool.append(key)
        ranked = self.rank_by_rule(pool)
        return ranked[0] if ranked else ""

    def complete_bundle(self) -> bool:
        """From BUNDLE_BY of the cap, with several products and no order out: a product without a line gets the fit the code
        knows for it (one step each, while steps allow); then, when every held line fits and the lines are within the limit,
        the code sends the order. Once per episode. A cart with a line that no longer fits is left to the model, with a note."""
        if self.bundle_done or not self.items():
            return False
        # the mark leaves the steps the filling and the order take (one per product still without a line, the order and one
        # re-read), so the order itself is out by BUNDLE_BY of the cap
        lacking_lines = len(self.bundle_state()[2])
        if (self.order_attempts or not self.roles.get("cart") or not self.roles.get("order")
                or self.step < max(6, round(self.max_steps * BUNDLE_BY)) or self.steps_left() < self.finish_reserve() + 1
                or self.cost >= self.cost_stop or self.time_left() < WRAP_UP_SECONDS + 20 or self.finished):
            return False
        self.bundle_done = True
        try:
            added = self.fill_bundle()
            if self.finished or self.dead:
                return True
            good, bad, missing = self.bundle_state()
            if not self.cart_ids or bad or not good:
                self.messages.append({"role": "user", "content": f"[bundle] step {self.step} of {self.max_steps}: " + self.bundle_note()
                                      + (" The code added " + "; ".join(added) + "." if added else "")
                                      + " Fix the lines that no longer fit (or add the missing ones) and order; decline only if a product has no fitting listing in the store.\n" + self.status()})
                return bool(added)
            budget, _ = self.walk_scope()
            total = self.bundle_total()
            if total is not None and budget is not None and total > budget + 1e-9:
                self.messages.append({"role": "user", "content": f"[bundle] the held lines total {total:g}, over the limit {budget:g}: swap the dearest line for a cheaper fit, then order.\n" + self.status()})
                return bool(added)
            why = ("Step " + str(self.step) + f" of {self.max_steps}: " + ("every requested product has a fitting line" if not missing else
                   "no fitting listing is known for " + ", ".join(missing) + ", and every held line fits") + (" (the code added " + "; ".join(added) + ")" if added else "") + ".")
            self.messages.append({"role": "user", "content": "[bundle] " + why + " The order goes out now through the usual checks."})
            order = self.roles["order"]
            call = {"id": f"bundle_order_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(self.neutral_args(order, "Order the cart."))}}
            self.messages.append({"role": "assistant", "content": "I order the cart now.", "tool_calls": [call]})
            replies = self.run_calls([call], planned=True)
            for call_id, reply_text in replies or [(call["id"], "Not sent.")]:
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text + "\n" + self.status()})
            self.trace.append({"kind": "bundle_order", "lines": dict(self.lines_for), "ordered": self.finished})
            return True
        except Exception as exc:
            self.trace.append({"kind": "bundle", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return False

    def closest_line(self, product: str) -> str:
        """For the code's own finish when no opened listing of a product states every condition: the opened, in-stock listing
        of that product within what the limit leaves whose sheet states the most conditions and contradicts the fewest
        (never one the fitment codes rule out). '' when none of that product was opened."""
        budget, _ = self.walk_scope()
        quantity = dict(self.items()).get(product, 1)
        total = self.bundle_total() or 0.0
        conds = self.product_conditions(product)
        targets = self.fit_targets()
        scored = []
        for key in self.sheets_of(product):
            row = self.shelf.rows.get(key) or {}
            if key in self.cart_ids or key in self.gone or row.get("stock") is False or row.get("price") is None:
                continue
            if (product != self.main_product() or not self.items()) and (self.fit_blocked(key) or (targets and self.fit_state(key, targets) == "unread")):
                continue                     # under fitment codes a close line must have its page read and not ruled out
            if (self.items() and self.product_of(key) != product) or (not self.items() and not self.of_kind(row)):
                continue
            if not self.items() and budget is not None:
                total, quantity = 0.0, self.walk_scope()[1]
            if budget is not None and total + row["price"] * quantity > budget + 1e-9:
                continue
            said = [self.shelf.says(key, p, v) for p, v in conds]
            if "no" in said:
                continue                     # a stated miss is never a close line
            scored.append((-said.count("yes"), row["price"], key))
        examined = [k for k in self.sheets_of(product) if (self.product_of(k) == product if self.items() else self.of_kind(self.shelf.rows.get(k) or {}))]
        if not scored and not examined and not targets and self.parse_sure():
            # no listing of the product was opened at all: a result row of it (in stock, within what the limit leaves, its
            # title contradicting nothing) stands in, the most conditions its title states first, then the cheapest. A
            # finish with a guessed line can still pass; a decline over a product never opened passes only on a true no-fit
            for key, row in self.shelf.rows.items():
                if key in self.shelf.sheets or key in self.cart_ids or key in self.gone or key in self.model_removed:
                    continue
                if row.get("stock") is False or row.get("price") is None or self.fit_blocked(key):
                    continue
                if (self.items() and self.product_of(key) != product) or (not self.items() and not self.of_kind(row)):
                    continue
                quantity = dict(self.items()).get(product, 1) if self.items() else self.walk_scope()[1]
                base = (self.bundle_total() or 0.0) if self.items() else 0.0
                if budget is not None and base + row["price"] * quantity > budget + 1e-9:
                    continue
                said = [self.shelf.says(key, p, v) for p, v in conds]
                if "no" in said:
                    continue
                scored.append((-said.count("yes"), row["price"], key))
        scored.sort()
        return scored[0][2] if scored else ""

    def fill_bundle(self, loose: bool = False) -> list[str]:
        """A product without a cart line gets the fit the code knows for it (one step each, while steps allow; one step for
        all of them with a whole-basket cart). `loose` (the code's own finish only): a product with no known fit gets its
        closest opened listing. The lines added."""
        added: list[str] = []
        good, bad, missing = self.bundle_state()
        if self.basket_form() and missing:
            picks = {}
            main = self.main_product()
            held = list(self.cart_ids)
            try:
                for product in sorted(missing, key=lambda p: p != main):
                    fit = self.product_fit(product, loose=loose) or (self.closest_line(product) if loose else "")
                    if fit and fit not in picks.values():
                        picks[product] = fit
                        if product == main:
                            self.cart_ids = held + [fit]     # the others are chosen to fit the main pick
            finally:
                self.cart_ids = held
            if not picks or self.steps_left() < 2 or self.finished or self.dead:
                return added
            keep = [(k, self.basket_counts.get(k, self.line_count(k))) for k in self.cart_ids]
            found = self.basket_call(keep + [(fit, dict(self.items()).get(product, 1)) for product, fit in picks.items()])
            if not found:
                return added
            tool, args = found
            self.agent_placed.update(picks.values())
            reply = self.env_call([{"name": tool, "args": args}])
            self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": args}], "reply": reply, "bundle": True})
            if reply is None or self.finished or self.dead:
                return added
            observation = reply.get("observation")
            self.code_booking(tool, args, observation)
            self.read_notices(tool, observation)
            self.note_observation(tool, observation)
            for product, fit in picks.items():
                if fit in self.gone:
                    continue
                self.lines_for[product] = fit
                added.append(f"{product}: {fit}")
            return added
        for product in missing:
            if self.steps_left() < self.finish_reserve() + 2 or self.finished or self.dead:
                break
            fit = self.product_fit(product, loose=loose) or (self.closest_line(product) if loose else "")
            found = self.add_call_args(fit) if fit else None
            if not found:
                self.trace.append({"kind": "bundle_unfilled", "product": product, "fit": fit, "sheets": self.sheets_of(product)[:6], "conds": self.product_conditions(product)})
                continue
            tool, args = found
            self.agent_placed.add(fit)
            reply = self.env_call([{"name": tool, "args": args}])
            self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": args}], "reply": reply, "bundle": True})
            if reply is None or self.finished or self.dead:
                break
            observation = reply.get("observation")
            self.code_booking(tool, args, observation)
            self.read_notices(tool, observation)
            self.note_observation(tool, observation)
            said = _compact(observation)
            if (isinstance(observation, dict) and observation.get("error")) or self.gone_said(said) or fit in self.gone:
                self.gone.add(fit)
                self.cart_ids = [i for i in self.cart_ids if i != fit]
                continue
            self.lines_for[product] = fit
            added.append(f"{product}: {fit}")
        return added

    def reread_lines(self, keys: list[str], keep: int | None = None) -> int:
        """The code re-opens held lines whose record predates the last announced change (one step each): the current stock
        and price land on the shelf. Steps spent."""
        tool = self.roles.get("view") or self.roles.get("details") or self.roles.get("stock")
        spent = 0
        for key in keys:
            row = self.shelf.rows.get(key) or {}
            if not tool or row.get("step", 0) > self.market_at or self.steps_left() < (self.finish_reserve() + 3 if keep is None else keep) or self.finished or self.dead:
                continue
            args = self.id_args(tool, key)
            if not args:
                continue
            reply = self.env_call([{"name": tool, "args": args}])
            self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": args}], "reply": reply, "reread": True})
            if reply is None or self.finished or self.dead:
                break
            spent += 1
            observation = reply.get("observation")
            self.book_call(tool, args, observation)
            self.read_notices(tool, observation)
            self.note_observation(tool, observation)
            self.freshened.add(key)
            if (self.shelf.rows.get(key) or {}).get("stock") is False:
                self.gone.add(key)
        return spent

    def send_bundle_order(self, why: str, call_id: str) -> None:
        """The code's own order of the held lines, through the usual holds, with its reason on the transcript."""
        order = self.roles["order"]
        self.messages.append({"role": "user", "content": "[bundle] " + why + " The order goes out now through the usual checks; if it is held or refused, fix the cart and order again."})
        call = {"id": f"{call_id}_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(self.neutral_args(order, "Order the cart."))}}
        self.messages.append({"role": "assistant", "content": "I order the cart now.", "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        for cid, reply_text in replies or [(call["id"], "Not sent.")]:
            self.messages.append({"role": "tool", "tool_call_id": cid, "content": reply_text + "\n" + self.status()})

    def bundle_recover(self) -> bool:
        """With several products in the cart, after a line was reported sold out: when the report named no line, the stale lines
        are re-read and the one shown out of stock is the dead one; the dead line comes out, the fit the code knows for its
        product goes in, and when every product then holds a fitting line the order goes out again. A bundle order never
        leaves with a product missing. Once per stockout report. True when a turn was spent."""
        if (not self.items() or not self.cart_ids or not self.roles.get("cart") or not self.roles.get("order") or self.finished
                or self.steps_left() < self.finish_reserve() + 3 or self.cost >= self.cost_stop or self.time_left() < WRAP_UP_SECONDS + 20):
            return False
        dead = [k for k in self.cart_ids if k in self.gone]
        unknown = self.dead_unknown > self.bundle_recovered_at
        if not dead and not unknown:
            return False
        if dead and self.market_at <= self.bundle_recovered_at and not unknown:
            return False
        self.bundle_recovered_at = max(self.market_at, self.dead_unknown)
        spent = 0
        try:
            if not dead:
                spent = self.reread_lines(list(self.cart_ids))
                if self.finished or self.dead:
                    return True
                dead = [k for k in self.cart_ids if k in self.gone or (self.shelf.rows.get(k) or {}).get("stock") is False]
                self.gone.update(dead)
                if not dead:
                    self.messages.append({"role": "user", "content": (f"[recovery by the code: {spent} store step(s)] a line was reported sold out, but no held line shows out of stock on "
                                                                     "re-reading. Check the cart view; take out any line the store will not sell and put the cheapest fit for that product in, then order.\n" + self.status())})
                    self.trace.append({"kind": "bundle_recover", "reread": spent, "dead": []})
                    return spent > 0
            self.track_lines()
            swapped, failed = [], []
            for key in dead:
                if self.steps_left() < self.finish_reserve() + 2:
                    break
                product = next((p for p, k in self.lines_for.items() if k == key), "")
                found = self.remove_args(key)
                if not found:
                    break
                tool, out_args = found
                reply = self.env_call([{"name": tool, "args": out_args}])
                self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": out_args}], "reply": reply, "bundle_recover": True})
                if reply is None or self.finished or self.dead:
                    return True
                self.code_booking(tool, out_args, reply.get("observation"))
                self.read_notices(tool, reply.get("observation"))
                spent += 1
                fit = self.product_fit(product) if product else ""
                added = self.add_call_args(fit) if fit else None
                if not added or self.steps_left() < self.finish_reserve() + 2:
                    self.trace.append({"kind": "bundle_add_failed", "product": product, "fit": fit})
                    failed.append(product or key)
                    continue
                tool, in_args = added
                self.agent_placed.add(fit)
                reply = self.env_call([{"name": tool, "args": in_args}])
                self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": in_args}], "reply": reply, "bundle_recover": True})
                if reply is None or self.finished or self.dead:
                    return True
                observation = reply.get("observation")
                self.code_booking(tool, in_args, observation)
                self.read_notices(tool, observation)
                self.note_observation(tool, observation)
                spent += 1
                said = _compact(observation)
                if (isinstance(observation, dict) and observation.get("error")) or self.gone_said(said) or fit in self.gone:
                    self.gone.add(fit)
                    self.cart_ids = [i for i in self.cart_ids if i != fit]
                    failed.append(product or key)
                    continue
                self.lines_for[product] = fit
                swapped.append(f"{product}: {key} (sold out) -> {fit} ({(self.shelf.rows.get(fit) or {}).get('price', '?')})")
            good, bad, missing = self.bundle_state()
            budget, _ = self.walk_scope()
            total = self.bundle_total()
            over = total is not None and budget is not None and total > budget + 1e-9
            self.trace.append({"kind": "bundle_recover", "steps": spent, "dead": dead, "swapped": swapped, "failed": failed, "missing": missing, "bad": bad})
            head = f"[recovery by the code: {spent} store step(s)] the store reported a held line sold out: " + ("; ".join(swapped) if swapped else ", ".join(dead) + " taken out") + "."
            if swapped and not bad and not missing and not failed and not over:
                self.send_bundle_order(head + " Every requested product holds a fitting line again.", "recover_order")
                return True
            self.messages.append({"role": "user", "content": head + (" No opened listing fits for: " + ", ".join(failed) + "." if failed else "")
                                  + " " + self.bundle_note() + " Open the cheapest unread listings of the missing product and put a fit in; a bundle order never goes out with a product "
                                  "missing; decline only if no listing of that product in the store fits.\n" + self.status()})
            return True
        except Exception as exc:
            self.trace.append({"kind": "bundle_recover", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return spent > 0

    def bundle_reprice(self) -> bool:
        """After a price was reported moved with several products in the cart: the stale lines are re-read (one step each) and,
        under a lowest-price rule, a line now dearer than a cheaper fit of its product is swapped out. Once per report."""
        if (not self.items() or not self.cart_ids or self.price_moved_at <= self.repriced_at or not self.roles.get("cart") or self.finished
                or self.steps_left() < self.finish_reserve() + 3 or self.cost >= self.cost_stop or self.time_left() < WRAP_UP_SECONDS + 20):
            return False
        self.repriced_at = self.price_moved_at
        try:
            spent = self.reread_lines(list(self.cart_ids))
            if self.finished or self.dead:
                return True
            note = self.bundle_cheapest()
            if spent or note:
                self.messages.append({"role": "user", "content": f"[price change: {spent} store step(s) re-read the held lines] " + (note or "the held lines were judged again at their current prices.")
                                      + " " + self.bundle_note() + "\n" + self.status()})
            self.trace.append({"kind": "bundle_reprice", "reread": spent, "swapped": bool(note)})
            return bool(spent or note)
        except Exception as exc:
            self.trace.append({"kind": "bundle_reprice", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return False

    def bundle_cheapest(self) -> str:
        """Under a lowest-price rule with several products: each held line is compared with the opened listings of its product,
        and a cheaper one whose own sheet states every condition of that product, in stock and within the limit, takes the
        line's place (two steps per swap). Once per cart state. The note for the transcript, or ''."""
        if self.plan.get("priority") != "cheapest" or not self.items() or not self.roles.get("cart") or not self.cart_ids:
            return ""
        state = self.cart_signature()
        if state in self.bundle_swapped:
            return ""
        self.bundle_swapped.add(state)
        self.track_lines()
        notes = []
        for product, key in list(self.lines_for.items()):
            if self.steps_left() < self.finish_reserve() + 3 or self.time_left() < WRAP_UP_SECONDS + 10 or self.finished or self.dead:
                break
            price = (self.shelf.rows.get(key) or {}).get("price")
            if price is None or key in self.model_removed:
                continue
            better = self.product_fit(product, below=price, leaving=key)
            if not better or better in self.vetoed or self.set_aside(better):
                continue
            if self.model_vetoes(better, self.product_conditions(product)):
                self.vetoed.add(better)
                continue
            found = self.remove_args(key)
            added = self.add_call_args(better)
            if not found or not added:
                continue
            tool, out_args = found
            reply = self.env_call([{"name": tool, "args": out_args}])
            self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": out_args}], "reply": reply, "bundle_swap": True})
            if reply is None or self.finished or self.dead:
                break
            self.code_booking(tool, out_args, reply.get("observation"))
            self.read_notices(tool, reply.get("observation"))
            tool, in_args = added
            self.agent_placed.add(better)
            reply = self.env_call([{"name": tool, "args": in_args}])
            self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": tool, "args": in_args}], "reply": reply, "bundle_swap": True})
            if reply is None or self.finished or self.dead:
                break
            observation = reply.get("observation")
            self.code_booking(tool, in_args, observation)
            self.read_notices(tool, observation)
            said = _compact(observation)
            if (isinstance(observation, dict) and observation.get("error")) or self.gone_said(said) or better in self.gone:
                self.gone.add(better)
                self.cart_ids = [i for i in self.cart_ids if i != better]
                back = self.add_call_args(key)
                if back and key not in self.gone and self.steps_left() >= self.finish_reserve() + 1:
                    again = self.env_call([{"name": back[0], "args": back[1]}])
                    self.trace.append({"kind": "env", "step": self.step, "actions": [{"name": back[0], "args": back[1]}], "reply": again, "bundle_swap": True})
                    if again is not None:
                        self.code_booking(back[0], back[1], again.get("observation"))
                continue
            self.lines_for[product] = better
            notes.append(f"{product}: {key} ({price:g}) -> {better} ({self.shelf.rows[better]['price']:g}), cheaper and its own sheet states every condition of that product")
        if not notes:
            return ""
        self.trace.append({"kind": "bundle_swap", "swaps": notes})
        return "[cheapest rule applied by the code per product: " + "; ".join(notes) + ".]"

    # ------------------------------------------------------------------ what the model sees: a digest and the last turns

    def sheet_text(self, key: str, limit: int) -> str:
        """One listing as the code holds it: price, stock, title, category and its full property sheet (merged labels)."""
        row = self.shelf.rows.get(key) or {}
        stock = {True: "in stock", False: "OUT OF STOCK"}.get(row.get("stock"), "stock ?")
        head = f"{key} | {row['price']:g} | {stock}" if row.get("price") is not None else f"{key} | price ? | {stock}"
        if row.get("group"):
            head += f" | {row['group']}"
        if row.get("title"):
            head += f" | {str(row['title'])[:90]}"
        sheet = self.shelf.sheets.get(key) or {}
        if not sheet:
            return head + " | no property sheet opened"
        body = "; ".join(f"{k}: {str(v)[:120]}" for k, v in sheet.items())
        if len(body) > limit:
            body = body[:limit] + " ..."
        return head + "\n  sheet: " + body

    def digest_sheets(self) -> list[str]:
        """The listings whose sheets the model must keep seeing: the cart lines, the fits the checks passed, then the cheapest
        opened listings on offer within the limit."""
        budget, quantity = self.walk_scope()
        order: list[str] = [k for k in self.cart_ids if k in self.shelf.rows]
        order += [k for k in sorted(self.passed) if k not in order and k not in self.gone]
        rest = [k for k, r in self.shelf.rows.items() if k in self.opened and r.get("sheet") and k not in order and k not in self.gone
                and r.get("stock") is not False and r.get("price") is not None
                and (budget is None or r["price"] * max(1, quantity) <= budget + 1e-9)]
        rest.sort(key=lambda k: (self.shelf.rows[k]["price"], k))
        order += rest
        return [self.sheet_text(k, DIGEST_SHEET_CHARS) for k in order[:DIGEST_SHEETS]]

    def digest(self, upto: int) -> str:
        """The code's summary of the turns the window no longer shows in full: the searches, the sheets opened, the exchanges
        with the shopper, the cart moves and the latest holds. The status under the newest result carries the rest."""
        parts = [f"[earlier turns, summarized by the code: {max(0, upto - 2)} transcript entries are not shown in full]"]
        if self.searches:
            shown = [f"{_compact(s['args'])[:120]} -> {s.get('total') if s.get('total') is not None else len(s.get('shown') or [])} hits" for s in self.searches[-8:]]
            parts.append("Searches made: " + " | ".join(shown))
        opened = sorted(k for k in self.opened if k in self.shelf.ids)
        if opened:
            parts.append(f"Listings opened ({len(opened)}): " + ", ".join(opened[:40]))
        if self.said:
            parts.append("Exchanges with the shopper:\n" + "\n".join(s[:300] for s in self.said[-8:]))
        if self.cart_ids:
            parts.append("Cart lines now: " + ", ".join(f"{k} ({(self.shelf.rows.get(k) or {}).get('price', '?')})" for k in self.cart_ids))
        elif self.cart_used:
            parts.append("The cart is empty now.")
        if self.gone:
            parts.append("Reported sold out (never order these): " + ", ".join(sorted(self.gone)))
        if self.hold_log:
            parts.append("Latest calls the code held (no step used):\n" + "\n".join("- " + h for h in self.hold_log[-3:]))
        if self.order_attempts:
            parts.append(f"Order attempts so far: {self.order_attempts}; refused: {self.refusals}. Nothing is bought until the episode ends.")
        sheets = self.digest_sheets()
        if sheets:
            parts.append("Property sheets as opened (the cart lines, the fits the checks passed, then the cheapest opened listings within the limit; "
                         "judge these listings from these records, not from memory):\n" + "\n".join(sheets))
        return "\n".join(parts)[:4000 + DIGEST_SHEETS * (DIGEST_SHEET_CHARS + 300)]

    def window(self) -> list[dict[str, Any]]:
        """What the model sees on a tool turn: the playbook, the opening, then the WHOLE episode as it happened (the store's
        own results, the code's notes), each entry without the status it carried at its time, and the current status as the
        last message. Only when the transcript passes HISTORY_CHARS are the oldest results cut to their head (with the ids
        they showed) until the rest fits in HISTORY_KEEP; that cut moves in jumps, so the entries before it stay byte-stable
        from turn to turn and the prompt cache keeps serving them. Past the soft line both sizes shrink."""
        msgs = self.messages
        if len(msgs) <= 2:
            return msgs
        limit, keep = (HISTORY_CHARS, HISTORY_KEEP) if not self.past_soft() else (int(HISTORY_CHARS * 0.65), int(HISTORY_KEEP * 0.65))
        body: list[dict[str, Any]] = []
        for m in msgs[2:]:
            content = m.get("content")
            if isinstance(content, str) and "[status]" in content:
                head = content.split("\n[status]")[0] if "\n[status]" in content else content.split("[status]")[0]
                m = {**m, "content": head.rstrip() or "(see the status below)"}
            elif m.get("role") == "assistant" and not content:
                m = {**m, "content": "(tool calls)"}
            body.append(m)
        sizes = [len(str(m.get("content") or "")) + len(_compact(m.get("tool_calls") or [])) for m in body]
        if sum(sizes) - sum(sizes[: self.compact_upto]) + sum(min(n, OLD_HEAD_CHARS + 200) for n in sizes[: self.compact_upto]) > limit:
            # the cut jumps forward to a turn boundary, far enough that the rest fits in `keep`
            total, upto = sum(sizes), 0
            while upto < len(body) - 1 and total > keep:
                total -= max(0, sizes[upto] - (OLD_HEAD_CHARS + 200))
                upto += 1
            while upto < len(body) and body[upto].get("role") != "assistant":
                upto += 1
            self.compact_upto = max(self.compact_upto, min(upto, len(body)))
        if self.compact_upto:
            ids = [k for k in self.shelf.rows]
            for i in range(min(self.compact_upto, len(body))):
                m = body[i]
                content = str(m.get("content") or "")
                if len(content) <= OLD_HEAD_CHARS + 200:
                    continue
                if i not in self.cut_cache:
                    # cut once and kept: the same entry reads the same on every later turn
                    if m.get("role") == "assistant":
                        self.cut_cache[i] = content[:OLD_HEAD_CHARS] + " ...(cut)"
                    else:
                        named = [k for k in ids if k in content][:30]
                        self.cut_cache[i] = content[:OLD_HEAD_CHARS] + " ...(an older result, cut" + ("; listings it showed: " + ", ".join(named) if named else "") + ")"
                body[i] = {**m, "content": self.cut_cache[i]}
        tail = self.status()
        if self.compact_upto:
            tail = self.digest(self.compact_upto) + "\n\n" + tail
        return msgs[:2] + body + [{"role": "user", "content": tail}]

    def bundle_decline_gate(self) -> str | None:
        """With several products, a decline never goes out while the cart holds lines that fit: a product without a line first
        gets the fit the code knows for it; when every product then holds a fitting line within the limit, the first attempt is
        held with the state of the bundle and the next one makes the code order it. A product no opened listing fits leaves
        the decline to the usual audit (never a partial order)."""
        if not self.items() or not self.cart_ids:
            return None
        good, bad, missing = self.bundle_state()
        if bad or not good or self.steps_left() < 2:
            return None
        if missing and self.steps_left() >= self.finish_reserve() + 2 and self.cost < self.cost_stop:
            added = self.fill_bundle()
            if self.finished or self.dead:
                return "Not run: the episode is over."
            good, bad, missing = self.bundle_state()
            if added:
                self.trace.append({"kind": "hold", "bundle_filled_before_decline": added})
        if missing:
            self.decline_blocks += 1
            if self.decline_blocks > DECLINE_BLOCKS:
                return None
            return ("Not run (no step used): the cart holds fitting lines for some products, so a decline is wrong unless a product has no fitting listing in the store. "
                    + self.bundle_note() + " Open the cheapest unread listings of the missing product and put a fit in, then order the complete cart; a decline is right "
                    "only when every listing of that product fails a condition by its own records (then say so in the decline).")
        budget, _ = self.walk_scope()
        total = self.bundle_total()
        if total is not None and budget is not None and total > budget + 1e-9:
            return None
        self.decline_blocks += 1
        if self.decline_blocks >= DECLINE_BLOCKS or self.steps_left() <= 4:
            order = self.roles.get("order")
            if order:
                self.trace.append({"kind": "hold", "bundle_decline_overruled": dict(self.lines_for)})
                call = {"id": f"bundle_order_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(self.neutral_args(order, "Order the cart."))}}
                self.messages.append({"role": "assistant", "content": "I order the held lines instead of declining.", "tool_calls": [call]})
                replies = self.run_calls([call], planned=True)
                for call_id, reply_text in replies or [(call["id"], "Not sent.")]:
                    self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text + "\n" + self.status()})
                if self.finished:
                    return "Not run: the code ordered the held lines instead (see above); the episode is finished."
                return "Not run (no step used): instead of declining, the code sent the order for the held lines; its result is above. If it was refused, fix the cart and order again."
        return ("Not run (no step used): a decline is wrong while the cart holds lines that fit their products. " + self.bundle_note()
                + " Order the held lines (add a fit for any product still missing when one is known); a decline can go out only when a requested product has no fitting listing in the store.")

    # ------------------------------------------------------------------ step savers: titles to ids, the read cache, gone ids

    def resolve_titles(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A listing named by its title where the tool wants its id is mapped to the id the shelf knows for that title."""
        if name not in {self.roles.get(r) for r in ("details", "view", "stock", "cart", "cart_remove", "order")} - {None}:
            return args
        properties = self.schemas[name].get("properties") or {}
        out = dict(args)
        for key, value in args.items():
            spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
            if not isinstance(value, str) or "enum" in spec or value.strip() in self.shelf.ids or len(words(value)) < 2:
                continue
            want = set(words(value))
            hits = [k for k, r in sorted(self.shelf.rows.items()) if r.get("title") and (flat(r["title"]) == flat(value) or want <= set(words(r["title"])))]
            if len(hits) == 1:
                out[key] = hits[0]
                self.trace.append({"kind": "resolved_title", "from": value[:80], "to": hits[0]})
        return out

    def remove_args(self, target: str) -> tuple[str, dict[str, Any]] | None:
        """The tool and arguments that take one line out: the cart tool's removing mode, else the separate remove tool."""
        if self.basket_form():
            return self.basket_call([(k, self.basket_counts.get(k, self.line_count(k))) for k in self.cart_ids if k != target])
        out_args = self.cart_args(False, target)
        if out_args and self.roles.get("cart"):
            return self.roles["cart"], out_args
        remover = self.roles.get("cart_remove")
        if remover and remover != self.roles.get("cart"):
            args = self.id_args(remover, target)
            if args:
                return remover, args
        return None

    def read_hold(self, name: str, args: dict[str, Any]) -> str | None:
        """A sheet opened by this same tool since the last announced change is served from the shelf (no step); a listing reported
        gone is not re-opened at all."""
        if name not in (self.roles.get("details"), self.roles.get("view"), self.roles.get("stock")):
            return None
        target = next((v.strip() for v in args.values() if isinstance(v, str) and v.strip() in self.shelf.ids), "")
        if not target:
            pointed = [v.strip() for v in args.values() if isinstance(v, str) and v.strip()]
            if (pointed and len(pointed[0]) >= 4 and " " not in pointed[0] and any(c.isdigit() for c in pointed[0])
                    and re.search(r"(?<![\w-])" + re.escape(pointed[0]) + r"(?![\w-])", self.raw_seen)):
                # the store wrote this id but the code's reader never found it: the model's reading wins
                self.raw_misses += 1
                self.trace.append({"kind": "raw_miss", "id": pointed[0][:60]})
                return None
            if pointed and (" " in pointed[0] or not any(c.isdigit() for c in pointed[0])):
                # a name or words, not a listing id: the store answers such a lookup with an empty record
                return (f"Not run (no step used): '{pointed[0][:40]}' is not a listing id. Search (or filter by section) first and open listings "
                        "by the ids the results show.")
            if pointed and len(self.shelf.ids) >= 3 and f"unknown|{pointed[0]}" not in self.keep_held:
                # no result ever showed this id: the call would most likely spend a step on an error
                self.keep_held.add(f"unknown|{pointed[0]}")
                return (f"Not run (no step used): no result so far showed a listing '{pointed[0]}'. Use an id exactly as a search result shows it; "
                        "if you are sure this id exists, send the same call again.")
            return None
        if target in self.gone:
            return (f"Not run (no step used): {target} was reported sold out; re-opening it cannot bring it back. "
                    + (self.fallback_note() or "Choose another listing that meets every requirement."))
        row = self.shelf.rows.get(target) or {}
        raw = self.broad_raw.pop(target, None)
        if raw is not None and raw[0] == name and raw[1] > self.market_at:
            # the code read this sheet itself after a search: the model's own read gets that same record, with no step
            self.cache_hits += 1
            return _compact(raw[2]) + "\n[note] This is the record the code read for you a moment ago; it cost no store step."
        if name not in self.opened_by.get(target, set()) or row.get("step", 0) <= self.market_at or not self.shelf.records.get(target):
            return None
        self.cache_hits += 1
        return (f"Not run (no step used): {target} was already opened with this tool and nothing has changed since. Its full record as opened: "
                + self.sheet_text(target, 5000))

    def run_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        """One model turn: each proposed call is held or queued; a finish runs its own holds; the rest goes to the store in
        envelopes of up to sixteen, always leaving the last step for the finish."""
        replies: list[tuple[str, str]] = []
        allowed = {t["function"]["name"] for t in self.tools_for_turn()}
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        runnable: list[tuple[str, str, dict[str, Any]]] = []
        queued_messages = 0
        for call in calls:
            held, name, args = self.call_hold(call, len(calls), allowed, planned, queued_messages)
            if held:
                replies.append((call["id"], held))
                self.hold_log.append(f"{call['function'].get('name')}: {held[:160]}")
                continue
            queued_messages += name == self.roles.get("message")
            if name == self.roles.get("cart"):
                try:
                    args = self.merge_basket(args)
                except Exception as exc:
                    self.trace.append({"kind": "merge_basket", "error": f"{type(exc).__name__}: {exc}"[:300]})
            runnable.append((call["id"], name, args))
        if runnable and runnable[0][1] in finishing:
            call_id, name, args = runnable[0]
            declining = self.is_decline(name, args)
            if not declining and self.roles.get("cart") and not self.cart_ids and (not self.empty_held or not self.cart_used):
                # invariant: an order with nothing in the cart would only be refused; when a verified fit is known the code
                # carts it and orders, otherwise the order waits (once when the cart was used before, always when it never was)
                self.empty_held = True
                fits = self.rank_by_rule([k for k in self.live_fits() if k not in self.gone]) if not self.items() and self.parse_sure() else []
                if fits and self.steps_left() >= 2:
                    self.code_order(fits[0], f"The order was sent with an empty cart; {fits[0]} meets every condition by its own sheet.")
                    replies.append((call_id, "Not run as sent (the cart was empty): the code put " + fits[0] + " in the cart and ordered it; the result is above."))
                    return replies
                replies.append((call_id, "Not run (no step used): the cart is empty, so this order would be refused. Put the verified pick in the cart first "
                                         "(re-check its stock and price), then order; if nothing fits, decline instead. If you are sure the cart holds the item, send the same call again."))
                return replies
            self.order_args = args if name == self.roles.get("order") else {}
            held = None
            if not declining and name == self.roles.get("order") and self.add_unread and (not self.items() or self.basket_form()):
                peeked = self.peek_after_add()
                if self.finished or self.dead:
                    replies.append((call_id, "Not run: the episode is over."))
                    return replies
                if peeked:
                    self.relax_done = max(self.relax_done, self.relaxed_at)
                    if self.rerank_cart("A note arrived after the add."):
                        replies.append((call_id, "Not run (no step used): " + peeked + " The code re-ranked and ordered (see above)."))
                        return replies
                    held = ("Not run (no step used): " + peeked + " Apply it: when the held line still meets every remaining condition within the limit and is the best "
                            "by the request's rule, order it again; otherwise put the best fit in the cart and order it.")
            if not held and not declining and name == self.roles.get("order"):
                held = self.probe_order_hold()
            if not held and not declining and name == self.roles.get("order"):
                try:
                    held = self.fit_order_hold()
                except Exception as exc:
                    self.trace.append({"kind": "fitment", "error": f"{type(exc).__name__}: {exc}"[:300]})
                if self.finished or self.dead:
                    replies.append((call_id, "Not run: the episode is over."))
                    return replies
            held = held or (self.decline_gate() if declining else (self.order_hold() if name == self.roles.get("order") else None))
            if held and declining and self.finished:
                replies.append((call_id, held))
                return replies
            if not held and not declining and name == self.roles.get("order"):
                try:
                    self.swap_note = self.enforce_cheapest() or self.bundle_cheapest()
                except Exception as exc:
                    self.trace.append({"kind": "swap", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if not held:
                held = self.last_look(declining)
            if not held:
                held = self.settle_duties(declining)
            if held:
                self.trace.append({"kind": "finish_held", "declining": declining, "text": held[:240]})
                replies.append((call_id, held))
                return replies
            if declining:
                runnable[0] = (call_id, name, self.name_missing(self.clean_decline(args, name), name))
        if runnable and runnable[0][1] not in finishing and finishing:
            try:
                runnable = self.fit_pre_reads(runnable, replies)
            except Exception as exc:
                self.trace.append({"kind": "fitment", "error": f"{type(exc).__name__}: {exc}"[:300]})
            if self.finished or self.dead or not runnable:
                return replies
            if not self.pre_read and self.long_world() and not self.cart_used:
                first_add = next((args for _, name, args in runnable if name == self.roles.get("cart")), None)
                if first_add is not None:
                    target = next((v.strip() for v in first_add.values() if isinstance(v, str) and v.strip() in self.shelf.ids), "")
                    target = target or ((_listed_ids(first_add) or [""])[-1])
                    try:
                        self.pre_add_reads(target)
                    except Exception as exc:
                        self.trace.append({"kind": "pre_add_read", "error": f"{type(exc).__name__}: {exc}"[:300]})
                    if self.finished or self.dead:
                        return replies
            pace = self.pace_hold(runnable)
            if pace:
                replies.extend((call_id, pace) for call_id, _, _ in runnable)
                return replies
            # a reply with many calls must not eat the last steps: two are always kept for the finish (an order and,
            # when it is refused, one repair), except for a call that itself finishes a pending checkout
            closing = [r for r in runnable if self.finishes_checkout(r[1])]
            ordinary = [r for r in runnable if not self.finishes_checkout(r[1])]
            if self.step >= self.pace_marks()[1] and not self.order_attempts:
                # late in the episode a reply opens at most a couple of sheets: the long reading batches at this point are what push
                # episodes into the forced finish
                readers = {self.roles.get("details"), self.roles.get("view")} - {None}
                reads = [r for r in ordinary if r[1] in readers]
                if len(reads) > LATE_READS:
                    late = reads[LATE_READS:]
                    late_ids = {r[0] for r in late}
                    replies.extend((call_id, f"Not run (no step used): past step {self.pace_marks()[1]} a reply opens at most {LATE_READS} listings; "
                                             "read those first, then put the fit in the cart and order.") for call_id, _, _ in late)
                    ordinary = [r for r in ordinary if r[0] not in late_ids]
            keep = max(0, self.steps_left() - 2 - len(closing) - self.held_back())
            replies.extend((call_id, "Not run (no step used): the last steps are kept for the finish (the order or the decline).") for call_id, _, _ in ordinary[keep:])
            runnable = closing + ordinary[:keep]
        for start in range(0, len(runnable), 16):
            self.send_chunk(runnable[start : start + 16], replies)
        order = {c["id"]: i for i, c in enumerate(calls)}
        replies.sort(key=lambda r: order.get(r[0], 0))
        return replies

    def pace_hold(self, runnable: list[tuple[str, str, dict[str, Any]]]) -> str | None:
        """Once the order is overdue (a verified fit known, the commit mark passed, no lowest-price rule), one turn made
        only of exploring calls (searches, sheets, views) is held without a step; the next such turn goes through."""
        if self.pace_held or not self.order_overdue():
            return None
        exploring = {self.roles.get("search"), self.roles.get("details"), self.roles.get("view"), self.roles.get("cart_view")} - {None}
        if not runnable or any(name not in exploring for _, name, _ in runnable):
            return None
        self.pace_held = True
        fits = self.verified_now()
        named = ", ".join(f"{k} ({(self.shelf.rows.get(k) or {}).get('price', '?')})" for k in fits[:3])
        return (f"Not run (no step used): the order is overdue (step {self.step} of {self.max_steps}, order due by step {self.pace_marks()[1]}) "
                f"and a verified fit is already known: {named}. Put the chosen fit in the cart and order now; if you still must look at "
                "something first, send the same calls again and they will run.")

    def opening_messages(self) -> list[dict[str, Any]]:
        """The system playbook and the first user turn: request, opening observation, the brief and the status."""
        opening = self.policy.get("opening")
        parts = ["SHOPPER REQUEST:\n" + str(self.policy.get("query") or "")]
        if opening is not None:
            parts.append("OPENING OBSERVATION:\n" + _compact(opening))
            self.records.append(opening)
            _collect_ids(opening, self.seen_ids, keys=tuple(self.shelf.id_keys))
        brief = self.brief()
        if brief:
            parts.append(brief)
        parts.append(self.status())
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]

    def proposed_calls(self, message: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
        """The model's tool calls in one canonical shape (string arguments, an id for each), and its text."""
        calls = []
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            self.call_counter += 1
            function = call.get("function") or {}
            arguments = function.get("arguments")
            calls.append({"id": call.get("id") or f"call_{self.call_counter}", "type": "function",
                          "function": {"name": function.get("name"), "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {})}})
        content = message.get("content") or ""
        return calls, content if isinstance(content, str) else _compact(content)

    def answer_all(self, calls: list[dict[str, Any]], content: str, replies: list[tuple[str, str]]) -> None:
        """The assistant turn and its tool results into the transcript, the status after the last result."""
        self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        for index, (call_id, text) in enumerate(replies):
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text + ("\n" + self.status() if index == len(replies) - 1 else "")})

    def commit_too_early(self, calls: list[dict[str, Any]], content: str) -> bool:
        """With the planner on, a cart, order or decline proposed before the questions: open more sheets first (once, when
        fewer than three were read), else the questions are planned now. True when the turn is used up."""
        committing = {self.roles.get("cart"), self.roles.get("order"), self.roles.get("decline")} - {None}
        if not self.planner_on or not any(c["function"]["name"] in committing for c in calls):
            return False
        # questions are planned from the listings opened so far; with too few of them the plan is a guess
        if not self.early_bounced and self.property_menu()[1] < 3 and self.steps_left() > 12 and self.time_left() > WRAP_UP_SECONDS + 60:
            self.early_bounced = True
            note = ("Not run (no step used): before choosing, open the details of two or three more listings of the requested kind "
                    "(several in one reply, preferring ones that differ from those you have seen); the questions to the shopper are planned from what they show.")
            self.answer_all(calls, content, [(c["id"], note) for c in calls])
            return True
        if self.interview(force=True):
            # the proposed commit never reached the store: say so, or the model may believe the cart holds the item
            names = ", ".join(sorted({str(c["function"]["name"]) for c in calls}))
            self.messages.append({"role": "user", "content": f"[note] Your call(s) to {names} were not run (no step used): the questions to the shopper came first. "
                                                             "If they still stand after the answers, send them again now."})
            return True
        return False

    def note_list(self, action: dict[str, Any], observation: Any) -> None:
        """Remembers a paged result list: the call, the argument that moves through the rows, the next value the store gave,
        the reported total and the rows shown, so the code can read the next page itself."""
        if not isinstance(observation, dict) or not isinstance(action.get("args"), dict):
            return
        inner = observation
        for _ in range(2):
            lists = [v for v in inner.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
            if lists:
                break
            nested = [v for v in inner.values() if isinstance(v, dict)]
            if not nested:
                return
            inner = nested[0]
        if not lists:
            return
        rows = [str(field(x, ID_KEYS) or next((v for k, v in x.items() if isinstance(v, str) and _ID_HINT.match(v) and any(c.isdigit() for c in v)), "")) for x in lists[0]]
        nexts = [v for k, v in inner.items() if isinstance(v, int) and not isinstance(v, bool) and re.search(r"next|token|cursor|offset", flat(k))]
        totals = [v for k, v in inner.items() if isinstance(v, int) and not isinstance(v, bool) and re.search(r"total|tally|count|hits|found", flat(k))]
        props = (self.schemas.get(action.get("name")) or {}).get("properties") or {}
        movers = [k for k, r in props.items() if isinstance(r, dict) and r.get("type") == "integer" and re.search(r"page|offset|row|start|from|token|cursor", flat(re.sub(r"[_\-]+", " ", k)))]
        signature = _compact({k: v for k, v in action["args"].items() if k not in movers})
        entry = self.paged.get((action.get("name"), signature)) or {"tool": action.get("name"), "args": dict(action["args"]), "rows": [], "read": 0}
        entry["rows"] += [r for r in rows if r and r not in entry["rows"]]
        entry["next"] = nexts[0] if nexts and len(movers) == 1 else None
        entry["mover"] = movers[0] if len(movers) == 1 else None
        entry["total"] = totals[0] if totals else None
        if entry["next"] is None and entry["mover"] and entry["total"] and rows and len(entry["rows"]) < entry["total"]:
            # no next value in the reply: a page number moves by one, a row offset by the rows shown
            at = action["args"].get(entry["mover"])
            at = at if isinstance(at, int) and not isinstance(at, bool) else None
            entry["next"] = ((at or 1) + 1) if "page" in flat(entry["mover"]) else ((at or 0) + len(rows))
        self.paged[(action.get("name"), signature)] = entry

    def bundle_pages(self, single: bool = False) -> bool:
        """With several products in a long world: a product's result list that runs over a few pages is read to its end by
        the code (a step a page, BUNDLE_PAGES per episode), and the cheapest unread rows of that product within the limit are
        opened (the main product's best listing often sits on page two). True when steps were spent."""
        if ((not self.items() and not single) or self.max_steps < BUNDLE_PAGE_FROM or self.bundle_paged >= BUNDLE_PAGES or self.finished or self.dead
                or self.order_attempts or self.steps_left() < self.finish_reserve() + 12 or self.time_left() < WRAP_UP_SECONDS + 40):
            return False
        budget, _ = self.walk_scope()
        for entry in self.paged.values():
            if entry.get("next") is None or not entry.get("mover") or not entry["rows"] or entry.get("total") is None or entry["total"] > BUNDLE_LIST_MAX:
                continue
            owners = [self.product_of(k) for k in entry["rows"][:6]]
            product = max(set(owners), key=owners.count) if owners else ""
            if single and not self.items():
                product = "" if any(self.of_kind(self.shelf.rows.get(k) or {}) for k in entry["rows"][:6]) else None
            if product is None or (not product and not single):
                continue
            args = shape_args(self.schemas_full.get(entry["tool"]) or {}, {**entry["args"], entry["mover"]: entry["next"]})
            if lacking(self.schemas_full.get(entry["tool"]) or {}, args):
                entry["next"] = None
                continue
            before = len(entry["rows"])
            entry["next"] = None             # the reply sets it again when another page follows
            reply = self.env_call([{"name": entry["tool"], "args": args}])
            self.bundle_paged += 1
            self.trace.append({"kind": "bundle_page", "tool": entry["tool"], "args": args, "product": product})
            if reply is None or self.finished or self.dead:
                return True
            self.book_call(entry["tool"], args, reply.get("observation"))
            self.read_notices(entry["tool"], reply.get("observation"))
            self.note_observation(entry["tool"], reply.get("observation"))
            fresh = entry["rows"][before:]
            quantity = dict(self.items()).get(product, 1) if self.items() else self.walk_scope()[1]
            cheap = sorted((r["price"], k) for k in fresh for r in [self.shelf.rows.get(k) or {}]
                           if k not in self.shelf.sheets and r.get("price") is not None and r.get("stock") is not False
                           and (budget is None or r["price"] * quantity <= budget + 1e-9)
                           and (self.product_of(k) == product if self.items() else self.of_kind(r)))
            read = self.code_reads([k for _, k in cheap[:BUNDLE_PAGE_READS]]) if cheap else 0
            self.messages.append({"role": "user", "content": f"[note] The code read the next page of the {product or self.plan.get('product') or 'result'} list ({len(fresh)} more rows)"
                                  + (f" and opened the {read} cheapest within the limit: " + ", ".join(k for _, k in cheap[:read]) if read else "")
                                  + ". Their records are in the store's replies.\n" + self.status()})
            return True
        return False

    def empty_repair(self) -> bool:
        """After an order the store refused for an empty basket: the code holds the best-verified line of each product (the
        closest opened one where none is verified) and sends the order again. Once per refusal. True when steps were spent."""
        if (self.empty_fault <= self.empty_repaired or self.finished or self.dead or not self.roles.get("cart")
                or not self.roles.get("order") or self.steps_left() < 2 or self.time_left() < 3):
            return False
        self.empty_repaired = self.empty_fault
        before = self.step
        try:
            if self.items():
                self.fill_bundle(loose=True)
            else:
                self.order_known_fit()
        except Exception as exc:
            self.trace.append({"kind": "empty_repair", "error": f"{type(exc).__name__}: {exc}"[:300]})
        self.trace.append({"kind": "empty_repair", "held": list(self.cart_ids)})
        if self.cart_ids and not self.finished and not self.dead and self.steps_left() >= 1:
            order = self.roles["order"]
            action = {"name": order, "args": self.neutral_args(order, "Order the cart.")}
            reply = self.env_call([action])
            self.trace.append({"kind": "empty_repair", "action": action, "reply": reply})
            if reply and not self.finished:
                self.read_notices(order, reply.get("observation"))
                self.note_observation(order, reply.get("observation"), refused=True)
        if not self.finished and self.step > before:
            self.messages.append({"role": "user", "content": "[note] The order was refused because the basket was empty; the code put "
                                  + (", ".join(self.cart_ids) or "nothing") + " in it and sent the order again.\n" + self.status()})
        return self.step > before

    def code_bundle_round(self) -> bool:
        """The code's own search for a product of several that still lacks a known fit (money or model spent): the cheapest
        unopened rows of that product within what the limit leaves are opened (CODE_BUNDLE_READS a round), and a product
        never searched gets one plain search by its name first. True when steps were spent."""
        if not self.items() or self.finished or self.dead or self.steps_left() < self.finish_reserve() + 3 or self.time_left() < FORCE_SECONDS + 10:
            return False
        budget, _ = self.walk_scope()
        total = self.bundle_total() or 0.0
        good, bad, missing = self.bundle_state()
        todo = [p for p in list(missing) + [p for p, k in self.lines_for.items() if k in bad] if not self.product_fit(p)]
        if self.plan.get("priority") == "cheapest":
            # a product whose known line or fit costs more than an unopened row of it: the cheaper rows are opened too
            for product, _ in self.items():
                pick = self.lines_for.get(product) or self.product_fit(product)
                price = (self.shelf.rows.get(pick) or {}).get("price") if pick else None
                if price is not None and any(k not in self.shelf.sheets and k not in self.gone and r.get("stock") is not False
                                             and r.get("price") is not None and r["price"] < price - 1e-9 and self.product_of(k) == product
                                             for k, r in self.shelf.rows.items()):
                    todo.append(product)
        main = self.main_product()
        for product in sorted(dict.fromkeys(todo), key=lambda p: p != main):
            tries = self.code_bundle_tries.get(product, 0)
            if tries >= CODE_BUNDLE_TRIES:
                continue
            quantity = dict(self.items()).get(product, 1)
            rows = sorted((r["price"], k) for k, r in self.shelf.rows.items()
                          if k not in self.shelf.sheets and k not in self.gone and r.get("stock") is not False and r.get("price") is not None
                          and (budget is None or r["price"] * quantity <= budget + 1e-9) and self.product_of(k) == product)
            self.code_bundle_tries[product] = tries + 1
            targets = self.fit_targets() if product != main else []
            if targets and self.fit_tool():
                # under fitment codes the pages decide: the cheapest rows of the product whose page is unread are opened
                pages = sorted((r["price"], k) for k, r in self.shelf.rows.items()
                               if not r.get("paged") and k not in self.gone and k not in self.fit_asked and r.get("stock") is not False
                               and r.get("price") is not None and (budget is None or r["price"] * quantity <= budget + 1e-9)
                               and self.product_of(k) == product)
                if pages:
                    read = self.fit_reads([k for _, k in pages[:CODE_BUNDLE_READS]], "code bundle")
                    self.trace.append({"kind": "code_bundle", "product": product, "pages": read})
                    if read:
                        return True
            if rows:
                read = self.code_reads([k for _, k in rows[:CODE_BUNDLE_READS]])
                self.trace.append({"kind": "code_bundle", "product": product, "read": read})
                if read:
                    return True
                continue
            if f"searched|{product}" in self.keep_held:
                continue
            self.keep_held.add(f"searched|{product}")
            args = self.text_search_args(product)
            if args and self.code_search(args):
                self.trace.append({"kind": "code_bundle", "product": product, "searched": True})
                return True
        return False

    def code_powered(self, routine: Any) -> Any:
        """Runs one of the code's own routines with the spend lines lifted and every model call off (the money is spent)."""
        spent, down = self.cost, self.llm_down
        if spent < self.cost_stop and not down:
            return routine()
        self.cost, self.llm_down = 0.0, True
        try:
            return routine()
        finally:
            self.cost, self.llm_down = max(self.cost, spent), down

    def code_single_round(self) -> bool:
        """The money spent on one product: while no opened listing fits, or under a lowest-price rule while unopened rows of
        the kind within the limit are cheaper than the best known fit, the code opens the cheapest of them (CODE_BUNDLE_READS
        a round, CODE_BUNDLE_TRIES rounds). True when steps were spent."""
        if (self.items() or self.finished or self.dead or self.open_pointers() or self.steps_left() < self.finish_reserve() + 3
                or self.time_left() < FORCE_SECONDS + 10 or self.code_bundle_tries.get("", 0) >= CODE_BUNDLE_TRIES):
            return False
        budget, quantity = self.walk_scope()
        fits = self.rank_by_rule([k for k in self.live_fits() if k not in self.gone])
        best = (self.shelf.rows.get(fits[0]) or {}).get("price") if fits else None
        if fits and self.plan.get("priority") != "cheapest":
            return False
        rows = sorted((r["price"], k) for k, r in self.shelf.rows.items()
                      if k not in self.shelf.sheets and k not in self.gone and r.get("stock") is not False and r.get("price") is not None
                      and (budget is None or r["price"] * quantity <= budget + 1e-9) and self.of_kind(r) and (best is None or r["price"] < best - 1e-9))
        if not rows:
            return self.bundle_pages(single=True)     # the candidates may sit on the next page of the list
        self.code_bundle_tries[""] = self.code_bundle_tries.get("", 0) + 1
        read = self.code_reads([k for _, k in rows[:CODE_BUNDLE_READS]])
        self.trace.append({"kind": "code_single", "read": read})
        return read > 0

    def replace_duty(self) -> bool:
        """A step owed before any swap (checking with the shopper first) is carried out by the code when a line has gone or an
        order was refused and no model turn will do it: one message, then the swap may follow. True when a step was spent."""
        owed = self.open_duties({"before_replace"})
        if (not owed or self.swap_asked or not self.roles.get("message") or self.messages_open() < 1 or not (self.gone or self.refusals)
                or self.finished or self.dead or self.steps_left() < self.finish_reserve() + 3):
            return False
        text = ("The item I picked can no longer be bought as it was, so I would like to replace it with another listing that meets your requirements. "
                + "As you asked: " + "; ".join(d["do"].strip().rstrip(".") for d in owed)[:300] + ".")
        sent = self.say_now(text, "Before replacing the item, I check with the shopper as the request asks.")
        self.swap_asked = True
        for duty in owed:
            duty["done"] = True
        return sent is not None

    def code_rerank(self) -> bool:
        """Once in the code's own rounds, one product: the opened fits are ranked by the request's rule against the held line
        (after a raise the best-placed rows the old limit kept out are opened first); a better one is swapped in and ordered."""
        if self.code_reranked or self.items() or not self.cart_ids or self.plan.get("priority") not in ("best", "cheapest"):
            return False
        self.code_reranked = True
        return self.rerank_cart("The code re-ranked the opened fits by the request's rule.")

    def code_interview(self) -> bool:
        """A planned question still open goes out in the code's own rounds (the reply is read by the code's hearing)."""
        return bool(self.planner_on) and self.interview()

    def stock_check(self) -> None:
        """A held line whose latest record shows it out of stock is gone, whatever the note that came with it said."""
        dead = [k for k in self.cart_ids if k not in self.gone and (self.shelf.rows.get(k) or {}).get("stock") is False]
        if dead:
            self.gone.update(dead)
            self.mark_change()
            self.trace.append({"kind": "stock_check", "dead": dead})

    def code_phase(self) -> None:
        """Past the stop line, or with the model gone: the code's own rounds (repairs after a refusal or a note, recoveries,
        fitment reads, re-ranking, the commit and the bundle completion) still run, with every model call off, until none of
        them acts; the finish follows. The spend lines guard model calls, so they are lifted for these rounds only."""
        self.stock_check()
        spent, down = self.cost, self.llm_down
        self.cost, self.llm_down = 0.0, True
        self.trace.append({"kind": "code_phase", "step": self.step, "cost": round(spent, 5), "model_down": down})
        try:
            for _ in range(12):
                if self.finished or self.dead or self.steps_left() <= self.finish_reserve() or self.time_left() < FORCE_SECONDS:
                    break
                acted = False
                routines = ((self.replace_duty, self.empty_repair, self.repair_refusal, self.bundle_budget_resend, self.bundle_recover, self.bundle_reprice, self.refill_candidates,
                             self.recover, self.fitment_round, self.bundle_pages, self.code_bundle_round, self.code_single_round, self.rerank_after_rise, self.relax_reorder, self.code_interview, self.code_rerank, self.commit_fit, self.complete_bundle)
                            if self.parse_sure() else (self.replace_duty, self.empty_repair, self.bundle_budget_resend, self.bundle_reprice, self.code_interview))
                for routine in routines:
                    try:
                        if routine():
                            acted = True
                            break
                    except Exception as exc:
                        self.trace.append({"kind": "code_phase", "error": f"{type(exc).__name__}: {exc}"[:300]})
                if not acted:
                    break
        finally:
            self.cost, self.llm_down = max(self.cost, spent), down

    def code_round(self) -> bool:
        """The code's rounds before a model turn. The safe ones always run: the repair of an empty-basket order, a resend
        after a raised limit, the review of a change, a pointer need, the re-read of held lines after a price move, the
        questions. The ones that act on the code's own reading of the store (cart swaps and removals, its own adds and
        orders, paging and extra reads, fallback searches) run only while that reading is trusted (parse_sure). True when
        one of them used the turn."""
        sure = self.parse_sure()
        if self.empty_repair() or (sure and self.repair_refusal()) or self.bundle_budget_resend():
            return True
        if self.review_reason:
            self.change_review()
        for routine, acts in ((self.resolve_pointers, False), (self.bundle_recover, True), (self.bundle_reprice, False), (self.refill_candidates, True),
                              (self.recover, True), (self.zero_fallback_round, True), (self.fitment_round, True), (self.bundle_pages, True),
                              (self.rerank_after_rise, True), (self.relax_reorder, True)):
            if (sure or not acts) and routine():
                return True
        if self.planner_on and self.interview():
            return True
        return bool(sure and (self.page_early() or self.commit_fit() or self.complete_bundle()))

    def model_loop(self, turns: int) -> None:
        """Model turns (with the code's rounds between them) until the episode ends or a way out of the loop is taken."""
        idle = failures = 0
        for _ in range(turns):
            if self.finished or self.dead or self.llm_down or self.steps_left() <= 0:
                break
            if self.time_left() < FORCE_SECONDS or (self.step > 0 and self.steps_left() <= self.finish_reserve()) or self.cost >= self.cost_stop:
                break
            if self.code_round():
                continue
            if self.tool_turns >= min(MAX_TURNS, self.max_steps + TOOL_TURN_SLACK):
                self.trace.append({"kind": "turn_cap", "turns": self.tool_turns, "calls": self.model_calls})
                break
            self.tool_turns += 1
            # past the hard line the dearer backup model is left out: the main (cheapest) model is tried twice instead
            models = [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL] if self.cost < self.cost_hard else [MAIN_MODEL, MAIN_MODEL]
            message = self.infer(self.window(), self.tools_for_turn(), models, 2500)
            if self.llm_down:
                break
            if message is None:
                failures += 1
                if failures >= 2:
                    break
                continue
            calls, content = self.proposed_calls(message)
            self.last_content = content
            if self.commit_too_early(calls, content):
                continue
            if not calls:
                idle += 1
                self.messages.append({"role": "assistant", "content": content or "(no tool call)"})
                self.messages.append({"role": "user", "content": "Every reply must call at least one of the supplied tools. Continue.\n" + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            self.trace.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in calls]})
            replies = self.run_calls(calls)
            self.answer_all(calls, content, replies)

    def run(self) -> list[dict[str, Any]]:
        """Plan, then model turns until the episode ends; the agent finishes itself on time, on the last steps, after two
        silent models or three replies without a tool call."""
        if self.finished:
            return self.trace
        self.setup()
        self.trace.append({"kind": "plan", "plan": self.plan, "roles": self.roles})
        self.messages = self.opening_messages()
        self.model_loop(self.turn_cap)
        if not self.finished and not self.dead and (self.llm_down or self.cost >= self.cost_stop):
            self.code_phase()          # out of money or model: the code's own moves go on before the finish
        if not self.finished and not self.dead:
            self.force_finish()        # every way out of the loop that leaves the episode open ends here
        if (not self.finished and not self.dead and not self.llm_down and self.cost < self.cost_stop and self.refused_at >= self.step - 1
                and self.steps_left() > self.finish_reserve() + 1 and self.time_left() > WRAP_UP_SECONDS):
            # the code's own finish drew a note (a refusal): the model reads it and acts for a few more turns
            self.trace.append({"kind": "after_finish_note", "step": self.step})
            self.model_loop(6)
        if not self.finished and not self.dead:
            # every path ends in a finish: a refused code finish handed back to the model is finished again here even when
            # no time was left for the model's turns
            self.force_finish()
        self.trace.append({"kind": "parse", "good": self.parse_good, "low": self.parse_low, "raw_misses": self.raw_misses, "sure": self.parse_sure(), "blind": self.blind()})
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.model_calls,
                           "tool_turns": self.tool_turns, "llm_down": self.llm_down, "seconds": round(time.monotonic() - self.started, 1)})
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
