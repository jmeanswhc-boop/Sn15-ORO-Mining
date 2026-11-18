from __future__ import annotations

import json
import re
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------------------------------------------- models
# The first model of each list does the work; the others are fallbacks for a failed or unreadable reply, used only
# while the task's spend is under SPEND_SOFT (the actor's under SPEND_HARD). Every routine call starts on the
# cheap model: a race of 90 tasks must stay near $0.039 a task.
ACTOR = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"]
READER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]    # reads the request and the tools once
CHECKER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]   # item checks, the early decline audit, the rank check
LISTENER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]  # shopper lines
PLANNER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]   # the question plan
DEAR = {"anthropic/claude-sonnet-5.5"}   # once the task's spend reaches SPEND_DEAR these go after the cheaper models
# The stronger model judges only the moments that decide the finish, in the order given whatever the spend tier
# (Wire.lineup firm): a cheap check's rejection before an order or of a line's only candidate, and the decline audits.
ADJUDICATOR = ["anthropic/claude-sonnet-5.5"]
AUDITOR = ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"]
SPEND_SOFT = 0.02         # USD per task: above it no fallback model
SPEND_DEAR = 0.025        # above it the dear models go last
SPEND_HARD = 0.06         # above it only the actor and the reading of shopper lines are asked (and no second opinion)
SPEND_STOP = 0.075        # above it the code finishes the errand without a model (a call that could cross it is not made)
DEAR_GUESS = 3.0          # a dear model not called yet in the task is priced at this many times the dearest call so far
SONNET_AUDITS = 1         # decline audits per task that go to the stronger model (only with a listing left to judge)
CHECKS_MOST = 8           # routine item checks per task (2 per product line + 4 when more); the order-stage ones are not counted
ADJUDICATE_MOST = 3       # second opinions on a rejection per task
RECHECK_PER_LINE = 2      # read listings judged again per product line with no verified pick before a decline
ROW_TEXT = 160            # longest free text of one search row shown to the actor (the records keep it whole)

# ---------------------------------------------------------------------------------------------------------------- clock
DEADLINE = 265.0          # every call ends before this many seconds
SANDBOX_END = 292.0       # the sandbox stops the task near 300 s; no world call may run past this
CLOSING = 70.0            # with less time than this left, the errand heads for its finish
LAST_CALL = 30.0          # with less than this left, the code finishes the errand itself (no model call)
MODEL_FLOOR = LAST_CALL + 6.0
FAILS_LIMIT = 3           # actor calls failing in a row before the code finishes
VIEW_CHARS = 9000         # longest observation text shown to the actor
RECORD_CHARS = 16000      # longest text of one value in the checks' prompts and the code's own reads
TURN_LIMIT = 80
ASK_DEFAULT = 3           # questions assumed allowed when the request states no budget
WINDOW = 5                # properties one question names
# (an answer asking for one thing at a time: "can we do one item and then the next?", "could you split them up?")
ONE_AT_A_TIME = re.compile(
    r"\bone (?:prompt|query|item|thing|question|subject|topic|property|bit)\b[^.?!]{0,40}\b(?:at a time|then|before|per|and wait|first)\b"
    r"|\bone (?:at a time|by one)\b|\b(?:split|break) (?:them|it|those|these) up\b|\bseparately\b|\bsmaller (?:bites|pieces|chunks)\b"
    r"|\btrickle\b|\bfirst thing and\b|\bper turn\b|\b(?:then|before) the next\b|\ba lot to (?:juggle|take in)\b|\bthere'?s a pile\b")
# (keys that are no property a shopper chooses by: generic words of any store, like LINK_WORDS, no suite vocabulary)
WINDOW_SKIP = r"\b(?:brand|maker|seller|sold by|vendor|title|name|description|blurb|options?|variants?|sku|price|stock|on hand|comments?|ids?)\b"
FIRST_ORDER_BY = 7        # with this many steps left after a refusal, only cart, view and finishing tools remain
FIRST_ORDER_DUE = 10      # at most this many steps kept for the first order and one repair (a quarter of the steps, at least FIRST_ORDER_BY)
IDLE_LIMIT = 3            # replies without a tool call before the code finishes
STUCK_LIMIT = 4           # replies in a row where nothing was sent before the backup model takes a turn
SWEEP_READS = 8           # at most this many shelf slices the code reads itself when a change leaves the whole shelf acceptable
RECALL_PASSES = 2         # code-run reads of unread search pages per task (once per requirements version)
RECALL_PAGES = 2          # pages one recall pass reads
RECALL_OPENS = 2          # cheapest unread in-budget rows of those pages it opens
RECALL_FLOOR_DECLINE = 7  # steps left needed for a recall pass held at a decline
RECALL_FLOOR_MID = 12     # steps left kept by a recall pass at mid-errand (nothing verified at half the steps)
POINTERS_MOST = 2         # needs given by pointing at another listing, looked up by the code per task
POINTER_STEPS = 4         # steps one look-up may use (its searches and details reads together)
POINTER_SEARCHES = 2
POINTER_READS = 2
HANDOVER_STEPS = 3        # steps per task that follow listings' hand-overs (reads and searches together)
HAND_OVER_RX = re.compile(
    r"\bfor (?:the |its |this item's )?([a-z][a-z0-9 /&\-]{1,40}?),\s*(?:go by|see|refer to|look (?:at|to)|use|check|consult|follow|rely on)\b"
    r"|\b(?:its |the )?([a-z][a-z0-9 /&\-]{1,40}?) (?:is|should be|must be|is to be|are to be|has to be) (?:taken|read|copied|drawn|sourced) from\b"
    r"|\btake (?:the |its )?([a-z][a-z0-9 /&\-]{1,40}?) from\b", re.I)
ELSEWHERE_RX = re.compile(r"\bnot (?:from |by |to )?this\b|rather than this|\banother\b|\bdifferent\b|\bother\b|\blisting\b", re.I)
HOLDS_RELEASE = 3         # advisory finish holds per task after which an order goes through
SWAPS_MOST = 2            # sold-out or over-budget cart lines the code replaces itself per task
RULE_ROUNDS = 2           # rule audits per (stage, cart version, requirements version)
RULE_CALLS = 3            # rule audit model calls per task
RULE_STEPS = 2            # world calls the rule audit may add per task
RULE_MOST = 6             # rules one audit judges
RULE_AUDITOR = CHECKER[:1]
SELLER_KEY = re.compile(r"seller|vendor|brand|maker|merchant|store|shop|manufactur|supplier|sold ?by|retailer")
LINK_WORDS = {"and", "or", "the", "for", "with", "of", "a", "an", "in", "on", "to", "by", "from", "at", "as", "is", "are", "be",
              "this", "that", "it", "its", "your", "our", "you", "all", "any", "not", "no"}
ASK_AFTER_SETS = 4        # listing detail records read before the first planned question
ASK_GRACE_STEPS = 3       # while fewer records were read and over 20 steps remain, this many more steps for them
ASK_AFTER_STEPS = 9       # or this many steps used

# ---------------------------------------------------------------------------------------------------------------- prompts
ACT_RULES = """You are a careful shopping assistant working for one shopper in a simulated store; the supplied tools are your only way to act. The errand scores 1 or 0, and it scores 1 only when the finish is right in every respect: the ordered item meets every requirement the shopper wrote, every hidden need the shopper reveals when asked, and every change the shopper announces along the way; the quantity and the budget are right; the cheapest (or best) item is chosen when the request asks for that; and every rule of procedure in the request is obeyed. When nothing in the store meets everything, the right finish is a decline, done exactly as the request says.

HOW THE STORE WORKS
- Each tool call costs one step, failed and invalid calls included, and steps are limited. Thinking costs nothing.
- One reply may carry several tool calls when none needs another's result (for example opening several listings' details); they run in order. An order or a decline always goes alone.
- Tool names, argument names and result shapes are different in every store. Read each tool's description and JSON schema and use exactly the names and allowed values given. Learn from the first results which tool shows which facts (price, stock, detailed properties).
- Early on, learn which category labels the store puts on listings of the requested kind. When neighbouring labels exist, the right one is the label that matches the request's wording as a whole; search and buy inside it.
- Searches differ from store to store; learn this one's from its first results. Rows may come best match first or in a fixed id order; a keyword search may need every word (adding a word shrinks the total) or take any word (adding a word grows the total, and listings matching more of the words rank higher); only a few rows or pages may be visible; a filter may work in one mode only (a keyword mode may ignore the category or price filter, a filter mode may ignore the words). Compare a reported total with the rows you can see: when it is larger, the fitting or cheapest listing may sit on a later page or beyond the visible ones. When every word is required, narrow with more exact words (or the category and price filters with a lower price limit) until the total fits on the visible pages. Keep queries short (one to three words) with the most distinctive words FIRST: some stores read only the first word or two of a query and ignore the rest, so a long query may search only its opening words (compare a longer query's total and rows with a shorter one that starts the same way; the notes report what they show). When every word is required, a required value with the item kind ("blue chair", the value first) lists exactly the listings that carry it. When any word matches, common words (the item kind, a unit) match most of the shelf and bury the rest: a required value's own distinctive word or figure alone ("herringbone", "30m", "0.4") lists just the listings that carry it, so search it alone and in the listings' other spellings ("30 m", "3 atm"), read every page of such a search when its total fits on them, and put the item kind with the values together only to rank the closest matches first. A keyword search may also match category names and property values. Notes after each search say what the totals so far show about this store's search.

THE SHOPPER AND HIDDEN NEEDS
- The shopper holds needs they did not write down and reveals one only when a message asks about that property. A message is answered only about the FIRST thing it asks, so each message asks about ONE property: a short, plain, friendly question that names the property (one or two example values are fine). No instructions, no lists, labels, positions or rules, never two questions in one.
- "Don't mind", "no preference", "anything is fine" means the property is free. A stated preference or limit is a hard requirement from then on, exactly as worded, numbers included.
- When questions are planned for you, they go out once you have opened the details of two or three listings of the requested kind, so open those in your first moves. The answers arrive in the conversation with their meaning; apply each revealed need as a hard requirement and search again with it and the stated values together.
- If you ask yourself: first open two or three listings' details, then ask about what the story hints at (who it is for, how it must look, where or how it is used), then about properties the listings carry that differ among candidates. Never ask what the request already settles and never go over the message budget.

RULES OF PROCEDURE
- The request may set rules for how to work: how many listings to look at or compare, what to check, what to tell the shopper, what a remark must say, how to decline. They are graded like requirements; follow each one literally and fully. "Look at" or "open" N listings means N fully fitting listings, each opened with the details tool.
- The code keeps a list of these steps and carries some out itself (a comparison of your finalists, a hold or reservation of the pick, an account report, and a message to the shopper before the order only when the request asks for one); the status line names the ones still open. An order or decline sent while a step is open is held until the step is done, and before any finish the code checks the request's other rules against what was done.
- Do only what the request and the shopper ask. Never send the shopper a basket summary, a confirmation request or a progress report the request does not ask for; messages are for questions about hidden needs and for answering the shopper.
- When the shopper asks you something (for example to confirm a change), it is answered for you while messages remain; the change itself is binding.
- What the shopper would merely like ("ideally", "would be nice", "if possible") is not a requirement: it only breaks ties between items that already meet everything, and never beats a cheapest rule.
- When the request leaves something only the shopper can settle (for example two wishes that may not both be met and the shopper must say which matters more), ask before you choose. A request that names no price limit has none: never ask for a budget.

CHANGES WHILE YOU WORK
- The shopper may change things: a new budget (higher or lower), a requirement dropped, replaced or added, a price rise, an item selling out. One line can carry several changes. Such lines can arrive inside any tool result (an extra field holding sentences) or with an order attempt. Each change binds from that moment: re-check the pick and the backups, and when the cheapest or best is asked for, choose again over the new acceptable set.
- An order attempt can be REFUSED: the episode goes on and the cart is as it was. Read the refusal, apply every change it states, take out of the cart whatever no longer fits (for example a sold-out item), put in the next verified fit, re-check its stock and price, and order again.
- Items can also sell out or change price without notice, so stock and price are re-checked right before every order attempt; an item whose current price times quantity is over the current budget no longer fits.

THE CART
- The errand may ask for one product or for several different products, each with its own number of units. The basket you order must hold exactly one listing per requested product, at that product's number of units, and nothing else.
- A cart call may raise the units of a listing that is already in the cart instead of setting them (one more of a held listing makes two). So never put in again a listing the cart holds: to change its units, take out or put in the difference. When a listing has to go (sold out, no longer fitting), take out all of its units before its replacement goes in. A note after each cart call says what the code counts in the cart and what the world's own basket reply showed.

HOW TO JUDGE A LISTING (this decides the score)
- The grader reads each requirement by MEANING, the way a careful shop assistant would, not by exact spelling. Read it from the listing's field for that property. A field whose name says the same property counts whatever its spelling: separators, a prefix, a shorter or longer name, or a usual alternative name ("fit" for a pants fit type, "neckline" for a collar type, "dial size" for a watch dial size, "color family" for a colour). When a field carrying the property's own name exists, it decides; otherwise such an alternative field decides; when no field names the property at all, the options list or the title may show the value. The description is never evidence.
- Values match by meaning: case, separators, singular and plural, abbreviations and common synonyms are the same ("tech wood" is technical wood, "win11" is windows 11, "embroidered" is embroidery, "solid" is plain); a more specific value that contains the required one meets it ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060); one element of a list is enough; a shade usually sold as that colour meets it (ivory or cream for off-white). A different value of the property does not (quartz is not mechanical, rosewood is not maple), nor does a word that only looks alike. A value that claims to cover every value of the property ("suitable for all occasions", "all seasons", "universal") names none of them and does not meet a required value; a list that also names the required value still meets it. A field that names another value of the property does not meet the required value just because the two often go together or one tends to imply the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); only a shade usually sold as the required colour, or a more specific value that contains it, counts.
- A field of a different property is not the requirement's property (a lining material is not the material, a length is not a style).
- Only what the shopper stated or revealed counts. Never rule out a listing for something the shopper did not ask for: an age group, gender, size, occasion, condition, quality or use you guess from the story or the description. The story is context; if it seems to matter, ask the shopper about that property instead of assuming. Your own questions are not requirements: only the shopper's answers are.
- Numbers are compared as amounts after converting units (4.5 cm is 45 mm); limits include the edge; a range counts only when all of it meets the limit. When no field names the property and the title is the evidence, an exact required value is met when the title shows its words and figures in the same order, even with more detail after them (9' x 12' 4" shows 9' x 12'; 9' x 13' does not).
- Respect exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
- The requested kind of product is met by the listing's category alone: a listing on the requested category IS that kind for the grader even when its title or description names another object (a container, a case, a refill, a set, an accessory), so never skip, reject or demote it for what it seems to be; judge it only on the requirements, price and stock. No property has to repeat the category name. When the store labels categories, the item must sit in the one whose label matches the request's wording as a whole (a request for "A or B" matches a category "A & B", not a narrower neighbour "B"). It must be in stock when the request wants things on hand, and price times quantity must be within the budget (equal is fine). A listing's price is in the budget's unit unless its record names another (45 against a budget of 1000 US cents is 45 cents); never rescale either.
- A decline is rarely the right finish. When no listing meets the strict wording but one meets each requirement by meaning (a synonym, a shade, a more specific value, an alternative field name), that listing is the fit; decline only when the search really found nothing that meets every requirement by meaning.

CHEAPEST OR BEST
- For the cheapest, the order must be the lowest-priced item among ALL items meeting every requirement; one fitting item is not enough. When a price filter is offered and holds, use it with the category and lower the price limit until the reported total fits on the visible pages; without one, search each required value with the item kind as a short query, the value first ("blue chair", "30m watch"), and read every page of it when its total fits on the readable pages (when it does not, add a second required value or use a narrower kind word); do the same with the kind's other names ("stool", "bench") and with the value alone; when a value cannot be searched (a figure the store drops), also search the product line of each fit or near-fit you find, its maker with one distinctive word of its title as two words, every page: a line's other sizes and colours are separate listings that share its property values and may cost less; sort the visible candidates by price yourself and open their details from the cheapest up; the first whose property fields meet everything, with every cheaper one opened and rejected, is the answer. Keep the next fitting ones as backups.
- For the best by some measure (most, largest, longest, highest...), compare ALL fitting items on that measure the same way and take the extreme. An item stating no value for the measure cannot win.

FINISHING
- For every requested product keep a ranked list of verified fits with a backup: a pick may sell out or change price when you order.
- Before every order attempt: the basket holds one verified listing per requested product at its units and nothing more; after the last cart change the stock and price of each line were re-read; the basket total is within the budget.
- A notice about stock, price, budget or a dropped requirement means re-check and re-pick the product it touches, never a decline by itself.
- Decline only after a thorough search finds nothing that meets every requirement, following the request's decline rule exactly. When the request says not to offer substitutes, the decline remark names, describes or points to no listing: no ids, titles or prices.
- Budget your steps: the first order attempt goes out while at least seven steps remain, so a refusal can still be handled. With 7 steps left only cart, stock-check and finishing tools remain. Decide once; do not swap the cart back and forth.
- Every reply carries at least one tool call. Keep the text short: the requirements now, the verified fits with price and stock, the next move."""

READ_REQUEST = """You prepare a shopping errand for an assistant. The shopper's request and the store's tools are below. Reply with a single compact JSON object (no indentation, no other text) with these keys.

"items": one entry per DIFFERENT product the shopper wants in this one order, each {"product": the kind of product in the request's own words, "quantity": the number of units of that product to order, "requirements": [...], "priority": "cheapest", "best" or "none"}. A shelf or category name that joins two kinds ("tables and chairs", "A or B") is still ONE product, and several units of one product are one entry with that quantity. The quantity is 1 unless the request plainly orders several units ("three of them", "2 packs"); a pair, set or kit sold as one listing is 1, and how many people share or use it, or how many old ones there were, is story.
  Each requirement is one string "property: constraint" with the shopper's exact value, limit, exclusion or closed list for THAT product ("pattern: herringbone", "drawers: at least 4", "material: not plastic"). Leave out price, stock, quantity, budgets and mere wishes. Do not repeat the product kind: words that are part of the category's own name (a season, an occasion, an audience in it) are covered by the category. The story is never a requirement (who it is for, the event or place where it is used, when, how many people share it, that it replaces an old one); an occasion, use or audience counts only when the request makes it the item's own property ("the occasion has to be business").
  "priority" is "cheapest" only when the request plainly asks for the cheapest or lowest-priced fitting listing of that product, "best" only for the best by a named measure; saying money is tight is not a cheapest request.
"priority_detail": for "best", the measure; else "".
"price_cap": the highest TOTAL price for the whole order, as the number the request writes in its own unit (never converted: "3050 US cents" gives 3050), or null.
"currency": the request's price unit, or null.
"in_stock_only": true when the request wants only listings on hand or in stock.
"message_budget": the number of messages the assistant may send the shopper, or null when not said.
"step_budget": the number of moves or steps the request allows, or null.
"rules": every instruction on HOW the assistant must work (how many listings to open or compare, what to hold or reserve, what to report, check or tell the shopper, what to do or say when nothing fits), copied word for word. What the shopper will do themselves is story.
"procedures": the same instructions, one object each: {"text": the request's own words, "kind": one of "survey" (open at least N listings before choosing), "compare" (put candidates side by side with a comparing tool), "hold" (hold, reserve or set aside the pick before buying), "report" (pull up an account, report, statement or summary), "confirm" (ask the shopper's OK before ordering), "tell_basket" (tell the shopper the basket, prices or total before ordering), "ask_before_change" (ask before switching to another listing), "decline_wording" (how to word the decline), "other"; "count": the number the step names or null; "timing": "before_order", "before_decline", "before_finish" or "any"}. Most requests state none: then [].
"soft_wants": what the shopper would like without needing it ("ideally", "preferably", "would be nice"). Never in requirements.
"ask_first": what only the shopper can settle and should be asked before choosing (for example which of two wishes wins when the request says both may not be possible; then both wishes go in soft_wants), or anything the request says to check with the shopper. Empty when none.
"decline_no_names": true when the request's rule for declining forbids naming, offering or suggesting other listings; false when it asks to name something; null when it gives no rule.
"references": requirements whose value the request does not name but points to on ANOTHER listing of the store (the value that listing gives or shows), one object each: {"property": the property, "kind": the kind of product that listing is, "seller": its seller, brand or shop as written, or "", "facts": the other details given to single that listing out, each "property: value", "for": the product the requirement is about}; such a requirement is left out of "requirements". Most requests have none: then [].
Every sentence lands somewhere: a requirement, a priority, a soft want, a rule, a budget, or story. When unsure whether a sentence is a rule, put it in "rules".
"roles": the exact tool name for each role, or null when no tool does it: "search" (finds listings), "details" (one listing's full record), "view" (a second single-listing view, if any), "compare" (several listings side by side), "hold" (holds or reserves one listing), "report" (an account report or statement), "message" (sends the shopper a message), "cart" (puts listings in or takes them out of the cart), "cart_view" (shows the cart), "order" (buys or checks out), "decline" (ends the errand without buying). One tool may finish both ways; then name it for "order" and for "decline".
"finish_modes": when one finishing tool both orders and declines through an argument, {"argument": its name, "order": the value that buys, "decline": the value that ends without buying}; else null.
"cart_modes": when the cart tool puts in and takes out through an argument, {"argument": its name, "put": the value that adds, "take": the value that removes}; else null.
"""

HEAR = """You help a shopping assistant understand what the shopper just said. Answer with ONE JSON object, nothing else, holding:
"meaning": one short sentence on what the shopper means.
"needs": new firm requirements on the product, each "property: constraint" with the exact value, limit, exclusion or closed list given.
"no_preference": properties the shopper said they do not care about.
"dropped": earlier requirements the shopper no longer needs.
"new_budget": the new highest total price as a number in the unit of the request's budget (the figure the line writes when it uses that unit: "13075 cents" gives 13075 for a budget in cents), or null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), else false.
"requirements_now": the full list of firm requirements on the product as they stand now, each "property: constraint": start from REQUIREMENTS SO FAR, keep each unless the shopper dropped or replaced it, for a replaced value keep only the newest, and add each new need the shopper's lines state. Never add a requirement from the request's story that REQUIREMENTS SO FAR does not hold; only the shopper's lines add requirements. No price, stock, quantity or product category.
"reply": when the shopper asks the assistant something or asks for a confirmation, a short plain answer to send back that names the change in substance (for example confirming the new value); else "".
"understood": false when the shopper did not follow the assistant's message and asks for it again in another form (to repeat it, say it simpler or shorter, ask one thing at a time); else true. A reply that says they have no preference is understood.
"references": the needs whose value the shopper does not name but points to on ANOTHER listing of the store (the value that listing gives, shows or states), one object each: {"property": the property whose value is wanted, "kind": the kind of product that other listing is, "seller": the seller, brand or shop that listing is from as written, or "", "facts": the other details the shopper gives to single that listing out, each "property: value", "for": the errand's product the need is about}; else []. Such a need goes ONLY here, never in needs or requirements_now: the assistant reads that listing and adds the value itself.
When the errand has several products, end each requirement in "needs" and "requirements_now" with " (for <product>)" naming the product it is about, as REQUIREMENTS SO FAR does.
One line can carry several changes; report each. A need on another property, even a related one (a seat height beside an overall height, a strap width beside a width), is a new requirement that replaces nothing: an earlier requirement is dropped or replaced only when the line says so about that same property. A replaced requirement is both dropped (the old) and a new need (the new). When the shopper says which of two wishes matters more, the preferred one is a need. A stated budget is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and kindly: folksy, figurative or roundabout words that name a value mean the shopper firmly wants that value; a value is excluded only when the shopper clearly says to avoid it. Words that only stress how firmly the shopper holds a value (only, nothing but, nothing short of, strictly, stays, sticking with, locked to, holding out for) are not part of it: write just the value words ("material: jute", "color: white"), never "<value> only"; only a shopper who also rules out mixes or other values in so many words adds an exclusion. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit. A new budget replaces the old one whether higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference or anything works, put the property asked about in no_preference. The assistant's own question is never a requirement: only what the shopper says is; values named as examples in the question do not become needs. A "don't mind" answer frees only the property the assistant's message asked about; it never drops or changes a need the shopper stated earlier, and when the message asked nothing (it only told something), such an answer changes nothing."""

READ_RULES = """- Read each requirement by MEANING, as a careful shop assistant would, not by exact spelling. Read it from the listing's field for that property: a field whose name says the same property counts whatever its spelling (separators, a prefix, a shorter or longer name, or a usual alternative name such as "fit" for a pants fit type, "neckline" for a collar type, "color family" for a colour). When a field carrying the property's own name exists, it decides; otherwise such an alternative field decides; only when no field names the property may the options list or the title show the value. The description is never evidence.
- Values match by meaning: case, separators, singular and plural, abbreviations and common synonyms are the same ("tech wood" is technical wood, "win11" is windows 11, "embroidered" is embroidery, "solid" is plain); a more specific value that contains the required one meets it ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060); one element of a list is enough; a shade usually sold as that colour meets it (ivory or cream for off-white). A different value of the property does not (quartz is not mechanical, rosewood is not maple), nor a word that only looks alike. A value that claims to cover every value of the property ("suitable for all occasions", "all seasons", "universal") names none of them and does not meet a required value; a list that also names the required value still meets it. A field that names another value of the property does not meet the required value just because the two often go together or one tends to imply the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); only a shade usually sold as the required colour, or a more specific value that contains it, counts.
- A field of a different property does not count (a lining material is not the material, a length is not a style).
- Numbers are compared as amounts after converting units (4.5 cm is 45 mm); limits include the edge; a range counts only when all of it meets the limit. When no field names the property and the title is the evidence, an exact required value is met when the title shows its words and figures in the same order, even with more detail after them (9' x 12' 4" shows 9' x 12'; 9' x 13' does not).
- A listing's price is in the same unit as the request's budget unless its record names another currency or unit (a price of 45 against a budget of 1000 US cents is 45 cents, well within it); never rescale a price or the budget.
- Only the requirements listed count: never fail a listing for an age group, gender, size, occasion, condition, season, quality or use the requirements do not name.
- The requested kind is met by the category label alone: a listing on the requested category is of that kind even when its title names another object (a container, a case, a refill, a set, an accessory); judge it only on the requirements, price and stock."""

CHECK_ITEM = """A shopping assistant is about to commit to one listing. The errand passes only if the listing meets every requirement of the product it is meant for. Decide from the store records shown and nothing else, reading them this way:
""" + READ_RULES + """
- A word that is part of the category's own name is met by the category; exclusions and closed lists hold.
- The listing must be of the requested kind; when listings carry category labels, its label is the one matching the request's wording as a whole, not a narrower or neighbouring one.
- Its price times the units wanted must fit the budget (equal fits); with several products, the basket total counts. Records showing it out of stock fail it.
- A wish ("ideally", "would be nice") is not a requirement.
Where records disagree (price, stock), the newest wins.
When the errand lists several products, first decide which product line the listing is for (by its kind), then judge it only against that line's requirements.
Reply with one JSON object only: {"item": the listing's id, "line": the number of the product line it is for (1 when there is one product), "fits": true or false, "problems": [short strings naming each requirement it fails or does not show]}. Say fits=false only for a listed requirement its fields clearly fail by meaning, or one that no field states and (when no field for it exists) neither the options list nor the title shows; a missing stock or price record is no reason to fail it."""

RANK_CHECK = """A shopping assistant is about to order an item for a request that asks for the PRIORITY. Below are the requirements, the chosen item and the listing records seen. Using the reading rules below (price times quantity within the budget; not shown out of stock), name the listings that may meet every requirement AND beat the chosen item on the priority (a lower price for the cheapest; a better value of the measure for the best). Leave out listings the records show failing a requirement, out of stock or over budget.
READING RULES:
""" + READ_RULES + """
Answer with ONE JSON object, nothing else: {"better": [ids, best first, at most 4], "reason": "one sentence"}."""

ASK_PLAN = """You plan the questions a shopping assistant sends to a shopper. The shopper holds hidden needs: each is a value of one property of one requested product, revealed only when a message asks about that exact property of that product; each message is answered only about the first thing it asks, and questions about anything else get a polite "don't mind". A hidden need left unasked loses the errand even when the listing bought happens to fit it, so every question goes on the likeliest property left. Choose up to QUESTION_COUNT different properties, most promising first:
0. first, anything listed below as MUST BE LEARNED FROM THE SHOPPER that has not been asked; ask it plainly. For a choice between two wishes, name both values with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?";
1. then properties the listings' details below actually carry. Read the story first: the hidden need is usually the property its concrete details concern without naming the value. What the user will touch or press, see up close or in photos, hear, measure or fit it against, read on the label, or how it was made points at that part or aspect of the item and its property: tags, labels or the fine print point to the material, fabric or ingredients; photos and looks to the pattern, colour or the visible part; a body part or a space it must fit to that part's shape or size; buying for someone else, or someone barely known, to whom it is made for (gender, age group); how it must feel to the hand, skin or foot first to the material or fabric; where, when or for what activity it is worn or used (a gym class, the office, outdoors) to its style or type, the property whose values name such uses; something done up or adjusted again and again to its closure or mechanism; what does the work inside it to its core component. Weather, a job, a schedule or a place that is not where the item is worn or used are background. Ask first about the property the story's details point at, by the name the listings use for it. Whom the item is made for is a candidate like any other: ask it when the story points at someone else, or when nothing else is hinted. Then prefer properties describing what the product is like (who it is for, colour, pattern, material, shape, type or style variant, closure, function) over measures (size, capacity, weight, dimensions, speed) and logistics (warranty, condition, variant or size lists, years). With several products, each product gets a question before any product gets a second, and each question names its product by the category its listings carry: a question about the right property of another product reveals nothing. Among properties of equal promise take the one carried by most of that product's listings, with values that differ between them; a property seen on only one or two listings is a poor question.
Never pick a property the request already settles (its requirements, the price, stock, quantity), never repeat one already asked, never spend two questions on near-duplicates (two kinds of style, two spellings of one property), and never ask about brand, maker, a product or model name, price, stock or delivery. Ask only about a property that appears as a key in the listings' details below, by the name the listings use for it. Each step-1 question is one short, plain, friendly sentence (at most 15 words) asking which <property> they would like for the <item kind>, naming the property by the listings' own name for it, with no example values. Never ask whether anything else matters, or which property matters most: a question that names no property reveals nothing. No lead-in, no status report, no instructions or lists, never a second property, and never a catch-all key such as options, variants, a description, a title or a name.
Answer with ONE JSON object, nothing else: {"questions": [{"property": "...", "question": "..."}, ...]}."""

WINDOW_PLAN = """You rank the properties a shopping assistant will name when it asks a shopper what they want. The shopper holds hidden needs: each is a value of one property of one requested product. One message names up to five properties of one product ("What a, b, c, d or e would you like for the <product>?"), and the shopper answers about the first of them that matters to them, so the order decides what is found. For EACH requested product, list up to TEN properties, most promising first:
- only properties that appear as keys in the listings' details below for that product's category, by the name the listings use;
- read the story first: the hidden need is usually the property its concrete details concern without naming the value. What the user will touch or press, see up close or in photos, hear, measure or fit it against, read on the label, or how it was made points at that part or aspect of the item: tags, labels or the fine print point to the material, fabric or ingredients; photos and looks to the pattern, colour or the visible part; a body part or a space it must fit to that part's shape or size; buying for someone else to whom it is made for (gender, age group); how it must feel first to the material; where, when or for what activity it is used to its style or type; something done up or adjusted again and again to its closure or mechanism; what does the work inside it to its core component;
- then properties describing what the product is like (who it is for, colour, pattern, material, shape, type or style, closure, function) before measures (size, capacity, weight, dimensions, speed) and logistics (warranty, condition, pack size, years);
- among equals, the property most listings of that product carry, with values that differ between them.
Never list a property the request already settles (its requirements, the price, stock, quantity), one already asked, near-duplicates (two spellings of one property), or brand, maker, seller, a model or product name, price, stock, delivery, options, variants, descriptions, titles or names.
Answer with ONE JSON object, nothing else: {"properties": [{"product": the product as the request names it, "property": "..."}, ...]}."""

NEXT_ONE = """
REMAINING question(s) are left and every one should be used: list that many, best first, when that many useful properties are listed. They go out in that order; the list is made again only when listings of another requested product have been read."""

BEFORE_DECLINE = """A shopping assistant is about to tell the shopper that nothing in the store fits; that is rarely the right finish. Check the listing records below against the requirements by these reading rules (the price must be within the budget and the item must not be shown out of stock):
""" + READ_RULES + """
Some listings below are known only from search rows (no details read, so their property fields are unknown). Such a row on the requested category (whatever object its title names) whose title, price and stock do not show it failing a requirement has not been checked: list it under "unopened", the likeliest first (its title naming a required value, or the requested kind with nothing against it).
Answer with ONE JSON object, nothing else: {"possible_fits": [ids of listings that seem to meet every requirement, best first, at most 5], "unopened": [ids of rows-only listings to open before declining, at most 6], "reason": "one sentence"}."""

RULE_AUDIT = """A shopping assistant is about to FINISH. The shopper's request set the rules of procedure below; each one is graded, and the errand fails when a rule that applies was not followed. Judge each rule only from the numbered record of what was done (each line: the tool called, its arguments, the start of its result).
- "done": a call in the record carries the rule out: the right kind of tool, the listing or listings the rule concerns, at the moment it names. A step the rule ties to the item being bought, or to "before ordering", counts only for the listings now in the cart, and only after the line marked LAST CART CHANGE when the rule says it comes after choosing.
- "not_done": the rule applies to this errand and to this finish, and nothing in the record carries it out.
- "n/a": the rule does not apply now (it is about declining while the assistant orders, or the reverse, or about a situation that never happened), it is a requirement on the product, its price or the budget rather than a step of work, or it is about what the shopper will do.
When a rule is not_done and ONE call of the tools below carries it out now, give that call, with arguments that follow the tool's schema exactly and listing ids copied from the record. Never a call that orders, declines, or puts listings in or takes them out of the cart: those stay with the assistant. A message to the shopper only when the rule itself asks to tell or ask the shopper something; then write it short and plain, in the rule's terms, from facts in the record.
Answer with ONE JSON object, nothing else: {"rules": [{"n": rule number, "status": "done" or "not_done" or "n/a", "why": "a few words"}], "call": {"name": tool name, "args": {...}} or null, "for_rule": the rule number the call carries out, or null}."""

DECLINE_WORDS = r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without|forgo|walk.?away|pass\b|none\b|^no$"
ORDER_WORDS = r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed|submit|^yes$"
PUT_WORDS = r"\badd|\bput\b|\bplus\b|insert|include|increase|\bmore\b|\bload"
TAKE_WORDS = r"remov|delet|discard|minus|\btake|decreas|\bless\b|\bdrop|clear|\bout\b|subtract|unload|\bundo"
SORRY = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
ID_LIKE = r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}"
CODE_LIKE = r"[A-Za-z0-9_\-]+"
COUNT_KEY = r"qty|quant|count|pieces|units?\b|number_?of|how_?many|copies"

# Generic verbs that say what an unfamiliar tool does; always read together with its argument schema.
FIND_CUE = re.compile(r"\b(?:search\w*|find\w*|look ?up|lookups?|brows\w*|query|queries|catalog\w*|filter\w*|scan\w*|list(?:s|ing)? (?:items|products|listings))\b")
READ_CUE = re.compile(r"\b(?:detail\w*|spec\w*|sheet|attribute\w*|propert\w*|facts?|full|record|info\w*|describe\w*|read\w*)\b")
TALK_CUE = re.compile(r"\b(?:message\w*|ask\w*|chat\w*|question\w*|send\w*|reply|replies|tell\w*|talk\w*|contact\w*|notify)\b")
PERSON_CUE = re.compile(r"\b(?:shopper|customer|buyer|client|user|patron)s?\b")
BAG_CUE = re.compile(r"\b(?:cart|carts|basket|baskets|bag|bags|trolley|tote)\b")
HOLD_CUE = re.compile(r"\b(?:hold|holds|holding|reserve|reserves|reserving|reservation|set aside|sets aside|put aside|keep aside|"
                      r"earmark\w*|pin|pins|pinned|save for later|lock in|locks in|shortlist\w*)\b")
COMPARE_CUE = re.compile(r"\b(?:compar\w*|side by side|line ?up|lines up|lined up|versus|contrast\w*|weigh up|stack up)\b")
REPORT_CUE = re.compile(r"\b(?:report|reports|reporting|account|accounts|statement|summary|summaries|summari[sz]\w*|recap|"
                        r"ledger|audit|tally|spending|spend)\b")
VIEW_CUE = re.compile(r"\b(?:show|shows|showing|view|views|display|displays|overview|look|page|card|panel|info|information|"
                      r"details?|describe\w*|preview|open\w*)\b")
BUY_CUE = re.compile(r"\b(?:check ?out|checks out|purchas\w*|buy|buys|pay|pays|settle)\b|\b(?:place|places|submit|submits|complete|"
                     r"completes|finali[sz]e\w*|confirm\w*)\b.{0,16}\border\b")
END_CUE = re.compile(r"\b(?:declin\w*|abstain\w*|give up|gives up|no purchase|without (?:buying|purchas\w*|an order|ordering)|"
                     r"abandon\w*|walk away|refuse to (?:buy|order|purchase)|nothing (?:fits|suitable|acceptable)|close out|"
                     r"end (?:the )?(?:errand|task|session) without)\b")
# Fault reading shared by every caller: "write" (cart, hold), "read" (lookups, compare, report), "talk" (messages).
FAULT_KEY = re.compile(r"\b(?:error|errors|err|issue|issues|fault|faults|failure|failures|failed|fail|problem|problems|denied|"
                       r"rejected|refused|refusal|invalid|exception)\b")
GOOD_FLAG = re.compile(r"\b(?:accepted|accept|ok|okay|success|successful|succeeded|done|added|removed|placed|confirmed|valid|"
                       r"held|reserved|saved|compared|stored)$")
BAD_TEXT = (r"\b(?:error|issue|fail|failed|fails|failure|fault|invalid|not found|no match|denied|rejected|refused|cannot|can't|"
            r"unable|could ?n[o']t|too many|too few|at least|at most|exceeds?|exceeded|unknown)\b")
CODE_ONLY = re.compile(r"^[A-Za-z]{1,4}[-_ ]?\d{1,4}$")
STATUS_KEY = re.compile(r"\b(?:status|code|result|outcome|state|reason|text|message|detail)\b")
# A request that says two wishes may not both be met, or asks the assistant to consult the shopper first.
TRADE_CUE = re.compile(r"\b(?:can'?t|cannot|can not|couldn'?t|unable to|won'?t)\s+(?:have|get|find|do|be)\s+both\b|\bnot both\b|"
                       r"\bboth (?:aren'?t|are not|can'?t|cannot|won'?t)\b|\bone or the other\b|\bif (?:that|this|it)(?:'s| is)? not "
                       r"(?:possible|available|doable)\b|\bif (?:that|this|it) isn'?t (?:possible|available|doable)\b|\btrade.?offs?\b|"
                       r"\bwhich (?:one )?matters more\b|\bif it comes down to\b|\bif (?:none|no listing|nothing) (?:has|have|offers?) both\b")
CHECK_BACK_CUE = re.compile(r"\b(?:ask|check with|confirm with|consult|run (?:it|that|this|them) by)\s+me\b|\blet me (?:choose|pick|decide)\b")
# A fault about the call itself (its tool or arguments), not about what the store holds: read over the fault text in words.
ARG_FAULT = re.compile(r"\b(?:invalid (?:arg\w*|param\w*|field\w*|input\w*|value\w*|json|call|request|tool)|unknown (?:tool|function|"
                       r"arg\w*|param\w*|field\w*|action)|missing (?:required )?(?:arg\w*|param\w*|field\w*|value\w*)|no such "
                       r"(?:tool|function|action)|not (?:a )?valid (?:json|arg\w*|param\w*)|validation)\b")
SPENT_FAULT = re.compile(r"^(?!.*\b(?:length|long|characters?|chars|words|size)\b)(?=.*\b(?:budget|limit|quota|allowance|maximum|max|"
                         r"no more)\b)(?=.*\b(?:exceed\w*|used up|reached|exhausted|spent|over|no more|run out)\b)|"
                         r"\btoo many (?:messages|questions)\b", re.S)


def fault_words(fault: str) -> str:
    return re.sub(r"[\s_\-]+", " ", fault.lower())


# ================================================================================================================ helpers
def as_text(piece: Any, ceiling: int = RECORD_CHARS) -> str:
    try:
        text_body = json.dumps(piece, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text_body = str(piece)
    return text_body if len(text_body) <= ceiling else text_body[:ceiling] + "...(cut)"


def clipped(piece: Any, chars: int, lvl: int = 0) -> Any:
    """The value with every string cut to `chars` (nesting past 8 levels dropped)."""
    if lvl > 8:
        return None
    if isinstance(piece, str):
        return piece if len(piece) <= chars else piece[:chars] + "..."
    if isinstance(piece, dict):
        return {fkey: clipped(val, chars, lvl + 1) for fkey, val in piece.items()}
    if isinstance(piece, list):
        return [clipped(val, chars, lvl + 1) for val in piece]
    return piece


def rows_trimmed(piece: Any, chars: int, in_row: bool = False, lvl: int = 0) -> Any:
    """The value with the strings inside listing rows (dicts that are items of a list) cut to `chars`; strings
    outside rows, such as a notice wrapped around the results, stay whole."""
    if lvl > 8:
        return piece
    if isinstance(piece, str):
        return piece if not in_row or len(piece) <= chars else piece[:chars] + "..."
    if isinstance(piece, dict):
        return {fkey: rows_trimmed(val, chars, in_row, lvl + 1) for fkey, val in piece.items()}
    if isinstance(piece, list):
        return [rows_trimmed(val, chars, in_row or isinstance(val, dict), lvl + 1) for val in piece]
    return piece


def json_object(text_body: Any) -> dict[str, Any] | None:
    """The first JSON object in a model answer (code fences and surrounding prose allowed)."""
    if not isinstance(text_body, str):
        return None
    envelope = text_body.strip()
    if envelope.startswith("```"):
        envelope = re.sub(r"^```[A-Za-z]*", "", envelope).strip()
        if envelope.endswith("```"):
            envelope = envelope[:-3].strip()
    for candidate in (envelope, envelope[envelope.find("{"): envelope.rfind("}") + 1] if "{" in envelope else ""):
        if not candidate:
            continue
        try:
            piece = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(piece, dict):
            return piece
    return None


def is_number(piece: Any) -> bool:
    return isinstance(piece, (int, float)) and not isinstance(piece, bool)


def long_strings(piece: Any, collected: list[str], lvl: int = 0) -> None:
    """Strings of four or more words inside a string or a list of them (shopper lines, notices)."""
    if lvl > 4:
        return
    if isinstance(piece, str):
        if len(piece.split()) >= 4:
            collected.append(piece)
    elif isinstance(piece, list):
        for thing in piece[:10]:
            long_strings(thing, collected, lvl + 1)


def walk_dicts(piece: Any, visit, lvl: int = 0, ceiling: int = 6) -> None:
    """Calls visit(d) on every dict inside value, outermost first."""
    if lvl > ceiling:
        return
    if isinstance(piece, dict):
        visit(piece)
        for thing in piece.values():
            if isinstance(thing, (dict, list)):
                walk_dicts(thing, visit, lvl + 1, ceiling)
    elif isinstance(piece, list):
        for thing in piece:
            walk_dicts(thing, visit, lvl + 1, ceiling)


def word_stems(text: str) -> set[str]:
    """The words of a product or category name, three letters or more, without plural endings."""
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]{3,}", str(text).lower())}


def words_meet(one: set[str], other: set[str]) -> int:
    """How many words of the first set are in the second, a shared start of four letters counting as the same word."""
    return sum(1 for a in one if a in other or (len(a) >= 4 and any(b[:4] == a[:4] for b in other if len(b) >= 4)))


def spoken(name: Any) -> str:
    """An identifier as lower-case words: 'putInBasket' -> 'put in basket', 'item_id' -> 'item id'."""
    text_body = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name or ""))
    text_body = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text_body)
    return re.sub(r"[\s_\-.:/]+", " ", text_body).strip().lower()


def kinds(pspec: Any) -> list[str]:
    if not isinstance(pspec, dict):
        return []
    found = pspec.get("type")
    return [found] if isinstance(found, str) else [k for k in (found or []) if isinstance(k, str)]


def fault_of(obs: Any, mode: str = "read") -> str:
    """Why a tool result is a failure ('' when it is not): an error-like key with a value, a success flag set to
    false, or a short status text that is only a code or reads as a refusal. 'read' also fails an empty result;
    'talk' (a shopper message) fails only on an explicit error key."""
    if obs in (None, "", [], {}):
        return "empty result" if mode == "read" else ""
    if isinstance(obs, str):
        said = obs.strip()
        if mode != "talk" and len(said) <= 240 and (CODE_ONLY.match(said) or re.search(BAD_TEXT, said.lower())):
            return said[:160]
        return ""
    if not isinstance(obs, dict):
        return ""
    for key, val in obs.items():
        words = spoken(key)
        if val is False and GOOD_FLAG.search(words):
            return f"{key} is false"
        if val not in (None, False, "", [], {}, 0) and FAULT_KEY.search(words) and (len(obs) <= 4 or len(words.split()) <= 2):
            return f"{key}: {as_text(val, 140)}"
        if mode != "talk" and isinstance(val, str) and len(obs) <= 3 and STATUS_KEY.search(words) \
                and (CODE_ONLY.match(val.strip()) or (len(val) <= 240 and re.search(BAD_TEXT, val.lower()))):
            return f"{key}: {val[:140]}"
    return ""


def edit_distance(a: str, b: str, cap: int = 3) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


ID_NAME = re.compile(r"\b(?:id|ids|ident\w*|item|items|product|products|article|articles|listing|listings|sku|skus|ref|refs|"
                     r"code|target|offer|offers|option|options|candidate|candidates|choice|choices|entry|entries|pick|picks|picked)\b")


def id_param(name: Any, pspec: Any) -> bool:
    """A scalar argument that names a listing: text (or a whole number) whose name or description speaks of ids or items."""
    if not isinstance(pspec, dict) or "enum" in pspec or not set(kinds(pspec) or ["string"]) & {"string", "integer"}:
        return False
    return bool(ID_NAME.search(spoken(name) + " " + str(pspec.get("description") or "").lower()[:120]))


def list_params(jschema: Any) -> list[tuple[str, int, int]]:
    """(name, minItems, maxItems or 0) of each array argument holding plain ids."""
    found = []
    props = (jschema or {}).get("properties") if isinstance(jschema, dict) else None
    for key, pspec in (props if isinstance(props, dict) else {}).items():
        if "array" not in kinds(pspec):
            continue
        inner = pspec.get("items") if isinstance(pspec.get("items"), dict) else {}
        if "enum" in inner or (kinds(inner) and not set(kinds(inner)) & {"string", "integer"}):
            continue
        low, high = pspec.get("minItems"), pspec.get("maxItems")
        found.append((key, int(low) if is_number(low) else 0, int(high) if is_number(high) else 0))
    return found


def enum_modes(tdef: dict[str, Any], yes_words: str, no_words: str) -> tuple[str, Any, Any] | None:
    """(argument, yes value, no value) for a tool whose enum argument chooses between two actions (order or decline,
    put or take). Each value is read by its own words and by what the tool's and the argument's descriptions say
    after it; in a two-value enum, one value that clearly reads one way makes the other the opposite."""
    jschema = tdef.get("parameters") if isinstance(tdef.get("parameters"), dict) else {}
    props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
    for key, pspec in props.items():
        options = pspec.get("enum") if isinstance(pspec, dict) else None
        if not isinstance(options, list) or len(options) < 2 or not all(isinstance(o, str) for o in options):
            continue
        told = (str(tdef.get("description") or "") + " " + str(pspec.get("description") or "")).lower()

        def reads(opt: str, rx: str) -> bool:
            if re.search(rx, spoken(opt)) or re.search(rx, opt.lower()):
                return True
            for hit in re.finditer(r"['\"`(]?\b" + re.escape(opt.lower()) + r"\b['\"`)]?\s*(?:=|:|-|means|to|will|for)?\s*([^,;.|]{0,60})", told):
                if re.search(rx, hit.group(1)):
                    return True
            return False
        no_side = [o for o in options if reads(o, no_words)]
        yes_side = [o for o in options if reads(o, yes_words) and o not in no_side]
        yes = yes_side[0] if len(yes_side) == 1 else None
        no = no_side[0] if len(no_side) == 1 else None
        if len(options) == 2:
            if yes is not None and no is None:
                no = next(o for o in options if o != yes)
            elif no is not None and yes is None:
                yes = next(o for o in options if o != no)
        if yes is not None and no is not None and yes != no:
            return key, yes, no
    return None


_ROW_FIELD = re.compile('(?:^|(?<= \\| ))([A-Za-z_][A-Za-z0-9_]{0,40}): ')
_ROW_WORDS = {'true': True, 'false': False, 'none': None, 'null': None}


def _row_value(field: str, text: str) -> Any:
    """A field of a listing written as text, as data: a flag or none, a whole number, or "name=value; name=value" pairs."""
    word = text.strip().lower()
    if word in _ROW_WORDS:
        return _ROW_WORDS[word]
    if re.fullmatch('-?(?:0|[1-9][0-9]{0,15})', text):
        return int(text)
    pairs = [part.partition('=') for part in text.split('; ')]
    if pairs and pairs[0][1] and all(0 < len(key) <= 48 and not re.search('[,.:|]', key) for key, eq, _ in pairs if eq):
        facts: dict[str, str] = {}
        last = ''
        for key, eq, said in pairs:
            if eq and key not in facts:
                facts[key] = said
                last = key
            else:
                facts[last] += '; ' + key + eq + said
        if len(facts) >= 2 or re.search('fact|spec|attribute|feature|detail', field, re.I):
            return facts
    return text


def _from_row(text: Any) -> dict[str, Any] | None:
    """A listing the store wrote as one line of "name: value" fields joined by " | " (race 15's stores), as a record; None
    for other text and for a line without a number or a flag in it (a note in that shape is not a listing). A field name
    met a second time belongs to the value before it."""
    if not isinstance(text, str) or ' | ' not in text:
        return None
    marks = list(_ROW_FIELD.finditer(text))
    if len(marks) < 3 or marks[0].start() != 0:
        return None
    fields: dict[str, str] = {}
    order: list[str] = []
    for index, mark in enumerate(marks):
        stop = marks[index + 1].start() - 3 if index + 1 < len(marks) else len(text)
        name, value = mark.group(1), text[mark.end():stop]
        if name in fields:
            fields[order[-1]] += ' | ' + name + ': ' + value
        else:
            fields[name] = value
            order.append(name)
    if len(fields) < 3:
        return None
    record = {name: _row_value(name, value) for name, value in fields.items()}
    return record if any(isinstance(v, (bool, int)) for v in record.values()) else None


# a reply part whose text is a message about the call, never a listing: kept as it came, under its own key
_WRAPPER_KEY = re.compile(r"refus|reject|error|notice|message|change", re.I)


def _records_of(node: Any, written: dict[int, tuple[Any, Any]] | None = None, depth: int = 0) -> Any:
    """A store reply with each listing written as text read as a record, wherever it sits: an entry of a list, or a
    reply part holding that one line. Everything else stays as it came. `written` keeps, for each new record, the
    store's own form of it (record id -> (record, form)), so what the model is shown can stay the store's text."""
    def keep(record: dict[str, Any] | None, form: Any) -> Any:
        if record is None:
            return form
        if written is not None:
            written[id(record)] = (record, form)
        return record

    if depth > 6:
        return node
    if isinstance(node, list):
        return [keep(_from_row(entry), entry) if isinstance(entry, str) else _records_of(entry, written, depth + 1) for entry in node]
    if isinstance(node, dict):
        values = list(node.values())
        if len(values) == 1 and isinstance(values[0], str) and not _WRAPPER_KEY.search(str(next(iter(node)))):
            return keep(_from_row(values[0]), node)
        return {key: value if isinstance(value, str) and _WRAPPER_KEY.search(str(key)) else _records_of(value, written, depth + 1)
                for key, value in node.items()}
    if isinstance(node, str):
        return keep(_from_row(node), node)
    return node


# a key that holds a listing's id without saying "id", read as words (spoken): its last word an id word or its first
# "ref" ("ref_ref", "listingCode", "item_number"), and never a name, title, model, brand or description
_ID_WORD_END = re.compile(r"(?:^| )(?:ref|code|sku|number|no|id|key)$|^ref(?: |$)")
_NOT_AN_ID = re.compile(r"\b(?:name|title|model|brand|desc\w*)\b")
# a key that says "id" as a word ("id", "item_id", "listingIds", "identifier"; not "width", "provider" or "video")
_ID_KEY = re.compile(r"(?:^|[ _])ids?(?:$|[ _])|ident")


def _listing_ids(node: Any, depth: int = 0) -> Any:
    """The reply with an "id" field put first in each record whose listing id sits under a key that does not say "id"
    (race 15 wrote "ref_ref"): the code finds a listing's records by their id field."""
    if depth > 6:
        return node
    if isinstance(node, list):
        return [_listing_ids(entry, depth + 1) for entry in node]
    if not isinstance(node, dict):
        return node
    out = {key: _listing_ids(value, depth + 1) for key, value in node.items()}
    if any(_ID_KEY.search(spoken(key)) for key in out):
        return out
    # (the first such field in the line's own order: every text-line store seen writes the id first)
    ref = next((value for key, value in out.items() if isinstance(value, str) and _ID_WORD_END.search(spoken(key))
                and not _NOT_AN_ID.search(spoken(key)) and re.fullmatch(ID_LIKE, value) and any(c.isdigit() for c in value)), None)
    return {"id": ref, **out} if ref else out


def _as_read(obs: Any) -> Any:
    """A store reply as the code reads it: listings written as text are records, each with its id field."""
    return _listing_ids(_records_of(obs))


def _short_rows(node: Any, chars: int, depth: int = 0) -> Any:
    """A browse result as the actor reads it: every listing written as a text line keeps all its fields, each value
    cut to `chars` (the records keep the whole line for the checks)."""
    if depth > 6:
        return node
    if isinstance(node, list):
        return [" | ".join(part if len(part) <= chars else part[:chars] + "..." for part in entry.split(" | "))
                if isinstance(entry, str) and _from_row(entry) is not None else _short_rows(entry, chars, depth + 1) for entry in node]
    if isinstance(node, dict):
        return {key: _short_rows(value, chars, depth + 1) for key, value in node.items()}
    return node


# ================================================================================================================ schema
class ArgCheck:
    """Coerces model-written arguments toward a JSON schema and lists what still breaks it."""

    TYPES = {
        "string": lambda val: isinstance(val, str),
        "integer": lambda val: isinstance(val, int) and not isinstance(val, bool),
        "number": is_number,
        "boolean": lambda val: isinstance(val, bool),
        "object": lambda val: isinstance(val, dict),
        "array": lambda val: isinstance(val, list),
        "null": lambda val: val is None,
    }

    @classmethod
    def type_list(cls, jschema: dict[str, Any]) -> list[str]:
        type_list = jschema.get("type")
        return [type_list] if isinstance(type_list, str) else list(type_list or [])

    @classmethod
    def conforms(cls, piece: Any, ktype: str) -> bool:
        test = cls.TYPES.get(ktype)
        return test(piece) if test else True

    @classmethod
    def convert(cls, piece: Any, type_list: list[str]) -> Any:
        if not type_list:
            return piece
        if any(cls.conforms(piece, fkey) for fkey in type_list):
            if "integer" in type_list and isinstance(piece, float) and piece.is_integer():
                return int(piece)
            return piece
        for ktype in type_list:
            if ktype == "integer":
                if isinstance(piece, float) and piece.is_integer():
                    return int(piece)
                if isinstance(piece, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", piece):
                    return int(float(piece))
            elif ktype == "number" and isinstance(piece, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", piece):
                num = float(piece)
                return int(num) if num.is_integer() else num
            elif ktype == "string" and is_number(piece):
                return str(piece)
            elif ktype == "boolean" and isinstance(piece, str) and piece.lower() in ("true", "false"):
                return piece.lower() == "true"
        return piece

    @classmethod
    def execute(cls, piece: Any, jschema: Any, where: str = "args") -> tuple[Any, list[str]]:
        if not isinstance(jschema, dict):
            return piece, []
        type_list = cls.type_list(jschema)
        piece = cls.convert(piece, type_list)
        if type_list and not any(cls.conforms(piece, fkey) for fkey in type_list):
            return piece, [f"{where}: expected {'/'.join(type_list)}"]
        faults: list[str] = []
        for keyword, rule_fn in cls.RULES:
            if keyword in jschema:
                piece, matched = rule_fn(piece, jschema, where)
                faults += matched
        return piece, faults

    @staticmethod
    def _enum(piece, jschema, where):
        options = jschema["enum"]
        if piece in options:
            return piece, []
        if isinstance(piece, str):
            for opt in options:
                if isinstance(opt, str) and opt.lower() == piece.strip().lower():
                    return opt, []
        return piece, [f"{where}: must be one of {options}"]

    @staticmethod
    def _const(piece, jschema, where):
        return piece, ([] if piece == jschema["const"] else [f"{where}: must equal {jschema['const']!r}"])

    @staticmethod
    def _min_length(piece, jschema, where):
        if isinstance(piece, str) and len(piece) < jschema["minLength"]:
            return piece, [f"{where}: shorter than {jschema['minLength']} characters"]
        return piece, []

    @staticmethod
    def _max_length(piece, jschema, where):
        return (piece[: jschema["maxLength"]] if isinstance(piece, str) else piece), []

    @staticmethod
    def _bound(call_name, test, words):
        def rule_fn(piece, jschema, where):
            ceiling = jschema[call_name]
            if is_number(piece) and is_number(ceiling) and test(piece, ceiling):
                return piece, [f"{where}: {words} {ceiling}"]
            return piece, []
        return rule_fn

    @staticmethod
    def _min_items(piece, jschema, where):
        if isinstance(piece, list) and len(piece) < jschema["minItems"]:
            return piece, [f"{where}: needs at least {jschema['minItems']} items"]
        return piece, []

    @staticmethod
    def _max_items(piece, jschema, where):
        if isinstance(piece, list) and len(piece) > jschema["maxItems"]:
            return piece, [f"{where}: allows at most {jschema['maxItems']} items"]
        return piece, []

    @classmethod
    def _items(cls, piece, jschema, where):
        if not isinstance(piece, list) or not isinstance(jschema["items"], dict):
            return piece, []
        faults, mended = [], []
        for idx, thing in enumerate(piece):
            thing, matched = cls.execute(thing, jschema["items"], f"{where}[{idx}]")
            mended.append(thing)
            faults += matched
        return mended, faults

    @classmethod
    def _properties(cls, piece, jschema, where):
        if not isinstance(piece, dict):
            return piece, []
        prop_specs = jschema.get("properties") or {}
        faults = [f"{where}.{call_name}: required" for call_name in jschema.get("required") or [] if call_name not in piece]
        if jschema.get("additionalProperties") is False:
            faults += [f"{where}.{call_name}: not an allowed argument" for call_name in piece if call_name not in prop_specs]
        mended = {}
        for call_name, thing in piece.items():
            if call_name in prop_specs:
                thing, matched = cls.execute(thing, prop_specs[call_name], f"{where}.{call_name}")
                faults += matched
            mended[call_name] = thing
        return mended, faults


ArgCheck.RULES = [
    ("enum", ArgCheck._enum),
    ("const", ArgCheck._const),
    ("minLength", ArgCheck._min_length),
    ("maxLength", ArgCheck._max_length),
    ("minimum", ArgCheck._bound("minimum", lambda val, lim: val < lim, "below minimum")),
    ("maximum", ArgCheck._bound("maximum", lambda val, lim: val > lim, "above maximum")),
    ("exclusiveMinimum", ArgCheck._bound("exclusiveMinimum", lambda val, lim: val <= lim, "must be above")),
    ("exclusiveMaximum", ArgCheck._bound("exclusiveMaximum", lambda val, lim: val >= lim, "must be below")),
    ("minItems", ArgCheck._min_items),
    ("maxItems", ArgCheck._max_items),
    ("items", ArgCheck._items),
    ("type", lambda val, s, p: ArgCheck._properties(val, s, p) if isinstance(val, dict) else (val, [])),
]


# ================================================================================================================ wire
class Wire:
    """Model calls and world calls through the sandbox proxy, bounded by the errand's clock and its spend.

    A world call carries exactly one action. When its reply is lost (timeout, 5xx, 504) the same envelope is sent
    again after a pause; if it still fails, the step stays pending and is replayed before any other action, so no
    different action ever lands on a used step (the world answers a replay from its cache)."""

    def __init__(self, bind: dict[str, Any]):
        self.t0 = monotonic()
        self.bind = dict(bind)
        self.model_client = ProxyClient(timeout=40, max_retries=1, retry_delay=1)
        self.world = ProxyClient(timeout=100, max_retries=1, retry_delay=1)
        self.spend = 0.0
        self.dearest: dict[str, float] = {}         # model -> the dearest call it made in this task
        self.tick = 0
        self.done = False
        self.broke = False                          # the world can no longer be reached (credit gone, repeated conflicts)
        self.credit_out = False                     # inference said 402: no more model calls in this task
        self.barred: set[str] = set()               # models the proxy refused (403)
        self.conflicts = 0                          # 409 answers that were not about a finished episode
        self.pending: dict[str, Any] | None = None  # the envelope of a step whose reply never came
        self.orphan: tuple[dict[str, Any], dict[str, Any]] | None = None   # (action, reply) replayed before a newer action
        self.retryable = False                      # the last failed post may be sent again on the same step
        self.calls = 0
        self.answered_by = ""                       # the model whose answer the last consult returned
        self.log: list[dict[str, Any]] = []

    def spare(self) -> float:
        return DEADLINE - (monotonic() - self.t0)

    def hard_spare(self) -> float:
        return SANDBOX_END - (monotonic() - self.t0)

    def guess_cost(self, mdl: str) -> float:
        """What one more call of a model may cost: its dearest call in this task, else the dearest call of any model
        (DEAR_GUESS times that for a dear model)."""
        if mdl in self.dearest:
            return self.dearest[mdl]
        top = max(self.dearest.values(), default=0.0)
        return top * DEAR_GUESS if mdl in DEAR else top

    def affordable(self, mdl: str) -> bool:
        """One more call of the model keeps the task's spend at or under SPEND_STOP."""
        return self.spend < SPEND_STOP and self.spend + self.guess_cost(mdl) <= SPEND_STOP

    def lineup(self, model_list: list[str], actor: bool, roof: float | None = None, firm: bool = False) -> list[str]:
        """The models to try, in order: the first one, then its fallbacks while the spend allows them; without a
        fallback the first model gets one more try. A helper stops at SPEND_HARD unless its call names a higher
        roof (reading what the shopper said must not stop before the actor does). A firm list (a decision the finish
        waits on) keeps its order and tries each model once, whatever the spend tier below the roof. A model whose
        next call could carry the spend past SPEND_STOP is left out."""
        if self.credit_out or self.spend >= SPEND_STOP or (not actor and self.spend >= (roof or SPEND_HARD)):
            return []
        usable = [m for m in model_list if m not in self.barred and self.affordable(m)]
        if not usable:
            return []
        if firm:
            return usable
        if self.spend >= SPEND_DEAR:
            usable.sort(key=lambda m: m in DEAR)
        rest = usable[1:] if self.spend < (SPEND_HARD if actor else SPEND_SOFT) else []
        return usable[:1] + (rest or usable[:1])

    def query_model(self, msgs: list[dict[str, Any]], model_list: list[str], token_cap: int,
            toolset: list[dict[str, Any]] | None = None, actor: bool = False, roof: float | None = None) -> dict[str, Any] | None:
        tries = self.lineup(model_list, actor, roof)
        for mdl in tries[:1] if len(model_list) == 1 else tries:
            spare = self.spare()
            if spare < MODEL_FLOOR or mdl in self.barred or self.credit_out:
                return None
            if not self.affordable(mdl):
                continue
            self.model_client.timeout = int(max(6, min(40, spare - LAST_CALL - 5)))
            envelope: dict[str, Any] = {"model": mdl, "messages": self.cached(msgs, mdl), "temperature": 0,
                                    "max_tokens": token_cap}
            if toolset:
                envelope["tools"] = toolset
                envelope["tool_choice"] = "auto"
            self.calls += 1
            answer = self.model_client.post_verbose("/inference/chat/completions", envelope)
            data = answer.data
            if isinstance(data, dict) and data.get("choices"):
                use_info = data.get("usage") or {}
                if is_number(use_info.get("cost")) and use_info["cost"] > 0:
                    self.spend += float(use_info["cost"])
                    self.dearest[mdl] = max(self.dearest.get(mdl, 0.0), float(use_info["cost"]))
                choice = data["choices"][0] if isinstance(data["choices"], list) and data["choices"] else {}
                msg = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(msg, dict) and (msg.get("content") or msg.get("tool_calls")):
                    return msg
            status = (answer.error or {}).get("status")
            if status == 402:
                self.credit_out = True
            elif status == 403:
                self.barred.add(mdl)
            elif status == 429 and self.spare() > MODEL_FLOOR + 10:
                sleep(2.0)       # (the proxy's shared rate limit: a short pause before the next try)
            self.log.append({"kind": "inference_failure", "model": mdl, "status": status,
                             "error": str(answer.error)[:300] if answer.error else "empty"})
        return None

    @staticmethod
    def cached(msgs: list[dict[str, Any]], mdl: str) -> list[dict[str, Any]]:
        """For Anthropic models the system text and the newest message are marked for the prompt cache."""
        if not mdl.startswith("anthropic/") or len(msgs) <= 2:
            return msgs
        marked = [dict(m) for m in msgs]
        for idx in (0, len(marked) - 1):
            content = marked[idx].get("content")
            if isinstance(content, str) and content:
                marked[idx]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        return marked

    def consult(self, ask_text: str, token_cap: int, model_list: list[str], roof: float | None = None,
                firm: bool = False) -> dict[str, Any] | None:
        """One helper question answered as a JSON object; an unreadable answer moves on to the next model. The model
        that answered is kept in answered_by ('' when none did)."""
        self.answered_by = ""
        for mdl in self.lineup(model_list, False, roof, firm):
            msg = self.query_model([{"role": "user", "content": ask_text}], [mdl], token_cap, roof=roof)
            answer = json_object((msg or {}).get("content") or "")
            if answer is not None:
                self.answered_by = mdl
            if answer is not None or self.spare() < MODEL_FLOOR or self.credit_out:
                return answer
        return None

    def step(self, action: dict[str, Any]) -> dict[str, Any] | None:
        """Sends one action on the next step; returns the world's reply, or None when none could be had."""
        if self.done or self.broke:
            return None
        if self.pending is not None:
            held = self.pending
            reply = self.post_step(held)
            if reply is None:
                return None
            self.pending = None
            self.take(reply)
            if held["action"] == action:
                return reply
            self.orphan = (held["action"], reply)
            if self.done:
                return None
        body = {**self.bind, "step": self.tick + 1, "action": action}
        reply = self.post_step(body)
        if reply is None:
            if self.retryable and not self.done and not self.broke:
                self.pending = body
            return None
        self.take(reply)
        return reply

    def take(self, reply: dict[str, Any]) -> None:
        told = reply.get("step")
        self.tick = told if isinstance(told, int) and not isinstance(told, bool) and told >= self.tick else self.tick + 1
        if reply.get("finished"):
            self.done = True

    def post_step(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Posts one envelope, again after 1, 2 and 4 s while the reply is lost and the sandbox clock allows."""
        self.retryable = False
        pause = 1.0
        for _ in range(4):
            room = self.hard_spare() - 2
            if room < 5:
                self.retryable = True
                return None
            self.world.timeout = int(min(100, room))
            answer = self.world.post_verbose("/environment/call", body)
            if isinstance(answer.data, dict) and ("observation" in answer.data or "finished" in answer.data):
                return answer.data
            error = answer.error or {}
            status, text_body = error.get("status"), str(error.get("body") or "")
            self.log.append({"kind": "env_failure", "step": body.get("step"), "status": status, "body": text_body[:300]})
            if status == 409:
                if re.search(r"finished|is over|has ended", text_body):
                    self.done = True
                    return None
                self.conflicts += 1
                expected = re.search(r"expected (\d+)", text_body)
                self.tick = int(expected.group(1)) - 1 if expected else self.tick + 1
                self.broke = self.broke or self.conflicts >= 3
                return None
            if status == 402:
                self.broke = True
                return None
            if status in (400, 401, 403, 404, 422):
                return None
            self.retryable = True
            if self.hard_spare() < pause + 10:
                return None
            sleep(pause)
            pause *= 2
        return None


# ================================================================================================================ tool roles
class ToolMap:
    """Which tool does what, read from the tools' names, descriptions and argument schemas. The reader's naming is
    kept only where the schema agrees with it; a role it left out is filled from the schema, so a finisher exists
    before the first step even when the reading failed. Also resolves the enum argument that switches a finishing
    tool between ordering and declining, and a cart tool between putting in and taking out."""

    ROLES = ("search", "details", "view", "compare", "hold", "report", "message", "cart", "cart_view", "order", "decline")
    END_WORDS = re.compile(r"\b(?:finish\w*|conclud\w*|end|ends|ending|final\w*|close|closes|wrap\w* up|complete\w*|settle\w*)\b")

    def __init__(self, toolset: list[dict[str, Any]]):
        self.defs: dict[str, dict[str, Any]] = {}
        for tdef in toolset:
            fn = tdef.get("function") if isinstance(tdef, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                self.defs[fn["name"]] = fn
        self.roles: dict[str, str] = {}
        self.finish_modes: tuple[str, Any, Any] | None = None
        self.cart_modes: tuple[str, Any, Any] | None = None
        self.notes: list[str] = []

    # -- schema reads
    def props(self, name: str) -> dict[str, Any]:
        jschema = (self.defs.get(name) or {}).get("parameters")
        found = jschema.get("properties") if isinstance(jschema, dict) else None
        return found if isinstance(found, dict) else {}

    def required(self, name: str) -> list[str]:
        jschema = (self.defs.get(name) or {}).get("parameters")
        found = jschema.get("required") if isinstance(jschema, dict) else None
        return [k for k in found if isinstance(k, str)] if isinstance(found, list) else []

    def said(self, name: str) -> str:
        return (spoken(name) + " " + str((self.defs.get(name) or {}).get("description") or "")).lower()

    def ids(self, name: str) -> list[str]:
        return [k for k, p in self.props(name).items() if id_param(k, p)]

    def free_texts(self, name: str) -> list[str]:
        return [k for k, p in self.props(name).items() if isinstance(p, dict) and "string" in (kinds(p) or ["string"])
                and "enum" not in p and not id_param(k, p)]

    def id_key(self, name: str | None) -> str | None:
        """The argument that names the listing for a single-listing tool."""
        if not name:
            return None
        ids = self.ids(name)
        req = self.required(name)
        pick = next((k for k in ids if k in req), ids[0] if ids else None)
        if pick:
            return pick
        strings = [k for k, p in self.props(name).items() if isinstance(p, dict) and "string" in (kinds(p) or ["string"]) and "enum" not in p]
        return strings[0] if len(strings) == 1 else None

    def count_key(self, name: str | None) -> str | None:
        props = self.props(name or "")
        nums = [k for k, p in props.items() if isinstance(p, dict) and set(kinds(p)) & {"integer", "number"} and "enum" not in p]
        return next((k for k in nums if re.search(COUNT_KEY, spoken(k).replace(" ", "_"))), nums[0] if len(nums) == 1 else None)

    def compare_shape(self) -> tuple[str, Any, int, int] | None:
        """('list', argument, minItems, maxItems or 0) or ('pair', [arguments], 2, count) for the compare tool."""
        name = self.roles.get("compare")
        if not name:
            return None
        lists = list_params(self.defs[name].get("parameters"))
        if lists:
            key, low, high = next((x for x in lists if x[0] in self.required(name)), lists[0])
            return "list", key, low, high
        ids = self.ids(name)
        return ("pair", ids[:4], 2, len(ids[:4])) if len(ids) >= 2 else None

    # -- agreement between a role and a tool's schema
    def agrees(self, role: str, name: str) -> bool:
        if name not in self.defs:
            return False
        req_ids = [k for k in self.ids(name) if k in self.required(name)]
        props = self.props(name)
        if role in ("details", "view", "hold"):
            return bool(self.id_key(name)) and not list_params(self.defs[name].get("parameters"))
        if role == "compare":
            return bool(list_params(self.defs[name].get("parameters"))) or len(self.ids(name)) >= 2
        if role in ("report", "cart_view"):
            return not req_ids
        if role == "message":
            return bool(self.free_texts(name) or [k for k, p in props.items() if isinstance(p, dict) and "string" in kinds(p)])
        if role == "cart":
            return bool(self.id_key(name) or list_params(self.defs[name].get("parameters")) or
                        any("object" in kinds(p) or "array" in kinds(p) for p in props.values()))
        if role == "decline":
            return not req_ids
        if role == "search":
            return bool(props)
        return True

    # -- guesses from the schema alone
    def guess(self) -> dict[str, str]:
        found: dict[str, str] = {}
        reads: list[str] = []
        for name, fn in self.defs.items():
            said, label = self.said(name), spoken(name)
            props, lists, ids = self.props(name), list_params(fn.get("parameters")), self.ids(name)
            req_ids = [k for k in ids if k in self.required(name)]
            texts = self.free_texts(name)
            finds = bool(FIND_CUE.search(said))
            bags = bool(BAG_CUE.search(said))
            buy_label = bool(BUY_CUE.search(label) or re.search(r"\border\b", label))
            if enum_modes(fn, ORDER_WORDS, DECLINE_WORDS) and not req_ids and not finds \
                    and (buy_label or BUY_CUE.search(said) or END_CUE.search(said) or self.END_WORDS.search(said)):
                found.setdefault("order", name)
                found.setdefault("decline", name)
                continue
            if END_CUE.search(label) or (END_CUE.search(said) and not buy_label and not req_ids and not finds and not bags):
                found.setdefault("decline", name)
                continue
            if (lists or len(ids) >= 2) and COMPARE_CUE.search(said):
                found.setdefault("compare", name)
                continue
            if (buy_label or (BUY_CUE.search(said) and not ids and not bags)) and not finds \
                    and not (TALK_CUE.search(said) and PERSON_CUE.search(said) and texts) and not re.search(PUT_WORDS, label):
                found.setdefault("order", name)
                continue
            if texts and TALK_CUE.search(said) and not (bags or finds) and not ids:
                found.setdefault("message", name)
                continue
            if (ids or lists) and (bags or re.search(PUT_WORDS, label)) and not finds and not HOLD_CUE.search(label):
                found.setdefault("cart", name)
                continue
            if ids and HOLD_CUE.search(said) and not bags:
                found.setdefault("hold", name)
                continue
            if not req_ids and REPORT_CUE.search(label) and not finds:
                found.setdefault("report", name)
                continue
            if not req_ids and bags and not texts:
                found.setdefault("cart_view", name)
                continue
            if not req_ids and REPORT_CUE.search(said) and not finds:
                found.setdefault("report", name)
                continue
            if finds and (texts or props) and not bags:
                found.setdefault("search", name)
                continue
            if self.id_key(name) and not lists and (READ_CUE.search(said) or VIEW_CUE.search(said)) and len(props) <= 3:
                reads.append(name)
        if reads:
            found.setdefault("details", next((r for r in reads if READ_CUE.search(self.said(r))), reads[0]))
            more = [r for r in reads if r != found["details"]]
            if more:
                found.setdefault("view", more[0])
        return found

    def settle(self, reading: dict[str, Any]) -> None:
        """Merges the reader's roles with the schema guesses, then resolves the enum modes."""
        told = reading.get("roles") if isinstance(reading.get("roles"), dict) else {}
        guessed = self.guess()
        roles: dict[str, str] = {}
        for role in self.ROLES:
            name = told.get(role)
            if isinstance(name, str) and name in self.defs and self.agrees(role, name):
                roles[role] = name
            elif guessed.get(role) and self.agrees(role, guessed[role]):
                roles[role] = guessed[role]
                if name:
                    self.notes.append(f"{role}: reader's {name} does not fit its schema; {guessed[role]} used")
        # (a duty tool never doubles as the cart or a finishing tool, and the message tool is not a finisher)
        busy = {roles.get(r) for r in ("cart", "order", "decline")} - {None}
        for role in ("compare", "hold", "report", "view"):
            if roles.get(role) in busy:
                roles.pop(role)
        for role in ("order", "decline"):
            if roles.get(role) and roles.get(role) in (roles.get("message"), roles.get("search")):
                roles.pop(role)
        if roles.get("view") == roles.get("details"):
            roles.pop("view", None)
        if not roles.get("details") and roles.get("view"):
            roles["details"] = roles.pop("view")
        one = roles.get("order") or roles.get("decline")
        if one and enum_modes(self.defs[one], ORDER_WORDS, DECLINE_WORDS):
            roles.setdefault("order", one)
            roles.setdefault("decline", one)
        self.roles = roles
        self.finish_modes = self.modes_of(roles.get("order"), ORDER_WORDS, DECLINE_WORDS, reading.get("finish_modes"), ("order", "decline")) \
            if roles.get("order") and roles.get("order") == roles.get("decline") else None
        self.cart_modes = self.modes_of(roles.get("cart"), PUT_WORDS, TAKE_WORDS, reading.get("cart_modes"), ("put", "take"))

    def modes_of(self, name: str | None, yes_rx: str, no_rx: str, told: Any, keys: tuple[str, str]) -> tuple[str, Any, Any] | None:
        if not name or name not in self.defs:
            return None
        found = enum_modes(self.defs[name], yes_rx, no_rx)
        if found:
            return found
        if isinstance(told, dict):
            key = told.get("argument")
            options = (self.props(name).get(key) or {}).get("enum") if isinstance(key, str) else None
            yes, no = told.get(keys[0]), told.get(keys[1])
            if isinstance(options, list) and yes in options and no in options and yes != no:
                return key, yes, no
        return None


# ================================================================================================================ records
class Records:
    """Every observation the store returned, and the views the checks and the question plan read from them."""

    def __init__(self):
        self.items: list[Any] = []
        self.ids: set[str] = set()
        self.read: set[str] = set()     # ids whose details (a record with a nested property map) were seen
        self.tokens: set[str] = set()   # every one-word value any result showed (the ids a call may name)

    def add(self, obs: Any) -> None:
        self.items.append(obs)

        def visit(d: dict[str, Any]) -> None:
            for key, piece in d.items():
                if isinstance(piece, str) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, piece):
                    self.ids.add(piece)
                if isinstance(piece, str) and len(piece) <= 64 and " " not in piece.strip() and piece.strip():
                    self.tokens.add(piece.strip())
                elif isinstance(piece, int) and not isinstance(piece, bool) and "id" in str(key).lower():
                    self.tokens.add(str(piece))
                elif isinstance(piece, list):
                    self.tokens.update(v.strip() for v in piece if isinstance(v, str) and len(v) <= 64 and " " not in v.strip() and v.strip())
                if isinstance(key, str) and re.search(r"\d", key) and re.fullmatch(ID_LIKE, key):
                    self.tokens.add(key)     # (a listing keyed by its id)
        walk_dicts(obs, visit)

    def known(self, ident: Any) -> bool:
        return str(ident) in self.tokens or str(ident) in self.ids

    def nearest(self, ident: str) -> str | None:
        """The seen id one typo away (two for ids of 10+ characters), when exactly one such id exists."""
        if not isinstance(ident, str) or len(ident) < 4:
            return None
        pool = self.ids or self.tokens
        best = [(edit_distance(ident, k), k) for k in pool if abs(len(k) - len(ident)) <= 2]
        best = sorted(b for b in best if b[0] <= (2 if len(b[1]) >= 10 else 1))
        if not best or (len(best) > 1 and best[1][0] == best[0][0]):
            return None
        return best[0][1]

    def price(self, thing: str) -> float | None:
        """The newest price seen on a record that carries the item's id."""
        found: list[float] = []

        def visit(d: dict[str, Any]) -> None:
            if not any(isinstance(val, str) and val == thing and "id" in str(key).lower() for key, val in d.items()):
                return
            for key, val in d.items():
                if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower()):
                    found.append(float(val))
                    return
        for rec in self.items:
            walk_dicts(rec, visit)
        return found[-1] if found else None

    def about(self, thing: str, hold: int = 4) -> list[str]:
        """The newest records (dicts) that carry the item's id as a value, outermost match only."""
        matched: list[str] = []

        def seek(piece: Any, lvl: int) -> None:
            if lvl > 6 or len(matched) >= 8:
                return
            if isinstance(piece, dict):
                if any(isinstance(val, str) and val == thing for val in piece.values()):
                    text_body = as_text(clipped(piece, 400))[:2500]
                    if text_body not in matched:
                        matched.append(text_body)
                    return
                for val in piece.values():
                    seek(val, lvl + 1)
            elif isinstance(piece, list):
                for val in piece:
                    seek(val, lvl + 1)

        for rec in self.items:
            seek(rec, 0)
        return matched[-hold:]

    def digest(self, char_cap: int = 24000, value_chars: int = 160) -> list[str]:
        """The records, newest first, within a character budget."""
        record_texts, char_total = [], 0
        for rec in reversed(self.items):
            text_body = as_text(clipped(rec, value_chars))
            if char_total + len(text_body) > char_cap:
                continue
            record_texts.append(text_body)
            char_total += len(text_body)
        return record_texts

    def cards(self, char_cap: int = 30000, cheaper_than: str = "") -> list[str]:
        """One line per listing seen: its search row and its details merged (the details win), DETAILS listings first,
        each value clipped; within a character budget. With cheaper_than, only that listing and the ones priced below
        it (or with no price seen) are kept."""
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower()
                          and re.fullmatch(ID_LIKE, val) and val in self.ids), None)
            if not ident or len(d) < 4:
                return
            card = merged.setdefault(ident, {})
            if ident not in order:
                order.append(ident)
            card.update({key: val for key, val in d.items()})
            if any(isinstance(val, dict) and len(val) >= 2 for val in d.values()):
                self.read.add(ident)

        for rec in self.items:
            walk_dicts(rec, visit)

        def price_of(ident: str) -> float | None:
            for key, val in (merged.get(ident) or {}).items():
                if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower()):
                    return float(val)
            return None

        cap_price = price_of(cheaper_than) if cheaper_than else None
        if cap_price is not None:
            order = [i for i in order if i == cheaper_than or price_of(i) is None or price_of(i) < cap_price]
        lines, total = [], 0
        for ident in sorted(order, key=lambda i: (i not in self.read, order.index(i))):
            card = clipped(merged[ident], 160)
            text_body = ("DETAILS " if ident in self.read else "ROW ") + as_text(card)
            if total + len(text_body) > char_cap:
                continue
            lines.append(text_body)
            total += len(text_body)
        return lines

    def rich_reads(self) -> int:
        """Listings read whose record carries a property map of 3+ fields (in flat records, 5+ short text fields)."""
        rich: set[str] = set()

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.read), None)
            if not ident:
                return
            maps = [val for val in d.values() if isinstance(val, dict)]
            if any(len(val) >= 3 for val in maps) or (not maps and sum(1 for val in d.values() if isinstance(val, str) and 0 < len(val) <= 70) >= 5):
                rich.add(ident)
        for rec in self.items:
            walk_dicts(rec, visit)
        return len(rich)

    def prop_menu(self) -> tuple[str, int]:
        """Property names on detail-like records (5+ short string fields), with counts and sample values."""
        tallies: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        sets = 0

        def visit(d: dict[str, Any]) -> None:
            nonlocal sets
            brief_vals = [(fkey, val) for fkey, val in d.items() if isinstance(val, str) and 0 < len(val) <= 70]
            if len(brief_vals) < 5:
                return
            sets += 1
            for key, piece in brief_vals:
                if re.search(r"\d", piece) and re.fullmatch(CODE_LIKE, piece) and ("-" in piece or "_" in piece):
                    continue
                call_name = re.sub(r"[_.]+", " ", str(key)).strip().lower()
                tallies[call_name] = tallies.get(call_name, 0) + 1
                value_counts = examples.setdefault(call_name, [])
                if piece.lower() not in value_counts and len(value_counts) < 5:
                    value_counts.append(piece.lower())

        for rec in self.items:
            walk_dicts(rec, visit)
        top_props = sorted(tallies, key=lambda fkey: -tallies[fkey])[:28]
        return "\n".join(f"- {fkey} (in {tallies[fkey]} listings): " + " | ".join(examples.get(fkey, [])) for fkey in top_props), sets

    def shelves(self) -> dict[str, str]:
        """The category label each listing's rows or records carry, by its id."""
        found: dict[str, str] = {}

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.ids), None)
            label = next((val for key, val in d.items() if isinstance(val, str) and 0 < len(val) <= 60 and "id" not in str(key).lower()
                          and re.search(r"categor|shelf|department|aisle", str(key).lower())), "")
            if ident and label:
                found[ident] = label

        for rec in self.items:
            walk_dicts(rec, visit)
        return found

    def menu_by_shelf(self, products: list[str]) -> str:
        """The properties of the listings read, per category, with how many listings carry each and example values. Long
        text, purchase-variant lists and values different on every listing are left out; so are the categories of no
        requested product when any category is one's (a listing looked up for reference is not for sale to this shopper)."""
        props: dict[str, dict[str, str]] = {}
        flat: dict[str, dict[str, str]] = {}

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.read), None)
            if not ident:
                return
            maps = [val for val in d.values() if isinstance(val, dict) and len(val) >= 3]
            into = props if maps else flat
            for block in maps or [d]:
                for key, piece in block.items():
                    if isinstance(piece, (str, int, float)) and not isinstance(piece, bool) and str(piece).strip() and "id" not in spoken(str(key)).split():
                        into.setdefault(ident, {})[spoken(str(key))] = str(piece).strip().lower()[:70]

        for rec in self.items:
            walk_dicts(rec, visit)
        # a store whose records have no property map: the flat record of a listing is all there is
        for ident, block in flat.items():
            if ident not in props and len(block) >= 5:
                props[ident] = block
        labels = self.shelves()
        groups: dict[str, list[dict[str, str]]] = {}
        for ident, block in props.items():
            groups.setdefault(labels.get(ident, ""), []).append(block)
        # narrowed only when every requested product has a category here: else the planner would never see the others' properties
        matched: dict[str, list[dict[str, str]]] = {}
        hits = 0
        for name in products:
            scores = {label: words_meet(word_stems(name), word_stems(label)) for label in groups if label}
            top = max(scores.values(), default=0)
            hits += bool(top)
            matched.update({label: groups[label] for label, score in scores.items() if top and score == top})
        if hits < len(products):
            matched = {}
        lines: list[str] = []
        for label, rows in sorted((matched or groups).items(), key=lambda kv: -len(kv[1])):
            values: dict[str, list[str]] = {}
            for block in rows:
                for name, text in block.items():
                    values.setdefault(name, []).append(text)
            kept = []
            for name, seen in values.items():
                mean = sum(len(v) for v in seen) / len(seen)
                if mean > 40 or 2 * sum(" / " in v for v in seen) >= len(seen) or (len(seen) >= 4 and len(set(seen)) == len(seen) and mean > 12):
                    continue
                kept.append((name, seen))
            if label:
                lines.append(f"category {label!r} ({len(rows)} listings read):")
            for name, seen in sorted(kept, key=lambda kv: -len(kv[1]))[:18]:
                lines.append(f"- {name} (in {len(seen)} listings): " + " | ".join(list(dict.fromkeys(seen))[:5]))
        return "\n".join(lines[:44])

    def label_lines(self) -> str:
        """Short values that recur under one key across listings (category-like labels), with counts."""
        tallies: dict[str, dict[str, int]] = {}

        def visit(d: dict[str, Any]) -> None:
            if len(d) < 4:
                return
            for key, piece in d.items():
                if isinstance(piece, str) and 0 < len(piece) <= 40 and not (re.search(r"\d", piece) and re.fullmatch(CODE_LIKE, piece)):
                    value_counts = tallies.setdefault(str(key), {})
                    value_counts[piece] = value_counts.get(piece, 0) + 1

        for rec in self.items:
            walk_dicts(rec, visit)
        lns = []
        for key, value_counts in tallies.items():
            if sum(value_counts.values()) >= 4 and 2 <= len(value_counts) <= 8 and max(value_counts.values()) >= 2:
                lns.append(f"- {key}: " + " | ".join(f"{val} (x{qty_n})" for val, qty_n in sorted(value_counts.items(), key=lambda kv: -kv[1])))
        return "\n".join(lns[:12])


# ================================================================================================================ needs
class Needs:
    """What the shopper wants now: the request's reading, what was revealed, dropped or freed, and the budget."""

    def __init__(self, wire: Wire, request: str):
        self.wire = wire
        self.request = request
        self.reading: dict[str, Any] = {}
        self.written: list[str] = []
        self.uncovered: list[str] = []
        self.indifferent: list[str] = []
        self.let_go: list[str] = []
        self.now_reqs: list[str] | None = None
        self.spend_cap: Any = None
        self.version = 0
        self.caught: set[str] = set()
        self.exchanges: list[str] = []
        self.reply_due = ""
        self.widened_at = -1
        self.widened_tick = -10 ** 6
        self.sweepable = False     # the search tool has a paging argument, so the code can read shelf slices itself
        self.lines: list[dict[str, Any]] = []   # one per requested product: product, quantity, requirements, priority
        self.garbled = False       # the shopper did not follow the last question
        self.pointers: list[dict[str, Any]] = []   # needs whose value sits on another listing: open, resolved or unresolved

    @staticmethod
    def prop_of(req: str) -> str:
        return req.split(":", 1)[0].strip().lower() if isinstance(req, str) and ":" in req else ""

    @staticmethod
    def same_prop(one: str, two: str) -> bool:
        """Two property names that are the same words (plural endings and one-letter spellings aside)."""
        def words(text_body: str) -> list[str]:
            return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", text_body.lower()) if w not in LINK_WORDS]
        a, b = words(one), words(two)
        return bool(a) and len(a) == len(b) and all(x == y or (len(x) >= 5 and edit_distance(x, y, 2) <= 1) for x, y in zip(a, b))

    def add_pointers(self, raw: Any, source: str) -> list[dict[str, Any]]:
        """Books the pointer needs a reading named (at most POINTERS_MOST per task, one per property): each needs a
        property and something that singles the other listing out (a seller or a fact)."""
        added = []
        for entry in raw if isinstance(raw, list) else []:
            if not isinstance(entry, dict) or len(self.pointers) >= POINTERS_MOST:
                continue
            prop = re.sub(r"\s+", " ", str(entry.get("property") or "")).strip()[:60]
            seller = re.sub(r"\s+", " ", str(entry.get("seller") or "")).strip()[:60]
            facts = entry.get("facts")
            facts = [facts] if isinstance(facts, str) else facts if isinstance(facts, list) else []
            facts = [str(f).strip()[:80] for f in facts if isinstance(f, (str, int, float)) and str(f).strip()][:4]
            if not prop or not (seller or facts) or any(self.same_prop(prop, p["property"]) for p in self.pointers):
                continue
            ptr = dict(property=prop, kind=str(entry.get("kind") or "").strip()[:80], seller=seller, facts=facts,
                       source=source, state="open", queue=[], seen=[])
            ptr["for"] = str(entry.get("for") or "").strip()[:80]
            ptr.update(steps=0, searches=0, reads=0)     # (the look-up's spend: steps, searches, details reads)
            self.pointers.append(ptr)
            added.append(ptr)
        return added

    def pointed(self, prop: str) -> bool:
        """A property whose value the shopper gave by pointing at another listing (asked about once is enough)."""
        return any(self.same_prop(prop, p["property"]) for p in self.pointers)

    def set_lines(self, reading: dict[str, Any]) -> None:
        """The errand's product lines from the reading ('items'), else one line from its single-product keys. The
        reading's single-product keys are then filled from the lines, so older readers of them keep working."""
        lines = []
        for entry in reading.get("items") if isinstance(reading.get("items"), list) else []:
            if not isinstance(entry, dict) or not str(entry.get("product") or "").strip():
                continue
            count = entry.get("quantity")
            count = int(count) if is_number(count) and 1 <= count <= 99 and float(count).is_integer() else 1
            raw = entry.get("requirements")
            raw = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
            reqs = [str(r).strip() for r in raw if isinstance(r, (str, int, float)) and str(r).strip()]
            rank = entry.get("priority") if entry.get("priority") in ("cheapest", "best") else "none"
            lines.append({"product": str(entry["product"]).strip(), "quantity": count, "requirements": reqs, "priority": rank})
        if not lines:
            count = reading.get("quantity")
            count = int(count) if is_number(count) and 1 <= count <= 99 and float(count).is_integer() else 1
            raw = reading.get("requirements")
            raw = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
            lines = [{"product": str(reading.get("product") or "").strip(), "quantity": count,
                      "requirements": [str(r) for r in raw if r],
                      "priority": reading.get("priority") if reading.get("priority") in ("cheapest", "best") else "none"}]
        self.lines = lines[:9]
        if len(self.lines) == 1:
            only = self.lines[0]
            reading.setdefault("product", only["product"])
            reading["quantity"] = only["quantity"]
            reading["requirements"] = only["requirements"]
            reading["priority"] = only["priority"] if only["priority"] != "none" else reading.get("priority", "none")
        else:
            reading["product"] = " + ".join(f"{ln['quantity']} x {ln['product']}" for ln in self.lines)
            reading["requirements"] = [f"{r} (for {ln['product']})" for ln in self.lines for r in ln["requirements"]]
            ranks = {ln["priority"] for ln in self.lines} - {"none"}
            reading["priority"] = "cheapest" if "cheapest" in ranks else "best" if ranks else reading.get("priority", "none")
            reading["quantity"] = min(ln["quantity"] for ln in self.lines)

    def several(self) -> bool:
        return len(self.lines) > 1

    def line_qty(self, idx: int | None) -> int:
        if idx is not None and 0 <= idx < len(self.lines):
            return self.lines[idx]["quantity"]
        return self.quantity()

    def line_for(self, words: str) -> int | None:
        """The product line whose product words appear most in a text (a listing's title or a shopper's line)."""
        if len(self.lines) == 1:
            return 0
        text_words = set(re.findall(r"[a-z0-9]{3,}", words.lower()))
        scores = []
        for idx, ln in enumerate(self.lines):
            mine = {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z0-9]{3,}", ln["product"].lower())}
            theirs = {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in text_words}
            scores.append((len(mine & theirs), -idx))
        best = max(scores) if scores else (0, 0)
        if best[0] == 0 or sum(1 for s in scores if s[0] == best[0]) > 1:
            return None
        return -best[1]

    def quantity(self) -> int:
        need_n = self.reading.get("quantity")
        return need_n if isinstance(need_n, int) and not isinstance(need_n, bool) and need_n >= 1 else 1

    def priority(self) -> str:
        return self.reading.get("priority") if self.reading.get("priority") in ("cheapest", "best") else "none"

    def props_left(self) -> list[str]:
        """The property requirements in force (price, budget, stock and quantity lines left out)."""
        left = self.now_reqs if self.now_reqs is not None else [r for r in self.written + self.uncovered if r not in self.let_go]
        return [r for r in left if r not in self.let_go and not re.search(r"\bprice\b|budget|quantity|\bstock\b|\bcents?\b", r.lower())]

    def text_body(self) -> str:
        if self.now_reqs is not None:
            items = list(self.now_reqs)
        else:
            items = [f"{req} (stated)" for req in self.written if req not in self.let_go]
            items += [f"{req} (revealed by the shopper)" for req in self.uncovered if req not in self.let_go]
        collected = "; ".join(items) if items else (
            "no property requirement is left: every in-stock listing in the requested category fits" if self.let_go else "see the request")
        if self.spend_cap is not None:
            collected += f"; total price at most {self.spend_cap} {self.reading.get('currency') or ''}".rstrip()
        if self.several():
            collected = ("products in one order: " + "; ".join(f"line {i + 1}: {ln['quantity']} x {ln['product']}"
                                                             for i, ln in enumerate(self.lines)) + ". Requirements: " + collected)
        else:
            collected += f"; quantity {self.quantity()}"
        if self.let_go:
            collected += "; no longer required: " + "; ".join(self.let_go)
        if self.indifferent:
            collected += "; shopper does not care about: " + ", ".join(self.indifferent)
        return collected

    def changes(self) -> str:
        """For the checks, which read the request's own text: the requirements the shopper dropped since."""
        if not self.let_go:
            return ""
        return ("\n\nCHANGES SINCE THE REQUEST (they override its text): the shopper DROPPED " + "; ".join(self.let_go)
                + ". A dropped requirement no longer counts: never fail, rank or prefer a listing on it.")

    def overview(self, step_count: int, msgs: int | None) -> str:
        reading = self.reading
        if not reading:
            return ""
        lns = ["A first reading of the request (check it against the request itself):"]
        if self.several():
            lns.append(f"- {len(self.lines)} different products in ONE order, one basket line each:")
            for idx, ln in enumerate(self.lines):
                lns.append(f"  line {idx + 1}: {ln['quantity']} x {ln['product']}" + (f" (the {ln['priority']})" if ln["priority"] != "none" else "")
                           + "".join(f"; {req}" for req in ln["requirements"]))
        else:
            if reading.get("product"):
                lns.append(f"- product: {reading.get('product')}")
            lns.append(f"- quantity: {self.quantity()}")
            lns += [f"- stated requirement: {req}" for req in reading.get("requirements") or []]
        if reading.get("price_cap") is not None:
            lns.append(f"- price cap (total): {reading.get('price_cap')} {reading.get('currency') or ''}".rstrip())
        if reading.get("in_stock_only"):
            lns.append("- only items in stock / on hand")
        if self.priority() != "none":
            detail_note = f" ({reading.get('priority_detail')})" if reading.get("priority_detail") else ""
            lns.append(f"- choose the {self.priority()}{detail_note} among all fitting items")
        else:
            lns.append("- no cheapest/best rule was found: any item meeting every requirement within the budget is right, so do not spend steps hunting for the lowest price (unless the request itself says otherwise)")
        lns += [f"- nice to have, NOT required (only a tie-breaker): {w}" for w in reading.get("soft_wants") or []]
        lns += [f"- must be learned from the shopper first: {a}" for a in reading.get("ask_first") or []]
        lns += [f"- rule: {req}" for req in reading.get("rules") or []]
        if msgs is not None:
            lns.append(f"- messages to the shopper allowed: {msgs}")
        lns.append(f"- steps allowed: {step_count}")
        return "\n".join(lns)

    def hear(self, lns: list[str], qtext: str | None) -> str:
        """Reads fresh shopper lines into the ledger; returns a note for the actor ('' when nothing was read)."""
        new_lines = [ln for ln in lns if ln not in self.caught]
        if not new_lines:
            return ""
        self.caught.update(new_lines)
        if self.wire.spare() < CLOSING:
            return ""
        ask_text = HEAR + "\n\nTHE SHOPPER'S REQUEST (for context):\n" + self.request
        so_far = self.text_body()
        ask_text += "\n\nREQUIREMENTS SO FAR:\n" + so_far
        if self.exchanges:
            ask_text += "\n\nEARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(self.exchanges[-12:])
        if qtext:
            ask_text += "\n\nTHE ASSISTANT ASKED:\n" + qtext
        ask_text += "\n\nTHE SHOPPER NOW SAID:\n" + "\n".join(new_lines)
        self.exchanges.append((f"asked: {qtext} -> " if qtext else "") + "shopper: " + " / ".join(new_lines))
        answer = self.wire.consult(ask_text, 2500, LISTENER, roof=SPEND_STOP)
        self.garbled = False
        if not answer:
            return ""
        if qtext is not None and answer.get("understood") is False and not answer.get("needs") and not is_number(answer.get("new_budget")):
            # (a reply that only asks for the question again reveals nothing: it is no "don't mind")
            self.garbled = True
            answer["no_preference"] = []
        remarks = []
        before = set(self.now_reqs or []) | set(self.uncovered) | set(self.written)
        # (a requirement stands on the first reading or on the shopper's own words: one whose property and value
        # words appear in neither came from the story and is left out)
        said = (so_far + " " + " ".join(self.exchanges) + " " + (qtext or "")).lower()

        def grounded(req: str) -> bool:
            words = [w for w in re.findall(r"[a-z]+", req.lower()) if len(w) >= 3]
            return not words or any(w in said for w in words)

        now_list = answer.get("requirements_now")
        if isinstance(now_list, list) and all(isinstance(req, str) for req in now_list):
            self.now_reqs = [req for req in now_list if req.strip() and grounded(req)]
        for new_need in answer.get("needs") or []:
            if isinstance(new_need, str) and new_need and new_need not in self.uncovered and grounded(new_need):
                self.uncovered.append(new_need)
                remarks.append(f"new requirement: {new_need}")
        held = {r.split(":", 1)[0].strip().lower() for r in (self.now_reqs if self.now_reqs is not None else self.written + self.uncovered)
                if isinstance(r, str) and ":" in r and r not in self.let_go}
        for prop_name in answer.get("no_preference") or []:
            if isinstance(prop_name, str) and prop_name and prop_name not in self.indifferent and prop_name.strip().lower() not in held:
                self.indifferent.append(prop_name)
        widened = False
        for old_need in answer.get("dropped") or []:
            if isinstance(old_need, str) and old_need and old_need not in self.let_go \
                    and not (is_number(answer.get("new_budget"))
                             and re.search(r"\b(?:prices?|budget|costs?|spend\w*|cents?|dollars?)\b", old_need.lower())):
                self.let_go.append(old_need)
                remarks.append(f"dropped: {old_need}")
                widened = widened or self.priority() != "none"
        spend_cap = answer.get("new_budget")
        if is_number(spend_cap) and spend_cap > 0 and spend_cap != self.spend_cap:
            self.spend_cap = spend_cap
            remarks.append(f"new budget: {spend_cap}")
        if answer.get("unavailable"):
            remarks.append("an item became unavailable; take it out of the cart and do not order it")
        if remarks:
            self.version += 1
            if widened:
                self.widened_at = self.version
                self.widened_tick = self.wire.tick
        if self.now_reqs is not None:
            # (a need the shopper revealed stays in force until the shopper drops it: a rebuilt list that lost it,
            # after a "don't mind" to some other message, gets it back)
            for need in self.uncovered:
                prop_name = need.split(":", 1)[0].strip().lower()
                kept = any(prop_name and prop_name in req.lower() for req in self.now_reqs)
                released = any(prop_name and prop_name in old.lower() for old in self.let_go)
                if prop_name and not kept and not released:
                    self.now_reqs.append(need)
        fresh = self.add_pointers(answer.get("references"), "shopper")
        for ptr in fresh:
            # (the pointer itself is no requirement a check can read: a line the model wrote for it this time is taken
            # out, and the value the other listing gives is added once the code has read it; a line naming a plain
            # value of that property is kept, and then the shopper's own value stands and no look-up runs)
            marks = [m for m in ["listing", ptr["seller"].lower()] + [w for w in re.findall(r"[a-z]{4,}", ptr["kind"].lower())] if m]

            def new_line(req: Any, ptr: dict[str, Any] = ptr) -> bool:
                return isinstance(req, str) and req not in before and self.same_prop(self.prop_of(req), ptr["property"])

            def pointer_line(req: Any, marks: list[str] = marks) -> bool:
                return new_line(req) and (not req.split(":", 1)[1].strip() or any(m in req.split(":", 1)[1].lower() for m in marks))
            if any(new_line(r) and not pointer_line(r) for r in (self.now_reqs or []) + self.uncovered):
                ptr["state"] = "named"
                self.wire.log.append({"kind": "pointer", "state": "named", "property": ptr["property"]})
            self.uncovered = [r for r in self.uncovered if not pointer_line(r)]
            if self.now_reqs is not None:
                self.now_reqs = [r for r in self.now_reqs if not pointer_line(r)]
            remarks = [r for r in remarks if not (r.startswith("new requirement: ") and pointer_line(r[len("new requirement: "):]))]
            if ptr["state"] != "open":
                continue
            remarks.append(f"the shopper's {ptr['property']} is the value given on another listing ("
                           + ", ".join(x for x in (ptr["kind"], ptr["seller"] and "from " + ptr["seller"], "; ".join(ptr["facts"])) if x)
                           + "): the code reads that listing next and adds the value as a requirement")
        if fresh:
            self.wire.log.append({"kind": "pointer_heard", "pointers": [{k: p[k] for k in ("property", "kind", "seller", "facts", "for")} for p in fresh]})
        if widened:
            # (decided after the restore above: a revealed need the rebuilt list lost is still in force)
            if not self.props_left() and self.priority() == "cheapest" and self.sweepable:
                remarks.append("no property requirement is left: every in-stock listing in the requested category fits, whatever "
                               "object it is (an accessory, a part or a set shelved there counts), and the cheapest on the whole shelf "
                               "wins. Rows come by relevance or id, not by price, so further slices of the shelf are read for you next "
                               "(its listing's pages, the label's own words, words naming the kinds of object in the titles); if no "
                               "'[shelf sweep done]' list follows, cover the shelf yourself with short queries, the first page of "
                               "each - the category label, each kind word of the label alone, each item name that recurs in the "
                               "titles seen - and the category listing's pages. Then make the lowest-priced in-stock row on the "
                               "requested category within the budget the cart's only line, re-check its stock and price, and order it")
            elif not self.props_left():
                remarks.append("no property requirement is left: every in-stock listing in the requested category fits, whatever "
                               "object it is (an accessory, a part or a set shelved there counts), and the cheapest on the whole shelf "
                               "wins. Rows come by relevance or id, not by price: cover the shelf with short queries, the first page "
                               "of each - the category label, each kind word of the label alone, each item name that recurs in the "
                               "titles seen - and the category listing's pages; then order the lowest in-stock price")
            else:
                remarks.append("a dropped requirement WIDENS what is acceptable: listings that failed only on it fit now, and "
                               "listings never searched for may beat the pick; search the item kind again without it, as short "
                               "queries of each remaining required value with the kind word, value first, reading "
                               "every page, and choose the cheapest over everything acceptable now")
        self.wire.log.append({"kind": "heard", "question": qtext is not None, "needs": len(answer.get("needs") or []),
                              "free": len(answer.get("no_preference") or []), "dropped": len(answer.get("dropped") or []),
                              "budget": is_number(spend_cap)})
        # (a reply goes back only when the shopper asked something: answering a plain statement spends a message and
        # draws a "don't mind" that reads like a change)
        asked_back = qtext is None and any("?" in ln for ln in new_lines) and not answer.get("no_preference")
        self.reply_due = str(answer.get("reply") or "").strip() if asked_back else ""
        return "Meaning: " + str(answer.get("meaning") or "") + (" (" + "; ".join(remarks) + ")" if remarks else "")


# ================================================================================================================ cart
class Basket:
    """The cart as the code knows it, and the basket the errand means to order.

    `units` is listing -> units the world's cart holds: counted from each accepted cart call and replaced by every
    basket the world shows (its reply is the truth). `aims` is product line -> (listing, units) the errand intends.
    The cart tool's put/take switch comes from its schema (ToolMap.cart_modes); without one, a call whose enum
    values differ from the first accepted call's, or a zero count, takes out. `adds` says whether a put on a held
    listing adds units (True) or sets them (False); the tool's description or the first basket reply settles it."""

    def __init__(self, tools: "ToolMap | None" = None, the_tool: str | None = None):
        self.the_tool = the_tool
        self.prop_specs = tools.props(the_tool) if tools is not None and the_tool else {}
        self.modes = tools.cart_modes if tools is not None else None
        self.id_key = tools.id_key(the_tool) if tools is not None else None
        self.count_key = tools.count_key(the_tool) if tools is not None else None
        said = tools.said(the_tool) if tools is not None and the_tool else ""
        self.adds: bool | None = False if re.search(r"\b(?:sets?|replac\w*|overwrit\w*|exactly|absolute)\b", said) else \
            True if re.search(r"\b(?:adds? (?:more|units|to)|increment\w*|increas\w*|on top)\b", said) else None
        self.put_form: str | None = None
        self.units: dict[str, float] = {}
        self.aims: dict[int, tuple[str, int]] = {}
        self.used = False
        self.blind = False                     # an accepted cart call named no listing the code could read: the count is unsure
        self.version = 0
        self.changed_at = 0
        self.viewed_at = -1
        self.last_view: Any = None
        self.pending: dict[str, float] = {}    # listing -> units expected after the last call if the tool sets them
        self.shown = -1                        # the cart version the world last showed (its counts are then the truth)
        # (a cart tool that takes the whole basket as one list of lines, each naming a listing and its units: the list
        # replaces the basket, so a listing left out of it is taken out)
        self.lines_key: str | None = None
        self.line_id: str | None = None
        self.line_count: str | None = None
        self.booked_args: Any = None
        self.booked_fresh: list[str] = []
        self.booked_taken: list[str] = []
        self.line_int = False                  # the store's listing ids are whole numbers (learned from a call)
        named = tools.ids(the_tool) if tools is not None and the_tool else []
        for key, pspec in (self.prop_specs.items() if not named else ()):
            inner = pspec.get("items") if isinstance(pspec, dict) and "array" in kinds(pspec) else None
            if not isinstance(inner, dict) or "object" not in (kinds(inner) or ["object"]):
                continue
            fields = inner.get("properties") if isinstance(inner.get("properties"), dict) else {}
            ident = next((k for k, p in fields.items() if id_param(k, p)), None)
            if ident or not fields:
                # (a list of objects with no field schema: the line's fields are learned from the first call)
                self.lines_key, self.line_id = key, ident
                self.line_count = next((k for k, p in fields.items() if k != ident and isinstance(p, dict) and set(kinds(p)) & {"integer", "number"}), None)
                self.line_int = bool(ident) and "integer" in kinds(fields.get(ident)) and "string" not in kinds(fields.get(ident))
                self.adds = False
                self.id_key = None             # (v010's free-text fallback would write the listing into a note argument)
                break

    def selection(self, call_args: dict[str, Any]) -> dict[str, float] | None:
        """Listing -> units of a whole-basket call (None when the tool takes no basket list or the call has none)."""
        lines = call_args.get(self.lines_key) if self.lines_key else None
        if not isinstance(lines, list):
            return None
        found: dict[str, float] = {}
        for line in lines:
            if isinstance(line, dict) and self.line_id is None:
                # (the line's id: a field named like an id (as id_param reads names) holding id-like text or a whole
                # number, else the one id-like text; its units: another whole number)
                held = [k for k, v in line.items() if (isinstance(v, str) and re.fullmatch(ID_LIKE, v) and any(c.isdigit() for c in v))
                        or (isinstance(v, int) and not isinstance(v, bool))]
                self.line_id = (next((k for k in held if ID_NAME.search(spoken(k))), None)
                                or next((k for k in held if isinstance(line[k], str)), None))
                self.line_int = self.line_id is not None and isinstance(line[self.line_id], int)
                self.line_count = next((k for k, v in line.items() if is_number(v) and not isinstance(v, bool) and k != self.line_id), self.line_count)
            ident = line.get(self.line_id) if isinstance(line, dict) and self.line_id else None
            if isinstance(ident, (str, int)) and not isinstance(ident, bool) and str(ident).strip():
                count = line.get(self.line_count) if self.line_count else None
                found[str(ident).strip()] = found.get(str(ident).strip(), 0.0) + (float(count) if is_number(count) else 1.0)
        return found

    def line_list(self, units: dict[str, float]) -> list[dict[str, Any]] | None:
        """A basket as the whole-basket tool takes it; None while the line's id field is unknown (no call is made)."""
        if not self.line_id:
            return None
        return [{self.line_id: int(thing) if self.line_int and str(thing).isdigit() else thing, **({self.line_count: int(qty) if float(qty).is_integer() else qty} if self.line_count else {})}
                for thing, qty in units.items() if qty > 0]

    def changes(self, call_args: dict[str, Any]) -> tuple[list[str], list[str]]:
        """(listings a whole-basket call puts in or sets to new units, listings it leaves out) against the cart as
        counted; for the call booked last, as it was before it."""
        if call_args is self.booked_args:
            return self.booked_fresh, self.booked_taken
        lines = self.selection(call_args) or {}
        fresh = [t for t, q in lines.items() if q > 0 and abs(self.units.get(t, 0.0) - q) > 1e-9]
        return fresh, [t for t in self.units if lines.get(t, 0.0) <= 0]

    def swap_args(self, old: str, new: str, count: int) -> dict[str, Any] | None:
        """One whole-basket call that replaces a listing by another at its units (None for other cart tools)."""
        if not self.lines_key:
            return None
        units = {t: q for t, q in self.units.items() if t != old}
        units[new] = float(count)
        lines = self.line_list(units)
        return {self.lines_key: lines} if lines is not None else None

    def form(self, call_args: dict[str, Any]) -> str:
        return as_text({fkey: val for fkey, val in call_args.items() if isinstance(self.prop_specs.get(fkey), dict) and "enum" in self.prop_specs[fkey]})

    def aim(self, call_args: dict[str, Any]) -> str | None:
        """The listing a cart call names: its id argument, else a one-word text argument that is not an enum."""
        if self.lines_key:
            lines = self.selection(call_args)
            fresh, taken = self.changes(call_args)
            return (fresh or taken or list(lines or {}) or [None])[0]
        if self.id_key and isinstance(call_args.get(self.id_key), (str, int)) and not isinstance(call_args.get(self.id_key), bool):
            return str(call_args[self.id_key]).strip() or None
        for key, piece in call_args.items():
            pspec = self.prop_specs.get(key) if isinstance(self.prop_specs.get(key), dict) else {}
            if isinstance(piece, str) and "enum" not in pspec and piece.strip() and " " not in piece.strip():
                return piece
        return None

    def count_arg(self, call_args: dict[str, Any]) -> float | None:
        if self.lines_key:
            thing = self.aim(call_args)
            return (self.selection(call_args) or {}).get(thing, 0.0) if thing else None
        keys = [self.count_key] if self.count_key else [k for k, p in self.prop_specs.items() if isinstance(p, dict)
                                                         and set(kinds(p)) & {"integer", "number"} and "enum" not in p]
        for key in keys:
            pspec = self.prop_specs.get(key) if isinstance(self.prop_specs.get(key), dict) else {}
            piece = call_args.get(key, pspec.get("default"))
            if is_number(piece):
                return float(piece)
        return None

    def is_take(self, call_args: dict[str, Any]) -> bool:
        if self.lines_key:
            fresh, taken = self.changes(call_args)
            return bool(taken) and not fresh
        if self.modes:
            key, _, take = self.modes
            pspec = self.prop_specs.get(key) if isinstance(self.prop_specs.get(key), dict) else {}
            return call_args.get(key, pspec.get("default")) == take
        if self.put_form is not None and self.form(call_args) != self.put_form:
            return True
        count = self.count_arg(call_args)
        return count is not None and count <= 0

    def is_put(self, call_args: dict[str, Any]) -> bool:
        return not self.is_take(call_args)

    def put_args(self, thing: str, count: int) -> dict[str, Any]:
        """Arguments that put `count` units of a listing in (for the code's own cart calls)."""
        call_args: dict[str, Any] = {}
        if self.lines_key:
            units = {t: q for t, q in self.units.items() if t != thing}
            units[thing] = float(count)
            lines = self.line_list(units)
            if lines is None:
                return None
            call_args[self.lines_key] = lines
        if self.id_key:
            call_args[self.id_key] = thing
        if self.modes:
            call_args[self.modes[0]] = self.modes[1]
        if self.count_key:
            call_args[self.count_key] = count
        for key, pspec in self.prop_specs.items():
            if key in call_args or not isinstance(pspec, dict):
                continue
            if isinstance(pspec.get("enum"), list) and pspec["enum"]:
                call_args[key] = next((o for o in pspec["enum"] if isinstance(o, str) and re.search(PUT_WORDS, spoken(o))), pspec["enum"][0])
        return call_args

    def take_args(self, thing: str, count: float) -> dict[str, Any] | None:
        """Arguments that take every unit of a listing out, or None when the tool shows no way to."""
        if self.lines_key:
            units = dict(self.units)
            left = units.get(thing, 0.0) - (count if count > 0 else units.get(thing, 0.0))
            if left > 0:
                units[thing] = left
            else:
                units.pop(thing, None)
            lines = self.line_list(units)
            return {self.lines_key: lines} if lines is not None else None
        call_args: dict[str, Any] = {self.id_key: thing} if self.id_key else {}
        if self.modes:
            call_args[self.modes[0]] = self.modes[2]
            if self.count_key:
                call_args[self.count_key] = int(count)
            return call_args
        if self.count_key and self.adds is False:
            call_args[self.count_key] = 0
            return call_args
        return None

    def held(self, thing: str) -> float:
        return self.units.get(thing, 0.0)

    def booked(self, call_args: dict[str, Any], obs: Any) -> str:
        """Books a cart call the world answered; returns a note on what the cart holds now."""
        self.used = True
        fault = fault_of(obs, "write")
        if fault:
            return f"[cart] the cart call failed ({fault}); nothing changed."
        if self.lines_key:
            lines = self.selection(call_args)
            if lines is None:
                self.blind = True
                return ""
            fresh, taken = self.changes(call_args)
            self.booked_args, self.booked_fresh, self.booked_taken = call_args, fresh, taken
            self.version += 1
            self.units = {t: q for t, q in lines.items() if q > 0}
            self.pending = {}
            seen = self.view_units(obs)
            if seen is not None:
                self.learn(seen)
            return self.summary()
        if self.put_form is None and not self.modes and not self.is_take(call_args):
            self.put_form = self.form(call_args)
        thing = self.aim(call_args)
        if not thing:
            self.blind = True
            return ""
        self.version += 1
        count = self.count_arg(call_args)
        count = 1.0 if count is None else count
        prior = self.held(thing)
        if self.is_take(call_args) and self.modes and count <= 0:
            count = prior
        if self.is_put(call_args):
            if prior > 0 and self.adds is not False:
                self.units[thing] = prior + count
                self.pending = {thing: count}         # the units if the tool sets instead of adding
            else:
                self.units[thing] = count if count > 0 else 0.0
        else:
            self.units[thing] = max(0.0, prior - count) if count > 0 else 0.0
        others = {t for t in self.units if t != thing}
        if self.units.get(thing, 0) <= 0:
            self.units.pop(thing, None)
        seen = self.view_units(obs)
        if seen is not None:
            if others <= set(seen):
                self.learn(seen)          # (the reply shows the whole cart: it is the truth)
            else:
                # (the reply shows only the line it changed: the other lines are kept as counted)
                kept = {t: q for t, q in self.units.items() if t not in seen}
                self.learn(seen)
                self.units.update(kept)
        return self.summary()

    def view_units(self, obs: Any) -> dict[str, float] | None:
        """Listing -> units read from a basket-like observation: rows holding a listing id and a count. The count is
        the field whose name says so (qty, quantity, count, pieces, units...), else the row's only small whole
        number. None when the observation shows no such rows."""
        rows: dict[str, float] = {}
        known = set(self.units) | set(self.pending) | {a[0] for a in self.aims.values()}

        def visit(d: dict[str, Any]) -> None:
            keyed = [str(val) for key, val in d.items() if isinstance(val, (str, int)) and not isinstance(val, bool)
                     and re.search(r"(?:^|[ _])id$|^id|ident|sku", spoken(key)) and " " not in str(val).strip()]
            loose = [val for val in d.values() if isinstance(val, str) and (val in known or re.fullmatch(ID_LIKE, val)) and " " not in val]
            ids = [i for i in keyed + loose if i in known] or keyed or loose
            if not ids:
                return
            money = r"price|cost|total|amount|stock|avail|rating|review|score|year|percent|discount|weight|size|fee|sum"
            counted = [val for key, val in d.items() if is_number(val) and re.search(COUNT_KEY, spoken(key).replace(" ", "_"))
                       and not re.search(money, spoken(key))]
            # (race 15's bag: "amount_held" is the units of a line that also shows its price)
            counted += [val for key, val in d.items() if isinstance(val, int) and not isinstance(val, bool)
                        and re.fullmatch(r"amount (?:held|ordered|taken|wanted|in cart|reserved)|amount", spoken(key))
                        and any(re.search(r"price|cost", spoken(k)) for k in d)]
            ints = [val for key, val in d.items() if isinstance(val, int) and not isinstance(val, bool) and 0 < val < 1000
                    and not re.search(money, spoken(key))]
            units = counted[0] if len(counted) == 1 else (ints[0] if len(ints) == 1 else None)
            if units is not None and units >= 0:
                rows[ids[0]] = rows.get(ids[0], 0) + float(units)
        walk_dicts(obs, visit)
        if not rows or not (set(rows) & (known or set(rows))):
            return None
        return {k: v for k, v in rows.items()}

    def learn(self, seen: dict[str, float]) -> None:
        """A cart view is the truth: it replaces the code's count and settles whether a put adds or sets."""
        for thing, expected_if_set in self.pending.items():
            if thing in seen:
                if abs(seen[thing] - expected_if_set) < 1e-9 and abs(self.units.get(thing, 0) - expected_if_set) > 1e-9:
                    self.adds = False
                elif seen[thing] > expected_if_set:
                    self.adds = True
        self.pending = {}
        self.units = {fkey: val for fkey, val in seen.items() if val > 0}
        self.shown = self.version

    def viewed(self, obs: Any, tick: int) -> None:
        self.last_view = obs
        self.viewed_at = tick
        seen = self.view_units(obs)
        if seen is not None:
            self.learn(seen)
        elif isinstance(obs, (dict, list)) and not self.ids_in(obs):
            self.units = {}

    def ids_in(self, obs: Any) -> bool:
        text_body = as_text(obs)
        return any(thing in text_body for thing in self.units)

    def summary(self) -> str:
        if not self.units:
            return "[cart as counted] empty."
        lns = ", ".join(f"{thing} x{int(qty_n) if float(qty_n).is_integer() else qty_n}" for thing, qty_n in self.units.items())
        return f"[cart as counted] {lns}."

    def lines_of(self, needs: "Needs", line_of: dict[str, int]) -> dict[int, list[str]]:
        """Product line -> the cart's listings on it (a listing on no known line is left out)."""
        found: dict[int, list[str]] = {}
        for thing in self.units:
            idx = 0 if not needs.several() else line_of.get(thing)
            if idx is None:
                idx = next((i for i, (aim_id, _) in self.aims.items() if aim_id == thing), None)
            if idx is not None:
                found.setdefault(idx, []).append(thing)
        return found

    def faults(self, needs: "Needs", line_of: dict[str, int]) -> list[str]:
        """What keeps the cart from holding exactly one listing per requested product at its units."""
        if not self.units:
            return []
        collected = []
        if not needs.several():
            if len(self.units) > 1:
                collected.append("the cart holds more than one listing (" + ", ".join(f"{i} x{int(q)}" for i, q in self.units.items())
                                 + "); take out every unit of each listing but the chosen one")
            elif any(abs(q - needs.line_qty(0)) > 1e-9 for q in self.units.values()):
                thing, have = next(iter(self.units.items()))
                change = "take out" if have > needs.line_qty(0) else "put in"
                collected.append(f"{thing} has {int(have)} units in the cart but the request needs {needs.line_qty(0)}; {change} "
                                 f"{int(abs(have - needs.line_qty(0)))} with the cart tool")
            return collected
        by_line = self.lines_of(needs, line_of)
        for idx, things in sorted(by_line.items()):
            want = needs.line_qty(idx)
            if len(things) > 1:
                collected.append(f"{needs.lines[idx]['product']} has {len(things)} listings in the cart ({', '.join(things)}); keep the "
                                 "chosen one and take every unit of the others out")
            for thing in things:
                have = self.units.get(thing, 0)
                if abs(have - want) > 1e-9:
                    collected.append(f"{thing} has {int(have)} units but {needs.lines[idx]['product']} needs {want}; "
                                     f"{'take out' if have > want else 'put in'} {int(abs(have - want))}")
        placed = {t for things in by_line.values() for t in things}
        loose = [t for t in self.units if t not in placed]
        if len(self.units) > len(needs.lines):
            collected.append(f"the cart holds {len(self.units)} listings for {len(needs.lines)} products: one listing per product, "
                             "take every unit of the extra ones out")
        if loose and (len(self.units) > len(needs.lines) or len(by_line) == len(needs.lines)):
            # (every product already has its listing, or there are more listings than products: these are extras)
            collected.append("these cart listings are for none of the requested products as far as the checks tell: "
                             + ", ".join(f"{t} x{int(self.units[t])}" for t in loose))
        missing = [ln["product"] for i, ln in enumerate(needs.lines) if i not in by_line]
        if missing and len(self.units) < len(needs.lines):
            collected.append("no listing in the cart yet for: " + "; ".join(missing))
        return collected

    def unit_fix(self, thing: str, have: float, want: int) -> dict[str, Any] | None:
        """One cart call that brings a held listing from `have` to `want` units, or None when the tool's semantics do
        not make the result certain (a put that may add or set, a take with no count)."""
        if not (self.count_key or self.line_count) or want < 1:
            return None
        if have > want:
            if self.modes:
                return self.take_args(thing, have - want)
            return self.put_args(thing, want) if self.adds is False else None
        if self.adds is False:
            return self.put_args(thing, want)
        return self.put_args(thing, int(want - have)) if self.adds is True else None

    def fix_calls(self, needs: "Needs", line_of: dict[str, int], good: set[str]) -> list[dict[str, Any]]:
        """The cart calls the code itself sends so the cart holds exactly one listing per product at its units: only
        those it is sure of, on counts the world last showed. Of several listings on one product the one the item
        check passed stays (none is taken out when that is not exactly one); a listing on no product goes when every
        product has a passed listing; units are set right on a listing whose product is known."""
        if not self.units or self.blind or self.shown != self.version or not self.the_tool:
            return []
        groups = {0: list(self.units)} if not needs.several() else self.lines_of(needs, line_of)
        drop: list[str] = []
        kept: dict[int, str] = {}
        for idx, things in groups.items():
            passed = [t for t in things if t in good]
            if len(things) == 1:
                kept[idx] = things[0]
            elif len(passed) == 1:
                kept[idx] = passed[0]
                drop += [t for t in things if t != passed[0]]
        placed = {t for things in groups.values() for t in things}
        loose = [t for t in self.units if t not in placed]
        if loose and len(kept) == len(needs.lines) and all(t in good for t in kept.values()):
            drop += loose
        if self.lines_key:
            # (one whole-basket call: the lines kept, each at its units)
            target = {t: q for t, q in self.units.items() if t not in drop}
            for idx, thing in sorted(kept.items()):
                if thing in good or not needs.several():
                    target[thing] = float(needs.line_qty(idx))
            lines = self.line_list(target) if target and target != self.units else None
            return [{self.lines_key: lines}] if lines is not None else []
        calls = []
        for thing in drop:
            undo = self.take_args(thing, self.units[thing])
            if undo is not None:
                calls.append(undo)
        for idx, thing in sorted(kept.items()):
            want, have = needs.line_qty(idx), self.units.get(thing, 0)
            if abs(have - want) > 1e-9 and (thing in good or not needs.several()):
                move = self.unit_fix(thing, have, want)
                if move is not None:
                    calls.append(move)
        return calls


# ================================================================================================================ duties
DUTY_KINDS = ("survey", "compare", "hold", "report", "confirm", "tell_basket", "ask_before_change", "decline_wording", "other")
GATING = ("survey", "compare", "hold", "report", "confirm", "tell_basket")
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
SURVEY_RX = re.compile(r"\b(?:check|look at|look over|look through|see|view|review|consider|survey|browse|open|read|"
                       r"examine|inspect|weigh|go through|explore)\b[^.;:!?]{0,40}?\b(?:at least|a minimum of|minimum of|"
                       r"no fewer than|not fewer than)\s+(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(?:[a-z-]+\s+){0,2}?"
                       r"(?:listings?|options?|items?|products?|candidates?|alternatives?|choices?|offers?|models?|results?)\b", re.I)
COMPARE_COUNT = re.compile(r"\b(?:compar\w*|side by side|line ?up)\b[^.;:!?]{0,30}?\b(?:at least\s+|a minimum of\s+|no fewer than\s+)?"
                           r"(\d+|two|three|four|five|six)\s+(?:[a-z-]+\s+){0,2}?(?:listings?|options?|items?|products?|candidates?|"
                           r"alternatives?|choices?|offers?|models?|results?|of them)\b", re.I)
TELL_CUE = re.compile(r"\b(?:confirm\w*|check with|ask|tell|let me know|show|message|send|update|inform|approv\w*|sign off|"
                      r"go[- ]ahead|ok|okay|run it by|summar\w*|recap|list|report)\b")
TIED_CUE = re.compile(r"\b(?:before|first|confirm\w*|approv\w*|go[- ]ahead|sign off|ok|okay|prior to)\b")
DUTY_CUES = {"hold": HOLD_CUE, "compare": COMPARE_CUE, "report": REPORT_CUE, "confirm": TELL_CUE, "tell_basket": TELL_CUE}
# (a request sentence that itself asks for a message before the purchase: all four present, no negation before the verb)
TELL_VERB = re.compile(r"\b(?:confirm\w*|check (?:it |this |that )?with|ask|tell|let (?:me|us) know|message|notify|update|inform|"
                       r"run (?:it|that|this|them) by|approv\w*|sign off|summar\w*|recap|show (?:me|us))\b")
CHANGE_WARN = re.compile(r"\bkeep (?:on )?(?:checking|re-?checking)\b|\b(?:can|may|might|could|will) change\b|\bre-?check|"
                         r"\bthings (?:can |may |might )?(?:change|shift|move)\b|\bsubject to change\b")
REPORT_ASK = re.compile(r"\b(?:send|give|write|message)\s+(?:me|us|the (?:shopper|buyer|customer))\b[^.;!?]{0,50}\b(?:report|summary|recap|rundown)\b")
BEFORE_WORD = re.compile(r"\b(?:before|prior to|ahead of|until|beforehand)\b")
PURCHASE_WORD = re.compile(r"\b(?:buy\w*|purchas\w*|order\w*|check ?out|checking out|pay\w*|place|placing|finali[sz]\w*|commit\w*)\b")
SHOPPER_WORD = re.compile(r"\b(?:me|us)\b|\b(?:shopper|customer|buyer|client)s?\b")
OK_WORD = re.compile(r"\b(?:confirm\w*|approv\w*|ok|okay|go[- ]ahead|sign off|permission|agree\w*|consent|ask)\b")
NOT_TELL = re.compile(r"\b(?:don'?t|do not|no need|needn'?t|never|without|not necessary|no reason)\b[^.;!?]{0,30}"
                      r"\b(?:ask|tell|confirm|check|message|bother|update|inform|run)")


def count_word(text_body: Any) -> int | None:
    low = str(text_body or "").strip().lower()
    if low.isdigit():
        return int(low)
    return NUMBER_WORDS.index(low) if low in NUMBER_WORDS else None


class Duty:
    def __init__(self, text_body: str, kind: str, count: int | None = None, timing: str = "any", stated: bool = True):
        self.text = text_body
        self.kind = kind
        self.count = count
        self.timing = timing
        self.stated = stated
        self.state = "open"         # open, done, failed, void
        self.tries = 0
        self.mark: Any = None       # what the step proved, for steps that reopen when the basket changes
        self.why = ""

    def brief(self) -> str:
        return f"{self.kind}{'(' + str(self.count) + ')' if self.count else ''}{'' if self.stated else '*'}: {self.text[:80]}"


class Duties:
    """The procedure the request states (open N listings, compare, hold, report, confirm, tell the basket), kept as
    code state. Every world result passes through observe(); a step is done only when a tool result proves it. The
    code carries the steps out itself with read, compare, hold, report and message tools only: never the cart, an
    order or a decline. Steps marked * were not stated: a comparison when a compare tool exists. A message about the
    basket goes to the shopper only when the request asks for one. Stated steps of no tracked kind ('other',
    'ask_before_change') are checked against the action record by the rule audit before a finish."""

    def __init__(self, request: str, tools: ToolMap, log: list[dict[str, Any]]):
        self.request = request
        self.tools = tools
        self.log = log
        self.items: list[Duty] = []
        self.opened: set[str] = set()
        self.compared: list[list[str]] = []
        self.held: set[str] = set()
        self.reported_at = -1
        self.told: Any = None

    # -- reading
    def read(self, entries: Any, msg_cap: int | None) -> None:
        heard = set(re.findall(r"[a-z0-9]+", self.request.lower()))
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            kind = str(entry.get("kind") or "").strip().lower().replace(" ", "_").replace("-", "_")
            text_body = re.sub(r"\s+", " ", str(entry.get("text") or "")).strip()
            words = re.findall(r"[a-z0-9]+", text_body.lower())
            if kind not in DUTY_KINDS or not words:
                continue
            if sum(1 for w in words if w in heard) < 0.6 * len(words):
                self.event(f"duty '{text_body[:60]}' -> void: the request does not say it")
                continue
            cue = DUTY_CUES.get(kind)
            if (cue is not None and not cue.search(text_body.lower())) or (kind in ("confirm", "tell_basket") and not TIED_CUE.search(text_body.lower())):
                self.event(f"duty '{text_body[:60]}' -> void: its words do not ask for {kind}")
                continue
            count = entry.get("count")
            count = int(count) if is_number(count) and 1 <= count <= 12 else None
            timing = str(entry.get("timing") or "any").strip().lower()
            timing = timing if timing in ("before_order", "before_decline", "before_finish", "any") else "any"
            self.add(Duty(text_body[:200], kind, count, timing))
        for hit in SURVEY_RX.finditer(self.request):
            found = count_word(hit.group(1))
            if found and 1 <= found <= 12:
                self.add(Duty(hit.group(0)[:200], "survey", found, "before_order"))
        hit = COMPARE_COUNT.search(self.request)
        if hit and count_word(hit.group(1)):
            if not any(d.kind == "compare" for d in self.items):
                self.add(Duty(hit.group(0)[:200], "compare", count_word(hit.group(1)), "before_order"))
            for duty in self.items:
                if duty.kind == "compare" and not duty.count:
                    duty.count = count_word(hit.group(1))
        roles = self.tools.roles
        for duty in list(self.items):
            if duty.kind == "compare" and not roles.get("compare"):
                self.items.remove(duty)
                self.add(Duty(duty.text, "survey", max(2, duty.count or 2), duty.timing, duty.stated))
            elif duty.kind == "hold" and not roles.get("hold"):
                duty.state, duty.why = "void", "no tool holds listings"
            elif duty.kind == "report" and not roles.get("report"):
                if roles.get("message"):
                    duty.kind = "tell_basket"
                else:
                    duty.state, duty.why = "void", "no report tool"
            elif duty.kind in ("confirm", "tell_basket", "ask_before_change") and not roles.get("message"):
                duty.state, duty.why = "void", "no message tool"
        if roles.get("compare") and not any(d.kind == "compare" for d in self.items):
            self.add(Duty("compare two or three finalists", "compare", 2, "before_order", stated=False))
        # (a message to the shopper before the order is sent only when the request asks for one: the reading's
        # procedures above, or one of the request's own sentences that asks to be told or asked before the purchase)
        if roles.get("message") and not any(d.kind in ("confirm", "tell_basket") for d in self.items):
            said = self.said_before_order()
            if said:
                self.add(Duty(said[1][:200], said[0], None, "before_order"))
        self.event("duties: " + json.dumps([d.brief() + " [" + d.state + "]" for d in self.items]))

    def said_before_order(self) -> tuple[str, str] | None:
        """(kind, sentence) when a request sentence asks the assistant to tell or ask the shopper before the purchase:
        it addresses the shopper, names a telling verb, a before-word and the purchase. 'confirm' when it asks for an
        OK, 'tell_basket' when it asks only to be told."""
        for sentence in re.split(r"(?<=[.!?;])\s+|\n+", self.request):
            low = sentence.lower()
            if not (TELL_VERB.search(low) and BEFORE_WORD.search(low) and PURCHASE_WORD.search(low) and SHOPPER_WORD.search(low)) \
                    or NOT_TELL.search(low):
                continue
            kind = "confirm" if OK_WORD.search(low) else "tell_basket"
            return kind, re.sub(r"\s+", " ", sentence).strip()
        return None

    def add(self, duty: Duty) -> None:
        for old in self.items:
            if old.kind == duty.kind and old.kind != "other":
                if duty.count and (old.count or 0) < duty.count:
                    old.count = duty.count
                old.stated = old.stated or duty.stated
                return
        self.items.append(duty)

    def event(self, text_body: str) -> None:
        self.log.append({"kind": "duty", "note": text_body[:400]})

    # -- what results prove
    def observe(self, call_name: str, call_args: dict[str, Any], obs: Any, cart_version: int) -> None:
        roles = self.tools.roles
        role = next((r for r in ("details", "view", "compare", "hold", "report") if roles.get(r) == call_name), None)
        if not role:
            return
        fault = fault_of(obs, "read" if role != "hold" else "write")
        named = [str(v) for k, v in call_args.items() if isinstance(v, (str, int)) and not isinstance(v, bool) and id_param(k, self.tools.props(call_name).get(k))]
        for key, low, high in list_params((self.tools.defs.get(call_name) or {}).get("parameters")):
            sent = call_args.get(key)
            named += [str(v) for v in (sent if isinstance(sent, list) else []) if isinstance(v, (str, int))]
        named = list(dict.fromkeys(named))
        if not named and self.tools.id_key(call_name) and call_args.get(self.tools.id_key(call_name)) is not None:
            named = [str(call_args[self.tools.id_key(call_name)])]
        if fault:
            self.event(f"{role} {call_name} failed: {fault}")
            return
        if role in ("details", "view"):
            self.opened.update(named[:1])
        elif role == "compare":
            shown = as_text(obs, 60000)
            rows = []
            walk_dicts(obs, lambda d: rows.append(d) if len(d) >= 2 else None)
            if len(named) >= 2 and (sum(1 for i in named if i in shown) >= 2 or len(rows) >= 2):
                self.compared.append(named)
                self.event(f"compare -> table for {len(named)} listings")
        elif role == "hold" and named:
            self.held.add(named[0])
        elif role == "report":
            self.reported_at = cart_version

    def refresh(self, picks: list[str], cart_version: int, quote: Any) -> None:
        for duty in self.items:
            if duty.state in ("void", "failed"):
                continue
            was = duty.state
            if duty.kind == "survey":
                duty.state = "done" if len(self.opened) >= (duty.count or 1) else "open"
            elif duty.kind == "compare":
                shape = self.tools.compare_shape()
                need = min(max(2, duty.count or 2), (shape[3] if shape and shape[3] else 5))
                tables = [c for c in self.compared if len(c) >= need]
                fresh = [c for c in tables if not picks or set(picks) & set(c) or not duty.stated]
                duty.state = "done" if fresh or (tables and duty.tries >= 2) else "open"
            elif duty.kind == "hold":
                duty.state = "done" if picks and all(p in self.held for p in picks) else "open"
            elif duty.kind == "report":
                duty.state = "done" if self.reported_at >= 0 and (self.reported_at == cart_version or duty.tries >= 2) else "open"
            elif duty.kind in ("confirm", "tell_basket"):
                duty.state = "done" if self.told is not None and (self.told == quote or duty.tries >= 2) else "open"
            if was != duty.state:
                self.event(f"duty {duty.kind} -> {duty.state} {duty.why}".strip())

    def open_for(self, stage: str) -> list[Duty]:
        """Open steps that must be done before an order ('order') or a decline ('decline')."""
        out = []
        for duty in self.items:
            if duty.state != "open" or duty.kind not in GATING:
                continue
            if stage == "decline":
                if duty.kind == "report" and duty.timing in ("before_decline", "before_finish", "any"):
                    out.append(duty)
            elif duty.timing != "before_decline":
                out.append(duty)
        return out

    def failed(self, duty: Duty, why: str) -> None:
        duty.state, duty.why = "failed", why
        self.event(f"duty {duty.kind} -> failed {why}")

    def open_text(self) -> str:
        shown = [d for d in self.items if d.state == "open" and d.kind not in ("decline_wording",)]
        if not shown:
            return ""
        return "[procedure steps still open] " + "; ".join(d.brief() + (f" (tool {self.tools.roles.get(d.kind)})" if self.tools.roles.get(d.kind) else "")
                                                       for d in shown)

    def decline_words(self) -> str:
        """The wording a decline must use, when the request quotes one."""
        rule = next((d for d in self.items if d.kind == "decline_wording"), None)
        if rule is None:
            return ""
        quoted = [q.strip() for q in re.findall(r"[\"“”]([^\"“”]{3,160})[\"“”]", rule.text) if q.strip()]
        return quoted[0] if quoted else ""


# ================================================================================================================ errand
class Errand:
    def __init__(self, pdata: dict[str, Any]):
        environment = pdata["environment"]
        self.wire = Wire(environment["binding"])
        self.pview = environment["policy_view"]
        self.request = str(self.pview.get("query") or "")
        self.toolset = [tdef for tdef in self.pview.get("tools") or [] if isinstance(tdef, dict) and isinstance(tdef.get("function"), dict)]
        self.shapes = {tdef["function"]["name"]: tdef["function"].get("parameters") or {"type": "object", "properties": {}} for tdef in self.toolset}
        self.step_cap = int(self.pview.get("max_steps") or 30)
        self.wire.done = bool(self.pview.get("finished"))
        self.archive = Records()
        self.needs = Needs(self.wire, self.request)
        self.tools = ToolMap(self.toolset)
        self.tools.settle({})
        self.role_map: dict[str, str] = dict(self.tools.roles)
        self.trolley = Basket(self.tools, self.role("cart"))
        self.duties = Duties(self.request, self.tools, self.wire.log)
        self.line_of: dict[str, int] = {}          # listing -> product line (from the item check, or its words)
        self.duty_due: str | None = None           # 'order' / 'decline' when a finish was held for open steps
        self.hold_queue: list[str] = []            # listings to hold before they go in the cart
        self.hold_tried: set[str] = set()
        self.survey_tried: set[str] = set()
        self.late_notes: list[str] = []            # replies to lost calls that arrived on a later replay
        self.flag = False                          # the last result carried a notice, a refusal or a shopper line
        self.read_queue: list[str] = []            # listings the code opens itself (a survey short of its count)
        self.vetted_at: tuple = ()
        self.held_ids: set[str] = set()            # signatures of calls held once for an unseen id
        self.dead: dict[str, str] = {}             # listing -> why the store turned down putting it in (until a put is accepted)
        self.failed_cart = ""                      # signature of the newest cart call the store turned down
        self.msgs_sent = 0
        self.msg_cap: int | None = None
        self.orders_tried = 0
        self.refusal_n = 0
        self.order_vetted = False
        self.rank_checked: set[str] = set()
        self.decline_vetted = False
        self.audit_done = False
        self.questions_log: list[str] = []
        self.asking = True
        self.rulings: dict[str, str] = {}
        self.checked_adds: set[str] = set()
        self.passed: set[str] = set()              # item|version the independent check found meeting everything
        self.flaws: dict[str, int] = {}            # item|version -> problems the first check named (rejections only)
        self.adjudicated: set[str] = set()         # item|version whose rejection went to the second opinion
        self.audit_fits: list[str] = []            # the newest decline audit's possible fits
        self.refused_lines: set[str] = set()        # lines the cart held when a checkout was refused
        self.refusals_seen: dict[str, int] = {}     # item -> refusals counted when it was last put in
        self.warned: set[str] = set()
        self.word_seen: list[tuple] = []           # (argument and mode, other arguments, query words, total) of first pages
        self.word_kind = ""                        # 'any' / 'all': how this store's keyword search treats several words
        self.lead_seen: list[tuple] = []           # (argument and mode, other arguments, query words, total, row marks)
        self.lead_bounds: dict[tuple, list] = {}   # argument and mode -> [largest k proven read, k's proven unread]
        self.lead_words = 0                        # K when the store reads only the first K words of a query; 0 unknown
        self.lead_ctx: tuple | None = None         # the argument and mode K was learned in
        self.last_calls: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.stuck = 0
        self.read_at: dict[str, int] = {}          # listing -> step of its newest details read
        self.fix_queue: list[dict[str, Any]] = []  # cart calls the code sends itself to correct the cart
        self.win_queue: dict[int, list[str]] = {}     # product line -> the planner's ranked properties
        self.win_state: dict[int, dict[str, Any]] = {}
        self.win_sent: dict[int, int] = {}
        self.win_done: dict[int, set[str]] = {}
        self.fixing = False                        # one of them is being sent (the cart screen lets it through)
        self.fix_rounds = 0
        self.auto_order: dict[str, Any] | None = None   # an order the code sends itself once the cart is settled
        self.holds_n = 0                           # advisory finish holds so far (the ledger's release counts them)
        self.held_gate = ""                        # the gate the newest finish hold came from
        self.sold_lines: set[str] = set()          # listings a refusal or a record showed sold out
        self.rule_queue: list[tuple[dict[str, Any], str]] = []   # the call a rule audit found missing, run by duty_step
        self.rule_rounds: dict[tuple, int] = {}    # (stage, cart version, requirements version) -> audits run
        self.rule_clear: set[tuple] = set()        # those keys whose audit let the finish go
        self.rule_calls = 0                        # rule audit model calls so far
        self.rule_steps = 0                        # world calls the rule audit added
        self.rule_msgs = 0                         # messages the rule audit sent
        self.rule_sent: set[str] = set()           # signatures of the calls the rule audit sent
        self.handovers: dict[str, list[dict[str, Any]]] = {}   # listing -> its page's hand-overs to other listings
        self.page_reads: dict[tuple[str, str], int] = {}     # (page tool, listing) -> step of its newest read

    # ----------------------------------------------------------------------------------------------- small reads
    @property
    def log(self) -> list[dict[str, Any]]:
        return self.wire.log

    def moves_left(self) -> int:
        return self.step_cap - self.wire.tick

    def role(self, call_name: str) -> str | None:
        return self.role_map.get(call_name)

    def ask_limit(self) -> int:
        """Messages the planned questions may use: the talk limit, which already leaves out what the request's own steps
        still owe. Each message is one more property covered, and that is worth more than a spare for a reply. A report
        the request asks to be sent that no step keeps a message for keeps one back from the questions (the actor
        sends it)."""
        if self.msg_cap is None:
            return ASK_DEFAULT
        kept = 1 if REPORT_ASK.search(self.request.lower()) and not self.telling() \
            and not any(d.kind == "report" and d.state != "void" for d in self.duties.items) else 0
        return max(0, self.talk_limit() - kept)

    def talk_limit(self) -> int:
        """Messages the code may send before the order: the budget less one for each message step the request states
        that is still owed (its basket message or confirmation)."""
        if self.msg_cap is None:
            return ASK_DEFAULT + 1
        owed = sum(1 for d in self.duties.items if d.kind in ("confirm", "tell_basket") and d.state == "open")
        return max(0, self.msg_cap - owed)

    def telling(self) -> bool:
        """A message to the shopper about the basket is still owed before the order."""
        return any(d.kind in ("confirm", "tell_basket") and d.state == "open" for d in self.duties.items)

    def msg_reserved(self, queued: int = 0) -> bool:
        """One more message from the actor would spend one kept for the basket message the request states."""
        return self.msg_cap is not None and self.telling() and self.msgs_sent + queued + 1 > self.talk_limit()

    def call_kind(self, call_name: str, call_args: dict[str, Any] | None) -> str | None:
        """'order', 'decline' or None for a call. When one tool does both, its mode value tells which: the value the
        schema resolved, else the words of the values sent."""
        order, decline = self.role("order"), self.role("decline")
        if call_name not in (order, decline) or not call_name:
            return None
        if order != decline:
            return "order" if call_name == order else "decline"
        modes = self.tools.finish_modes
        if modes:
            key, buy, end = modes
            pspec = self.tools.props(call_name).get(key) or {}
            sent = (call_args or {}).get(key, pspec.get("default") if isinstance(pspec, dict) else None)
            if sent == end:
                return "decline"
            if sent == buy:
                return "order"
        for piece in (call_args or {}).values():
            if isinstance(piece, str) and len(piece) <= 40:
                if re.search(DECLINE_WORDS, piece.lower()):
                    return "decline"
                if re.search(ORDER_WORDS, piece.lower()):
                    return "order"
        return "order"

    def finishers(self) -> set[str]:
        return {self.role("order"), self.role("decline")} - {None}

    def status_line(self) -> str:
        spare = self.moves_left()
        pieces = [f"[status] steps used {self.wire.tick} of {self.step_cap}, {spare} left",
                 f"messages sent {self.msgs_sent}" + (f" of {self.msg_cap}" if self.msg_cap is not None else ""),
                 f"about {int(max(0, self.wire.spare() - LAST_CALL))} s left"]
        text_body = "; ".join(pieces) + ".\n[requirements now] " + self.needs.text_body()
        if self.trolley.used:
            text_body += "\n" + self.trolley.summary()
        money = self.budget_line()
        if money:
            text_body += "\n" + money
        steps_open = self.duties.open_text()
        if steps_open:
            text_body += "\n" + steps_open
        if spare <= 2:
            text_body += " LAST STEPS: order now if the cart holds a verified fit, otherwise decline."
        elif spare <= 5 or self.wire.spare() < CLOSING:
            text_body += " FINISH NOW: make the cart one verified listing per product at its units, re-check stock, then order (or decline if nothing fits)."
        elif self.orders_tried == 0 and spare <= self.order_due() + 7 and spare > self.order_due() and self.narrowing():
            text_body += f" Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when {self.order_due()} steps remain, and a refused order needs about 4 steps to repair."
        elif self.orders_tried == 0 and spare <= self.order_due() and self.narrowing():
            text_body += " Searching has ended (details only for the cart's line, listings a check named, or while nothing is verified): put your best verified fit in the cart, re-check its stock and price, and order; a refused order still needs about 4 steps to repair."
        return text_body

    # ----------------------------------------------------------------------------------------------- setup
    def read_request(self) -> None:
        tool_rows = [f"- {tdef['function']['name']}: {tdef['function'].get('description', '')} | arguments: {as_text(tdef['function'].get('parameters') or {})}" for tdef in self.toolset]
        fixed_part = READ_REQUEST + "\n\nTOOLS:\n" + "\n".join(tool_rows)
        var_part = "REQUEST:\n" + self.request
        if self.pview.get("opening") is not None:
            var_part += "\n\nOPENING OBSERVATION:\n" + as_text(self.pview.get("opening"))
        content = [{"type": "text", "text": fixed_part, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": var_part}]
        reading: dict[str, Any] = {}
        for mdl in self.wire.lineup(READER, False):
            msg = self.wire.query_model([{"role": "user", "content": content}], [mdl], 3500)
            reading = json_object((msg or {}).get("content") or "") or {}
            if isinstance(reading.get("roles"), dict) or isinstance(reading.get("items"), list):
                break
            self.wire.log.append({"kind": "setup_retry", "model": mdl})
            if self.wire.spare() < CLOSING + 60 or self.wire.credit_out:
                break
        # (the schema decides which tool plays which role; the reader's naming counts where the schema agrees)
        self.tools.settle(reading)
        self.role_map = dict(self.tools.roles)
        self.trolley = Basket(self.tools, self.role("cart"))
        self.needs.sweepable = bool(self.role("search")) and self.page_arg() is not None
        spend_cap = reading.get("message_budget")
        if is_number(spend_cap) and spend_cap >= 0:
            self.msg_cap = int(spend_cap)
        else:
            # (the reading missed it: a number written before "messages" in the request)
            told = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:more\s+|short\s+)?(?:messages?|questions?)\b",
                             self.request.lower())
            self.msg_cap = count_word(told.group(1)) if told else None
        self.needs.set_lines(reading)
        self.needs.reading = reading
        self.needs.written = [str(req) for req in reading.get("requirements") or [] if req]
        for ptr in self.needs.add_pointers(reading.get("references"), "request"):
            # (a requirement line that only restates the pointer is no value a check can read)
            marks = [w for w in ("listing", ptr["seller"].lower()) if w]

            def restates(req: Any, ptr: dict[str, Any] = ptr, marks: list[str] = marks) -> bool:
                return isinstance(req, str) and self.needs.same_prop(self.needs.prop_of(req), ptr["property"]) and any(m in req.lower() for m in marks)
            self.needs.written = [r for r in self.needs.written if not restates(r)]
            for ln in self.needs.lines:
                ln["requirements"] = [r for r in ln["requirements"] if not restates(r)]
            if isinstance(reading.get("requirements"), list):
                reading["requirements"] = [r for r in reading["requirements"] if not restates(r)]
            self.log.append({"kind": "pointer_read", "pointer": {k: ptr[k] for k in ("property", "kind", "seller", "facts", "for")}})
        self.duties.read(reading.get("procedures"), self.msg_cap)
        self.log.append({"kind": "roles", "roles": self.role_map, "finish_modes": self.tools.finish_modes,
                         "cart_modes": self.tools.cart_modes, "notes": self.tools.notes[:6]})
        self.needs.spend_cap = reading.get("price_cap")
        if reading.get("ask_first") and not (TRADE_CUE.search(self.request.lower()) or CHECK_BACK_CUE.search(self.request.lower())):
            # (a question the reading invented: only a request that says two wishes may clash, or asks to be consulted,
            # spends a message on it before choosing)
            self.log.append({"kind": "ask_first_dropped", "items": reading.get("ask_first")})
            reading["ask_first"] = []
        # (no budget question of the code's own: a request with no price limit is read as having none)
        step_count = reading.get("step_budget")
        if is_number(step_count) and 0 < step_count < self.step_cap:
            self.step_cap = int(step_count)

    # ----------------------------------------------------------------------------------------------- tools on offer
    def order_due(self) -> int:
        """Steps kept back for the first order attempt and the repair a refusal needs: a quarter of the errand's
        steps, at least FIRST_ORDER_BY and at most FIRST_ORDER_DUE. A longer errand whose procedure still owes steps at
        the finish (hold, compare, report, the basket message) keeps one more step for each, up to a third of its steps."""
        due = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, self.step_cap // 4 + 1))
        if self.step_cap // 4 + 1 > FIRST_ORDER_DUE:
            owed = sum(1 for d in self.duties.items if d.state == "open" and d.kind in ("hold", "compare", "report", "confirm", "tell_basket"))
            due = max(due, min(self.step_cap // 3, due + min(4, owed)))
        return due

    def narrowing(self) -> bool:
        return bool(self.role("order") and self.role("cart"))

    def offered(self) -> list[dict[str, Any]]:
        spare = self.moves_left()
        call_rank, decl, trolley = self.role("order"), self.role("decline"), self.role("cart")
        name_set: set[str] | None = None
        if spare <= 1:
            name_set = {call_rank, decl}
        elif spare <= 3:
            name_set = {call_rank, decl, trolley}
        elif (spare <= self.order_due() and self.orders_tried == 0) or (spare <= FIRST_ORDER_BY and self.refusal_n > 0):
            name_set = {call_rank, decl, trolley, self.role("cart_view"), self.role("view") or self.role("details")}
            # (a check that names listings to open, nothing verified yet, or a widening change: a few late reads stay)
            late = self.__dict__.get("late_n", 0) < 3
            if late and ((self.unverified() and spare > 5) or (self.named_open() and spare >= min(FIRST_ORDER_BY + 2, self.order_due()))):
                name_set.add(self.role("details"))
            if late and (self.widened_late() or self.__dict__.get("named_pages")) and spare > 4:
                name_set.add(self.role("search"))
        toolset = self.toolset
        if name_set is not None and call_rank and trolley:
            trimmed = [tdef for tdef in self.toolset if tdef["function"]["name"] in name_set]
            toolset = trimmed or toolset
        if (self.asking or self.msg_reserved()) and self.role("message"):
            # (questions are sent by the code; the last message is kept for the basket summary before the order)
            toolset = [tdef for tdef in toolset if tdef["function"]["name"] != self.role("message")]
        return toolset

    def in_reserve(self) -> bool:
        """The steps kept for the first order (or for the repair after a refusal) have begun."""
        spare = self.moves_left()
        return bool(self.role("order") and self.role("cart")) and spare > 3 and (
            (spare <= self.order_due() and self.orders_tried == 0) or (spare <= FIRST_ORDER_BY and self.refusal_n > 0))

    def unverified(self) -> bool:
        return not any(k.rpartition("|")[2] == str(self.needs.version) and k.rpartition("|")[0] not in self.refused_lines
                       for k in self.passed)

    def named_open(self) -> set[str]:
        """Listings a check named that were seen, are not read yet and were not in a refused cart."""
        return {i for i in self.__dict__.get("named", set()) if i in self.archive.ids} - self.archive.read - self.refused_lines

    def widened_late(self) -> bool:
        """The newest change widened the acceptable set, and it came within the steps the order keeps."""
        return self.needs.widened_at == self.needs.version and self.wire.tick - self.needs.widened_tick <= self.order_due()

    def begin_batch(self) -> None:
        self.late_q = 0

    def late_hold(self, call_name: str, call_args: dict[str, Any]) -> str:
        """In the reserve, a details read of a cart line or a verified fit (the stock re-check) always goes; any other read goes only while
        fewer than 3 late reads were made or queued in this reply, and only for a listing a check named, while nothing is
        verified, or (a search) right after a widening change."""
        details, search = self.role("details"), self.role("search")
        if call_name not in (details, search) or not self.in_reserve():
            return ""
        named = {val for val in call_args.values() if isinstance(val, str)}
        if call_name == details and named & (set(self.trolley.units) | {k.rpartition("|")[0] for k in self.passed}):
            return ""
        queued = self.__dict__.get("late_q", 0)
        spare = self.moves_left() - queued
        allowed = False
        if self.__dict__.get("late_n", 0) + queued < 3:
            if call_name == details:
                allowed = (self.unverified() and spare > 5) or bool(named & self.named_open()) and spare >= min(FIRST_ORDER_BY + 2, self.order_due())
            else:
                allowed = (self.widened_late() or json.dumps(call_args, sort_keys=True, default=str) in self.__dict__.get("named_pages", set())) and spare > 4
        if allowed:
            self.late_q = queued + 1
            return ""
        return ("Not run (no step used): the steps left are kept for the order. Now details are read only for the cart's line, a "
                "listing a check named, or (while nothing is verified) up to 3 listings in all: put your best verified fit in the "
                "cart, re-check its stock and price, and order.")

    # ----------------------------------------------------------------------------------------------- checks
    def field_names(self) -> str:
        """The property field names on the listing details read so far, spellings that differ only by separators
        merged, with how many listings carry each (the main field of a property is the one most listings carry)."""
        prop_list, _ = self.archive.prop_menu()
        if not prop_list:
            return ""
        return ("\n\nFIELD NAMES ON THIS STORE'S LISTINGS (separator-only spellings merged; how many listings read carry each):\n"
                + prop_list)

    def check_item(self, thing: str, aim_text: str) -> str:
        """The independent item check (cheap model); a rejection that decides the finish gets a second opinion from
        the stronger model, whose verdict stands. Returns the ruling the actor reads ('' when the item passes or no
        verdict was had)."""
        if self.wire.spare() < CLOSING + 20:
            return ""
        key = f"{thing}|{self.needs.version}"
        if key in self.rulings:
            ruling = self.rulings[key]
            if ruling and aim_text == "ORDER" and key not in self.adjudicated and self.adjudicable(thing, aim_text):
                # (an earlier cheap rejection now stands between the errand and its order: the second opinion is asked)
                archive = self.archive.about(thing)
                first = {"fits": False, "ruling": ruling}
                verdict = self.second_opinion(thing, aim_text, self.item_prompt(thing, aim_text, archive), first, archive) if archive else first
                if verdict is not first:
                    return self.rule(thing, verdict)
            return ruling
        archive = self.archive.about(thing)
        if not archive:
            return ""
        made = self.__dict__.get("checks_made", 0)
        if aim_text != "ORDER" and made >= max(CHECKS_MOST, 2 * len(self.needs.lines) + 4):
            # (the task's routine checks are spent: the item goes unverified, so the order keeps its full pre-order check)
            self.log.append({"kind": "verify_skipped", "item": thing})
            return ""
        if aim_text != "ORDER":
            self.checks_made = made + 1
        ask_text = self.item_prompt(thing, aim_text, archive)
        answer = self.wire.consult(ask_text, 3000, CHECKER)
        first_by = self.wire.answered_by
        self.log.append({"kind": "verify", "item": thing, "aim": aim_text, "by": first_by, "result": answer})
        self.place_line(thing, answer, archive)
        if isinstance(answer, dict) and answer.get("fits") is False:
            self.flaws[key] = len(self.problems_of(answer)) or 1
            if first_by in ADJUDICATOR:
                self.adjudicated.add(key)      # (the stronger model already gave this verdict as the fallback)
            elif self.adjudicable(thing, aim_text):
                answer = self.second_opinion(thing, aim_text, ask_text, answer, archive)
        return self.rule(thing, answer)

    def item_prompt(self, thing: str, aim_text: str, archive: list[str]) -> str:
        ask_text = CHECK_ITEM + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request + self.needs.changes()
        if self.needs.several():
            ask_text += "\n\nTHE ERRAND'S PRODUCT LINES:\n" + "\n".join(f"{i + 1}. {ln['quantity']} x {ln['product']}" + "".join(f"; {r}" for r in ln["requirements"])
                                                                  for i, ln in enumerate(self.needs.lines))
        ask_text += "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + self.needs.text_body()
        ask_text += f"\n\nTHE ITEM TO {aim_text}: {thing}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n" + "\n".join(archive)
        ask_text += self.handover_lines(thing)
        if self.trolley.last_view is not None:
            ask_text += "\n\nLATEST CART VIEW:\n" + as_text(self.trolley.last_view)[:3000]
        ask_text += self.field_names()
        label_lines = self.archive.label_lines()
        if label_lines:
            ask_text += "\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the item's category label must be the one that matches the request's wording as a whole, not a neighbouring one):\n" + label_lines
        return ask_text

    def place_line(self, thing: str, answer: dict[str, Any] | None, archive: list[str]) -> None:
        """The product line a check put the listing on, else (the first time) the line its records' words name."""
        line_no = (answer or {}).get("line")
        if is_number(line_no) and 1 <= line_no <= len(self.needs.lines):
            self.line_of[thing] = int(line_no) - 1
        elif thing not in self.line_of:
            guess = self.needs.line_for(" ".join(archive)[:3000])
            if guess is not None:
                self.line_of[thing] = guess

    def adjudicable(self, thing: str, aim_text: str) -> bool:
        """A cheap check's rejection decides the finish: it comes before an order, or at a cart add while the item's
        product line has no verified listing (and the item was not in a refused cart); the item is within the budget
        as last priced, not shown sold out and not turned down by the store. At most ADJUDICATE_MOST per task, under
        SPEND_HARD."""
        if self.__dict__.get("adjudged", 0) >= ADJUDICATE_MOST or self.wire.spend >= SPEND_HARD or self.wire.spare() < CLOSING + 20:
            return False
        idx = self.line_of.get(thing, 0 if not self.needs.several() else None)
        if aim_text != "ORDER" and (idx in self.verified_picks() or thing in self.refused_lines):
            return False
        cap, cost = self.needs.spend_cap, self.archive.price(thing)
        if is_number(cap) and is_number(cost) and cost * self.needs.line_qty(idx) > cap + 1e-9:
            return False
        return thing not in self.dead and not self.gone(thing)

    def second_opinion(self, thing: str, aim_text: str, ask_text: str, first: dict[str, Any], archive: list[str]) -> dict[str, Any]:
        """The same check asked of the stronger model; its verdict replaces the first one unless it is unreadable. Only
        a verdict that stands uses one of the task's ADJUDICATE_MOST second opinions."""
        self.adjudicated.add(f"{thing}|{self.needs.version}")
        second = self.wire.consult(ask_text, 3000, ADJUDICATOR, firm=True)
        stands = isinstance(second, dict) and isinstance(second.get("fits"), bool)
        self.log.append({"kind": "adjudicate", "item": thing, "aim": aim_text, "by": self.wire.answered_by,
                         "first": first, "second": second, "replaced": stands})
        if not stands:
            return first
        self.adjudged = self.__dict__.get("adjudged", 0) + 1
        self.place_line(thing, second, archive)
        return second

    def rule(self, thing: str, answer: dict[str, Any] | None) -> str:
        """Books a check's verdict under the requirements in force and returns the ruling the actor reads."""
        key = f"{thing}|{self.needs.version}"
        if not answer or answer.get("fits") is not False:
            self.rulings[key] = ""
            if answer:
                self.passed.add(key)
                self.__dict__.setdefault("pass_price", {})[thing] = self.archive.price(thing)
            return ""
        faults = self.problems_of(answer)[:5]
        ruling = f"An independent check says {thing} does NOT meet every requirement: " + "; ".join(faults) + ". Unless you are certain the check is wrong, choose another fully verified fit."
        self.rulings[key] = ruling
        return ruling

    @staticmethod
    def problems_of(answer: dict[str, Any]) -> list[str]:
        """A check's named problems (one written as a plain string counts as one)."""
        named = answer.get("problems")
        if isinstance(named, str):
            return [named] if named.strip() else []
        return [str(p) for p in named] if isinstance(named, list) else []

    def rank_check(self, thing: str) -> str:
        """For a cheapest or best request: listings in the records that may fit and beat the chosen one."""
        priority = self.needs.priority()
        # (asked again only when the requirements, the item's price or the budget changed, not on every refusal)
        key = f"{thing}|{self.needs.version}|{self.archive.price(thing)}|{self.needs.spend_cap}"
        if priority == "none" or key in self.rank_checked or self.wire.spare() < CLOSING + 30 or self.moves_left() < 4:
            return ""
        self.rank_checked.add(key)
        detail_note = self.needs.reading.get("priority_detail") or ""
        what = "cheapest (lowest price)" if priority == "cheapest" else f"best ({detail_note})" if detail_note else "best"
        ask_text = RANK_CHECK.replace("PRIORITY", what) + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request + self.needs.changes()
        ask_text += "\n\nCURRENT REQUIREMENTS:\n" + self.needs.text_body()
        ask_text += (f"\n\nTHE CHOSEN ITEM: {thing}\n\nLISTINGS SEEN (one card each; DETAILS = its full record was read, ROW = seen only "
                     "in search rows, fields unknown: a ROW on the requested category (whatever object its title names) that beats the chosen item and is not ruled out by "
                     "its title, price or stock MAY fit and is named" + ("; only listings priced below the chosen one are shown"
                     if priority == "cheapest" else "") + "):\n"
                     + "\n".join(self.archive.cards(20000, thing if priority == "cheapest" else "")))
        ask_text += self.field_names()
        answer = self.wire.consult(ask_text, 2500, CHECKER)
        self.log.append({"kind": "rank", "result": answer})
        # (a line that was in the cart when a checkout was refused is taken as gone: it is never offered as better)
        better = [str(x) for x in (answer or {}).get("better") or [] if isinstance(x, (str, int)) and str(x) != thing
                  and str(x) not in self.refused_lines][:4]
        self.__dict__.setdefault("named", set()).update(better)
        if not better:
            return ""
        return (f"A rank check thinks these listings may also fit and beat {thing} on the {what}: " + ", ".join(better)
                + (f" ({answer.get('reason')})" if answer.get("reason") else "")
                + ". Check them (details, stock, price) before ordering; if each fails, order the current pick.")

    def pool_open(self) -> bool:
        """Some product line with no verified pick has a read listing the order-stage check may still pass (see
        recheck_pool): only then is there something for the stronger model to judge before a decline."""
        picks = self.verified_picks()
        return any(self.recheck_pool(i) for i in range(max(1, len(self.needs.lines))) if i not in picks)

    def review_decline(self, early: bool = False, pooled: bool | None = None) -> str:
        """The decline audit; an early one (nothing verified late in the episode) does not count against the two
        audits a decline may trigger. A decline audit goes to the stronger model only while a read listing is left to
        judge (pooled, else pool_open now) and at most SONNET_AUDITS times per task; otherwise to the cheap checker."""
        reread = len(self.archive.read) > self.__dict__.get("audit_reads", 10 ** 9) and self.__dict__.get("audits", 0) < 2
        if (self.audit_done and not reread) or self.wire.spare() < CLOSING + 30:
            return ""
        self.audit_done = True
        if not early:
            self.audits = self.__dict__.get("audits", 0) + 1
        cards = self.archive.cards()
        self.audit_reads = len(self.archive.read)
        if not cards:
            return ""
        before = self.__dict__.get("audit_ids")
        fresh = sorted(self.archive.read - before) if before is not None else []
        self.audit_ids = set(self.archive.read)
        ask_text = BEFORE_DECLINE + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request + self.needs.changes()
        ask_text += "\n\nCURRENT REQUIREMENTS:\n" + self.needs.text_body()
        ask_text += "\n\nLISTINGS SEEN (one card each; DETAILS = its full record was read, ROW = seen only in search rows):\n" + "\n".join(cards)
        if fresh:
            ask_text += ("\n\nOPENED SINCE THE LAST CHECK (their DETAILS cards above are new: judge each of them first, field by field, "
                         "by the reading rules): " + ", ".join(fresh))
        ask_text += self.field_names()
        # (an audit a decline waits on goes to the stronger model first when a read listing is left to judge, once per
        # task; the early one, mid-errand, and the others stay cheap)
        dear = not early and self.__dict__.get("sonnet_audits", 0) < SONNET_AUDITS and (self.pool_open() if pooled is None else pooled)
        if dear:
            self.sonnet_audits = self.__dict__.get("sonnet_audits", 0) + 1
            answer = self.wire.consult(ask_text, 3000, AUDITOR, firm=True)
        else:
            answer = self.wire.consult(ask_text, 3000, CHECKER)
        self.log.append({"kind": "audit", "early": early, "dear": dear, "by": self.wire.answered_by, "result": answer})
        conforms = [str(x) for x in (answer or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        if answer:
            self.audit_fits = conforms
        unopened = [str(x) for x in (answer or {}).get("unopened") or [] if isinstance(x, (str, int))
                    and str(x) not in self.archive.read and str(x) in self.archive.ids][:6]
        shelves, cap, qty, hit_shelf = self.shelf_labels(), self.needs.spend_cap, self.needs.quantity(), self.__dict__.get("hit_shelf", {})
        hits = [i for i in sorted(self.__dict__.get("all_hits", []), key=lambda i: -self.__dict__.get("hit_score", {}).get(i, 0)) if i not in self.archive.read
                and not (shelves and hit_shelf.get(i) and hit_shelf.get(i) not in shelves)
                and not (is_number(cap) and is_number(self.archive.price(i)) and self.archive.price(i) * qty > cap)]
        unopened = (hits[:4] + [i for i in unopened if i not in hits])[:6]
        pages = self.open_pages(2) if self.moves_left() > 5 else []
        if pages:
            unopened = unopened[:4]
            self.__dict__.setdefault("named_pages", set()).update(json.dumps(p, sort_keys=True, default=str) for p in pages)
        self.__dict__.setdefault("named", set()).update(conforms + unopened)
        named_list = self.__dict__.setdefault("named_list", [])
        named_list += [i for i in conforms + unopened if i not in named_list]
        if not conforms and not unopened and not pages:
            return ""
        said = []
        if conforms:
            said.append("An independent check thinks these listings may meet every requirement: " + ", ".join(conforms)
                        + (f" ({answer.get('reason')})" if answer.get("reason") else "") + ".")
        if pages:
            said.append("These searches carry required values (or a price limit within the budget) and matched more listings than were "
                        "read, so fitting ones may sit on their unread pages: " + "; ".join(f"{self.role('search')} {json.dumps(p, ensure_ascii=False)}" for p in pages)
                        + ". Read those pages first (one step shows a whole page of rows) and open the in-budget rows of the requested kind they show.")
        if unopened:
            said.append("These listings of the requested kind were seen only in search rows and never opened, so their fields "
                        "are unchecked: " + ", ".join(unopened) + ". Open their details first (several in one reply).")
        return " ".join(said) + " Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."

    def named_unread(self) -> list[str]:
        """Listings an all-words search or a check named that are still unread: seen, within the budget as last priced, on
        the requested category when known, not in a refused cart; the all-words hits first, then in the order named."""
        cap, qty = self.needs.spend_cap, self.needs.quantity()
        shelves, hit_shelf = self.shelf_labels(), self.__dict__.get("hit_shelf", {})
        out = []
        ranks = self.__dict__.get("hit_score", {})
        ranked = sorted(self.__dict__.get("all_hits", []), key=lambda i: -ranks.get(i, 0))
        for ident in dict.fromkeys(ranked + list(self.__dict__.get("named_list", []))):
            cost = self.archive.price(ident)
            if ident in self.archive.read or ident in self.refused_lines or ident not in self.archive.ids \
                    or (is_number(cap) and is_number(cost) and cost * qty > cap) or (shelves and hit_shelf.get(ident) and hit_shelf[ident] not in shelves):
                continue
            out.append(ident)
        return out

    def details_call(self, ident: str) -> dict[str, Any] | None:
        """Arguments for opening one listing: an earlier details call's arguments with the listing id swapped in, else the
        details tool's one id-like text argument."""
        tool = self.role("details")
        for entry in reversed(self.log):
            for act in (entry.get("sent") or []) if entry.get("kind") == "env" else []:
                args = act.get("args") if act.get("name") == tool else None
                if isinstance(args, dict) and any(isinstance(v, str) and v in self.archive.ids for v in args.values()):
                    return {k: (ident if isinstance(v, str) and v in self.archive.ids else v) for k, v in args.items()}
        jschema = self.shapes.get(tool or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
        key = next((k for k in texts if k in (jschema.get("required") or [])), None) or (texts[0] if len(texts) == 1 else None)
        return {key: ident} if key else None

    def open_named(self) -> bool:
        """Sends the details reads a held decline queued (scripted, one reply), then the decline audit over the new records;
        the reads a recall pass queued (open_quiet) go without the audit."""
        queue = self.__dict__.pop("open_queue", None)
        quiet = self.__dict__.pop("open_quiet", False)
        if not queue or self.wire.done or self.wire.broke:
            return False
        calls = [(i, self.details_call(i)) for i in queue]
        proposals = [{"id": f"open_{n}_{self.wire.tick}", "type": "function",
                      "function": {"name": self.role("details"), "arguments": json.dumps(args)}} for n, (_, args) in enumerate(calls) if args]
        if not proposals:
            return False
        self.chat.append({"role": "assistant", "content": "I open the cheapest in-budget rows those pages showed." if quiet
                          else "Before declining, I open the listings the checks named.", "tool_calls": proposals})
        self.covering = True
        try:
            responses = self.dispatch(proposals, scripted=True)
        finally:
            self.covering = False
        if quiet:
            if responses:
                cid, text_body = responses[-1]
                responses[-1] = (cid, text_body + "\n[recall] Judge these records against every requirement by meaning (a field of "
                                 "the property's own name decides); buy one that fits the way the request's procedure says (cart at its "
                                 "units, stock and price re-read, then the order).")
            self.post_replies(responses)
            return True
        note = self.review_decline() if self.wire.spare() > CLOSING + 30 else ""
        if note and responses:
            cid, text_body = responses[-1]
            responses[-1] = (cid, text_body + "\n[check after opening] " + note)
        self.post_replies(responses)
        return True

    def cheaper_fit(self, thing: str) -> str:
        """Under a cheapest rule: the lowest-priced listing the item check passed under the current requirements that
        costs less than the chosen one (a line of a refused cart is taken as gone)."""
        if self.needs.priority() != "cheapest" or not thing:
            return ""
        mine = self.archive.price(thing)
        if mine is None:
            return ""
        best: tuple[str, float] | None = None
        mine_line = self.line_of.get(thing, 0 if not self.needs.several() else None)
        for key in self.passed:
            ident, _, version = key.rpartition("|")
            if version != str(self.needs.version) or ident == thing or ident in self.refused_lines:
                continue
            if self.needs.several() and (mine_line is None or self.line_of.get(ident) != mine_line):
                continue
            if self.needs.several() and self.needs.lines[mine_line]["priority"] != "cheapest":
                continue
            cost = self.archive.price(ident)
            if cost is not None and cost < mine and (best is None or cost < best[1]):
                best = (ident, cost)
        return best[0] if best else ""

    def chosen(self) -> str | None:
        """The line the order would buy: the one booked line, else the newest booked one."""
        if self.trolley.units:
            return list(self.trolley.units)[-1]
        return None

    def cart_empty(self) -> bool:
        """The store has a cart, its order tool buys what the cart holds, and the cart is known to hold nothing: no
        accepted cart call, or the counted ones emptied it (not when an accepted call named no listing the code read)."""
        call_rank = self.role("order")
        if not self.role("cart") or not call_rank or self.tools.ids(call_rank) or self.trolley.units:
            return False
        return not self.trolley.blind

    def verified(self, ident: str) -> bool:
        return f"{ident}|{self.needs.version}" in self.passed and ident not in self.refused_lines and ident not in self.dead

    def verified_picks(self) -> dict[int, str]:
        """Product line -> the listing the code would order for it: a verified listing in the cart, else the verified
        listing for that line (the cheapest under a cheapest rule, else the one checked last)."""
        found: dict[int, str] = {}
        for idx, things in self.trolley.lines_of(self.needs, self.line_of).items():
            # (a cart line shown sold out is no pick: the forced finish then swaps it for a verified one, see forced_move)
            good = [t for t in things if self.verified(t) and not self.line_sold(t)]
            if good:
                found[idx] = good[0]
        cheap = self.needs.priority() == "cheapest"
        for key in reversed(list(self.passed)):
            ident = key.rpartition("|")[0]
            if not self.verified(ident) or self.line_sold(ident):
                continue
            idx = 0 if not self.needs.several() else self.line_of.get(ident)
            if idx is None or idx in found and found[idx] in self.trolley.units:
                continue
            prior = found.get(idx)
            here, there = self.archive.price(ident), self.archive.price(prior) if prior else None
            if prior is None or (cheap and is_number(here) and is_number(there) and here < there):
                found[idx] = ident
        return found

    # ----------------------------------------------------------------------------------------------- questions
    def text_arg(self, the_tool: str) -> str | None:
        jschema = self.shapes.get(the_tool or "") or {}
        prop_specs = jschema.get("properties") or {}
        req_names = list(jschema.get("required") or [])
        str_params = [fkey for fkey, val in prop_specs.items() if isinstance(val, dict) and (val.get("type") == "string" or "string" in (val.get("type") or [])) and "enum" not in val]
        aim = next((fkey for fkey in req_names if fkey in str_params), None) or (str_params[0] if len(str_params) == 1 else None)
        if not aim or any(fkey != aim for fkey in req_names):
            return None
        return aim

    def message_call(self, text_body: str, cid: str) -> dict[str, Any] | None:
        the_tool = self.role("message")
        aim = self.text_arg(the_tool) if the_tool else None
        if not aim:
            return None
        call_args, faults = ArgCheck.execute({aim: text_body}, self.shapes[the_tool])
        if faults:
            return None
        return {"id": cid, "type": "function", "function": {"name": the_tool, "arguments": json.dumps(call_args)}}

    def pose_planned(self, urge: bool = False, pace: bool = False) -> bool:
        """Sends the next planned question, one at a time: each question is chosen after the answers to the earlier
        ones were read, so a 'don't mind' or a revealed need steers the next pick. Questions start once a few listings'
        details were read (or at once when something must be learned from the shopper first). With several products
        they keep pace with the products read, so that no product's share goes on another's properties: `pace` keeps
        that even when something else urges a question (a cart move can wait for no question; a finish waits for all)."""
        the_tool = self.role("message")
        ceiling = self.ask_limit()
        if not the_tool or self.msgs_sent >= ceiling:
            self.asking = False
            return False
        if self.wire.tick == 0:
            return False     # (the errand never opens with a message)
        sets = self.archive.rich_reads()
        # (what the request says must be settled with the shopper goes out once one listing was read, never as the
        # opening move)
        pressing = bool(self.needs.reading.get("ask_first")) and not self.questions_log and sets >= 1
        thin = sets < ASK_AFTER_SETS and not pressing and not urge
        wait_until = min(ASK_AFTER_STEPS, self.step_cap // 4)
        if thin and (self.wire.tick < wait_until or (self.wire.tick < wait_until + ASK_GRACE_STEPS and self.moves_left() > 20)):
            return False
        if self.moves_left() <= 10 or self.wire.spare() < CLOSING + 40:
            self.asking = False
            return False
        waiting = self.__dict__.get("plan_wait")
        if waiting is not None and sets <= waiting and not urge:
            return False
        many = len(self.needs.lines)
        read_now = self.products_read()
        # (with listings read and none of them traceable to a product, the count says nothing, so it holds nothing back)
        blind = read_now == 0 and sets >= ASK_AFTER_SETS
        if many > 1 and not blind and (pace or not urge) and self.moves_left() > 20 and self.msgs_sent >= (ceiling * read_now + many - 1) // many:
            return False
        last = self.__dict__.get("last_asked")
        if self.needs.garbled and last and not last.get("again"):
            # (the shopper did not follow the last question: the same property once more, short and with no examples)
            self.needs.garbled = False
            last["again"] = True
            prop_words = " ".join(re.findall(r"[a-z0-9]+", str(last.get("property") or "").lower())[:4])
            product = " ".join(re.findall(r"[a-z0-9]+", str((self.needs.lines[0] if len(self.needs.lines) == 1 else {}).get("product") or "").lower())[:2])
            short = f"Which {prop_words} do you want" + (f" for the {product}?" if product and len(prop_words.split()) + len(product.split()) <= 2 else "?")
            proposal = self.message_call(short, f"again_{self.msgs_sent + 1}_{self.wire.tick}")
            if proposal and prop_words:
                self.chat.append({"role": "assistant", "content": f"The shopper did not follow; I ask about {prop_words} again, shorter.", "tool_calls": [proposal]})
                responses = self.dispatch([proposal], scripted=True)
                self.post_replies(responses)
                for _, text_body in responses[:1]:
                    self.questions_log.append(f"- {last.get('property')}: asked again \"{short}\" -> {text_body.split(chr(10) + '[status]')[0][:400]}")
                self.asking = self.msgs_sent < ceiling
                return True
        if not pressing and self.send_window(ceiling, urge, sets, read_now):
            return True
        remaining = ceiling - self.msgs_sent
        scripted = [] if self.__dict__.get("queue_for") != read_now else self.__dict__.get("ask_queue") or []
        scripted = [qitem for qitem in scripted if not self.asked_before(str(qitem.get("property") or ""), str(qitem.get("question") or ""))
                    and not self.settled(str(qitem.get("property") or ""))]
        if scripted:
            return self.send_planned(scripted, ceiling, urge, sets)
        ask_text = ASK_PLAN.replace("QUESTION_COUNT", str(remaining)) + NEXT_ONE.replace("REMAINING", str(remaining))
        ask_text += "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
        ask_text += "\n\nREQUIREMENTS SO FAR:\n" + self.needs.text_body()
        if self.questions_log:
            ask_text += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.questions_log)
        if self.needs.reading.get("ask_first"):
            ask_text += "\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in self.needs.reading.get("ask_first") or [])
        prop_list = self.archive.menu_by_shelf([ln["product"] for ln in self.needs.lines]) or self.archive.prop_menu()[0]
        ask_text += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR, per category (name, number of listings, example values):\n" + (prop_list or "(none seen yet)")
        # the first plan reads the story's hints, which is where the needs show; the later ones mostly follow it
        answer = self.wire.consult(ask_text, 2500, PLANNER if self.questions_log else PLANNER[::-1])
        scripted = [qitem for qitem in (answer or {}).get("questions") or [] if isinstance(qitem, dict) and str(qitem.get("question") or "").strip()]
        # (a second name for a property already asked costs a message and finds nothing: 'gender', then 'recommended gender')
        scripted = [qitem for qitem in scripted if not self.asked_before(str(qitem.get("property") or ""), str(qitem.get("question") or ""))]
        # (a property the request or an earlier answer already settles is never asked again)
        scripted = [qitem for qitem in scripted if not self.settled(str(qitem.get("property") or ""))]
        self.queue_for = read_now
        return self.send_planned(scripted, ceiling, urge, sets)

    def menu_props(self, idx: int) -> list[str]:
        """The properties the listings read carry for one product line, as the menu shows them (most carried first):
        the menu's section whose category label shares the most words with the product (the whole menu when it has
        no sections)."""
        menu = self.archive.menu_by_shelf([ln["product"] for ln in self.needs.lines]) or self.archive.prop_menu()[0]
        sections: dict[str, list[str]] = {}
        label = ""
        for row in menu.splitlines():
            head = re.match(r"category '(.*)' \(", row) or re.match(r'category "(.*)" \(', row)
            if head:
                label = head.group(1)
                continue
            item = re.match(r"- (.+?) \(in \d+ listings?\)", row)
            if item:
                sections.setdefault(label, []).append(item.group(1).strip())
        if len(sections) <= 1:
            return next(iter(sections.values()), [])
        product = word_stems(self.needs.lines[idx]["product"]) if idx < len(self.needs.lines) else set()
        best = max(sections, key=lambda lab: (words_meet(product, word_stems(lab)), len(sections[lab])))
        return sections[best] if words_meet(product, word_stems(best)) else []

    def window_plan(self, read_now: int) -> None:
        """Ranks, once per set of products read, the properties to name in windows: the planner reads the story's hints
        and the listings' keys; each product line gets its own ranked list."""
        if self.__dict__.get("win_plans", 0) >= 2:
            return
        self.win_plans = self.__dict__.get("win_plans", 0) + 1
        ask_text = WINDOW_PLAN + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
        ask_text += "\n\nREQUIREMENTS SO FAR:\n" + self.needs.text_body()
        if self.questions_log:
            ask_text += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.questions_log)
        prop_list = self.archive.menu_by_shelf([ln["product"] for ln in self.needs.lines]) or self.archive.prop_menu()[0]
        ask_text += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR, per category (name, number of listings, example values):\n" + (prop_list or "(none seen yet)")
        answer = self.wire.consult(ask_text, 1500, PLANNER if self.questions_log else PLANNER[::-1])
        queues: dict[int, list[str]] = {}
        for entry in (answer or {}).get("properties") or []:
            if not isinstance(entry, dict) or not str(entry.get("property") or "").strip():
                continue
            idx = self.needs.line_for(str(entry.get("product") or "")) if self.needs.several() else 0
            name = re.sub(r"\s+", " ", str(entry["property"])).strip().lower()[:50]
            if idx is not None and name not in queues.setdefault(idx, []):
                queues[idx].append(name)
        self.win_queue = queues
        self.win_plan_for = read_now

    def next_window(self, idx: int) -> list[str]:
        """Up to WINDOW properties of a product line not named yet: the planner's order first, then the menu's."""
        done = self.win_done.setdefault(idx, set())
        out: list[str] = []
        for name in list(self.win_queue.get(idx, [])) + self.menu_props(idx):
            low = name.lower()
            if low in done or low in out or self.settled(name) or re.search(WINDOW_SKIP, low):
                continue
            out.append(low)
            if len(out) >= WINDOW:
                break
        return out

    def window_text(self, idx: int, window: list[str]) -> str:
        product = str(self.needs.lines[idx]["product"]).strip() if idx < len(self.needs.lines) else ""
        listed = window[0] if len(window) == 1 else ", ".join(window[:-1]) + " or " + window[-1]
        return f"What {listed} would you like" + (f" for the {product}?" if product else "?")

    def send_single(self, ceiling: int, urge: bool, sets: int) -> bool:
        """After the shopper asked for one thing at a time: one property per message, in v010's wording, for the rest of
        the task. The window it was asked about goes on first, its first property next; then the product lines take
        turns (fewest questions first, among those whose listings were seen), each line's queued window properties in
        their order before its next ranked ones. False when no line has a property left (v010's planner goes on, one
        property per message)."""
        queue = self.__dict__.setdefault("single_first", [])
        sent = self.__dict__.setdefault("single_sent", {})
        count = max(1, len(self.needs.lines))
        order = sorted(range(count), key=lambda i: (self.win_sent.get(i, 0) + sent.get(i, 0), i))
        if self.needs.several():
            seen = [i for i in order if any(words_meet(word_stems(self.needs.lines[i]["product"]), word_stems(lab))
                                            for lab in self.archive.shelves().values())]
            order = seen or order
        first = self.__dict__.pop("single_next", None)
        if first is not None:
            order = [first] + [i for i in order if i != first]
        for idx in order:
            for name in [n for i, n in queue if i == idx] + self.next_window(idx):
                if (idx, name) in queue:
                    queue.remove((idx, name))
                if self.settled(name) or self.asked_before(name, f"for the {self.needs.lines[idx]['product']}" if idx < len(self.needs.lines) else ""):
                    continue
                self.win_done.setdefault(idx, set()).add(name.lower())
                sent[idx] = sent.get(idx, 0) + 1
                product = str(self.needs.lines[idx]["product"]).strip() if idx < len(self.needs.lines) else ""
                return self.send_planned([{"property": name, "question": f"Which {name} would you like" + (f" for the {product}?" if product else "?")}],
                                         ceiling, urge, sets)
        return False

    def send_window(self, ceiling: int, urge: bool, sets: int, read_now: int) -> bool:
        """One message naming a window of up to WINDOW properties of one product line, the line with the fewest
        windows first. The shopper answers about the first property it has a point on: a need disclosed (or a "don't
        mind" for one named property) settles the properties named before it and that one, and the rest lead that
        line's next window; an answer that settles nothing sends the same window reversed, once, and then the next.
        False (the one-question planner goes on) when no line has two properties left to name."""
        if self.__dict__.get("win_single"):
            return self.send_single(ceiling, urge, sets)
        if not self.win_queue or self.__dict__.get("win_plan_for") != read_now:
            self.window_plan(read_now)
        count = max(1, len(self.needs.lines))
        order = sorted(range(count), key=lambda i: (self.win_sent.get(i, 0), i))
        if self.needs.several():
            read = [i for i in order if any(words_meet(word_stems(self.needs.lines[i]["product"]), word_stems(lab))
                                            for lab in self.archive.shelves().values())]
            order = read or order
        for idx in order:
            state = self.win_state.setdefault(idx, {"window": [], "reversed": False})
            window = list(state["window"]) or self.next_window(idx)
            if not state["window"]:
                state["reversed"] = False
            if len(window) >= 2:
                break
        else:
            return False
        proposal = None
        while len(window) >= 2 and proposal is None:
            text = self.window_text(idx, window)
            proposal = self.message_call(text, f"win_{self.msgs_sent + 1}_{self.wire.tick}")
            if proposal is None:
                window = window[:-1]         # (longer than the tool takes: a shorter window)
        if proposal is None:
            return False
        needs_before, free_before = set(self.needs.uncovered), set(self.needs.indifferent)
        self.chat.append({"role": "assistant", "content": f"I will ask the shopper about {', '.join(window)}.", "tool_calls": [proposal]})
        responses = self.dispatch([proposal], scripted=True)
        self.post_replies(responses)
        self.win_sent[idx] = self.win_sent.get(idx, 0) + 1
        # (what this answer added: hear may rebuild its lists, so not a slice)
        fresh = [n for n in self.needs.uncovered if n not in needs_before]
        spared = [f for f in self.needs.indifferent if f not in free_before]

        def named(prop: str, label: str) -> bool:
            have, want = word_stems(prop), word_stems(label)
            return bool(have and want and (have <= want or want <= have)) or Needs.same_prop(prop, label)
        said = " ".join(text_body for _, text_body in responses[:1]).split("\n[note]")[0]
        if not fresh and (self.needs.garbled or ONE_AT_A_TIME.search(said.lower())):
            # (the shopper did not follow a window, or asked for one thing at a time: the rest of the task asks one
            # property per message, this window's properties first and in their order; no reversed resend)
            self.win_single = True
            self.needs.garbled = False
            self.single_first = [(idx, name) for name in window]
            self.single_next = idx
            state["window"], state["reversed"] = [], False
            self.questions_log.append(f"- nothing settled: asked \"{text}\" -> {said[:400]}")
            self.log.append({"kind": "window", "line": idx, "window": window, "disclosed": [], "free": spared, "cut": None, "one_at_a_time": True})
            self.asking = self.msgs_sent < ceiling
            return True
        # (a need on a property outside the window settles none of the window's properties)
        hit = next((i for i, label in enumerate(window) if any(named(Needs.prop_of(need) or need, label) for need in fresh)), None)
        frees = [i for i, label in enumerate(window) if any(named(str(f), label) for f in spared)]
        if hit is not None:
            # (a need disclosed for a property is no "don't mind" for it)
            self.needs.indifferent = [f for f in self.needs.indifferent if not named(str(f), window[hit])]
        cut = hit if hit is not None else (frees[0] if len(frees) == 1 and not self.needs.garbled else None)
        done = self.win_done.setdefault(idx, set())
        if cut is not None:
            done.update(window[:cut + 1])
            rest = window[cut + 1:]
            state["window"] = (rest + [n for n in self.next_window(idx) if n not in rest])[:WINDOW] if rest else []
            state["reversed"] = False
        elif not state["reversed"]:
            state["window"] = list(reversed(window))
            state["reversed"] = True
        else:
            done.update(window)
            state["window"] = []
            state["reversed"] = False
        # (logged under the properties this answer settled, so a property named but not settled can still be asked
        # one at a time; the question and the answer stay in the line for the planner)
        settled = window[:cut + 1] if cut is not None else (window if not state["window"] else [])
        for _, text_body in responses[:1]:
            self.questions_log.append(f"- {', '.join(settled) or 'nothing settled'}: asked \"{text}\" -> {text_body.split(chr(10) + '[status]')[0][:400]}")
        self.log.append({"kind": "window", "line": idx, "window": window, "disclosed": fresh, "free": spared, "cut": cut})
        self.asking = self.msgs_sent < ceiling
        return True

    def send_planned(self, scripted: list[dict[str, Any]], ceiling: int, urge: bool, sets: int) -> bool:
        """Sends the first question of the list that can be sent, and keeps the rest of the list for the next turn."""
        proposal = None
        waiting = list(scripted)
        while waiting and not proposal:
            thing = waiting.pop(0)
            proposal = self.message_call(str(thing.get("question")).strip(), f"ask_{self.msgs_sent + 1}_{self.wire.tick}")
            if proposal:
                label = (str(thing.get("property")), str(thing.get("question")).strip())
                self.last_asked = {"property": label[0]}
        self.ask_queue = waiting
        if not proposal:
            if self.moves_left() > 12 and not urge:
                self.plan_wait = sets     # (planned again once another listing is opened)
                return False
            self.asking = False
            return False
        self.chat.append({"role": "assistant", "content": f"I will ask the shopper about {label[0]}.", "tool_calls": [proposal]})
        responses = self.dispatch([proposal], scripted=True)
        self.post_replies(responses)
        for _, text_body in responses[:1]:
            self.questions_log.append(f"- {label[0]}: asked \"{label[1]}\" -> {text_body.split(chr(10) + '[status]')[0][:400]}")
        self.asking = self.msgs_sent < ceiling
        return True

    def products_read(self) -> int:
        """How many requested products have a listing read, by the words of its category or title."""
        labels = self.archive.shelves()
        seen = [word_stems(labels.get(ident, "")) | word_stems(self.title_of(ident)) for ident in self.archive.read]
        return sum(1 for ln in self.needs.lines if any(words_meet(word_stems(ln["product"]), words) for words in seen))

    def asked_before(self, prop: str, question: str = "") -> bool:
        """The property, or another name for it, has had its question: one name's words hold the other's. In a basket the
        same property of another product is a question of its own, when both questions show which product they are on."""
        product = set().union(*[word_stems(ln["product"]) for ln in self.needs.lines]) if self.needs.lines else set()
        words = word_stems(prop) - product or word_stems(prop)
        several = len(self.needs.lines) > 1
        here = self.needs.line_for(question) if several and question else None
        for line in self.questions_log:
            earlier, _, rest = line.partition(":")
            other = word_stems(earlier.strip("- ")) - product or word_stems(earlier.strip("- "))
            if not (words and other and (words <= other or other <= words)):
                continue
            there = self.needs.line_for(rest.split('" ->', 1)[0]) if several else None
            if here is None or there is None or here == there:
                return True
        return False

    def settled(self, prop: str) -> bool:
        """A property the requirements in force or the shopper's 'don't mind' answers already cover: its words and
        a requirement's property words agree (one set holds the other)."""
        product = {w[:-1] if len(w) > 3 and w.endswith("s") else w for ln in self.needs.lines for w in re.findall(r"[a-z0-9]+", ln["product"].lower())}

        def words(text_body: str) -> set[str]:
            return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", text_body.lower())
                    if w not in LINK_WORDS} - product
        asked = words(prop)
        if not asked:
            return False
        if self.needs.pointed(prop):
            return True     # (asked again it points at the same listing)
        names = [r.split(":", 1)[0] for r in self.needs.written + self.needs.uncovered + (self.needs.now_reqs or []) if ":" in r]
        names += list(self.needs.indifferent)
        return any(have and asked <= have for have in (words(n) for n in names))

    def reply_back(self, text_body: str) -> bool:
        # (a reply may use the spare message; with no spare it never takes the last planned question's message)
        spare_kept = self.talk_limit() > self.ask_limit()
        if not self.role("message") or self.msgs_sent >= self.talk_limit() or self.moves_left() <= 4 or self.wire.spare() < CLOSING \
                or (self.asking and not spare_kept and self.msgs_sent >= self.ask_limit() - 1 and self.moves_left() > 10
                    and self.wire.spare() >= CLOSING + 40):
            return False
        proposal = self.message_call(text_body, f"reply_{self.wire.tick}")
        if not proposal:
            return False
        self.chat.append({"role": "assistant", "content": "I will answer the shopper's question.", "tool_calls": [proposal]})
        self.post_replies(self.dispatch([proposal], scripted=True))
        return True

    def post_replies(self, responses: list[tuple[str, str]]) -> None:
        for idx, (cid, text_body) in enumerate(responses):
            if idx == len(responses) - 1:
                if self.late_notes:
                    text_body += "\n" + "\n".join(self.late_notes)
                    self.late_notes = []
                text_body += "\n" + self.status_line()
            self.chat.append({"role": "tool", "tool_call_id": cid, "content": text_body})

    # ----------------------------------------------------------------------------------------------- finishing
    def plain_args(self, call_name: str, text_body: str, kind: str = "") -> dict[str, Any]:
        """Arguments the code fills itself for a finishing call: the resolved mode value for an order or a decline,
        an enum value reading the same way (never one reading the other way), the text for free-text fields."""
        jschema = self.shapes.get(call_name) or {}
        prop_specs = jschema.get("properties") or {}
        call_args: dict[str, Any] = {}
        words = DECLINE_WORDS if kind == "decline" else ORDER_WORDS if kind == "order" else ""
        against = ORDER_WORDS if kind == "decline" else DECLINE_WORDS if kind == "order" else ""
        modes = self.tools.finish_modes if call_name == self.role("order") == self.role("decline") else None
        if modes and kind in ("order", "decline"):
            call_args[modes[0]] = modes[1] if kind == "order" else modes[2]
        for key in jschema.get("required") or []:
            pspec = prop_specs.get(key) or {}
            if key in call_args:
                continue
            if pspec.get("enum"):
                fit = [val for val in pspec["enum"] if words and isinstance(val, str) and re.search(words, spoken(val))]
                neutral = [val for val in pspec["enum"] if not (against and isinstance(val, str) and re.search(against, spoken(val)))]
                call_args[key] = fit[0] if fit else (neutral or pspec["enum"])[0]
            elif pspec.get("type") == "string" or "string" in (pspec.get("type") or []):
                call_args[key] = text_body[: pspec.get("maxLength", len(text_body))]
            elif pspec.get("type") == "boolean":
                call_args[key] = True
            elif pspec.get("type") in ("integer", "number"):
                call_args[key] = max(1, pspec.get("minimum", 1))
        if len(prop_specs) == 1:
            key, pspec = next(iter(prop_specs.items()))
            if key not in call_args and isinstance(pspec, dict) and pspec.get("type") == "string" and "enum" not in pspec:
                call_args[key] = text_body[: pspec.get("maxLength", len(text_body))]
        return call_args

    def scrub_decline(self, call_args: dict[str, Any], call_name: str) -> dict[str, Any]:
        if self.needs.reading.get("decline_no_names") is False:
            return call_args
        named = any(tok and tok in self.archive.ids
                    for piece in call_args.values() if isinstance(piece, str)
                    for tok in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", piece))
        if not (self.needs.reading.get("decline_no_names") or named):
            return call_args
        prop_specs = (self.shapes.get(call_name) or {}).get("properties") or {}
        return {key: (SORRY[: (prop_specs.get(key) or {}).get("maxLength", len(SORRY))]
                      if isinstance(piece, str) and "enum" not in (prop_specs.get(key) or {}) else piece)
                for key, piece in call_args.items()}

    def decline_move(self) -> dict[str, Any] | None:
        decl = self.role("decline")
        if not decl:
            return None
        quoted = self.duties.decline_words()
        words = (quoted.rstrip(".") + ". " + SORRY) if quoted else SORRY
        return {"name": decl, "args": self.scrub_decline(self.plain_args(decl, words, "decline"), decl)}

    def order_move(self) -> dict[str, Any] | None:
        call_rank = self.role("order")
        if not call_rank:
            return None
        call_args = self.plain_args(call_rank, "Please place the order for the basket.", "order")
        pick = next(iter(self.verified_picks().values()), None) or self.chosen()
        key = self.tools.id_key(call_rank) if self.tools.ids(call_rank) else None
        if key and pick:
            # (a store whose order tool names the listing itself, with no cart)
            call_args[key] = pick
            count = self.tools.count_key(call_rank)
            if count:
                call_args[count] = self.needs.line_qty(self.line_of.get(pick))
        return {"name": call_rank, "args": call_args}

    def forced_move(self, refusals: int) -> dict[str, Any] | None:
        """The code's next finishing move: order the cart when it holds the verified basket; put the verified picks in
        first when it lacks them (taking out a listing a verified one replaces, while steps allow); send the basket
        message the procedure owes when a step is left for it; decline only when nothing passed the item check, after
        two refused attempts, or when the cart is empty and nothing can be put in (an empty cart is never ordered)."""
        left = self.moves_left()
        cart = dict(self.trolley.units)
        if refusals >= 2 or not self.role("order"):
            return self.decline_move() or self.order_move()
        # (a listing the store would not put in, or whose newest record shows it sold out, is no pick: verified_picks)
        picks = self.verified_picks()
        if self.role("cart") and self.trolley.lines_key and left >= 2:
            on_line = self.trolley.lines_of(self.needs, self.line_of)
            target = {t: q for t, q in cart.items() if not any(t in things and idx in picks for idx, things in on_line.items())}
            for idx, ident in sorted(picks.items()):
                target[ident] = float(self.needs.line_qty(idx))
            lines = self.trolley.line_list(target)
            put = {self.trolley.lines_key: lines}
            if lines is not None and target and target != cart and self.role("cart") + as_text(put) != self.failed_cart:
                return {"name": self.role("cart"), "args": put}
        if self.role("cart") and self.trolley.id_key:
            on_line = self.trolley.lines_of(self.needs, self.line_of)
            missing = [(idx, ident) for idx, ident in sorted(picks.items()) if ident not in cart]
            stale = [t for idx, things in on_line.items() for t in things if idx in picks and picks[idx] != t]
            if stale and left >= 2 + len(missing) + 1:
                undo = self.trolley.take_args(stale[0], cart.get(stale[0], 1))
                if undo is not None:
                    return {"name": self.role("cart"), "args": undo}
            # (every missing put and the order; with an empty cart one put and the order are enough to try)
            if missing and left >= 2 and (not cart or left >= len(missing) + 1):
                idx, ident = missing[0]
                put = self.trolley.put_args(ident, self.needs.line_qty(idx))
                if self.role("cart") + as_text(put) != self.failed_cart:
                    return {"name": self.role("cart"), "args": put}
        if self.cart_empty():
            return self.decline_move() or self.order_move()
        told = self.basket_message(left)
        if told is not None:
            return told
        if cart or picks or (self.trolley.used and self.trolley.view_units(self.trolley.last_view) is None):
            return self.read_or_order() if cart else self.order_move()
        return self.decline_move() or self.order_move()

    def basket_message(self, left: int) -> dict[str, Any] | None:
        """The basket summary the procedure owes before the order, as the code's own move: only for a cart with no
        faults, with a message left in the budget and a step left for the order after it."""
        duty = next((d for d in self.duties.items if d.kind in ("confirm", "tell_basket") and d.state == "open" and d.tries < 2), None)
        if duty is None or left < 2 or not self.trolley.units or self.trolley.faults(self.needs, self.line_of) \
                or (self.msg_cap is not None and self.msgs_sent >= self.msg_cap):
            return None
        self.duties.refresh(self.picks(), self.trolley.version, self.quote())
        if duty.state != "open":
            return None
        made = self.message_call(self.basket_text(duty.kind == "confirm" or duty.stated), f"forced_tell_{self.wire.tick}")
        if not made:
            return None
        duty.tries += 1
        return {"name": made["function"]["name"], "args": json.loads(made["function"]["arguments"])}

    def finish_now(self) -> None:
        """The code-only finish: no model call, one action per step, at most a few steps."""
        refusals = 0
        for _ in range(5):
            if self.wire.done or self.wire.broke or self.moves_left() <= 0 or self.wire.hard_spare() < 8:
                return
            move = self.forced_move(refusals)
            if move is None:
                return
            sent_before = self.msgs_sent
            resp, text_body = self.send_one(move["name"], move["args"])
            self.log.append({"kind": "forced_finish", "action": move, "finished": self.wire.done, "note": text_body[-300:]})
            if self.msgs_sent > sent_before:
                self.duties.told = self.quote()
                self.duties.refresh(self.picks(), self.trolley.version, self.quote())
            if resp is None and self.wire.pending is None:
                return
            if self.call_kind(move["name"], move["args"]) and not self.wire.done:
                refusals += 1

    # ----------------------------------------------------------------------------------------------- observations
    def refused(self, call_name: str, obs: Any, resp: dict[str, Any]) -> bool:
        """An order call whose episode did not finish (unless the fault is about the call's own tool or arguments, so
        the store never weighed the order), or any result holding a refusal-named field."""
        if isinstance(obs, dict) and any(re.search(r"\b(?:refus\w*|reject\w*)\b", spoken(k)) and v not in (None, False, "", [], {}) for k, v in obs.items()):
            return True
        if call_name != self.role("order") or resp.get("finished"):
            return False
        return not ARG_FAULT.search(fault_words(fault_of(obs, "write")))

    def notes_for(self, call_name: str, obs: Any, refused: bool) -> str:
        remarks = []
        fault = fault_words(fault_of(obs, "talk")) if call_name == self.role("message") else ""
        if fault and SPENT_FAULT.search(fault):
            # (the store turned a message down for the budget: no more messages, whatever count the request gave)
            self.msg_cap = self.msgs_sent if self.msg_cap is None else min(self.msg_cap, self.msgs_sent)
            remarks.append("That budget is used up; do not call this tool again.")
        if refused and call_name == self.role("decline") and getattr(self, "last_kind", None) == "decline":
            # (the store turned the decline down: it holds that something can still be bought - the audit runs again)
            self.audit_done = False
            self.decline_vetted = False
            remarks.append("THE DECLINE WAS REFUSED and the episode continues: the store does not accept that nothing fits. Read the refusal, apply what it says, look again at the closest candidates (their own properties, stock and price against the requirements now), and order the one that fits; decline again only if each one truly fails.")
        elif refused:
            self.refusal_n += 1
            self.order_vetted = False
            self.refused_lines.update(self.trolley.units)
            self.mark_sold(obs)
            remarks.append("THE ORDER WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (take out any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again.")
        elif isinstance(obs, dict) and len(obs) >= 2 and call_name != self.role("message"):
            lns: list[str] = []
            for piece in obs.values():
                if isinstance(piece, list) and piece and all(isinstance(val, (str, dict)) for val in piece) and any(isinstance(val, str) for val in piece):
                    long_strings([val for val in piece if isinstance(val, str)], lns)
            if lns:
                remarks.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(lns[:4]))
        return " ".join(remarks)

    @staticmethod
    def shelf_of(row: dict[str, Any]) -> str:
        """The category label a record carries ('' when none)."""
        return next((val.strip().lower() for key, val in row.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                     and re.search(r"categor|shelf|department|aisle", str(key).lower()) and "id" not in str(key).lower()), "")

    @staticmethod
    def sold_out(row: dict[str, Any]) -> bool:
        """A record whose stock field says it cannot be bought now."""
        for key, val in row.items():
            name = str(key).lower()
            if not re.search(r"stock|avail|on.?hand", name):
                continue
            if re.search(r"out|sold", name):
                if val is True:
                    return True
            elif val is False or (is_number(val) and val <= 0) or (
                    isinstance(val, str) and re.search(r"\b(?:out|sold|none|unavailable|not|no|false)\b", val.lower())):
                return True
        return False

    def shelf_labels(self) -> set[str]:
        """The category label(s) most of the listings read so far sit on; empty while none is known."""
        tally: dict[str, int] = {}
        counted: set[str] = set()

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.archive.read), None)
            shelf = self.shelf_of(d) if ident and ident not in counted else ""
            if shelf:
                counted.add(ident)
                tally[shelf] = tally.get(shelf, 0) + 1
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        top = max(tally.values(), default=0)
        return {shelf for shelf, n in tally.items() if n == top}

    @staticmethod
    def row_mark(row: dict[str, Any]) -> str:
        """A result row's identity: its id (a key named id, else one ending in id), else the whole row."""
        for exact in (True, False):
            for key, val in row.items():
                name = str(key).lower()
                if (name == "id" if exact else name.endswith("id")) and isinstance(val, (str, int)) and not isinstance(val, bool):
                    return str(val)
        return json.dumps(row, sort_keys=True, default=str)

    def search_keys(self, call_args: dict[str, Any]) -> tuple[list[tuple], tuple, bool, set]:
        """(free-text arguments as (key, words in order, repeats dropped), the mode (the enum arguments), first page?,
        the paging keys) of a search call. A word is a whitespace word with its outer punctuation stripped; a text is
        kept only when every word is plain letters or digits of 2+ characters (stores split hyphens, slashes and
        decimals and drop 1-letter words in different ways, so other words cannot be counted). A paging
        argument (a page, an offset; not a page size or a price) is on the first page when it is at most its schema
        minimum (else its default, else 0, or 1 for a page number)."""
        props = self.shapes.get(self.role("search") or "")
        props = props.get("properties") if isinstance(props, dict) else None
        props = props if isinstance(props, dict) else {}

        def spec(key: str) -> dict[str, Any]:
            pspec = props.get(key)
            return pspec if isinstance(pspec, dict) else {}
        texts, mode, first, paging = [], [], True, set()
        for key, val in call_args.items():
            name = str(key).lower()
            if isinstance(val, str) and spec(key).get("enum"):
                mode.append((key, val))
            elif isinstance(val, str):
                words = tuple(dict.fromkeys(w for w in (t.strip("\"'()[]{},.;:!?|/") for t in val.lower().split())
                                            if re.search(r"[a-z0-9]", w)))
                if words and all(re.fullmatch(r"[a-z0-9]{2,}", w) for w in words):
                    texts.append((key, words))
            elif is_number(val) and re.search(r"page|offset|first|start|skip|from|cursor", name) \
                    and not re.search(r"size|per_?page|limit|price|cost", name):
                paging.add(key)
                low = spec(key).get("minimum", spec(key).get("default"))
                first = first and val <= (low if is_number(low) else 1 if "page" in name else 0)
        return texts, tuple(sorted(mode)), first, paging

    def word_rule(self, call_args: dict[str, Any], obs: dict[str, Any], rows: list) -> str:
        """'any' when a query that extends another (the same words first, then more) matched more listings, 'all' when
        it matched fewer, '' while unknown; same mode and other arguments, first pages, per free-text argument. Only
        extensions count: a store that reads only the first words of a query gives unrelated totals for reordered words."""
        texts, mode, first, paging = self.search_keys(call_args)
        if not first:
            return self.word_kind
        total = max([val for val in obs.values() if isinstance(val, int) and not isinstance(val, bool)] + [len(rows)])
        for key, words in texts:
            ctx = (key, mode)
            others = tuple(sorted((k2, str(v2)) for k2, v2 in call_args.items() if k2 != key and k2 not in paging))
            for seen_ctx, seen_others, seen_words, seen_total in self.word_seen:
                if seen_ctx != ctx or seen_others != others or seen_total == total:
                    continue
                if len(words) > len(seen_words) and words[: len(seen_words)] == seen_words:
                    self.word_kind = "any" if total > seen_total else "all"
                elif len(seen_words) > len(words) and seen_words[: len(words)] == words:
                    self.word_kind = "any" if total < seen_total else "all"
            self.word_seen.append((ctx, others, words, total))
        return self.word_kind

    def lead_rule(self, call_args: dict[str, Any], obs: dict[str, Any], rows: list) -> int:
        """K when the store reads only the first K words of a query, 0 while unknown. First-page searches in one mode that
        share every other argument are compared, per free-text argument: a pair agreeing on its first k words shows the
        same total and rows when no word after the k-th was read (K <= k), and different ones when one was (K > k). K is
        claimed when the bounds meet, which also shows the argument is read in that mode. A 'same' pair counts only when
        one query extends the other, and one at or below a proven 'read' depth is a coincidence (word variants, words
        every listing on a shelf carries). Only queries of plain words count (see search_keys)."""
        texts, mode, first, paging = self.search_keys(call_args)
        if not first:
            return self.lead_words
        total = max([val for val in obs.values() if isinstance(val, int) and not isinstance(val, bool)] + [len(rows)])
        marks = tuple(self.row_mark(r) for r in rows)
        for key, words in texts:
            ctx = (key, mode)
            others = tuple(sorted((k2, str(v2)) for k2, v2 in call_args.items() if k2 != key and k2 not in paging))
            bound = self.lead_bounds.setdefault(ctx, [-1, set()])
            for seen_ctx, seen_others, seen_words, seen_total, seen_marks in self.lead_seen:
                if seen_ctx != ctx or seen_others != others or seen_words == words:
                    continue
                k = 0
                while k < min(len(words), len(seen_words)) and words[k] == seen_words[k]:
                    k += 1
                if total != seen_total or marks != seen_marks:
                    bound[0] = max(bound[0], k)
                elif k and len(set(marks)) >= 3 and k == min(len(words), len(seen_words)):
                    bound[1].add(k)
            self.lead_seen.append((ctx, others, words, total, marks))
        claims = sorted((min(k for k in high if k > low), ctx) for ctx, (low, high) in self.lead_bounds.items()
                        if any(k > low for k in high))
        claims = [(k, ctx) for k, ctx in claims if k == self.lead_bounds[ctx][0] + 1]
        self.lead_words, self.lead_ctx = claims[0] if claims else (0, None)
        return self.lead_words

    def page_arg(self) -> tuple[str, bool, float, float | None] | None:
        """The search tool's paging argument from its schema: (name, counts whole pages?, first value, highest or None)."""
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        for key, pspec in (props if isinstance(props, dict) else {}).items():
            name = str(key).lower()
            if not isinstance(pspec, dict) or pspec.get("type") not in ("integer", "number") \
                    or not re.search(r"page|offset|first|start|skip|from|cursor", name) or re.search(r"size|per_?page|limit|price|cost", name):
                continue
            by_page = "page" in name and not re.search(r"offset|skip|first|start|from", name)
            low = pspec.get("minimum", pspec.get("default"))
            low = low if is_number(low) else (1 if by_page else 0)
            return key, by_page, low, (pspec.get("maximum") if is_number(pspec.get("maximum")) else None)
        return None

    @staticmethod
    def stem(word: str) -> str:
        return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word

    def shelf_rows(self) -> dict[str, tuple[float, str, str]]:
        """In-stock rows seen on the requested category, within the budget: id -> (price, title, maker)."""
        shelves, cap, qty = self.shelf_labels(), self.needs.spend_cap, self.needs.quantity()
        found: dict[str, tuple[float, str, str]] = {}
        if not shelves:
            return found

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.archive.ids), None)
            if not ident or ident in self.refused_lines or self.shelf_of(d) not in shelves:
                return
            if self.sold_out(d):
                found.pop(ident, None)
                return
            cost = next((val for key, val in d.items() if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower())), None)
            title = next((val for key, val in d.items() if isinstance(val, str) and 0 < len(val) <= 300 and re.search(r"title|heading|name", str(key).lower())
                          and not re.search(r"categor|shelf|brand|make|manufactur|vendor|id", str(key).lower())), "")
            maker = " ".join(val for key, val in d.items() if isinstance(val, str) and re.search(r"brand|make|manufactur|vendor", str(key).lower()))
            if cost is not None:
                found[ident] = (float(cost), title, maker)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        for ident, (cost, title, maker) in list(found.items()):
            newest = self.archive.price(ident)
            cost = newest if newest is not None else cost
            if is_number(cap) and cost * qty > cap:
                found.pop(ident)
            else:
                found[ident] = (float(cost), title, maker)
        return found

    def label_queries(self) -> list[str]:
        """Keyword queries of the requested shelf label's own words: its first two, then the last word of each part (the
        kind it names), then its other words."""
        out: list[str] = []
        for label in sorted(self.shelf_labels()):
            parts, cur = [], []
            for word in label.lower().split():
                word = word.strip(".,;:!?'\"()[]")
                if word in LINK_WORDS or not re.fullmatch(r"[a-z0-9][a-z0-9'-]*", word):
                    if cur:
                        parts.append(cur)
                    cur = []
                else:
                    cur.append(word)
            if cur:
                parts.append(cur)
            words = [w for part in parts for w in part]
            heads = [part[-1] for part in parts]
            out += ([" ".join(words[:2])] if len(words) >= 2 else []) + heads + [w for w in words if w not in heads]
        return list(dict.fromkeys(out))

    def kind_queries(self, pairs: bool) -> list[str]:
        """Words naming kinds of object on the shelf, from the titles of the in-stock shelf rows seen: two neighbouring
        title words (pairs) or single words, of 3+ letters each, found in 2+ distinct titles, that are not function words
        or the row's maker and not all words of the label; the one on the cheapest such row first."""
        label = {self.stem(w) for s in self.shelf_labels() for w in re.findall(r"[a-z0-9]+", s)}
        titles: dict[str, tuple[float, str]] = {}
        for cost, title, maker in self.shelf_rows().values():
            low = title.lower().strip()
            if low:
                titles[low] = (min(cost, titles.get(low, (cost, ""))[0]), maker.lower())
        count: dict[str, int] = {}
        cheapest: dict[str, float] = {}
        shown: dict[str, str] = {}
        for title, (cost, maker) in titles.items():
            makers = set(re.findall(r"[a-z]+", maker))
            words = re.findall(r"[a-z]+", title)
            plain = [len(w) >= 3 and w not in LINK_WORDS and w not in makers for w in words]
            met: set[str] = set()
            grams = [(words[i], words[i + 1]) for i in range(len(words) - 1) if plain[i] and plain[i + 1]] if pairs                 else [(words[i],) for i in range(len(words)) if plain[i]]
            for gram in grams:
                base = " ".join(self.stem(w) for w in gram)
                if all(self.stem(w) in label for w in gram) or base in met:
                    continue
                met.add(base)
                count[base] = count.get(base, 0) + 1
                cheapest[base] = min(cheapest.get(base, cost), cost)
                shown.setdefault(base, " ".join(gram))
        return [shown[b] for b in sorted((b for b in count if count[b] >= 2), key=lambda b: (cheapest[b], -count[b], b))]

    def sweep_next(self, turn: int) -> tuple[dict[str, Any] | None, bool]:
        """(the next shelf slice to read, is it a page of the shelf listing?): the listing's next readable page while one
        is left, then in turn a kind pair, a kind word and a label query (first pages of queries not yet run, in the mode
        the actor's keyword searches used). Measured offline: the listing's pages first, then kind words from the titles
        ranked by their cheapest carrier, beat label-word queries page for page."""
        found = self.page_arg()
        shelves = self.shelf_labels()
        book = self.__dict__.get("page_book", {})
        if not found or not shelves or not book:
            return None, False
        key, _, low, _ = found
        tried = self.__dict__.get("sweep_tried", set())
        lists = [w for w in book.values() if w.get("id_order") and w["shelves"] and w["shelves"] <= shelves and self.next_page(w) is not None]
        listing = max(lists, key=lambda w: w["total"], default=None)
        ranked = [w for w in book.values() if w.get("id_order") is False and self.free_text(w["args"])]
        ranked = ranked or [w for w in book.values() if w.get("id_order") is None and self.free_text(w["args"])]
        template, text_key = None, None
        if ranked:
            props = (self.shapes.get(self.role("search") or "") or {}).get("properties") or {}
            args = ranked[-1]["args"]

            def fixed(k: str, v: Any) -> bool:
                return not isinstance(v, str) or bool((props.get(k) or {}).get("enum")) or v.strip().lower() in shelves
            texts = [k for k, v in args.items() if not fixed(k, v)]
            text_key = self.lead_ctx[0] if self.lead_ctx and self.lead_ctx[0] in texts else max(texts, key=lambda k: len(args[k].split()), default=None)
            template = {k: v for k, v in args.items() if k != text_key and fixed(k, v)} if text_key else None
        done = {tuple(self.stem(w) for w in re.findall(r"[a-z0-9]{2,}", " ".join(self.free_text(w["args"])).lower())[:2])
                for w in book.values() if low in w["read"] and w.get("id_order") is not True}

        def query(text: str) -> dict[str, Any] | None:
            sig = tuple(self.stem(w) for w in re.findall(r"[a-z0-9]{2,}", text.lower())[:2])
            if template is None or not sig or sig in done:
                return None
            args = {**template, text_key: text}
            return None if json.dumps(args, sort_keys=True, default=str) in tried else args
        sources = [
            lambda: [{**listing["args"], key: self.next_page(listing)}] if listing else [],
            lambda: [q for q in (query(t) for t in self.kind_queries(True)[:6]) if q],
            lambda: [q for q in (query(t) for t in self.kind_queries(False)[:6]) if q],
            lambda: [q for q in (query(t) for t in self.label_queries()) if q],
        ]
        for pick in [0] + [1 + (turn + shift) % 3 for shift in range(3)]:
            picks = [a for a in sources[pick]() if json.dumps(a, sort_keys=True, default=str) not in tried]
            if picks:
                return picks[0], pick == 0
        return None, False

    def sweep_step(self) -> bool:
        """Under a cheapest rule, once a change left no property requirement: reads one more slice of the shelf (scripted),
        while the steps an order and one repair need stay free; after the last one, lists the cheapest rows seen."""
        nd = self.needs
        if nd.priority() != "cheapest" or not nd.let_go or nd.props_left() or not self.role("search") or self.wire.done:
            return False
        keep = 3 + (1 if self.trolley.units else 0)     # (take the old line out, put the pick in, re-check it, order)
        if self.orders_tried == 0:
            keep = max(keep + 3, FIRST_ORDER_BY)        # (and the repair a refused first order needs)
        swept = self.__dict__.get("swept", 0)
        if swept >= SWEEP_READS or self.moves_left() <= keep or self.wire.spare() < CLOSING + 40:
            return False
        turn = self.__dict__.get("sweep_turn", 0)
        call_args, listing = self.sweep_next(turn)
        if call_args is None:
            return False
        self.swept = swept + 1
        self.sweep_turn = turn + (0 if listing else 1)
        self.__dict__.setdefault("sweep_tried", set()).add(json.dumps(call_args, sort_keys=True, default=str))
        proposal = {"id": f"sweep_{self.swept}_{self.wire.tick}", "type": "function",
                    "function": {"name": self.role("search"), "arguments": json.dumps(call_args)}}
        self.chat.append({"role": "assistant", "content": "No property requirement is left, so the cheapest in-stock listing on the "
                          "whole shelf wins: I read one more slice of the shelf before choosing.", "tool_calls": [proposal]})
        self.covering = True
        try:
            responses = self.dispatch([proposal], scripted=True)
        finally:
            self.covering = False
        more = (self.swept < SWEEP_READS and self.moves_left() > keep and self.wire.spare() >= CLOSING + 40
                and self.sweep_next(self.sweep_turn)[0] is not None)
        if not more and responses:
            rows = sorted(self.shelf_rows().items(), key=lambda kv: (kv[1][0], kv[0]))[:6]
            if rows:
                cid, text_body = responses[-1]
                responses[-1] = (cid, text_body + "\n[shelf sweep done] The cheapest in-stock rows on the requested category within "
                                 "the budget, from everything seen: " + "; ".join(f"{i} '{t[:80]}' {c:g}" for i, (c, t, _) in rows)
                                 + ". The lowest-priced one is the pick unless a change rules it out: make it the cart's only "
                                 "line, re-check its stock and price, and order it.")
        self.post_replies(responses)
        return True

    @staticmethod
    def result_rows(obs: Any) -> list:
        """The listing rows of a search result: its first non-empty list of dicts, one level down when the result is
        wrapped with a notice or a shopper line; [] when none."""
        def rows_of(piece: dict[str, Any]) -> list | None:
            lists = [val for val in piece.values() if isinstance(val, list) and all(isinstance(r, dict) for r in val)]
            return next((val for val in lists if val), lists[0] if lists else None)
        obs = _as_read(obs)
        if not isinstance(obs, dict):
            return []
        rows = rows_of(obs)
        if rows is None:
            inner = next((val for val in obs.values() if isinstance(val, dict) and rows_of(val) is not None), None)
            rows = rows_of(inner) if inner is not None else None
        return rows or []

    def told_basket(self) -> bool:
        """The basket message the procedure owes went out for the cart as it is: only the order may follow."""
        return any(d.kind in ("confirm", "tell_basket") and d.state == "done" for d in self.duties.items)

    def recall_ready(self, floor: int) -> bool:
        """A recall pass may start: passes left in the task, none yet under these requirements, a search with pages and
        a details tool, at least floor steps left, time to spare, and no basket message waiting for its order."""
        return (self.__dict__.get("recalls", 0) < RECALL_PASSES and f"recall|{self.needs.version}" not in self.warned
                and bool(self.role("search")) and bool(self.role("details")) and self.page_arg() is not None
                and self.moves_left() >= floor and self.wire.spare() > CLOSING + 40 and not self.told_basket()
                and not self.wire.done and not self.wire.broke)

    def recall_pages(self) -> list[dict[str, Any]]:
        """The unread pages a recall pass reads (searches already run, only the paging value changed): first the next
        page of each walk that showed a listing since lost (turned down at the cart, in a refused cart, or sold out after
        it passed its check), then open_pages; none the code already tried."""
        found = self.page_arg()
        if not found:
            return []
        tried = self.__dict__.get("recall_tried", set())
        lost = set(self.dead) | set(self.refused_lines) | {t for t in self.__dict__.get("pass_price", {}) if self.gone(t)}
        first = []
        for walk in self.__dict__.get("page_book", {}).values():
            nxt = self.next_page(walk) if lost & walk.get("marks", set()) else None
            if nxt is not None:
                first.append({**walk["args"], found[0]: nxt})
        pages: list[dict[str, Any]] = []
        seen = set(tried)
        for args in first + self.open_pages(RECALL_PAGES + 2):
            sig = json.dumps(args, sort_keys=True, default=str)
            if sig not in seen:
                seen.add(sig)
                pages.append(args)
        return pages[:RECALL_PAGES]

    def recall_hold(self) -> str:
        """At a decline with RECALL_FLOOR_DECLINE steps or more left: when searches already run have unread pages that
        matter, the decline is held (no step) and the next turn's recall_step reads them and opens their best rows."""
        if not self.recall_ready(RECALL_FLOOR_DECLINE):
            return ""
        pages = self.recall_pages()
        if not pages:
            return ""
        self.warned.add(f"recall|{self.needs.version}")
        self.recalls = self.__dict__.get("recalls", 0) + 1
        self.recall_due = pages
        self.log.append({"kind": "recall", "at": "decline", "pages": pages})
        return ("Not run yet (no step used). Before any decline, your searches still have unread pages that may hold a fit: the "
                "code reads " + ("that page" if len(pages) == 1 else "those pages") + " now and opens the cheapest in-budget, in-stock "
                "rows of the requested kind they show. Judge those records against every requirement by meaning, buy one that fits, "
                "and decline only if each one fails.")

    def recall_step(self) -> bool:
        """A code-run recall pass (scripted, at most RECALL_PAGES page reads, then RECALL_OPENS details reads queued for
        open_named): the one a decline hold asked for, else, with nothing verified and nothing in the cart once half the
        steps are used (not while questions are being asked), one that keeps RECALL_FLOOR_MID steps (and the order's
        reserve) free. The opened rows are the cheapest unread, in-stock, in-budget rows of the requested kind that
        the pages read in this pass showed."""
        due = self.__dict__.pop("recall_due", None)
        if due is not None:
            pages, floor, why = due, 4, "decline"
        else:
            if self.asking or self.trolley.units or self.verified_picks() or self.wire.tick * 2 < self.step_cap or self.in_reserve():
                return False
            floor = max(RECALL_FLOOR_MID, self.order_due() + 2)
            if not self.recall_ready(floor + 2):
                return False
            pages = self.recall_pages()
            if not pages:
                return False
            self.warned.add(f"recall|{self.needs.version}")
            self.recalls = self.__dict__.get("recalls", 0) + 1
            why = "mid"
            self.log.append({"kind": "recall", "at": why, "pages": pages})
        if self.wire.done or self.wire.broke:
            return False
        pages = pages[:max(0, min(RECALL_PAGES, self.moves_left() - floor - 1))]
        if not pages:
            return False
        tried = self.__dict__.setdefault("recall_tried", set())
        tried.update(json.dumps(a, sort_keys=True, default=str) for a in pages)
        tool = self.role("search")
        proposals = [{"id": f"recall_{n}_{self.wire.tick}", "type": "function", "function": {"name": tool, "arguments": json.dumps(a)}}
                     for n, a in enumerate(pages)]
        self.chat.append({"role": "assistant", "content": ("Before any decline, I read the unread pages of the searches that matter."
                          if why == "decline" else "Nothing verified fits yet at half the steps, so I read the unread pages of the "
                          "searches that matter before searching on."), "tool_calls": proposals})
        start = len(self.log)
        self.covering = True
        try:
            responses = self.dispatch(proposals, scripted=True)
        finally:
            self.covering = False
        rows: list = []
        for entry in self.log[start:]:
            sent = (entry.get("sent") or [{}]) if entry.get("kind") == "env" else [{}]
            if isinstance(sent[0], dict) and sent[0].get("name") == tool and isinstance(entry.get("reply"), dict):
                rows += self.result_rows(entry["reply"].get("observation"))
        hits = [i for i in dict.fromkeys(self.row_hits(rows)) if i not in self.dead and i not in self.refused_lines and not self.gone(i)]
        hits.sort(key=lambda i: (self.archive.price(i) if is_number(self.archive.price(i)) else 1e18, i))
        queue = hits[:max(0, min(RECALL_OPENS, self.moves_left() - floor))]
        self.log.append({"kind": "recall_rows", "at": why, "rows": len(rows), "hits": hits[:8], "opens": queue})
        if queue:
            self.open_queue = queue
            self.open_quiet = True
            note = ("[recall] The code opens the cheapest unread, in-stock, in-budget rows of the requested kind these pages show "
                    "next: " + ", ".join(queue) + ".")
        else:
            note = "[recall] These pages show no unread, in-stock row of the requested kind within the budget."
        if responses:
            cid, text_body = responses[-1]
            responses[-1] = (cid, text_body + "\n" + note)
        self.post_replies(responses)
        return True

    # ----------------------------------------------------------------------------------------------- pointer needs
    @staticmethod
    def squash(text_body: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", text_body.lower())

    def seller_text(self, rec: dict[str, Any]) -> str:
        """The record's seller-like fields (seller, vendor, brand, maker, shop ...) joined; '' when it has none."""
        found: list[str] = []

        def visit(d: dict[str, Any]) -> None:
            for key, val in d.items():
                name = spoken(str(key))
                if isinstance(val, str) and 0 < len(val) <= 80 and SELLER_KEY.search(name) and "id" not in name.split():
                    found.append(val)
        walk_dicts(rec, visit)
        return " ".join(found)

    def from_seller(self, rec: dict[str, Any], seller: str) -> bool | None:
        """True when the record names the seller (in a seller-like field, or anywhere when it has none), False when its
        seller-like fields name another one, None when it shows no seller at all."""
        want = self.squash(seller)
        if not want:
            return True
        fields = self.squash(self.seller_text(rec))
        if fields:
            return want in fields or (len(fields) >= 4 and fields in want)
        return True if want in self.squash(as_text(rec)) else None

    @staticmethod
    def fact_met(body: str, fact: str) -> bool:
        """A 'property: value' detail shows in a record's text: each part of the value (split at 'and', commas,
        slashes), its numbers as whole numbers and its words with separators ignored."""
        value = fact.split(":", 1)[1] if ":" in fact else fact
        parts = [p.strip() for p in re.split(r"\band\b|[,;/]", value.lower()) if p.strip()]
        flat = re.sub(r"[^a-z0-9]+", "", body)
        for part in parts:
            nums = re.findall(r"\d+(?:\.\d+)?", part)
            words = re.sub(r"[^a-z]+", "", re.sub(r"\d+(?:\.\d+)?", " ", part))
            if nums and not all(re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", body) for n in nums):
                return False
            if not nums and words and words not in flat:
                return False
        return bool(parts)

    def prop_value(self, rec: Any, prop: str) -> str:
        """The value a record gives for a property: the field named with the property's words (or holding them, or held
        by them), or a {name, value} pair naming it; list values joined; '' when none."""
        want = [self.stem(w) for w in re.findall(r"[a-z0-9]+", prop.lower()) if w not in LINK_WORDS]
        if not want:
            return ""

        def same(a: str, b: str) -> bool:
            return a == b or (len(a) >= 5 and edit_distance(a, b, 2) <= 1)
        found: list[tuple[int, int, str]] = []

        def offer(key: str, val: Any) -> None:
            if isinstance(val, bool) or isinstance(val, dict):
                return
            if isinstance(val, list):
                text_body = ", ".join(str(v) for v in val if isinstance(v, (str, int, float)) and not isinstance(v, bool))
            elif isinstance(val, (str, int, float)):
                text_body = str(val)
            else:
                return
            text_body = text_body.strip()
            have = [self.stem(w) for w in re.findall(r"[a-z0-9]+", spoken(key)) if w not in LINK_WORDS]
            if not text_body or len(text_body) > 120 or not have or "id" in have:
                return
            if len(have) == len(want) and all(same(a, b) for a, b in zip(have, want)):
                score = 3
            elif all(any(same(w, h) for h in have) for w in want):
                score = 2
            elif all(any(same(h, w) for w in want) for h in have) and not set(have) <= {"name", "title", "type", "value", "kind"}:
                score = 1
            else:
                return
            found.append((score, -len(found), text_body))

        def visit(d: dict[str, Any]) -> None:
            label = next((d[k] for k in d if re.fullmatch(r"name|label|attribute|property|key|field", str(k).lower()) and isinstance(d[k], str)), None)
            if label and "value" in d:
                offer(label, d["value"])
            for key, val in d.items():
                offer(str(key), val)
        walk_dicts(rec, visit)
        return max(found)[2] if found else ""

    def pointer_records(self, ptr: dict[str, Any]) -> list[tuple[tuple, str, dict[str, Any], str]]:
        """(score, id, its records merged (newest fields win), the property's value there) of each listing seen that may be
        the one a pointer names, best first: a seller match, then facts met, then kind words; listings of another
        seller left out."""
        newest: dict[str, dict[str, Any]] = {}

        def visit(d: dict[str, Any]) -> None:
            ident = self.row_mark(d)
            if ident in self.archive.ids and len(d) >= 3:
                newest.setdefault(ident, {}).update(d)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        kind_words = {self.stem(w) for w in re.findall(r"[a-z]{3,}", ptr["kind"].lower()) if w not in LINK_WORDS}
        ranked = []
        for ident, rec in newest.items():
            seller = self.from_seller(rec, ptr["seller"])
            if seller is False:
                continue
            body = as_text(rec, 20000).lower()
            facts = sum(1 for f in ptr["facts"] if self.fact_met(body, f))
            kinds_n = sum(1 for w in kind_words if w in body)
            ranked.append(((seller is True, facts, kinds_n, ident in self.archive.read), ident, rec, self.prop_value(rec, ptr["property"])))
        ranked.sort(key=lambda r: r[0], reverse=True)
        return ranked

    def pointer_fits(self, ptr: dict[str, Any], score: tuple, loose: bool = False) -> bool:
        """The record is the one the pointer names: the seller matches (when one was named) and the facts are met (all
        of up to two, all but one of more); loose, one fact (or none when none was given) is enough."""
        facts = len(ptr["facts"])
        need = (min(1, facts) if loose else facts if facts <= 2 else facts - 1)
        seller_ok = score[0] or not ptr["seller"]
        return seller_ok and score[1] >= need and (bool(ptr["seller"]) or score[1] >= 1)

    def pointer_settle(self, ptr: dict[str, Any], loose: bool = False) -> str:
        """Resolves the pointer from the records seen when one of them is the listing it names and gives the property's
        value (a details record that names it in no field is read once by the cheap model); the note, or ''."""
        for score, ident, rec, value in self.pointer_records(ptr):
            if not self.pointer_fits(ptr, score, loose):
                continue
            if not value and ident in self.archive.read and not ptr.get("asked") and self.wire.spare() > CLOSING + 20:
                ptr["asked"] = True
                ask_text = (f"A shopper wants the {ptr['property']} of a product to be the one this store listing gives. From the record "
                            f"below, copy that value exactly as the record writes it. Answer with ONE JSON object, nothing else: "
                            f"{{\"value\": the value, or null when the record names no {ptr['property']}}}.\n\nRECORD:\n" + as_text(clipped(rec, 400), 6000))
                answer = self.wire.consult(ask_text, 300, LISTENER, roof=SPEND_STOP)
                got = (answer or {}).get("value")
                value = str(got).strip()[:120] if isinstance(got, (str, int, float)) and not isinstance(got, bool) else ""
            if value:
                return self.pointer_resolve(ptr, ident, value, loose)
        return ""

    def pointer_resolve(self, ptr: dict[str, Any], ident: str, value: str, loose: bool) -> str:
        nd = self.needs
        line = nd.line_for(ptr["for"]) if nd.several() and ptr["for"] else None
        need = f"{ptr['property']}: {value}" + (f" (for {nd.lines[line]['product']})" if line is not None else "")
        if not nd.several():
            nd.uncovered = [r for r in nd.uncovered if not nd.same_prop(nd.prop_of(r), ptr["property"])]
            if nd.now_reqs is not None:
                nd.now_reqs = [r for r in nd.now_reqs if not nd.same_prop(nd.prop_of(r), ptr["property"])]
        nd.uncovered.append(need)
        if nd.now_reqs is not None:
            nd.now_reqs.append(need)
        nd.version += 1
        ptr.update(state="resolved", listing=ident, value=value)
        self.log.append({"kind": "pointer", "state": "resolved", "property": ptr["property"], "listing": ident, "value": value,
                         "need": need, "steps": ptr["steps"], "loose": loose})
        return (f"[reference] The {ptr['property']} the shopper pointed at is the one listing {ident} gives: {value}. It is now a "
                f"requirement ({need}); listings judged before are judged again under it.")

    def pointer_drop(self, ptr: dict[str, Any], why: str) -> str:
        """Gives a pointer up: the best near match (seller and one fact) still resolves it; else it binds nothing."""
        note = self.pointer_settle(ptr, loose=True)
        if note:
            return note
        ptr["state"] = "unresolved"
        self.log.append({"kind": "pointer", "state": "unresolved", "property": ptr["property"], "why": why, "steps": ptr["steps"],
                         "binding": False})
        return (f"[reference] The listing the shopper pointed at for the {ptr['property']} was not found, so that need cannot be "
                "checked and binds nothing: go on with the other requirements (it is no reason to decline).")

    def pointer_search(self, ptr: dict[str, Any]) -> dict[str, Any] | None:
        """Search arguments for the listing a pointer names, in the mode the errand's searches used, first page, with no
        category or price filter: a seller argument when the tool has one, else the seller's name in the query, with
        the kind's words on the first search and alone on the second (kind and a fact when no seller was named)."""
        tool = self.role("search")
        jschema = self.shapes.get(tool or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        need = set(jschema.get("required") or [])
        paging = self.page_arg()

        def spec(key: str) -> dict[str, Any]:
            return props.get(key) if isinstance(props.get(key), dict) else {}

        def texty(key: str) -> bool:
            return not spec(key).get("enum") and "string" in (kinds(spec(key)) or ["string"])
        maker = next((k for k in props if texty(k) and SELLER_KEY.search(spoken(str(k)))), None)
        skip = re.compile(r"categor|shelf|department|aisle|\bnext\b|token|cursor|continu|page|offset|sort|\border\b|after|before|price|cost|\bcap\b|max|min|limit")
        last: dict[str, Any] = {}
        for entry in reversed(self.log):
            sent = [a for a in (entry.get("sent") or []) if a.get("name") == tool and isinstance(a.get("args"), dict)] if entry.get("kind") == "env" else []
            if sent:
                last = sent[0]["args"]
                break
        texts = [k for k in props if texty(k) and k != maker and not skip.search(spoken(str(k)))]
        used = [k for k in texts if isinstance(last.get(k), str) and last[k].strip()]
        text_key = (self.lead_ctx[0] if self.lead_ctx and self.lead_ctx[0] in texts else None) \
            or (max(used, key=lambda k: len(last[k].split())) if used else None) \
            or next((k for k in texts if k in need), None) or (texts[0] if len(texts) == 1 else None)
        if not text_key and not maker:
            return None
        args: dict[str, Any] = {}
        for key, val in last.items():
            if key in (text_key, maker) or (paging and key == paging[0]):
                continue
            if spec(key).get("enum") and not re.search(r"categor|shelf|department|aisle", spoken(str(key))):
                args[key] = val          # (the mode the errand's searches ran in)
            elif key in need:
                args[key] = val
        if paging and paging[0] in need:
            args[paging[0]] = int(paging[2]) if float(paging[2]).is_integer() else paging[2]
        kind_words = " ".join(w for w in re.findall(r"[A-Za-z0-9]+", ptr["kind"]) if w.lower() not in LINK_WORDS)
        seller, first = ptr["seller"], ptr["searches"] == 0
        if maker and seller:
            args[maker] = seller
            words = kind_words if first else ""
        elif seller:
            words = (seller + " " + kind_words).strip() if first else seller
        else:
            fact = ptr["facts"][0].split(":", 1)[-1].strip() if ptr["facts"] else ""
            words = (kind_words + " " + fact).strip() if first else kind_words
        if self.lead_words and self.lead_ctx and self.lead_ctx[0] == text_key:
            words = " ".join(words.split()[: self.lead_words])
        if text_key and (words or text_key in need):
            args[text_key] = words or seller or kind_words
        return args or None

    def reference_step(self) -> bool:
        """Looks up the listing an open pointer need names (scripted, one step per turn): the records already seen
        first (no step), then the seller's rows already seen, then up to POINTER_SEARCHES searches and POINTER_READS
        details reads of the rows best matching the seller and the facts, POINTER_STEPS steps in all, while the steps
        the order and one repair need stay free. The value that listing gives for the property becomes a requirement
        'property: value'; a pointer left unresolved is logged and binds nothing."""
        ptr = next((p for p in self.needs.pointers if p["state"] == "open"), None)
        if ptr is None or self.wire.done or self.wire.broke:
            return False
        note = self.pointer_settle(ptr)
        if not note:
            room = self.moves_left() - max(FIRST_ORDER_BY, self.order_due() - 2) - 1
            if ptr["steps"] >= POINTER_STEPS or room <= 0 or self.wire.spare() < CLOSING + 30 or not self.role("search"):
                note = self.pointer_drop(ptr, "no steps or time left for the look-up")
        if note:
            self.chat.append({"role": "user", "content": note + "\n" + self.status_line()})
            return True
        if ptr["seller"] and not ptr["queue"] and not ptr["reads"]:
            ptr["queue"] = [i for score, i, _, _ in self.pointer_records(ptr) if score[0] and i not in self.archive.read][:POINTER_READS]
        queue = [i for i in ptr["queue"] if i not in self.archive.read]
        if queue and ptr["reads"] < POINTER_READS and self.role("details") and self.details_call(queue[0]):
            ident = queue.pop(0)
            ptr["queue"], ptr["reads"], ptr["steps"] = queue, ptr["reads"] + 1, ptr["steps"] + 1
            responses = self.pointer_call(self.role("details"), self.details_call(ident),
                                          f"The shopper's {ptr['property']} is the one another listing gives; I open {ident} to read it.")
        elif ptr["searches"] < POINTER_SEARCHES:
            call_args = self.pointer_search(ptr)
            ptr["searches"] += 1
            sig = json.dumps(call_args, sort_keys=True, default=str)
            if call_args is None or sig in ptr["seen"]:
                return self.reference_step()
            ptr["seen"].append(sig)
            ptr["steps"] += 1
            start = len(self.log)
            responses = self.pointer_call(self.role("search"), call_args,
                                          f"The shopper's {ptr['property']} is the one another listing gives; I search for that listing.")
            found = {self.row_mark(r) for entry in self.log[start:] if entry.get("kind") == "env" and isinstance(entry.get("reply"), dict)
                     for r in self.result_rows(entry["reply"].get("observation")) if isinstance(r, dict)}
            # (rows of another seller are already left out; a seller match ranks first, rows showing no seller follow)
            ptr["queue"] = [i for _, i, _, _ in self.pointer_records(ptr) if i in found and i not in self.archive.read][:POINTER_READS + 1]
        else:
            note = self.pointer_drop(ptr, "no listing of the seller met the facts")
            self.chat.append({"role": "user", "content": note + "\n" + self.status_line()})
            return True
        note = self.pointer_settle(ptr)
        if not note and (ptr["steps"] >= POINTER_STEPS or (ptr["searches"] >= POINTER_SEARCHES and not ptr["queue"])
                         or (ptr["reads"] >= POINTER_READS)):
            note = self.pointer_drop(ptr, "the look-up's steps are spent")
        if note and responses:
            cid, text_body = responses[-1]
            responses[-1] = (cid, text_body + "\n" + note)
        self.post_replies(responses)
        return True

    @staticmethod
    def same_kind(prop: str, other: str) -> bool:
        """Two property names for the same thing: the same words, or one's words inside the other's."""
        a, b = word_stems(prop), word_stems(other)
        return Needs.same_prop(prop, other) or bool(a and b and (a <= b or b <= a))

    def binding(self, prop: str) -> bool:
        """A property some requirement in force is about."""
        return any(self.same_kind(prop, Needs.prop_of(req)) for req in self.needs.props_left() if Needs.prop_of(req))

    def note_handovers(self, obs: Any) -> str:
        """Hand-overs on listing pages: a page whose own text says one of its properties is to be taken from another
        listing ("For storage, go by the Business Laptops listing from Ramiqiri (8 gb memory), not this item"). Each is
        kept with its listing in the shape of the shopper's pointers (property, kind, seller, facts), so the same
        look-up finds the other listing; the listing's property is then that listing's value. Returns a note on those
        about a property the requirements name ('' when none is new)."""
        told: list[str] = []
        sellers: set[str] = set()
        shelves = sorted({lab.lower() for lab in self.archive.shelves().values() if lab}, key=len, reverse=True)

        def known(d: dict[str, Any]) -> None:
            sellers.update(self.squash(w) for w in re.findall(r"[A-Za-z]{4,}", self.seller_text(d)))

        def visit(d: dict[str, Any]) -> None:
            ident = self.row_mark(d)
            if ident not in self.archive.ids:
                return
            own = self.squash(self.seller_text(d))
            for key, val in d.items():
                # (any text of the record but its name: race 15 wrote hand-overs in "shop_comments", another store may not)
                if not isinstance(val, str) or len(val) < 30 or re.search(r"\b(?:title|name|headline)\b", spoken(key)):
                    continue
                for sentence in re.split(r"(?<=[.;])\s+", val):
                    hit = HAND_OVER_RX.search(sentence)
                    if not hit or not ELSEWHERE_RX.search(sentence):
                        continue
                    prop = next(g for g in hit.groups() if g).strip().lower()
                    kept = self.handovers.setdefault(ident, [])
                    if any(self.same_kind(prop, h["property"]) for h in kept):
                        continue
                    low = sentence.lower()
                    kind = next((s for s in shelves if s in low), "")
                    names = [w for w in re.findall(r"\b([A-Z][a-z]{3,})(?:'s)?\b", sentence)
                             if self.squash(w) not in own and w.lower() not in kind and w.lower() not in ("from", "this", "take", "that", "with")]
                    seller = next((w for w in names if self.squash(w) in sellers), "") \
                        or next((m for m in re.findall(r"\b(?:from|by)\s+([A-Z][a-z]{3,})", sentence) if m in names), "")
                    inside = re.search(r"\(([^)]{3,120})\)", sentence)
                    facts = [f.strip() for f in re.split(r",|\band\b", inside.group(1)) if f.strip()][:4] if inside else []
                    ptr = {"property": prop, "kind": kind, "seller": seller, "facts": facts, "text": sentence.strip()[:240],
                           "state": "open", "searches": 0, "listing": "", "value": ""}
                    kept.append(ptr)
                    self.log.append({"kind": "handover", "listing": ident, "pointer": {k: ptr[k] for k in ("property", "kind", "seller", "facts")}})
                    if self.binding(prop):
                        told.append(f"[hand-over] {ident}'s page says its {prop} comes from another listing (\"{sentence.strip()[:180]}\"): "
                                    f"{ident}'s {prop} is that listing's, not its own field's; the code looks that listing up before {ident} is bought.")

        for rec in self.archive.items:
            walk_dicts(rec, known)
        walk_dicts(obs, visit)
        return " ".join(told)

    def settle_handover(self, owner: str, ptr: dict[str, Any]) -> bool:
        """The other listing of a hand-over among the records read: the best match by seller and facts that was read and
        names the property."""
        for score, ident, rec, value in self.pointer_records(ptr):
            if ident == owner or not self.pointer_fits(ptr, score, loose=True):
                continue
            if ident in self.archive.read and value:
                ptr.update(state="read", listing=ident, value=value)
                self.log.append({"kind": "handover", "listing": owner, "state": "read", "from": ident, "value": value})
                return True
            return False
        return False

    def handover_step(self) -> bool:
        """Follows the hand-over of a listing the errand may buy (in the cart, or the check passed it) on a property the
        requirements name: the other listing is read (searched for first when no row of it was seen), HANDOVER_STEPS
        steps a task, while the order's steps stay free. Once its value is read the listing is checked again with it."""
        if self.wire.done or self.wire.broke or not self.role("details") or not self.handovers:
            return False
        if self.__dict__.get("handover_steps", 0) >= HANDOVER_STEPS or self.moves_left() <= max(FIRST_ORDER_BY, self.order_due()) + 1 \
                or self.wire.spare() < CLOSING + 30:
            return False
        owners = list(dict.fromkeys(list(self.trolley.units) + [k.rpartition("|")[0] for k in self.passed]))
        for owner in owners:
            for ptr in self.handovers.get(owner, []):
                if ptr["state"] != "open" or not self.binding(ptr["property"]):
                    continue
                if self.settle_handover(owner, ptr):
                    # (judged again with the other listing's value: the item check reads it in the hand-over note)
                    self.passed = {k for k in self.passed if k.rpartition("|")[0] != owner}
                    self.rulings = {k: v for k, v in self.rulings.items() if k.rpartition("|")[0] != owner}
                    self.checked_adds = {k for k in self.checked_adds if k.rpartition("|")[0] != owner}
                    self.chat.append({"role": "user", "content": f"[hand-over] {owner}'s {ptr['property']} is the one {ptr['listing']} gives: "
                                      f"{ptr['value']}. Judge {owner} on that value (it is checked again before any order).\n" + self.status_line()})
                    return True
                rows = [i for score, i, _, _ in self.pointer_records(ptr) if i != owner and i not in self.archive.read and self.pointer_fits(ptr, score, loose=True)]
                why = f"{owner}'s {ptr['property']} is the one another listing gives; I "
                if rows and self.details_call(rows[0]):
                    responses = self.pointer_call(self.role("details"), self.details_call(rows[0]), why + f"open {rows[0]} to read it.")
                elif ptr["searches"] < 1 and self.role("search") and self.pointer_search(ptr):
                    search_args = self.pointer_search(ptr)
                    ptr["searches"] += 1
                    responses = self.pointer_call(self.role("search"), search_args, why + "search for that listing.")
                else:
                    ptr["state"] = "lost"
                    self.log.append({"kind": "handover", "listing": owner, "state": "lost", "property": ptr["property"]})
                    continue
                self.handover_steps = self.__dict__.get("handover_steps", 0) + 1
                self.post_replies(responses)
                return True
        return False

    def handover_lines(self, thing: str) -> str:
        """For the item check: the listing's hand-overs on properties the requirements name, with the other listing's
        value once it was read."""
        lines = []
        for ptr in self.handovers.get(thing, []):
            if not self.binding(ptr["property"]):
                continue
            got = (f"that listing is {ptr['listing']}, whose record gives {ptr['property']}: {ptr['value']}" if ptr["state"] == "read"
                   else "that listing's record was not found, so this listing's own field decides")
            lines.append(f"- \"{ptr['text']}\": its {ptr['property']} is the {ptr['property']} of that other listing; {got}.")
        return ("\n\nHAND-OVERS ON THIS LISTING'S OWN PAGE (a property its page says is taken from another listing is judged by that listing's value):\n"
                + "\n".join(lines)) if lines else ""

    def pointer_call(self, tool: str, call_args: dict[str, Any], why: str) -> list[tuple[str, str]]:
        """One scripted call of the pointer look-up; the replies are posted by the caller."""
        proposal = {"id": f"ref_{self.wire.tick}_{len(self.log)}", "type": "function", "function": {"name": tool, "arguments": json.dumps(call_args)}}
        self.chat.append({"role": "assistant", "content": why, "tool_calls": [proposal]})
        self.covering = True
        try:
            return self.dispatch([proposal], scripted=True)
        finally:
            self.covering = False

    def free_text(self, call_args: dict[str, Any]) -> list[str]:
        """The free-text argument values of a search call (enum-valued arguments left out)."""
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        props = props if isinstance(props, dict) else {}
        return [val for key, val in call_args.items() if isinstance(val, str) and not (props.get(key) or {}).get("enum")]

    def book_page(self, call_args: dict[str, Any], rows: list, total: int) -> None:
        """Books a search page in its walk (the same call with only the paging argument changed)."""
        found = self.page_arg()
        if not found or not rows:
            return
        key, _, low, _ = found
        at = call_args.get(key, low)
        if not is_number(at):
            return
        base = {k: v for k, v in call_args.items() if k != key}
        walk = self.__dict__.setdefault("page_book", {}).setdefault(
            json.dumps(base, sort_keys=True, default=str), {"args": base, "read": set(), "size": 0, "total": 0, "shelves": set()})
        walk["read"].add(at)
        walk["size"] = max(walk["size"], len(rows))
        walk["total"] = max(walk["total"], total)
        walk["shelves"] |= {self.shelf_of(r) for r in rows if isinstance(r, dict)} - {""}
        walk.setdefault("marks", set()).update(self.row_mark(r) for r in rows if isinstance(r, dict))
        prices = [self.row_price(r) for r in rows if isinstance(r, dict)]
        walk["dearest"] = max([walk.get("dearest", 0.0)] + [p for p in prices if p is not None])
        if at == low and total <= len(rows):
            walk["every"] = [r for r in rows if isinstance(r, dict)]     # (a first page showing every match)
        marks = [self.row_mark(r) for r in rows if isinstance(r, dict)]
        if len(marks) >= 5:
            # (5+ rows in ascending id order: a fixed listing order, not a ranking)
            walk["id_order"] = walk.get("id_order", True) is not False and (marks == sorted(marks) or marks == sorted(
                marks, key=lambda m: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", m)]))

    def next_page(self, walk: dict[str, Any]) -> Any:
        """The first unread paging value of a walk that can still show unseen rows, else None."""
        key, by_page, low, high = self.page_arg()
        low = min([low] + [v for v in walk["read"] if is_number(v)])
        size = max(1, walk["size"])
        step = 1 if by_page else size
        for i in range(1, 400):
            val = low + i * step
            if (high is not None and val > high) or (i * size >= walk["total"]):
                return None
            if val not in walk["read"]:
                return val
        return None

    def value_score(self, call_args: dict[str, Any]) -> int:
        """How many property requirements in force have a value word among a search's leading query words."""
        def near(a: str, b: str) -> bool:
            return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))
        vals = [re.findall(r"[a-z0-9]{3,}", re.sub(r"\(for [^)]*\)", "", req.partition(":")[2].lower())) for req in self.needs.props_left()]
        words = [w for text in self.free_text(call_args) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
        words = words[: self.lead_words] if self.lead_words else words
        return sum(1 for value_words in vals if any(near(w, v) for w in words for v in value_words))

    @staticmethod
    def row_price(row: dict[str, Any]) -> float | None:
        """The price a search row shows (its first number under a price- or cost-named key), else None."""
        return next((float(val) for k2, val in row.items() if is_number(val) and not isinstance(val, bool)
                     and re.search(r"price|cost", str(k2).lower())), None)

    def row_hits(self, rows: list) -> list[str]:
        """Search rows not read yet that are in stock, within the budget and (when it is known) on the requested category."""
        cap, qty = self.needs.spend_cap, self.needs.quantity()
        shelves = self.shelf_labels()
        hits = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            ident = self.row_mark(r)
            cost = next((val for k2, val in r.items() if is_number(val) and re.search(r"price|cost", str(k2).lower())), None)
            shelf = self.shelf_of(r)
            if ident in self.archive.ids and ident not in self.archive.read and not (is_number(cap) and is_number(cost) and cost * qty > cap) \
                    and not self.sold_out(r) and not (shelves and shelf and shelf not in shelves):
                hits.append(ident)
                self.__dict__.setdefault("hit_shelf", {})[ident] = shelf
        return hits

    def limit_of(self, call_args: dict[str, Any]) -> float | None:
        """The numeric limit a search call set (the same test search_note uses: a number above 1 in an argument that is
        not a paging, size, count or lower-bound one), the largest when several; None when it set none."""
        paging = self.search_keys(call_args)[3]
        limits = [val for key, val in call_args.items() if is_number(val) and not isinstance(val, bool) and val > 1 and key not in paging
                  and not re.search(r"page|size|count|result|min|low|from|start", str(key).lower())]
        return max(limits) if limits else None

    def capped_walk(self, walk: dict[str, Any], shelves: set[str]) -> bool:
        """A walk that is a capped browse worth paging: its limit is at or under the budget for one unit, no row it showed
        costs more, and its rows sit on the requested category (while no read listing shows the category: its rows'
        category labels, else its query words, share a word with a requested product)."""
        cap, limit = self.needs.spend_cap, self.limit_of(walk["args"])
        if not is_number(cap) or limit is None or limit > cap / max(1, self.needs.quantity()) + 1e-9:
            return False
        if walk.get("dearest", 0.0) > limit + 1e-9:
            return False        # (a row above the limit: the store ignored it, or the number is not a price limit)
        if shelves:
            return bool(walk["shelves"] & shelves)
        product = {self.stem(w) for ln in self.needs.lines for w in re.findall(r"[a-z0-9]{3,}", str(ln.get("product") or "").lower())}
        said = {self.stem(w) for text in (walk["shelves"] or self.free_text(walk["args"])) for w in re.findall(r"[a-z0-9]{3,}", text.lower())}
        return bool(product & said)

    def open_pages(self, most: int = 2) -> list[dict[str, Any]]:
        """Ready arguments for the next unread page of each search whose total is larger than the pages read and that
        either leads with a required value or is a capped browse (capped_walk): the ones carrying more required values
        first, then the narrower; then the capped browses, fewest pages left first."""
        found = self.page_arg()
        if not found:
            return []

        shelves = self.shelf_labels()
        picks = []
        for walk in self.__dict__.get("page_book", {}).values():
            nxt = self.next_page(walk)
            if nxt is None or (shelves and walk["shelves"] and not walk["shelves"] & shelves):
                continue
            score = self.value_score(walk["args"])
            if score:
                picks.append((0, -score, walk["total"], {**walk["args"], found[0]: nxt}))
            elif self.capped_walk(walk, shelves):
                size = max(1, walk["size"])
                left = max(0, walk["total"] - size * len(walk["read"]))
                picks.append((1, 0, -(-left // size), {**walk["args"], found[0]: nxt}))
        picks.sort(key=lambda p: p[:3])
        return [p[3] for p in picks[:most]]

    def value_probe(self, raw: list[str]) -> tuple[str, str]:
        """(property, value-first query) when a keyword query's leading words hold a word of a required property's NAME
        and no word of any required value or of the product: a store's keyword index holds listing values, titles and
        categories, so such a query keeps only the listings that repeat the name. The query offered is the value from
        its first figure (comparison words dropped), else the whole value, then the product as the request words it."""
        if self.word_kind == "any":
            return "", ""
        lead = [w for w in re.findall(r"[a-z0-9]+", " ".join(raw).lower()) if len(w) >= 2][: self.lead_words or 2]
        named = self.needs.lines[0]["product"] if self.needs.lines and not self.needs.several() else ("" if self.needs.several() else self.needs.reading.get("product"))
        product = " ".join(re.findall(r"[a-z0-9]+", str(named or "").lower())[:3])

        def near(a: str, b: str) -> bool:
            if any(c.isdigit() for c in a + b):
                return a.lstrip("0") == b.lstrip("0") or (a.isdigit() and b.startswith(a) and not b[len(a):len(a) + 1].isdigit())
            return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))
        reqs = [(req.partition(":")[0].strip(), re.split(r"[,;]", re.sub(r"\((?:stated|revealed[^)]*|for [^)]*)\)", "", req.partition(":")[2]))[0].strip().lower())
                for req in self.needs.props_left() if ":" in req]
        keep = [w for _, value in reqs for w in re.findall(r"[a-z0-9]{2,}", value)] + product.split()
        if not lead or any(near(w, k) for w in lead for k in keep):
            return "", ""
        for name, value in reqs:
            if value and any(near(w, n) for w in lead for n in re.findall(r"[a-z0-9]{2,}", name.lower())):
                digit = re.search(r"\d", value)
                probe = f"{(value[digit.start():] if digit else value).strip(' .;')} {product}".strip()
                head = [w for w in re.findall(r"[a-z0-9]+", probe.lower()) if len(w) >= 2][: self.lead_words or 2]
                for walk in self.__dict__.get("page_book", {}).values():
                    ran = [w for t in self.free_text(walk["args"]) for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) >= 2][: self.lead_words or 2]
                    if len(ran) == len(head) and all(near(a, b) for a, b in zip(head, ran)):
                        return "", ""
                return name, probe
        return "", ""

    def query_shape(self, call_args: dict[str, Any]) -> dict[str, Any] | None:
        """A search call's filters (every argument but the query's and the paging ones), the query's argument (the one
        whose read depth was learned, else the free-text one with most words that is not a category label), its stemmed
        words in order with repeats dropped, its mode (the enum arguments) and whether it asks for the first page.
        None without a free-text argument."""
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        props = props if isinstance(props, dict) else {}
        texts = [k for k, v in call_args.items() if isinstance(v, str) and not (props.get(k) or {}).get("enum")]
        if not texts:
            return None
        shelves = self.shelf_labels()
        plain = [k for k in texts if call_args[k].strip().lower() not in shelves] or texts
        _, mode, first, paging = self.search_keys(call_args)
        key = self.lead_ctx[0] if self.lead_ctx and self.lead_ctx[0] in texts else max(plain, key=lambda k: (len(call_args[k].split()), k))
        words = tuple(dict.fromkeys(self.stem(w) for w in re.findall(r"[a-z0-9]+", call_args[key].lower())))
        filters = tuple(sorted((k, json.dumps(v, sort_keys=True, default=str)) for k, v in call_args.items() if k != key and k not in paging))
        return {"filters": filters, "key": key, "words": words, "mode": mode, "first": first}

    def read_words(self, shape: dict[str, Any], words: tuple) -> frozenset:
        """The words of a query the store reads: its first lead_words when that depth was learned for this argument
        and mode, else all of them; as a set (the order of read words does not change the match)."""
        if self.lead_words and self.lead_ctx and self.lead_ctx == (shape["key"], shape["mode"]):
            return frozenset(words[: self.lead_words])
        return frozenset(words)

    def book_query(self, call_args: dict[str, Any], total: int) -> None:
        """Books a first-page search (rows or none) for repeat_hold, with the requirements and cart versions it ran under."""
        shape = self.query_shape(call_args)
        if shape is None or not shape["first"] or not shape["words"]:
            return
        self.__dict__.setdefault("first_pages", []).append({**shape, "total": total, "args": dict(call_args),
                                                            "at": (self.needs.version, self.trolley.version)})

    def repeat_hold(self, call_args: dict[str, Any]) -> str:
        """A hold (no step) for a first-page search that repeats earlier ones under the same requirements and cart, once
        per kind of repeat; sent again, it goes. (1) It reads the same words under the same filters as a first page
        already read: it would show the same rows. (2) It is the fourth first page whose query starts with the same two
        words under the same filters."""
        shape = self.query_shape(call_args)
        if shape is None or not shape["first"] or not shape["words"]:
            return ""
        filters, key, words = shape["filters"], shape["key"], shape["words"]
        now = (self.needs.version, self.trolley.version)
        sent = [rec for rec in self.__dict__.get("first_pages", []) if rec["filters"] == filters and rec["key"] == key and rec["at"] == now]
        held = self.__dict__.setdefault("repeat_held", set())
        found = self.page_arg()
        tool = self.role("search")

        def next_of(rec: dict[str, Any]) -> dict[str, Any] | None:
            if not found:
                return None
            base = {k: v for k, v in rec["args"].items() if k != found[0]}
            walk = self.__dict__.get("page_book", {}).get(json.dumps(base, sort_keys=True, default=str))
            nxt = self.next_page(walk) if walk else None
            return {**base, found[0]: nxt} if nxt is not None else None
        read = self.read_words(shape, words)
        twin = next((rec for rec in sent if self.read_words(rec, rec["words"]) == read), None)
        if twin is not None and ("same", filters, key, read) not in held:
            held.add(("same", filters, key, read))
            more = next_of(twin)
            depth = f"only the first {self.lead_words} word(s) of a query" if len(read) < len(words) else "these words, in any order"
            return ("Not run (no step used): this search reads the same words under the same filters as your earlier search "
                    f"{json.dumps(twin['args'], ensure_ascii=False)} (the store reads {depth}), so it would show the same rows. "
                    + (f"Read its next page instead: {tool} {json.dumps(more, ensure_ascii=False)}. " if more else
                       "That search showed every match it has. ")
                    + "Or search another required value's own word, or the item kind alone. If you are sure, send the same call again.")
        pair = words[:2]
        alike = [rec for rec in sent if len(pair) == 2 and rec["words"][:2] == pair]
        mark = json.dumps(call_args, sort_keys=True, default=str)
        let_through = self.__dict__.setdefault("repeat_pass", set())
        pair_holds = self.__dict__.setdefault("pair_holds", {})
        if len(alike) < 3 or mark in let_through or pair_holds.get((filters, key, pair), 0) >= 3:
            return ""      # (a call sent again after its hold goes through; at most three holds per pair of words)
        pair_holds[(filters, key, pair)] = pair_holds.get((filters, key, pair), 0) + 1
        let_through.add(mark)
        ideas = []
        largest = max(alike, key=lambda rec: rec["total"])
        more = next_of(largest)
        if more:
            ideas.append(f"read the next page of the largest of them: {tool} {json.dumps(more, ensure_ascii=False)}")
        led = {rec["words"][0] for rec in sent if rec["words"]}
        _, probe = self.value_probe([str(v) for v in self.free_text(call_args)])
        if not probe:
            line = self.needs.line_for(" ".join(words))
            named = self.needs.lines[line]["product"] if line is not None and line < len(self.needs.lines) else ""
            product = " ".join(re.findall(r"[a-z0-9]+", str(named).lower())[:2])
            values = [w for req in self.needs.props_left() if ":" in req
                      for w in re.findall(r"[a-z0-9]{3,}", re.sub(r"\((?:stated|revealed[^)]*|for [^)]*)\)", "", req.partition(":")[2].lower()))]
            fresh = next((w for w in values if self.stem(w) not in pair and self.stem(w) not in led), "")
            probe = f"{fresh} {product}".strip() if fresh else ""
        if probe:
            ideas.append(f"a query led by a required value's own word, such as '{probe}'")
        ideas.append("the requested category with a price limit at the budget, reading its pages")
        return (f"Not run (no step used): {len(alike)} first-page searches starting with '{' '.join(str(call_args[key]).split()[:2])}' under the same filters "
                "ran already, and re-wording after those words has not shown a fit. Instead, in this order: " + "; ".join(ideas)
                + ". If you are sure, send the same call again.")

    def search_note(self, call_args: dict[str, Any], obs: Any) -> str:
        """A search result whose reported total is larger than the rows it shows (the rest stay unseen, whatever their
        price or fit), or whose rows break a numeric limit the call set (a filter the search ignored)."""
        if not isinstance(obs, dict):
            return ""

        def rows_of(piece: dict[str, Any]) -> list | None:
            lists = [val for val in piece.values() if isinstance(val, list) and all(isinstance(r, dict) for r in val)]
            return next((val for val in lists if val), lists[0] if lists else None)
        rows = rows_of(obs)
        if rows is None:
            # (a result wrapped together with a shopper message or a refusal: the rows sit one level down)
            obs = next((val for val in obs.values() if isinstance(val, dict) and rows_of(val) is not None), None)
            if obs is None:
                return ""
            rows = rows_of(obs) or []
        totals = [val for val in obs.values() if isinstance(val, int) and not isinstance(val, bool) and val > len(rows)]
        self.book_page(call_args, rows, max(totals or [len(rows)]))
        self.book_query(call_args, max(totals or [len(rows)]))
        remarks = []
        kind = self.word_rule(call_args, obs, rows)
        lead = self.lead_rule(call_args, obs, rows)
        if lead:
            texts, mode, _, _ = self.search_keys(call_args)
            words = next((w for key, w in texts if key == self.lead_ctx[0]), ()) if mode == self.lead_ctx[1] else ()
            counted = f" (this query counted only: {' '.join(words[:lead])})" if len(words) > lead else ""
            remarks.append(f"This store's keyword search ({self.lead_ctx[0]}) reads only the FIRST {lead} word(s) of a query and "
                           f"ignores the rest{counted}: put the {lead} most distinctive words first (a required value and the item "
                           "kind, the value first), and search other values in separate short queries.")
        texts, mode, first, paging_keys = self.search_keys(call_args)
        raw = [val for key, val in call_args.items() if isinstance(val, str) and key not in dict(mode)]
        if first and raw:
            # (a store that drops one-character words: a query of such pieces only finds nothing)
            pieces = [p for t in raw for p in re.findall(r"[a-z0-9]+", t.lower())]
            if pieces and all(len(p) == 1 for p in pieces) and not rows and not self.__dict__.get("short_told"):
                self.short_told = True
                remarks.append("This query found nothing: this store's search skips one-character words, so a figure like 0.4 or "
                               "5 x 7 searches nothing (or only its other words). Search a word value, or the unit word with the item "
                               "kind, and read every page.")
            prop, probe = self.value_probe(raw)
            told = self.__dict__.get("name_told", 0)
            if probe and told < 2 and self.moves_left() > self.order_due() + 3:
                self.name_told = told + 1
                remarks.append((f"This query leads with words of a property's NAME ({prop}) and none of a required value: a keyword "
                                "search matches listing values, titles and categories, and a field's name only where some value or "
                                f"title repeats it, so it misses listings whose {prop} field holds just the value." if not told
                                else f"Again led by a property's name ({prop}).")
                               + f" Search the value first, then the item kind: '{probe}' (a figure the store cannot read drops out, "
                               "leaving its unit word), and read every page of it.")
            marks = tuple(self.row_mark(r) for r in rows)
            others = tuple(sorted((k2, str(v2)) for k2, v2 in call_args.items() if k2 not in paging_keys and not (isinstance(v2, str) and k2 not in dict(mode))))
            sig = (mode, others, max(totals or [len(rows)]), marks)
            book = self.__dict__.setdefault("page_sigs", {})
            query = " ".join(raw)
            if len(set(marks)) >= 3 and sig in book and book[sig] != query:
                remarks.append(f"Same total and rows as your earlier search '{book[sig]}': the store read both alike, so re-spelling "
                               "it shows nothing new. Read its next page, or change its leading words.")
            book.setdefault(sig, query)
            # (an all-words store showing every match: each row carries the words it read, often in fields its title hides)
            hits = []
            if (kind == "all" or self.lead_words) and not totals and rows:
                hits = self.row_hits(rows)
                score = self.value_score(call_args)
                ranks = self.__dict__.setdefault("hit_score", {})
                ranks.update({i: max(ranks.get(i, 0), score) for i in hits})
                found = self.__dict__.setdefault("all_hits", [])
                found += [i for i in hits if i not in found]
                if hits and len(texts) and len(rows) <= 12:
                    remarks.append("Every word is required here and every match is shown: each row carries the words the store read "
                                   "somewhere in its record, often in fields its title does not show. Open the in-budget ones before "
                                   "searching on: " + ", ".join(hits[:6]) + ".")
            if (kind == "all" or self.lead_words) and not self.__dict__.get("hits_back"):
                # (the rule is known only now: earlier searches that showed every match and carry a required value)
                self.hits_back = True
                here = json.dumps({k: v for k, v in call_args.items() if k not in paging_keys}, sort_keys=True, default=str)
                back: list[str] = []
                for walk_key, walk in self.__dict__.get("page_book", {}).items():
                    if walk_key != here and walk.get("every") and self.value_score(walk["args"]):
                        fresh = [i for i in self.row_hits(walk["every"]) if i not in back and i not in hits]
                        ranks = self.__dict__.setdefault("hit_score", {})
                        ranks.update({i: max(ranks.get(i, 0), self.value_score(walk["args"])) for i in fresh})
                        back += fresh
                found = self.__dict__.setdefault("all_hits", [])
                found += [i for i in back if i not in found]
                if back:
                    remarks.append("This store's keyword search needs every word, so earlier searches that showed every match listed only "
                                   "rows carrying all of their words, often in fields the titles do not show. Open the in-budget ones not "
                                   "read yet before searching on: " + ", ".join(back[:6]) + ".")
        if not rows:
            return " ".join(remarks)
        if totals:
            if kind == "any":
                how = ("this store's keyword search takes ANY of the words and ranks the listings matching more of them first, so "
                       "adding words widens it: a required value's own distinctive word or figure alone lists only the listings "
                       "carrying it (often few enough to read every page and compare prices); the item kind with the values ranks "
                       "the closest matches first; use the category mode (or a price filter if one holds) for the rest.")
            elif kind == "all":
                how = ("this store's keyword search needs EVERY word, so more exact words narrow it, as do the category and price "
                       "filters with a lower price limit.")
            else:
                how = ("then narrow it for the rest: when every word is required, add exact words; when any word matches (a longer "
                       "query matching more listings shows that), put the required values in the query so fitting listings rank first; "
                       "a price filter, when one holds, also helps.")
            remarks.append(f"This search matched {max(totals)} listings but shows only {len(rows)} on this page, and only a few pages can "
                           "be read: the listings that matter may be among the unseen ones. Read the further pages of this same search "
                           "first (as many as can be read); if the total is larger than those pages, " + how)
        paging = self.search_keys(call_args)[3]
        limits = [val for key, val in call_args.items() if is_number(val) and val > 1 and key not in paging
                  and not re.search(r"page|size|count|result|min|low|from|start", str(key).lower())]
        if limits:
            cap = max(limits)
            over = [r for r in rows if any(is_number(val) and val > cap and "price" in str(key).lower() for key, val in r.items())]
            if over:
                remarks.append(f"{len(over)} of these rows cost more than the limit {cap} this call set: this search mode ignored that filter "
                               "(and may ignore others, such as the category). Use the mode whose filters hold, or check every row yourself.")
        return " ".join(remarks)

    def shopper_lines(self, call_name: str, obs: Any, refused: bool) -> list[str]:
        lns: list[str] = []
        if call_name == self.role("message"):
            if isinstance(obs, str):
                lns = [obs]
            elif isinstance(obs, dict) and "error" not in obs:
                long_strings([val for val in obs.values() if isinstance(val, (str, list))], lns)
        elif isinstance(obs, dict):
            for piece in obs.values():
                if isinstance(piece, list) and piece and all(isinstance(val, (str, dict)) for val in piece) and any(isinstance(val, str) for val in piece):
                    long_strings([val for val in piece if isinstance(val, str)], lns)
                elif isinstance(piece, str) and len(piece.split()) >= 4 and refused:
                    lns.append(piece)
        return lns

    # ----------------------------------------------------------------------------------------------- one batch
    def screen(self, proposal: dict[str, Any], permitted: set[str], scripted: bool, batch: int, queued_messages: int) -> tuple[str | None, str, dict[str, Any]]:
        """(refusal text or None, tool name, checked args) for one proposed call."""
        func = proposal.get("function") or {}
        call_name, raw_args = func.get("name"), func.get("arguments")
        if call_name not in self.shapes:
            return f"Not run (no step used): '{call_name}' is not one of the tools. Use one of: {', '.join(sorted(permitted))}.", call_name, {}
        if not scripted and call_name == self.role("message") and self.msg_reserved(queued_messages):
            # (the request's procedure owes the shopper a basket message before the order: the actor never spends it)
            return ("Not run (no step used): the last message left is kept for the basket summary, which the code sends right "
                    "before the order. Do not message the shopper now; go on with the errand."), call_name, {}
        if not scripted and call_name not in permitted:
            if call_name == self.role("message") and self.asking:
                return "Not run (no step used): questions to the shopper are planned and sent for you after you have opened a few listings' details.", call_name, {}
            return "Not run (no step used): too few steps are left for this; only these tools can be used now: " + ", ".join(sorted(permitted)) + ".", call_name, {}
        call_args = raw_args if isinstance(raw_args, dict) else (json_object(raw_args) if (raw_args or "").strip() else {})
        if call_args is None:
            return "Not run (no step used): the arguments were not a valid JSON object.", call_name, {}
        call_args, faults = ArgCheck.execute(call_args, self.shapes[call_name])
        if faults:
            return "Not run (no step used): invalid arguments: " + "; ".join(faults[:6]), call_name, call_args
        call_args, unseen = self.mend_ids(call_name, call_args)
        if unseen and not scripted:
            sig = call_name + as_text(call_args)
            if sig not in self.held_ids:
                self.held_ids.add(sig)
                return (f"Not run (no step used): {', '.join(unseen[:3])} never appeared in any tool result, so it is not an id this store "
                        "showed. Copy listing ids exactly as the results wrote them; if you are sure, send the same call again."), call_name, call_args
        if call_name in self.finishers() and batch > 1:
            return "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls.", call_name, call_args
        if call_name == self.role("message") and self.msg_cap is not None and self.msgs_sent + queued_messages >= self.msg_cap:
            return "Not run (no step used): the message budget is used up. Do not message the shopper again.", call_name, call_args
        sig = call_name + as_text(call_args)
        if call_name not in self.finishers() and call_name not in (self.role("message"), self.role("cart")) and sig in self.last_calls:
            return "Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else.", call_name, call_args
        if call_name == self.role("search") and not scripted:
            # (a first page that repeats earlier ones: held once, with the next page or another way to search)
            repeat = self.repeat_hold(call_args)
            if repeat:
                self.log.append({"kind": "repeat_hold", "args": call_args})
                return repeat, call_name, call_args
        if call_name == self.role("cart") and sig == self.failed_cart and not scripted:
            # (the store just turned this exact cart call down; sent again it gets the same answer and costs a step)
            thing = self.trolley.aim(call_args)
            why = self.dead.get(thing or "", "")
            return (f"Not run (no step used): the store just turned down this exact cart call{' (' + why + ')' if why else ''} and "
                    "would answer the same. Put in another verified listing for that product (read its stock first), change the "
                    "call, or finish."), call_name, call_args
        if call_name == self.role("cart") and self.trolley.modes and self.trolley.count_key and self.trolley.is_take(call_args):
            count = self.trolley.count_arg(call_args)
            thing = self.trolley.aim(call_args)
            if count is not None and count <= 0 and thing and self.trolley.held(thing) > 0:
                # (a take of zero units takes nothing out in stores that read the count literally: take every unit held)
                call_args = {**call_args, self.trolley.count_key: int(self.trolley.held(thing))}
                sig = call_name + as_text(call_args)
        held = "" if scripted else self.late_hold(call_name, call_args)
        if held:
            return held, call_name, call_args
        if call_name == self.role("cart") and not self.fixing:
            refusal = self.screen_cart(call_args, sig)
            if refusal:
                return refusal, call_name, call_args
        return None, call_name, call_args

    def mend_ids(self, call_name: str, call_args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Listing ids in a call to a listing tool (details, view, compare, hold, cart, a finisher taking an id): an id
        no result ever showed is put back to the one seen id a typo away; ids with no such neighbour are returned."""
        listing_tools = {self.role(r) for r in ("details", "view", "compare", "hold", "cart", "order", "decline")} - {None}
        if call_name not in listing_tools or not self.archive.tokens:
            return call_args, []
        props = self.tools.props(call_name)
        keys = {k for k, p in props.items() if id_param(k, p)}
        if call_name in {self.role(r) for r in ("details", "view", "hold", "cart")}:
            keys |= {self.tools.id_key(call_name)} - {None}
        arrays = {k for k, _, _ in list_params((self.tools.defs.get(call_name) or {}).get("parameters"))}
        unseen: list[str] = []

        def fix(val: Any) -> Any:
            if not isinstance(val, str) or not val.strip() or " " in val.strip() or self.archive.known(val):
                return val
            near = self.archive.nearest(val)
            if near:
                self.log.append({"kind": "id_mended", "sent": val, "used": near})
                return near
            unseen.append(val)
            return val
        mended = {}
        for key, val in call_args.items():
            if key in arrays and isinstance(val, list):
                fixed: list[Any] = []
                for v in val:
                    v = fix(v)
                    if v not in fixed:
                        fixed.append(v)
                mended[key] = fixed
            elif key in keys:
                mended[key] = fix(val)
            else:
                mended[key] = val
        return mended, unseen

    def screen_cart(self, call_args: dict[str, Any], sig: str) -> str:
        thing = self.trolley.aim(call_args)
        mark = f"{sig}#{self.__dict__.get('puts', {}).get(thing, 0)}"
        mine = self.archive.price(thing) if thing else None
        cheaper_put = any(is_number(mine) and is_number(self.archive.price(p)) and self.archive.price(p) < mine
                          for p in self.__dict__.get("batch_puts", []) if p != thing)
        swap = self.needs.priority() == "cheapest" and (cheaper_put or bool(self.cheaper_fit(thing)))
        moved = thing in self.__dict__.get("pass_price", {}) and self.__dict__["pass_price"][thing] != mine
        if thing and not self.trolley.is_put(call_args) and f"{thing}|{self.needs.version}" in self.passed and not swap and not moved \
                and self.refusal_n == self.refusals_seen.get(thing, self.refusal_n) and mark not in self.warned:
            # (a line the check passed, taken out with no change or refusal since: held once per time it was put in)
            self.warned.add(mark)
            least = " It is also the cheapest verified fit." if self.needs.priority() == "cheapest" and not self.cheaper_fit(thing) else ""
            return (f"Not run (no step used): an independent check found {thing} meets every requirement now in force, reading each "
                    "property by meaning; the category label alone decides the item kind, whatever object its title or description "
                    f"names.{least} Nothing has changed since. If a change or a refusal rules it out, one of its fields clearly fails "
                    "a requirement, or you are replacing it with a cheaper listing, send the same call again.")
        if not self.trolley.is_put(call_args) or not thing:
            return ""
        if self.trolley.held(thing) > 0 and self.trolley.adds is not False and sig not in self.warned:
            self.warned.add(sig)
            return (f"Not run (no step used): the cart already holds {thing} x{int(self.trolley.held(thing))}, and this cart tool adds units to a held line, "
                    f"so this call would raise it. To order it, leave the cart as it is; to change the units, take the difference out or in. If you really mean to add, send the same call again.")
        if self.moves_left() > 3 and f"{thing}|{self.needs.version}" not in self.checked_adds:
            self.checked_adds.add(f"{thing}|{self.needs.version}")
            ruling = self.check_item(thing, "ADD TO THE CART")
            if ruling:
                return "Not run (no step used). " + ruling + " If you are certain it fits, send the same call again."
            better = self.rank_check(thing) if not self.needs.several() else ""
            if better:
                return "Not run (no step used). " + better.replace("before ordering", "before putting it in the cart").replace("order the current pick", "send the same call again")
        idx = self.line_of.get(thing, 0 if not self.needs.several() else None)
        if self.needs.several() and idx is not None:
            rivals = [t for t in (self.trolley.selection(call_args) or self.trolley.units) if t != thing and self.line_of.get(t) == idx]
            if rivals and f"rival|{sig}" not in self.warned:
                self.warned.add(f"rival|{sig}")
                return (f"Not run (no step used): the cart already holds {', '.join(rivals)} for {self.needs.lines[idx]['product']}, and the "
                        "order needs one listing per product. If this one replaces it, take every unit of the old one out first; if you "
                        "really mean to add it, send the same call again.")
        hold = self.role("hold")
        if hold and thing not in self.duties.held and thing not in self.hold_tried and self.moves_left() > 4 \
                and any(d.kind == "hold" and d.state == "open" for d in self.duties.items):
            # (the request asks for a hold before the purchase: the code holds the listing first, one step)
            self.hold_queue.append(thing)
            self.hold_tried.add(thing)
            return (f"Not run yet (no step used): the request asks to hold the chosen listing before buying it, so the code holds "
                    f"{thing} for you now; then send this cart call again.")
        need_n = self.needs.line_qty(idx)
        count = self.trolley.count_arg(call_args)
        aim = need_n - self.trolley.held(thing) if self.trolley.adds is not False else need_n
        if count is not None and abs(count - aim) > 1e-9 and sig not in self.warned:
            self.warned.add(sig)
            what = f"{self.needs.lines[idx]['product']} needs {need_n} units" if self.needs.several() and idx is not None else \
                f"the request needs a quantity of {need_n} of one listing"
            return f"Not run (no step used): {what}, but this call asks for {int(count)}. If you are sure, send the same call again."
        return ""

    # ----------------------------------------------------------------------------------------------- the hold ledger
    def gated(self, gate: str, text_body: str | None) -> str | None:
        """Names the gate a finish hold comes from (for the ledger); passes the text on."""
        if text_body:
            self.held_gate = gate
        return text_body

    def hard_stop(self) -> str:
        """What no release lets an order past: an empty cart, a total over the budget, or a line shown sold out."""
        if self.cart_empty():
            return ("Not run (no step used): the cart is empty as counted, so the store would turn this order down. Put your "
                    "verified pick in the cart first (one step), then order; if nothing fits, decline.")
        over = self.basket_total()
        if over is not None and is_number(self.needs.spend_cap) and over > self.needs.spend_cap + 1e-9:
            return (f"Not run (no step used): the basket totals {over:g} by the latest prices seen, over the budget of "
                    f"{self.needs.spend_cap:g}. {self.budget_line()} Replace a line with a verified listing whose unit price "
                    "is within the figure shown (the units stay as requested), then order.")
        sold = [t for t in self.trolley.units if self.line_sold(t)]
        if sold:
            return (f"Not run (no step used): the newest record shows {', '.join(sold)} sold out. Take it out of the cart, put in "
                    "another verified listing for that product (read its stock and price first), then order.")
        # (a line no check passed under the requirements in force, most often after a change: one cheap check, booked per
        # requirements version, so a change is never ordered past unread)
        for thing in [t for t in self.trolley.units if not self.verified(t)][:3]:
            ruling = self.check_item(thing, "ORDER") if self.moves_left() > 3 else ""
            if ruling:
                return "Not run yet (no step used). " + ruling
        return ""

    def released(self) -> bool:
        """The ledger leans toward finishing: after HOLDS_RELEASE advisory holds in the task, or once the steps kept
        for the order have begun, the advisory gates step aside (the procedure's own steps and the hard stops stay)."""
        return self.holds_n >= HOLDS_RELEASE or self.moves_left() <= self.order_due() + 1

    def screen_finish(self, cid: str, call_name: str, call_args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """A held-back answer for an order or decline (None lets it go), and the args to send. Every hold is booked in
        the ledger with its gate; the advisory ones count toward the release, the code's own work (cart fixes, the
        re-read, reads and recalls it runs) and the procedure's open steps do not."""
        kind = self.call_kind(call_name, call_args)
        self.held_gate = ""
        if kind == "order" and self.released():
            stop = self.hard_stop()
            mark = f"hard|{self.needs.version}|{self.trolley.version}"
            held = None
            if stop and self.__dict__.get("hard_holds", {}).get(mark, 0) < 2 and self.moves_left() > 1:
                hard = self.__dict__.setdefault("hard_holds", {})
                hard[mark] = hard.get(mark, 0) + 1
                held = self.gated("hard", stop)
            else:
                held = (self.gated("pages", self.pages_gate_text) if self.pages_gate() else None) or self.gated("duty", self.duty_gate("order"))
        else:
            held, call_args = self.finish_gates(cid, call_name, call_args, kind)
        advisory = self.held_gate not in ("", "duty", "hard", "cart_fix", "reread", "decline_opens", "recall", "pages")
        if held:
            self.holds_n += 1 if advisory else 0
        self.log.append({"kind": "hold" if held else "finish_go", "finish": kind, "gate": self.held_gate or "-",
                         "holds": self.holds_n, "released": kind == "order" and self.released(), "left": self.moves_left()})
        return held, call_args

    def finish_gates(self, cid: str, call_name: str, call_args: dict[str, Any], kind: str | None) -> tuple[str | None, dict[str, Any]]:
        """The finish gates before the release (each hold names its gate through gated)."""
        if kind == "order" and self.cart_empty() and f"empty|{self.trolley.version}" not in self.warned:
            # (an order of an empty cart is turned down and, on the last step, ends the errand unfinished)
            self.warned.add(f"empty|{self.trolley.version}")
            return self.gated("hard", "Not run (no step used): the cart is empty as counted, so the store would turn this order down. Put your "
                              "verified pick in the cart first (one step), then order; if nothing fits, decline. If you are sure the cart "
                              "holds the item, send the order again."), call_args
        if kind == "order" and self.moves_left() > 3:
            thing = self.chosen()
            cheaper = self.cheaper_fit(thing) if thing else ""
            told = self.__dict__.setdefault("cheaper_told", {})
            if cheaper and told.get((thing, cheaper), 0) < 2:
                told[(thing, cheaper)] = told.get((thing, cheaper), 0) + 1
                return self.gated("cheaper", f"Not run yet (no step used). The independent item check found that {cheaper} meets every requirement, and "
                                  f"it costs less than {thing}. The cheapest fit is asked for and the check's reading stands over yours: take "
                                  f"{thing} out of the cart, put {cheaper} in, re-read its stock and price, and order it."), call_args
        if kind == "order" and self.order_vetted and self.vetted_at != (self.needs.version, self.trolley.version):
            self.order_vetted = False      # (a change or a cart call since the last vetting: vet again)
        if kind == "order" and not self.order_vetted and self.moves_left() > 2 and self.wire.spare() > CLOSING:
            self.order_vetted = True
            self.vetted_at = (self.needs.version, self.trolley.version)
            faults = self.trolley.faults(self.needs, self.line_of)
            over = self.basket_total()
            if over is not None and is_number(self.needs.spend_cap) and over > self.needs.spend_cap + 1e-9:
                faults.append(f"the basket totals {over:g} by the latest prices seen, over the budget of {self.needs.spend_cap:g}")
            if faults and self.moves_left() > 3:
                # (held again while the same cart stays faulty, at most twice for one cart state)
                held_n = self.__dict__.setdefault("fault_holds", {})
                mark = (self.needs.version, self.trolley.version)
                held_n[mark] = held_n.get(mark, 0) + 1
                if held_n[mark] < 2:
                    self.order_vetted = False
                fixes = self.cart_fix_plan()
                if fixes:
                    # (the code sends the corrections itself, then the re-read and this order: no model turn for them)
                    self.fix_queue = fixes
                    self.auto_order = {"call": (call_name, call_args), "version": self.needs.version, "reads": 0, "sends": 0, "kind": "order"}
                    self.log.append({"kind": "cart_fix", "faults": faults, "calls": fixes})
                    return self.gated("cart_fix", "Not run yet (no step used). The cart as counted is not ready: " + "; ".join(faults) + f". The code makes "
                                      f"the cart changes itself now ({len(fixes)} cart call{'s' if len(fixes) > 1 else ''}), re-reads the stock and "
                                      "price, and sends this order again once the cart is right; read the results."), call_args
                return self.gated("cart_faults", "Not run yet (no step used). The cart as counted is not ready: " + "; ".join(faults) + ". Fix it, then order."), call_args
            unchecked = [t for t in self.trolley.units if not self.verified(t)][:3] or ([self.chosen()] if self.chosen() else [])
            for thing in unchecked:
                ruling = self.check_item(thing, "ORDER")
                if ruling:
                    return self.gated("item_check", "Not run yet (no step used). " + ruling), call_args
            thing = self.chosen()
            better = self.rank_check(thing) if thing and not self.needs.several() else ""
            if better:
                return self.gated("rank", "Not run yet (no step used). " + better), call_args
            # (the first vetting of the task always shows the checklist: the actor may owe a procedure step the code does
            # not track; later ones, after a refusal or a cart fix, skip it once the code has settled the cart)
            due = self.reread_due() if self.cart_settled() and self.__dict__.get("checklist_shown") else None
            self.checklist_shown = True
            if due:
                if self.moves_left() >= len(due) + 2 and self.wire.spare() > CLOSING + 10:
                    # (only the re-read after the last cart change is missing: the code reads, then orders)
                    self.auto_order = {"call": (call_name, call_args), "version": self.needs.version, "reads": 0, "sends": 0, "kind": "order"}
                    return self.gated("reread", "Not run yet (no step used). The cart checks out: every line passed the item check under the requirements "
                                      "in force, and the units and the total are right. Only a stock and price re-read after the last cart change "
                                      "is missing, so the code re-reads " + ", ".join(due[:3]) + " now and then sends this order itself."), call_args
                due = None
            if due is None:
                first = "one listing per requested product, each at its units," if self.needs.several() else "exactly one line, the chosen item at the required quantity,"
                return self.gated("checklist", "Not run yet (no step used). Pre-order check: (1) the cart holds " + first + " and nothing else; "
                                  "(2) after your last cart change you re-read the stock and price of what you order with a tool that shows them; it is in stock and the total is within the current budget; "
                                  "(3) its own properties meet every stated requirement, every need the shopper revealed and every change announced; "
                                  "(4) if the cheapest or best was asked for, no better fitting listing exists among everything you could find; "
                                  "(5) every procedural rule in the request has been followed. If anything is unchecked, check it now; otherwise call the order tool again."), call_args
            # (settled and freshly read: the order goes on to the procedure gate without the checklist turn)
            self.log.append({"kind": "order_settled", "cart": dict(self.trolley.units)})
        if kind == "order" and self.pages_gate():
            return self.gated("pages", self.pages_gate_text), call_args
        if kind == "order":
            return self.gated("duty", self.duty_gate("order")) or None, call_args
        if kind != "decline":
            return None, call_args
        # (a decline: every product with a verified, in-stock listing is held once per requirements version)
        picks = self.verified_picks()
        whole = len(picks) == max(1, len(self.needs.lines)) and not any(self.line_sold(t) for t in picks.values())
        if whole and self.role("order") and self.moves_left() > len(self.needs.lines) + 1 and f"picks|{self.needs.version}" not in self.warned:
            self.warned.add(f"picks|{self.needs.version}")
            self.decline_blocked = True
            what = "; ".join(f"{self.needs.lines[i]['product']}: {t}" for i, t in sorted(picks.items())) if self.needs.several() else picks[0]
            return self.gated("decline_picks", "Not run (no step used): the independent check found a fitting, in-stock listing for every "
                              f"product ({what}). Put each in the cart at its units, re-check stock and price, and order; decline only if "
                              "one of its fields clearly fails a requirement."), call_args
        queue = self.named_unread()[:3]
        if queue and not self.__dict__.get("decline_opens") and self.role("details") and self.moves_left() > len(queue) + 3 \
                and self.wire.spare() > CLOSING + 40:
            # (named listings nobody opened: their records are read by the code before any decline)
            self.decline_opens = True
            self.decline_vetted = True
            self.open_queue = queue
            return self.gated("decline_opens", "Not run yet (no step used). Listings the checks named were never opened, so their records are read for you now, "
                              "before any decline: " + ", ".join(queue) + ". Judge each against the requirements by meaning (a field of the "
                              "property's own name decides; when none exists, an alternative field, the options list or the title may show the "
                              "value), buy one that fits, and decline only if each one fails."), call_args
        # (whether a read listing is left to judge is taken before the re-check settles it: it decides whether the
        # re-check runs and whether this decline's audit may go to the stronger model)
        pooled = self.pool_open()
        recheck = self.decline_recheck() if pooled else ""
        if recheck:
            return self.gated("decline_recheck", recheck), call_args
        if not pooled:
            self.log.append({"kind": "decline_recheck", "skipped": "no read listing left to judge"})
        recall = self.recall_hold()
        if recall:
            return self.gated("recall", recall), call_args
        reread = len(self.archive.read) > self.__dict__.get("audit_reads", 10 ** 9) and self.__dict__.get("audits", 0) < 2
        if reread and self.decline_vetted and self.moves_left() > 3 and self.wire.spare() > CLOSING + 30:
            # (listings were read since the audit, most often the rows it named: their records are checked too)
            audit_note = self.review_decline(pooled=pooled)
            if audit_note:
                return self.gated("audit", "Not run yet (no step used). " + audit_note), call_args
        if not self.decline_vetted and self.moves_left() > 6 and self.wire.spare() > CLOSING + 30:
            self.decline_vetted = True
            audit_note = self.review_decline(pooled=pooled)
            if audit_note:
                return self.gated("audit", "Not run yet (no step used). " + audit_note), call_args
            return self.gated("decline_checklist", "Not run yet (no step used). Before declining, be sure: you searched in several ways (different words, filters, pages), you checked every candidate's own properties, "
                              "and each one fails a stated requirement, a revealed need, the budget or stock. Remember that close synonyms and any key spelling count. "
                              "If something does fit, buy it instead; if nothing fits, call the decline tool again."), call_args
        grounds = self.decline_grounds()
        if not grounds and self.moves_left() > 4 and f"grounds|{self.needs.version}" not in self.warned:
            # (a decline needs a requirement the records show failing; with none, the closest listing is tried first)
            self.warned.add(f"grounds|{self.needs.version}")
            return self.gated("decline_grounds", "Not run (no step used): nothing seen so far fails a requirement the records can show: no "
                              "check rejected a listing, and none read is sold out or over the budget. Open the closest in-budget, in-stock "
                              "listing of the requested kind, put it in the cart at its units if its fields fit, and order it; decline only "
                              "when its own fields fail a requirement."), call_args
        self.log.append({"kind": "decline_grounds", "grounds": grounds or "none (released)"})
        gate = self.duty_gate("decline")
        if gate:
            return self.gated("duty", gate), call_args
        return None, self.scrub_decline(call_args, call_name)

    def decline_grounds(self) -> str:
        """Why a decline may go ('' when nothing checkable failed): a check rejected a listing on named problems, the store
        turned a listing down, a listing read is sold out or over the budget, or no listing was seen at all."""
        if self.flaws:
            return "a check named a failed requirement"
        if self.dead:
            return "the store turned a listing down"
        if not self.archive.ids:
            return "no listing was found"
        cap, qty = self.needs.spend_cap, self.needs.quantity()
        for ident in sorted(self.archive.read)[:40]:
            cost = self.archive.price(ident)
            if self.line_sold(ident) or (is_number(cap) and is_number(cost) and cost * qty > cap + 1e-9):
                return "a listing read is sold out or over the budget"
        return ""

    def decline_recheck(self) -> str:
        """Before a decline: on each product line with no verified pick, the read listings closest to fitting are
        judged by the order-stage check, up to RECHECK_PER_LINE each (a rejection there gets the second opinion; a
        verdict already booked costs nothing). When every line then has a verified pick, the decline is held with the
        picks to buy, once per requirements version; a line left without one lets the decline go on."""
        mark = f"recheck|{self.needs.version}"
        n_lines = max(1, len(self.needs.lines))
        if mark in self.warned or not self.role("order") or self.moves_left() <= max(4, n_lines + 1) \
                or self.wire.spare() <= CLOSING + 30:
            return ""
        open_lines = [i for i in range(n_lines) if i not in self.verified_picks()]
        for idx in open_lines:
            for ident in self.recheck_pool(idx)[:RECHECK_PER_LINE]:
                if self.wire.spare() <= CLOSING + 30:
                    break
                self.recheck_one(ident)
                if idx in self.verified_picks():
                    break
            if idx not in self.verified_picks():
                break
        picks = self.verified_picks()
        self.log.append({"kind": "decline_recheck", "open_lines": open_lines, "picks": dict(picks)})
        if any(i not in picks for i in range(n_lines)) or any(self.gone(t) for t in picks.values()) \
                or (not open_lines and self.needs.several() and self.__dict__.get("decline_blocked")):
            return ""
        self.warned.add(mark)
        in_cart = all(t in self.trolley.units for t in picks.values())
        if self.needs.several():
            what = "; ".join(f"{self.needs.lines[i]['product']}: {t}" for i, t in sorted(picks.items()))
            how = ("The cart holds them: re-read their stock and price" if in_cart else
                   "Put each in the cart at its units, re-read their stock and price" if self.role("cart") else "Re-read their stock and price")
            return ("Not run (no step used): before any decline the independent check judged the closest listings read again, and "
                    f"every product has a listing meeting every requirement now in force ({what}). {how}, and order them; decline "
                    "only if one of their fields clearly fails a requirement.")
        pick, qty = picks[0], self.needs.line_qty(0)
        how = ("It is in the cart: re-read its stock and price" if in_cart else
               f"Put it in the cart ({qty} unit{'s' if qty != 1 else ''}), re-read its stock and price" if self.role("cart") else "Re-read its stock and price")
        return (f"Not run (no step used): before any decline the independent check judged the closest listings read again, and it "
                f"finds {pick} meeting every requirement now in force, reading each property by meaning. {how}, and order it; "
                "decline only if one of its fields clearly fails a requirement.")

    def recheck_pool(self, idx: int) -> list[str]:
        """Read listings for product line idx that the order-stage check may still pass: within the budget as last priced,
        not shown sold out, not turned down by the store, not in a refused cart and not already settled by the second
        opinion; fewest problems a check named first, then those the decline audit called possible fits, then the
        cheapest."""
        cap, version = self.needs.spend_cap, self.needs.version
        ranked = []
        for ident in self.archive.read:
            key = f"{ident}|{version}"
            if key in self.adjudicated or key in self.passed or ident in self.refused_lines or ident in self.dead:
                continue
            where = self.line_of.get(ident) if self.needs.several() else 0
            if where is None:
                where = self.needs.line_for(" ".join(self.archive.about(ident))[:3000])
            if where != idx:
                continue
            cost = self.archive.price(ident)
            if (is_number(cap) and is_number(cost) and cost * self.needs.line_qty(idx) > cap + 1e-9) or self.gone(ident):
                continue
            ranked.append((self.flaws.get(key, 0), ident not in self.audit_fits, cost if is_number(cost) else 1e18, ident))
        return [entry[-1] for entry in sorted(ranked)]

    def recheck_one(self, ident: str) -> None:
        """The order-stage check of one listing (a cheap rejection, new or booked earlier, gets the second opinion)."""
        key = f"{ident}|{self.needs.version}"
        if key in self.rulings and not self.rulings[key] and key not in self.passed:
            del self.rulings[key]      # (no verdict was had before: asked again)
        self.check_item(ident, "ORDER")

    def basket_total(self) -> float | None:
        """The cart's total by the newest prices seen (None when a price is unknown)."""
        total = 0.0
        for thing, units in self.trolley.units.items():
            cost = self.archive.price(thing)
            if cost is None:
                return None
            total += cost * units
        return total if self.trolley.units else None

    @staticmethod
    def headroom(cap: float, rest: float, units: float, whole: bool) -> float:
        """The most one unit of a line may cost: (budget - the other lines' units x unit prices) / its units, floored to
        the price grain (a whole unit when the budget and the prices are whole numbers, else a hundredth), never below 0."""
        room = (cap - rest) / max(units, 1e-9)
        room = float(int(room // 1)) if whole else int(room * 100 + 1e-9) / 100.0
        return max(0.0, room)

    def budget_line(self) -> str:
        """The budget arithmetic the actor reads: the total by line (units x newest unit price) against the budget in
        force, and the most each line's unit may cost; '' with no budget, an unknown price or nothing to work out."""
        cap = self.needs.spend_cap
        if not is_number(cap):
            return ""
        prices = {t: self.archive.price(t) for t in self.trolley.units}
        if any(not is_number(c) for c in prices.values()):
            return ""
        whole = float(cap).is_integer() and all(float(c).is_integer() for c in prices.values())
        if not prices:
            qty = self.needs.line_qty(0)
            if len(self.needs.lines) == 1 and qty > 1:
                return f"[budget] {cap:g} for {qty} units: each unit may cost at most {self.headroom(cap, 0.0, qty, whole):g}."
            return ""
        total = sum(c * self.trolley.units[t] for t, c in prices.items())
        parts = []
        for thing, units in self.trolley.units.items():
            cost = prices[thing]
            room = self.headroom(cap, total - cost * units, units, whole)
            parts.append(f"{int(units)} x {thing} at {cost:g} = {cost * units:g} (unit may cost at most {room:g})")
        state = f"over the budget by {total - cap:g}" if total > cap + 1e-9 else f"within it, {cap - total:g} left"
        return f"[budget] {cap:g}; cart total {total:g}, {state}: " + "; ".join(parts) + "."

    def over_lines(self) -> list[str]:
        """Cart lines to replace because the total is over the budget, dearest line first: only for a cart that holds
        one listing per product with every price known."""
        cap, total = self.needs.spend_cap, self.basket_total()
        if not is_number(cap) or total is None or total <= cap + 1e-9 or len(self.trolley.units) != len(self.needs.lines):
            return []
        return sorted(self.trolley.units, key=lambda t: -(self.archive.price(t) or 0.0) * self.trolley.units[t])

    def cart_settled(self) -> bool:
        """The cart as counted is ready to order: every listing passed the item check under the requirements in force
        and still shows the price it was checked at and no sell-out, one listing per product at its units, and the total
        is known and within the budget."""
        units = self.trolley.units
        if not units or self.trolley.blind or self.trolley.faults(self.needs, self.line_of):
            return False
        if len(units) != len(self.needs.lines):
            return False
        total = self.basket_total()
        if total is None or (is_number(self.needs.spend_cap) and total > self.needs.spend_cap + 1e-9):
            return False
        checked_at = self.__dict__.get("pass_price", {})
        return all(self.verified(t) and not self.gone(t) and checked_at.get(t) == self.archive.price(t) for t in units)

    def reread_due(self) -> list[str]:
        """Cart listings whose record was not read after the last cart change (none when the store has no details tool)."""
        if not self.role("details"):
            return []
        return [t for t in self.trolley.units if self.read_at.get(t, -1) <= self.trolley.changed_at]

    def cart_fix_plan(self) -> list[dict[str, Any]]:
        """The cart calls the code sends itself to correct the cart (Basket.fix_calls), when the steps allow them, the
        re-read and the order; at most three rounds per task."""
        if not self.role("cart") or self.fix_rounds >= 3 or self.wire.spare() < CLOSING + 10:
            return []
        good = {t for t in self.trolley.units if self.verified(t)}
        calls = self.trolley.fix_calls(self.needs, self.line_of, good)
        if not calls or self.moves_left() < len(calls) + 3:
            return []
        self.fix_rounds += 1
        return calls

    def page_tools(self) -> list[str]:
        """The tools that show one listing: one argument names it and no other is required, and no other role (search,
        cart, message, finish, compare, hold, report) took the tool; the details and view tools, and any other whose
        words read as showing, not holding, buying, ending, putting in, taking out, messaging or searching."""
        mark = tuple(sorted(self.role_map.items()))
        cached = self.__dict__.get("page_tool_list")
        if cached is not None and cached[0] == mark:
            return cached[1]
        busy = {self.role(r) for r in ("search", "cart", "cart_view", "message", "order", "decline", "compare", "hold", "report")} - {None}
        found: list[str] = []
        for name in self.tools.defs:
            key = self.tools.id_key(name)
            if name in busy or not key or any(k != key for k in self.tools.required(name)):
                continue
            said, label = self.tools.said(name), spoken(name)
            shows = (READ_CUE.search(said) or VIEW_CUE.search(said)) and key in self.tools.ids(name) and not (
                HOLD_CUE.search(said) or BUY_CUE.search(said) or END_CUE.search(said) or re.search(PUT_WORDS, label)
                or re.search(TAKE_WORDS, label) or TALK_CUE.search(label) or FIND_CUE.search(label) or BAG_CUE.search(label))
            if name in (self.role("details"), self.role("view")) or shows:
                found.append(name)
        self.page_tool_list = (mark, found)
        return found

    def changes_warned(self) -> bool:
        """The request warns that things can change while the errand runs (then reads count only after the last cart change)."""
        return bool(CHANGE_WARN.search(self.request.lower()))

    def page_args(self, tool: str, ident: str) -> dict[str, Any] | None:
        """Arguments that show one listing with a page tool (None when its schema will not take the id)."""
        key = self.tools.id_key(tool)
        if not key:
            return None
        call_args, faults = ArgCheck.execute({key: ident}, self.shapes.get(tool) or {})
        return None if faults else call_args

    def pages_due(self) -> list[tuple[str, str]]:
        """(page tool, listing) reads an order still owes: each cart listing with each page tool that has not read it,
        and, when the request warns that things can change, each read made before the last cart change; the reads
        never made first."""
        again = self.changes_warned()
        due: list[tuple[bool, str, str]] = []
        for ident in self.trolley.units:
            for tool in self.page_tools():
                at = self.page_reads.get((tool, ident))
                if (at is None or (again and at <= self.trolley.changed_at)) and self.page_args(tool, ident) is not None:
                    due.append((at is not None, tool, ident))
        return [(tool, ident) for _, tool, ident in sorted(due, key=lambda d: d[0])]

    def pages_gate(self) -> bool:
        """Holds an order (no step used) while reads are due, at most twice per cart version, queueing as many as the
        steps allow with the order and one more kept free; the code makes them next (page_step)."""
        if self.wire.spare() <= CLOSING + 10:
            return False
        due = self.pages_due()
        room = self.moves_left() - 2
        mark = f"pages|{self.trolley.version}"
        holds = self.__dict__.setdefault("page_holds", {})
        if not due or room < 1 or holds.get(mark, 0) >= 2:
            return False
        holds[mark] = holds.get(mark, 0) + 1
        self.page_queue = due[:room]
        again = " (again after the last cart change: the request says things can change)" if self.changes_warned() else ""
        self.pages_gate_text = ("Not run yet (no step used). Before an order the store wants each listing you order read with every tool "
                                f"that shows one listing{again}; the code reads " + ", ".join(f"{i} with {t}" for t, i in due[:room][:6])
                                + " now. Check that the records still fit every requirement, then order again.")
        return True

    def read_or_order(self) -> dict[str, Any] | None:
        """The code's own finish (finish_now): a read the order still owes, one per move while a step stays for the order
        (three at most), else the order."""
        due = self.pages_due()
        if due and self.moves_left() >= 2 and self.__dict__.get("forced_reads", 0) < 3:
            self.forced_reads = self.__dict__.get("forced_reads", 0) + 1
            tool, ident = due[0]
            return {"name": tool, "args": self.page_args(tool, ident)}
        return self.order_move()

    def page_step(self) -> bool:
        """Makes the (page tool, listing) reads an order was held for (scripted, one reply)."""
        queue = self.__dict__.pop("page_queue", None)
        if not queue or self.wire.done or self.wire.broke or self.moves_left() < len(queue) + 1:
            return False
        proposals = [{"id": f"page_{n}_{self.wire.tick}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
                     for n, (tool, args) in enumerate((tool, self.page_args(tool, ident)) for tool, ident in queue) if args]
        if not proposals:
            return False
        self.chat.append({"role": "assistant", "content": "Before the order I read each listing in the cart with the store's page tools.",
                          "tool_calls": proposals})
        self.covering = True
        try:
            self.post_replies(self.dispatch(proposals, scripted=True))
        finally:
            self.covering = False
        return True

    def auto_step(self) -> bool:
        """The code's own moves toward an order the checks settled, one per turn: a queued cart correction, the stock and
        price re-read after the last cart change, then the order. A notice, a change, a failure or an unsettled cart on
        the way hands the errand back to the actor, and the next order is vetted again."""
        if self.wire.done or self.wire.broke or self.moves_left() <= 1:
            self.fix_queue, self.auto_order = [], None
            return False
        if self.fix_queue:
            call_args = self.fix_queue.pop(0)
            proposal = {"id": f"fix_{self.wire.tick}_{len(self.fix_queue)}", "type": "function",
                        "function": {"name": self.role("cart"), "arguments": json.dumps(call_args)}}
            self.chat.append({"role": "assistant", "content": "I correct the cart so it holds one listing per product at its units.",
                              "tool_calls": [proposal]})
            before = self.wire.tick
            self.fixing = True
            try:
                responses = self.dispatch([proposal], scripted=True)
            finally:
                self.fixing = False
            self.post_replies(responses)
            if self.flag or self.failed_cart or self.wire.tick == before:
                self.fix_queue, self.auto_order = [], None
            return True
        plan, self.auto_order = self.auto_order, None
        if plan is None:
            return False
        due = self.reread_due()
        call_args = self.details_call(due[0]) if due else None
        if plan["version"] != self.needs.version or self.needs.reply_due or not self.cart_settled() \
                or self.moves_left() <= 2 or self.wire.spare() < CLOSING \
                or (due and (plan["reads"] > len(self.needs.lines) or not call_args or self.moves_left() < len(due) + 2)):
            self.order_vetted = False
            self.log.append({"kind": "auto_order_dropped", "plan": plan["kind"]})
            self.chat.append({"role": "user", "content": "[note] The code did not send the order itself: the cart or the requirements "
                              "are not settled as it counts them (read the results above). Make the cart right, re-read stock and price, "
                              "then order.\n" + self.status_line()})
            return False
        if due:
            proposal = {"id": f"reread_{self.wire.tick}", "type": "function",
                        "function": {"name": self.role("details"), "arguments": json.dumps(call_args)}}
            self.chat.append({"role": "assistant", "content": f"Before the order I re-read the stock and price of {due[0]}.",
                              "tool_calls": [proposal]})
            before = self.wire.tick
            self.covering = True
            try:
                responses = self.dispatch([proposal], scripted=True)
            finally:
                self.covering = False
            self.post_replies(responses)
            plan["reads"] += 1
            if self.flag or self.wire.tick == before:
                self.order_vetted = False
            else:
                self.auto_order = plan
            return True
        if plan["sends"] >= 2:
            return False
        plan["sends"] += 1
        call_name, call_args = plan["call"]
        proposal = {"id": f"auto_order_{self.wire.tick}", "type": "function",
                    "function": {"name": call_name, "arguments": json.dumps(call_args)}}
        self.chat.append({"role": "assistant", "content": "The cart is verified and freshly read; I place the order.", "tool_calls": [proposal]})
        before = self.wire.tick
        responses = self.dispatch([proposal], scripted=True)
        self.post_replies(responses)
        self.log.append({"kind": "auto_order", "plan": plan["kind"], "sent": self.wire.tick != before, "finished": self.wire.done})
        if self.wire.tick == before and self.duty_due == "order":
            self.auto_order = plan     # (the procedure's open steps run first, then the order goes again)
        return True

    def duty_gate(self, stage: str) -> str:
        """Holds a finish (no step used) while a procedure step for it is open and the steps left allow doing it;
        the code then carries the steps out before the actor's next turn."""
        self.duties.refresh(self.picks(), self.trolley.version, self.quote())
        steps_open = self.gate_duties(stage)
        doable = [d for d in steps_open if d.tries < 2 and (d.stated or self.moves_left() > len(steps_open) + 4)]
        if not doable or self.moves_left() <= len(doable) + 2 or self.wire.spare() < CLOSING:
            # (the tracked steps are done or cannot be: the request's other rules are checked against the record)
            return self.rule_audit(stage)
        self.duty_due = stage
        return ("Not run yet (no step used). The request's procedure still has open steps: " + "; ".join(d.brief() for d in doable)
                + ". The code carries them out now; read the results, then send the " + ("order" if stage == "order" else "decline") + " again.")

    def gate_duties(self, stage: str) -> list[Duty]:
        """The open steps a finish waits for; the basket summary and the hold wait for a basket that is ready."""
        steps_open = self.duties.open_for(stage)
        if stage == "order" and (self.trolley.faults(self.needs, self.line_of) or not self.trolley.units):
            steps_open = [d for d in steps_open if d.kind not in ("confirm", "tell_basket", "hold")]
        return sorted(steps_open, key=lambda d: d.kind in ("confirm", "tell_basket"))

    # ----------------------------------------------------------------------------------------------- stated-rule audit
    def rule_texts(self) -> list[str]:
        """The request's rules no tracked step proves: stated steps of kind 'other' or 'ask_before_change'; when the
        reading gave no procedures at all, its rules (said in the request) that share few words with every tracked step."""
        def words(text_body: str) -> set[str]:
            return {w for w in re.findall(r"[a-z0-9]+", text_body.lower()) if w not in LINK_WORDS}
        heard = words(self.request)
        texts = [d.text for d in self.duties.items if d.stated and d.state != "void" and d.kind in ("other", "ask_before_change")]
        tracked = [words(d.text) for d in self.duties.items if d.kind not in ("other", "ask_before_change")]
        procs = self.needs.reading.get("procedures")
        for rule in (self.needs.reading.get("rules") or []) if not (isinstance(procs, list) and procs) else []:
            if not isinstance(rule, str) or not rule.strip():
                continue
            mine = words(rule)
            if not mine or len(mine & heard) < 0.6 * len(mine):
                continue
            if any(len(mine & have) >= 0.6 * len(mine) for have in tracked + [words(t) for t in texts]):
                continue
            texts.append(re.sub(r"\s+", " ", rule).strip()[:200])
        return list(dict.fromkeys(texts))[:RULE_MOST]

    def action_record(self) -> str:
        """Every answered world call as one numbered line (tool, arguments, start of the result), with a marker after
        the newest cart change."""
        envs = [e for e in self.log if e.get("kind") == "env" and isinstance(e.get("reply"), dict) and e.get("sent")]
        last_cart = max((i for i, e in enumerate(envs) if (e["sent"][0] or {}).get("name") == self.role("cart")), default=-1)
        rows = []
        for idx, entry in enumerate(envs):
            act = entry["sent"][0] or {}
            obs = entry["reply"].get("observation")
            wide = 260 if act.get("name") in (self.role("message"), self.role("hold"), self.role("report"), self.role("compare")) else 100
            rows.append(f"{entry.get('step')}. {act.get('name')} {as_text(act.get('args'), 120)} -> {as_text(obs, wide)}")
            if idx == last_cart:
                rows.append("---- LAST CART CHANGE ----")
        text_body = "\n".join(rows)
        return text_body if len(text_body) <= 9000 else "(earlier calls cut)\n" + text_body[-9000:]

    def rule_audit(self, stage: str) -> str:
        """Before an order or a decline: one cheap check of the request's other rules against the action record. A rule
        the record does not show done gets the one non-finishing call the check names (run by duty_step through the
        usual screen), or one note for the actor. At most RULE_ROUNDS audits per cart and requirements state, RULE_CALLS
        per task, RULE_STEPS added calls and one message; never an order, a decline or a cart change."""
        key = (stage, self.trolley.version, self.needs.version)
        if key in self.rule_clear or self.rule_rounds.get(key, 0) >= RULE_ROUNDS or self.rule_calls >= RULE_CALLS:
            return ""
        if self.moves_left() < 4 or self.wire.spare() < CLOSING + 20 or self.wire.credit_out:
            return ""
        rules = self.rule_texts()
        if not rules:
            return ""
        self.rule_rounds[key] = self.rule_rounds.get(key, 0) + 1
        self.rule_calls += 1
        banned = self.finishers() | {self.role("cart")}
        tool_rows = [f"- {t['function']['name']}: {as_text(t['function'].get('description', ''), 140)} | arguments: "
                     f"{as_text(t['function'].get('parameters') or {}, 300)}" for t in self.toolset if t["function"]["name"] not in banned]
        cart = ", ".join(f"{int(u)} x {t}" for t, u in self.trolley.units.items()) or "empty"
        ask_text = (RULE_AUDIT.replace("FINISH", "place the order" if stage == "order" else "decline (end without buying)")
                    + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
                    + "\n\nRULES TO JUDGE:\n" + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))
                    + "\n\nTHE CART NOW: " + cart
                    + "\n\nRECORD OF WHAT WAS DONE (oldest first):\n" + (self.action_record() or "(nothing yet)")
                    + "\n\nTOOLS THAT MAY BE CALLED:\n" + "\n".join(tool_rows))
        answer = self.wire.consult(ask_text, 900, RULE_AUDITOR) or {}
        verdicts = [v for v in answer.get("rules") or [] if isinstance(v, dict)]
        missing = [v for v in verdicts if str(v.get("status") or "").strip().lower().replace(" ", "_") == "not_done"
                   and is_number(v.get("n")) and 1 <= int(v["n"]) <= len(rules)]
        made, why = self.rule_proposal(answer.get("call"), stage)
        self.log.append({"kind": "rule_audit", "stage": stage, "rules": rules, "verdicts": verdicts[:RULE_MOST],
                         "call": answer.get("call"), "sent": bool(made), "why": why, "model": self.wire.answered_by})
        if not answer or not missing:
            self.rule_clear.add(key)
            return ""
        named = "; ".join(f"'{rules[int(v['n']) - 1][:120]}' ({str(v.get('why') or '')[:80]})" for v in missing[:3])
        if made:
            self.rule_queue = [(made, f"The request's rule is not done yet; I do it now: {named[:200]}.")]
            self.duty_due = stage
            return ("Not run yet (no step used). The request's rules are checked against what was done; not done yet: " + named
                    + ". The code carries out the missing step now; read its result, then send the "
                    + ("order" if stage == "order" else "decline") + " again.")
        # (nothing the code may call does it: the actor is told once for this state, and the next finish goes)
        self.rule_clear.add(key)
        return ("Not run yet (no step used). The request's rules are checked against what was done; not done yet: " + named
                + ". Carry out each with one tool call now if a tool does it (never by changing the cart for it), then send the "
                + ("order" if stage == "order" else "decline") + " again; if none applies, send it again as it is.")

    def rule_proposal(self, call: Any, stage: str) -> tuple[dict[str, Any] | None, str]:
        """(proposal, '') for the call a rule audit named when the code may send it, else (None, why not)."""
        if not isinstance(call, dict) or not call.get("name"):
            return None, "no call"
        name, args = str(call.get("name")), call.get("args")
        args = args if isinstance(args, dict) else (json_object(args) if isinstance(args, str) and args.strip() else {})
        if name not in self.shapes or name in self.finishers() or name == self.role("cart") or self.call_kind(name, args):
            return None, "not a tool the audit may call"
        if self.rule_steps >= RULE_STEPS or self.moves_left() < 4:
            return None, "no steps to spare"
        if name == self.role("message"):
            room = self.talk_limit() if self.msg_cap is not None else ASK_DEFAULT + 1
            if self.rule_msgs >= 1 or self.msgs_sent >= room or (self.msg_cap is not None and self.msgs_sent >= self.msg_cap):
                return None, "no message to spare"
        checked, faults = ArgCheck.execute(args or {}, self.shapes[name])
        if faults:
            return None, "arguments do not fit the schema: " + "; ".join(faults[:2])
        # (the same call made since the last cart change already shows whatever it can show)
        sig = name + as_text(checked)
        sent_since = {(e["sent"][0] or {}).get("name", "") + as_text((e["sent"][0] or {}).get("args"))
                      for e in self.log if e.get("kind") == "env" and e.get("sent") and isinstance(e.get("step"), int)
                      and e["step"] > self.trolley.changed_at}
        if sig in self.rule_sent or sig in sent_since:
            return None, "the same call was already made"
        self.rule_sent.add(sig)
        return {"id": f"rule_{stage}_{self.wire.tick}", "type": "function", "function": {"name": name, "arguments": json.dumps(checked)}}, ""

    def picks(self) -> list[str]:
        return list(self.trolley.units)

    def quote(self) -> Any:
        """What a basket summary states: each cart listing with its units and newest price."""
        return tuple(sorted((t, u, self.archive.price(t)) for t, u in self.trolley.units.items())) or None

    def dispatch(self, proposals: list[dict[str, Any]], scripted: bool = False) -> list[tuple[str, str]]:
        responses: list[tuple[str, str]] = []
        self.begin_batch()
        permitted = {tdef["function"]["name"] for tdef in self.offered()}
        to_send: list[tuple[str, str, dict[str, Any]]] = []
        queued_messages = 0
        cart_held = False
        self.batch_puts = []
        for proposal in proposals:
            func = proposal.get("function") or {}
            if func.get("name") == self.role("cart"):
                raw = func.get("arguments")
                put_args = raw if isinstance(raw, dict) else (json_object(raw) if isinstance(raw, str) and raw.strip() else {})
                if isinstance(put_args, dict) and self.trolley.is_put(put_args) and self.trolley.aim(put_args):
                    self.batch_puts.append(self.trolley.aim(put_args))
        for proposal in proposals:
            if cart_held and proposal["function"]["name"] == self.role("cart"):
                # (a held cart change makes the later ones in the same reply stale)
                responses.append((proposal["id"], "Not run (no step used): an earlier cart change in this reply was not run; read why, "
                                                  "then send the cart changes again."))
                continue
            refusal, call_name, call_args = self.screen(proposal, permitted, scripted, len(proposals), queued_messages)
            if refusal:
                cart_held = cart_held or (call_name == self.role("cart") and isinstance(call_args, dict) and bool(call_args)
                                          and not self.trolley.is_put(call_args))
                responses.append((proposal["id"], refusal))
                continue
            if call_name == self.role("message"):
                queued_messages += 1
            to_send.append((proposal["id"], call_name, call_args))
        if to_send and to_send[0][1] in self.finishers():
            cid, call_name, call_args = to_send[0]
            held, call_args = self.screen_finish(cid, call_name, call_args)
            if held:
                responses.append((cid, held))
                return responses
            to_send[0] = (cid, call_name, call_args)
        if to_send and to_send[0][1] not in self.finishers() and self.finishers():
            hold = max(0, self.moves_left() - 1)
            for cid, _, _ in to_send[hold:]:
                responses.append((cid, "Not run (no step used): only the last step is left, and it must be the order (or the decline)."))
            to_send = to_send[:hold]
        # (one action per world call; a reply that brings a notice, a refusal or a failure stops the rest of the
        # batch unsent, so the actor reads it before anything else goes out)
        stop = ""
        for cid, call_name, call_args in to_send:
            if self.wire.done or self.wire.broke or self.moves_left() <= 0:
                responses.append((cid, "Not run: the episode is over or no steps are left."))
                continue
            if stop:
                responses.append((cid, f"Not run (no step used): {stop} Read it first; send this call again if it still makes sense."))
                continue
            resp, text_body = self.send_one(call_name, call_args)
            responses.append((cid, text_body))
            if resp is None:
                stop = "an earlier call in this reply failed."
            elif self.flag:
                stop = "an earlier call in this reply brought a notice, a refusal or a change."
        call_rank = {c["id"]: i for i, c in enumerate(proposals)}
        responses.sort(key=lambda req: call_rank.get(req[0], 0))
        return responses

    def send_one(self, call_name: str, call_args: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        """Sends one action on the next step and books its reply; returns (the reply, the text the actor reads)."""
        prior = self.wire.tick
        resp = self.wire.step({"name": call_name, "args": call_args})
        if self.wire.orphan is not None:
            # (the reply to an earlier lost call came back on its replay: it is booked like any other)
            action, late = self.wire.orphan
            self.wire.orphan = None
            self.log.append({"kind": "env", "step": prior, "sent": [action], "reply": late, "late": True})
            late_text = self.absorb(action.get("name"), action.get("args") or {}, late.get("observation"), late, True)
            self.late_notes.append(f"[the earlier call {action.get('name')} that failed did reach the store; its result] {late_text[:3000]}")
        self.log.append({"kind": "env", "step": self.wire.tick, "sent": [{"name": call_name, "args": call_args}], "reply": resp})
        self.flag = False
        if resp is None:
            if self.wire.done:
                return None, "Not run: the episode is over."
            if self.wire.pending is None:
                return None, ("The store did not take this call (no result came back). Nothing changed: send it again if it "
                              "still makes sense, or your next move.")
            return None, ("The call got no reply (network or server); it stays queued and is sent again, unchanged, before any "
                          "other call. Send it again, or your next move.")
        return resp, self.absorb(call_name, call_args, resp.get("observation"), resp, True)

    def absorb(self, call_name: str, call_args: dict[str, Any], obs: Any, resp: dict[str, Any], last: bool) -> str:
        """Books one answered call and returns the text the actor reads for it."""
        written, obs = obs, _as_read(obs)
        msg_tool, basket_tool = self.role("message"), self.role("cart")
        if call_name == msg_tool:
            self.msgs_sent += 1
        self.last_kind = self.call_kind(call_name, call_args)
        if self.last_kind == "order":
            self.orders_tried += 1
        if call_name in (basket_tool, msg_tool) or call_name in self.finishers():
            self.last_calls = []
        else:
            self.last_calls = (self.last_calls + [call_name + as_text(call_args)])[-2:]
        self.archive.add(obs)
        if (call_name in (self.role("details"), self.role("view")) or call_name in self.page_tools()) and not fault_of(obs, "read"):
            self.archive.read.update(str(val) for val in call_args.values() if isinstance(val, (str, int)) and str(val) in self.archive.ids)
            self.read_at.update({str(val): self.wire.tick for val in call_args.values() if isinstance(val, (str, int))})
            key = self.tools.id_key(call_name)
            if key and isinstance(call_args.get(key), (str, int)) and not isinstance(call_args.get(key), bool):
                self.page_reads[(call_name, str(call_args[key]))] = self.wire.tick
        self.duties.observe(call_name, call_args, obs, self.trolley.version)
        if self.moves_left() < self.order_due() and call_name in (self.role("details"), self.role("search")) and not self.__dict__.get("covering") \
                and not (call_name == self.role("details") and {val for val in call_args.values() if isinstance(val, str)} & set(self.trolley.units)):
            self.late_n = self.__dict__.get("late_n", 0) + 1
        remarks = []
        if call_name == basket_tool:
            self.order_vetted = False
            self.trolley.changed_at = self.wire.tick
            remarks.append(self.trolley.booked(call_args, obs))
            put_item = self.trolley.aim(call_args)
            fault = fault_of(obs, "write")
            accepted = not fault
            self.failed_cart = "" if accepted else call_name + as_text(call_args)
            if put_item and self.trolley.is_put(call_args):
                if accepted:
                    self.dead.pop(put_item, None)
                elif put_item not in self.trolley.units and not ARG_FAULT.search(fault_words(fault)):
                    # (the store will not put this listing in: no pick, forced finish or check counts on it any more)
                    self.dead[put_item] = fault[:120]
                    remarks.append(f"{put_item} could not be put in the cart, so it no longer counts as a pick: choose another "
                                   "verified listing for that product (read its stock and price first).")
            if put_item and self.trolley.is_put(call_args) and accepted:
                self.refusals_seen[put_item] = self.refusal_n
                puts = self.__dict__.setdefault("puts", {})
                puts[put_item] = puts.get(put_item, 0) + 1
                idx = self.line_of.get(put_item, 0 if not self.needs.several() else self.needs.line_for(" ".join(self.archive.about(put_item, 2))))
                if idx is not None:
                    self.line_of.setdefault(put_item, idx)
                    self.trolley.aims[idx] = (put_item, self.needs.line_qty(idx))
            elif put_item and accepted and put_item not in self.trolley.units:
                self.trolley.aims = {i: a for i, a in self.trolley.aims.items() if a[0] != put_item}
            for other in (self.trolley.booked_fresh[1:] if accepted and call_args is self.trolley.booked_args else []):
                idx = self.line_of.get(other, 0 if not self.needs.several() else self.needs.line_for(" ".join(self.archive.about(other, 2))))
                if idx is not None:
                    self.line_of.setdefault(other, idx)
                    self.trolley.aims[idx] = (other, self.needs.line_qty(idx))
            if put_item and self.trolley.is_put(call_args) and accepted and f"{put_item}|{self.needs.version}" in self.passed:
                remarks.append(f"An independent check found {put_item} meets every requirement now in force, reading each property "
                               "by meaning; keep it unless a new change rules it out.")
        if call_name == self.role("cart_view"):
            self.trolley.viewed(obs, self.wire.tick)
            remarks.append(self.trolley.summary())
        refused = self.refused(call_name, obs, resp if last else {"finished": False})
        remarks.append(self.notes_for(call_name, obs, refused))
        if call_name in self.page_tools() or call_name in (self.role("details"), self.role("view"), self.role("compare")):
            remarks.append(self.note_handovers(obs))
        if call_name == self.role("search"):
            remarks.append(self.search_note(call_args, obs))
            self.__dict__.get("named_pages", set()).discard(json.dumps(call_args, sort_keys=True, default=str))
        if (last and not self.__dict__.get("early_audit") and call_name not in (basket_tool, msg_tool) and call_name not in self.finishers()
                and FIRST_ORDER_BY < self.moves_left() <= self.step_cap // 3 + 2 and not self.trolley.units
                and not any(key.rpartition("|")[2] == str(self.needs.version) for key in self.passed)):
            # (nothing verified with a third of the steps left: the decline audit runs now, while its rows can be opened)
            self.early_audit = True
            ahead = self.review_decline(early=True)
            if ahead:
                remarks.append("Nothing verified fits yet. " + ahead)
        qtext = " ".join(str(val) for val in call_args.values() if isinstance(val, str)) if call_name == msg_tool else None
        caught = self.shopper_lines(call_name, obs, refused)
        if caught:
            cap_before = self.needs.spend_cap
            remarks.append(self.needs.hear(caught, qtext))
            if is_number(self.needs.spend_cap) and self.needs.spend_cap != cap_before:
                # (a budget notice: the cart's lines are priced again at units x newest unit price against the new budget)
                money = self.budget_line()
                over = self.over_lines()
                remarks.append((money or f"[budget] now {self.needs.spend_cap:g}.")
                               + (" The cart no longer fits: replace a line with a cheaper verified listing whose unit price is within "
                                  "the figure shown (the code swaps one itself when a checked one fits); the units stay as requested."
                                  if over else ""))
                self.log.append({"kind": "budget_notice", "before": cap_before, "now": self.needs.spend_cap, "note": money, "over": over})
        self.flag = bool(refused or (caught and call_name != msg_tool))
        if call_name == basket_tool and not fault_of(obs, "write") and not self.fixing:
            extra = [f for f in self.trolley.faults(self.needs, self.line_of) if not f.startswith("no listing in the cart yet")]
            if extra and len(self.trolley.units) > len(self.needs.lines):
                # (the bag the store returned holds more listings than the errand has products: said at once)
                remarks.append("[cart check] " + "; ".join(extra) + ".")
        text_body = as_text(self.actor_view(call_name, written), VIEW_CHARS)
        if last and resp.get("finished"):
            text_body += " [the episode is finished]"
        summary = " ".join(qty_n for qty_n in remarks if qty_n)
        return text_body + ("\n[note] " + summary if summary else "")

    def actor_view(self, call_name: str, obs: Any) -> Any:
        """What the actor reads of a result: a search's (or any unmapped browse tool's) rows with their long free text
        cut to ROW_TEXT; a listing's own record, a comparison, the cart, a shopper reply and a finish stay whole. The
        records keep every result whole, so the checks still read it all."""
        whole = {self.role(r) for r in ("details", "view", "compare", "cart", "cart_view", "message", "order", "decline",
                                        "hold", "report")} - {None}
        return obs if call_name in whole else rows_trimmed(_short_rows(obs, ROW_TEXT), ROW_TEXT)

    # ----------------------------------------------------------------------------------------------- procedure steps
    def fill_required(self, tool: str, call_args: dict[str, Any], cue: Any, words: str) -> dict[str, Any]:
        """The other required arguments of a duty call: an enum value the cue reads in (else the first), true for a
        flag, the units for a count, the given words for free text."""
        props = self.tools.props(tool)
        for key in self.tools.required(tool):
            pspec = props.get(key) if isinstance(props.get(key), dict) else {}
            if key in call_args:
                continue
            if isinstance(pspec.get("enum"), list) and pspec["enum"]:
                call_args[key] = next((o for o in pspec["enum"] if isinstance(o, str) and cue is not None and cue.search(spoken(o))), pspec["enum"][0])
            elif "boolean" in kinds(pspec):
                call_args[key] = True
            elif set(kinds(pspec)) & {"integer", "number"}:
                call_args[key] = max(1, int(pspec.get("minimum", 1)) if is_number(pspec.get("minimum")) else 1)
            elif "string" in (kinds(pspec) or ["string"]):
                call_args[key] = words[: int(pspec["maxLength"])] if is_number(pspec.get("maxLength")) else words
        return call_args

    def title_of(self, ident: str) -> str:
        found: list[str] = []

        def visit(d: dict[str, Any]) -> None:
            if not any(str(v) == ident for v in d.values() if isinstance(v, (str, int))):
                return
            for key, val in d.items():
                if isinstance(val, str) and 0 < len(val) <= 200 and re.search(r"title|name|heading|headline|label", spoken(key)) \
                        and not re.search(r"brand|vendor|maker|categor|shelf|\bid\b", spoken(key)):
                    found.append(val)
                    return
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return re.sub(r"\s+", " ", found[-1]).strip()[:60] if found else ""

    def gone(self, ident: str) -> bool:
        """The newest record of a listing shows it out of stock."""
        state: list[bool] = []

        def visit(d: dict[str, Any]) -> None:
            if any(str(v) == ident for k, v in d.items() if isinstance(v, (str, int)) and "id" in str(k).lower()):
                state.append(self.sold_out(d))
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return bool(state) and state[-1]

    # ----------------------------------------------------------------------------------------------- sold-out lines
    def line_sold(self, ident: str) -> bool:
        """A refusal named the listing sold out, or its newest record shows it out of stock."""
        return ident in self.sold_lines or self.gone(ident)

    def mark_sold(self, obs: Any) -> None:
        """A refused order whose text says a listing is out of stock: the cart lines it names (by id or by the start of
        the title), or the only line, are booked sold out."""
        text_body = as_text(obs, 8000).lower()
        if not re.search(r"out of stock|sold out|no longer (?:available|in stock|sold|stocked)|unavailable|not available|"
                         r"none left|no stock|discontinued|cannot be bought|can no longer be bought", text_body):
            return
        named = [t for t in self.trolley.units if t.lower() in text_body
                 or (len(self.title_of(t)) >= 12 and self.title_of(t).lower()[:24] in text_body)]
        if not named and len(self.trolley.units) == 1:
            named = list(self.trolley.units)
        if named:
            self.sold_lines.update(named)
            self.log.append({"kind": "sold_line", "items": named})

    def swap_pick(self, idx: int, thing: str) -> str:
        """The listing that replaces a sold-out cart line: the cheapest in-stock listing for that product, within the
        budget with the other lines, that the item check passes under the requirements in force; listings passed earlier
        come first, and those not yet passed now are checked again (two at most)."""
        cap, qty = self.needs.spend_cap, self.needs.line_qty(idx)
        rest = sum((self.archive.price(t) or 0.0) * u for t, u in self.trolley.units.items() if t != thing)
        once = {k.rpartition("|")[0] for k in self.passed}
        pool = []
        for ident in once | set(self.archive.read):
            if ident == thing or ident in self.dead or ident in self.refused_lines or ident in self.trolley.units or self.line_sold(ident):
                continue
            if (self.line_of.get(ident) if self.needs.several() else 0) != idx:
                continue
            cost = self.archive.price(ident)
            if not is_number(cost) or (is_number(cap) and cost * qty + rest > cap + 1e-9):
                continue
            pool.append((ident not in once, cost, ident))
        checks = 0
        for _, _, ident in sorted(pool):
            if not self.verified(ident):
                if checks >= 2 or self.wire.spare() < CLOSING + 20:
                    break
                checks += 1
                self.check_item(ident, "ORDER")
            if self.verified(ident):
                return ident
        return ""

    def swap_step(self) -> bool:
        """Replaces a sold-out cart line by code: at most SWAPS_MOST per task, with 4 or more steps left and time to spare.
        The line comes out, the swap_pick goes in at its units, then auto_step re-reads it and sends the order (4 steps).
        The other lines of a cart refused for that sell-out count again, checked under the requirements in force."""
        if self.wire.done or self.wire.broke or self.fix_queue or not self.role("cart") or not self.role("order") \
                or not (self.trolley.id_key or self.trolley.lines_key):
            return False
        if self.__dict__.get("swaps", 0) >= SWAPS_MOST or self.moves_left() < 4 or self.wire.spare() <= CLOSING:
            return False
        sold = [t for t in self.trolley.units if self.line_sold(t)]
        # (a cart whose total the budget in force no longer covers: its dearest lines, two at most, are tried in turn)
        over = [] if sold else self.over_lines()[:2]
        cap = self.needs.spend_cap
        for thing in sold[:1] or over:
            mark = f"swap|{thing}|{self.trolley.version}|{self.needs.version}|{cap}"
            if mark in self.warned:
                continue
            self.warned.add(mark)
            idx = self.line_of.get(thing, 0 if not self.needs.several() else None)
            if idx is None:
                self.log.append({"kind": "swap", "line": thing, "pick": "", "why": "line unknown"})
                continue
            for other in list(self.trolley.units):
                if other != thing and (thing in self.sold_lines or not sold):
                    self.refused_lines.discard(other)
                if other != thing and not self.verified(other) and self.wire.spare() > CLOSING + 20:
                    self.check_item(other, "ORDER")
            sub = self.swap_pick(idx, thing)
            if self.trolley.lines_key:
                take, put = {}, self.trolley.swap_args(thing, sub, self.needs.line_qty(idx)) if sub else None
            else:
                take = self.trolley.take_args(thing, self.trolley.held(thing) or 1)
                put = self.trolley.put_args(sub, self.needs.line_qty(idx)) if sub else None
            order = self.order_move()
            self.log.append({"kind": "swap", "line": thing, "why": "sold out" if sold else "over budget", "pick": sub, "left": self.moves_left()})
            if not sub or take is None or put is None or order is None:
                continue
            self.swaps = self.__dict__.get("swaps", 0) + 1
            self.fix_queue = [c for c in (take, put) if c]
            self.auto_order = {"call": (order["name"], order["args"]), "version": self.needs.version, "reads": 0, "sends": 0, "kind": "swap"}
            why = "is sold out" if sold else f"takes the cart over the budget of {cap:g} ({self.budget_line()})"
            self.chat.append({"role": "user", "content": f"[swap] {thing} {why}. The code takes it out of the cart, puts in {sub} (the "
                              "cheapest listing for that product the independent check passes, in stock and within the budget with "
                              "the other lines), re-reads it and orders.\n" + self.status_line()})
            return self.auto_step()
        return False

    def finalists(self, most: int) -> list[str]:
        """Listings worth comparing: the cart's, then the verified ones, then the other opened ones within the
        budget and in stock (cheapest first under a cheapest rule); with several products, those of the first
        cart listing's product."""
        cap, cheap = self.needs.spend_cap, self.needs.priority() == "cheapest"
        head = self.picks()
        line = self.line_of.get(head[0]) if head and self.needs.several() else None
        good = [k.rpartition("|")[0] for k in self.passed if self.verified(k.rpartition("|")[0])]
        rest = [i for i in sorted(self.archive.read) if i not in self.refused_lines]

        def fine(ident: str) -> bool:
            cost = self.archive.price(ident)
            if is_number(cap) and is_number(cost) and cost * self.needs.line_qty(self.line_of.get(ident)) > cap:
                return False
            if line is not None and self.line_of.get(ident) not in (None, line):
                return False
            return not self.gone(ident)
        if cheap:
            rest.sort(key=lambda i: (self.archive.price(i) if is_number(self.archive.price(i)) else 1e18, i))
        ranked = [i for i in dict.fromkeys(head + good + rest) if i in head or fine(i)]
        return ranked[:max(0, most)]

    def compared_text(self) -> str:
        """'Report: I compared A, B and C side by side. ' for the newest comparison sent ('' when none was)."""
        tool = self.role("compare")
        for entry in reversed(self.log):
            sent = [a for a in entry.get("sent") or [] if a.get("name") == tool] if entry.get("kind") == "env" and tool else []
            ids = [str(i) for a in sent for v in (a.get("args") or {}).values() if isinstance(v, list) for i in v][:5]
            if ids:
                return "Report: I compared " + ", ".join(f"{(self.title_of(i) or 'listing')[:40]} ({i})" for i in ids) + " side by side. "
        return ""

    def basket_text(self, ask: bool) -> str:
        """A short basket summary built only from tool results: each line, its units and price, the total, the budget."""
        parts, total, priced = [], 0.0, True
        cur = str(self.needs.reading.get("currency") or "").strip()
        for thing, units in self.trolley.units.items():
            title, cost = self.title_of(thing), self.archive.price(thing)
            label = f"{int(units)} x {title or 'listing'} ({thing})"
            if is_number(cost):
                total += cost * units
                label += f" at {cost:g}{(' ' + cur) if cur else ''} each"
            else:
                priced = False
            parts.append(label)
        text_body = ("Before I place the order, here is your basket: " if ask else "Your basket before I order: ") + "; ".join(parts)
        if priced:
            text_body += f". Total {total:g}{(' ' + cur) if cur else ''}"
            if is_number(self.needs.spend_cap):
                text_body += f", within your budget of {self.needs.spend_cap:g}" if total <= self.needs.spend_cap else f" (budget {self.needs.spend_cap:g})"
        return text_body + (". Shall I go ahead?" if ask else ". I will place this order now.")

    def duty_call(self, duty: Duty) -> tuple[dict[str, Any], str] | None:
        """(proposal, why) that carries out one open step, or None when it cannot be done now."""
        cid = f"duty_{duty.kind}_{self.wire.tick}_{duty.tries}"
        if duty.kind == "survey":
            short = (duty.count or 1) - len(self.duties.opened)
            pool = [i for i in self.named_unread() + sorted(self.archive.ids - self.archive.read)
                    if i not in self.refused_lines and i not in self.survey_tried]
            pool = [i for i in dict.fromkeys(pool) if not self.gone(i)]
            if short <= 0 or not pool or not self.role("details"):
                return None
            call_args = self.details_call(pool[0])
            if not call_args:
                return None
            self.survey_tried.add(pool[0])
            return ({"id": cid, "type": "function", "function": {"name": self.role("details"), "arguments": json.dumps(call_args)}},
                    f"The request asks to look at {duty.count} listings; I open {pool[0]}.")
        if duty.kind == "compare":
            shape = self.tools.compare_shape()
            if not shape:
                return None
            _, key, low, high = shape
            # (the pick first, then the finalists, up to the schema's limit or 5; after a failed table, 3, then 2)
            roof = high or 5
            if isinstance(duty.mark, int):
                roof = min(roof, 3 if duty.mark > 3 else 2)
            ids = self.finalists(roof if duty.stated else min(roof, 3))
            if len(ids) < max(2, low or 0):
                return None
            call_args = {key: ids} if shape[0] == "list" else dict(zip(key, ids))
            duty.mark = len(ids)
            call_args = self.fill_required(self.role("compare"), call_args, COMPARE_CUE, "compare")
            return ({"id": cid, "type": "function", "function": {"name": self.role("compare"), "arguments": json.dumps(call_args)}},
                    f"I put {len(ids)} finalists side by side with the comparing tool.")
        if duty.kind == "hold":
            todo = [t for t in self.picks() if t not in self.duties.held]
            return self.hold_call(todo[0], cid) if todo else None
        if duty.kind == "report":
            tool = self.role("report")
            call_args = self.fill_required(tool, {}, REPORT_CUE, "account report")
            return ({"id": cid, "type": "function", "function": {"name": tool, "arguments": json.dumps(call_args)}},
                    "The request asks for the account report; I pull it up.")
        if duty.kind in ("confirm", "tell_basket"):
            if not self.trolley.units or (self.msg_cap is not None and self.msgs_sent >= self.msg_cap) or self.moves_left() <= 3:
                return None
            text_body = self.basket_text(duty.kind == "confirm" or duty.stated)
            if re.search(r"compar|side by side|line up|report", duty.text.lower()):
                text_body = self.compared_text() + text_body
            proposal = self.message_call(text_body, cid)
            return (proposal, "Before the order I tell the shopper what the basket holds.") if proposal else None
        return None

    def hold_call(self, thing: str, cid: str) -> tuple[dict[str, Any], str] | None:
        tool = self.role("hold")
        key = self.tools.id_key(tool)
        if not tool or not key:
            return None
        call_args = {key: thing}
        count = self.tools.count_key(tool)
        if count:
            call_args[count] = self.needs.line_qty(self.line_of.get(thing))
        call_args = self.fill_required(tool, call_args, HOLD_CUE, "hold for the shopper")
        return ({"id": cid, "type": "function", "function": {"name": tool, "arguments": json.dumps(call_args)}},
                f"The request asks to hold the pick before buying it; I hold {thing}.")

    def run_duty(self, made: tuple[dict[str, Any], str], duty: Duty | None) -> bool:
        proposal, why = made
        sent_before = self.msgs_sent
        self.chat.append({"role": "assistant", "content": why, "tool_calls": [proposal]})
        self.covering = True
        try:
            responses = self.dispatch([proposal], scripted=True)
        finally:
            self.covering = False
        if duty is not None and duty.kind in ("confirm", "tell_basket") and self.msgs_sent > sent_before:
            self.duties.told = self.quote()
        self.post_replies(responses)
        self.duties.refresh(self.picks(), self.trolley.version, self.quote())
        if duty is not None:
            self.duties.event(f"duty {duty.kind} ran ({proposal['function']['name']}) -> {duty.state}")
        return True

    def duty_step(self) -> bool:
        """Carries out one procedure step by code before the actor's turn: a hold a cart call is waiting for, a step
        a held finish is waiting for, or a comparison once finalists exist. Only read, compare, hold, report and
        message tools are called here; the cart, the order and the decline stay with the actor."""
        if self.wire.done or self.wire.broke or self.moves_left() <= 2 or self.wire.spare() < CLOSING:
            self.hold_queue, self.duty_due, self.rule_queue = [], None, []
            return False
        while self.hold_queue:
            thing = self.hold_queue.pop(0)
            made = self.hold_call(thing, f"duty_hold_{self.wire.tick}")
            if made:
                return self.run_duty(made, next((d for d in self.duties.items if d.kind == "hold"), None))
        if self.rule_queue:
            # (the step a rule audit found missing: one call, through the usual screen; never a finish or a cart change)
            made = self.rule_queue.pop(0)
            if made[0]["function"]["name"] not in self.finishers() | {self.role("cart")} and self.moves_left() >= 3:
                self.rule_steps += 1
                self.rule_msgs += 1 if made[0]["function"]["name"] == self.role("message") else 0
                return self.run_duty(made, None)
        self.duties.refresh(self.picks(), self.trolley.version, self.quote())
        stage = self.duty_due
        if stage is None:
            duty = next((d for d in self.duties.items if d.kind == "compare" and d.state == "open" and d.tries == 0), None)
            ready = duty is not None and (self.picks() or any(self.verified(k.rpartition("|")[0]) for k in self.passed)) \
                and self.moves_left() - 1 >= (4 if duty.stated else 6) and not self.in_reserve() \
                and len(self.finalists(5)) >= (max(2, duty.count or 2) if duty.stated else 2)
            if ready:
                made = self.duty_call(duty)
                if made:
                    duty.tries += 1
                    return self.run_duty(made, duty)
            return False
        steps_open = self.gate_duties(stage)
        for duty in steps_open:
            if duty.tries >= 2 or (not duty.stated and self.moves_left() <= 5):
                continue
            if duty.kind == "survey" and self.moves_left() <= len(steps_open) + 3:
                self.duties.failed(duty, "no steps to spare")
                continue
            made = self.duty_call(duty)
            if made is None:
                self.duties.failed(duty, "could not be done with what the store showed")
                continue
            if duty.kind != "survey":
                duty.tries += 1
            return self.run_duty(made, duty)
        self.duty_due = None
        return False

    # ----------------------------------------------------------------------------------------------- loop
    def tool_calls(self, msg: dict[str, Any], seq_no: int) -> tuple[list[dict[str, Any]], int]:
        proposals = []
        for proposal in msg.get("tool_calls") or []:
            if not isinstance(proposal, dict):
                continue
            seq_no += 1
            func = proposal.get("function") or {}
            arguments = func.get("arguments") if isinstance(func.get("arguments"), str) else json.dumps(func.get("arguments") or {})
            proposals.append({"id": proposal.get("id") or f"call_{seq_no}", "type": "function", "function": {"name": func.get("name"), "arguments": arguments}})
        return proposals, seq_no

    def execute(self) -> list[dict[str, Any]]:
        if self.wire.done:
            return self.log
        self.read_request()
        self.log.append({"kind": "plan", "plan": self.needs.reading, "roles": self.role_map})
        opener_obs = self.pview.get("opening")
        opener = "SHOPPER REQUEST:\n" + self.request
        if opener_obs is not None:
            opener += "\n\nOPENING OBSERVATION:\n" + as_text(opener_obs, VIEW_CHARS)
            self.archive.add(_as_read(opener_obs))
        overview = self.needs.overview(self.step_cap, self.msg_cap)
        if overview:
            opener += "\n\n" + overview
        opener += "\n\n" + self.status_line()
        self.chat = [{"role": "system", "content": ACT_RULES}, {"role": "user", "content": opener}]
        idle_n = fails = seq_no = 0
        for _ in range(TURN_LIMIT):
            if self.wire.done or self.wire.broke or self.moves_left() <= 0:
                break
            # (the code finishes by itself: little time or money left, no model to ask, repeated step conflicts, or
            # the last steps with something to order: two with a filled cart, three with an empty one - a put that
            # may be turned down, a second put or a decline, the order). Money runs out when no actor model's next
            # call, priced at its dearest one so far, fits under SPEND_STOP.
            if self.wire.spare() < LAST_CALL or self.wire.credit_out or self.wire.conflicts >= 2 \
                    or not any(self.wire.affordable(m) for m in ACTOR if m not in self.wire.barred):
                break
            if self.moves_left() <= (2 if self.trolley.units else 3) and (self.trolley.units or self.verified_picks()):
                break
            if self.needs.reply_due:
                text_body, self.needs.reply_due = self.needs.reply_due, ""
                if self.reply_back(text_body):
                    continue
            if self.reference_step():
                # (a need the shopper gave by pointing at another listing is looked up before anything is bought)
                continue
            if self.handover_step():
                continue
            if self.swap_step():
                continue
            if self.duty_step():
                continue
            if self.page_step():
                continue
            if self.auto_step():
                continue
            if self.open_named():
                continue
            if not self.asking and self.sweep_step():
                continue
            if self.recall_step():
                # (a decline hold's pass runs whatever the asking state; the mid-errand one only once questions are done)
                continue
            if self.asking and self.pose_planned(urge=self.stuck >= 3):
                # (three replies in a row sent nothing: the planned question goes out now, or the message tool is freed)
                continue
            model_list = ACTOR if self.stuck < STUCK_LIMIT else ACTOR[1:] + ACTOR[:1]
            msg = self.wire.query_model(self.chat, model_list, 2500, self.offered(), actor=True)
            if msg is None:
                fails += 1
                if fails >= FAILS_LIMIT:
                    break
                continue
            fails = 0
            proposals, seq_no = self.tool_calls(msg, seq_no)
            content = msg.get("content") or ""
            if not isinstance(content, str):
                content = as_text(content)
            binding_moves = {self.role("cart"), self.role("order"), self.role("decline")} - {None}
            finishing = {self.role("order"), self.role("decline")} - {None}
            if self.asking and any(c["function"]["name"] in binding_moves for c in proposals) \
                    and self.pose_planned(urge=True, pace=not any(c["function"]["name"] in finishing for c in proposals)):
                continue
            if self.asking and any(c["function"]["name"] == self.role("message") for c in proposals) and len(proposals) == 1:
                # (the actor wants to ask now: the planned question goes out now, or, with none planned, the actor asks)
                if self.pose_planned():
                    continue
            if not proposals:
                idle_n += 1
                self.chat.append({"role": "assistant", "content": content or "(no tool call)"})
                self.chat.append({"role": "user", "content": "Every reply must call at least one of the supplied tools. Continue.\n" + self.status_line()})
                if idle_n >= IDLE_LIMIT:
                    break
                continue
            idle_n = 0
            self.log.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in proposals]})
            self.chat.append({"role": "assistant", "content": content or None, "tool_calls": proposals})
            step_before = self.wire.tick
            self.post_replies(self.dispatch(proposals))
            self.stuck = self.stuck + 1 if self.wire.tick == step_before else 0
        if not self.wire.done and not self.wire.broke:
            self.finish_now()
        self.log.append({"kind": "end", "finished": self.wire.done, "steps": self.wire.tick, "model_calls": self.wire.calls,
                         "cost": round(self.wire.spend, 5), "seconds": round(DEADLINE - self.wire.spare(), 1),
                         "duties": [d.brief() + " -> " + d.state + (" " + d.why if d.why else "") for d in self.duties.items]})
        return self.log


def agent_main(pdata: dict[str, Any]) -> list[dict[str, Any]]:
    errand = Errand(pdata)
    try:
        return errand.execute()
    except Exception as err:
        errand.log.append({"kind": "crash", "error": f"{type(err).__name__}: {err}"[:500]})
        try:
            if not errand.wire.done and not errand.wire.broke:
                errand.finish_now()
        except Exception as second_err:
            errand.log.append({"kind": "crash", "error": f"{type(second_err).__name__}: {second_err}"[:500]})
        return errand.log


__all__ = ["agent_main"]
