"""Kraus C: a shopping agent for one ORO store episode.
Reads request and tool schemas once; maps tools to roles
(search, details, message(+status), cart(+view), order,
decline); requirement ledger grows per revealed need.
Hidden-need questions: one message at a time, each product
before any twice; two properties share a message only while
another is left, until one is answered, never after a
one-at-a-time reply. Basket shown pre-order only if a request
rule asks. Text-line listings read as records. A light model
drives tools; code checks cart changes, finish vs stock,
budget, records; finishes itself if time/steps run short.
Entry point: agent_main(problem_data) -> episode log.
"""
from __future__ import annotations

import json
import re
import weakref
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

ACTOR = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-haiku-5.5"]
READER = ["anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"]
CHECKER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
AUDITOR = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
LISTENER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
PLANNER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]

DEADLINE = 265.0
CLOSING = 70.0
LAST_CALL = 28.0
CLOCK_STEP = 30
CLOCK_FROM = 45.0
CH_SYNONYM_MIN = 4
VIEW_CHARS = 16000
SEARCH_TEXT = 240
RECORD_TEXT = 700
SOFT_SPEND = 0.03
SPEND_CAP = 0.045
TURN_LIMIT = 80
PACE_GAP = 0.15
SLOW_GAP = 1.0
SLOW_FOR = 30.0
DOWN_AFTER = 45.0
MODEL_TRIES = 2
WORLD_TRIES = 5
CODE_FINISHES = 4
CODE_PAGES = 3
CODE_READS = 8
JOINERS = r"\s+and\s+|\s*&\s*|\s*/\s*|\s+\+\s+"
PUT_WORDS = {"add", "put", "hold", "set", "insert", "place", "keep", "include", "in", "plus", "increase", "append"}
DROP_WORDS = {"remove", "drop", "release", "delete", "take", "out", "clear", "discard", "minus", "decrease", "subtract", "off", "unhold"}
FAULT_KEYS = r"error|fault|refus|reject|denied|invalid|fail"
ASK_DEFAULT = 3
ASK_OPEN = 5
PAIR_LAST = False
PAIR_FROM = 2             # question messages from which a message may ask two properties (never an unproven pair last)
MENU_READS_WIDE = 80
FIRST_ORDER_BY = 7
FIRST_ORDER_DUE = 10
IDLE_LIMIT = 3
STUCK_LIMIT = 4
SWEEP_READS = 8
WALK_PAGES = 9
WALK_TOTAL = 12
VALUE_PAGES = 4
VALUE_TOTAL = 8
VALUE_TOTAL_PER = 5
LABEL_TRIES = 4
LABEL_OPENS = 3
CEIL_CALLS = 4
CEIL_OPENS = 3
LINK_WORDS = {"and", "or", "the", "for", "with", "of", "a", "an", "in", "on", "to", "by", "from", "at", "as", "is", "are", "be",
              "this", "that", "it", "its", "your", "our", "you", "all", "any", "not", "no"}
ASK_AFTER_SETS = 2
ASK_GRACE_STEPS = 3
ASK_AFTER_STEPS = 5
TIE_SEARCHES = 3
TIE_OPENS = 4
TOP_COVER = 3
FIT_PAGES = 12
FIT_TOTAL = 30
FIT_OPENS = 3

ACT_RULES = """You are a careful shopping assistant working for one shopper in a simulated store; the supplied tools are your only way to act. The errand scores 1 or 0, and it scores 1 only when the finish is right in every respect: the ordered item meets every requirement the shopper wrote, every hidden need the shopper reveals when asked, and every change the shopper announces along the way; the quantity and the budget are right; the cheapest (or best) item is chosen when the request asks for that; and every rule of procedure in the request is obeyed. When nothing in the store meets everything, the right finish is a decline, done exactly as the request says.

HOW THE STORE WORKS
- Each tool call costs one step, failed and invalid calls included, and steps are limited. Thinking costs nothing.
- One reply may carry several tool calls when none needs another's result (for example opening several listings' details); they run in order. An order or a decline always goes alone.
- A details tool that takes a LIST of listing ids shows all their full records for ONE step (up to its item limit): when you open two or more listings, use it.
- Tool names, argument names and result shapes are different in every store. Read each tool's description and JSON schema and use exactly the names and allowed values given. Learn from the first results which tool shows which facts (price, stock, detailed properties).
- Early on, learn which category labels the store puts on listings of the requested kind. When neighbouring labels exist, the right one is the label that matches the request's wording as a whole; search and buy inside it.
- Searches differ from store to store; learn this one's from its first results. Rows may come best match first or in a fixed id order; a keyword search may need every word (adding a word shrinks the total) or take any word (adding a word grows the total, and listings matching more of the words rank higher); only a few rows or pages may be visible; a filter may work in one mode only (a keyword mode may ignore the category or price filter, a filter mode may ignore the words). Compare a reported total with the rows you can see: when it is larger, the fitting or cheapest listing may sit on a later page or beyond the visible ones. When every word is required, narrow with more exact words (or the category and price filters with a lower price limit) until the total fits on the visible pages. Keep queries short (one to three words) with the most distinctive words FIRST: some stores read only the first word or two of a query and ignore the rest, so a long query may search only its opening words (compare a longer query's total and rows with a shorter one that starts the same way; the notes report what they show). When every word is required, a required value with the item kind ("blue chair", the value first) lists exactly the listings that carry it. When any word matches, common words (the item kind, a unit) match most of the shelf and bury the rest: a required value's own distinctive word or figure alone ("herringbone", "30m", "0.4") lists just the listings that carry it, so search it alone and in the listings' other spellings ("30 m", "3 atm"), read every page of such a search when its total fits on them, and put the item kind with the values together only to rank the closest matches first. A keyword search may also match category names and property values. Notes after each search say what the totals so far show about this store's search.

THE SHOPPER AND HIDDEN NEEDS
- The shopper holds needs they did not write down and reveals one only when a message asks about that property. The FIRST thing a message asks is answered most reliably, so a message asks about one property, or two at most with the likelier first: a short, plain, friendly question that names the property, with no example values. No instructions, no lists, labels, positions or rules.
- "Don't mind", "no preference", "anything is fine" means the property is free. A stated preference or limit is a hard requirement from then on, exactly as worded, numbers included.
- When questions are planned for you, they go out once you have opened the details of two or three listings of the requested kind, so open those in your first moves. The answers arrive in the conversation with their meaning; apply each revealed need as a hard requirement and search again with it and the stated values together.
- If you ask yourself: first open two or three listings' details, then ask about what the story hints at (who it is for, how it must look, where or how it is used), then about properties the listings carry that differ among candidates. Never ask what the request already settles and never go over the message budget.

RULES OF PROCEDURE
- The request may set rules for how to work: how many listings to look at or compare, what to check, what to tell the shopper, what a remark must say, how to decline. They are graded like requirements; follow each one literally and fully. "Look at" or "open" N listings means N fully fitting listings, each opened with the details tool.
- When the shopper asks you something (for example to confirm a change), it is answered for you while messages remain; the change itself is binding.
- What the shopper would merely like ("ideally", "would be nice", "if possible") is not a requirement: it only breaks ties between items that already meet everything, and never beats a cheapest rule.
- When the request leaves something only the shopper can settle (for example no price limit, or two wishes that may not both be met and the shopper must say which matters more), ask before you choose.

CHANGES WHILE YOU WORK
- The shopper may change things: a new budget (higher or lower), a requirement dropped, replaced or added, a price rise, an item selling out. One line can carry several changes. Such lines can arrive inside any tool result (an extra field holding sentences) or with an order attempt. Each change binds from that moment: re-check the pick and the backups, and when the cheapest or best is asked for, choose again over the new acceptable set.
- An order attempt can be REFUSED: the episode goes on and the cart is as it was. Read the refusal, apply every change it states, take out of the cart whatever no longer fits (for example a sold-out item), put in the next verified fit, re-check its stock and price, and order again.
- Items can also sell out or change price without notice, so stock and price are re-checked right before every order attempt; an item whose current price times quantity is over the current budget no longer fits.

SEVERAL PRODUCTS
- A request may ask for several different products, each with its own quantity, requirements and hidden needs. Work on every product: search each one's category, verify a fit for each, and order them together, one line per product at its quantity. A cart tool that takes a list of lines may replace the whole cart with that list, so send every product's line in each such call. The cheapest or best rule applies within each product.
- A requirement may give its value by pointing at another listing or page ("the same colour as the jumpsuit sold by X"): find that listing first (search the words that name it, such as its shop or brand with its kind), read its record, take the value from the named field, then search the requested product with that value.

THE CART
- The cart tool may ADD units to a line that is already in the cart rather than set them: putting 1 more of a held line makes 2. Never put a line in again that the cart already holds; to change units, take the difference out or in. A note after each cart call says what the code counts in the cart.

HOW TO JUDGE A LISTING
- Read each requirement from the listing's field for that property, by the value's own words as this store writes them in that field. A field whose name says the same property counts whatever its spelling: separators, a prefix, a shorter or longer name, or a usual alternative name ("fit" for a pants fit type, "neckline" for a collar type, "dial size" for a watch dial size, "color family" for a colour). When a field carrying the property's own name exists, it decides, and so does a field of that name with one more leading word (pants length for length, clothing material for material): when either one writes the value, the requirement is met. Two fields whose names differ only by separators (size class, size_class) are one field: the requirement is met when either one writes the value. Otherwise such an alternative field decides; when no field names the property at all, the options list or the title may show the value. The description is never evidence, with one exception: a model name or number required for the item itself is met wherever its record writes it (title, fields, overview or notes), spaces, hyphens and case aside.
- A product that must work with, fit or be listed for ANOTHER product of the request (an accessory for a main item) fits only when its own record names the main item's model name or number, or another name the main item's record says the same item is sold under; TIE EVIDENCE lines, when given, are computed from the records and decide it. Such a tie is read from the records, never asked about.
- Values match by their words, as this store writes them: case, separators, singular and plural and spelling variants are the same; a more specific value that contains the required one meets it ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060); one element of a list is enough. A synonym, an abbreviation, a related word or a shade ("tech wood" for technical wood, "win11" for windows 11, "embroidered" for embroidery, "solid" for plain, ivory or cream for off-white) meets a value only when no listing of the category writes the value itself in that field: when other listings write "matte" as their finish, a "satin" finish does not meet matte. The initials of the value's words are the value itself, whatever other listings write ("sff" for small form factor, "ssd" for solid state drive). A value joined with another by a plus ("wool + nylon") does not meet one of its parts when the field also carries the other part alone ("nylon"). FIELD EVIDENCE lines, when given, are computed from this store's records and say which case holds. A different value of the property does not (quartz is not mechanical, rosewood is not maple), nor does a word that only looks alike. A value that claims to cover every value of the property ("suitable for all occasions", "all seasons", "universal") names none of them and does not meet a required value; a list that also names the required value still meets it. A field that names another value of the property does not meet the required value just because the two often go together or one tends to imply the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); only a shade usually sold as the required colour, or a more specific value that contains it, counts.
- A field of a different property is not the requirement's property (a lining material is not the material, a length is not a style, a strap length is not a strap type): it never stands in for a property whose own field the category's listings carry.
- Only what the shopper stated or revealed counts. Never rule out a listing for something the shopper did not ask for: an age group, gender, size, occasion, condition, quality or use you guess from the story or the description. The story is context; if it seems to matter, ask the shopper about that property instead of assuming. Your own questions are not requirements: only the shopper's answers are.
- A requirement that the item fit, work with or suit a model (a vehicle, device or machine model) is met by a sentence of its record, the description and notes included, that names that model or a model family covering it ("every ABC-47xx model" covers ABC-4710: each x stands for one character), unless an "except" or "excluding" after it names that very model; a sentence about another part ("Part X also fits: ...") says nothing about this item. FIT EVIDENCE lines, when given, are computed by code and settle it.
- Numbers are compared as amounts after converting units (4.5 cm is 45 mm); limits include the edge; a range counts only when all of it meets the limit. When no field names the property and the title is the evidence, an exact required value is met when the title shows its words and figures in the same order, even with more detail after them (9' x 12' 4" shows 9' x 12'; 9' x 13' does not).
- Figures in a title ("55 x 94 Inch", "117 x 228 cm") do not say which dimension each one is: never rule a listing out, or skip opening it, for the figures in its title; read the fields of its details.
- Respect exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
- The requested kind of product is met by the listing's category alone: a listing on the requested category IS that kind even when its title or description names another object (a container, a case, a refill, a set, an accessory), so never skip, reject or demote it for what it seems to be; judge it only on the requirements, price and stock. No property has to repeat the category name. When the store labels categories, the item must sit in the one whose label matches the request's wording as a whole (a request for "A or B" matches a category "A & B", not a narrower neighbour "B"). It must be in stock when the request wants things on hand, and price times quantity must be within the budget (equal is fine). A listing's price is in the budget's unit unless its record names another (45 against a budget of 1000 US cents is 45 cents); never rescale either.
- A decline is rarely the right finish. When no listing meets the strict wording but one meets each requirement under these rules (a more specific value, another name of the same field, or a synonym or shade where no listing of the category writes the value itself in that field), that listing is the fit; decline only when the search really found nothing that meets every requirement under these rules.

CHEAPEST OR BEST
- For the cheapest, the order must be the lowest-priced item among ALL items meeting every requirement; one fitting item is not enough. When the keyword search needs every word and matches property values, the value-and-kind search lists the candidates that carry the value: read all its pages and open them by price; a category listing with a price limit shows the whole category's prices but not its values: use it to see which cheaper rows exist (lower the limit until the total fits), then check those rows' properties. Search each required value with the item kind as a short query, the value first ("blue chair", "30m watch"), and read every page of it when its total fits on the readable pages (when it does not, add a second required value or use a narrower kind word); do the same with the kind's other names ("stool", "bench") and with the value alone; when a value cannot be searched (a figure the store drops), also search the product line of each fit or near-fit you find, its maker with one distinctive word of its title as two words, every page: a line's other sizes and colours are separate listings that share its property values and may cost less; sort the visible candidates by price yourself and open their details from the cheapest up; the first whose property fields meet everything, with every cheaper one opened and rejected, is the answer. Keep the next fitting ones as backups.
- For the best by some measure (most, largest, longest, highest...), compare ALL fitting items on that measure the same way and take the extreme. An item stating no value for the measure cannot win.

FINISHING
- Keep a ranked list of verified fits with at least one backup: the first pick may sell out at checkout.
- THE BASKET STATEMENT: right before every order the code itself sends the shopper one message stating the basket (each line with its quantity and price, the total) and asking to go ahead; the shopper's acknowledgement is consent, so never wait for a yes and never message the shopper yourself (a message is kept for it). Once the basket was stated, keep the cart as stated while it can still be bought. When a refusal, a change or a record makes a different cart necessary (a line sold out, the budget cut, a better fit under a changed request), change it: the code states the new basket before the order while a message is left, and when none is left you still order the best fit. Never empty the cart or decline because no statement is possible: a decline is right only when nothing in the store fits.
- Before every order attempt: the cart holds exactly one line per product asked for (one line when one product is asked), each the chosen item at its required quantity, and nothing else; after your last cart change you have re-read the chosen item's stock and price, and it is in stock and within budget.
- Decline only after a thorough search finds nothing that meets every requirement, following the request's decline rule exactly. When the request says not to offer substitutes, the decline remark names, describes or points to no listing: no ids, titles or prices.
- Budget your steps: the first order attempt goes out while at least seven steps remain, so a refusal can still be handled. With 7 steps left only cart, stock-check and finishing tools remain. Decide once; do not swap the cart back and forth.
- Every reply carries at least one tool call. Keep the text short: the requirements now, the verified fits with price and stock, the next move."""

NO_STATEMENT = ("- THE BASKET STATEMENT: right before every order the code itself sends the shopper one message stating the basket (each line with its quantity and price, the total) and asking to go ahead; the shopper's acknowledgement is consent, so never wait for a yes and never message the shopper yourself (a message is kept for it). Once the basket was stated, keep the cart as stated while it can still be bought. When a refusal, a change or a record makes a different cart necessary (a line sold out, the budget cut, a better fit under a changed request), change it: the code states the new basket before the order while a message is left, and when none is left you still order the best fit. Never empty the cart or decline because no statement is possible: a decline is right only when nothing in the store fits.", '- THE QUESTIONS: the code itself sends the shopper the questions about hidden needs; never message the shopper yourself unless the shopper asked you something. When a refusal, a change or a record makes a different cart necessary (a line sold out, the budget cut, a better fit under a changed request), change it, and still order the best fit. Never empty the cart or decline because the messages are used up: a decline is right only when nothing in the store fits.')

READ_REQUEST = """Below are a shopper's request and a store's tool list. Answer with ONE compact JSON object (no indentation), nothing else, holding these keys:

"product": the kind of product asked for, in the request's words (for several products, all of them joined with "and").
"items": one entry per DIFFERENT product the request asks to buy, in its order: {"product": its kind in the request's words, "quantity": units of one listing of it (as for "quantity" below), "requirements": its firm requirements (as for "requirements" below)}. One entry when one product is asked.
"quantity": units of one listing to order (an integer; 1 when the request does not ask for several). Something sold as one item, such as a pair of shoes or a set, is quantity 1. Only a plain request to buy several units sets it ("I need three of them", "order 2 packs"); how many people will use or share it, or how many old ones there were, is story and leaves it at 1.
"requirements": a list of strings, one per firm requirement on the product (for several products: every product's, each written "property (product): constraint"), each "property: constraint" with the shopper's exact value, limit, exclusion or closed list. Leave out price, stock, quantity, budgets and mere wishes, and do not restate the product kind or category: words that belong to the category's name (a season, an occasion or an audience in it) are covered by the category, not a separate requirement. A requirement is a property of the product with the value the request says it must have ("the pattern has to be herringbone", "with a mechanical movement", "at least 4 drawers", "the occasion has to be party"). The story around it is never a requirement: whom it is for, an event it will be worn or used at ("to wear at his wedding", "for the staff party"), where, when or how it will be used, how many people share it, or that it replaces an old one ("a new watch" asks for a replacement, not for a condition). An occasion, use or audience is a requirement only when the request makes it the item's own property ("the occasion has to be business", "a blazer made for business"). A requirement's value is words or figures the request itself writes as the item's value; a quality, reason or wish the story gives (natural and breathable, slim enough for a case, works with what I already have, covers the whole drop) is a hint, never a requirement.
"price_cap": the highest total price as the number the request writes, in the request's own unit (never converted: "3050 US cents" gives 3050), or null when none is given.
"currency": the price unit of the request, or null.
"in_stock_only": true when the request wants only items on hand or in stock.
"priority": "cheapest" only when the request plainly asks for the cheapest or lowest-priced fitting item; "best" only when it plainly asks for the best by some measure; else "none". Saying money is tight or every penny counts is not a cheapest request; it only explains the cap.
"priority_detail": for "best", the measure; else "".
"message_budget": how many messages the shopper allows you to send, as an integer, or null when not said.
"step_budget": how many moves or steps the request allows, as an integer, or null when not said.
"rules": every procedural rule the request sets for the assistant (how many listings to look at, what to check or tell, how to decline, what not to do, what to include), quoted word for word. What the shopper will do themselves is story, not a rule.
"soft_wants": what the shopper would like without strictly needing it (ideally, preferably, would be nice, if possible). Keep these out of "requirements".
"ask_first": what only the shopper can tell and should be asked before choosing: for example the budget when no limit is given, which of two wishes matters more when the request says both may not be possible (then put both wishes in soft_wants, not in requirements), or anything the request says to check with the shopper. Empty when none.
"hints": the story details that point at a property of a product without stating its value (what it is used for or with, where, by whom, what it must fit, match or survive), each "property (product): the detail in a few words" ("property: detail" for one product), the property named as listings would likely name it. When a detail may concern more than one property, name the likeliest first and the next after a slash ("colour / style (chair): must suit a room with one painted wall"): what the item must match, suit or sit well with (a room, a wall, an outfit, things it is used with) and how it looks to the eye point first to its colour, then its pattern or style. Leave out a detail that only explains a requirement the request already states (why it must be the stated material or size). Weather, a job, a schedule, travel, or a place where the item is not worn or used is background, never a hint; so is a purpose every listing of the kind serves (storage for a dresser, warmth for a coat). Empty when none.
Every sentence of the request lands somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or plain story. When unsure whether a sentence is a rule, put it in "rules".
"references": requirements whose value the request gives only by pointing at ANOTHER listing or page ("the same colour as the jumpsuit sold by X", "within 5 of the figure on the page for Y", "a colour other than the one on Z's page"), each "property (product): as shown on <how the request names that listing>" ("property (product): different from the one shown on <how the request names that listing>" when the value must differ from that listing's; "property (product): counterpart of the one shown on <how the request names that listing>" when it asks for the counterpart of that listing's value); also put each in "requirements" in that form. Empty when none.
"reference_targets": for each entry of "references", the listing it points at: {"property": its property, "product": its product, "seller": the shop or brand name the request gives for that listing, exactly as written, "kind": the kind of product that listing is, "details": [each other value the request names for that listing, short and in the request's words ("blue", "4gb", "net weight 167.262186 g")], "relation": "same", "differ" (the value must NOT be that listing's: set apart from, different from, other than, not the one), "counterpart" (the value the store's pairing rule on that listing's page pairs with that listing's value: "the counterpart of the one on ...", "a type that works alongside the value shown on ...", "compatible with what that page gives"; "nothing other than" or "nowt but" that listing's value is "same"), "within", "at most" or "at least" (how the value must relate to that listing's), "tolerance": the number for "within", else null, "unit": its unit, else ""}. Empty when none.
"decline_no_names": true when the request's decline rule forbids offering, naming or suggesting substitutes or alternatives; false when it asks you to name something; null when it gives no rule.
"ties": for each product that must work with, fit or be listed as compatible with ANOTHER product of this request (an accessory bought together with its main item): {"product": that product in the request's words, "main": the product it must work with, in the request's words, "code": the model name or number the request gives for that main product, exactly as written, or ""}. Such a condition goes here only, not in "requirements"; the main product's own model name or number stays a requirement of the main product ("model (<main product>): <code>"). Empty when none.
"roles": an object naming the tool for each role, with the exact tool names from the list, or null when no tool fits: "search", "details", "view", "message", "message_status", "cart", "cart_view", "order", "decline". "message" sends a message to the shopper; "message_status" only reports how many messages are used or left; "cart" adds to or takes from the cart; "order" buys or checks out; "decline" ends the errand without buying.
"""

HEAR = """You help a shopping assistant understand what the shopper just said. Answer with ONE JSON object, nothing else, holding:
"meaning": one short sentence on what the shopper means.
"needs": new firm requirements on the product, each "property: constraint" with the exact value, limit, exclusion or closed list given. After a comparison word (falls short of, nothing less than, at least) a value that is words, not a figure, is the value itself, written as the shopper writes it: "falls short of not available" gives "property: not available" ("not available" and "none" can be a field's own values).
"no_preference": properties the shopper said they do not care about.
"dropped": earlier requirements the shopper no longer needs.
"new_budget": the new highest total price as a number in the unit of the request's budget (the figure the line writes when it uses that unit: "13075 cents" gives 13075 for a budget in cents), or null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), else false.
"meta_only": true when the line only asks the assistant to ask less at once (one question at a time, fewer things in one go, only the first now and the rest later) and states no value: it answers nothing yet, so put nothing in needs, no_preference or dropped for it, and its figurative words name no value; else false. A line that only says the shopper did not follow or understand the question, or asks for it again in shorter or simpler words, is not meta_only.
"requirements_now": the full list of firm requirements on the product as they stand now, each "property: constraint": start from REQUIREMENTS SO FAR, keep each unless the shopper dropped or replaced it, for a replaced value keep only the newest, and add each new need the shopper's lines state. Never add a requirement from the request's story that REQUIREMENTS SO FAR does not hold; only the shopper's lines add requirements. No price, stock, quantity or product category.
"reply": when the shopper asks the assistant something or asks for a confirmation, a short plain answer to send back that names the change in substance (for example confirming the new value); else "".
"references": for each need in "needs" whose value the shopper gives only by pointing at another listing or page, the listing it points at: {"property": its property, "seller": the shop or brand name the line gives for that listing, exactly as written, "kind": the kind of product that listing is, "details": [each other value the line names for that listing, short and in the line's words], "relation": "same", "differ" (the value must NOT be that listing's: set apart from, different from, other than, not the one), "counterpart" (the value the store's pairing rule on that listing's page pairs with that listing's value: "the counterpart of the one on ...", "a type that works alongside the value shown on ...", "compatible with what that page gives"; "nothing other than" or "nowt but" that listing's value is "same"), "within", "at most" or "at least", "tolerance": the number for "within", else null, "unit": its unit, else ""}; [] when none.
One line can carry several changes; report each. A need on another property, even a related one (a seat height beside an overall height, a strap width beside a width), is a new requirement that replaces nothing: an earlier requirement is dropped or replaced only when the line says so about that same property. A replaced requirement is both dropped (the old) and a new need (the new). When the shopper says which of two wishes matters more, the preferred one is a need. A stated budget is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and kindly: folksy, figurative or roundabout words that name a value mean the shopper firmly wants that value; a value is excluded only when the shopper clearly says to avoid it. Words that only stress how firmly the shopper holds a value (only, nothing but, nothing short of, no ... other than, not settle for ... other than, strictly, stays, sticking with, locked to, holding out for) are not part of it: write just the value words ("material: jute", "color: white"), never "<value> only"; only a shopper who also rules out mixes or other values in so many words adds an exclusion. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit. A new budget replaces the old one whether higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference or anything works, put the property asked about in no_preference (when the message asked about two properties, only the FIRST one). A reply that only says the shopper did not follow or understand the question ("that was rather dense", "you've lost me", "could you put it another way") is this shopper's way of having no preference when the question named the product the property belongs to: then put the property asked about in no_preference too; when several products are asked for and the question did not say which one, the property is still open (put nothing in no_preference). In a request for several products, write each need "property (product): constraint", the product taken from the line (a line about "toe style of the women's boots" gives "toe style (women's boots): ..."). The assistant's own question is never a requirement: only what the shopper says is; values named as examples in the question do not become needs. A "don't mind" answer frees only the property the assistant's message asked about; it never drops or changes a need the shopper stated earlier, and when the message asked nothing (it only told something), such an answer changes nothing. A need given by pointing at another listing keeps the line's own pointing words as its value ("finish: unlike the one shown on Y's kettle page"); the values the line names for that other listing describe that listing, not the product: they are no needs, drops or freed properties."""

GO_STEP = """A shopper's request gave a shopping assistant the numbered rules below. Name the rules that make the assistant send the shopper a message that asks nothing about what the products should be like: asking the shopper to approve, confirm or agree to the basket or the order before it is placed, or telling the shopper the basket, the choice or the total before ordering. Rules about which listings to look at, what to buy, how to decline or how many messages are allowed do not count. Answer with ONE JSON object, nothing else: {"steps": [the numbers of those rules]}; an empty list when no rule does."""
GO_ASK = re.compile(r"\b(?:ask|get|need|want|wait for|seek|request|have)\b[^.;]{0,20}\b(?:me|my)\b[^.;]{0,30}"
                    r"\b(?:approv\w*|confirm\w*|go[- ]ahead|ok|okay|sign[- ]?off|consent|yes|nod)\b"
                    r"|\b(?:my|me to)\s+(?:approv\w*|confirm\w*|go[- ]ahead|ok\b|okay|sign[- ]?off|consent)"
                    r"|\bconfirm\w*\b[^.;]{0,40}\bwith me\b|\bcheck with me\b|\bask me (?:first|before)\b"
                    r"|\b(?:check|clear|run|go over)\s+(?:it|them|this|that|the (?:\w+ ){0,2}(?:basket|cart|order|picks?|choices?|selection|items?|purchase))"
                    r"\s+(?:with|by|past) me\b", re.I)
GO_TELL = re.compile(r"\b(?:give|tell|send|show|report|let)\s+me\s+(?:know\s+)?(?:the|your|what|which|how much)\b[^.;,]{0,30}?"
                     r"\b(?:total|basket|cart|order total|final (?:basket|cart|order))\b", re.I)


def go_words(rule: str) -> bool:
    """A rule whose own words ask the shopper for approval of the order (ask me to approve, my go-ahead, confirm with me)
    or to be told the basket or total."""
    text = " ".join(str(rule or "").split())
    return bool(GO_ASK.search(text) or GO_TELL.search(text))


def word_set(text: Any) -> set[str]:
    """Returns the set of lowercase letter and digit runs in the text; the episode compares such sets to tell
    whether a must-ask item only restates a rule that wants the order approved or told.
    """
    return set(re.findall(r"[a-z0-9]+", str(text or "").lower()))


READ_RULES = """- Read each requirement from the listing's field for that property, by the value's own words as this store writes them in that field: a field whose name says the same property counts whatever its spelling (separators, a prefix, a shorter or longer name, or a usual alternative name such as "fit" for a pants fit type, "neckline" for a collar type, "color family" for a colour). When a field carrying the property's own name exists, it decides, and so does a field of that name with one more leading word (pants length for length, clothing material for material): when either one writes the value, the requirement is met. Two fields whose names differ only by separators (size class, size_class) are one field: the requirement is met when either one writes the value. Otherwise such an alternative field decides; only when no field names the property may the options list or the title show the value. The description is never evidence, with one exception: a model name or number required for the item itself is met wherever its record writes it (title, fields, overview or notes), spaces, hyphens and case aside.
- A product that must work with another product of the request is judged on that tie by the TIE EVIDENCE lines, when given: its record must name one of the main item's model names.
- Values match by their words, as this store writes them: case, separators, singular and plural and spelling variants are the same; a more specific value that contains the required one meets it ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060); one element of a list is enough. A synonym, an abbreviation, a related word or a shade ("tech wood" for technical wood, "win11" for windows 11, "embroidered" for embroidery, "solid" for plain, ivory or cream for off-white) meets a value only when no listing of the category writes the value itself in that field: when other listings write "matte" as their finish, a "satin" finish does not meet matte. The initials of the value's words are the value itself, whatever other listings write ("sff" for small form factor, "ssd" for solid state drive). A value joined with another by a plus ("wool + nylon") does not meet one of its parts when the field also carries the other part alone ("nylon"). FIELD EVIDENCE lines, when given, are computed from this store's records and say which case holds. A different value of the property does not (quartz is not mechanical, rosewood is not maple), nor a word that only looks alike. A value that claims to cover every value of the property ("suitable for all occasions", "all seasons", "universal") names none of them and does not meet a required value; a list that also names the required value still meets it. A field that names another value of the property does not meet the required value just because the two often go together or one tends to imply the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); only a shade usually sold as the required colour, or a more specific value that contains it, counts.
- A field of a different property does not count (a lining material is not the material, a length is not a style, a strap length is not a strap type): it never stands in for a property whose own field the category's listings carry.
- A requirement that the item fit, work with or suit a model (a vehicle, device or machine model) is met by a sentence of its record, the description and notes included, that names that model or a model family covering it ("every ABC-47xx model" covers ABC-4710: each x stands for one character), unless an "except" or "excluding" after it names that very model; a sentence about another part ("Part X also fits: ...") says nothing about this item. FIT EVIDENCE lines, when given, are computed by code and settle it.
- Numbers are compared as amounts after converting units (4.5 cm is 45 mm); limits include the edge; a range counts only when all of it meets the limit. When no field names the property and the title is the evidence, an exact required value is met when the title shows its words and figures in the same order, even with more detail after them (9' x 12' 4" shows 9' x 12'; 9' x 13' does not).
- A listing's price is in the same unit as the request's budget unless its record names another currency or unit (a price of 45 against a budget of 1000 US cents is 45 cents, well within it); never rescale a price or the budget.
- Only the requirements listed count: never fail a listing for an age group, gender, size, occasion, condition, season, quality or use the requirements do not name.
- The requested kind is met by the category label alone: a listing on the requested category is of that kind even when its title names another object (a container, a case, a refill, a set, an accessory); judge it only on the requirements, price and stock.
- A required value whose words all belong to the item's own category label (a label naming one kind, not two joined kinds) is met by that category itself: no field has to write it. CATEGORY lines, when given, say where this holds."""

CHECK_ITEM = """You are the last check before a shopping assistant commits to an item. The errand scores 1 only when the item meets every requirement below. Judge it ONLY from the records shown, by these rules:
""" + READ_RULES + """
- Words that belong to the category's own name are met by the category.
- Exclusions and closed lists are respected.
- The item must be the requested kind; when listings carry a category label, it must be the label matching the request's wording as a whole, not a narrower or neighbouring one.
- Price times quantity must be within the budget (equal is fine). If the records show it out of stock, it fails.
- A mere wish ("ideally", "would be nice") marked as not required is not a requirement.
When records disagree (for example on price or stock), the newest one counts.
Answer with ONE JSON object, nothing else: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement not met or not shown]}. Give fits=false only for a listed requirement that the item's fields clearly fail under these rules (a FIELD EVIDENCE line saying the item does NOT meet a requirement is such a failure), or that neither a field nor (when no field for it exists) the options list or the title states; do not fail an item for a missing record of stock or price."""

RANK_CHECK = """A shopping assistant is about to order an item for a request that asks for the PRIORITY. Below are the requirements, the chosen item and the listing records seen. Using the reading rules below (price times quantity within the budget; not shown out of stock), name the listings that may meet every requirement AND beat the chosen item on the priority (a lower price for the cheapest; a better value of the measure for the best). Leave out listings the records show failing a requirement, out of stock or over budget. Figures in a ROW's title (such as 55 x 94) do not say which dimension each one is, so they never rule a ROW out.
READING RULES:
""" + READ_RULES + """
Answer with ONE JSON object, nothing else: {"better": [ids, best first, at most 4], "reason": "one sentence"}."""

ASK_PLAN = """You plan the questions a shopping assistant sends to a shopper. The shopper holds hidden needs: each is a value of one property of the product, revealed only when a message asks about that exact property; the first thing a message asks is answered most reliably, and questions about anything else get a polite "don't mind". Choose up to QUESTION_COUNT different properties, most promising first:
0. first, anything listed below as MUST BE LEARNED FROM THE SHOPPER that has not been asked; ask it plainly. For the budget, ask the most they want to spend. For a choice between two wishes, name both values with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?";
1. then properties the listings' details below actually carry. Ask a product only about properties shown for its own category (its CATEGORY section or RANKED list below); a property seen only in another product's category, or only on a listing the request refers to, is never asked of it. Read the story first (STORY HINTS below, when given, name its details and the property each concerns: they come first): the hidden need is usually the property its concrete details concern without naming the value. What the user will touch or press, see up close or in photos, hear, measure or fit it against, read on the label, or how it was made points at that part or aspect of the item and its property: tags, labels or the fine print point to the material, fabric or ingredients; photos and looks to the pattern, colour or the visible part; a body part or a space it must fit to that part's shape or size; buying for someone else, or someone barely known, to whom it is made for (gender, age group); how it must feel to the hand, skin or foot first to the material or fabric; where, when or for what activity it is worn or used (a gym class, the office, outdoors) to its style or type, the property whose values name such uses; something done up or adjusted again and again to its closure or mechanism; what does the work inside it to its core component. Weather, a job, a schedule or a place that is not where the item is worn or used are background. Ask first about the property the story's details point at, by the name the listings use for it, choosing among the eight properties of the product's RANKED list below (the properties on which the store's listings differ most): the one the story's details point at; when they point at none of them, the one that most defines this kind of item, the property a shopper buying it is most likely to insist on (what kind or variant it is, or its main look, feel, taste or make), not simply the first of the list. Whom the item is made for is a candidate like any other: ask it when the story points at someone else; otherwise it stands where the RANKED list puts it. Then prefer properties describing what the product is like (pattern, material, shape, type or style variant, closure, length, function) over logistics (warranty, condition, pack size, variant or size lists, years).
Never pick a property the request already settles (its requirements, the price, stock, quantity), never repeat one already asked, never spend two questions on near-duplicates (two kinds of style, two spellings of one property), and never ask about brand, maker, a product or model name, price, stock or delivery. Ask only about a property that appears as a key in the listings' details below, by the name the listings use for it; a property under STORY HINTS that no key below matches may be asked by its plain name, unless the hint is marked [no key] (enough of its product's listings were read without that field): then ask the key below whose values differ the way its detail describes, or leave it. Each step-1 question is one short, plain, friendly sentence (at most 15 words) asking which <property> they would like for the <item kind>, naming the property by the listings' own name for it (example values alone never reveal a need), with no example values. No lead-in, no status report, no instructions or lists, and never a catch-all key such as options, variants, a description, a title or a name. Each question asks about ONE property of ONE product (the step-0 choice between two wishes aside): never two properties in one question and never properties of two products, since a reply reveals at most one need and a question naming two subjects may get no answer at all. In "second", give the next most promising property of the SAME product (not another name for the first), or leave it empty: it may be asked later, in a message of its own.
When the request asks for SEVERAL products, the hidden needs may sit on any of them: spread the questions so every product gets one before any gets a second (a product with no property left aside), take each product's properties from its own category's list below, write the property as "property (product)", and name the product in the question ("Which toe style would you like for the boots?").
Answer with ONE JSON object, nothing else: {"questions": [{"property": "...", "second": "... or empty", "question": "..."}, ...]}."""

NEXT_ONE = """
This time only ONE question is sent; the shopper's answer is read before the next is chosen. REMAINING question(s) are left in all and every one should be used: put the single most promising property first in the list (one or two more may follow as spares); after a revealed need, the next question goes to a different property and story detail."""

BEFORE_DECLINE = """A shopping assistant is about to tell the shopper that nothing in the store fits; that is rarely the right finish. Check the listing records below against the requirements by these reading rules (the price must be within the budget and the item must not be shown out of stock):
""" + READ_RULES + """
Some listings below are known only from search rows (no details read, so their property fields are unknown). Such a row on the requested category (whatever object its title names) whose title, price and stock do not show it failing a requirement has not been checked (figures in a title, such as 55 x 94, do not say which dimension each one is, so they never show a row failing): list it under "unopened", the likeliest first (its title naming a required value, or the requested kind with nothing against it).
Answer with ONE JSON object, nothing else: {"possible_fits": [ids of listings that seem to meet every requirement, best first, at most 5], "unopened": [ids of rows-only listings to open before declining, at most 6], "reason": "one sentence"}."""

REF_HOP = """A shopper's request takes the PROPERTY of the product it asks for from another listing in the store, described as: DESCRIPTION. Below are the records of listings that may be that listing. Answer with ONE JSON object, nothing else, holding:
"target": the id of the record that IS the described listing: sold by that seller, of that kind, and showing every detail named, read by meaning (a figure given in another unit counts when it is the same amount); null when none of them is.
"missing": the details the target does not show ([] when it shows them all).
"go_to": when the target's own text sends the reader to a DIFFERENT listing for the PROPERTY (take it from, use, refer to, look at or go by another listing), that listing as {"seller": its shop or brand name as written, "kind": its kind of product, "details": [each value the text names for it, short and as written: a colour, a size with its unit, a material]}; else null.
"value": when there is no go_to, the target's own value of the PROPERTY, as its record writes it.
"goes_with": the values the target's text says go together with, pair with or count the same as that value ([] when none). When the text states a pairing rule ("Our <property> rule: a goes together with b, and c goes together with d"), goes_with holds the value the rule pairs with the target's own value, read in either direction (for the value d: ["c"]).
A listing the text names only as what the target is not ("not the green one from X"), or as a version, size or colour sold elsewhere, is no go_to; nor is one it names for another property. A record that does not show the described details is not the target, whatever its text says."""

TIE_NAMES = """Below are the records of one store listing. List every model name or model number that this listing says the item itself has or is also sold or known as (its own model number and each other name given for the same item), exactly as the records write them. Leave out the names of other products it works with or fits, part numbers of other items, and plain words. Answer with ONE JSON object, nothing else: {"names": [...]}"""

DECLINE_WORDS = r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without"
ORDER_WORDS = r"commit|buy(?!er)|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed"      # (X3: not "buyer")
SORRY = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
ID_LIKE = r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}"
CODE_LIKE = r"[A-Za-z0-9_\-]+"


def as_text(piece: Any, ceiling: int = VIEW_CHARS) -> str:
    """Renders any value as compact JSON (non-ASCII kept, str() when it cannot be serialised) for a prompt,
    cutting it at the ceiling (VIEW_CHARS by default) and marking the cut with a trailing note.
    """
    try:
        text_body = json.dumps(piece, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text_body = str(piece)
    return text_body if len(text_body) <= ceiling else text_body[:ceiling] + "...(cut)"


def clipped(piece: Any, chars: int, lvl: int = 0) -> Any:
    """Returns a copy of a nested value in which every string longer than chars is cut and ends with an
    ellipsis; dicts and lists are copied level by level, anything else is kept as is, and parts nested
    deeper than eight levels become None. Used to keep observation records short in prompts.
    """
    if lvl > 8:
        return None
    if isinstance(piece, str):
        return piece if len(piece) <= chars else piece[:chars] + "..."
    if isinstance(piece, dict):
        return {fkey: clipped(val, chars, lvl + 1) for fkey, val in piece.items()}
    if isinstance(piece, list):
        return [clipped(val, chars, lvl + 1) for val in piece]
    return piece


def json_object(text_body: Any) -> dict[str, Any] | None:
    """Pulls one JSON object out of a model reply: non-strings give None, a surrounding code fence (with
    or without a language tag) is removed, then the whole text and, failing that, the span from the first
    opening brace to the last closing brace are parsed in turn. The first parse that yields a dict is
    returned; None when neither does.
    """
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
    """True for an int or float that is not a bool."""
    return isinstance(piece, (int, float)) and not isinstance(piece, bool)


def long_strings(piece: Any, collected: list[str], lvl: int = 0) -> None:
    """Appends to collected every string of four or more words found in the value, entering lists (their
    first ten entries) down to four levels but never dicts; returns nothing. The episode uses it to gather
    sentences, such as shopper lines or refusal texts, from an observation.
    """
    if lvl > 4:
        return
    if isinstance(piece, str):
        if len(piece.split()) >= 4:
            collected.append(piece)
    elif isinstance(piece, list):
        for thing in piece[:10]:
            long_strings(thing, collected, lvl + 1)


def walk_dicts(piece: Any, visit, lvl: int = 0, ceiling: int = 6) -> None:
    """Calls visit on every dict inside a nested value, a dict before its children, going through dicts and
    lists down to ceiling levels (six by default). Almost every scan of world observations (ids, prices,
    labels, fields) runs through it.
    """
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


def junk_key(key: Any) -> bool:
    """Tells whether a field key (before any parenthesis, camelCase split) names something that is not a
    product property: brand, seller, store, category, title, listing, description, price, availability,
    sku, url and the like, or a key that is wholly a name, notes, stock, images, links, make, shelf or id.
    """
    part = re.split(r"\s*\(", str(key))[0]
    words = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower()))
    return bool(re.search(r"\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|"
                          r"heading|listing|description|overview|price|cost|amount|availability|sku|url)\b", words)
                or re.fullmatch(r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)", words))


def prop_key(name: Any) -> str:
    """Normalises a property name to one comparison key: the part before any parenthesis, camelCase split,
    letters only, a plain plural s dropped from longer words, all joined without spaces.
    """
    part = re.split(r"\s*\(", str(name))[0]
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower())
    return "".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words)


def edits_within(a: str, b: str, most: int) -> bool:
    """True when string a becomes b within the given number of single-character insertions, deletions or
    substitutions. It fills an edit-distance table row by row and stops early when the lengths differ by
    more than the limit or a whole row already exceeds it.
    """
    if abs(len(a) - len(b)) > most:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > most:
            return False
        prev = cur
    return prev[-1] <= most


def same_prop(a: Any, b: Any) -> bool:
    """Loose test that two property names mean the same property, by their prop_key forms: equal keys, or
    keys of four or more letters that share the first three and differ by one edit (two once the shorter
    has six letters), or a shorter key of five or more letters that starts the longer one.
    """
    x, y = prop_key(a), prop_key(b)
    if not x or not y:
        return False
    if x == y:
        return True
    short = min(len(x), len(y))
    if short >= 4 and x[:3] == y[:3] and edits_within(x, y, 1 if short < 6 else 2):
        return True
    return short >= 5 and (x.startswith(y) or y.startswith(x))


def prop_words(name: Any) -> list[str]:
    """Splits a property name (before any parenthesis, camelCase split) into lowercase letter words, each
    longer word with a plain plural s removed.
    """
    part = re.split(r"\s*\(", str(name))[0]
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]


def spelled_alike(lo: str, hi: str) -> bool:
    """Decides whether two spellings of one word differ only trivially, lo being the shorter or equal one.
    At equal length exactly one pair of neighbouring letters may be swapped; when hi is one or two letters
    longer, removing that many soft letters (a vowel or y, or one letter of a doubled pair) from hi must
    give lo. Any other length gap fails.
    """
    gap = len(hi) - len(lo)
    if gap == 0:
        diff = [i for i in range(len(lo)) if lo[i] != hi[i]]
        return len(diff) == 2 and diff[1] == diff[0] + 1 and lo[diff[0]] == hi[diff[1]] and lo[diff[1]] == hi[diff[0]]
    if gap not in (1, 2):
        return False

    def soft(word: str, i: int) -> bool:
        """True when the letter at position i is a vowel or y, or sits next to an identical letter."""
        return word[i] in "aeiouy" or (i > 0 and word[i - 1] == word[i]) or (i + 1 < len(word) and word[i + 1] == word[i])
    for i in range(len(hi)):
        if not soft(hi, i):
            continue
        one = hi[:i] + hi[i + 1:]
        if gap == 1:
            if one == lo:
                return True
            continue
        for j in range(len(one)):
            if soft(one, j) and one[:j] + one[j + 1:] == lo:
                return True
    return False


def strict_prop(a: Any, b: Any) -> bool:
    """Stricter match of two property names than same_prop: equal prop_key forms, or keys of four or more
    letters with the same first three that spelled_alike accepts, or word lists where the longer is the
    shorter (five letters or more in all) plus one trailing word of at most three letters.
    """
    x, y = prop_key(a), prop_key(b)
    if not x or not y:
        return False
    if x == y:
        return True
    lo, hi = sorted((x, y), key=len)
    if len(lo) >= 4 and lo[:3] == hi[:3] and spelled_alike(lo, hi):
        return True
    wa, wb = prop_words(a), prop_words(b)
    short, longer = sorted((wa, wb), key=len)
    return len(longer) == len(short) + 1 and longer[:len(short)] == short and len(longer[-1]) <= 3 and len("".join(short)) >= 5


BOUND_WORDS = r"\b(?:no|not)\s+(?:more|less|fewer|over|under|above|below|beyond|exceeding)\b"
CH_BOUND = r"\b(?:at least|at most|no more than|no less than|more than|less than|up to|or more|or less|or larger|or smaller|or higher|" \
           r"or lower|minimum|maximum|between|within|under|over|above|below|than)\b"
CJ_COMPARE = (r"\b(?:falls?\s+short\s+of|short\s+of|nothing\s+(?:less|other)\s+than|nothing\s+short\s+of|nothing\s+but|"
              r"no\s+(?:less|worse)\s+than|at\s+least|(?:less|more|better|worse)\s+than|exactly|only|just)\b")
CH_SKIP = {"and", "the", "a", "an", "of"}
LITERAL_VALUES: set[str] = set()


def ch_words(text: Any) -> list[str]:
    """Lowercase words and numbers (decimals kept whole) of a text, without and, the, a, an and of; the word
    list that the value-writing checks compare.
    """
    return [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text or "").lower()) if w not in CH_SKIP]


def ch_word_in(word: str, pool: list[str]) -> bool:
    """True when the word occurs in the pool: the same word, a singular or plural form of it (both
    alphabetic), or, for alphabetic words of six or more letters, one with the same first two letters
    within one edit.
    """
    for other in pool:
        if other == word:
            return True
        if word.isalpha() and other.isalpha():
            if same_kind(word, other):
                return True
            if len(word) >= 6 and len(other) >= 6 and word[:2] == other[:2] and edits_within(word, other, 1):
                return True
    return False


def ch_alts(value: Any) -> list[list[str]]:
    """Splits a requirement value into alternatives on the word or, each a ch_words list, after removing the
    stated or revealed tags, quotes and stress words such as only, just, exactly and strictly.
    """
    plain = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", str(value or "").lower()).replace('"', " ")
    plain = re.sub(r"\b(?:only|just|exactly|strictly)\b", " ", plain)
    return [ws for ws in (ch_words(part) for part in re.split(r"\bor\b", plain)) if ws]


def ch_writes(value: Any, text: Any) -> bool:
    """True when the text writes the required value, testing the value's alternatives against the text's
    words with ch_match.
    """
    return ch_match(ch_alts(value), ch_words(text))


def ch_match(alts: list[list[str]], pool: list[str]) -> bool:
    """Checks whether a word pool writes any one of the value alternatives. An alternative matches when all
    its words are in the pool; when three or four alphabetic words have their initials as a pool word;
    when a single vowel-less word of three or four letters is the initials of consecutive pool words; or
    when its words run together equal consecutive pool words run together (tried for multi-word or
    numeric cases). An empty pool never matches.
    """
    if not pool:
        return False
    for ws in alts:
        if all(ch_word_in(w, pool) for w in ws):
            return True
        if 3 <= len(ws) <= 4 and all(w.isalpha() for w in ws) and "".join(w[0] for w in ws) in pool:
            return True
        if len(ws) == 1 and ws[0].isalpha() and 3 <= len(ws[0]) <= 4 and not re.search(r"[aeiouy]", ws[0]):
            n = len(ws[0])
            if any(all(p.isalpha() for p in pool[i:i + n]) and "".join(p[0] for p in pool[i:i + n]) == ws[0]
                   for i in range(len(pool) - n + 1)):
                return True
        target = "".join(ws)
        if len(ws) >= 2 or re.search(r"\d", target) or any(re.search(r"\d", p) for p in pool):
            for i in range(len(pool)):
                run = ""
                for j in range(i, len(pool)):
                    run += pool[j]
                    if run == target:
                        return True
                    if len(run) >= len(target):
                        break
    return False


def ch_touches(value: Any, text: Any) -> bool:
    """Looser companion of ch_writes: true when any word of any value alternative is in the text's words, a
    bare number of the value appears as a standalone number, or a value word of three or more letters
    shares its first three letters with a text word. Used to tell a field that comes near the value.
    """
    pool = ch_words(text)
    low = str(text or "").lower()
    for ws in ch_alts(value):
        for w in ws:
            if ch_word_in(w, pool):
                return True
            if re.fullmatch(r"\d+(?:\.\d+)?", w) and re.search(r"(?<![\d.])" + re.escape(w) + r"(?!\d|\.\d)", low):
                return True
            if len(w) >= 3 and any(len(p) >= 3 and (p.startswith(w[:3]) or w.startswith(p[:3])) for p in pool):
                return True
    return False


def ch_parts(text: Any) -> list[str]:
    """Splits a field value into lowercase parts on commas, semicolons and bars; the whole value when it has
    no separator, an empty list when blank.
    """
    low = str(text or "").strip().lower()
    return [p.strip() for p in re.split(r"[,;|]", low) if p.strip()] or ([low] if low else [])


def ch_sides(part: Any) -> list[str]:
    """Splits a value part on plus signs and returns the sides when there are two or more (a blend such as
    two joined materials), else an empty list.
    """
    sides = [s.strip() for s in re.split(r"\s*\+\s*", str(part)) if s.strip()]
    return sides if len(sides) >= 2 else []


def figure_written(fig: str, said: str) -> bool:
    """True when the figure appears in the text as a number of its own. A decimal figure counts at once; a
    whole figure counts only where no word follows it or the next word has at most three letters, such as
    a short unit.
    """
    for hit in re.finditer(r"(?<![\d.])" + re.escape(fig) + r"(?!\d|\.\d)", said):
        if "." in fig:
            return True
        tail = said[hit.end():]
        word = re.match(r"\s*([a-z]+)", tail)
        if not word or len(word.group(1)) <= 3:
            return True
    return False


EXCLUDE_WORDS = r"\b(?:no(?!-)|not|without|nothing|never|neither|nor|except|excluding|avoid\w*|anything but|other than|rather than|instead of)\b"


def positive_value(value: Any, name: Any = "") -> bool:
    """Tells whether a requirement value asks for something rather than rules it out. Yes/no-style words and
    values in LITERAL_VALUES pass; bound phrases (no more than, not over) are ignored; a piece of four or
    more content words fails as no plain value. Otherwise a value without an exclusion word passes, as
    does one with content before that word, or one whose words after it only repeat the property name.
    """
    low = str(value).lower().strip()
    if re.fullmatch(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*", low):
        return True
    if LITERAL_VALUES and " ".join(re.findall(r"[a-z0-9]+", re.sub(r"\((?:stated|revealed[^)]*)\)", " ", low))) in LITERAL_VALUES:
        return True

    def content(text: str) -> list[str]:
        """Words of three or more letters in the text, without linking words."""
        return [w for w in re.findall(r"[a-z]{3,}", text) if w not in LINK_WORDS]
    low = re.sub(BOUND_WORDS, " ", low)
    low = re.sub(r"\b(?:no|not)\s+[a-z]+er\b(?=\s+than\b)", " ", low)
    if re.search(r"\d", low):
        low = re.sub(r"\b(?:no|not)\s+[a-z]+er\b", " ", low)
    body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", low)
    if any(len(content(part)) >= 4 for part in re.split(r",|;|/|&|\(|\)|\bor\b|\band\b", body)):
        return False
    hit = re.search(EXCLUDE_WORDS, low)
    if not hit:
        return True
    if content(low[:hit.start()]):
        return True
    rest = content(low[hit.end():])
    own = set(prop_words(name)) | set(re.findall(r"[a-z]+", str(name).lower()))
    return bool(rest) and bool(own) and all(w in own or w.rstrip("s") in own for w in rest)


def value_seen(value: Any, samples: Any) -> bool:
    """True when sample text shows a requirement value: one of its content words (the last letter dropped
    from longer ones) starts a word in the samples, or one of its figure runs (single figures or figures
    joined by x, colon or slash) appears with at most four other characters between the figures.
    """
    pool = str(samples or "").lower()
    if not pool.strip():
        return False
    body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", str(value).lower())
    for w in re.findall(r"[a-z]{3,}", body):
        if w not in LINK_WORDS and re.search(r"\b" + re.escape(w[:-1] if len(w) > 4 else w), pool):
            return True
    for run in re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body):
        figs = re.findall(r"\d+(?:\.\d+)?", run)
        if re.search(r"(?<![\d.])" + r"\D{0,4}".join(re.escape(f) for f in figs) + r"(?![\d.])", pool):
            return True
    return False


def kind_words(text: Any) -> set[str]:
    """Words of three or more letters in a text, without linking words and with trailing s removed, as a set."""
    return {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", str(text).lower()) if w not in LINK_WORDS}


def kind_tokens(text: Any) -> list[str]:
    """Words of three or more letters in a text, in order, with possessive endings and linking words removed."""
    return [w for w in re.findall(r"[a-z]{3,}", re.sub(r"'s\b", "", str(text).lower().replace("’", "'"))) if w not in LINK_WORDS]


def plural_forms(word: Any) -> set[str]:
    """Builds the singular and plural spellings of a word so kind words can be matched across number.
    Besides the word itself it adds the s and es plurals, ies for a final y, and ves for a final f or fe;
    for a word already ending in s, es, ies or ves it adds the shorter singular forms instead.
    Only forms of three or more characters are kept.
    """
    w = str(word)
    out = {w, w + "s", w + "es"}
    if w.endswith("y"):
        out.add(w[:-1] + "ies")
    if w.endswith("fe"):
        out.add(w[:-2] + "ves")
    elif w.endswith("f"):
        out.add(w[:-1] + "ves")
    if w.endswith("ies"):
        out.add(w[:-3] + "y")
    if w.endswith("ves"):
        out |= {w[:-3] + "f", w[:-3] + "fe"}
    if w.endswith("es"):
        out.add(w[:-2])
    if w.endswith("s"):
        out.add(w[:-1])
    return {x for x in out if len(x) >= 3}


def same_kind(a: Any, b: Any) -> bool:
    """True when two kind words are equal or one is a singular or plural form of the other."""
    return a == b or str(b) in plural_forms(a) or str(a) in plural_forms(b)


def kind_in(word: Any, pool: Any) -> bool:
    """True when the word matches some word of the pool as the same kind."""
    return any(same_kind(word, other) for other in pool)


def kinds_within(words: Any, pool: Any) -> bool:
    """True when every word of the list matches some word of the pool as the same kind."""
    return all(kind_in(w, pool) for w in words)


def kind_score(own: Any, words: Any, shared: Any) -> float:
    """Scores how well a category label's words cover a product's own words: 1 for each own word found, 0.5
    when that word also belongs to another product of the request.
    """
    return sum(0.5 if kind_in(w, shared) else 1.0 for w in own if kind_in(w, words))


def name_pattern(name: Any) -> str:
    """Builds a regular expression for a product name: its kind tokens in order, each in any singular or
    plural form with an optional possessive, joined by spaces, apostrophes or hyphens. Empty for a name
    without such tokens.
    """
    pats = []
    for w in kind_tokens(name):
        forms = sorted(plural_forms(w), key=lambda x: (-len(x), x))
        pats.append(r"(?:" + "|".join(re.escape(x) for x in forms) + r")(?:'s\b|'|\b)")
    return r"[\s'-]+".join(pats)


VALUE_LINKS = r"at least|at most|no more than|no less than|more than|less than|up to|or more|or less|or larger|or smaller|" \
              r"or higher|or lower|minimum|maximum|between|within|under|over|above|below|exactly|only|either|neither|nor|than|plus"


def one_prop_hint(hint: str) -> str:
    """Rewrites a story hint whose property part lists several names split by slashes so it names one
    property: the first name keeps the product tag and the detail, and the others follow in an or-note at
    the end. A hint without a slash, a colon or two names, or one that fails to parse, comes back as is.
    """
    try:
        head, sep, detail = hint.partition(":")
        if "/" not in head or not sep:
            return hint
        tag = re.search(r"\(([^()]*)\)\s*$", head)
        names = [n.strip() for n in (head[: tag.start()] if tag else head).split("/") if n.strip()]
        if len(names) < 2:
            return hint
        return f"{names[0]}{' (' + tag.group(1).strip() + ')' if tag else ''}:{detail} (or {', '.join(names[1:])})"
    except Exception:
        return hint


def lead_value(detail: Any, request: str, product: str) -> str:
    """Finds a value the request places right before the product's name. The detail must open with a single
    word (followed by punctuation, its end or a linking word) of three or more letters that is not a
    linking word or the product's own kind word; the word is returned when the request writes it directly
    in front of the product's first kind word (any plural or possessive). Returns an empty string
    otherwise or on any error.
    """
    try:
        low = str(detail).lower()
        hit = re.match(r"\s*([a-z][a-z-]*[a-z])(?:\s*[,;.()]|\s*$|\s+(?:" + "|".join(sorted(LINK_WORDS)) + r")\b)", low)
        if not hit:
            return ""
        word = hit.group(1)
        kinds = [w for w in re.findall(r"[a-z]{3,}", str(product).lower()) if w not in LINK_WORDS]
        if len(word) < 3 or word in LINK_WORDS or not kinds or word.rstrip("s") in {k.rstrip("s") for k in kinds}:
            return ""
        first = kinds[0].rstrip("s")
        toks = re.findall(r"[a-z][a-z'-]*", str(request).lower())
        for i in range(len(toks) - 1):
            if toks[i] == word and toks[i + 1].split("'")[0].rstrip("s") == first:
                return word
    except Exception:
        return ""
    return ""


def story_guess(req: Any, request: str, refs: list[str]) -> list[str]:
    """Spots a stated requirement whose value the request does not write
    (a story detail read as a value).
    Returns the value's words and figure runs missing from the request,
    matched loosely (stems, prefixes, one edit).
    Nothing for references, written figures, or short values found.
    For a yes/no value the property name is checked instead.
    The episode may move such requirements out of the stated list.
    """
    req = str(req)
    head, _, value = req.partition(":")
    low = value.strip().lower()
    untag = lambda x: tuple(s.strip().lower() for s in (re.split(r"\s*\(", str(x).partition(":")[0])[0], str(x).partition(":")[2]))
    if not low or req in refs or untag(req) in {untag(r) for r in refs} or ref_text(low) \
            or re.search(r"as shown|same as|the page|listing|sold by|redirect|refer", low):
        return []
    if re.fullmatch(r"\W*(?:yes|no|true|false|required|included|needed)\W*", low):
        low = re.split(r"\s*\(", head)[0].lower()
    body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", low)
    said = str(request).lower()
    figs = re.findall(r"\d+(?:\.\d+)?", body)
    if figs and all(figure_written(f, said) for f in figs):
        return []
    said_words = set(re.findall(r"[a-z]+", said))
    said_stems = {s[:-1] if s.endswith("s") and len(s) > 3 else s for s in said_words}
    missing = []
    content = [w for w in re.findall(r"[a-z]{3,}", body) if w not in LINK_WORDS]
    for w in content:
        stem = w[:-2] if w.endswith("es") and len(w) > 4 else w[:-1] if w.endswith("s") and len(w) > 3 else w
        if not any(s == w or (s.startswith(stem) and len(s) <= len(stem) + 3) or (stem.startswith(s) and len(s) >= max(4, len(stem) - 2))
                   or (len(w) >= 4 and s.startswith(w)) or (len(w) >= 4 and len(s) >= 4 and s[0] == w[0] and edits_within(s, w, 1))
                   for s in said_words) \
                and not any(len(s) >= 3 and stem.startswith(s) and len(s) >= len(stem) - 3 for s in said_stems):
            missing.append(w)
    if len(content) <= 2 and any(len(w) >= 4 and w not in missing for w in content):
        return []
    for run in re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body):
        figs = re.findall(r"\d+(?:\.\d+)?", run)
        if len(figs) >= 2 and not re.search(r"(?<![\d.])" + r"\D{0,4}".join(re.escape(f) for f in figs) + r"(?!\d)", said):
            missing.append(run)
    return missing


REF_WORDS = (r"\bas shown\b|\bshown on\b|\bsame as\b|\bpage\b|\bsold by\b|\bredirect|\brefer(?:s|red|ring)?\b|\blisting points\b|\bpoints you\b"
             r"|\bcounterpart\b|\bpointed to\b")
REF_STEPS = 0.4
REF_CALLS = 12
REF_HOPS = 8
DIFFER_WORDS = (r"\bset apart from\b|\bapart from\b|\bdistinct from\b|\bdifferent (?:from|to|than)\b|\bdiffer(?:s|ent|ing)?\s+from\b"
                r"|(?<!nothing )(?<!none )(?<!nowt )(?<!naught )\bother than\b|\bunlike\b|\bnot the same as\b|\banything but\b"
                r"|\banything bar\b|\banything except\b"
                r"|\bsteer clear of\b|\bkeep clear of\b|\bnot the one\b|\bnot the value\b")
SAME_WORDS = (r"\b(?:nothing|none|nowt|naught)\s+(?:other|else)\s+than\b|\b(?:nothing|nowt|naught)\s+but\b"
              r"|\bnone other than\b")
PAIR_WORDS = (r"\bcounterpart\b|\bworks? (?:alongside|together with|in tandem with)\b|\bcompatible with\b"
              r"|\bpair(?:s|ed)? (?:up )?with\b|\bgoes (?:together )?with\b|\bplays? nicely\b|\bpartner(?:ed)? (?:of|with|to)\b")
LINK_TAIL = re.compile(r",?\s*\b(?:or else\b|or otherwise\b|whether\b|or (?:as )?followed\b|or traced\b)[^()]*$", re.I)
OR_STATED = re.compile(r"\(\s*or\s+(?:stated|pointed to|shown|given|named|written|listed)[^)]{0,24}\)", re.I)
REF_TAIL = re.compile(r"\b(?:i|i'm|i'll|i've|i'd|me|my|we|you|your|that's|it's)\b")
REF_VOCATIVES = {"mate", "matey", "friend", "sir", "madam", "love", "pal", "buddy", "lad", "lass", "dear", "captain", "skipper", "please"}
RULE_PAIR = re.compile(r"((?:[A-Za-z0-9 /\"'-]|\.(?=\d))+?)\s+(?:goes|go|pairs|pair|is paired|are paired|matches|match)\s+"
                       r"(?:together\s+)?with\s+((?:[A-Za-z0-9 /\"'-]|\.(?=\d))+?)(?=\s*(?:,|;|\.(?:\s|$)|\band\b|\bwhile\b|$))", re.I)


def rule_partners(text: Any, value: Any) -> list[str]:
    """Reads pairing rules of the form a goes with b (or pairs or matches with b) from a listing's text and
    returns the partner values of the given value, read in either direction, comparing sides by their
    ref_toks sets. Used when a request asks for the counterpart of another listing's value; an empty
    list on any error.
    """
    try:
        want = set(ref_toks(value))
        if not want:
            return []
        out: list[str] = []
        for hit in RULE_PAIR.finditer(str(text or "")):
            left = re.sub(r"^(?:and|while|or|but)\s+", "", hit.group(1).strip(), flags=re.I).strip(" \"'")
            right = hit.group(2).strip().strip(" \"'")
            lt, rt = set(ref_toks(left)), set(ref_toks(right))
            lead = re.match(r"(which|that|it|this|these|they|those)\b\s*(.*)$", left, re.I)
            rest = lead.group(2).strip() if lead else ""
            if lead and lead.group(1).lower() not in ("which", "it", "they") and rest and not re.fullmatch(
                    r"ones?|(?:typically|usually|often|always|also|best|only|normally|generally|commonly|mostly|neatly|nicely)\b.*",
                    rest, re.I):
                left, lt = rest, set(ref_toks(rest))      # (RPb: "that fitted sheet": a determiner before the value)
            elif lead:
                # (RP: "<...> is monofilament, which goes together with monofilament": the pronoun is the value before it)
                head = ref_toks(str(text or "")[:hit.start(1)])
                if len(head) >= len(want) and set(head[len(head) - len(want):]) == want and rt and rt != want and right not in out:
                    out.append(right)
                continue
            if lt and lt == want and right not in out:
                out.append(right)
            elif rt and rt == want and left not in out:
                out.append(left)
        return out
    except Exception:
        return []
POINT_WORDS = (r"\bwithin\b|\bat (?:most|least)\b|\bno (?:more|less) than\b|\bsame as\b|\bas shown\b|\bthe value\b|\bthe one\b"
               r"|\bthe counterpart of\b|\bcounterpart of\b|\bno (?:lower|higher|smaller|larger) than\b"
               r"|\bwhat'?s given\b|\bworks? alongside\b|\bcompatible with\b|\bnothing other than\b|\bnowt but\b")
REF_DIFFER_STEPS = 0.3
REF_DIFFER_BLIND = 5
SELLER_KEYS = r"shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by"


def ref_text(value: Any) -> bool:
    """True when a value points at another listing instead of naming a value (as shown, same as, page, sold
    by, refers, counterpart and similar wording).
    """
    return bool(re.search(REF_WORDS, str(value or "").lower()))


def ref_toks(text: Any) -> list[str]:
    """Tokens for comparing reference details: lowercase words and figures, a number joined to its short
    unit unless the unit is a linking word, linking words and single letters dropped, and plural endings
    (s, or es after ch, sh, x or z) removed from longer words.
    """
    low = re.sub(r"(\d)(?:\s+|\s*-\s*)([a-z]{1,8})\b", lambda h: h.group(0) if h.group(2) in LINK_WORDS else h.group(1) + h.group(2),
                 str(text or "").lower())
    out = []
    for tok in re.findall(r"\d+(?:[.,:/]\d+)*[a-z]*|[a-z][a-z0-9]*", low):
        if tok in LINK_WORDS or (len(tok) == 1 and not tok.isdigit()):
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") and not tok[-2].isdigit():
            tok = tok[:-2] if re.search(r"(?:ch|sh|x|z)es$", tok) else tok[:-1]
        out.append(tok)
    return out


def ref_figs(text: Any) -> set[tuple[float, str]]:
    """Collects (number, unit) pairs from a text, commas removed: each figure followed by a unit word of up
    to eight letters, slash-separated figures each paired with that unit, units singularised and linking
    words skipped. Used to check that a candidate listing shows the measurements a reference names.
    """
    out: set[tuple[float, str]] = set()
    try:
        low = str(text or "").lower().replace(",", "")
        for hit in re.finditer(r"(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*-?\s*([a-z]{1,8})\b", low):
            unit = hit.group(2)
            if len(unit) >= 3 and unit.endswith("s") and not unit.endswith("ss"):
                unit = unit[:-2] if re.search(r"(?:ch|sh|x|z)es$", unit) else unit[:-1]
            if unit in LINK_WORDS:
                continue
            for fig in re.split(r"\s*/\s*", hit.group(1)):
                out.add((float(fig), unit))
    except Exception:
        return out
    return out


def ref_name(text: Any) -> str:
    """Compact form of a seller or listing name: lowercase, possessive s removed, only letters and digits."""
    return re.sub(r"[^a-z0-9]", "", re.sub(r"'s\b", "", str(text or "").lower()))


def ref_relation(value: Any) -> tuple[str, float | None, str]:
    """Classifies how a value given by reference relates to the other listing's value and returns
    (relation, tolerance, unit). It returns same for wording that forbids anything else, counterpart for
    pairing words, differ for words such as unlike or apart from, within with the number and unit of a
    within phrase, then at most or at least for limit wording; plain same otherwise.
    """
    low = str(value or "").lower()
    if re.search(SAME_WORDS, low):
        return "same", None, ""
    if re.search(PAIR_WORDS, low):
        return "counterpart", None, ""
    if re.search(DIFFER_WORDS, low):
        return "differ", None, ""
    hit = re.search(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?", low)
    if hit:
        return "within", float(hit.group(1)), (hit.group(2) or "").strip()
    if re.search(r"\b(?:at most|no more than|up to|or less|or lower|or smaller|no higher than|no larger than|no greater than|no longer than|not (?:any )?more than|not (?:any )?higher than)\b", low):
        return "at most", None, ""
    if re.search(r"\b(?:at least|no less than|or more|or higher|or larger|no lower than|no smaller than|no shorter than|no fewer than|not (?:any )?less than|not (?:any )?lower than)\b", low):
        return "at least", None, ""
    return "same", None, ""


def ref_parse(text: Any, ok: Any = None) -> dict[str, Any] | None:
    """Parses a description of another listing into {seller, kind, details}. The seller is the first
    capitalised word of four or more letters, not at the very start, that the ok filter accepts; the kind
    comes from a phrase such as the X sold by before it, else from the phrase right after it. Details are
    the remaining clauses (split on commas, and, with, which, in, at), at most six, up to a clause with
    first- or second-person words or a form of address. Or-asides and trailing hedges are removed first.
    """
    try:
        body = re.split(r"\(\s*or\b", OR_STATED.sub(" ", str(text or "")))[0]
        body = LINK_TAIL.sub("", body)
        hit = next((m for m in re.finditer(r"(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])", body) if m.start() > 0
                    and (ok is None or ok(m.group(1)))), None)
        if not hit:
            return None
        before, after, kind = body[:hit.start()], body[hit.end():], ""
        sold = re.search(r".*(?:\bthe|\bfor|\ba|\ban)\s+(.*?)\s+(?:(?:sold|made|offered|listed)\s+by|from)\s*$", before, re.I | re.S)
        if sold:
            kind = sold.group(1)
        else:
            lead = re.match(r"\s*([^,;(]*?)(?=\s+(?:with|that|which|in|at|having|from|of|for|sold)\b|[,;]|$)", after, re.I)
            if lead:
                kind, after = lead.group(1), after[lead.end():]
        parts = re.split(r",|;|\band\b|\bwith\b|\bthat\s+(?:is|are|has|have|comes?(?:\s+as)?)\b|\bwhich\b|\bin\b|\bat\b", after, flags=re.I)
        details = []
        for part in parts:
            low_part = part.strip().lower()
            if REF_TAIL.search(low_part) or low_part.strip(" .!?") in REF_VOCATIVES:
                break
            words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.%:/&'-]*", part) if w.lower() not in LINK_WORDS]
            if words:
                details.append(" ".join(words))
        return {"seller": hit.group(1), "kind": kind.strip()[:80], "details": details[:6]}
    except Exception:
        return None


def prop_named(prop: Any, text: Any, exact: bool = False) -> bool:
    """True when the text writes the property's name (before any parenthesis, camelCase split) as consecutive
    words, an s or es plural on either side allowed unless exact is set. A name shorter than three
    characters never counts.
    """
    part = re.sub(r"([a-z])([A-Z])", r"\1 \2", re.split(r"\s*\(", str(prop or ""))[0])
    words = re.findall(r"[a-z0-9]+", part.lower())
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
    if not words or len(" ".join(words)) < 3:
        return False

    def same(a: str, b: str) -> bool:
        """Word equality for prop_named, also accepting an s or es plural of either word when not exact."""
        return a == b or (not exact and ((len(b) >= 3 and a in (b + "s", b + "es")) or (len(a) >= 3 and b in (a + "s", a + "es"))))
    n = len(words)
    return any(all(same(t, w) for t, w in zip(toks[i: i + n], words)) for i in range(len(toks) - n + 1))


def ref_span(prop: Any, source: Any) -> str:
    """Returns the source text that follows the property's name (its non-linking words, any separators
    between them), at most 500 characters, cut at an or-aside or a sentence end; empty when the name is
    not found or on error. It isolates the part of a request that describes the referenced value.
    """
    try:
        source = OR_STATED.sub(" ", str(source))
        words = [w for w in re.findall(r"[a-z0-9]+", re.split(r"\s*\(", str(prop))[0].lower()) if w not in LINK_WORDS]
        hit = re.search(r"\b" + r"\W+".join(map(re.escape, words)) + r"\b", str(source).lower()) if words else None
        if not hit:
            return ""
        return re.split(r"\(\s*or\b|(?<=[.!?])\s", str(source)[hit.end(): hit.end() + 500])[0]
    except Exception:
        return ""


def key_words(key: Any) -> list[str]:
    """Splits a field key into lowercase letter runs and digit runs, breaking camelCase first."""
    return re.findall(r"[a-z]+|\d+", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key)).lower())


def code_tokens(text: Any) -> list[str]:
    """Lowercase letter runs and digit runs of a text, so model codes compare regardless of spacing, hyphens
    and case.
    """
    return re.findall(r"[a-z]+|\d+", str(text or "").lower())


def names_code(text: Any, code: Any) -> bool:
    """True when a record's text names a model code: the code's tokens appear in a row once sentences about
    another part fitting and except or excluding tails are removed. Failing that, a family statement in
    the text that covers the code (see family_fit) decides.
    """
    want = code_tokens(code)
    n = len(want)
    if n == 0:
        return False
    raw = str(text or "")
    have = code_tokens(EXCEPT_TAIL.sub(" ", OTHER_PART_FITS.sub(" ", raw)))
    if any(have[i: i + n] == want for i in range(len(have) - n + 1)):
        return True
    return bool(family_fit(raw, code))


FAMILY_RULE = re.compile(r"\b(?:every|all|any)\s+((?:[a-z]+[\s-]?)?\d*)([x*#?]{1,4})\s+(?:models?|versions?|variants?|units?)\b"
                         r"([^.;\n]{0,200})", re.I)
FAMILY_EXCEPT = re.compile(r"^\s*,?\s*(?:except(?:\s+for)?|but\s+not|excluding|save\s+for)\s+(.+)$", re.I)
OTHER_PART_FITS = re.compile(r"\b(?:part|item|piece)\s+[a-z0-9][\w-]*\s+also\s+(?:fits|works\s+with|suits)\b\s*:?[^.;\n]*", re.I)
EXCEPT_TAIL = re.compile(r",?\s*\b(?:except(?:\s+for)?|but\s+not|excluding)\s+[^.;\n]*", re.I)


MODEL_CODE = re.compile(r"\b[A-Za-z]{1,6}-?\d{3,}[A-Za-z0-9]*\b|\b[A-Za-z]{0,6}-?\d{1,4}[xX*#?]{2,4}\b")
PART_SERIES = re.compile(r"\bpart\s+series\s*[:#-]?\s*([A-Za-z]{1,5}-?\d{1,4})\b", re.I)

# (CS) every series of a list ("Takes part series VI1, VI4"), and the series a part's own code carries ("[SI6-6465]")
SERIES_CODE = r"[A-Za-z]{1,5}-?\d{1,4}(?![A-Za-z0-9])"
SERIES_LIST = re.compile(r"\bpart\s+series\s*[:#-]?\s*(" + SERIES_CODE + r"(?:\s*(?:,\s*(?:and|or)\b|[,/&;|]|\band\b|\bor\b)\s*"
                         + SERIES_CODE + r")*)", re.I)
PART_CODE = re.compile(r"[\[(]\s*(?:part\s+)?([A-Za-z]{1,5}\d{1,3})-\d{2,6}\s*[\])]", re.I)


def series_key(code: Any) -> str:
    """(CSb) A series compared by its letters and figures (case and hyphen aside)."""
    return re.sub(r"[^A-Z0-9]", "", str(code or "").upper())


def part_series(text: Any) -> list[str]:
    """(CS) The part series a main item's record says it takes, as written, in order."""
    out: list[str] = []
    for hit in SERIES_LIST.finditer(str(text or "")):
        for code in re.findall(SERIES_CODE, hit.group(1)):
            if code_like(code) and series_key(code) not in {series_key(c) for c in out}:
                out.append(code)
    return out


def part_code_series(text: Any) -> list[str]:
    """(CS) The series a part's own code carries ("[SI6-6465]" and "(part VI4-1035)" are series SI6 and VI4)."""
    out: list[str] = []
    for hit in PART_CODE.finditer(str(text or "")):
        code = hit.group(1).upper()
        if code_like(code) and code not in out:
            out.append(code)
    return out


def family_rules(text: Any) -> list[tuple[Any, set[str], str, str]]:
    """Finds statements such as every ABC-47xx model in a text and returns, for each, a pattern for the base
    followed by as many characters as wildcards, the set of codes named after an except or excluding,
    the matched sentence and the base's letter prefix. Bases shorter than two characters are skipped.
    """
    out = []
    for hit in FAMILY_RULE.finditer(str(text or "")):
        base = "".join(code_tokens(hit.group(1)))
        if len(base) < 2:
            continue
        pat = re.compile("^" + re.escape(base) + "[a-z0-9]{%d}$" % len(hit.group(2)))
        gone: set[str] = set()
        cut = FAMILY_EXCEPT.match(hit.group(3) or "")
        if cut:
            for piece in re.findall(r"[a-z]+[\s-]?\d[\w-]*|\d{3,}[\w-]*", cut.group(1), re.I):
                gone.add("".join(code_tokens(piece)))
        lead = re.match(r"[a-z]+", base)
        out.append((pat, gone, hit.group(0).strip(), lead.group(0) if lead else ""))
    return out


def family_fit(text: Any, code: Any) -> bool | None:
    """Decides whether family statements in the text cover a model code: None when the code is empty or no
    statement shares its letter prefix, else True when one statement's pattern matches the code and does
    not except it, False otherwise.
    """
    want = "".join(code_tokens(code))
    if not want:
        return None
    lead = re.match(r"[a-z]+", want)
    rules = [r for r in family_rules(text) if r[3] == (lead.group(0) if lead else "")]
    if not rules:
        return None
    return any(pat.match(want) and want not in gone for pat, gone, _, _ in rules)


def code_like(code: Any) -> bool:
    """True when a string looks like a model code: it has a numeric token, three or more characters in all,
    and is not one token shorter than four characters.
    """
    toks = code_tokens(code)
    return bool(toks) and any(t.isdigit() for t in toks) and len("".join(toks)) >= 3 and not (len(toks) == 1 and len(toks[0]) < 4)


def rank_keys(tables: list[dict[str, Any]]) -> list[str]:
    """Ranks a product's listing keys by usefulness to ask. Counts key names
    per table (scalar non-empty values, non-junk, <=40 chars) with values;
    names sharing a word and much of their values, or always same values
    in same tables, join the commoner's family. Names with >32-char mean
    values drop. Score: family top count x rank_prior, less for one value,
    half values list/ratio-like, or near-all listings differ. Returns
    names best first (ties: more distinct values); feeds RANKED questions.
    """
    count: dict[str, int] = {}
    values: dict[str, set[str]] = {}
    per: dict[str, dict[int, str]] = {}
    for t_no, table in enumerate(tables):
        seen: set[str] = set()
        for key, val in table.items():
            if isinstance(val, bool) or not isinstance(val, (str, int, float)):
                continue
            name = " ".join(key_words(key))
            text_v = str(val).strip().lower()
            if not name or not text_v or len(name) > 40 or name in seen or junk_key(key):
                continue
            seen.add(name)
            count[name] = count.get(name, 0) + 1
            values.setdefault(name, set()).add(text_v[:60])
            per.setdefault(name, {})[t_no] = text_v[:60]
    family: dict[str, str] = {}

    def twin(a: str, b: str) -> bool:
        """True when key b appears in at least two tables, every one of them also holding key a with the very
        same value.
        """
        both = set(per.get(a, {})) & set(per.get(b, {}))
        return len(both) >= 2 and len(both) == len(per.get(b, {})) and all(per[a][i] == per[b][i] for i in both)
    for name in sorted(count, key=lambda x: (-count[x], len(x))):
        kw = set(name.split())
        root = next((f for f in dict.fromkeys(family.values()) if (set(f.split()) & kw
                     and len(values[f] & values[name]) >= max(1, min(len(values[f]), len(values[name])) // 2)) or twin(f, name)), None)
        family[name] = root or name
    score: dict[str, int] = {}
    variety: dict[str, set[str]] = {}
    total: dict[str, int] = {}
    for name, n in count.items():
        if sum(len(v) for v in values[name]) / len(values[name]) > 32:
            continue
        root = family[name]
        score[root] = max(score.get(root, 0), n)
        total[root] = max(total.get(root, 0), n)
        variety.setdefault(root, set()).update(values[name])
    ranked = []
    for name, n in score.items():
        d = len(variety[name])
        factor = 1.0
        if d <= 1:
            factor = 0.6
        elif sum(1 for v in variety[name] if re.search(r"[:|]|/.*\d|\d.*/", v)) * 2 >= d:
            factor = 0.4
        elif total[name] >= 3 and d >= 0.75 * total[name]:
            factor = 0.5
        ranked.append((-(n * factor), -d, name))
    return [name for _, _, name in sorted(ranked)]


def catch_all_key(name: Any) -> bool:
    """Tells whether a key is a catch-all or an identifier rather than a property worth asking: options,
    variants, warranty, package or sku, a model, part, serial or catalog number, code or id, or mpn, upc,
    ean, gtin or asin.
    """
    words = " ".join(key_words(name))
    if re.search(r"\b(?:options?|variants?|warranty|package|sku)\b", words):
        return True
    return bool(re.fullmatch(r"(?:(?:item|product|manufacturer|mfr)\s+)?(?:model|part|serial|catalog)(?:\s+(?:number|no|num|code|id|name))?"
                             r"|mpn|upc|ean|gtin|asin", words)
                or re.search(r"\b(?:model|part|serial)\s+(?:number|no|num|code|id)\b", words))


ALIKE_GENERIC = {"type", "of", "the", "and", "for", "size", "kind"}


def alike_score(prop: Any, key: Any) -> tuple[float, bool]:
    """Scores how closely a field key resembles a property name (before any parenthesis) and returns
    (score, lexical). Each shared word adds 3 (0.5 for generic words such as type, size, kind), each pair
    of differing words with the same first four letters adds 2, and a pair listed in ALIKE_WORDS adds 2.
    Lexical is true when a non-generic word or a four-letter prefix matched; identical or empty word sets
    give (0, False).
    """
    a, b = set(key_words(re.split(r"\s*\(", str(prop))[0])), set(key_words(key))
    if not a or not b or a == b:
        return 0.0, False
    score, lexical = 0.0, False
    for w in a & b:
        score += 0.5 if w in ALIKE_GENERIC else 3.0
        lexical = lexical or w not in ALIKE_GENERIC
    for x in a - b:
        for y in b - a:
            if len(x) >= 4 and len(y) >= 4 and x[:4] == y[:4]:
                score += 2.0
                lexical = True
    return score, lexical


def odd_key(name: Any) -> bool:
    """True for keys that make poor question subjects: names holding digits, filter or availability words,
    e.g., i.e. or etc, or names that start with is or has.
    """
    words = " ".join(key_words(name))
    return bool(re.search(r"\d", words) or re.search(r"\b(?:filters?|available|availability|e g|i e|etc)\b", words)
                or re.match(r"(?:is|has)\b", words))


def core_words(name: Any, product: Any = "") -> frozenset:
    """The property name's own words (before any parenthesis) as a frozenset, trailing s removed, without
    the product's words and linking words.
    """
    own = {w.rstrip("s") for w in key_words(product)}
    return frozenset(w.rstrip("s") for w in key_words(re.split(r"\s*\(", str(name))[0])
                     if w.rstrip("s") not in own and w not in LINK_WORDS)


def name_values(product: Any, labels: Any, records: list[dict[str, Any]], held: Any) -> list[tuple[str, str]]:
    """Finds property values the product name itself already writes: (key, phrase) pairs.
    Walks the listing field maps, junk and catch-all keys skipped.
    Takes whole values and their parts of one to three words found in the name,
    not label or linking words, the full name or its head noun.
    Phrases already held are skipped.
    Each phrase goes to the key that carries it most often.
    Phrases inside a longer found one are dropped.
    """
    name_toks = re.findall(r"[a-z0-9]+", str(product).lower())
    name = " " + " ".join(name_toks) + " "
    head = name_toks[-1].rstrip("s") if name_toks else ""
    label = {w.rstrip("s") for s in labels for w in re.findall(r"[a-z0-9]+", str(s).lower())}
    held_low = " " + " ".join(re.findall(r"[a-z0-9]+", str(held).lower())) + " "
    found: dict[str, dict[str, int]] = {}
    for fields in records:
        for key, val in (fields or {}).items():
            if junk_key(key) or catch_all_key(key) or odd_key(key):
                continue
            kname = " ".join(key_words(key))
            if not kname:
                continue
            for part in {str(val)} | set(re.split(r"\s*[,/;|]\s*", str(val))):
                words = re.findall(r"[a-z0-9]+", part.lower())
                if not words or len(words) > 3:
                    continue
                phrase = " ".join(words)
                if len(phrase) < 3 or all(w.rstrip("s") in label for w in words) or all(w in LINK_WORDS for w in words):
                    continue
                if phrase == " ".join(name_toks) or (len(words) == 1 and words[0].rstrip("s") == head):
                    continue
                if f" {phrase} " in name:
                    found.setdefault(phrase, {})
                    found[phrase][kname] = found[phrase].get(kname, 0) + 1
    out: list[tuple[str, str]] = []
    for phrase, keys in found.items():
        if f" {phrase} " in held_low:
            continue
        best = sorted(keys.items(), key=lambda kv: (-kv[1], len(kv[0])))
        if len(best) > 1 and best[1][1] == best[0][1] and not (set(best[0][0].split()) & set(best[1][0].split())):
            continue
        out.append((best[0][0], phrase))
    return [(k, p) for k, p in out if not any(p != q and f" {p} " in f" {q} " for _, q in out)]


# ================================================================================================================ text records
TX_PAIR_MARKS = (": ", " = ", "=", " -> ", " - ")
TX_FIELD_MARKS = (" | ", "\n", " || ", " ; ", "; ", "\t", " // ", " · ", " • ", "|", ";", ", ")
TX_NAME = r"[A-Za-z][A-Za-z0-9_ ./&'()-]{0,38}"
TX_BULLET = r"^\s*(?:[-*•·]|\d{1,3}[.)])\s+"
TX_MIN_FIELDS = 3          # named fields a text needs to count as a record
TX_NOTE_FIELDS = 5         # without an id, this many fields are needed (a note with colons stays a note)
TX_SHARE = 0.15            # share of the separated parts that must carry a name (0.7 for a comma)
TX_COMMA_SHARE = 0.7
TX_ID_KEY = (r"(?:^|[_ -])(?:id|idx|sku|pid|ref|asin|ident\w*)(?:$|[_ -])|^id(?-i:[A-Z0-9])|(?-i:Ident)|(?<![a-z])"
             r"(?!(?:paid|valid|grid|solid|rigid|fluid|liquid|hybrid|rapid|void|acid|humid|android|said|laid)$)[a-z]*id$")


def tx_value(text: str) -> Any:
    """A field's text as a structured result would hold it: a boolean, None, a number, else the text."""
    bare = text.strip()
    low = bare.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    money = re.fullmatch(r"[$€£]?\s?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)", bare)
    if money and not re.match(r"0\d", money.group(1)):
        number = float(money.group(1).replace(",", ""))
        return int(number) if number.is_integer() and "." not in money.group(1) else number
    return bare


def tx_pair(text: str, mark: str) -> tuple[str, str] | None:
    """(name, value) when the text opens with a short field name and the mark."""
    head, found, rest = text.partition(mark)
    if not found and text.endswith(mark.strip()):
        head, found, rest = text[:-len(mark.strip())], mark, ""
    name = re.sub(TX_BULLET, "", head).strip()
    if not found or not re.fullmatch(TX_NAME, name) or len(name.split()) > 5:
        return None
    return name, rest.strip()


def tx_fields(text: str, cut: str, mark: str) -> tuple[list[tuple[str, str]], int]:
    """The (name, value) fields of a text cut at `cut` (a part without a name continues the field before it), and how many
    parts carried a name."""
    parts = [p for p in text.split(cut) if p.strip()]
    if not parts or not tx_pair(parts[0].strip(), mark):
        return [], 0
    fields: list[tuple[str, str]] = []
    named = 0
    for part in parts:
        pair = tx_pair(part.strip(), mark)
        if pair:
            fields.append(pair)
            named += 1
        elif fields:
            fields[-1] = (fields[-1][0], fields[-1][1] + cut + part.strip())
    return fields, named


def tx_block(text: str) -> dict[str, Any] | None:
    """Attributes written inside one field: "a=b; c=d" parts, or short "name: value" parts between commas or semicolons."""
    bare = text.strip()
    if len(bare) < 5:
        return None
    if "=" in bare:
        out: dict[str, Any] = {}
        last = ""
        for part in [p for p in re.split(r";\s+|;$|\s\|\s|\n", bare) if p.strip()]:
            name, found, value = part.partition("=")
            if found and 0 < len(name.strip()) <= 60 and name.strip() not in out:
                last = name.strip()
                got = tx_value(value)
                out[last] = value.strip() if isinstance(got, bool) or (got is None and value.strip()) else got
            elif last:
                out[last] = f"{out[last]}; {part.strip()}"
            else:
                return None
        return out if len(out) >= 2 else None
    for cut in (", ", "; "):
        if cut in bare and ": " in bare:
            pairs = [tx_pair(p.strip(), ": ") for p in bare.split(cut)]
            if len(pairs) >= 2 and all(pairs) and all("." not in n and len(v) <= 40 for n, v in pairs):
                return {n: tx_value(v) for n, v in pairs}
    return None


def tx_record(text: str) -> dict[str, Any] | None:
    """The record a text spells field by field (separators found by trying them: the first field separator that works,
    with the pair mark naming the most fields); None for prose."""
    best: tuple[int, list[tuple[str, str]]] | None = None
    for cut in TX_FIELD_MARKS:
        if cut not in text:
            continue
        pieces = len([p for p in text.split(cut) if p.strip()])
        for mark in TX_PAIR_MARKS:
            fields, named = tx_fields(text, cut, mark)
            if named < TX_MIN_FIELDS or named < (TX_COMMA_SHARE if cut == ", " else TX_SHARE) * pieces:
                continue
            if len({n for n, _ in fields}) * 2 <= len(fields):
                continue
            if best is None or named > best[0]:
                best = (named, fields)
        if best:
            break
    if best is None:
        return None
    record: dict[str, Any] = {}
    for name, value in best[1]:
        if name in record:
            continue
        inner = tx_block(value)
        record[name] = inner if inner else value.strip() if re.search(TX_ID_KEY, name, re.I) else tx_value(value)
    return record


def tx_listing(record: Any) -> bool:
    """A restored record that reads as a listing: an id-named field or an id-like value, or many fields."""
    if not isinstance(record, dict):
        return False
    if len(record) >= TX_NOTE_FIELDS or any(re.search(TX_ID_KEY, str(k), re.I) for k in record):
        return True
    return any(isinstance(v, str) and 3 <= len(v.strip()) <= 64 and " " not in v.strip() and re.search(r"\d", v)
               and re.fullmatch(ID_LIKE, v.strip()) for v in record.values())


def tx_text(text: str) -> Any:
    """A text a tool returned as its structured form (a record, or a list of records one per line or paragraph); None when
    it holds no listing record."""
    bare = text.strip()
    if len(bare) < 12:
        return None
    if bare[0] in "{[":
        try:
            found = json.loads(bare)
            return found if isinstance(found, (dict, list)) else None
        except (ValueError, RecursionError):
            pass
    for parts in (re.split(r"\n\s*\n", bare), bare.splitlines()):
        if len(parts) >= 2:
            rows = [tx_record(p) for p in parts]
            good = [r for r in rows if r and tx_listing(r)]
            if len(good) >= 2 and len(good) * 5 >= len(parts) * 4 and len({next(iter(r)) for r in good}) == 1:
                return good
    found = tx_record(bare)
    return found if tx_listing(found) else None


def tx_keyed(node: dict[str, Any]) -> bool:
    """An object that is a record already: an id-named field with a value, or an id-like value."""
    for key, val in node.items():
        if isinstance(val, (str, int)) and not isinstance(val, bool) and str(val).strip() and re.search(TX_ID_KEY, str(key), re.I):
            return True
        if isinstance(val, str) and re.search(r"\d", val) and re.fullmatch(ID_LIKE, val.strip()):
            return True
    return False


def text_records(piece: Any, lvl: int = 0) -> tuple[Any, bool]:
    """The observation with its listing texts as objects, and whether anything was turned. Objects and lists keep their
    shape; only string values outside records are read (a whole reply, a list item, a wrapper's value)."""
    if lvl > 6:
        return piece, False
    if isinstance(piece, str):
        found = tx_text(piece)
        return (found, True) if found is not None else (piece, False)
    if isinstance(piece, list):
        out, hit = [], False
        for val in piece:
            new, got = text_records(val, lvl + 1)
            out.append(new)
            hit = hit or got
        return (out, True) if hit else (piece, False)
    if isinstance(piece, dict):
        keyed = tx_keyed(piece)      # (a record already, or a wrapper holding an id: its own texts stay text, its rows are read)
        if keyed and any(re.search(r"title|price|name|cost|amount", str(k), re.I) for k in piece):
            return piece, False      # (a listing record: its list fields - variants, highlights - stay as they are)
        out_d, hit = {}, False
        for key, val in piece.items():
            readable = isinstance(val, list) if keyed else isinstance(val, (str, list, dict))
            new, got = text_records(val, lvl + 1) if readable else (val, False)
            out_d[key] = new
            hit = hit or got
        return (out_d, True) if hit else (piece, False)
    return piece, False


def fixed_arg(spec: Any) -> Any:
    """The value a required argument other than a message's text can take without a choice: its const, its default, or
    for an enum the value naming the shopper or a message (else the first); None when the schema leaves it open."""
    if not isinstance(spec, dict):
        return None
    if "const" in spec:
        return spec["const"]
    if spec.get("default") is not None:
        return spec["default"]
    options = [v for v in spec.get("enum") or [] if v is not None]
    if options:
        return next((v for v in options if re.search(r"shopper|buyer|customer|user|client|question|ask|message|text|chat",
                                                     str(v), re.I)), options[0])
    return None


def id_texts(piece: Any, lvl: int = 0) -> Any:
    """The observation with numeric listing ids (4+ digits under an id-named key) written as text."""
    if lvl > 6:
        return piece
    if isinstance(piece, list):
        return [id_texts(v, lvl + 1) for v in piece]
    if isinstance(piece, dict):
        return {k: (str(v) if isinstance(v, int) and not isinstance(v, bool) and abs(v) >= 1000 and re.search(TX_ID_KEY, str(k), re.I)
                    else id_texts(v, lvl + 1)) for k, v in piece.items()}
    return piece




def id_key(key: Any) -> bool:
    """A key that names a listing id: "id", or an id word inside it (TX_ID_KEY: item_id, product_id_ref, sku, ref)."""
    return str(key).lower() == "id" or bool(re.search(TX_ID_KEY, str(key), re.I))


def record_id(d: Any) -> str:
    """The listing id of a record: the id-like text under its id key, else ''."""
    if not isinstance(d, dict):
        return ""
    for key, val in d.items():
        if isinstance(val, str) and id_key(key) and re.fullmatch(ID_LIKE, val.strip()):
            return val.strip()
    return ""


def shelf_last(value: Any) -> str:
    """The last step of a category path (A > B > C gives C); a plain name as it is."""
    parts = [p.strip() for p in re.split(r"\s*(?:>|/|\||»)\s*", str(value or "")) if p.strip()]
    return parts[-1] if parts else str(value or "").strip()


def shelf_norm(text: Any) -> str:
    """A category name as compared: lower case, single spaces, no outer punctuation."""
    return " ".join(str(text or "").lower().split()).strip(" .,:;'\"")


def shelf_hit(value: Any, names: set[str]) -> bool:
    """Whether a field value is one of the request's category names, whole or as the last step of a path (A > B > C)."""
    plain = shelf_norm(value)
    if not plain:
        return False
    last = shelf_norm(re.split(r"\s*(?:>|/|\||»)\s*", plain)[-1])
    return plain in names or last in names


def lensed(piece: Any) -> Any:
    """Returns a JSON-copied observation with the field names later code looks for (an uncopyable value
    comes back unchanged). A top-level dict without an error key gets one from a non-empty problem,
    issue, fault or failure string. Every dict, the top one included, is then passed to visit, which adds
    standard id, price, in_stock and title keys taken from store-specific names. Applied to each tool
    reply the episode absorbs.
    """
    try:
        piece = json.loads(json.dumps(piece))
    except (TypeError, ValueError):
        return piece
    if isinstance(piece, dict) and not any(str(key).lower() == "error" for key in piece):
        flag = next((key for key, val in piece.items() if isinstance(val, str) and val.strip()
                     and re.fullmatch(r"problems?|issues?|faults?|failures?", str(key).lower())), None)
        if flag is not None:
            piece["error"] = piece[flag]

    def visit(d: dict[str, Any]) -> None:
        """Adds standard keys to one record dict, placed first. A record without an id-named key holding an
        id-like value takes its id from a ref or sku key, or is left alone when it has none or already has an
        id key. A missing price comes from the single numeric amount, fee or charge-like key, a missing
        in_stock from the single boolean on-hand or orderable-like key, and a missing title from the single
        headline, name, label or caption string not naming a seller, category or id.
        """
        front: dict[str, Any] = {}
        if not any(isinstance(val, str) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, val) for key, val in d.items()):
            ref = next((val for key, val in d.items() if isinstance(val, str) and re.fullmatch(ID_LIKE, val)
                        and any(w in ("ref", "refs", "sku", "skus") for w in key_words(key))), None)
            if ref is None or "id" in d:
                return
            front["id"] = ref
        elif "id" not in d and not any(str(key).lower().endswith("id") and isinstance(val, (str, int)) and not isinstance(val, bool)
                                       for key, val in d.items()):
            # (L1: an id word inside the key - product_id_ref - is the listing id the code reads as "id")
            refs = [(not re.search(r"(?:^|[_ -])id(?:$|[_ -])|[a-z]id$|Id$|ID$", str(key)), val) for key, val in d.items()
                    if isinstance(val, str) and re.fullmatch(ID_LIKE, val) and re.search(TX_ID_KEY, str(key), re.I)]
            if refs:
                front["id"] = sorted(refs, key=lambda kv: kv[0])[0][1]      # (L1b: a key whose own word is id first)
        names = {str(key).lower(): key for key in d}
        if not any(re.search(r"price|cost", low) and is_number(d[key]) for low, key in names.items()):
            picks = [key for low, key in names.items() if is_number(d[key])
                     and re.search(r"amount|fee|charge|pay|rate|value|(?<!per)cents?\b|money", low)
                     and not re.search(r"total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|size|weight|width|height|length|found|match", low)]
            if len(picks) == 1:
                front["price"] = d[picks[0]]
        if not any(re.search(r"stock|avail", low) for low in names):
            picks = [key for low, key in names.items() if isinstance(d[key], bool)
                     and re.search(r"hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store", low)]
            if len(picks) == 1:
                front["in_stock"] = d[picks[0]]
        if not any(re.search(r"title|heading", low) for low in names):
            picks = [key for low, key in names.items() if isinstance(d[key], str) and re.search(r"headline|name|label|caption", low)
                     and not re.search(r"shop|seller|store|vendor|brand|make|categor|shelf|department|aisle|id", low)]
            if len(picks) == 1:
                front["title"] = d[picks[0]]
        if front:
            rest = {key: val for key, val in d.items() if key not in front}
            d.clear()
            d.update(front)
            d.update(rest)
    walk_dicts(piece, visit)
    return piece


class ArgCheck:

    """Checks and repairs tool-call arguments against a tool's JSON schema; the episode also runs it on empty
    arguments to see what a tool requires. TYPES maps schema type names to Python tests, and RULES (set
    after the class) pairs schema keywords (enum, const, length and numeric bounds, item counts, items, and
    type for object properties) with rule functions. execute returns the possibly converted value with a
    list of fault strings, each prefixed by the argument path.
    """
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
        """The schema's type entry as a list: one string becomes a one-item list, a missing type an empty one."""
        type_list = jschema.get("type")
        return [type_list] if isinstance(type_list, str) else list(type_list or [])

    @classmethod
    def conforms(cls, piece: Any, ktype: str) -> bool:
        """True when the value passes the test for the named schema type; an unknown type name always passes."""
        test = cls.TYPES.get(ktype)
        return test(piece) if test else True

    @classmethod
    def convert(cls, piece: Any, type_list: list[str]) -> Any:
        """Coerces a value towards one of the allowed schema types. A value that already conforms is kept, apart
        from a whole float turned into an int where integer is allowed. Otherwise it tries each type in turn:
        whole floats and integer strings to int, numeric strings to a number, numbers to a string, and true or
        false strings to a bool. Anything else comes back unchanged.
        """
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
        """Validates a value against a schema and returns (value, faults). A non-dict schema accepts anything.
        The value is first converted; when it still matches none of the listed types, one expected-type fault
        is returned. Otherwise each RULES keyword present in the schema runs in order, each able to repair
        the value and add faults.
        """
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
        """Checks a value against the schema's enum list and returns it with its faults.
        A string matching an option apart from case and outer spaces is replaced by that option.
        Any other value not in the list gets a must-be-one-of fault naming the options.
        """
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
        """Faults a value that differs from the schema's const."""
        return piece, ([] if piece == jschema["const"] else [f"{where}: must equal {jschema['const']!r}"])

    @staticmethod
    def _min_length(piece, jschema, where):
        """Faults a string shorter than the schema's minLength."""
        if isinstance(piece, str) and len(piece) < jschema["minLength"]:
            return piece, [f"{where}: shorter than {jschema['minLength']} characters"]
        return piece, []

    @staticmethod
    def _max_length(piece, jschema, where):
        """Cuts a string to the schema's maxLength instead of faulting it."""
        return (piece[: jschema["maxLength"]] if isinstance(piece, str) else piece), []

    @staticmethod
    def _bound(call_name, test, words):
        """Builds a rule function for one numeric limit keyword (minimum, maximum or the exclusive ones): the test
        decides when a number breaks the limit, and the words name the breach in the fault.
        """
        def rule_fn(piece, jschema, where):
            """Reads the limit from the schema and, when both the value and the limit are numbers and the test
            fires, returns a fault naming the breach and the limit; the value itself is never changed.
            """
            ceiling = jschema[call_name]
            if is_number(piece) and is_number(ceiling) and test(piece, ceiling):
                return piece, [f"{where}: {words} {ceiling}"]
            return piece, []
        return rule_fn

    @staticmethod
    def _min_items(piece, jschema, where):
        """Faults a list with fewer entries than the schema's minItems."""
        if isinstance(piece, list) and len(piece) < jschema["minItems"]:
            return piece, [f"{where}: needs at least {jschema['minItems']} items"]
        return piece, []

    @staticmethod
    def _max_items(piece, jschema, where):
        """Faults a list with more entries than the schema's maxItems."""
        if isinstance(piece, list) and len(piece) > jschema["maxItems"]:
            return piece, [f"{where}: allows at most {jschema['maxItems']} items"]
        return piece, []

    @classmethod
    def _items(cls, piece, jschema, where):
        """Runs execute on every entry of a list against the schema's items schema, with the entry's index
        added to the path, and returns the repaired list with all entry faults. Non-lists, or an items
        entry that is not a dict, pass untouched.
        """
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
        """Checks an argument object: faults each required name that is missing and, when additionalProperties
        is false, each name the schema does not define. Defined properties are checked recursively with
        execute; undefined ones are kept unchanged in the returned object. RULES reaches it through the type
        keyword for dict values.
        """
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



class WorldReply:
    """A world call's outcome in the proxy client's shape: data (the reply's JSON) or error ({status, body})."""

    def __init__(self, data: Any = None, error: Any = None):
        """Stores the reply data and the error, at most one of which is normally set."""
        self.data, self.error = data, error


WORLD_LINKS: dict[str, Any] = {}        # binding -> a weak reference to the errand's Wire (its paced sender)


def link_key(binding: dict[str, Any]) -> str:
    """Stable text key for a world binding (JSON with sorted keys), used to find the episode's paced sender
    in WORLD_LINKS.
    """
    return json.dumps(binding, sort_keys=True, default=str)


def _env_post(binding: dict[str, Any], step: int, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Sends one action to the world (POST /environment/call) and returns the reply's JSON (observation, finished, step),
    or {'__error__': {'status': ..., 'body': ...}} when the call failed."""
    payload = dict(binding)
    payload["step"] = step
    payload["action"] = {"name": name, "args": args}
    link = WORLD_LINKS.get(link_key(binding))
    wire = link() if link else None
    answer = (wire.send(wire.world, "/environment/call", payload) if wire is not None
              else ProxyClient(timeout=95, max_retries=1, retry_delay=1).post_verbose("/environment/call", payload))
    if isinstance(answer.data, dict):
        return answer.data
    return {"__error__": dict(answer.error or {})}


OWN_ENV_POST = _env_post


class Wire:

    """Transport for one episode: sends model requests to the inference proxy and actions to the world, both
    paced by one send method. It tracks the time left before DEADLINE, the model spend against SPEND_CAP
    (scaled by an optional scale attribute for several-product requests), the world step counter tick,
    and whether the episode is finished (done) or stopped by a 402 world reply (broke). It remembers models
    that refused calls (banned, tool_banned), spare models a refusal named, failure streaks and a log of
    failed calls. Creating it registers its world sender in WORLD_LINKS.
    """
    def __init__(self, bind: dict[str, Any]):
        """Sets up the clients and counters for one episode: the start time, a copy of the world binding, two
        proxy clients (model and world, one retry each) and the paced world sender in WORLD_LINKS. It also
        zeroes spend, tick and the failure count, clears done and broke, starts the call log, and resets
        the pacing marks (last send, slow-down end, model resume, last reply), the no_model and mute
        switches, the banned and spare model lists and the last call's failure kind.
        """
        self.t0 = monotonic()
        self.bind = dict(bind)
        self.model_list = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
        self.world = ProxyClient(timeout=95, max_retries=1, retry_delay=1)
        WORLD_LINKS[link_key(self.bind)] = weakref.ref(self)
        self.spend = 0.0
        self.tick = 0
        self.done = False
        self.broke = False
        self.log: list[dict[str, Any]] = []
        self.last_send = 0.0
        self.slow_until = 0.0
        self.model_resume = 0.0
        self.last_reply = self.t0
        self.misses = 0
        self.no_model = False
        self.banned: set[str] = set()
        self.tool_banned: set[str] = set()
        self.spare_models: list[str] = []
        self.call_models: set[str] = set()
        self.mute = False
        self.call_fail = ""

    def spare(self) -> float:
        """Seconds left before the episode's DEADLINE, counted from the wire's creation."""
        return DEADLINE - (monotonic() - self.t0)

    def spent_out(self) -> bool:
        """True when model spend has reached SPEND_CAP, scaled by the wire's scale attribute when set."""
        return self.spend >= SPEND_CAP * self.__dict__.get("scale", 1.0)

    def send(self, client: Any, path: str, body: dict[str, Any]) -> Any:
        """Posts one body to the proxy client at the given path, pacing calls: it waits until PACE_GAP after the
        last send (SLOW_GAP while a slow-down runs) and, for inference paths, until the model resume time,
        never longer than the time left minus a second. A 429 or 503 failure starts a slow-down of SLOW_FOR
        seconds. Returns the client's reply.
        """
        gap = SLOW_GAP if monotonic() < self.slow_until else PACE_GAP
        wait = self.last_send + gap - monotonic()
        if "/inference/" in path:
            wait = max(wait, self.model_resume - monotonic())
        wait = min(wait, self.spare() - 1)
        if wait > 0:
            sleep(wait)
        self.last_send = monotonic()
        answer = client.post_verbose(path, body)
        if answer.data is None and (answer.error or {}).get("status") in (429, 503):
            self.slow_until = monotonic() + SLOW_FOR
        return answer

    def post_env(self, env_body: dict[str, Any], acts: list[dict[str, Any]]) -> Any:
        """One world request for the actions: through the _env_post seam for a single action, and for a batch when a
        wrapper replaced the seam (one action at a time, joined in the batch reply's shape); else the batch as is."""
        seam = globals().get("_env_post", OWN_ENV_POST)
        if seam is OWN_ENV_POST:
            return self.send(self.world, "/environment/call", env_body)      # (the same request as the seam would send)
        first = int(env_body.get("step") or self.tick + 1)
        observations, last = [], None
        for spot, action in enumerate(acts):
            reply = seam(self.bind, first + spot, action.get("name"), action.get("args") or {})
            if not isinstance(reply, dict) or "__error__" in reply:
                if spot == 0:
                    return WorldReply(None, (reply or {}).get("__error__") if isinstance(reply, dict) else {"status": None, "body": ""})
                break
            last = reply
            observations.append(reply.get("observation"))
            if reply.get("finished"):
                break
        if len(acts) == 1:
            return WorldReply(last)
        joined = {"observations": observations, "step": (last or {}).get("step", first + len(observations) - 1),
                  "finished": bool((last or {}).get("finished"))}
        return WorldReply(joined)

    def pause(self, seed: str, tries: int, status: Any) -> float:
        """Backoff delay for a retry: doubling per try (from 2 seconds for a 429, else 1), capped at 8, then spread
        by a factor of 0.75 to 1.25 derived from the seed text, the try count and the world step.
        """
        mix = (len(seed) * 7 + sum(ord(ch) for ch in seed[-64:]) + 31 * tries + 13 * self.tick) % 101 / 100.0
        return min(8.0, (2.0 if status == 429 else 1.0) * 2 ** tries) * (0.75 + 0.5 * mix)

    def down(self, at: float | None = None) -> bool:
        """True when model calls are off (no_model) or have failed since the last reply more than DOWN_AFTER
        seconds before the given moment (now by default).
        """
        return self.no_model or (self.misses > 0 and (monotonic() if at is None else at) - self.last_reply > DOWN_AFTER)

    def usable(self, model_list: list[str], tools: bool = False) -> bool:
        """True when model calls are allowed (not no_model, not mute) and some model of the list or the spare
        models is not banned, nor banned for tool calls when tools are wanted.
        """
        return not self.no_model and not self.mute and any(
            mdl not in self.banned and not (tools and mdl in self.tool_banned) for mdl in list(model_list) + self.spare_models)

    @staticmethod
    def plain(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Copies chat messages, joining any list content (such as cache-marked text parts) into one string with
        blank lines between the parts; used to retry a model that rejected the structured form.
        """
        out = []
        for msg in msgs:
            msg = dict(msg)
            if isinstance(msg.get("content"), list):
                msg["content"] = "\n\n".join(str(part.get("text") or "") for part in msg["content"] if isinstance(part, dict))
            out.append(msg)
        return out

    def query_model(self, msgs: list[dict[str, Any]], model_list: list[str], token_cap: int,
            toolset: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        """Chat completion (temperature 0, token cap, optional tools); reply or None.
        Tries given then spare models, skipping banned, refused and tool-banned ones.
        Sonnet is skipped at the spend cap or past SOFT_SPEND.
        MODEL_TRIES tries a model; the timeout fits the time left.
        Usage cost adds to spend; a reply with content or calls returns.
        402 or a credit 403 stops all calls.
        401/403/404 ban the model.
        400/413/422 retry once with plain messages.
        Other errors back off; a slow model passes to the next.
        A service that is down stops all calls.
        """
        self.call_fail = ""
        self.call_models = set()
        if self.mute or self.no_model:
            return None
        seed = ""
        kinds: set[str] = set()
        refused_here: set[str] = set()
        ladder = list(model_list) + [mdl for mdl in self.spare_models if mdl not in model_list]
        lone = len({mdl for mdl in ladder if mdl not in self.banned}) <= 1
        for pos, mdl in enumerate(ladder):
            if self.no_model:
                return self.failed(kinds)
            if mdl in self.banned or mdl in refused_here or (toolset and mdl in self.tool_banned):
                continue
            if "sonnet" in mdl:
                cheaper_next = pos + 1 < len(model_list) and "sonnet" not in model_list[pos + 1]
                guess = sum(len(json.dumps(m, default=str)) for m in msgs) / 2.3 * 2e-6 + min(token_cap, 1500) * 1e-5
                scale = self.__dict__.get("scale", 1.0)
                if self.spent_out() or (cheaper_next and (self.spend >= SOFT_SPEND * scale or self.spend + guess > SPEND_CAP * scale)):
                    continue
            plain = False
            for _ in range(1 if self.down() else MODEL_TRIES):
                spare = self.spare()
                if spare < LAST_CALL - 10:
                    return self.failed(kinds)
                self.model_list.timeout = int(max(8, min(30 if self.down() else 75, spare - LAST_CALL + 15)))
                envelope: dict[str, Any] = {"model": mdl, "messages": self.plain(msgs) if plain else self.cached(msgs, mdl),
                                            "temperature": 0, "max_tokens": token_cap}
                if toolset:
                    envelope["tools"] = toolset
                    envelope["tool_choice"] = "auto"
                self.call_models.add(mdl)
                started = monotonic()
                answer = self.send(self.model_list, "/inference/chat/completions", envelope)
                data = answer.data
                if isinstance(data, dict) and data.get("choices"):
                    self.last_reply, self.misses = monotonic(), 0
                    use_info = data.get("usage") or {}
                    if is_number(use_info.get("cost")):
                        self.spend += float(use_info["cost"])
                    msg = data["choices"][0].get("message") or {}
                    if msg.get("content") or msg.get("tool_calls"):
                        return msg
                    kinds.add("empty")
                    self.log.append({"kind": "inference_failure", "model": mdl, "error": "empty"})
                    break
                self.misses += 1
                self.log.append({"kind": "inference_failure", "model": mdl,
                                 "error": str(answer.error)[:300] if answer.error else "empty"})
                error = answer.error or {}
                status, body = error.get("status"), str(error.get("body") or "")
                said = body.lower()
                if status == 402 or (status == 403 and re.search(r"key limit|insufficient|credit", said)):
                    self.no_model = True
                    return self.failed(kinds)
                if status in (401, 403, 404):
                    kinds.add("hard")
                    refused_here.add(mdl)
                    if status == 404 and toolset and "tool" in said:
                        self.tool_banned.add(mdl)
                    elif not (status == 403 and re.search(r"moderat|flagged", said)):
                        self.banned.add(mdl)
                    listed = re.search(r'allowed_models"?\s*:\s*\[([^\]]*)', body) if status == 403 else None
                    if listed and not self.spare_models:
                        self.spare_models = [m for m in re.findall(r'"([^"\s]{3,100})"', listed.group(1))
                                             if m not in self.banned and m not in model_list][:2]
                        ladder += [m for m in self.spare_models if m not in ladder]
                        lone = len({m for m in ladder if m not in self.banned}) <= 1
                    break
                if status in (400, 413, 422):
                    kinds.add("hard")
                    if plain:
                        refused_here.add(mdl)
                        break
                    plain = True
                    continue
                if isinstance(status, int) and 400 <= status < 500 and status not in (408, 425, 429):
                    kinds.add("hard")
                    refused_here.add(mdl)
                    break
                kinds.add("soft")
                if monotonic() - started > 20:
                    if self.down() and (lone or len(self.call_models) >= 2):
                        return self.failed(kinds)
                    break
                if not seed:
                    seed = json.dumps(msgs[-1:], default=str)[-600:]
                wait = self.pause(seed + mdl, min(self.misses - 1, 6), status)
                if self.spare() - wait < LAST_CALL - 10:
                    return self.failed(kinds)
                self.model_resume = monotonic() + wait
                if self.down(self.model_resume) and (lone or len(self.call_models) >= 2):
                    return self.failed(kinds)
        return self.failed(kinds)

    def failed(self, kinds: set[str]) -> None:
        """Records why a model call came back empty-handed in call_fail (empty when a model returned nothing,
        refused when every failure was a hard refusal, else blank) and returns None.
        """
        self.call_fail = "empty" if "empty" in kinds else "refused" if kinds == {"hard"} else ""
        return None

    @staticmethod
    def cached(msgs: list[dict[str, Any]], mdl: str) -> list[dict[str, Any]]:
        """Marks prompt-cache points for Anthropic models: with more than two messages, the first and last
        string contents are turned into one text part carrying an ephemeral cache_control. Other models and
        short conversations get the messages unchanged; the input list is never modified.
        """
        if not mdl.startswith("anthropic/") or len(msgs) <= 2:
            return msgs
        marked = [dict(m) for m in msgs]
        for idx in (0, len(marked) - 1):
            content = marked[idx].get("content")
            if isinstance(content, str) and content:
                marked[idx]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        return marked

    def consult(self, ask_text: str, token_cap: int, model_list: list[str]) -> dict[str, Any] | None:
        """Sends one user prompt through query_model and returns the JSON object parsed from the reply text, or
        None when there is no reply or no object.
        """
        msg = self.query_model([{"role": "user", "content": ask_text}], model_list, token_cap)
        return json_object((msg or {}).get("content") or "")

    def act(self, acts: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Sends one action, or a batch of them, to the world at step tick + 1 and returns the reply data, or
        None. A dict reply updates tick (from its step, or by the number of actions) and sets done when the
        world says finished. Failures are logged. 402 sets broke; 409 either ends the episode (finished),
        resynchronises the step from an expected-step message, or moves past a reused step up to twice. 403
        and 422 end at once, 400, 401 and 404 after one retry, others after WORLD_TRIES failures with pause
        backoff, and any wait that would leave under two seconds ends the call.
        """
        env_body: dict[str, Any] = dict(self.bind)
        env_body["step"] = self.tick + 1
        if len(acts) == 1:
            env_body["action"] = acts[0]
        else:
            env_body["actions"] = acts
        tries = reused = 0
        for _ in range(3 + WORLD_TRIES):
            spare = self.spare()
            if spare < 2:
                return None
            self.world.timeout = int(max(3, min(95, spare - 1)))
            answer = self.post_env(env_body, acts)
            if isinstance(answer.data, dict):
                data = answer.data
                self.tick = data["step"] if isinstance(data.get("step"), int) else self.tick + len(acts)
                if data.get("finished"):
                    self.done = True
                return data
            error = answer.error or {}
            status_line, envelope = error.get("status"), str(error.get("body") or "")
            self.log.append({"kind": "env_failure", "status": status_line, "body": envelope[:300]})
            if status_line == 402:
                self.broke = True
                return None
            if status_line == 409:
                if "finished" in envelope:
                    self.done = True
                    return None
                expected = re.search(r"expected (\d+)", envelope)
                if not expected:
                    if "reused" in envelope and reused < 2:
                        reused += 1
                        env_body["step"] += 1
                        continue
                    return None
                self.tick = int(expected.group(1)) - 1
                env_body["step"] = self.tick + 1
                continue
            if status_line in (403, 422) or tries >= WORLD_TRIES - 1 or (status_line in (400, 401, 404) and tries >= 1):
                return None
            wait = self.pause(json.dumps(env_body, default=str)[-600:], tries, status_line)
            tries += 1
            if self.spare() - wait < 2:
                return None
            sleep(wait)
        return None


class Records:

    """Archive of everything the world returned in one episode. items holds each observation as received;
    ids collects the id-like values seen under id keys; read holds the ids whose details were seen (a
    record with a nested field map); label_of maps an id to the category, shelf or department label
    first seen with it. Its methods look up prices and records of one id, build record digests and
    listing cards for prompts, count detailed reads per shelf, and list field names and values for
    planning questions.
    """
    def __init__(self):
        """Starts an empty archive: no observations, no known ids, no read ids
        and no category labels.
        """
        self.items: list[Any] = []
        self.ids: set[str] = set()
        self.read: set[str] = set()
        self.label_of: dict[str, str] = {}

    def add(self, obs: Any) -> None:
        """Stores one observation and indexes it. Every nested dict adds its id-like values under id-named keys
        to ids. A dict's own id (an id or _id key, or a key ending in id holding an id-like value) joins
        read when the dict also holds a nested map with two or more scalar values, and gets the dict's short
        category, shelf, department or aisle label in label_of unless it already has one.
        """
        self.items.append(obs)

        def visit(d: dict[str, Any]) -> None:
            """Indexes one record dict: collects its id-like values under id-named keys, finds the record's own id,
            marks it read when a nested map carries at least two scalar fields, and stores a category-like label
            of up to 60 characters for it when none is stored yet.
            """
            for key, piece in d.items():
                if isinstance(piece, str) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, piece):
                    self.ids.add(piece)
            ident = next((piece for key, piece in d.items() if isinstance(piece, str) and str(key).lower() in ("id", "_id")
                          or (isinstance(piece, str) and str(key).lower().endswith("id") and re.fullmatch(ID_LIKE, piece))), None)
            if ident and any(isinstance(val, dict) and sum(1 for v in val.values() if isinstance(v, (str, int, float))) >= 2
                             for val in d.values()):
                self.read.add(ident)
            own = next((piece.strip() for key, piece in d.items() if isinstance(piece, str) and 0 < len(piece.strip()) <= 60
                        and re.search(r"categor|shelf|department|aisle", str(key).lower()) and "id" not in str(key).lower()), "")
            if ident and own:
                self.__dict__.setdefault("label_of", {}).setdefault(ident, own)
        walk_dicts(obs, visit)

    def price(self, thing: str) -> float | None:
        """Latest known price of a listing: scans every stored observation for dicts whose id-named key holds
        exactly this id and takes their first numeric price or cost field. Returns the last value found, or
        None when no record gives one.
        """
        found: list[float] = []

        def visit(d: dict[str, Any]) -> None:
            """Takes the first numeric price or cost value from a dict that names the wanted id under an id key;
            other dicts are passed over.
            """
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
        """Returns the stored records that mention a listing: every dict holding the id as one of its string
        values, as JSON with strings clipped to 400 characters and each record to 2500, duplicates skipped.
        The search stops at eight records and at depth six, does not look inside a matched dict, and keeps
        the last hold matches (four by default). Used to show a pick's own records to a check prompt.
        """
        matched: list[str] = []

        def seek(piece: Any, lvl: int) -> None:
            """Walks one nested value for dicts that hold the id among their string values and appends each as a
            clipped text; a matched dict is not entered further, and the walk stops past depth six or once eight
            matches exist.
            """
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
        """Newest-first JSON texts of the stored observations, with every string clipped to value_chars.
        Each text is kept while the running total stays within char_cap; a text that would pass it is
        skipped and later, shorter ones may still be taken.
        """
        record_texts, char_total = [], 0
        for rec in reversed(self.items):
            text_body = as_text(clipped(rec, value_chars))
            if char_total + len(text_body) > char_cap:
                continue
            record_texts.append(text_body)
            char_total += len(text_body)
        return record_texts

    def cards(self, char_cap: int = 30000, cheaper_than: str = "", skip: set[str] | None = None, first: Any = ()) -> list[str]:
        """Builds one text card per listing for prompts.
        Every record with a known id and 4+ keys or a map merges into its card.
        skip leaves ids out; cheaper_than keeps only cheaper listings.
        Order: given ids, read ids, then first sight; sold-out cards last.
        Read cards start with DETAILS, others with ROW.
        Cards past char_cap are skipped.
        """
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        def visit(d: dict[str, Any]) -> None:
            """Merges one dict into the card of the known id it carries, when the dict has four or more keys or a
            nested map of two or more entries; the id is added to the card order on first sight and marked read
            when such a map is present.
            """
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower()
                          and re.fullmatch(ID_LIKE, val) and val in self.ids), None)
            if not ident or (len(d) < 4 and not any(isinstance(v, dict) and len(v) >= 2 for v in d.values())):
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
            """Numeric price or cost field of a merged card, used for the cheaper_than filter;
            None when the card has none.
            """
            for key, val in (merged.get(ident) or {}).items():
                if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower()):
                    return float(val)
            return None

        cap_price = price_of(cheaper_than) if cheaper_than else None
        if skip:
            order = [i for i in order if i not in skip or i == cheaper_than]
        if cap_price is not None:
            order = [i for i in order if i == cheaper_than or price_of(i) is None or price_of(i) < cap_price]
        lines, total = [], 0
        front = set(first or ())
        ranked = sorted(order, key=lambda i: (i not in front, i not in self.read, order.index(i)))
        texts = {i: ("DETAILS " if i in self.read else "ROW ") + as_text(clipped(merged[i], 160)) for i in ranked}
        if sum(len(t) for t in texts.values()) > char_cap:
            def gone(i: str) -> bool:
                """True when any field of the merged card says the listing is out of stock."""
                return any(Errand.key_stock(key, val) is False for key, val in merged[i].items())
            ranked = sorted(ranked, key=lambda i: (i not in front, i not in self.read, gone(i), order.index(i)))
        for ident in ranked:
            text_body = texts[ident]
            if total + len(text_body) > char_cap:
                continue
            lines.append(text_body)
            total += len(text_body)
        return lines

    def label_for(self, d: dict[str, Any]) -> str:
        """Category label of a record: its own short category, shelf, department or aisle value, else the label
        stored in label_of for an id it holds under an id key; empty when neither exists.
        """
        own = next((val.strip() for key, val in d.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                    and re.search(r"categor|shelf|department|aisle", str(key).lower()) and "id" not in str(key).lower()), "")
        if own:
            return own
        labels = self.__dict__.get("label_of") or {}
        return next((labels[val] for key, val in d.items() if isinstance(val, str) and val in labels and "id" in str(key).lower()), "")

    def rich_reads(self, shelf: str = "") -> int:
        """Counts the read listings whose stored records are detailed: a dict naming the id with a nested map of
        three or more entries, or with no nested map but five or more short strings. With a shelf, only
        records whose label matches it (case aside) count.
        """
        rich: set[str] = set()

        def visit(d: dict[str, Any]) -> None:
            """Adds the read id of one dict to the rich set when the dict has a nested map of three or more entries,
            or no nested map and at least five short strings, skipping dicts from another shelf when a shelf is
            given.
            """
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.read), None)
            if not ident:
                return
            if shelf and self.label_for(d).strip().lower() != shelf.strip().lower():
                return
            maps = [val for val in d.values() if isinstance(val, dict)]
            if any(len(val) >= 3 for val in maps) or (not maps and sum(1 for val in d.values() if isinstance(val, str) and 0 < len(val) <= 70) >= 5):
                rich.add(ident)
        for rec in self.items:
            walk_dicts(rec, visit)
        return len(rich)

    def shelves_read(self) -> list[str]:
        """Category labels of the listings whose details were read (dicts with a read id, a label and a nested
        map), most frequent first. The episode uses it to pick the shelf whose fields feed the question
        menu.
        """
        tally: dict[str, int] = {}

        def visit(d: dict[str, Any]) -> None:
            """Counts one dict toward its label when it names a read id, has a category label and holds a nested
            map.
            """
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.read), None)
            shelf = self.label_for(d) if ident else ""
            if ident and shelf and any(isinstance(val, dict) for val in d.values()):
                tally[shelf] = tally.get(shelf, 0) + 1
        for rec in self.items:
            walk_dicts(rec, visit)
        return sorted(tally, key=lambda s: -tally[s])

    def prop_menu(self, shelf: str = "") -> tuple[str, int]:
        """Summarises the property fields of read listings: (menu text, records used).
        With a shelf, only records on it count.
        A record counts with five or more short fields (two on a shelf without a map).
        Keys are tallied with up to five example values each.
        Row-only keys drop once a detailed record exists.
        The menu lists the 28 most common keys for question prompts.
        """
        tallies: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        sets = 0
        mapped, row_tally = 0, {}

        def on_shelf(d: dict[str, Any]) -> bool:
            """True when the dict carries the wanted shelf as a category-like field value, or its label matches it."""
            return any(isinstance(val, str) and val.strip().lower() == shelf.lower() and re.search(r"categor|shelf|department|aisle", str(key).lower())
                       for key, val in d.items()) or self.label_for(d).strip().lower() == shelf.strip().lower()

        def seek(piece: Any, inside: bool, lvl: int) -> None:
            """Walks a nested value down to depth six, passing each dict to visit once it lies on the wanted shelf
            (or no shelf is given).
            Being on the shelf carries down to every dict nested inside.
            """
            if lvl > 6:
                return
            if isinstance(piece, dict):
                inside = inside or not shelf or on_shelf(piece)
                if inside:
                    visit(piece)
                for val in piece.values():
                    if isinstance(val, (dict, list)):
                        seek(val, inside, lvl + 1)
            elif isinstance(piece, list):
                for val in piece:
                    seek(val, inside, lvl + 1)

        def visit(d: dict[str, Any]) -> None:
            """Tallies one record's short non-junk string fields for the menu. A record with fewer than five such
            fields is skipped, unless a shelf is given and it has two or more and no nested map. Each key name
            (underscores and dots turned into spaces) is counted with up to five distinct lowercase example
            values, skipping hyphenated or underscored codes with digits; records with fewer than five fields
            and an id key also count as row-like.
            """
            nonlocal sets, mapped
            brief_vals = [(fkey, val) for fkey, val in d.items() if isinstance(val, str) and 0 < len(val) <= 70 and not junk_key(fkey)]
            if len(brief_vals) < 5 and not (shelf and len(brief_vals) >= 2 and not any(isinstance(v, dict) for v in d.values())):
                return
            sets += 1
            rowlike = len(brief_vals) < 5 and bool(record_id(d))      # (M1: an id-like value under an id key)
            mapped += 0 if rowlike else 1
            for key, piece in brief_vals:
                if re.search(r"\d", piece) and re.fullmatch(CODE_LIKE, piece) and ("-" in piece or "_" in piece):
                    continue
                call_name = re.sub(r"[_.]+", " ", str(key)).strip().lower()
                tallies[call_name] = tallies.get(call_name, 0) + 1
                if rowlike:
                    row_tally[call_name] = row_tally.get(call_name, 0) + 1
                value_counts = examples.setdefault(call_name, [])
                if piece.lower() not in value_counts and len(value_counts) < 5:
                    value_counts.append(piece.lower())

        for rec in self.items:
            seek(rec, False, 0)
        if mapped:
            for fkey in [fkey for fkey in tallies if row_tally.get(fkey, 0) >= tallies[fkey]]:
                tallies.pop(fkey)
        top_props = sorted(tallies, key=lambda fkey: -tallies[fkey])[:28]
        return "\n".join(f"- {fkey} (in {tallies[fkey]} listings): " + " | ".join(examples.get(fkey, [])) for fkey in top_props), sets

    def label_lines(self) -> str:
        """Lists the fields whose short values repeat across records, as label-like evidence for prompts. Over
        dicts of four or more keys it counts each key's string values of up to 40 characters (digit-bearing
        codes skipped) and keeps keys with four or more values, two to eight distinct and one seen at least
        twice. Each line gives the key with its values and counts, most common first; at most 12 lines.
        """
        tallies: dict[str, dict[str, int]] = {}

        def visit(d: dict[str, Any]) -> None:
            """Counts the short string values of one dict per key, skipping dicts with fewer than four keys and
            code-like values that contain digits.
            """
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


class Needs:

    """Requirement state of one episode: the first reading of the request and how the shopper's replies
    change it. reading holds the parsed request (product or several items, quantity, requirements,
    references, price cap, priority, soft wants, must-ask items, rules); written and uncovered are the
    stated and revealed needs, let_go the dropped ones, indifferent the properties the shopper does not
    mind, now_reqs a full current list once the listener model gives one, and spend_cap the budget.
    hear sends each new shopper line to the listener model and folds the result in, bumping version on
    change; text_body, overview and changes write the state into prompts.
    """
    def __init__(self, wire: Wire, request: str):
        """Starts the requirement state for a request, holding the wire for model calls: an empty reading, no
        stated, revealed, dropped or no-preference entries, no current list (now_reqs None) and no budget. It
        also sets version 0, no lines caught, no exchanges or reply due, no widening yet, sweepable off, and
        empty resolved and reference-news lists.
        """
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
        self.sweepable = False
        self.resolved: list[str] = []
        self.ref_news: list[dict[str, Any]] = []

    def items(self) -> list[dict[str, Any]]:
        """Product rows of a several-product request: each reading item with a non-empty product becomes
        {product, quantity (a positive int, else 1), requirements (strings with a colon)}. Returns the rows
        only when there are two or more, else an empty list.
        """
        rows = []
        for row in self.reading.get("items") or []:
            if isinstance(row, dict) and str(row.get("product") or "").strip():
                qty = row.get("quantity")
                reqs = row.get("requirements") if isinstance(row.get("requirements"), list) else []
                rows.append({"product": str(row["product"]).strip(), "quantity": qty if isinstance(qty, int) and not isinstance(qty, bool) and qty >= 1 else 1,
                             "requirements": [str(r) for r in reqs if r and ":" in str(r)]})
        return rows if len(rows) >= 2 else []

    def quantity(self) -> int:
        """Quantity of a single-product request: the reading's quantity when it is a positive int, else 1; always
        1 for a several-product request, whose quantities sit on its rows.
        """
        if self.items():
            return 1
        need_n = self.reading.get("quantity")
        return need_n if isinstance(need_n, int) and not isinstance(need_n, bool) and need_n >= 1 else 1

    def counts(self) -> list[int]:
        """Quantity of each product row, or a one-item list with the single quantity."""
        return [row["quantity"] for row in self.items()] or [self.quantity()]

    def items_text(self) -> str:
        """Prompt note for a several-product request: the order holds exactly one line per product at its
        quantity, each listing is judged by its own product's requirements, and the priority applies within
        each product, followed by the numbered products. Empty for a single product.
        """
        rows = self.items()
        if not rows:
            return ""
        return ("THE REQUEST ASKS FOR SEVERAL PRODUCTS; the order holds exactly one line per product, at its quantity, and nothing "
                "else; each listing is judged only against the requirements of the product it is (its category decides), and the "
                "cheapest or best rule applies within each product: " + " | ".join(
                    f"{n + 1}) {row['product']} x{row['quantity']}" for n, row in enumerate(rows)))

    def priority(self) -> str:
        """The request's ranking rule from the reading: cheapest or best, else none."""
        return self.reading.get("priority") if self.reading.get("priority") in ("cheapest", "best") else "none"

    def props_left(self) -> list[str]:
        """Property requirements still in force: the current list (or stated plus revealed), minus dropped ones
        and minus any about price, budget, quantity, stock or cents.
        """
        left = self.now_reqs if self.now_reqs is not None else [r for r in self.written + self.uncovered if r not in self.let_go]
        return [r for r in left if r not in self.let_go and not re.search(r"\bprice\b|budget|quantity|\bstock\b|\bcents?\b", r.lower())]

    def text_body(self) -> str:
        """Writes the current requirements as one line for prompts: the current list, or the stated and
        revealed needs tagged as such, minus dropped ones (with fallback text when none is left). It adds the
        budget and currency, the quantity or the several-product note, the dropped needs, the properties the
        shopper does not mind, and the tie and category notes when the episode has set them.
        """
        if self.now_reqs is not None:
            items = list(self.now_reqs)
        else:
            items = [f"{req} (stated)" for req in self.written if req not in self.let_go]
            items += [f"{req} (revealed by the shopper)" for req in self.uncovered if req not in self.let_go]
        collected = "; ".join(items) if items else (
            "no property requirement is left: every in-stock listing in the requested category fits" if self.let_go else "see the request")
        if self.spend_cap is not None:
            collected += f"; total price at most {self.spend_cap} {self.reading.get('currency') or ''}".rstrip()
        collected += f"; quantity {self.quantity()}" if not self.items() else "; " + self.items_text()
        if self.let_go:
            collected += "; no longer required: " + "; ".join(self.let_go)
        if self.indifferent:
            collected += "; shopper does not care about: " + ", ".join(self.indifferent)
        if self.__dict__.get("tie_text"):
            collected += "; " + self.tie_text
        if self.__dict__.get("cat_text"):
            collected += "; " + self.cat_text
        return collected

    def changes(self) -> str:
        """Prompt paragraph naming the requirements the shopper dropped and saying they no longer count for
        failing, ranking or preferring a listing; empty when nothing was dropped.
        """
        if not self.let_go:
            return ""
        return ("\n\nCHANGES SINCE THE REQUEST (they override its text): the shopper DROPPED " + "; ".join(self.let_go)
                + ". A dropped requirement no longer counts: never fail, rank or prefer a listing on it.")

    def overview(self, step_count: int, msgs: int | None) -> str:
        """Bullet summary of the first reading for the main prompt, empty without a reading: the product, each
        product line or the quantity, stated requirements, values given by reference, the price cap, an
        in-stock-only note, and the cheapest or best rule or a note that any fitting item will do. Then soft
        wants marked not required, must-ask items, rules, the message allowance when known and the steps
        allowed.
        """
        reading = self.reading
        if not reading:
            return ""
        lns = ["A first reading of the request (check it against the request itself):"]
        if reading.get("product"):
            lns.append(f"- product: {reading.get('product')}")
        if self.items():
            lns += [f"- product {n + 1}: {row['product']}, quantity {row['quantity']} (one cart line)" for n, row in enumerate(self.items())]
            lns.append("- the order holds exactly these lines, one per product, and nothing else; send every line in each cart call when the cart tool takes a list of lines")
        else:
            lns.append(f"- quantity: {self.quantity()}")
        lns += [f"- stated requirement: {req}" for req in reading.get("requirements") or []]
        lns += [f"- value given by reference (read that listing first, then search with its value): {req}" for req in reading.get("references") or []]
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

    @staticmethod
    def line_names(needs: list[Any], lns: list[str]) -> dict[str, str]:
        """Renames new needs to the shopper's own word forms. For each property: value need whose property words
        (three or more letters each) do not appear as such in the shopper's lines, it looks for a run of
        words in the lines that start with those property words in order and rewrites the need with that run,
        keeping any product tag and the value. Returns a map from old need to renamed need.
        """
        words = re.findall(r"[a-z]+", " ".join(lns).lower())
        joined = " ".join(words)
        out: dict[str, str] = {}
        for need in needs:
            if not isinstance(need, str) or ":" not in need:
                continue
            head, _, value = need.partition(":")
            tag = head[head.find("("):].strip() if "(" in head else ""
            parts = re.findall(r"[a-z]+", re.split(r"\s*\(", head)[0].lower())
            if not parts or any(len(p) < 3 for p in parts) or re.search(r"\b" + r"\s+".join(parts) + r"\b", joined):
                continue
            for i in range(len(words) - len(parts) + 1):
                run = words[i: i + len(parts)]
                if run != parts and all(w.startswith(p) for p, w in zip(parts, run)):
                    out[need] = (" ".join(run) + (" " + tag if tag else "")) + ":" + value
                    break
        return out

    def line_fields(self, needs: list[Any], lns: list[str], done: dict[str, str]) -> dict[str, str]:
        """Maps heard needs onto the store's own field names when the shopper named the field, not the property.
        Only 'property: value' needs whose property repeats a requirement from the request, and whose property name
        is not itself echoed in the shopper's lines, are considered. Through field_hook, a field key whose words occur
        in the lines and whose seen values include the need's value becomes the new property name (product tag kept).
        Returns {need: renamed need}; empty when no field hook is set.
        """
        hook = self.__dict__.get("field_hook")
        if not hook:
            return {}
        words = re.findall(r"[a-z]+", " ".join(lns).lower())
        joined = " " + " ".join(words) + " "
        phrases = {" ".join(words[i: i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}
        out: dict[str, str] = {}
        for need in needs:
            if not isinstance(need, str) or ":" not in need or need in done:
                continue
            head, _, value = need.partition(":")
            name = re.split(r"\s*\(", head)[0].strip()
            tag = head[head.find("("):].strip() if "(" in head else ""
            if not name or not any(same_prop(w.split(":", 1)[0], name) for w in self.written):
                continue
            if any(same_prop(name, p) for p in phrases if len(p) >= 3 and p not in LINK_WORDS):
                continue
            fields = hook(head) or {}
            for key, vals in sorted(fields.items(), key=lambda kv: -len(str(kv[0]))):
                kw = " ".join(key_words(key))
                if not kw or f" {kw} " not in joined or same_prop(kw, name):
                    continue
                if value_seen(value, " | ".join(str(v) for v in vals)):
                    out[need] = kw + (" " + tag if tag else "") + ":" + value
                    break
        return out

    def drop_named(self, old_need: str, lns: list[str]) -> bool:
        """Checks shopper lines truly released a listener-marked drop. Takes written
        requirements with same property and product tag (none: heard needs
        neither written nor fresh); accepts if none match or a fresh need on that
        property replaces it. Else it stands if a 1-3 word line phrase names the
        property, or (field_hook) a store field key whose seen values hold a
        non-reference requirement value, naming no other requirement's property
        (logged drop_by_field). False keeps it; any error = confirmed drop.
        """
        try:
            name = str(old_need).split(":", 1)[0]
            tag = re.findall(r"\(([^)]*)\)", name)

            def fits_tag(w: str) -> bool:
                """True when a requirement has the dropped need's property and, if that need carries a product tag, the same tag
                or none.
                """
                return same_prop(w.split(":", 1)[0], name) and (not tag or re.findall(r"\(([^)]*)\)", w.split(":", 1)[0]) in ([], tag))
            stated = [w for w in self.written if fits_tag(w)]
            if not stated:
                fresh = [n for n in self.__dict__.get("fresh_needs") or [] if isinstance(n, str) and ":" in n]
                stated = [w for w in self.uncovered if fits_tag(w) and w not in self.written and w not in fresh]
                if not stated:
                    return True
                if any(fits_tag(n) and n != old_need for n in fresh):
                    return True
            words = re.findall(r"[a-z]+", " ".join(lns).lower())
            phrases = {" ".join(words[i: i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}
            phrases = {p for p in phrases if len(p) >= 3 and p not in LINK_WORDS}
            if any(same_prop(w.split(":", 1)[0], p) for w in stated for p in phrases):
                return True
            hook = self.__dict__.get("field_hook")
            if hook:
                for w in stated:
                    head, _, value = w.partition(":")
                    value = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", value)
                    if ref_text(value) or not ch_alts(value):
                        continue
                    fields = hook(head) or {}
                    others = [r.split(":", 1)[0] for r in self.written + self.uncovered if isinstance(r, str) and ":" in r
                              and not same_prop(r.split(":", 1)[0], name)]
                    for p in phrases:
                        if any(strict_prop(o, p) for o in others):
                            continue
                        if any(strict_prop(key, p) and value_seen(value, " | ".join(vals)) for key, vals in fields.items()):
                            self.wire.log.append({"kind": "drop_by_field", "need": w[:200], "phrase": p})
                            return True
            return False
        except Exception:
            return True

    def literal_values(self, answer: dict[str, Any]) -> None:
        """Swaps word values of heard needs for the exact value the store's fields use.
        For each 'property: value' need without digits or a reference, comparative phrasing (CJ_COMPARE) is stripped;
        when that changes the value, or the value is not positive for the property, a field_hook field of the same
        property holding a value equal to the bare words replaces it, and the bare value joins LITERAL_VALUES.
        Swaps apply to the answer's needs and requirements_now and are logged as literal_value.
        """
        hook = self.__dict__.get("field_hook")
        if not hook:
            return
        swaps: dict[str, str] = {}
        for need in answer.get("needs") or []:
            if not isinstance(need, str) or ":" not in need:
                continue
            head, _, value = need.partition(":")
            if re.search(r"\d", value) or ref_text(value):
                continue
            bare = " ".join(re.findall(r"[a-z0-9]+", re.sub(CJ_COMPARE, " ", value.lower())))
            if not bare or (bare == " ".join(re.findall(r"[a-z0-9]+", value.lower())) and positive_value(value, head)):
                continue
            prop = re.split(r"\s*\(", head)[0].strip()
            for key, vals in (hook(head) or {}).items():
                if not strict_prop(key, prop):
                    continue
                own = next((v for v in vals if " ".join(re.findall(r"[a-z0-9]+", str(v).lower())) == bare), None)
                if own is not None:
                    LITERAL_VALUES.add(bare)
                    swaps[need] = f"{head.strip()}: {str(own).strip().lower()}"
                    break
        if not swaps:
            return
        answer["needs"] = [swaps.get(n, n) if isinstance(n, str) else n for n in answer.get("needs") or []]
        if isinstance(answer.get("requirements_now"), list):
            answer["requirements_now"] = [swaps.get(r, r) if isinstance(r, str) else r for r in answer["requirements_now"]]
        self.wire.log.append({"kind": "literal_value", "needs": list(swaps.values())[:4]})

    def pinned(self, prop_name: str, qtext: str | None) -> bool:
        """Tells whether a property the shopper called a no-preference is still pinned by a requirement of the request.
        Written requirements not let go are scanned, skipping ones whose product tag shares no kind word with the
        property's tag, ones with a non-positive value and ones story_guess marks as guessed from the story.
        A same_prop match, or a property key ending with the requirement's key (4+ chars), returns True;
        errors return False.
        """
        try:
            refs = [str(r) for r in self.reading.get("references") or []] if isinstance(self.reading.get("references"), list) else []
            refs += list(self.__dict__.get("resolved") or [])
            tag = re.findall(r"\(([^)]*)\)", str(prop_name))
            pk = prop_key(prop_name)
            for req in self.written:
                if req in self.let_go or ":" not in req:
                    continue
                head, _, value = req.partition(":")
                rtag = re.findall(r"\(([^)]*)\)", head)
                if tag and rtag and not (kind_words(tag[-1]) & kind_words(rtag[-1])):
                    continue
                if not positive_value(value, head) or story_guess(req, self.request, refs):
                    continue
                rk = prop_key(head)
                if same_prop(head, prop_name) or (len(rk) >= 4 and pk.endswith(rk)):
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def valued(needs: Any, lns: list[str]) -> bool:
        """True when, for some heard need, its property words (3+ letters) and its value words (link words aside) all
        occur in the shopper's lines, compared in lower case. hear keeps the needs of an answer the listener flagged
        as meta only when this holds.
        """
        said = " ".join(str(ln) for ln in lns).lower()
        for need in needs if isinstance(needs, list) else []:
            head, _, value = str(need).partition(":")
            names = [w for w in prop_words(head) if len(w) >= 3]
            values = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in LINK_WORDS]
            if names and values and all(w in said for w in names + values):
                return True
        return False

    def hear(self, lns: list[str], qtext: str | None) -> str:
        """One HEAR listener call: shopper lines to need changes.
        Sends new lines with the request, needs, recent exchanges and question.
        No call under CLOSING s; a lost call re-queues the lines.
        A meta-only answer echoing no value is emptied.
        New needs go to uncovered, references to ref_news, no-preference to
        indifferent, confirmed drops to let_go; a new budget is taken.
        Changes bump the version; drops add search advice.
        An unprompted question makes a reply due.
        Returns 'Meaning: ...'.
        """
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
        answer = self.wire.consult(ask_text, 2500, LISTENER)
        if not answer:
            if self.wire.mute or self.wire.misses > 0:
                self.caught.difference_update(new_lines)
                self.exchanges.pop()
                self.pending = list(self.__dict__.get("pending") or []) + [(list(new_lines), qtext)]
                self.pending_at = monotonic()
            return ""
        try:
            if answer.get("meta_only") is True and qtext and not self.valued(answer.get("needs"), new_lines):
                answer = dict(answer, needs=[], no_preference=[], dropped=[], references=[], requirements_now=None,
                              new_budget=None, unavailable=False, reply="")
                self.meta_last = qtext
                self.wire.log.append({"kind": "meta_only", "line": " / ".join(new_lines)[:200]})
        except Exception:
            pass
        try:
            renames = self.line_names(answer.get("needs") or [], new_lines)
        except Exception:
            renames = {}
        try:
            renames.update(self.line_fields(answer.get("needs") or [], new_lines, renames))
        except Exception:
            pass
        if renames:
            answer["needs"] = [renames.get(n, n) if isinstance(n, str) else n for n in answer.get("needs") or []]
            if isinstance(answer.get("requirements_now"), list):
                answer["requirements_now"] = [renames.get(r, r) if isinstance(r, str) else r for r in answer["requirements_now"]]
            self.wire.log.append({"kind": "need_renamed", "names": list(renames.values())[:4]})
        masked = new_lines
        hook = self.__dict__.get("ref_hook")
        if hook:
            try:
                masked = hook(answer, new_lines, qtext) or new_lines
            except Exception as err:
                self.wire.log.append({"kind": "ref_line_error", "error": f"{type(err).__name__}: {err}"[:300]})
                masked = new_lines
        remarks = []
        said = (so_far + " " + " ".join(self.exchanges) + " " + (qtext or "")).lower()

        def grounded(req: str) -> bool:
            """True when a requirement has no word of 3+ letters, or one of them already occurs in the requirements,
            exchanges or question so far.
            """
            words = [w for w in re.findall(r"[a-z]+", req.lower()) if len(w) >= 3]
            return not words or any(w in said for w in words)

        try:
            self.literal_values(answer)
        except Exception as err:
            self.wire.log.append({"kind": "literal_value_error", "error": f"{type(err).__name__}: {err}"[:300]})
        self.fresh_needs = [n for n in answer.get("needs") or [] if isinstance(n, str)]
        now_list = answer.get("requirements_now")
        if isinstance(now_list, list) and all(isinstance(req, str) for req in now_list):
            self.now_reqs = [req for req in now_list if req.strip() and grounded(req)]
        for new_need in answer.get("needs") or []:
            if isinstance(new_need, str) and new_need and new_need not in self.uncovered and grounded(new_need):
                self.uncovered.append(new_need)
                remarks.append(f"new requirement: {new_need}")
        try:
            infos = [r for r in answer.get("references") or [] if isinstance(r, dict)] if isinstance(answer.get("references"), list) else []
            for new_need in answer.get("needs") or []:
                if isinstance(new_need, str) and new_need in self.uncovered and (ref_text(new_need.partition(":")[2])
                                                                                  or new_need in (answer.get("ref_needs") or [])):
                    self.ref_news = list(self.__dict__.get("ref_news") or []) + [{"need": new_need, "infos": infos, "line": " ".join(new_lines)}]
        except Exception:
            pass
        held = {r.split(":", 1)[0].strip().lower() for r in (self.now_reqs if self.now_reqs is not None else self.written + self.uncovered)
                if isinstance(r, str) and ":" in r and r not in self.let_go}
        for prop_name in answer.get("no_preference") or []:
            if isinstance(prop_name, str) and prop_name and prop_name not in self.indifferent and prop_name.strip().lower() not in held:
                if self.pinned(prop_name, qtext):
                    self.pinned_heard = self.__dict__.get("pinned_heard", 0) + 1
                    continue
                self.indifferent.append(prop_name)
        widened = False
        for old_need in answer.get("dropped") or []:
            if isinstance(old_need, str) and old_need and old_need not in self.let_go \
                    and not (is_number(answer.get("new_budget"))
                             and re.search(r"\b(?:prices?|budget|costs?|spend\w*|cents?|dollars?)\b", old_need.lower())):
                if not self.drop_named(old_need, masked):
                    self.wire.log.append({"kind": "drop_kept", "need": old_need[:200]})
                    continue
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
            for need in self.uncovered:
                prop_name = need.split(":", 1)[0].strip().lower()
                kept = any(prop_name and prop_name in req.lower() for req in self.now_reqs)
                released = any(prop_name and prop_name in old.lower() for old in self.let_go)
                if prop_name and not kept and not released:
                    self.now_reqs.append(need)
            for need in self.written:
                try:
                    if need in self.let_go or need in self.now_reqs or self.drop_named(need, masked):
                        continue
                    name, tag = need.split(":", 1)[0], re.findall(r"\(([^)]*)\)", need.split(":", 1)[0])
                    if any(same_prop(name, old.split(":", 1)[0]) and re.findall(r"\(([^)]*)\)", old.split(":", 1)[0]) == tag for old in self.let_go):
                        continue
                    if not any(same_prop(name, req.split(":", 1)[0]) and re.findall(r"\(([^)]*)\)", req.split(":", 1)[0]) == tag
                               for req in self.now_reqs if isinstance(req, str)):
                        self.now_reqs.append(need)
                except Exception:
                    continue
        if widened:
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
        asked_back = qtext is None and any("?" in ln for ln in new_lines) and not answer.get("no_preference")
        self.reply_due = str(answer.get("reply") or "").strip() if asked_back else ""
        return "Meaning: " + str(answer.get("meaning") or "") + (" (" + "; ".join(remarks) + ")" if remarks else "")


class CartBook:

    """Counts cart units per listing id from calls sent and replies seen. Cart
    tool JSON schema gives the call shape: line-object array (id, count
    fields) or flat args: one id string, optional number. First good call's
    enum args set the put form; other forms are removals. Later replies
    show if a repeat put adds or replaces a count (adds). Replies or cart
    views showing ids with one quantity fix counts. Writes the '[cart
    as counted]' summary and faults vs requested products, quantities.
    """
    def __init__(self, the_tool: str | None, jschema: dict[str, Any]):
        """Starts an empty count for one cart tool, keeping its schema properties as prop_specs. Sets units (id -> count),
        put_form, adds (unknown at first), used, changed_at, viewed_at, last_view and pending (the count a repeat put
        would leave if it replaced the line).
        """
        self.the_tool = the_tool
        self.prop_specs = (jschema or {}).get("properties") or {}
        self.put_form: str | None = None
        self.units: dict[str, float] = {}
        self.adds: bool | None = None
        self.used = False
        self.changed_at = 0
        self.viewed_at = -1
        self.last_view: Any = None
        self.pending: dict[str, float] = {}

    def form(self, call_args: dict[str, Any]) -> str:
        """Renders the call's enum-valued arguments as text; equal text means the same kind of cart action."""
        return as_text({fkey: val for fkey, val in call_args.items() if isinstance(self.prop_specs.get(fkey), dict) and "enum" in self.prop_specs[fkey]})

    def line_spec(self) -> tuple[str, str, str] | None:
        """Finds an array property of the cart schema whose items have a non-enum string id field (preferring names
        containing 'id') and a numeric count field. Returns (array key, id field, count field), or None when the
        cart tool takes flat arguments.
        """
        for key, pspec in self.prop_specs.items():
            items = pspec.get("items") if isinstance(pspec, dict) and pspec.get("type") == "array" else None
            fields = (items or {}).get("properties") if isinstance(items, dict) else None
            if not isinstance(fields, dict):
                continue
            ids = [k for k, s in fields.items() if isinstance(s, dict) and s.get("type") == "string" and "enum" not in s]
            ids = [k for k in ids if "id" in k.lower()] or ids
            counts = [k for k, s in fields.items() if isinstance(s, dict) and s.get("type") in ("integer", "number")]
            if ids and counts:
                return key, ids[0], counts[0]
        return None

    def lines(self, call_args: dict[str, Any]) -> list[tuple[str, float | None]]:
        """Lists the (id, count) lines a cart call carries.
        With an array schema each row with a non-empty string id gives its stripped id and numeric count (None when
        missing); otherwise the single flat id (aim_flat) with count_arg. Empty when no id is found.
        """
        spec = self.line_spec()
        if spec and isinstance(call_args.get(spec[0]), list):
            out = []
            for row in call_args[spec[0]]:
                if isinstance(row, dict) and isinstance(row.get(spec[1]), str) and row[spec[1]].strip():
                    out.append((row[spec[1]].strip(), float(row[spec[2]]) if is_number(row.get(spec[2])) else None))
            return out
        thing = self.aim_flat(call_args)
        return [(thing, self.count_arg(call_args))] if thing else []

    def aim(self, call_args: dict[str, Any]) -> str | None:
        """The id a cart call acts on: the first line with a positive or missing count, else the first line; None when
        the call names no id.
        """
        found = self.lines(call_args)
        put = [ln for ln in found if ln[1] is None or ln[1] > 0]
        return (put or found)[0][0] if found else None

    def aim_flat(self, call_args: dict[str, Any]) -> str | None:
        """Picks the id from flat cart arguments: the first non-enum string argument that is a single token without
        spaces. None when there is none.
        """
        for key, piece in call_args.items():
            pspec = self.prop_specs.get(key) if isinstance(self.prop_specs.get(key), dict) else {}
            if isinstance(piece, str) and "enum" not in pspec and piece.strip() and " " not in piece.strip():
                return piece
        return None

    def count_arg(self, call_args: dict[str, Any]) -> float | None:
        """Reads the unit count of a cart call. For an array call it is the count of the line aim picks; for flat calls
        it is the first numeric non-enum argument, falling back to its schema default. None when nothing numeric is
        given.
        """
        if self.line_spec() and isinstance(call_args.get(self.line_spec()[0]), list):
            found = self.lines(call_args)
            put = [ln for ln in found if ln[1] is None or ln[1] > 0]
            return (put or found)[0][1] if found else None
        for key, pspec in self.prop_specs.items():
            if isinstance(pspec, dict) and pspec.get("type") in ("integer", "number") and "enum" not in pspec:
                piece = call_args.get(key, pspec.get("default"))
                if is_number(piece):
                    return float(piece)
        return None

    def is_put(self, call_args: dict[str, Any]) -> bool:
        """True when the call puts units in: no put form is known yet, or the call's enum form equals it."""
        return self.put_form is None or self.form(call_args) == self.put_form

    def held(self, thing: str) -> float:
        """Units of the given id in the count now (0 when absent)."""
        return self.units.get(thing, 0.0)

    def booked(self, call_args: dict[str, Any], obs: Any) -> str:
        """Updates the count after a cart call and returns the new summary (empty text for an error reply or no id).
        It marks the book used, and the first successful call fixes the put form. An array call replaces the whole
        count with its lines and marks the cart as not adding. A flat put adds to an existing line while adds is not
        known False (recording pending for learn), else sets the count; a removal subtracts. A reply that shows ids
        with quantities then corrects the count (partially when it shows only this id), and a removal whose reply
        looks like an empty cart drops the line.
        """
        self.used = True
        if isinstance(obs, dict) and obs.get("error"):
            return ""
        if self.put_form is None:
            self.put_form = self.form(call_args)
        if self.line_spec() and isinstance(call_args.get(self.line_spec()[0]), list):
            self.adds = False
            self.units = {}
            for thing, count in self.lines(call_args):
                if count is None or count > 0:
                    self.units[thing] = self.units.get(thing, 0.0) + (1.0 if count is None else count)
            seen = self.view_units(obs)
            if seen is not None:
                self.learn(seen)
            return self.summary()
        thing = self.aim(call_args)
        if not thing:
            return ""
        count = self.count_arg(call_args)
        count = 1.0 if count is None else count
        prior = self.held(thing)
        if self.is_put(call_args):
            if prior > 0 and self.adds is not False:
                self.units[thing] = prior + count
                self.pending = {thing: count}
            else:
                self.units[thing] = count if count > 0 else 0.0
        else:
            self.units[thing] = max(0.0, prior - count) if count > 0 else 0.0
        if self.units.get(thing, 0) <= 0:
            self.units.pop(thing, None)
        seen = self.view_units(obs)
        if seen is not None:
            self.learn(seen, partial=set(seen) <= {thing})
        elif (count <= 0 or not self.is_put(call_args)) and self.shows_empty(obs, thing):
            self.units.pop(thing, None)
            self.pending.pop(thing, None)
        return self.summary()

    def shows_empty(self, obs: Any, thing: str) -> bool:
        """True when a reply looks like an empty cart for this id: all list values empty, all numeric values zero, and
        no other counted id in its text. False when the id is not counted.
        """
        if not isinstance(obs, dict) or thing not in self.units:
            return False
        lists = [v for v in obs.values() if isinstance(v, list)]
        nums = [v for v in obs.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        text_body = as_text(obs)
        return bool(lists) and all(not v for v in lists) and bool(nums) and all(v == 0 for v in nums) \
            and not any(other in text_body for other in self.units if other != thing)

    def view_units(self, obs: Any) -> dict[str, float] | None:
        """Extracts {id: quantity} from a cart reply or view. Every dict walked (walk_dicts) that holds an id string
        (known or ID_LIKE, no spaces) and exactly one integer from 1 to 999 other than the price counts as a line.
        Returns None when no line is found, or when ids are known and none of the found ids is one of them.
        booked, viewed and learn rely on it to correct the count.
        """
        rows: dict[str, float] = {}
        known = set(self.units) | set(self.pending)

        def visit(d: dict[str, Any]) -> None:
            """Checks one dict for a cart line: collects id-like strings and integer fields between 1 and 999 (price
            excluded), narrows several integers to the single quantity-named key when exactly one exists, and adds the
            integer under a known id if present, else under the first id.
            """
            ids = [val for val in d.values() if isinstance(val, str) and (val in known or re.fullmatch(ID_LIKE, val)) and " " not in val]
            pairs = [(fkey, val) for fkey, val in d.items() if isinstance(val, int) and not isinstance(val, bool) and 0 < val < 1000
                     and str(fkey) != "price"]
            if len(pairs) > 1:
                named = [(fkey, val) for fkey, val in pairs if re.search(r"qty|quant|count|units|number|total|held|ordered", str(fkey).lower())
                         and not re.search(r"price|cost|due|sum|fee|charge", str(fkey).lower())]
                pairs = named if len(named) == 1 else pairs
            ints = [val for _, val in pairs]
            if len(ints) == 1 and ids:
                ident = next((i for i in ids if i in known), ids[0])
                rows[ident] = rows.get(ident, 0) + ints[0]
        walk_dicts(obs, visit)
        if not rows or not (set(rows) & (known or set(rows))):
            return None
        return rows

    def learn(self, seen: dict[str, float], partial: bool = False) -> None:
        """Adopts the counts seen in a cart reply. Each pending repeat put is compared first: a seen count equal to the
        replacing value (while the count differs) means puts set the count (adds False), a larger one means they add
        (adds True). The count then becomes the seen positive lines, merged into the current count when partial.
        """
        for thing, expected_if_set in self.pending.items():
            if thing in seen:
                if abs(seen[thing] - expected_if_set) < 1e-9 and abs(self.units.get(thing, 0) - expected_if_set) > 1e-9:
                    self.adds = False
                elif seen[thing] > expected_if_set:
                    self.adds = True
        self.pending = {}
        if partial:
            self.units.update(seen)
            self.units = {fkey: val for fkey, val in self.units.items() if val > 0}
            return
        self.units = {fkey: val for fkey, val in seen.items() if val > 0}

    def viewed(self, obs: Any, tick: int) -> None:
        """Records a cart view and its tick, learns the counts it shows, and clears the count when the view is a dict
        or list that mentions none of the counted ids.
        """
        self.last_view = obs
        self.viewed_at = tick
        seen = self.view_units(obs)
        if seen is not None:
            self.learn(seen)
        elif isinstance(obs, (dict, list)) and not self.ids_in(obs):
            self.units = {}

    def ids_in(self, obs: Any) -> bool:
        """True when the text of a reply mentions any counted id."""
        text_body = as_text(obs)
        return any(thing in text_body for thing in self.units)

    def summary(self) -> str:
        """Formats the count as a '[cart as counted]' line of 'id xN' entries, or says the cart is empty.
        Whole counts print without decimals.
        """
        if not self.units:
            return "[cart as counted] empty."
        lns = ", ".join(f"{thing} x{int(qty_n) if float(qty_n).is_integer() else qty_n}" for thing, qty_n in self.units.items())
        return f"[cart as counted] {lns}."

    def faults(self, need_n: int, counts: list[int] | None = None) -> list[str]:
        """Lists problems of the counted cart against the request, as instructions for the model. With quantities for
        several products it requires one line per product and matching sorted unit counts. For a single product it
        flags extra lines and a lone line whose units differ from need_n, saying how many to take out or put in.
        Empty when the count is empty.
        """
        if not self.units:
            return []
        if counts and len(counts) > 1:
            held = sorted(int(q) for q in self.units.values())
            if len(self.units) != len(counts):
                return [f"the cart holds {len(self.units)} line(s) ({', '.join(f'{i} x{int(q)}' for i, q in self.units.items())}) but "
                        f"the request asks for {len(counts)} products, one line each; every product needs its line and nothing else"]
            if held != sorted(counts):
                return [f"the lines hold {held} units but the products need {sorted(counts)} (one line per product at its quantity)"]
            return []
        collected = []
        if len(self.units) > 1:
            collected.append("the cart holds more than one line (" + ", ".join(f"{i} x{int(qty_n)}" for i, qty_n in self.units.items()) + "); take out every line but the chosen one")
        for thing, qty_n in self.units.items():
            if len(self.units) == 1 and abs(qty_n - need_n) > 1e-9:
                change = "take out" if qty_n > need_n else "put in"
                collected.append(f"the line {thing} holds {int(qty_n)} units but the request needs {need_n}; {change} {int(abs(qty_n - need_n))} with the cart tool")
        return collected


class Errand:
    """One shopping episode: policy view to order or decline. Built from env
    binding and policy view: request, tool definitions with parameter
    schemas, step cap. Owns Wire (model/world calls), Records
    (observations), Needs (requirements), CartBook (cart), role-to-tool
    map. State: messages sent/cap, order/refusal counts, passed/refused
    listings, questions log, model chat, references, product ties, basket
    statement. Methods read request, plan questions, run code-driven
    searches, reads, cart/decline steps, screen/dispatch model tool calls.
    """
    def __init__(self, pdata: dict[str, Any]):
        """Sets up an episode from the payload's environment: a Wire on its binding, the policy view, the request
        (query), the tools that have a function entry with their parameter schemas, the step cap (max_steps,
        default 30) and the finished flag. Creates Records and Needs (with ref_fix and store_fields as its reference
        and field hooks), an empty role map and an empty CartBook, and starts the counters and sets for messages,
        orders, refusals, checks, questions, ties, the basket statement and go-ahead rules.
        Clears LITERAL_VALUES left from an earlier episode.
        """
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
        self.needs.ref_hook = self.ref_fix
        self.role_map: dict[str, str] = {}
        self.trolley = CartBook(None, {})
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
        self.passed: set[str] = set()
        self.refused_lines: set[str] = set()
        self.refusals_seen: dict[str, int] = {}
        self.warned: set[str] = set()
        self.word_seen: list[tuple] = []
        self.word_kind = ""
        self.lead_seen: list[tuple] = []
        self.lead_bounds: dict[tuple, list] = {}
        self.lead_words = 0
        self.lead_ctx: tuple | None = None
        self.last_calls: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.stuck = 0
        self.refs: list[dict[str, Any]] = []
        LITERAL_VALUES.clear()
        self.needs.field_hook = self.store_fields
        self.stated_sig: tuple | None = None
        self.stated_text = ""
        self.statements = 0
        self.ties: list[dict[str, Any]] = []
        self.go_rules: list[str] = []               # the request's rules that ask for a told basket or a go-ahead
        self.go_key: tuple | None = None
        self.last_obs: Any = None

    @property
    def log(self) -> list[dict[str, Any]]:
        """Event log of the episode, kept on the Wire."""
        return self.wire.log

    def moves_left(self) -> int:
        """Steps still allowed: the step cap minus the steps the Wire has counted."""
        return self.step_cap - self.wire.tick

    def role(self, call_name: str) -> str | None:
        """Tool name mapped to a role (search, details, message, cart, order, decline and others), or None."""
        return self.role_map.get(call_name)

    def msgs_total(self) -> int:
        """Message allowance: the learned cap, else ASK_DEFAULT + 1 when the basket is stated before orders, else
        ASK_OPEN.
        """
        return self.msg_cap if self.msg_cap is not None else (ASK_DEFAULT + 1 if self.states_basket() else ASK_OPEN)

    def msgs_left(self) -> int:
        """Messages still available for the shopper, never below zero."""
        return max(0, self.msgs_total() - self.msgs_sent)

    def states_basket(self) -> bool:
        """The basket is stated to the shopper before each order only when a rule of the request asks for it (to approve
        the basket or the order, or to be told the basket or the total first) and the store has a message, a cart and an
        order tool; otherwise every message goes to questions."""
        return bool(self.go_rules) and bool(self.role("message") and self.role("cart") and self.role("order"))

    def go_ahead_rules(self, reading: dict[str, Any]) -> list[str]:
        """The request's rules that ask for the basket statement: one short model call over the rules read, joined with
        the word test (which alone decides when the call is lost, and reads the request's sentences when no model read
        the request at all)."""
        rules = [" ".join(str(r).split()) for r in reading.get("rules") or [] if str(r or "").strip()]
        if not rules and reading.get("code_reading"):
            rules = [s.strip() for s in re.split(r"(?<=[.!?;])\s+", self.request or "") if s.strip()]
        able = bool(self.role("message") and self.role("cart") and self.role("order"))
        key = (tuple(rules), able)
        if key == self.go_key:
            return self.go_rules
        self.go_key = key
        if not able:
            return []
        found = [r for r in rules if go_words(r)]
        if rules and not reading.get("code_reading"):
            listed = "\n".join(f"{n}. {r}" for n, r in enumerate(rules))
            try:
                answer = self.wire.consult(GO_STEP + "\n\nRULES:\n" + listed, 200, LISTENER)
            except Exception:
                answer = None
            picked = (answer or {}).get("steps") if isinstance(answer, dict) else None
            if isinstance(picked, list):
                found = []          # (the model read the rules: its picks decide; the word test is for a lost call)
            for n in picked if isinstance(picked, list) else []:
                if isinstance(n, int) and not isinstance(n, bool) and 0 <= n < len(rules) and rules[n] not in found:
                    found.append(rules[n])
            self.log.append({"kind": "go_rules", "rules": len(rules), "model": picked, "found": found})
        return found

    def spare_due(self) -> bool:
        """True when shopper-revealed needs are active (not let go) and, on a multi-product request, every product has
        at least one, matched by the need's product tag. ask_limit then holds one more message back from
        questions.
        """
        found = [n for n in self.needs.uncovered if n not in self.needs.let_go]
        if not found:
            return False
        rows = self.needs.items()
        if not rows:
            return True
        got = {self.product_of(str(n).split(":", 1)[0]) for n in found}
        return all(k in got for k in range(len(rows)))

    def shelf_has(self, k: int | None, prop: Any) -> bool:
        """Tells whether listings on a product's home shelves carry a field matching a property's name. Field keys of
        the rows (ch_rows) on those shelves are compared by words: a key whose words contain, or are contained in,
        the property's words is a match. Too little evidence (no home shelf, under 4 rows with fields, no keys) or
        an error counts as True; a miss is logged as shelf_gate and returns False.
        """
        try:
            name = re.split(r"\s*\(", str(prop or ""))[0].strip()
            want = set(key_words(name)) - {"type", "the", "of"}
            if not want:
                return True
            homes = self.home_shelves()
            home = set(homes[k if k is not None else 0]) if homes and (k or 0) < len(homes) else set()
            if not home:
                return True
            keys: set[str] = set()
            seen_n = 0
            for row in self.ch_rows().values():
                if row["fields"] and row["shelf"] in home:
                    seen_n += 1
                    keys |= {" ".join(key_words(f)) for f in row["fields"]}
            if not keys or seen_n < 4:
                return True
            for key in keys:
                have = set(key.split())
                if have and (want <= have or have <= want):
                    return True
            self.log.append({"kind": "shelf_gate", "product": k, "property": name[:60]})
            return False
        except Exception:
            return True

    def rank_props(self, k: int | None) -> list[str]:
        """Ranks the field keys that best tell product k's listings apart, for question planning. Up to 10 rows with
        fields from the product's home shelves (at least 2 needed) are ordered by rank_keys; catch-all and odd keys,
        keys settled by requirements (settled_keys) and keys sharing a word with a property still required for that
        product are left out. Returns the remaining key names.
        """
        homes = self.home_shelves()
        home = set(homes[k if k is not None else 0]) if homes and (k or 0) < len(homes) else set()
        if not home:
            return []
        rows = [r for r in self.ch_rows().values() if r["fields"] and r["shelf"] in home]
        if len(rows) < 2:
            return []
        several = bool(self.needs.items())
        stated: set[str] = set()
        for req in self.needs.props_left():
            head = str(req).partition(":")[0]
            tags = re.findall(r"\(([^)]*)\)", head)
            if several and (not tags or self.product_of(f"x ({tags[-1]})") != k):
                continue
            stated |= set(key_words(re.split(r"\s*\(", head)[0]))
        stated -= set(LINK_WORDS) | {"type"}
        out = []
        settled = self.settled_keys(k, rows[:10])
        for name in rank_keys([r["fields"] for r in rows[:10]]):
            if catch_all_key(name) or (stated and set(name.split()) & stated) or odd_key(name) or name in settled:
                continue
            out.append(name)
        return out

    def ranked_note(self, ok: Any = None, only: int | None = None) -> str:
        """The RANKED list for the planner: per product (only product `only` when given), its first eight open properties
        by rank_props; `ok(label)` (the planner's freshness test) drops asked and settled ones."""
        rows = self.needs.items()
        lines = []
        for k in ([only] if only is not None else (range(len(rows)) if rows else [None])):
            product = rows[k]["product"] if rows and k is not None else str(self.needs.reading.get("product") or "item")
            keys = []
            for name in self.rank_props(k):
                label = f"{name} ({product})" if rows else name
                if ok is not None and not ok(label):
                    continue
                if ok is None and (not self.plan_open(k, label) or self.asked_for(k, label)):
                    continue
                keys.append(name)
                if len(keys) >= 8:
                    break
            if keys:
                lines.append(f"- {product}: " + ", ".join(keys))
        if not lines:
            return ""
        return ("\n\nRANKED (for each product, its open properties on which the store's listings read so far differ most, a "
                "property on more listings first; when no concrete detail of the story points at a property of the product, "
                "ask the one that most defines this kind of item, the property a shopper buying it is most likely to insist on, "
                "not simply the first of its list):\n" + "\n".join(lines))

    def ask_limit(self) -> int:
        """Messages that may go to questions: the whole cap when no basket statement is needed; otherwise the total
        minus one for the statement and one more when spare_due holds.
        """
        if not self.states_basket():
            return self.msg_cap if self.msg_cap is not None else ASK_OPEN
        return max(0, self.msgs_total() - 1 - (1 if self.spare_due() else 0))

    def call_kind(self, call_name: str, call_args: dict[str, Any] | None) -> str | None:
        """Classifies a call as 'order' or 'decline', or None for any other tool. When one tool fills both roles, a
        short string argument (40 chars or less) matching DECLINE_WORDS or ORDER_WORDS decides, checked in argument
        order; the default is 'order'.
        """
        order, decline = self.role("order"), self.role("decline")
        if call_name not in (order, decline) or not call_name:
            return None
        if order != decline:
            return "order" if call_name == order else "decline"
        for piece in (call_args or {}).values():
            if isinstance(piece, str) and len(piece) <= 40:
                if re.search(DECLINE_WORDS, piece.lower()):
                    return "decline"
                if re.search(ORDER_WORDS, piece.lower()):
                    return "order"
        return "order"

    def finishers(self) -> set[str]:
        """Names of the mapped order and decline tools, the calls that close the episode."""
        return {self.role("order"), self.role("decline")} - {None}

    def clock_text(self) -> str:
        """Wall-clock warning for the status line: empty while more than CLOCK_FROM seconds remain before the last
        call; then the seconds left rounded down to CLOCK_STEP, or 'under' one step.
        """
        left = max(0.0, self.wire.spare() - LAST_CALL)
        if left > CLOCK_FROM:
            return ""
        step = int(left // CLOCK_STEP) * CLOCK_STEP
        return f"about {step} s left" if step > 0 else f"under {CLOCK_STEP} s left"

    def status_line(self) -> str:
        """Builds the status text the model sees each turn: steps used and left, messages sent (noting the one kept
        for the basket statement), a clock warning, the current requirements with reference, product-step and tie
        lines, whether the stated basket still matches the cart, the tools usable now when some are withheld, and
        the counted cart. It refreshes needs.cat_text on the way. As steps or model budget run short it appends one
        urgency instruction: last steps, finish now, model budget used, or the order deadline (order_due) near or
        reached.
        """
        spare = self.moves_left()
        pieces = [f"[status] steps used {self.wire.tick} of {self.step_cap}, {spare} left",
                 f"messages sent {self.msgs_sent}" + (f" of {self.msg_cap}" if self.msg_cap is not None else "")
                 + (" (one kept for the basket statement the code sends before the order)" if self.states_basket() and self.msgs_left() > 0 else "")]
        clock = self.clock_text()
        if clock:
            pieces.append(clock)
        try:
            self.needs.cat_text = self.cat_text()
        except Exception as err:
            self.log.append({"kind": "cat_text_error", "error": f"{type(err).__name__}: {err}"[:300]})
        series = ""
        if self.ties:
            try:
                self.needs.tie_text = self.tie_text()      # (CS: the wording follows the part series seen)
                raw = self.series_line()
                if raw != self.__dict__.get("series_last"):
                    series = raw or ("\n[part series] no seen main item with a part series is in stock now."
                                     if self.__dict__.get("series_last") else "")      # (CSc / CSd: once, on the change)
                    self.series_last = raw      # (CSb: sent again only when it changed)
            except Exception as err:
                self.log.append({"kind": "series_error", "error": f"{type(err).__name__}: {err}"[:300]})
        text_body = "; ".join(pieces) + ".\n[requirements now] " + self.needs.text_body() + self.ref_line() + self.product_steps()
        tie_now = str(self.needs.__dict__.get("tie_text") or "") if self.ties else ""
        text_body += (f"\n[ties] {tie_now}." if tie_now else "") + series
        if self.stated_sig is not None:
            text_body += "\n[basket stated to the shopper] " + (
                "it matches the cart." if self.cart_sig() == self.stated_sig else
                "the cart changed since: " + ("the code states the new basket before the next order." if self.msgs_left() > 0
                                              else "no message is left for a new statement; when the cart holds the best fit, order it anyway (never decline for that)."))
        usable = [tdef["function"]["name"] for tdef in self.offered()]
        if len(usable) < len(self.toolset):
            text_body += "\n[tools usable now] " + ", ".join(usable) + " (any other tool call is refused without using a step" + (
                "; questions to the shopper are planned and sent for you" if self.asking else "") + ")."
        if self.trolley.used:
            text_body += "\n" + self.trolley.summary()
        if spare <= (3 if self.states_basket() else 2):
            text_body += (" LAST STEPS: order the cart now if it holds a candidate for " + ("every product asked for" if self.needs.items()
                          else "the item") + "; otherwise put the best candidate in (a verified fit, else the cheapest in-stock, "
                          "in-budget listing whose record shows no requirement failing) and order. Decline only when the records "
                          "show that nothing can fit.")
        elif spare <= (6 if self.states_basket() else 5) or self.wire.spare() < CLOSING:
            text_body += (" FINISH NOW: fix the cart to " + ("one line per product asked for" if self.needs.items() else "one line")
                          + " (a verified fit, else the cheapest in-stock, in-budget listing whose record shows no requirement "
                          "failing; a sold-out line out), re-check its stock, then order. Decline only when the records show that "
                          "nothing can fit.")
        elif self.wire.spent_out():
            several = bool(self.needs.items())
            text_body += (" MODEL BUDGET USED: stop exploring; within the next few steps put the best candidate of "
                          + ("every product asked for (one line each, at its quantity)" if several else "the item asked for")
                          + " in the cart (a verified fit where one exists, else the closest fit the records show), re-check its "
                          "stock and price, and order. A decline is right only when the records show that "
                          + ("some product has" if several else "the store has") + " no listing that can meet its requirements.")
        elif self.orders_tried == 0 and spare <= self.order_due() + 7 and spare > self.order_due() and self.narrowing():
            text_body += f" Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when {self.order_due()} steps remain, and a refused order needs about 4 steps to repair."
        elif self.orders_tried == 0 and spare <= self.order_due() and self.narrowing():
            text_body += " Searching has ended (details only for the cart's line, listings a check named, or, while some product has no verified fit, up to 3 more reads or searches in all): put your best verified fit in the cart, re-check its stock and price, and order; a refused order still needs about 4 steps to repair."
        return text_body

    def status_tool(self) -> str | None:
        """Finds a tool that reports the message allowance. The mapped message_status role wins when it is not also the
        message, cart, order or decline tool. Otherwise the first unmapped tool with no required arguments whose
        name or description speaks of messages, chats or questions and also of remaining, used, limit or quota
        is picked. The guess is cached in status_guess.
        """
        named = self.role("message_status")
        if named and named not in {self.role(r) for r in ("message", "cart", "order", "decline")}:
            return named
        if "status_guess" not in self.__dict__:
            pick = None
            taken = {self.role(r) for r in ("search", "details", "view", "message", "cart", "cart_view", "order", "decline")} - {None}
            for tdef in self.toolset:
                fn = tdef["function"]
                name = fn["name"]
                needed = (self.shapes.get(name) or {}).get("required")
                if name in taken or (isinstance(needed, list) and needed):
                    continue
                said = (re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ") + " " + str(fn.get("description") or "")).lower()
                if re.search(r"\bmessag\w*|\bchats?\b|\bquestions?\b", said) and re.search(
                        r"\bleft\b|\bremain\w*|\bused\b|\ballowance\b|\busage\b|\bbudget\b|\blimit\w*|\bquota\b|\bhow many\b", said):
                    pick = name
                    break
            self.status_guess = pick
        return self.status_guess

    def learn_msgs(self, obs: Any, status: bool) -> None:
        """Learns message counts from a reply. Integer fields are read as left, used or cap by their key words; outside
        a status reply only keys about messages, questions, chats or asks count. A higher used count raises
        msgs_sent; the cap comes from a cap field or used plus left, and is kept when it lies between 1 and 50 and
        is not below the messages sent (logged as message_budget).
        """
        found: dict[str, int] = {}
        named: dict[str, int] = {}

        def visit(d: dict[str, Any]) -> None:
            """Scans one dict for non-negative integer fields and files each under left, used or cap by its key words
            (left or remain; used, sent, spent or asked; cap, limit, budget, max, allow, total or quota), keeping the
            first value per slot. Without status, keys must mention messages, questions, chats or asks.
            """
            for key, val in d.items():
                if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                    continue
                name = " ".join(key_words(key))
                spoken = bool(re.search(r"messag|question|chat|\bask|\bqa\b", name))
                if not status and not spoken:
                    continue
                if not spoken and re.search(r"step|turn|action|call|move|page|request|token|second|minute|item|result"
                                            r"|budget|price|cost|spend|money|cent|dollar|amount", name):
                    continue
                slot = ("left" if re.search(r"left|remain", name) else "used" if re.search(r"used|sent|spent|asked", name)
                        else "cap" if re.search(r"cap|limit|budget|max|allow|total|quota", name) else "")
                if slot:
                    found.setdefault(slot, val)
                    if spoken:
                        named.setdefault(slot, val)
        walk_dicts(obs, visit)
        merged = dict(named)
        if "cap" not in named and "left" not in named:
            merged.update({slot: found[slot] for slot in ("cap", "left") if slot in found})
        if "used" not in merged and "used" in found:
            merged["used"] = found["used"]
        found = merged
        if "used" in found and found["used"] > self.msgs_sent:
            self.msgs_sent = found["used"]
        cap = found.get("cap")
        if cap is None and "left" in found:
            cap = found.get("used", self.msgs_sent) + found["left"]
        if is_number(cap) and 0 < cap <= 50 and cap >= self.msgs_sent and cap != self.msg_cap:
            self.msg_cap = int(cap)
            self.log.append({"kind": "message_budget", "cap": self.msg_cap, "seen": found})

    def msg_probe(self) -> bool:
        """Calls the message-status tool once per episode, as a scripted step, to learn the message cap. It runs only
        when such a tool and a message tool exist, the cap is unknown, at least 12 steps remain and empty arguments
        pass the tool's schema. Returns True when the call was sent.
        """
        if self.__dict__.get("msg_probed") or self.wire.done or self.wire.broke:
            return False
        self.msg_probed = True
        tool = self.status_tool()
        if not tool or not self.role("message") or self.msg_cap is not None or self.moves_left() < 12:
            return False
        call_args, faults = ArgCheck.execute({}, self.shapes.get(tool) or {"type": "object", "properties": {}})
        if faults:
            return False
        proposal = {"id": f"status_{self.wire.tick}", "type": "function", "function": {"name": tool, "arguments": json.dumps(call_args)}}
        self.chat.append({"role": "assistant", "content": "I check how many messages to the shopper are allowed.", "tool_calls": [proposal]})
        self.post_replies(self.dispatch([proposal], scripted=True))
        return True

    def cart_sig(self) -> tuple:
        """Signature of the counted cart: sorted (id, rounded units) pairs for lines above zero."""
        return tuple(sorted((str(i), int(round(q))) for i, q in self.trolley.units.items() if q > 0))

    def basket_valid(self) -> bool:
        """True when the counted cart can be stated and ordered: it has lines, none of them is a dead line, and its
        total at recorded prices stays within the spending cap. No cap, or a line without a recorded price, counts
        as valid. stated_lock requires it.
        """
        if not self.trolley.units or set(self.trolley.units) & self.dead_lines():
            return False
        cap = self.needs.spend_cap
        if not is_number(cap):
            return True
        total = 0.0
        for ident, qty in self.trolley.units.items():
            price = self.archive.price(ident)
            if not is_number(price):
                return True
            total += float(price) * qty
        return total <= float(cap) + 1e-9

    def cart_over_cap(self) -> tuple[float, float] | None:
        """Returns (total, budget) when the counted cart costs more than the spending cap, summing recorded price times
        units per line. None when the cart is empty, no cap is known, a line has no recorded price, or the total
        fits the cap.
        """
        cap = self.needs.spend_cap
        if not self.trolley.units or not is_number(cap):
            return None
        total = 0.0
        for ident, qty in self.trolley.units.items():
            price = self.archive.price(ident)
            if not is_number(price):
                return None
            total += float(price) * qty
        return (total, float(cap)) if total > float(cap) + 1e-9 else None

    def stated_lock(self) -> bool:
        """True when a basket statement is required and was sent, no message is left, the cart still matches the
        stated basket and that basket is valid.
        """
        return (self.states_basket() and self.stated_sig is not None and self.msgs_left() <= 0
                and self.cart_sig() == self.stated_sig and self.basket_valid())

    def statement_due(self) -> bool:
        """True when the basket must be stated before ordering: statements are required, the cart has lines that differ
        from the last stated basket, a message is left, 2+ steps remain and the episode is still live.
        """
        return (self.states_basket() and bool(self.trolley.units) and self.cart_sig() != self.stated_sig
                and self.msgs_left() > 0 and self.moves_left() >= 2 and not self.wire.done and not self.wire.broke)

    def statement_text(self) -> str:
        """Writes the message that states the basket to the shopper before an order. Each counted line gives its units,
        a shortened title from the listing rows and its id, with the unit price and the request's currency when
        known; the total follows when every price is known, with a note on the budget, and the text asks whether to
        place the order. Titles are cut to 80, then 40, then no characters until the text fits the message
        argument's maxLength (default 1000), else it is truncated. Empty when the message tool has no text argument
        or the cart is empty.
        """
        the_tool = self.role("message")
        aim = self.text_arg(the_tool) if the_tool else None
        if not aim or not self.trolley.units:
            return ""
        spec = ((self.shapes.get(the_tool) or {}).get("properties") or {}).get(aim) or {}
        cap_len = int(spec["maxLength"]) if is_number(spec.get("maxLength")) else 1000
        rows = self.ch_rows()
        unit = " ".join(str(self.needs.reading.get("currency") or "").split())
        unit = f" {unit}" if unit and len(unit) <= 20 else ""

        def fig(x: Any) -> str:
            """Formats an amount as a whole number or with two decimals."""
            return f"{int(x)}" if float(x).is_integer() else f"{float(x):.2f}"
        lines, total, priced = [], 0.0, True
        for ident, qty in self.trolley.units.items():
            title = " ".join(str((rows.get(ident) or {}).get("title") or "").split())
            price = self.archive.price(ident)
            lines.append((ident, int(round(qty)), title, price))
            if is_number(price):
                total += float(price) * qty
            else:
                priced = False
        cap = self.needs.spend_cap
        text_body = ""
        for room in (80, 40, 0):
            parts = []
            for ident, qty, title, price in lines:
                name = (title if len(title) <= room else title[:room].rsplit(" ", 1)[0] + "...") if room and title else ""
                piece = f"{qty} x {name} ({ident})" if name else f"{qty} x {ident}"
                parts.append(piece + (f" at {fig(price)}{unit} each" if is_number(price) else ""))
            text_body = "Here is the basket I am about to order for you: " + "; ".join(parts) + "."
            if priced:
                text_body += f" Total: {fig(total)}{unit}"
                if is_number(cap):
                    text_body += (f", within your budget of {fig(cap)}{unit}" if total <= float(cap) + 1e-9
                                  else f" (your budget is {fig(cap)}{unit})")
                text_body += "."
            text_body += " Shall I place the order?"
            if len(text_body) <= cap_len:
                return text_body
        return text_body[:cap_len]

    def news_mark(self) -> tuple:
        """Snapshot of what an order depends on: requirement version, budget, dead lines and the cart signature."""
        return (self.needs.version, str(self.needs.spend_cap), tuple(sorted(self.dead_lines())), self.cart_sig())

    def view_cart(self) -> str:
        """Sends the separate cart-view tool as a code step when the cart changed since the last view (and the view
        tool is not the cart tool), at least 3 steps remain and empty arguments pass its schema. Returns the reply
        text, or empty text when no view was sent.
        """
        tool = self.role("cart_view")
        if not tool or tool == self.role("cart") or self.trolley.viewed_at >= self.trolley.changed_at or self.moves_left() < 3:
            return ""
        call_args, faults = ArgCheck.execute({}, self.shapes.get(tool) or {"type": "object", "properties": {}})
        if faults:
            return ""
        responses = self.send_chunk([(f"view_{self.wire.tick}", tool, call_args)])
        return responses[0][1] if responses else ""

    def state_gate(self, cid: str | None, finishing: bool = False) -> str:
        """Runs before an order call to state the basket when a statement is due. With two or more lines it first views
        the cart; if that changed the requirements, budget, dead lines or cart, it returns a 'Not run yet' note asking
        to fix the cart (unless finishing). It then sends the statement; when the shopper's reply changed something
        it returns a note to apply the change. Otherwise it keeps the statement note for the call id and returns
        empty text so the order goes ahead.
        """
        if not self.statement_due():
            return ""
        before = self.news_mark()
        if len(self.trolley.units) >= 2:
            seen = self.view_cart()
            if seen and self.news_mark() != before and not finishing:
                return ("Not run yet (one step used, for a cart view before the basket statement). The cart view or a line it "
                        "carried changed what the order depends on: " + seen[:900] + "\n" + self.trolley.summary() + " Make the "
                        "cart hold exactly the lines to buy (one per product, at its quantity), then call the order tool again.")
            if not self.statement_due():
                return ""
        sent, note = self.send_statement()
        if not sent:
            return ""
        said = self.stated_say()
        if note and not finishing:
            return ("Not run yet (one step used, for the basket statement). " + said + " The shopper's reply carries a change: "
                    + note + " Apply it: if the cart must change, change it (the code states the new basket before the next order "
                    "while a message is left); else call the order tool again.")
        if cid:
            self.stmt_note = (cid, said)
        return ""

    def stated_say(self) -> str:
        """Code note telling the model that the basket was stated before this order, with the message number and text."""
        return (f"[code] Before this order the code stated the basket to the shopper (message {self.msgs_sent}"
                + (f" of {self.msg_cap}" if self.msg_cap is not None else "") + f"): \"{self.stated_text}\"")

    def send_statement(self) -> tuple[bool, str]:
        """Sends the basket statement through the message tool as a code step, with the stating flag set during the
        send. On success it records the cart signature and text as stated, counts and logs the statement; a failed
        or error reply returns (False, ''). When the shopper's reply changed the requirement version, budget or dead
        lines, it returns (True, the reply's note) cut to 900 characters, else (True, '').
        """
        text_body = self.statement_text()
        proposal = self.message_call(text_body, f"state_{self.wire.tick}") if text_body else None
        if not proposal:
            return False, ""
        sig = self.cart_sig()
        before = (self.needs.version, str(self.needs.spend_cap), tuple(sorted(self.dead_lines())))
        fn = proposal["function"]
        self.stating, self.last_obs = True, None
        try:
            responses = self.send_chunk([(proposal["id"], fn["name"], json.loads(fn["arguments"]))])
        finally:
            self.stating = False
        obs = self.last_obs
        if not responses or obs is None or (isinstance(obs, dict) and obs.get("error")):
            self.log.append({"kind": "statement_failed", "reply": responses[0][1][:300] if responses else ""})
            return False, ""
        self.stated_sig, self.stated_text = sig, text_body
        self.statements += 1
        self.log.append({"kind": "statement", "cart": dict(self.trolley.units), "text": text_body})
        if (self.needs.version, str(self.needs.spend_cap), tuple(sorted(self.dead_lines()))) != before:
            reply = responses[0][1]
            return True, (reply.split("\n[note] ", 1)[1] if "\n[note] " in reply else reply)[:900]
        return True, ""

    def tie_register(self, reading: dict[str, Any]) -> None:
        """Registers product ties from the request reading: a product that must be listed as working with another
        (main) product. A tie with known, different product and main indexes becomes a tie record holding both
        indexes and names and its model code when code-like. A tie with a code but no main becomes a 'fits: code'
        requirement on its product unless a requirement already names the code, and the requirement version is
        bumped. Written requirements of a tied product that merely restate the tie are removed from the needs and the
        reading's items. Sets needs.tie_text and logs the ties.
        """
        self.ties = []
        rows = self.needs.items()
        raw = reading.get("ties") if isinstance(reading.get("ties"), list) else []
        extra: list[str] = []
        for tie in raw:
            if not isinstance(tie, dict):
                continue
            k = self.product_of(f"x ({tie.get('product')})") if rows else None
            m = self.product_of(f"x ({tie.get('main')})") if rows else None
            code = " ".join(str(tie.get("code") or "").split())[:60]
            if m is None and code_like(code) and (k is not None or not rows):
                req = f"fits ({rows[k]['product']}): {code}" if rows else f"fits: {code}"
                if not any(names_code(r, code) for r in self.needs.written) and req not in extra:
                    extra.append(req)
                continue
            if not rows or k is None or m is None or k == m or any(t["k"] == k for t in self.ties):
                continue
            self.ties.append({"k": k, "main": m, "product": rows[k]["product"], "main_name": rows[m]["product"],
                              "code": code if code_like(code) else "", "sig": None, "ready": False})
        if extra:
            self.needs.written = list(self.needs.written) + extra
            reading["requirements"] = list(self.needs.written)
            for req in extra:
                tags = re.findall(r"\(([^)]*)\)", req.partition(":")[0])
                k = self.product_of(f"x ({tags[-1]})") if tags and rows else None
                items = reading.get("items") or []
                if k is not None and k < len(items) and isinstance(items[k], dict) and isinstance(items[k].get("requirements"), list):
                    items[k]["requirements"] = list(items[k]["requirements"]) + [req]
            self.needs.version += 1
            self.log.append({"kind": "tie_fits", "added": extra})
        if not self.ties:
            return

        def restates(value: str, k: int | None) -> bool:
            """True when a requirement value only restates a tie of product k: it names the tie's code, or shares a kind
            word with the main product and speaks of working with, compatibility, fit, going with, being listed for or
            pairing.
            """
            for tie in self.ties:
                if k is not None and k != tie["k"]:
                    continue
                if tie["code"] and names_code(value, tie["code"]):
                    return True
                if kind_words(tie["main_name"]) & kind_words(value) and re.search(r"\b(?:work|compat|fit|go(?:es)? with|listed for|pair)", value.lower()):
                    return True
            return False
        gone = []
        for req in list(self.needs.written):
            head, _, value = str(req).partition(":")
            tags = re.findall(r"\(([^)]*)\)", head)
            k = self.product_of(f"x ({tags[-1]})") if tags else None
            if any(k == tie["k"] for tie in self.ties) and restates(value, k):
                gone.append(req)
        if gone:
            self.needs.written = [r for r in self.needs.written if r not in gone]
            reading["requirements"] = list(self.needs.written)
            for k, row in enumerate(reading.get("items") or []):
                if isinstance(row, dict) and isinstance(row.get("requirements"), list):
                    row["requirements"] = [r for r in row["requirements"] if not restates(str(r).partition(":")[2], k)]
            self.log.append({"kind": "tie_restated", "left": gone})
        self.needs.tie_text = self.tie_text()
        self.log.append({"kind": "ties", "ties": [{key: t[key] for key in ("product", "main_name", "code")} for t in self.ties]})

    def series_pairs(self, tie: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """(CS) The seen in-stock listings of a tie's main product whose records name the part series they take, and those of
        the tied product whose own part code carries a series; each by id, cheapest first (CSb: rows map by their own
        category, cached per archive size)."""
        rows = self.ch_rows()
        dead = set(self.dead_lines())
        homes = self.home_shelves()
        key = (len(self.archive.items), len(dead), tie["k"], tie["main"], repr(homes))
        memo = self.__dict__.setdefault("series_memo", {})
        if key in memo:
            return memo[key]
        hosts: dict[str, list[str]] = {}
        parts: dict[str, list[str]] = {}
        for ident, row in rows.items():
            if ident in dead or row.get("stock") is False:
                continue
            shelf = row.get("shelf") or ""
            k = (next((n for n, home in enumerate(homes) if shelf in home), None) if shelf else
                 self.product_index(ident) if any(homes) else None)      # (CSc: no archive walk while no product has a home)
            if k == tie["main"]:
                got = part_series(row.get("text") or "")
                if got:
                    hosts[ident] = got
            elif k == tie["k"]:
                got = part_code_series(f"{row.get('title') or ''} {row.get('text') or ''}")
                if got:
                    parts[ident] = got

        def cost(i: str) -> float:
            price = rows[i].get("price")
            return float(price) if is_number(price) else float("inf")
        memo[key] = ({i: hosts[i] for i in sorted(hosts, key=lambda i: (cost(i), i))},
                     {i: parts[i] for i in sorted(parts, key=lambda i: (cost(i), i))})
        return memo[key]

    def series_line(self) -> str:
        """(CS) For each tie in a store whose main items name part series: the seen main items with the series they take (once
        per main product), the seen parts whose code carries a series letter those use, and the pairs that match; about
        1,200 characters at most ('' when no seen main item names a series)."""
        out: list[str] = []
        named: set[int] = set()
        for tie in self.ties:
            hosts, parts = self.series_pairs(tie)
            if not hosts:
                continue
            heads = {re.sub(r"[^A-Z]", "", series_key(s)) for got in hosts.values() for s in got}
            parts = {p: got for p, got in parts.items() if any(re.sub(r"[^A-Z]", "", s) in heads for s in got)}
            fits = {h: [p for p in parts if {series_key(s) for s in hosts[h]} & {series_key(s) for s in parts[p]}] for h in hosts}
            pairs, listed = [], set()
            while len(pairs) < 6 and any(fits.values()):
                for h in hosts:      # (CSe: round the main items in price order, each its cheapest part not listed yet)
                    while fits[h] and fits[h][0] in listed:
                        fits[h].pop(0)
                    if fits[h] and len(pairs) < 6:
                        listed.add(fits[h][0])
                        pairs.append((h, fits[h].pop(0)))
            matched = {h for h, _ in pairs} | {p for _, p in pairs}
            if tie["main"] not in named:
                named.add(tie["main"])
                out.append(f"A {tie['main_name']} listing names the part series it takes; a part works with it only when its own "
                           f"part code carries one of them ([AB1-2345] or (part AB1-2345) is series AB1; a search of a series "
                           f"code alone lists its parts). Seen {tie['main_name']}: "
                           + "; ".join(f"{h} takes {', '.join(hosts[h])}" for h in sorted(hosts, key=lambda i: i not in matched)[:6])
                           + ".")
            fit = {p for _, p in pairs[:6]}
            part_ids = [p for p in parts if p not in fit][:4]
            text = (f"{tie['product']} fitting pairs: " + "; ".join(f"{h} with {p}" for h, p in pairs[:6]) + "." if pairs else
                    f"No seen {tie['product']} fits a seen {tie['main_name']} yet.")      # (CSc: the pairs first)
            if part_ids:
                text += f" Other {tie['product']} seen: " + "; ".join(f"{p} carries {', '.join(parts[p])}" for p in part_ids) + "."
            out.append(text)
        line = " ".join(out)
        if len(line) > 1600:
            line = line[:1600].rsplit(". ", 1)[0] + "."
        return ("\n[part series] " + line) if line else ""

    def tie_text(self) -> str:
        """Phrases each tie as a rule: the tied product's own record must name one of the main product's model codes
        (or the chosen main's model name when none is known), read from records and never asked.
        """
        parts = []
        for tie in self.ties:
            codes = self.tie_codes(tie, ask=False)
            parts.append(f"the {tie['product']} must be listed as working with the {tie['main_name']}: its own record must name "
                         + (" or ".join(f"'{c}'" for c in codes) if codes else
                            f"a part series the chosen {tie['main_name']} takes, in its own part code" if self.series_pairs(tie)[0] else
                            f"the chosen {tie['main_name']}'s model name")
                         + " (read from the records, never asked)")
        return "; ".join(parts)

    def tie_line(self) -> str:
        """The '[ties]' status line, or empty text when no tie is registered."""
        return ("\n[ties] " + self.tie_text() + ".") if self.ties else ""

    def record_text(self, ident: str) -> str:
        """Collects all text the observations hold about a listing id. Every dict, at any depth of each stored record,
        with a string value equal to the id contributes its strings, read up to 4 levels deep and 60 list entries
        per list. The joined text is cached in record_memo per id and number of stored records.
        Tie and fit checks search this text for model codes and rules.
        """
        memo = self.__dict__.setdefault("record_memo", {})
        key = (ident, len(self.archive.items))
        if key in memo:
            return memo[key]
        parts: list[str] = []

        def grab(x: Any, lvl: int = 0) -> None:
            """Appends every string found inside a value to parts, descending through dict values and the first 60 items
            of each list. Stops below 4 levels of nesting; numbers and other scalars are ignored.
            record_text joins the collected strings.
            """
            if lvl > 4:
                return
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict):
                for v in x.values():
                    grab(v, lvl + 1)
            elif isinstance(x, list):
                for v in x[:60]:
                    grab(v, lvl + 1)

        def visit(d: dict[str, Any]) -> None:
            """Grabs a dict's text when one of its string values is the listing id."""
            if any(isinstance(v, str) and v == ident for v in d.values()):
                grab(d)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        memo[key] = " ".join(parts)
        return memo[key]

    def tie_main_pick(self, tie: dict[str, Any]) -> str | None:
        """Chooses the main product's listing a tie refers to. A cart line of the main product wins; else the cheapest
        listing of it that passed checks at the current requirement version and is neither dead nor refused; else,
        when the tie has a code, the cheapest read listing of the main product whose record names that code.
        None when nothing qualifies.
        """
        m = tie["main"]
        lines = [i for i in self.trolley.units if self.product_index(i) == m]
        if lines:
            return lines[0]
        ver, dead = str(self.needs.version), self.dead_lines()

        def cost(i: str) -> float:
            """Recorded price of a listing, or infinity when unknown, for cheapest-first choice."""
            price = self.archive.price(i)
            return float(price) if is_number(price) else float("inf")
        passed = [key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == ver]
        mine = [i for i in passed if self.product_index(i) == m and i not in dead and i not in self.refused_lines]
        if mine:
            return min(mine, key=lambda i: (cost(i), i))
        if tie["code"]:
            named = [i for i in self.archive.read if self.product_index(i) == m and i not in dead
                     and names_code(self.record_text(i), tie["code"])]
            if named:
                return min(named, key=lambda i: (cost(i), i))
        return None

    def tie_names(self, pick: str) -> list[str] | None:
        """Asks a checker model (TIE_NAMES prompt) for the model names of the chosen main listing, from up to 4 of its
        records. Names that look like codes and occur in its record text are kept, part-series codes found in the
        record go first, duplicates by code tokens are removed, and up to 6 are returned. None when model time is
        short (under CLOSING + 30 s), no records exist or the call fails.
        """
        if self.wire.spare() < CLOSING + 30:
            return None
        records = self.archive.about(pick, 4)
        if not records:
            return None
        answer = self.wire.consult(TIE_NAMES + "\n\nRECORDS:\n" + "\n".join(records), 600, CHECKER[:1])
        if answer is None:
            return None
        names = [" ".join(str(n).split())[:60] for n in answer.get("names") or [] if isinstance(n, (str, int))]
        body = self.record_text(pick)
        kept = [n for n in names if code_like(n) and names_code(body, n)]
        series = part_series(body)      # (CS: every series of a list, "VI1, VI4")
        kept = list(dict.fromkeys(series + [n for n in kept if not any(code_tokens(n) == code_tokens(s) for s in series)]))
        self.log.append({"kind": "tie_names", "pick": pick, "names": kept, "said": names[:8]})
        return kept[:6]

    def tie_codes(self, tie: dict[str, Any], ask: bool = True) -> list[str]:
        """Gathers the model codes a tied product's record may name to show it works with the main product: the tie's
        own code plus names read from the main pick. With ask set, tie_names runs once per read pick and the result
        is cached. When no names are cached for the current pick and the tie has its own code, the last names are
        reused. Codes are deduplicated by tokens and ready marks that names are known. A code set that grows after
        being known bumps the requirement version; with ask, a changed set also refreshes the tie text.
        """
        codes = [tie["code"]] if tie["code"] else []
        pick = self.tie_main_pick(tie)
        memo = self.__dict__.setdefault("tie_memo", {})
        if pick and pick in self.archive.read and pick not in memo and ask:
            got = self.tie_names(pick)
            if got is not None:
                memo[pick] = got
        names = memo.get(pick) if pick else None
        if names is None and tie["code"]:
            names = tie.get("last_names")
        if names is not None:
            tie["last_names"] = list(names)
        codes += list(names or [])
        out: list[str] = []
        for code in codes:
            if not any(code_tokens(code) == code_tokens(o) for o in out):
                out.append(code)
        tie["ready"] = names is not None
        sig = tuple(sorted(" ".join(code_tokens(c)) for c in out))
        known = self.__dict__.setdefault("tie_known", {}).setdefault(tie["main"], set())
        fresh = set(sig) - known
        if fresh and known:
            self.needs.version += 1
            self.log.append({"kind": "tie_names_grown", "product": tie["product"], "names": out})
        known |= set(sig)
        if sig != tie.get("sig"):
            tie["sig"] = sig
            if ask:
                self.needs.tie_text = self.tie_text()
        return out

    def tie_hit(self, ident: str, codes: list[str]) -> str | bool | None:
        """Returns the first code named in a listing's record text; False when none is named and the listing was read;
        None when it was not read yet.
        """
        body = self.record_text(ident)
        for code in codes:
            if names_code(body, code):
                return code
        return False if ident in self.archive.read else None

    def fit_view(self, thing: str) -> tuple[list[str], list[str], list[str]]:
        """Checks requirement codes against a listing's own compatibility rules (family_rules). For each open property
        of the listing's product whose value holds a model code, the first of up to two codes that family_fit can
        decide gives a FIT EVIDENCE line or a failure note. Returns (evidence lines, failures, met properties); all
        empty when the record has no such rule.
        """
        lines: list[str] = []
        fails: list[str] = []
        met: list[str] = []
        body = self.record_text(thing)
        if not family_rules(body):
            return lines, fails, met
        several = bool(self.needs.items())
        k = self.product_index(thing)
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            tags = re.findall(r"\(([^)]*)\)", head)
            if several and tags:
                kk = self.product_of(f"x ({tags[-1]})")
                if kk is not None and k is not None and kk != k:
                    continue
            prop = re.split(r"\s*\(", head)[0].strip()
            for code in [c for c in re.findall(r"[A-Za-z]{1,8}[\s-]?\d{2,}[A-Za-z0-9-]*", value) if code_like(c)][:2]:
                fit = family_fit(body, code)
                if fit is None:
                    continue
                rule = "; ".join(r[2] for r in family_rules(body))[:200]
                if fit:
                    lines.append(f"- FIT EVIDENCE: {thing}'s record says '{rule}', which covers {code}: '{req}' is met.")
                    met.append(prop)
                else:
                    fails.append(f"{prop}: {code} is not covered by its own rule '{rule}'")
                break
        return lines, fails, met

    def tie_view(self, thing: str) -> tuple[list[str], list[str]]:
        """Reports tie evidence for a listing of a tied product, using the codes from tie_codes without a model call.
        It gives a TIE EVIDENCE line when the record names a model code of the main product, a failure when it names
        none and the main's codes are complete (ready), or a note that the tie is not settled while codes are
        incomplete. Returns (lines, fails); empty without ties, a matching product or codes.
        """
        lines: list[str] = []
        fails: list[str] = []
        if not self.ties:
            return lines, fails
        k = self.product_index(thing)
        for tie in self.ties:
            if k is None or k != tie["k"]:
                continue
            codes = self.tie_codes(tie, ask=False)
            if not codes:
                continue
            hit = self.tie_hit(thing, codes)
            names = " / ".join(codes)
            if hit:
                lines.append(f"- TIE EVIDENCE: {thing}'s record names '{hit}', a model name of the {tie['main_name']}: it is "
                             "listed as working with it (met).")
            elif hit is False and tie["ready"]:
                fails.append(f"works with the {tie['main_name']} ({tie['product']}): its record names none of the "
                             f"{tie['main_name']}'s model names ({names})")
            elif hit is False:
                lines.append(f"- TIE EVIDENCE: {thing}'s record names none of {names}; the other model names of the "
                             f"{tie['main_name']} are not known yet, so the tie is not settled.")
        return lines, fails

    def tie_rows(self, tie: dict[str, Any], codes: list[str]) -> list[str]:
        """Lists unread, live rows that may satisfy a tie: not sold out, on the tied product's home shelves (or off the
        main product's shelves when it has none), with a record text naming one of the codes, and with price times
        quantity within the budget. Rows showing more of the product's required values come first, then lower
        prices.
        """
        k = tie["k"]
        homes = self.home_shelves()
        home = set(homes[k]) if k < len(homes) else set()
        main_home = set(homes[tie["main"]]) if tie["main"] < len(homes) else set()
        cap = self.needs.spend_cap
        counts = self.needs.counts()
        qty = counts[k] if k < len(counts) else 1
        values = [set(kind_words(plain)) for _, plain in self.code_values(k)]
        out = []
        for ident, row in self.ch_rows().items():
            if ident in self.archive.read or ident in self.dead_lines() or row["stock"] is False:
                continue
            if (home and row["shelf"] not in home) or (not home and main_home and row["shelf"] in main_home):
                continue
            if not any(names_code(self.record_text(ident), c) for c in codes):
                continue
            price = row["price"]
            if is_number(cap) and is_number(price) and price * qty > cap:
                continue
            shown = sum(1 for v in values if v and v <= row["words"])
            out.append((-shown, price if is_number(price) else float("inf"), ident))
        return [i for _, _, i in sorted(out)]

    def tie_search_call(self, words: str) -> dict[str, Any] | None:
        """Builds search arguments for a model code. It finds the search tool's free-text argument (a non-enum string
        named like query, text or keyword, not the page token), then reuses each planned query from code_queries
        with the text replaced, first without numeric arguments, then whole; the first that passes the schema check
        wins, else the bare text argument. None when no search tool or text argument exists.
        """
        tool = self.role("search")
        jschema = self.shapes.get(tool or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        tok = self.token_key()
        texts = [key for key, spec in props.items() if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum") and key != tok]
        text_key = next((key for key in texts if re.search(r"^q$|query|text|keyword|needle|term|word|phrase|string|search", key.lower())),
                        texts[0] if texts else None)
        if not tool or not text_key:
            return None
        for call_args in self.code_queries(None):
            if text_key not in call_args:
                continue
            for base in ({key: val for key, val in call_args.items() if key == text_key
                          or not (isinstance(props.get(key), dict) and props[key].get("type") in ("integer", "number"))}, dict(call_args)):
                base[text_key] = words
                fixed, faults = ArgCheck.execute(base, jschema)
                if not faults:
                    return fixed
        fixed, faults = ArgCheck.execute({text_key: words}, jschema)
        return None if faults else fixed

    def tie_step(self) -> bool:
        """Takes one scripted step toward settling product ties. It is skipped near the order deadline (order_due + 4
        steps), with under CLOSING + 40 s of model time, when the wire is done, broken or mute, or without a search
        tool. For each tie it searches the next unsearched model code (up to TIE_SEARCHES codes per tie); once per
        tie and code set it queues up to TIE_OPENS matching rows (tie_rows) and opens them.
        Returns True when a call was sent.
        """
        if not self.ties or self.wire.done or self.wire.broke or self.wire.mute or not self.role("search"):
            return False
        if self.moves_left() <= self.order_due() + 4 or self.wire.spare() < CLOSING + 40:
            return False
        searched = self.__dict__.setdefault("tie_searched", set())
        opened = self.__dict__.setdefault("tie_opened", set())
        for tie in self.ties:
            codes = self.tie_codes(tie, ask=True)
            for code in codes[:TIE_SEARCHES]:
                mark = " ".join(code_tokens(code))
                if mark in searched:
                    continue
                searched.add(mark)
                call_args = self.tie_search_call(code)
                if not call_args:
                    continue
                proposal = {"id": f"tie_{self.wire.tick}", "type": "function",
                            "function": {"name": self.role("search"), "arguments": json.dumps(call_args)}}
                self.chat.append({"role": "assistant", "content": f"I search the model name {code}: the {tie['product']} must be "
                                  f"listed as working with the {tie['main_name']}.", "tool_calls": [proposal]})
                self.post_replies(self.dispatch([proposal], scripted=True))
                return True
            mark = (tie["k"], tie.get("sig"))
            if codes and mark not in opened:
                opened.add(mark)
                rows = self.tie_rows(tie, codes)[:TIE_OPENS]
                if rows:
                    self.open_queue = rows
                    self.open_purpose = "quota"
                    self.open_say = f"I open the listings that name {' / '.join(codes)} for the {tie['product']}, the closest first."
                    if self.open_named():
                        return True
        return False


    def read_request(self) -> None:
        """Reads the request once at the start with a model call (READ_REQUEST prompt) over the tools, the request and
        the opening observation, the fixed part marked for caching. It tries the reader models plus a listener, up
        to three rounds while failures look retryable and time allows, until an answer carries a roles map.
        Without one it falls back to code_reading, keeping the model's other fields. The result goes to use_reading.
        """
        tool_rows = [f"- {tdef['function']['name']}: {tdef['function'].get('description', '')} | arguments: {as_text(tdef['function'].get('parameters') or {})}" for tdef in self.toolset]
        fixed_part = READ_REQUEST + "\n\nTOOLS:\n" + "\n".join(tool_rows)
        var_part = "REQUEST:\n" + self.request
        if self.pview.get("opening") is not None:
            var_part += "\n\nOPENING OBSERVATION:\n" + as_text(self.pview.get("opening"))
        content = [{"type": "text", "text": fixed_part, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": var_part}]
        reading: dict[str, Any] = {}
        self.read_content = content
        ladder = READER + LISTENER[:1]
        for turn, mdl in enumerate(ladder * 3):
            if turn >= len(ladder) and (self.wire.misses == 0 or self.wire.call_fail == "refused" or self.wire.down()
                                        or not self.wire.usable(ladder) or self.wire.spare() < CLOSING + 60):
                break
            msg = self.wire.query_model([{"role": "user", "content": content}], [mdl], 3000)
            reading = json_object((msg or {}).get("content") or "") or {}
            if isinstance(reading.get("roles"), dict):
                break
            self.wire.log.append({"kind": "setup_retry", "model": mdl})
        if not isinstance(reading.get("roles"), dict):
            try:
                reading = {**self.code_reading(), **{key: val for key, val in reading.items() if key != "roles"}}
            except Exception as err:
                self.wire.log.append({"kind": "code_reading_error", "error": f"{type(err).__name__}: {err}"[:300]})
                reading = {}
            self.read_at = monotonic()
            self.wire.log.append({"kind": "code_reading", "roles": reading.get("roles"), "product": reading.get("product"),
                                  "price_cap": reading.get("price_cap"), "items": len(reading.get("items") or [])})
        self.use_reading(reading)

    def seed_joined(self) -> None:
        """E1c: the request's own "from <name>" category names that join two kinds at one joiner ("Wall Hooks and Coat
        Racks") are labels merge_joined (CA F1) knows before any search, so a reading that split such a category into
        two products merges at once, under merge_joined's own tests. A capture holding a figure or another "from" spans
        two request lines and is no name."""
        if len(self.needs.items()) < 2:
            return
        seen = self.__dict__.setdefault("joined_seen", {})
        for hit in re.finditer(r"\bfrom\s+([^;.()\n]{2,80}?)\s*(?=[;.)\n]|$)", str(self.request or "")):
            label = hit.group(1).strip()
            if not re.search(r"\d|\bfrom\b", label, re.I) and len([p for p in re.split(JOINERS, label.lower()) if p.strip()]) == 2 \
                    and not self.named_apart(label):      # (SH: a half that is another named category: two categories)
                seen.setdefault(label.lower(), label)
        if seen:
            note = self.merge_joined({})
            if note:
                self.log.append({"kind": "joined_at_reading", "note": note[:200]})

    def page_roles_fix(self) -> None:
        """(DS) Layout A: details is the page with a listing's properties, view its offer page with the notes. A reading that
        put them the other way round is swapped when both tools' descriptions say so plainly."""
        det, view = self.role("details"), self.role("view")
        if not det or not view or det == view or not self.split_pages() or det not in self.page_tools() \
                or view not in self.page_tools():
            return      # (DSb: both must be single-listing page tools)
        descs = {str(t["function"].get("name")): str(t["function"].get("description") or "").lower() for t in self.toolset}
        facts = r"\b(?:propert(?:y|ies)|technical|specs?|specifications?|facts|fact sheet|attributes?|particulars|characteristics|parameters)\b"
        notes = r"\b(?:offer|notes?|fine print|remarks?)\b"
        d, v = descs.get(det, ""), descs.get(view, "")
        if re.search(notes, d) and not re.search(facts, d) and re.search(facts, v) and not re.search(notes, v):
            self.role_map["details"], self.role_map["view"] = view, det
            self.log.append({"kind": "page_roles_swapped", "details": view, "view": det})

    def use_reading(self, reading: dict[str, Any], again: bool = False) -> None:
        """Applies a request reading at start or re-read (again=True). Keeps
        real-tool roles (plus a used cart tool on re-read); new CartBook unless
        cart used. Sets sweepable (paged search), message cap (message_budget),
        Wire time scale by product count, written requirements (no property or
        value: soft_wants). Registers ties, story values, references, go-ahead
        rules; drops ask-first items repeating one. Budget: price_cap (kept on
        re-read after needs change); none + message tool: ask first. A lower
        step_budget cuts the step cap; re-read bumps requirement version.
        """
        if not again:
            self.steps_given = self.step_cap
        old_cart = self.role("cart")
        role_map = reading.get("roles") if isinstance(reading.get("roles"), dict) else {}
        self.role_map = {fkey: val for fkey, val in role_map.items() if isinstance(val, str) and val in self.shapes}
        if again and self.trolley.used and old_cart:
            self.role_map["cart"] = old_cart
        try:
            self.page_roles_fix()      # (DS)
        except Exception as err:
            self.log.append({"kind": "page_roles_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if not (again and self.trolley.used):
            self.trolley = CartBook(self.role("cart"), self.shapes.get(self.role("cart") or "", {}))
        self.needs.sweepable = bool(self.role("search")) and self.page_arg() is not None
        spend_cap = reading.get("message_budget")
        if is_number(spend_cap) and spend_cap >= 0:
            self.msg_cap = int(spend_cap)
        self.needs.reading = reading
        try:
            self.seed_joined()      # (E1c)
        except Exception as err:
            self.wire.log.append({"kind": "seed_joined_error", "error": f"{type(err).__name__}: {err}"[:300]})
        try:
            self.wire.scale = 1.0 + 0.35 * min(3, max(0, len(self.needs.items()) - 1))
        except Exception:
            self.wire.scale = 1.0
        named = [str(req) for req in reading.get("requirements") or [] if req]
        self.needs.written = [req for req in named if ":" in req and req.split(":", 1)[0].strip() and req.split(":", 1)[1].strip()]
        loose = [req for req in named if req not in self.needs.written]
        if loose:
            reading["soft_wants"] = [str(w) for w in reading.get("soft_wants") or [] if w] + loose
            reading["requirements"] = list(self.needs.written)
        try:
            self.tie_register(reading)
        except Exception as err:
            self.wire.log.append({"kind": "tie_register_error", "error": f"{type(err).__name__}: {err}"[:300]})
        try:
            self.story_values(reading)
        except Exception as err:
            self.wire.log.append({"kind": "story_values_error", "error": f"{type(err).__name__}: {err}"[:300]})
        try:
            self.ref_register()
        except Exception as err:
            self.wire.log.append({"kind": "ref_register_error", "error": f"{type(err).__name__}: {err}"[:300]})
        try:
            self.go_rules = self.go_ahead_rules(reading)
        except Exception as err:
            self.wire.log.append({"kind": "go_rules_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if self.go_rules and reading.get("ask_first"):
            told = [word_set(r) for r in self.go_rules]
            kept = []
            for item in reading.get("ask_first") or []:
                words = word_set(item)
                if words and (go_words(str(item)) or any(len(words & t) >= 0.6 * len(words) for t in told)):
                    self.wire.log.append({"kind": "ask_first_go_dropped", "item": str(item)[:200]})
                    continue
                kept.append(item)
            reading["ask_first"] = kept
        if not (again and self.needs.version > 0):
            self.needs.spend_cap = reading.get("price_cap")
        if reading and self.needs.spend_cap is None and self.role("message"):
            reading["ask_first"] = ["the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"] + [str(a) for a in reading.get("ask_first") or [] if a]
        if again:
            self.step_cap = self.__dict__.get("steps_given", self.step_cap)
        step_count = reading.get("step_budget")
        if is_number(step_count) and 0 < step_count < self.step_cap:
            self.step_cap = int(step_count)
        if again:
            self.needs.version += 1

    def story_values(self, reading: dict[str, Any]) -> None:
        """Cleans the reading's hints and moves guessed values out of the requirements. Hints given as a dict become
        'property: value' lines, each kept to one property. When story_guess marks some, but at most half, of two or
        more written requirements as guessed from the story, they leave the requirements and items and become hints
        to ask about. Hints that written_hints turns into requirements are dropped; at most 12 hints stay.
        """
        raw = reading.get("hints")
        if isinstance(raw, dict):
            raw = [f"{v}" if ":" in str(v) else f"{k}: {v}" for k, vals in raw.items()
                   for v in (vals if isinstance(vals, list) else [vals]) if v]
        hints = [str(h).strip() for h in raw or [] if isinstance(h, (str, int, float)) and str(h).strip()] if isinstance(raw, (list, tuple)) else []
        hints = [one_prop_hint(h) for h in hints]
        own_hints = list(hints)
        refs = [str(r) for r in reading.get("references") or []] if isinstance(reading.get("references"), list) else []
        moved = [req for req in self.needs.written if story_guess(req, self.request, refs)]
        if moved and len(self.needs.written) >= 2 and 2 * len(moved) <= len(self.needs.written):
            self.needs.written = [req for req in self.needs.written if req not in moved]
            reading["requirements"] = list(self.needs.written)

            def base(req: str) -> tuple[str, str]:
                """Splits a requirement into its lowered property name (without product tag) and lowered value."""
                head, _, value = str(req).partition(":")
                return re.split(r"\s*\(", head)[0].strip().lower(), value.strip().lower()
            gone = {base(req) for req in moved}
            for row in reading.get("items") or []:
                if isinstance(row, dict) and isinstance(row.get("requirements"), list):
                    row["requirements"] = [r for r in row["requirements"] if base(r) not in gone]
            hints += [f"{req.split(':', 1)[0].strip()}: the story suggests {req.split(':', 1)[1].strip()} (not stated as a value; ask)" for req in moved]
            self.wire.log.append({"kind": "story_values", "moved": moved})
        try:
            written = self.written_hints(reading, own_hints)
            hints = [h for h in hints if h not in written]
        except Exception as err:
            self.wire.log.append({"kind": "written_hint_error", "error": f"{type(err).__name__}: {err}"[:300]})
        reading["hints"] = hints[:12]

    def written_hints(self, reading: dict[str, Any], hints: list[str]) -> list[str]:
        """Promotes hints whose value the request itself states into written requirements. For each 'property: detail'
        hint (junk keys skipped), lead_value looks for the value word in the request for the hint's product (its
        tag, else each item's product, else the main product). When no written requirement of that product already
        has the property or that word, 'property: word' is added to written, the reading's requirements and the
        matching item. Returns the hints moved, each logged as written_hint.
        """
        rows = [r for r in reading.get("items") or [] if isinstance(r, dict)]
        moved: list[str] = []
        for hint in hints:
            head, sep, detail = str(hint).partition(":")
            prop = re.split(r"\s*\(", head)[0].strip()
            if not sep or not prop or junk_key(prop):
                continue
            tag = re.findall(r"\(([^)]*)\)", head)
            names = [tag[-1]] if tag else [str(r.get("product") or "") for r in rows] or [str(reading.get("product") or "")]
            word, name = "", ""
            for name in names:
                word = lead_value(detail, self.request, name)
                if word:
                    break
            if not word:
                continue
            mine = kind_words(name)
            clash = False
            for req in self.needs.written:
                rhead, _, rvalue = req.partition(":")
                rtag = re.findall(r"\(([^)]*)\)", rhead)
                if rtag and mine and not (kind_words(rtag[-1]) & mine):
                    continue
                if same_prop(rhead, prop) or re.search(r"\b" + re.escape(word) + r"\b", rvalue.lower()):
                    clash = True
            if clash:
                continue
            req = f"{head.strip()}: {word}"
            self.needs.written.append(req)
            reading["requirements"] = list(self.needs.written)
            for row in rows:
                if isinstance(row.get("requirements"), list) and (len(rows) == 1 or kind_words(row.get("product") or "") & mine):
                    row["requirements"].append(req)
                    break
            moved.append(hint)
            self.wire.log.append({"kind": "written_hint", "hint": str(hint)[:200], "requirement": req})
        return moved

    def open_hints(self, k: int | None = None) -> list[str]:
        """Lists the reading's hints still worth asking about, for all products or product k. On a multi-product request
        each hint's product comes from product_of. A hint stays when its property is open by its samples (open_prop)
        and no answered question covered it (asked_inside); otherwise hint_alt may offer an alternative property
        from the hint. Alternatives come first in the result.
        """
        out, alts = [], []
        for hint in self.needs.reading.get("hints") or []:
            prop = str(hint).split(":", 1)[0].strip()
            if not prop:
                continue
            j = self.product_of(prop) if self.needs.items() else None
            if k is not None and j not in (k, None):
                continue
            if self.open_prop(j if self.needs.items() else None, prop, self.prop_line(j, prop), by_samples=True) \
                    and not self.asked_inside(j if self.needs.items() else None, prop):
                out.append(str(hint))
                continue
            alt = self.hint_alt(j if self.needs.items() else None, str(hint), prop)
            if alt:
                alts.append(alt)
        return alts + out

    def prop_line(self, k: int | None, prop: str) -> str:
        """Finds the property menu line for a property, from the store fields seen so far. The menu (prop_menu) comes
        from the first read shelf among product k's home shelves, else from the item's own shelf or all records,
        and is cached per product until new observations arrive. Returns the line whose key strictly matches the
        property; empty text otherwise or on error.
        """
        try:
            memo = self.__dict__.setdefault("menu_memo", {}).get(k)
            if memo and memo[0] == len(self.archive.items):
                menu = memo[1]
            else:
                menu = ""
                if self.needs.items():
                    homes = self.home_shelves()
                    home = homes[k] if k is not None and k < len(homes) else set()
                    shelves = [s for s in self.archive.shelves_read() if s.strip().lower() in home]
                    menu = self.archive.prop_menu(shelves[0])[0] if shelves else ""
                else:
                    own = self.own_shelf()
                    menu = self.archive.prop_menu(own)[0] if own else self.archive.prop_menu()[0]
                self.menu_memo[k] = (len(self.archive.items), menu)
            for line in menu.splitlines():
                if strict_prop(line.strip("- ").split(" (in ")[0].strip(), prop):
                    return line
        except Exception:
            return ""
        return ""

    def field_values(self, k: int | None, prop: str) -> set[str]:
        """Normalised sample values on the property's menu line; empty when the line lists none."""
        line = self.prop_line(k, prop)
        values = line.split("):", 1)[1].split("|") if "):" in line else []
        return {" ".join(re.findall(r"[a-z0-9]+", v.lower())) for v in values} - {""}

    def same_field(self, k: int | None, a: str, b: str) -> bool:
        """True when two property names match strictly, or match loosely (same_prop) and their menu lines share a
        sample value.
        """
        if strict_prop(a, b):
            return True
        try:
            return same_prop(a, b) and bool(self.field_values(k, a) & self.field_values(k, b))
        except Exception:
            return False

    def hinted_prop(self, k: int | None, prop: str) -> bool:
        """True when one of the reading's hints is about this property (strict match), for product k on a
        multi-product request or for the single item.
        """
        name = str(prop).split("(")[0].strip()
        for hint in self.needs.reading.get("hints") or []:
            head = str(hint).split(":", 1)[0]
            if strict_prop(head, name) and (k is None or not self.needs.items() or self.product_of(head) in (k, None)):
                return True
        return False

    def product_reads(self, k: int | None) -> int:
        """Counts the rich reads (rich_reads) recorded on product k's home shelves on a multi-product request, or on
        the item's own shelf (all shelves when none is known) otherwise. A product without home shelves counts 0,
        and errors count as 0.
        """
        try:
            if self.needs.items():
                homes = self.home_shelves()
                home = homes[k] if k is not None and k < len(homes) else set()
                return sum(self.archive.rich_reads(s) for s in home) if home else 0
            own = self.own_shelf()
            return self.archive.rich_reads(own) if own else self.archive.rich_reads()
        except Exception:
            return 0

    def plain_hint(self, k: int | None, prop: str, hinted: bool = False) -> bool:
        """True when a hinted property has no field key in the store: a hint names it (or hinted is set), at least 2
        product reads exist, and no menu line matches it, strictly or loosely. hint_mark tags such hints.
        Errors return False.
        """
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            if not name or not (hinted or self.hinted_prop(k, name)) or self.prop_line(k, name) or self.product_reads(k) < 2:
                return False
            menu = (self.__dict__.get("menu_memo", {}).get(k) or (0, ""))[1]
            return not any(same_prop(line.strip("- ").split(" (in ")[0].strip(), name) for line in menu.splitlines())
        except Exception:
            return False

    def hint_mark(self, hint: str) -> str:
        """Returns the hint with '[no key]' appended when the store's fields show no key for its property (plain_hint),
        else the hint unchanged.
        """
        try:
            prop = str(hint).split(":", 1)[0].strip()
            k = self.product_of(prop) if self.needs.items() else None
            return f"{hint} [no key]" if self.plain_hint(k, prop, hinted=True) else str(hint)
        except Exception:
            return str(hint)

    @staticmethod
    def hint_alts_of(detail: str) -> list[str]:
        """Splits the '(or ...)' parts of a hint detail into alternative property names, at commas, 'or' and slashes."""
        return [a.strip() for m in re.findall(r"\(or ([^()]+)\)", str(detail)) for a in re.split(r",|\bor\b|/", m) if a.strip()]

    @staticmethod
    def name_inside(a: Any, b: Any) -> bool:
        """True when the content words (3+ letters, no link words) of one property name are a subset of the other's."""
        wa = {w for w in prop_words(a) if len(w) >= 3 and w not in LINK_WORDS}
        wb = {w for w in prop_words(b) if len(w) >= 3 and w not in LINK_WORDS}
        return bool(wa) and bool(wb) and (wa <= wb or wb <= wa)

    def asked_inside(self, k: int | None, head: str) -> bool:
        """True when an answered question's label already covers the property for product k: by name_inside, the words
        of a label part lie within the property's, or the reverse. Label parts are split at ';' and tied to products
        by their tag or, failing that, by the question text. Questions marked not answered do not count, and errors
        return False.
        """
        try:
            several = bool(self.needs.items())
            for ln in self.questions_log:
                label = ln.split(":", 1)[0].strip("- ")
                if "[not answered]" in label:
                    continue
                said = re.search(r'asked "(.*?)" -> ', ln)
                for part in label.split(";"):
                    j = self.product_of(part) if several else None
                    if several and j is None and said:
                        j = self.product_of(said.group(1))
                    if (not several or k is None or j in (k, None)) and self.name_inside(part, head):
                        return True
        except Exception:
            return False
        return False

    def hint_alt(self, k: int | None, hint: str, head: str) -> str:
        """Offers a hint's alternative property after its first property drew no preference. It applies only when a
        quiet ask (quiet_asks) touched that property; the first '(or ...)' alternative with a menu line becomes the
        key, cached per hint. Returns a hint line for that key, with the product tag and a note that the first
        property drew no preference, while the key is unspent and open in the plan; empty text otherwise.
        """
        try:
            memo = self.__dict__.setdefault("hint_alts", {})
            several = bool(self.needs.items())
            if not memo.get(hint):
                quiet = self.__dict__.get("quiet_asks") or []
                if not any((self.same_field(k, q, head) or self.name_inside(q, head)) and (not several or self.product_of(q) in (k, None))
                           for q in quiet):
                    return ""
                for alt in self.hint_alts_of(hint.partition(":")[2]):
                    line = "" if junk_key(alt) or strict_prop(alt, head) else self.prop_line(k, alt)
                    if line:
                        memo[hint] = line.strip("- ").split(" (in ")[0].strip()
                        break
            key = memo.get(hint) or ""
            if not key or key.lower() in self.__dict__.get("spent_asks", set()) or not self.plan_open(k, key):
                return ""
            tag = re.search(r"\(([^()]*)\)\s*$", hint.partition(":")[0])
            detail = re.sub(r"\s*\(or [^()]+\)", "", hint.partition(":")[2]).strip()
            return f"{key}{' (' + tag.group(1).strip() + ')' if tag else ''}: {detail} (its first property, {head}, drew no preference)"
        except Exception:
            return ""

    def hint_mapped(self, k: int | None, prop: str) -> bool:
        """True when a hint for product k maps to the property: the hint's own property or one of its '(or ...)'
        alternatives matches it strictly, or all of the property's content words occur in the hint text.
        quiet_reask requires it before re-asking, and twin_key accepts a menu key that a hint maps to.
        Errors return False.
        """
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            words = [w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
            several = bool(self.needs.items())
            for hint in self.needs.reading.get("hints") or []:
                head, _, detail = str(hint).partition(":")
                if several and k is not None and self.product_of(head) not in (k, None):
                    continue
                if any(strict_prop(n, name) for n in [re.split(r"\s*\(", head)[0].strip()] + self.hint_alts_of(detail) if n):
                    return True
                low = str(hint).lower()
                if words and all(re.search(r"\b" + re.escape(w), low) for w in words):
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def name_twin(a: Any, b: Any) -> bool:
        """True when two different property names look like variants of each other: a head word (all content words but
        the last) of one equals, or shares a 4+ letter prefix with, a content word of the other. Strictly equal
        names are not twins.
        """
        if strict_prop(a, b):
            return False

        def words(name: Any) -> tuple[list[str], list[str]]:
            """Content words (3+ letters, no link words) of a name, plus the same list without its last word when it has
            more than one.
            """
            ws = [w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
            return ws, (ws[:-1] if len(ws) > 1 else ws)

        def match(x: str, y: str) -> bool:
            """Words match when equal, or when both have 4+ letters and one begins with the other."""
            return x == y or (min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x)))
        wa, ha = words(a)
        wb, hb = words(b)
        return any(match(x, y) for x in ha for y in wb) or any(match(x, y) for x in wa for y in hb)

    def twin_of(self, k: int | None, a: str, b: str) -> bool:
        """True when two property names differ but are twins by name (name_twin) or by shared field values (same_field)
        for product k.
        """
        try:
            return not strict_prop(a, b) and (self.name_twin(a, b) or self.same_field(k, a, b))
        except Exception:
            return False

    def twin_key(self, k: int | None, prop: str) -> str:
        """Looks in product k's property menu for a twin of a property that the story points to. A menu key qualifies
        when it is not junk, not spent, a twin of the property, open in the plan and not nested with the property's
        name; it is returned when one of its extra words (not the property's or the product's kind words) occurs in
        the request or the product's hints, or when a hint maps to it. Empty text otherwise.
        """
        try:
            several = bool(self.needs.items())
            self.prop_line(k, prop)
            menu = (self.__dict__.get("menu_memo", {}).get(k) or (0, ""))[1]
            product = self.needs.items()[k]["product"] if several and k is not None else str(self.needs.reading.get("product") or "")
            own = kind_tokens(product)
            story = " ".join([self.request.lower()] + [str(h).lower() for h in self.needs.reading.get("hints") or []
                                                       if not several or self.product_of(str(h).split(":", 1)[0]) in (k, None)])
            spent = self.__dict__.get("spent_asks", set())
            mine = set(prop_words(prop))
            for line in menu.splitlines():
                key = line.strip("- ").split(" (in ")[0].strip()
                if not key or junk_key(key) or key.lower() in spent or not self.twin_of(k, key, prop) or not self.plan_open(k, key):
                    continue
                if self.name_inside(key, prop):
                    continue
                words = [w for w in prop_words(key) if len(w) >= 3 and w not in LINK_WORDS and w not in mine and not kind_in(w, own)]
                if any(re.search(r"\b" + re.escape(w), story) for w in words) or self.hint_mapped(k, key):
                    return key
        except Exception:
            return ""
        return ""

    def values_form(self, k: int | None, prop: str, product: str) -> str:
        """Writes a choice question from a property's sample values: up to three short values (30 chars or less, with
        letters, no list punctuation) that appear on no other menu line, asked as a pick among them for the product.
        Needs at least two such values; empty text otherwise. distinct_ask uses it to re-ask with concrete options.
        Errors return empty text.
        """
        try:
            line = self.prop_line(k, prop)
            if "):" not in line:
                return ""
            menu = (self.__dict__.get("menu_memo", {}).get(k) or (0, ""))[1]

            def norm(v: str) -> str:
                """Lowercases a value and keeps only its letter and digit runs, space-joined."""
                return " ".join(re.findall(r"[a-z0-9]+", v.lower()))
            others = {norm(v) for ln in menu.splitlines() if "):" in ln and ln.strip() != line.strip()
                      for v in ln.split("):", 1)[1].split("|")}
            picks: list[str] = []
            for v in (v.strip() for v in line.split("):", 1)[1].split("|")):
                if v and len(v) <= 30 and re.search(r"[a-z]", v.lower()) and not re.search(r"[,;/]", v) \
                        and norm(v) not in others and norm(v) not in [norm(p) for p in picks]:
                    picks.append(v)
            picks = picks[:3]
            if len(picks) < 2:
                return ""
            return f"For the {product}, would you like {', '.join(picks[:-1])} or {picks[-1]}?"
        except Exception:
            return ""

    def look_alikes(self, k: int | None, prop: str, lexical_only: bool = False, avoid: str = "") -> list[str]:
        """Finds up to two other menu keys of product k that look like the property, so a question can set them apart.
        Keys are skipped when junk, catch-all, strictly the same, holding digits or over 4 words, sharing a sample
        set with the property's line (one within the other), covered by the avoid text, or a superset of the
        property's words. The rest need an alike_score of 2.0 or more (lexical only when asked), are ordered by
        score and listing count, and nested names are not picked twice.
        """
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            own = self.prop_line(k, name)
            menu = (self.__dict__.get("menu_memo", {}).get(k) or (0, ""))[1]

            def samples(line: str) -> set[str]:
                """Lowered sample values of a menu line, as a set."""
                return {v.strip().lower() for v in line.split("):", 1)[1].split("|") if v.strip()} if "):" in line else set()
            mine = samples(own)
            said = set(key_words(avoid))
            found: list[tuple[float, int, str]] = []
            for line in menu.splitlines():
                key = line.strip("- ").split(" (in ")[0].strip()
                if not key or junk_key(key) or catch_all_key(key) or strict_prop(key, name):
                    continue
                if re.search(r"\d", key) or len(key_words(key)) > 4:
                    continue
                theirs = samples(line)
                if mine and theirs and (theirs <= mine or mine <= theirs):
                    continue
                if said and set(key_words(key)) <= said:
                    continue
                if {w.rstrip("s") for w in key_words(key)} > {w.rstrip("s") for w in key_words(name)}:
                    continue
                score, lexical = alike_score(name, key)
                if score < 2.0 or (lexical_only and not lexical):
                    continue
                count = re.search(r"\(in (\d+) listing", line)
                found.append((-score, -(int(count.group(1)) if count else 0), key))
            picked: list[str] = []
            for _, _, key in sorted(found):
                words = set(key_words(key))
                if any(words <= set(key_words(p)) or set(key_words(p)) <= words for p in picked):
                    continue
                picked.append(key)
                if len(picked) == 2:
                    break
            return picked
        except Exception:
            return []

    def apart_text(self, prop: str, product: str, alikes: list[str]) -> str:
        """Phrases the which-property question for a product, naming up to two look-alike properties to set aside
        when there are any.
        """
        name = re.split(r"\s*\(", str(prop))[0].strip()
        if not alikes:
            return f"Which {name} would you like for the {product}?"
        return f"Apart from the {' and the '.join(alikes[:2])}, which {name} would you like for the {product}?"

    def apart_plain(self, prop: str, text: str) -> str:
        """Rewrites a plain which-property question into the apart form when the property has lexical look-alikes on
        the product's menu, so the question sets those properties apart. Other question texts, unknown products
        on a multi-product request and errors leave the text unchanged; rewrites are logged as apart_first.
        The product comes from the property's tag or the reading.
        """
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            tied = re.search(r"\(([^)]+)\)", str(prop))
            product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
            if str(text).strip().lower() != f"which {name} would you like for the {product}?".lower():
                return text
            several = bool(self.needs.items())
            k = self.product_of(prop) if several else None
            if several and k is None:
                return text
            alikes = [a for a in self.look_alikes(k, name, lexical_only=True) if not self.plan_open(k, a)]      # (E3b)
            if not alikes:
                return text       # (E3: an open, unasked look-alike may be the hidden need: never set aside)
            self.log.append({"kind": "apart_first", "property": name[:60], "alikes": alikes})
            return self.apart_text(name, product, alikes)
        except Exception:
            return text

    def distinct_ask(self, prop: str, sent: str = "") -> tuple[str, str] | None:
        """Plans a re-ask for a property, at most once per product (twin_asks). It prefers the apart form when
        look-alike keys exist (sent text is avoided). On a single-product request it gives up when other ranked
        properties remain unasked. Otherwise it asks with the property's sample values (values_form), or switches
        to a twin key (twin_key) and asks with its values or a which-question.
        Returns (property label, question) or None.
        """
        try:
            several = bool(self.needs.items())
            k = self.product_of(prop) if several else None
            done = self.__dict__.setdefault("twin_asks", set())
            if (several and k is None) or (k if several else -1) in done:
                return None
            product = self.needs.items()[k]["product"] if several else str(self.needs.reading.get("product") or "item")
            target = re.split(r"\s*\(", str(prop))[0].strip()
            alikes = self.look_alikes(k, target, False, sent)
            if alikes:
                done.add(k if several else -1)
                self.log.append({"kind": "apart_reask", "property": target[:60], "alikes": alikes})
                return (f"{target} ({product})" if several else target), self.apart_text(target, product, alikes)
            if not several:
                try:
                    left = [n for n in self.rank_props(None) if not self.asked_for(None, n) and not strict_prop(n, target)]
                except Exception:
                    left = []
                if left:
                    self.log.append({"kind": "values_reask_skipped", "property": target[:60], "next": left[0][:60]})
                    return None
            text = self.values_form(k, target, product)
            twin = "" if text else self.twin_key(k, prop)
            if twin:
                target = twin
                text = self.values_form(k, twin, product) or f"Which {twin} would you like for the {product}?"
            if not text:
                return None
            done.add(k if several else -1)
            return (f"{target} ({product})" if several else target), text
        except Exception:
            return None

    def asked_for(self, k: int | None, prop: Any) -> bool:
        """True when the questions log shows the property was asked for product k. Labels lose their bracket tags and
        are split on ';' and 'and'; each part is tied to a product by its tag or the question text, and its name
        must match the property strictly. Unlike asked_inside, unanswered questions count too.
        Errors return False.
        """
        try:
            several = bool(self.needs.items())
            name = re.split(r"\s*\(", str(prop or ""))[0].strip()
            if not name:
                return False
            for ln in self.questions_log:
                label = re.sub(r"\s*\[[^\]]*\]", "", ln.split(":", 1)[0].strip("- "))
                said = re.search(r'asked "(.*?)" -> ', ln)
                for part in (p.strip() for p in re.split(r";|\band\b", label) if p.strip()):
                    j = self.product_of(part) if several else None
                    if several and j is None and said:
                        j = self.product_of(said.group(1))
                    if (not several or k is None or j == k) and strict_prop(re.split(r"\s*\(", part)[0].strip(), name):
                        return True
        except Exception:
            return False
        return False

    def alt_key(self, k: int | None, prop: str) -> str:
        """Finds an alternative property key from the hints for product k: a hint whose property is the same field as,
        or nested with, the given property supplies its '(or ...)' alternatives, and the first alternative with a
        menu line whose key is new (not the property, not spent, not asked) and open in the plan is returned.
        Empty text when none qualifies.
        """
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            if not name:
                return ""
            several = bool(self.needs.items())
            spent = self.__dict__.get("spent_asks", set())
            for hint in self.needs.reading.get("hints") or []:
                head_full, _, detail = str(hint).partition(":")
                head = re.split(r"\s*\(", head_full)[0].strip()
                if not head or (several and self.product_of(head_full) not in (k, None)):
                    continue
                if not (self.same_field(k, name, head) or self.name_inside(name, head)):
                    continue
                for alt in self.hint_alts_of(detail):
                    if junk_key(alt) or strict_prop(alt, head) or strict_prop(alt, name) or self.name_inside(alt, name):
                        continue
                    line = self.prop_line(k, alt)
                    key = line.strip("- ").split(" (in ")[0].strip() if line else ""
                    if not key or strict_prop(key, name) or key.lower() in spent or self.asked_for(k, key) or not self.plan_open(k, key):
                        continue
                    return key
        except Exception:
            return ""
        return ""

    def collide_alt(self, prop: str) -> str:
        """Returns a hint alternative (alt_key) to ask instead of the property, labelled with the product on a
        multi-product request, at most once per product (twin_asks). Empty text when the product is unknown, it
        already had its re-ask, or no alternative exists.
        """
        try:
            several = bool(self.needs.items())
            k = self.product_of(prop) if several else None
            done = self.__dict__.setdefault("twin_asks", set())
            if (several and k is None) or (k if several else -1) in done:
                return ""
            key = self.alt_key(k, prop)
            if not key:
                return ""
            done.add(k if several else -1)
            return f"{key} ({self.needs.items()[k]['product']})" if several else key
        except Exception:
            return ""

    def holds_plain(self, text: Any, plain_q: str) -> bool:
        """True when a question text contains, by word stems, every content word of the plain which-question (frame
        words and link words aside).
        """
        frame = {"which", "would", "you", "like", "for", "the"}
        have = {self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())}
        need = [self.stem(w) for w in re.findall(r"[a-z0-9]+", str(plain_q).lower()) if w not in frame and w not in LINK_WORDS]
        return bool(need) and all(w in have for w in need)

    def alt_untried(self, k: int) -> bool:
        """True when some hint for product k (or for no product) names a property that was already asked and still has
        an alternative key (alt_key) that is unasked, unspent and open in the plan.
        Errors return False.
        """
        try:
            for hint in self.needs.reading.get("hints") or []:
                head_full = str(hint).split(":", 1)[0]
                if self.product_of(head_full) not in (k, None):
                    continue
                head = re.split(r"\s*\(", head_full)[0].strip()
                if head and self.asked_for(k, head) and self.alt_key(k, head):
                    return True
        except Exception:
            return False
        return False

    def quiet_reask(self, prop: str) -> str:
        """Picks a twin property (twin_key) to ask about for a hint-mapped property, at most once per product.
        It holds back while an open hint already carries a no-preference alternative and, on a multi-product
        request, until every other product has had a question. Returns the twin key, with the product label when
        several products are asked for, or empty text.
        """
        try:
            several = bool(self.needs.items())
            k = self.product_of(prop) if several else None
            done = self.__dict__.setdefault("twin_asks", set())
            if (several and k is None) or (k if several else -1) in done or not self.hint_mapped(k, prop):
                return ""
            if any(" drew no preference)" in h for h in self.open_hints(k)):
                return ""
            if several:
                sent = {self.product_of(ln.split(":", 1)[0]) for ln in self.questions_log}
                sent |= {self.product_of(said.group(1)) for ln in self.questions_log for said in [re.search(r'asked "(.*?)" -> ', ln)] if said}
                if any(j != k and j not in sent for j in range(len(self.needs.items()))):
                    return ""
            twin = self.twin_key(k, prop)
            if not twin:
                return ""
            done.add(k if several else -1)
            return f"{twin} ({self.needs.items()[k]['product']})" if several else twin
        except Exception:
            return ""

    @staticmethod
    def menu_trim(menu: str, reads: int) -> str:
        """Drops menu lines found in only 0 or 1 listings once at least 4 reads exist, keeping the menu whole when
        that would leave nothing.
        """
        if reads < 4 or not menu:
            return menu
        kept = [ln for ln in menu.splitlines() if not re.search(r"\(in [01] listings?\)", ln)]
        return "\n".join(kept) if kept else menu

    def first_reads(self) -> bool:
        """Queues a few listing reads per product before any question, so the menu of store fields is known. It runs
        once per product with a known home shelf, needs a details tool and no batch tool; the target is 2 reads when
        the step cap is under MENU_READS_WIDE, else 4, 3 or 2 by product count. Rows come from menu_rows,
        interleaved across products and capped by the steps left after the order reserve and planned questions.
        Sets open_queue, open_recheck, open_say and open_purpose 'menu'; returns True when rows were queued.
        """
        try:
            if self.batch_tool() or not self.role("details"):
                return False
            done_k = self.__dict__.setdefault("first_reads_done_k", set())
            rows = self.needs.items()
            homes = self.home_shelves()
            qtys = [row["quantity"] for row in rows] or [self.needs.quantity()]
            room = self.moves_left() - self.order_due() - 2 * max(0, self.ask_limit() - self.msgs_sent) - 8
            if self.split_pages():
                room -= max(1, len(rows))      # (FRb: layout A keeps one offer-page read per product for the order)
            target = 4 if len(rows) <= 1 else 3 if len(rows) == 2 else 2      # (FR: the wide sample at every step limit)
            wants: list[tuple[int, int, list[str]]] = []
            taken: list[str] = []
            for k in range(len(rows) or 1):
                home = homes[k] if k < len(homes) else set()
                have = self.product_reads(k if rows else None)
                if k in done_k or not home:
                    continue
                if have >= target or room <= 0:
                    done_k.add(k)
                    continue
                picks = self.menu_rows(home, target - have, taken, qtys[min(k, len(qtys) - 1)])
                taken += picks
                if picks:
                    wants.append((have, k, picks))
                else:
                    done_k.add(k)
            queue = [picks[n] for n in range(target) for _, _, picks in sorted(wants) if n < len(picks)][: max(0, room)]
            done_k.update(k for _, k, picks in wants if set(picks) & set(queue))      # (FRc: a product cut by the room comes back)
            if not queue:
                return False
            self.open_queue = queue
            self.open_recheck = True
            self.open_say = "Before asking the shopper, I open a few listings of each product asked for to learn their fields."
            self.open_purpose = "menu"
            self.log.append({"kind": "first_reads", "ids": queue})
            return True
        except Exception as err:
            self.log.append({"kind": "first_reads_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def menu_search(self) -> bool:
        """Before the first question on a multi-product request, searches for products that have no home shelf yet.
        Each such product gets its first planned query (code_queries) once, all sent together as one scripted turn
        with a note to the model. It needs a search tool, no message or question so far, at least CLOSING + 90 s of
        model time and enough steps beyond the order reserve and planned questions. Returns True when searches were
        sent; errors are logged.
        """
        try:
            rows = self.needs.items()
            search = self.role("search")
            if len(rows) < 2 or self.questions_log or self.msgs_sent or not search or self.wire.spare() < CLOSING + 90:
                return False
            done = self.__dict__.setdefault("menu_searched", set())
            homes = self.home_shelves()
            want = [k for k in range(len(rows)) if k not in done and not (homes[k] if k < len(homes) else set())]
            room = self.moves_left() - self.order_due() - 2 * max(0, self.ask_limit() - self.msgs_sent) - 12
            if not want or room < len(want):
                return False
            calls = []
            for k in want:
                done.add(k)
                queries = self.code_queries(k)
                if queries:
                    calls.append((k, queries[0]))
            if not calls:
                return False
            proposals = [{"id": f"menu_search_{k}_{self.wire.tick}", "type": "function",
                          "function": {"name": search, "arguments": json.dumps(call_args)}} for k, call_args in calls]
            names = ", ".join(rows[k]["product"] for k, _ in calls)
            self.chat.append({"role": "assistant", "content": f"Before asking the shopper, I search the store for the {names}: no "
                              "listing of " + ("these products" if len(calls) > 1 else "this product") + " has shown up yet.",
                              "tool_calls": proposals})
            self.log.append({"kind": "menu_search", "products": [k for k, _ in calls], "calls": [c for _, c in calls]})
            self.post_replies(self.dispatch(proposals, scripted=True))
            return True
        except Exception as err:
            self.log.append({"kind": "menu_search_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def near_asked(self, prop: Any) -> bool:
        """True when an answered question already covered the property's core words (core_words, relative to the
        product name) for the same product. Question labels in questions_log are split at ';' and each part is
        compared; questions marked not answered are ignored. On multi-product requests a part must carry the same
        product's tag. Errors return False.
        """
        try:
            rows = self.needs.items()
            k = self.product_of(str(prop)) if rows else None
            product = rows[k]["product"] if rows and k is not None else str(self.needs.reading.get("product") or "")
            mine = core_words(prop, product)
            if not mine:
                return False
            for ln in self.questions_log:
                head = ln.split(":", 1)[0].strip("- ")
                if "[not answered]" in head:
                    continue
                for part in (p.strip() for p in head.split(";") if p.strip()):
                    j = self.product_of(part) if rows else None
                    if rows and j != k:
                        continue
                    if core_words(part, product) == mine:
                        return True
            return False
        except Exception:
            return False

    def settled_prop(self, prop: Any) -> bool:
        """True when settled_keys already marked the property's field key as settled for its product, or for the
        single item (stored under -1). The property name is reduced to its key words first.
        Errors and an empty settled_by record return False.
        """
        try:
            got = self.__dict__.get("settled_by") or {}
            if not got:
                return False
            k = self.product_of(str(prop)) if self.needs.items() else None
            name = " ".join(key_words(re.split(r"\s*\(", str(prop))[0]))
            return bool(name) and name in got.get(-1 if k is None else k, set())
        except Exception:
            return False

    def cover_top(self, scripted: list[dict[str, Any]], ok: Any, remaining: int) -> list[dict[str, Any]]:
        """Makes sure the top ranked properties get asked when questions run short. It takes up to TOP_COVER ranked
        properties that pass ok and were not asked; when no more questions remain than those, and the first planned
        question does not already cover one of them (on the last question, the very first), a which-question for
        the top property is put in front. Returns the scripted questions; errors are logged.
        """
        try:
            top: list[str] = []
            for name in self.rank_props(None):
                if ok(name) and not self.asked_for(None, name):
                    top.append(name)
                if len(top) >= TOP_COVER:
                    break
            if not scripted or not top or remaining > len(top):
                return scripted
            product = str(self.needs.reading.get("product") or "item")
            first = scripted[0] if scripted else {}
            planned = re.split(r"\s*\(", str(first.get("property") or ""))[0].strip()
            last_top = remaining == 1 and not (planned and (strict_prop(planned, top[0]) or core_words(planned, product) == core_words(top[0], product)))
            if planned and not last_top and any(strict_prop(planned, t) or core_words(planned, product) == core_words(t, product) for t in top):
                return scripted
            if planned and not last_top and self.split_pages() \
                    and any(strict_prop(planned, t) or core_words(planned, product) == core_words(t, product) for t in self.rank_props(None)[:8]):
                self.log.append({"kind": "cover_top_kept", "planned": planned[:60], "top": top, "left": remaining})
                return scripted      # (E2c: layout A, the planner's pick among the RANKED entries it was shown stands)
            if planned and self.pair_room(self.ask_limit(), remaining) and not self.must_ask(planned) and not self.twin_follows(planned) \
                    and self.pair_fits(planned, top[0]):
                # (E2b: the planner's pick stays first; the open top-ranked property goes with it as its second)
                self.log.append({"kind": "cover_top_pair", "planned": planned[:60], "second": top[0][:60], "left": remaining})
                return [dict(first, second=top[0])] + list(scripted[1:])
            self.log.append({"kind": "cover_top", "planned": planned[:60], "asked": top[0][:60], "top": top, "left": remaining})
            return [{"property": top[0], "question": f"Which {top[0]} would you like for the {product}?"}] + list(scripted)
        except Exception as err:
            self.log.append({"kind": "cover_top_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return scripted

    def settled_keys(self, k: int | None, rows: list[dict[str, Any]]) -> set[str]:
        """Marks field keys of the given rows that need no question for product k. Each key's seen values and row count
        are gathered first; only plain values of properties still required for that product (props_left) take part,
        reference values aside. A key is settled when it lexically matches such a property whose value is among its
        seen values, or when it shows a single value on two or more rows and that value appears in the request text.
        Settled keys are stored in settled_by per product (-1 for the single item) and returned; errors return an
        empty set.
        """
        try:
            vals: dict[str, set[str]] = {}
            hold: dict[str, int] = {}
            for r in rows:
                for key, val in (r.get("fields") or {}).items():
                    name = " ".join(key_words(key))
                    v = str(val).strip().lower()
                    if name and v:
                        vals.setdefault(name, set()).add(v)
                        hold[name] = hold.get(name, 0) + 1
            several = bool(self.needs.items())
            reqs: list[tuple[str, str]] = []
            for req in self.needs.props_left():
                head, _, value = str(req).partition(":")
                if several and self.product_of(head) != k:
                    continue
                plain = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", value).strip().lower()
                if plain and not ref_text(plain):
                    reqs.append((re.split(r"\s*\(", head)[0].strip(), plain))
            said = " " + " ".join(re.findall(r"[a-z0-9]+", self.request.lower())) + " "
            out: set[str] = set()
            for name, seen in vals.items():
                for prop, plain in reqs:
                    _, lexical = alike_score(prop, name)
                    if lexical and value_seen(plain, " | ".join(sorted(seen))):
                        out.add(name)
                        break
                if name in out or len(seen) != 1 or hold.get(name, 0) < 2:
                    continue
                only = " ".join(re.findall(r"[a-z0-9]+", next(iter(seen))))
                if len(only) >= 3 and f" {only} " in said:
                    out.add(name)
            if out:
                self.__dict__.setdefault("settled_by", {}).setdefault(-1 if k is None else k, set()).update(out)
            return out
        except Exception:
            return set()

    def ref_fixed(self) -> set[int]:
        """On a multi-product request, finds which products a reference fixes: indexes from the product tags of the
        reading's references, else from reference_targets. Returns an empty set when the request has fewer than
        two products or when every product would count as fixed.
        Only the products' indexes are returned, not the references.
        """
        rows = self.needs.items()
        if len(rows) < 2:
            return set()
        out: set[int] = set()
        refs = self.needs.reading.get("references")
        for ref in refs if isinstance(refs, list) else []:
            for tag in re.findall(r"\(([^)]*)\)", str(ref).partition(":")[0]):
                k = self.product_of(f"x ({tag})")
                if k is not None:
                    out.add(k)
        if not out:
            for t in self.needs.reading.get("reference_targets") or []:
                if isinstance(t, dict) and t.get("product"):
                    k = self.product_of(f"x ({t['product']})")
                    if k is not None:
                        out.add(k)
        return out if len(out) < len(rows) else set()

    def plan_open(self, k: int | None, prop: str) -> bool:
        """Tells whether a property is still open for questions on product k, through open_prop; hinted properties are
        judged by their menu samples.
        """
        try:
            hinted = self.hinted_prop(k, prop)
            return self.open_prop(k, prop, self.prop_line(k, prop) if hinted else "", by_samples=hinted)
        except Exception:
            return self.open_prop(k, prop)

    def settles(self, req: str, samples: str = "") -> bool:
        """True when a requirement's value settles its property. Reference wording (as shown, same as, sold by,
        redirects) settles it; a value that is not positive does not, nor one story_guess marks as guessed from the
        request and exchanges. With samples, a word value without digits must match a sample word by stem or within
        one edit.
        """
        value = str(req).partition(":")[2]
        low = value.lower()
        if re.search(r"as shown|same as|the page|sold by|redirect|refer", low) or ref_text(low):
            return True
        if not positive_value(value, str(req).partition(":")[0]):
            return False
        refs = [str(r) for r in self.needs.reading.get("references") or []] if isinstance(self.needs.reading.get("references"), list) else []
        refs += list(self.needs.__dict__.get("resolved") or [])
        if story_guess(req, self.request + " " + " ".join(self.needs.exchanges), refs):
            return False
        if samples and not re.search(r"\d", value):
            words = [w for w in re.findall(r"[a-z]{3,}", low) if w not in LINK_WORDS]
            pool = set(re.findall(r"[a-z]+", samples))
            if words and not any(re.search(r"\b" + re.escape(w[:-1] if len(w) > 4 else w), samples)
                                 or (len(w) >= 4 and any(len(s) >= 4 and edits_within(s, w, 1) for s in pool)) for w in words):
                return False
        return True

    def order_due(self) -> int:
        """Steps that must remain when the first order is due. The base runs from FIRST_ORDER_BY to FIRST_ORDER_DUE by
        step cap, raised to a third of the cap (at most 16), and to 8 plus the product count after an event or a dead
        cart line; the result stays within half the cap, plus one step when the basket is stated first.
        """
        base = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, self.step_cap // 4 + 1))
        due = max(base, min(16, self.step_cap // 3))
        if self.__dict__.get("event_seen") or (self.__dict__.get("dead") and set(self.trolley.units) & self.__dict__["dead"]):
            try:
                due = max(due, 8 + max(1, len(self.needs.items())))
            except Exception:
                due = max(due, 9)
        return max(base, min(due, self.step_cap // 2)) + (1 if self.states_basket() else 0)

    def narrowing(self) -> bool:
        """True when both an order tool and a cart tool are mapped; status_line warns about the order deadline only
        then.
        """
        return bool(self.role("order") and self.role("cart"))

    def offered(self) -> list[dict[str, Any]]:
        """Lists the tool definitions the model may call now. With 1 step left only order and decline; with 3 or fewer
        also the cart; inside the order reserve the cart view, details, stock tool and, within limits, more details
        or search (a swap, unverified picks, named listings, a late widening, backup rows). Trimming applies only
        when order and cart tools exist. The message tool is withheld while the code plans questions (asking).
        """
        spare = self.moves_left()
        call_rank, decl, trolley = self.role("order"), self.role("decline"), self.role("cart")
        name_set: set[str] | None = None
        if spare <= 1:
            name_set = {call_rank, decl}
        elif spare <= 3:
            name_set = {call_rank, decl, trolley}
        elif (spare <= self.order_due() and self.orders_tried == 0) or (spare <= FIRST_ORDER_BY and self.refusal_n > 0):
            name_set = {call_rank, decl, trolley, self.role("cart_view"), self.role("view") or self.role("details"), self.stock_tool()}
            if spare > 3 and self.swap_open():
                name_set.add(self.role("details"))
            late = self.__dict__.get("late_n", 0) < 3
            if late and ((self.unverified() and spare > 5) or (self.named_open() and spare >= min(FIRST_ORDER_BY + 2, self.order_due()))):
                name_set.add(self.role("details"))
            if late and (self.widened_late() or self.__dict__.get("named_pages") or (self.unverified() and spare > 5)) and spare > 4:
                name_set.add(self.role("search"))
            if spare >= 7 and self.backup_rows():
                name_set.add(self.role("details"))
        toolset = self.toolset
        if name_set is not None and call_rank and trolley:
            trimmed = [tdef for tdef in self.toolset if tdef["function"]["name"] in name_set]
            toolset = trimmed or toolset
        if self.asking and self.role("message"):
            toolset = [tdef for tdef in toolset if tdef["function"]["name"] != self.role("message")]
        return toolset

    def in_reserve(self) -> bool:
        """True when order and cart tools exist, more than 3 steps remain and the order reserve is reached (before any
        order, or at FIRST_ORDER_BY steps after a refusal).
        """
        spare = self.moves_left()
        return bool(self.role("order") and self.role("cart")) and spare > 3 and (
            (spare <= self.order_due() and self.orders_tried == 0) or (spare <= FIRST_ORDER_BY and self.refusal_n > 0))

    def unverified(self) -> bool:
        """True when no listing has passed checks at the current requirement version (refused and dead lines aside).
        On a multi-product request it is also True when the passed listings' known shelves leave some product's
        home shelves uncovered.
        """
        dead = self.dead_lines()
        live = [k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == str(self.needs.version)
                and k.rpartition("|")[0] not in self.refused_lines and k.rpartition("|")[0] not in dead]
        if not live:
            return True
        if not self.needs.items():
            return False
        try:
            got = {self.shelf_known(i) for i in live} - {""}
            return bool(got) and any(home and not home & got for home in self.home_shelves())
        except Exception:
            return False

    def named_open(self) -> set[str]:
        """Ids a check named that are in the records but not yet read or refused."""
        return {i for i in self.__dict__.get("named", set()) if i in self.archive.ids} - self.archive.read - self.refused_lines

    def widened_late(self) -> bool:
        """True when the latest requirement change widened the search and came within order_due steps."""
        return self.needs.widened_at == self.needs.version and self.wire.tick - self.needs.widened_tick <= self.order_due()

    def begin_batch(self) -> None:
        """Resets the per-batch count and ids of late reads allowed in the order reserve."""
        self.late_q = 0
        self.late_ids = set()

    def late_hold(self, call_name: str, call_args: dict[str, Any]) -> str:
        """Holds back details and search calls inside the order reserve, returning a refusal note (no step used) or
        empty text to let the call run. Other lookup tools are judged as details. Details of a cart line or a
        passed listing always run. Up to 3 late calls in all may run: details for an unverified pick or a named
        listing, or a search after a late widening, for a named page or while unverified, each with enough steps
        left. Up to 2 cheaper backup rows and the read swap_open names are also allowed; the note lists backup
        rows still open.
        """
        details, search = self.role("details"), self.role("search")
        if call_name not in (details, search) and call_name in self.lookup_tools() and details:
            return self.late_hold(details, call_args)
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
                allowed = (self.widened_late() or json.dumps(call_args, sort_keys=True, default=str) in self.__dict__.get("named_pages", set())
                           or (self.unverified() and spare > 5)) and spare > 4
        if not allowed and call_name == details and spare - 1 >= 6 and self.__dict__.get("backup_n", 0) < 2:
            backups = [i for i in self.backup_rows() if i not in self.__dict__.get("late_ids", set())][: 2 - self.__dict__.get("backup_n", 0)]
            if named & set(backups):
                allowed = True
                self.backup_n = self.__dict__.get("backup_n", 0) + 1
        opened = self.swap_open() if not allowed and call_name == details else ""
        if opened and opened in named and spare > 2:
            allowed = True
            self.swap_done = True
        if allowed:
            self.late_q = queued + 1
            self.__dict__.setdefault("late_ids", set()).update(named)
            return ""
        extra = []
        if call_name == details and spare - 1 >= 6 and self.__dict__.get("backup_n", 0) < 2:
            extra = [i for i in self.backup_rows() if i not in self.__dict__.get("late_ids", set())][: 2 - self.__dict__.get("backup_n", 0)]
        return ("Not run (no step used): the steps left are kept for the order. Now details are read only for the cart's line, a "
                "listing a check named, or (while nothing is verified) up to 3 listings in all: put your best verified fit in the "
                "cart, re-check its stock and price, and order."
                + (" Cheaper unread rows that may back up the pick can still be read: " + ", ".join(extra) + "." if extra else ""))

    def backup_rows(self) -> list[str]:
        """Return up to six unread listing ids priced below the cheapest verified fit, as backups for a cheapest-
        priority request. Runs while fewer than two backups were used (backup_n) and only after a stock event, a
        dead cart line or named unread listings. For each product in cg_products it takes ed_pool entries that are
        unread, not gone, over, pointer, ruled or clashing, priced below that product's lowest verified fitting
        price; sorted by price. Errors log backup_error and give [].
        """
        if self.needs.priority() != "cheapest" or self.__dict__.get("backup_n", 0) >= 2:
            return []
        if not (self.__dict__.get("event_seen") or self.dead_lines() or self.named_open()):
            return []
        out: list[tuple[float, str]] = []
        try:
            for k in self.cg_products():
                pool = self.ed_pool(k)
                tops = [e["price"] for e in pool if e["verified"] and e["fit"] and e["price"] is not None]
                if not tops:
                    continue
                out += [(e["price"], e["id"]) for e in pool if not e["read"] and not e["gone"] and not e["over"] and not e["pointer"]
                        and not e["ruled"] and not e["clash"] and e["price"] is not None and e["price"] < min(tops)]
        except Exception as err:
            self.log.append({"kind": "backup_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []
        return [i for _, i in sorted(out)][:6]

    def field_names(self) -> str:
        """Build the prompt section that lists the field names found on this store's listings (spellings that differ
        only by separators merged, with how many read listings carry each), from archive.prop_menu. Empty string
        when there are none.
        """
        prop_list, _ = self.archive.prop_menu()
        if not prop_list:
            return ""
        return ("\n\nFIELD NAMES ON THIS STORE'S LISTINGS (separator-only spellings merged; how many listings read carry each):\n"
                + prop_list)

    def check_item(self, thing: str, aim_text: str, second: str = "") -> str:
        """Independent listing check: '' on a pass, else why a requirement fails.
        Skipped near the budget end or for a meaning-cleared item.
        Rulings are memoised by item, needs version and read count.
        Reference pointer or field failures reject without a model call.
        Else CHECKER gets the request, requirements and scope,
        reference notes and records, own-note redirects,
        the item's records, the cart view and field evidence.
        A pass records the item in passed with its price.
        A fail returns up to 5 problems.
        Logged as verify.
        """
        if self.wire.spare() < CLOSING + 20:
            return ""
        key = f"{thing}|{self.needs.version}"
        if key in self.__dict__.get("near_ok", set()):
            return ""
        refs = [str(r) for r in self.needs.reading.get("references") or [] if r] if isinstance(self.needs.reading.get("references"), list) else []
        tracked = list(self.__dict__.get("refs") or [])
        own = self.own_redirects(thing)
        memo = key + (f"|{len(self.archive.read)}" if refs or any(r.get("status") != "resolved" for r in tracked) else "") + ("|second" if second else "") \
            + (f"|own{len(own)}" if own else "")
        if memo in self.rulings:
            return self.rulings[memo]
        pointer = self.ref_pointer_note(thing)
        if pointer:
            ruling = f"An independent check says {thing} does NOT meet every requirement: {pointer}. Choose another listing."
            self.rulings[memo] = ruling
            self.log.append({"kind": "verify", "item": thing, "result": {"item": thing, "fits": False, "problems": ["reference chain"]}})
            return ruling
        archive = self.archive.about(thing)
        if not archive:
            return ""
        evidence = self.field_view(thing)
        if evidence["fails"]:
            ruling = (f"An independent check says {thing} does NOT meet every requirement: " + "; ".join(evidence["fails"][:3])
                      + ". Choose another listing.")
            self.rulings[memo] = ruling
            self.log.append({"kind": "verify", "item": thing, "result": {"item": thing, "fits": False, "problems": evidence["fails"][:3],
                                                                        "by": "field evidence"}})
            return ruling
        ask_text = CHECK_ITEM + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request + self.needs.changes()
        ask_text += "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + self.needs.text_body()
        if self.needs.items():
            ask_text += ("\n\nThe item below is one of the products asked for (its category decides which): judge it only against that "
                         "product's requirements and quantity; other products' requirements do not apply to it."
                         + (f" It is in the cart with {int(self.trolley.held(thing))} units." if self.trolley.held(thing) > 0 else ""))
        ref_note = self.ref_note(thing)
        if ref_note:
            ask_text += ref_note
        refs = self.ref_untracked(refs, tracked)
        if refs:
            words = {w for r in refs for w in re.findall(r"[a-z]{4,}", r.lower().split(":", 1)[-1])}
            pointed = [i for i in sorted(self.archive.read) if i != thing and any(
                w in " ".join(self.archive.about(i, 1)).lower() for w in words)][:3]
            if pointed:
                ask_text += ("\n\nRECORDS OF READ LISTINGS THE REQUEST POINTS AT (a requirement given by reference takes its value from "
                             "the field it names in the listing it means):\n" + "\n".join(r for i in pointed for r in self.archive.about(i, 1)))
        ask_text += second
        if own:
            ask_text += "\n\nNOTE (the code read this item's own notes): " + " ".join(
                f"this sentence of its notes may send {prop} to another listing: \"{sent[:240]}\". If it does (it tells the reader to take "
                f"{prop} from, refer to, go by or look to another listing), {prop} comes from that listing, so this item's own "
                f"{prop} field does not count for '{req}', which is then met only when the record of that other listing shows it; a "
                "listing the notes name only as another version sold elsewhere, or as what this item is not, changes nothing."
                for req, prop, sent in own)
        ask_text += f"\n\nTHE ITEM TO {aim_text}: {thing}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n" + "\n".join(archive)
        if self.trolley.last_view is not None:
            ask_text += "\n\nLATEST CART VIEW:\n" + as_text(self.trolley.last_view)[:3000]
        ask_text += self.field_names()
        if evidence["lines"]:
            ask_text += ("\n\nFIELD EVIDENCE (computed by code from this store's records of the item's category; it says how this "
                         "store writes each required value and what the item shows):\n" + "\n".join(evidence["lines"]))
        label_lines = self.archive.label_lines()
        if label_lines:
            ask_text += "\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the item's category label must be the one that matches the request's wording as a whole, not a neighbouring one):\n" + label_lines
        answer = self.wire.consult(ask_text, 3000, CHECKER)
        self.log.append({"kind": "verify", "item": thing, "result": answer})
        if not answer or answer.get("fits") is not False:
            self.rulings[memo] = ""
            if answer:
                self.passed.add(key)
                self.__dict__.setdefault("pass_price", {})[thing] = self.archive.price(thing)
            return ""
        faults = [str(p) for p in answer.get("problems") or []][:5]
        ruling = f"An independent check says {thing} does NOT meet every requirement: " + "; ".join(faults) + ". Unless you are certain the check is wrong, choose another fully verified fit."
        self.rulings[memo] = ruling
        return ruling

    def ch_rows(self) -> dict[str, dict[str, Any]]:
        """Return the code rows of all archived listings (code_rows), cached in ch_rows_memo and rebuilt only when the
        number of archived records changes.
        """
        memo = self.__dict__.get("ch_rows_memo")
        if memo and memo[0] == len(self.archive.items):
            return memo[1]
        rows = self.code_rows()
        self.ch_rows_memo = (len(self.archive.items), rows)
        return rows

    def store_fields(self, label: Any = "") -> dict[str, list[str]]:
        """Collect this store's field values by field name from the code rows, limited to the category labels the item
        belongs to. For several products the labels are the home labels of the product that product_of finds in the
        given label; for one product, own_shelf or else the first home label set. Empty values and junk keys are
        skipped. Returns {field name: [values]}, or {} on any error.
        """
        try:
            rows = self.ch_rows()
            shelves: set[str] = set()
            homes = self.home_shelves()
            if self.needs.items():
                k = self.product_of(str(label))
                shelves = set(homes[k]) if k is not None and k < len(homes) else set()
            else:
                own = self.own_shelf()
                shelves = {own} if own else set(homes[0]) if homes else set()
            out: dict[str, list[str]] = {}
            for row in rows.values():
                if not row["fields"] or (shelves and row["shelf"] not in shelves):
                    continue
                for key, val in row["fields"].items():
                    if str(val).strip() and not junk_key(key):
                        out.setdefault(str(key), []).append(str(val))
            return out
        except Exception:
            return {}

    def ch_redirected(self, thing: str, prop: str) -> bool:
        """Tell whether a listing's own notes may send a property to another listing. When own_redirects is available,
        true if any of its entries for the item names the same property (an error also counts as true). Otherwise it
        scans up to eight records of the item sentence by sentence for the property's words next to a redirect verb
        such as refer, go by, look to or comes from.
        """
        own = getattr(self, "own_redirects", None)
        if callable(own):
            try:
                return any(any(same_prop(str(x).partition(":")[0], prop) for x in (got if isinstance(got, (list, tuple)) else [got])[:2])
                           for got in own(thing) or [])
            except Exception:
                return True
        words = [w for w in prop_words(prop) if len(w) >= 3]
        if not words:
            return False
        named = r"\b" + r"\w*[\s_-]+".join(re.escape(w) for w in words)
        for body in self.archive.about(thing, 8):
            for sentence in re.split(r"(?<=[.!?;])\s+", body.lower()):
                if re.search(named, sentence) and re.search(
                        r"\b(?:refer|taken|take it|go by|goes by|look to|look at|comes? from|read from)\b", sentence):
                    return True
        return False

    def field_view(self, thing: str) -> dict[str, list[str]]:
        """Assemble the code-side field evidence for one listing as {lines, rules, fails}. Starts from field_view_base,
        adds the lines and fails of tie_view, then those of fit_view while dropping earlier fails on properties
        fit_view found met. Last, requirements met by the listing's own category label (cat_covered) lose their
        fails and gain a CATEGORY line. An error in any step is logged and that step is skipped.
        """
        view = self.field_view_base(thing)
        try:
            lines, fails = self.tie_view(thing)
        except Exception as err:
            self.log.append({"kind": "tie_view_error", "error": f"{type(err).__name__}: {err}"[:300]})
            lines, fails = [], []
        if lines or fails:
            view = {"lines": list(view["lines"]) + lines, "rules": list(view["rules"]), "fails": list(view["fails"]) + fails}
        try:
            f_lines, f_fails, f_met = self.fit_view(thing)
        except Exception as err:
            self.log.append({"kind": "fit_view_error", "error": f"{type(err).__name__}: {err}"[:300]})
            f_lines, f_fails, f_met = [], [], []
        if f_lines or f_fails:
            kept = [f for f in view["fails"] if not any(same_prop(f.split(":", 1)[0].strip(), p) for p in f_met)]
            view = {"lines": list(view["lines"]) + f_lines, "rules": list(view["rules"]), "fails": kept + f_fails}
        try:
            covered, shelf = self.cat_covered(thing)
        except Exception as err:
            self.log.append({"kind": "cat_cover_error", "error": f"{type(err).__name__}: {err}"[:300]})
            covered, shelf = [], ""
        if covered:
            props = [re.split(r"\s*\(", r.partition(":")[0])[0].strip() for r in covered]
            kept = [f for f in view["fails"] if not any(same_prop(f.split(":", 1)[0].strip(), p) or strict_prop(f.split(":", 1)[0].strip(), p)
                                                        for p in props)]
            view = {"lines": list(view["lines"]) + [f"- CATEGORY: '{r}' is met by {thing}'s own category label '{shelf}' (a value "
                                                    "of a one-kind category label needs no field)." for r in covered],
                    "rules": list(view["rules"]), "fails": kept}
        return view

    def cat_reqs(self, shelf: str, k: int | None) -> list[str]:
        """List the remaining requirements that a category label meets by itself. Labels that join kinds or carry no
        kind words give nothing. For several products only requirements tagged for product k count. Values that are
        references, contain digits or are not positive are skipped; a value counts when all its kind words appear in
        the label's kind words.
        """
        if not shelf or re.search(JOINERS, shelf.lower()):
            return []
        label = kind_words(shelf)
        if not label:
            return []
        several = bool(self.needs.items())
        out = []
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            if several:
                tags = re.findall(r"\(([^)]*)\)", head)
                if not tags or self.product_of(f"x ({tags[-1]})") != k:
                    continue
            plain = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", value)
            if ref_text(plain) or re.search(r"\d", plain) or not positive_value(plain, head):
                continue
            words = kind_words(plain)
            if words and words <= label:
                out.append(str(req))
        return out

    def cat_covered(self, thing: str) -> tuple[list[str], str]:
        """Return the requirements met by a listing's known category label (via cat_reqs) together with that label.
        With several products and no product match for the listing, no requirements are returned.
        """
        shelf = self.shelf_known(thing)
        k = self.product_index(thing) if self.needs.items() else None
        if self.needs.items() and k is None:
            return [], shelf
        return self.cat_reqs(shelf, k), shelf

    def cat_text(self) -> str:
        """Describe, for each product (or the single request) whose home label set has exactly one label, which
        requirements that category meets by itself so no field has to write them. Parts are joined with semicolons;
        empty when none apply.
        """
        homes = self.home_shelves()
        rows = self.needs.items()
        parts = []
        for k in (range(len(rows)) if rows else [None]):
            home = sorted(homes[k if k is not None else 0]) if homes and (k or 0) < len(homes) else []
            if len(home) != 1:
                continue
            for req in self.cat_reqs(home[0], k):
                parts.append(f"{req} (met by the category '{home[0]}' itself: no field has to write it)")
        return "; ".join(parts)

    def field_view_base(self, thing: str) -> dict[str, list[str]]:
        """Return field_facts for one listing, memoised in ch_view_memo by item, archive size, needs version and the
        remaining requirements; an error logs field_facts_error and gives empty lists. When a meaning check
        confirmed some of the item's values for this needs version (near_said), matching field lines get a note that
        the value is met in other words and fails on those properties are dropped.
        """
        key = (thing, len(self.archive.items), self.needs.version, "|".join(self.needs.props_left()))
        memo = self.__dict__.setdefault("ch_view_memo", {})
        if key not in memo:
            try:
                memo[key] = self.field_facts(thing)
            except Exception as err:
                self.log.append({"kind": "field_facts_error", "error": f"{type(err).__name__}: {err}"[:300]})
                memo[key] = {"lines": [], "rules": [], "fails": []}
        said = (self.__dict__.get("near_said") or {}).get(f"{thing}|{self.needs.version}")
        if not said:
            return memo[key]
        try:
            view = memo[key]
            props = [p for p, _, _ in said]
            lines = []
            for ln in view["lines"]:
                head = re.match(r"- (.+?) = '", ln)
                hit = next(((p, v, s) for p, v, s in said if head and (same_prop(head.group(1), p) or strict_prop(head.group(1), p))), None)
                lines.append(ln + (f" A meaning check confirmed its '{hit[2][:80]}' as '{hit[1][:60]}' (the same value in other words): it "
                                   "meets it." if hit else ""))
            fails = [f for f in view["fails"] if not any(same_prop(f.split(":", 1)[0].strip(), p) or strict_prop(f.split(":", 1)[0].strip(), p)
                                                         for p in props)]
            return {"lines": lines, "rules": list(view["rules"]), "fails": fails}
        except Exception:
            return memo[key]

    def field_facts(self, thing: str) -> dict[str, list[str]]:
        """{lines, rules, fails}: how the store writes each open requirement and
        whether one listing shows it.
        Needs fields on the item's row and 3+ read listings of its category.
        At most 12 requirements; no reference wording.
        A numeric limit fails when peers carry a field this listing lacks.
        Other values: the field by name, or peers writing it.
        A line says met by field, key or title, redirected, excluded,
        no such field, or joined with a value only peers carry.
        A field named for the property decides a fail.
        Used by the checks and audits.
        """
        out: dict[str, list[str]] = {"lines": [], "rules": [], "fails": []}
        rows = self.ch_rows()
        me = rows.get(thing)
        if not me or not me["fields"]:
            return out
        several = bool(self.needs.items())
        shelf = me["shelf"]
        if several and not shelf:
            return out
        k = next((n for n, home in enumerate(self.home_shelves()) if shelf in home), None) if several else None
        peers = {i: r for i, r in rows.items() if r["fields"] and (not shelf or r["shelf"] == shelf)}
        if len(peers) < 3:
            return out
        carry: dict[str, int] = {}
        table: dict[str, list[tuple[str, str, list[str]]]] = {}
        for i, r in peers.items():
            for key, val in r["fields"].items():
                if str(val).strip() and not junk_key(key):
                    carry[key] = carry.get(key, 0) + 1
                    elems = ch_parts(val)
                    elems += [str(val).strip().lower()] if len(elems) > 1 else []
                    table.setdefault(key, []).extend((i, p, ch_words(p)) for p in elems)

        def clip(text: Any, n: int = 70) -> str:
            """Collapse whitespace in a value and cut it to n characters (70 by default), ending with an ellipsis when cut."""
            text = " ".join(str(text).split())
            return text if len(text) <= n else text[: n - 3] + "..."
        for req in self.needs.props_left()[:12]:
            head, _, raw = str(req).partition(":")
            value = " ".join(re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw).split())
            prop = re.split(r"\s*\(", head)[0].strip()
            if not prop or not value or ref_text(value) or re.search(r"as shown|same as|the page|sold by|redirect|refer", value.lower()):
                continue
            if re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
                j = self.product_of(head) if several else None
                own_w = set(prop_words(prop))
                named_b = [key for key in carry if carry[key] >= 2 and strict_prop(key, prop)]
                kin_b = {w for w in own_w if len(w) >= 3 and w not in LINK_WORDS}
                if prop and kin_b and named_b and not (several and j is not None and j != k) and not any(
                        str(v).strip() and (strict_prop(key, prop) or set(prop_words(key)) & kin_b) for key, v in me["fields"].items()):
                    keys_b = " / ".join(f"'{x}'" for x in named_b[:3])
                    out["lines"].append(f"- {prop} = '{clip(value)}': this store's field for it is {keys_b}, on "
                                        f"{sum(1 for r in peers.values() if any(str(r['fields'].get(x, '')).strip() for x in named_b))} of "
                                        f"{len(peers)} read listings of this category. This listing has no such field, so the limit is not "
                                        "shown met: it does NOT meet it (another field's figure does not stand in).")
                    out["fails"].append(f"{prop}: it has no {named_b[0]} field, which this store's listings of the category carry, so "
                                        f"'{clip(value)}' is not shown met")
                continue
            if re.fullmatch(r"\s*\d+(?:\.\d+)?\s+[a-z]+\s*", value.lower()):
                continue
            j = self.product_of(head) if several else None
            if several and j is not None and j != k:
                continue
            positive = positive_value(value, head)
            if not positive and not re.search(EXCLUDE_WORDS, value.lower()):
                continue
            alts = ch_alts(value)
            if not alts:
                continue
            named = sorted((key for key in carry if strict_prop(key, prop)), key=lambda x: -carry[x])
            by_name = bool(named)
            if not positive and by_name:
                flat = " ".join(re.findall(r"[a-z0-9]+", value.lower()))
                if any(" ".join(words) == flat for key in named for _, _, words in table.get(key, [])):
                    positive = True
                    LITERAL_VALUES.add(flat)
            if not named:
                if not positive or (several and j is None):
                    continue
                found = [key for key in carry if carry[key] >= 2 and any(ch_match(alts, words) for _, _, words in table.get(key, []))]
                if not found:
                    continue
                top = max(found, key=lambda x: carry[x])
                named = sorted((key for key in carry if strict_prop(key, top)), key=lambda x: -carry[x])
            on = sum(1 for r in peers.values() if any(str(r["fields"].get(key, "")).strip() for key in named))
            writers = sorted({i for key in named for i, _, words in table.get(key, []) if i != thing and ch_match(alts, words)})
            alone = {" ".join(words) for key in named for i, _, words in table.get(key, []) if i != thing}
            samples = list(dict.fromkeys(clip(r["fields"][key], 40) for r in peers.values() for key in named
                                         if str(r["fields"].get(key, "")).strip()))[:6]
            keys_txt = " / ".join(f"'{x}'" for x in named[:3])
            tail = f"{len(writers)} other listing(s) write '{clip(value)}' itself in it."
            if not positive:
                bare = " ".join(re.sub(EXCLUDE_WORDS, " ", value.lower()).split())
                bare_alts = ch_alts(bare)
                held = {i for key in named for i, _, words in table.get(key, []) if i != thing and bare_alts and ch_match(bare_alts, words)}
                if bare_alts:
                    tail = (f"{len(held)} other listing(s) write '{clip(bare)}' there, the value it excludes (a listing that writes "
                            "it does not meet it).")
            base = (f"- {prop} = '{clip(value)}': this store's field for it is {keys_txt}"
                    + ("" if by_name else f" (no field is named '{prop}'; this one's values write the value)")
                    + f", on {on} of {len(peers)} read listings of this category; " + tail)
            own_kinds = kind_words(value)
            naming = [] if writers or not own_kinds else [
                i for i, r in rows.items() if i != thing and (not shelf or r["shelf"] == shelf) and own_kinds <= r["words"]][:3]
            if not positive:
                out["rules"].append(base)
            elif writers:
                out["rules"].append(base + " So only a value that writes those words meets it (containment and one list element "
                                    "count; a synonym or related word does not)." + ("" if by_name else " With no field named for "
                                    "the property, a listing whose title writes the value meets it too, unless a field of its own "
                                    "that shares the property's name says otherwise."))
            elif naming:
                out["rules"].append(base + " Its values seen: " + " | ".join(samples) + f"; listings of this category name '{clip(value)}' "
                                    f"in their title or text ({', '.join(naming)}), so the store has its own word for it: a synonym or "
                                    "related value meets it only once such a listing's record shows this field writes it otherwise.")
            elif on < CH_SYNONYM_MIN:
                out["rules"].append(base + " Its values seen: " + " | ".join(samples) + f"; none writes it, but too few read listings "
                                    f"carry this field to tell whether this store writes '{clip(value)}' itself: a value of the same "
                                    "meaning is not shown to meet it yet.")
            else:
                out["rules"].append(base + " Its values seen: " + " | ".join(samples) + "; none writes it, so a value of the same "
                                    "meaning may meet it.")
            mine = [(key, str(me["fields"][key])) for key in named if str(me["fields"].get(key, "")).strip()]
            shown = "; ".join(f"{key}: '{clip(v, 90)}'" for key, v in mine)
            own_words = prop_words(prop)
            qualified = [(key, str(v)) for key, v in me["fields"].items() if own_words and key not in named and str(v).strip()
                         and not junk_key(key) and len(prop_words(key)) == len(own_words) + 1 and prop_words(key)[1:] == own_words]
            qual_writes = [(key, v) for key, v in qualified if ch_writes(value, v)]
            qual_near = bool(qual_writes) or any(ch_touches(value, v) for _, v in qualified)
            qual_note = (f" Its {qual_writes[0][0]} '{clip(qual_writes[0][1], 60)}' writes it (a qualified key of the same property: "
                         "a field named for the property with one more leading word is the same property's field under its fuller "
                         "name), so it meets it."
                         if qual_writes else "")
            if self.ch_redirected(thing, prop):
                out["lines"].append(base + " This listing's own notes may send this property to another listing, so its own "
                                    "field may not decide.")
                continue
            if not positive:
                out["lines"].append(base + " This listing: " + (shown or "no such field") + " (an exclusion: the listing must not "
                                    "carry the excluded value).")
                continue
            kin_words = {w for w in prop_words(prop) if len(w) >= 3 and w not in LINK_WORDS}
            kin = [key for key, v in me["fields"].items() if str(v).strip() and not junk_key(key) and set(prop_words(key)) & kin_words
                   and not ch_writes(value, v)]
            title_says = not by_name and not kin and ch_writes(value, me.get("title") or "")
            if not mine and qual_writes:
                out["lines"].append(base + " This listing has no such field." + qual_note)
                continue
            if not mine and title_says:
                out["lines"].append(base + f" This listing has no such field, and no field is named for the property: its title "
                                    f"writes '{clip(value)}': it meets it.")
                continue
            if not mine:
                pw = {w for w in prop_words(prop) if len(w) >= 3}
                related = sorted(((key, v) for key, v in me["fields"].items() if key not in named and str(v).strip() and not junk_key(key)
                                  and (set(prop_words(key)) & pw or ch_writes(value, v))), key=lambda kv: not ch_writes(value, kv[1]))[:4]
                out["lines"].append(base + " This listing has no such field" + (
                    ": its fields that may name the property: " + "; ".join(f"{key}: '{clip(v, 60)}'" for key, v in related)
                    if related else "") + ". Another name of the SAME property counts; a field of another property (a height "
                    "for a type, a frame material for the material) does not stand in.")
                continue
            parts = [p for _, v in mine for p in ch_parts(v)] + [v.strip().lower() for _, v in mine if len(ch_parts(v)) > 1]

            def knocked(part: str) -> list[str]:
                """For one part of a field value that splits into sides with one side matching the requested value, return the
                other sides that do not match and that other listings write alone in this field; empty otherwise.
                """
                sides = ch_sides(part)
                if not sides or not any(ch_match(alts, ch_words(s)) for s in sides):
                    return []
                return [s for s in sides if not ch_match(alts, ch_words(s)) and " ".join(ch_words(s)) in alone]
            clean = [p for p in parts if ch_match(alts, ch_words(p)) and not knocked(p)]
            joined = [(p, knocked(p)) for p in parts if ch_match(alts, ch_words(p)) and knocked(p)]
            if clean:
                out["lines"].append(base + f" This listing: {shown} - it writes '{clip(value)}': it meets it.")
            elif qual_writes:
                out["lines"].append(base + f" This listing: {shown}." + qual_note)
            elif title_says:
                out["lines"].append(base + f" This listing: {shown} - with no field named for the property, its title writes "
                                    f"'{clip(value)}': it meets it.")
            elif joined:
                part, others = joined[0]
                out["lines"].append(base + f" This listing: {shown} - '{clip(part)}' joins '{clip(value)}' with '{others[0]}', which "
                                    "other listings carry alone in this field: " + ("this field does not meet it." + qual_note
                                                                                     if qual_writes else "it does NOT meet it."))
                if by_name and not qual_near:
                    out["fails"].append(f"{prop}: its {named[0]} is '{clip(part)}', which joins {clip(value)} with {others[0]} (a "
                                        f"value this store also writes alone there), so it is not {clip(value)}")
            elif writers:
                out["lines"].append(base + f" This listing: {shown} - it does not write '{clip(value)}', which other listings of "
                                    "this category write in this field: " + ("this field does not meet it." + qual_note if qual_writes
                                                                             else "it does NOT meet it (a synonym or related word does not "
                                                                             "count where the store writes the value itself)."))
                if by_name and not qual_near and not any(ch_touches(value, v) for _, v in mine):
                    out["fails"].append(f"{prop}: its {named[0]} is '{clip(mine[0][1], 60)}', while this store writes "
                                        f"'{clip(value)}' itself in that field on other listings")
            elif naming:
                out["lines"].append(base + f" This listing: {shown} - it does not write '{clip(value)}'. No read listing writes it in "
                                    "this field yet (its values: " + " | ".join(samples) + f"), but listings of this category name "
                                    f"'{clip(value)}' in their title or text ({', '.join(naming)}): the store has its own word for it, "
                                    "so this listing's other value is not shown to meet it.")
            elif on < CH_SYNONYM_MIN:
                out["lines"].append(base + f" This listing: {shown}. No read listing of this category writes '{clip(value)}' in "
                                    "this field (its values: " + " | ".join(samples) + f"), but only {on} read listing(s) carry it: too "
                                    f"few to tell whether this store writes '{clip(value)}' itself, so a value of the same meaning is "
                                    "not shown to meet it yet.")
            else:
                out["lines"].append(base + f" This listing: {shown}. No read listing of this category writes '{clip(value)}' in "
                                    "this field (its values: " + " | ".join(samples) + "), so a value of the same meaning may meet it.")
        return out

    def rank_check(self, thing: str) -> str:
        """CHECKER: may other seen listings fit and beat the pick on priority?
        Once per item/needs version/refusal count (rank_checked), with
        priority, CLOSING + 30 spare budget, 4+ moves. RANK_CHECK: request,
        needs, reference note, multi-product same-product rule, listing cards
        (cheaper if cheapest), field names/evidence rules. <= 4 better ids not
        refused, dead or field-check failing (drops: rank_dropped) join
        rank_better (max 8) and named. Returns a check-before-order note or ''.
        """
        priority = self.needs.priority()
        key = f"{thing}|{self.needs.version}|{self.refusal_n}"
        if priority == "none" or key in self.rank_checked or self.wire.spare() < CLOSING + 30 or self.moves_left() < 4:
            return ""
        self.rank_checked.add(key)
        detail_note = self.needs.reading.get("priority_detail") or ""
        what = "cheapest (lowest price)" if priority == "cheapest" else f"best ({detail_note})" if detail_note else "best"
        ask_text = RANK_CHECK.replace("PRIORITY", what) + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request + self.needs.changes()
        ask_text += "\n\nCURRENT REQUIREMENTS:\n" + self.needs.text_body()
        ask_text += self.ref_note()
        if self.needs.items():
            ask_text += ("\n\nSeveral products are asked for: only listings of the SAME product (category) as the chosen item compete "
                         "with it; listings of the other products never beat it.")
        ask_text += (f"\n\nTHE CHOSEN ITEM: {thing}\n\nLISTINGS SEEN (one card each; DETAILS = its full record was read, ROW = seen only "
                     "in search rows, fields unknown: a ROW on the requested category (whatever object its title names) that beats the chosen item and is not ruled out by "
                     "its title, price or stock MAY fit and is named" + ("; only listings priced below the chosen one are shown"
                     if priority == "cheapest" else "") + "):\n"
                     + "\n".join(self.archive.cards(20000, thing if priority == "cheapest" else "", self.ref_foreign(), {thing})))
        ask_text += self.field_names()
        rules = self.field_view(thing)["rules"]
        if rules:
            ask_text += ("\n\nFIELD EVIDENCE (computed by code from this store's records of the chosen item's category: how this "
                         "store writes each required value):\n" + "\n".join(rules))
        answer = self.wire.consult(ask_text, 2500, CHECKER)
        self.log.append({"kind": "rank", "result": answer})
        dead = self.dead_lines()
        better = [str(x) for x in (answer or {}).get("better") or [] if isinstance(x, (str, int)) and str(x) != thing
                  and str(x) not in self.refused_lines and str(x) not in dead][:4]
        dropped = {x: self.field_view(x)["fails"][:2] for x in better if self.field_view(x)["fails"]}
        if dropped:
            self.log.append({"kind": "rank_dropped", "items": {x: [str(f)[:160] for f in v] for x, v in dropped.items()}})
        better = [x for x in better if x not in dropped]
        self.rank_better = list(dict.fromkeys(list(better) + [x for x in self.__dict__.get("rank_better") or []
                                                               if x not in self.dead_lines() and x not in self.refused_lines]))[:8]
        self.__dict__.setdefault("named", set()).update(better)
        if not better:
            return ""
        return (f"A rank check thinks these listings may also fit and beat {thing} on the {what}: " + ", ".join(better)
                + (f" ({answer.get('reason')})" if answer.get("reason") else "")
                + ". Check them (details, stock, price) before ordering; if each fails, order the current pick.")

    def review_decline(self, early: bool = False) -> str:
        """Audits the seen listings before a decline; returns a note of what to try first.
        Runs once, again only after new reads (at most 2 audits), with spare time.
        AUDITOR gets the request, requirements, reference note and listing cards,
        new cards first, and the field names.
        Up to 5 fits without field fails become audit_fits.
        Unopened picks: slack rows, top search hits and the model's unopened ids.
        With moves to spare it also names unread search pages and new queries.
        Named ids join the named list.
        Returns '' when nothing is named.
        """
        reread = len(self.archive.read) > self.__dict__.get("audit_reads", 10 ** 9) and self.__dict__.get("audits", 0) < 2
        if (self.audit_done and not reread) or self.wire.spare() < CLOSING + 30:
            return ""
        self.audit_done = True
        if not early:
            self.audits = self.__dict__.get("audits", 0) + 1
        cards = self.archive.cards(16000, "", self.ref_foreign(), set(self.archive.read) - set(self.__dict__.get("audit_ids") or ())
                                   if self.__dict__.get("audit_ids") is not None else ())
        self.audit_reads = len(self.archive.read)
        if not cards:
            return ""
        before = self.__dict__.get("audit_ids")
        fresh = sorted(self.archive.read - before) if before is not None else []
        self.audit_ids = set(self.archive.read)
        ask_text = BEFORE_DECLINE + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request + self.needs.changes()
        ask_text += "\n\nCURRENT REQUIREMENTS:\n" + self.needs.text_body()
        if self.ties:
            try:
                ask_text += self.series_line().replace("\n[part series] ", "\n\nPART SERIES: ")      # (CS)
            except Exception as err:
                self.log.append({"kind": "series_error", "error": f"{type(err).__name__}: {err}"[:300]})
        ask_text += self.ref_note()
        ask_text += "\n\nLISTINGS SEEN (one card each; DETAILS = its full record was read, ROW = seen only in search rows):\n" + "\n".join(cards)
        if fresh:
            ask_text += ("\n\nOPENED SINCE THE LAST CHECK (their DETAILS cards above are new: judge each of them first, field by field, "
                         "by the reading rules): " + ", ".join(fresh))
        ask_text += self.field_names()
        answer = self.wire.consult(ask_text, 6000, AUDITOR)
        self.log.append({"kind": "audit", "result": answer})
        conforms = [str(x) for x in (answer or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        conforms = [x for x in conforms if not self.field_view(x)["fails"]]
        self.audit_fits = conforms
        unopened = [str(x) for x in (answer or {}).get("unopened") or [] if isinstance(x, (str, int))
                    and str(x) not in self.archive.read and str(x) in self.archive.ids][:6]
        shelves, cap, qty, hit_shelf = self.shelf_labels(), self.needs.spend_cap, self.needs.quantity(), self.__dict__.get("hit_shelf", {})
        hits = [i for i in sorted(self.__dict__.get("all_hits", []), key=lambda i: -self.__dict__.get("hit_score", {}).get(i, 0)) if i not in self.archive.read
                and not (shelves and hit_shelf.get(i) and hit_shelf.get(i) not in shelves)
                and not (is_number(cap) and is_number(self.archive.price(i)) and self.archive.price(i) * qty > cap)]
        try:
            slack = [i for i in self.slack_rows() if i not in self.archive.read]
        except Exception:
            slack = []
        unopened = list(dict.fromkeys(slack + hits[:4] + [i for i in unopened if i not in hits]))[:6]
        pages = self.open_pages(2) if self.moves_left() > 5 else []
        if pages:
            unopened = unopened[:4]
            self.__dict__.setdefault("named_pages", set()).update(json.dumps(p, sort_keys=True, default=str) for p in pages)
        try:
            combos = self.combined_queries() if self.moves_left() > 5 else []
        except Exception as err:
            self.log.append({"kind": "combo_error", "error": f"{type(err).__name__}: {err}"[:300]})
            combos = []
        if combos:
            self.__dict__.setdefault("named_pages", set()).update(json.dumps(p, sort_keys=True, default=str) for p in combos)
        self.__dict__.setdefault("named", set()).update(conforms + unopened)
        named_list = self.__dict__.setdefault("named_list", [])
        named_list += [i for i in conforms + unopened if i not in named_list]
        if not conforms and not unopened and not pages and not combos:
            return ""
        said = []
        if conforms:
            said.append("An independent check thinks these listings may meet every requirement: " + ", ".join(conforms)
                        + (f" ({answer.get('reason')})" if answer.get("reason") else "") + ".")
        if pages:
            said.append("These searches carry required values and matched more listings than were read, so fitting ones may sit on "
                        "their unread pages: " + "; ".join(f"{self.role('search')} {json.dumps(p, ensure_ascii=False)}" for p in pages)
                        + ". Read those pages first (one step shows a whole page of rows) and open the in-budget rows of the requested kind they show.")
        if combos:
            said.append("These searches combine the required values and were never run (when every word is required, each lists only "
                        "the listings carrying all of them): " + "; ".join(f"{self.role('search')} {json.dumps(p, ensure_ascii=False)}" for p in combos)
                        + ". Run them (the search tool stays usable for them) and open the in-budget rows of the requested kind they show.")
        if unopened:
            said.append("These listings of the requested kind were seen only in search rows and never opened, so their fields "
                        "are unchecked: " + ", ".join(unopened) + ". Open their details first (several in one reply).")
        return " ".join(said) + " Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."

    def named_unread(self) -> list[str]:
        """List listings that searches or checks named but the agent has not read: search hits by hit_score, then
        named_list. Read, refused or unknown ids are dropped, as are ones over the spend cap at the needed quantity
        and ones whose search row carried a category label outside the request's labels.
        """
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
        """Build the details tool's arguments for one listing id. Reuses the latest logged details call whose arguments
        held a known listing id, with the new id put in its place; otherwise uses the schema's required free-text
        argument, or its only one. Returns None when no argument fits.
        """
        tool = self.role("details")
        for entry in reversed(self.log):
            for act in (entry.get("actions") or []) if entry.get("kind") == "env" else []:
                args = act.get("args") if act.get("name") == tool else None
                if isinstance(args, dict) and any(isinstance(v, str) and v in self.archive.ids for v in args.values()):
                    return {k: (ident if isinstance(v, str) and v in self.archive.ids else v) for k, v in args.items()}
        jschema = self.shapes.get(tool or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
        key = next((k for k in texts if k in (jschema.get("required") or [])), None) or (texts[0] if len(texts) == 1 else None)
        return {key: ident} if key else None

    def open_reads(self, queue: list[str], say: str, batch: tuple[str, str, int] | None, reader: str | None = None) -> list[tuple[str, str]]:
        """Send scripted read calls for a queue of listing ids and return the tool replies. With a batch spec (tool,
        array argument, ids per call) the ids are grouped into list arguments; otherwise each id gets its own call
        built by lookup_call for the reader or the details tool. An assistant message with the say text and calls
        goes into the chat, the calls are dispatched with covering set, and the newly archived records are kept in
        last_reads.
        """
        if batch:
            tool, arg, most = batch
            calls = [{arg: queue[i: i + most]} for i in range(0, len(queue), most)]
        else:
            tool = reader or self.role("details")
            calls = [args for args in (self.lookup_call(i, tool) for i in queue) if args] if tool else []
        proposals = [{"id": f"open_{n}_{self.wire.tick}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
                     for n, args in enumerate(calls)]
        if not proposals:
            return []
        self.chat.append({"role": "assistant", "content": say, "tool_calls": proposals})
        before = len(self.archive.items)
        self.covering = True
        try:
            responses = self.dispatch(proposals, scripted=True)
        finally:
            self.covering = False
        self.last_reads = self.archive.items[before:]
        return responses

    def open_named(self) -> bool:
        """Pre-turn scripted reads of queued listings; True if replies posted.
        open_queue, else put_queue (post-put stock re-reads), with recheck flag,
        purpose, floor, reader tool, say text. Menu/quota/walk reads cap at moves
        past order reserve (order_due; FIRST_ORDER_BY after an order try), logged
        reserve_skip. Cart reads keep stale lines and, bar put re-reads, uncarted
        ids. 2+ ids batch if allowed (cart: proven batch; custom reader:
        stock-showing); a first batch batch_ok rejects clears batch_spec, then a
        few single reads. No recheck: review_decline note on last reply.
        """
        queue = self.__dict__.pop("open_queue", None)
        put_read = False
        if not queue and self.__dict__.get("put_queue") and not self.wire.done and not self.wire.broke:
            queue, put_read = self.__dict__.pop("put_queue"), True
            self.open_recheck, self.open_purpose, self.open_tool = True, "cart", self.__dict__.pop("put_tool", None)
            self.open_say = "I read the stock of what I just put in the cart: a listing can sell out right after it is put in."
        if not queue or self.wire.done or self.wire.broke:
            return False
        queue = list(dict.fromkeys(str(i) for i in queue))
        recheck = self.__dict__.pop("open_recheck", False)
        purpose = str(self.__dict__.pop("open_purpose", "") or "")
        floor = self.__dict__.pop("open_floor", None)
        reader = self.__dict__.pop("open_tool", None)
        say = self.__dict__.pop("open_say", "") or ("Before ordering, I re-read the cart lines." if recheck
                                                   else "Before declining, I open the listings the checks named.")
        if purpose in ("menu", "quota", "walk") and self.role("order") and self.role("cart") and not self.wire.mute:
            try:
                keep = self.order_due() if self.orders_tried == 0 else FIRST_ORDER_BY
                room = self.moves_left() - keep - 1
                if room < (1 if self.batch_tool() and len(queue) >= 2 else len(queue)) and (self.trolley.units or not self.unverified()):
                    trimmed = queue[: max(0, room)] if not (self.batch_tool() and len(queue) >= 2) else []
                    self.log.append({"kind": "reserve_skip", "purpose": purpose, "ids": queue[:6], "kept": trimmed})
                    queue = trimmed
                    if not queue:
                        return False
            except Exception as err:
                self.log.append({"kind": "reserve_skip_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if purpose == "cart":
            stale = set(self.stale_lines())
            queue = [i for i in queue if i in stale or (i not in self.trolley.units and not put_read)]
            if not queue:
                return False
        batch = self.batch_tool() if len(queue) >= 2 and purpose != "pages" else None      # (A2: each page through its own tool)
        if batch and purpose == "cart" and not self.__dict__.get("batch_proven"):
            batch = None
        if batch and reader and not ((self.__dict__.get("tool_caps") or {}).get(batch[0]) or {}).get("stock"):
            batch = None
        responses = self.open_reads(queue, say, batch, reader)
        if batch and responses and not self.__dict__.get("batch_proven"):
            if self.batch_ok(self.__dict__.get("last_reads") or [], queue):
                self.batch_proven = True
            else:
                self.batch_spec = None
                self.log.append({"kind": "batch_dropped", "tool": batch[0]})
                most = {"menu": 0, "quota": 2, "walk": 2}.get(purpose, 3)
                keep = int(floor) if is_number(floor) else self.order_due() if purpose in ("menu", "quota") else 3
                n = min(most, len(queue), self.moves_left() - keep - 1)
                if n > 0 and not self.wire.done and (reader or self.role("details")):
                    self.post_replies(responses)
                    responses = self.open_reads(queue[:n], say, None, reader)
                    if not responses:
                        return True
        if not responses:
            return False
        note = self.review_decline() if self.wire.spare() > CLOSING + 30 and not recheck else ""
        if note and responses:
            cid, text_body = responses[-1]
            responses[-1] = (cid, text_body + "\n[check after opening] " + note)
        self.post_replies(responses)
        return True

    def last_stock(self, ident: str) -> bool | None:
        """Return the latest stock state recorded for one listing across all archived records: rows are matched by
        row_mark and the last stock_state that is not None wins. None when no record states it; rank_verify skips
        listings for which this is False.
        """
        state: bool | None = None

        def visit(d: dict[str, Any]) -> None:
            """Update the enclosing stock state from one dict when its row mark is the listing id and stock_state gives a
            value.
            """
            nonlocal state
            if self.row_mark(d) == ident:
                got = self.stock_state(d)
                if got is not None:
                    state = got
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return state

    def rank_verify(self, thing: str) -> str:
        """For a cheapest-priority request, item-check up to two rank_better listings that are read, not yet passed for
        this needs version, not refused, dead or seen out of stock, free of field fails, and priced within the spend
        cap at their product's quantity. Then returns cheaper_fit for the chosen item. Errors log rank_verify_error
        and give an empty string.
        """
        try:
            if self.needs.priority() != "cheapest":
                return ""
            ver = str(self.needs.version)
            cap = self.needs.spend_cap
            done = 0
            for x in self.__dict__.get("rank_better") or []:
                if done >= 2 or self.wire.spare() < CLOSING + 30:
                    break
                if x not in self.archive.read or f"{x}|{ver}" in self.passed or x in self.refused_lines or x in self.dead_lines() \
                        or self.last_stock(x) is False or self.field_view(x)["fails"]:
                    continue
                k = self.product_index(x) if self.needs.items() else None
                qty = self.needs.items()[k]["quantity"] if k is not None and k < len(self.needs.items()) else self.needs.quantity()
                cost = self.archive.price(x)
                if not is_number(cost) or (is_number(cap) and is_number(qty) and cost * qty > cap):
                    continue
                done += 1
                self.check_item(x, "ORDER")
            return self.cheaper_fit(thing)
        except Exception as err:
            self.log.append({"kind": "rank_verify_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def cheaper_fit(self, thing: str) -> str:
        """For a cheapest-priority request, return the lowest-priced listing that passed the item check for the current
        needs version and costs less than the given item. Skips the item itself, refused and dead lines, listings
        with field fails and, with several products, listings on another category label. Gives an empty string when
        none is cheaper, the priority differs or the item's price is unknown.
        """
        if self.needs.priority() != "cheapest" or not thing:
            return ""
        mine = self.archive.price(thing)
        if mine is None:
            return ""
        kind = self.shelf_known(thing)
        best: tuple[str, float] | None = None
        dead = self.dead_lines()
        for key in self.passed:
            ident, _, version = key.rpartition("|")
            if version != str(self.needs.version) or ident == thing or ident in self.refused_lines or ident in dead:
                continue
            if self.needs.items() and (not kind or self.shelf_known(ident) != kind):
                continue
            cost = self.archive.price(ident)
            if cost is not None and cost < mine and (best is None or cost < best[1]):
                if self.field_view(ident)["fails"]:
                    continue
                best = (ident, cost)
        return best[0] if best else ""

    def shelf_known(self, thing: str) -> str:
        """Return a listing's category label: the last label shelf_of finds in an archived dict whose id-named key
        holds the listing id, else the archive's label_of entry, lowercased. Empty when unknown. Used to keep swaps,
        cheaper picks and product matches on the same category.
        """
        found = ""

        def visit(d: dict[str, Any]) -> None:
            """Keep the category label of a dict whose id-named key equals the listing id, if it has one."""
            nonlocal found
            if any(isinstance(val, str) and val == thing and "id" in str(key).lower() for key, val in d.items()):
                found = self.shelf_of(d) or found
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return found or str((self.archive.__dict__.get("label_of") or {}).get(thing) or "").strip().lower()

    def chosen(self) -> str | None:
        """Return the id of the line added to the cart most recently, or None when the cart is empty."""
        if self.trolley.units:
            return list(self.trolley.units)[-1]
        return None

    def shelf_names(self) -> set[str]:
        """K1: the category names the request gives its products ("1 of the same listing from Fishing Lines", "(from
        Internal SSDs)") and the products' own names, as shelf_norm writes them."""
        memo = self.__dict__.get("shelf_names_memo")
        if memo is not None:
            return memo
        names: set[str] = set()
        try:
            for hit in re.finditer(r"\bfrom\s+([^;.()\n]{2,80}?)\s*(?=[;.)\n]|$)", str(self.request or "")):
                names.add(shelf_norm(hit.group(1)))
            for row in self.needs.items() or [{"product": self.needs.reading.get("product")}]:
                names.add(shelf_norm(row.get("product")))
        except Exception:
            pass
        names.discard("")
        self.shelf_names_memo = names
        return names

    def shelf_lens(self, piece: Any) -> Any:
        """K1: worlds rename the category field (section_path, group_path). A record (record_id) with no field named like
        a category, whose one field holding one of the request's category names (shelf_names) is the same field in 2
        records, teaches that field: from then on it moves to "category" (its last path step) in each record of the
        lensed copy, and the records read before are relabelled."""
        try:
            records: list[dict[str, Any]] = []
            walk_dicts(piece, lambda d: records.append(d) if record_id(d) else None)
            names = self.shelf_names()
            if self.__dict__.get("shelf_alias") is None and names and records:
                votes = self.__dict__.setdefault("shelf_votes", {})
                for d in records:
                    if self.shelf_named(d):
                        continue
                    hits = [key for key, val in d.items() if isinstance(val, str) and 0 < len(val.strip()) <= 80
                            and not id_key(key) and shelf_hit(val, names)]
                    if len(hits) == 1:
                        votes.setdefault(hits[0], set()).add(record_id(d))
                agreed = [key for key, ids in votes.items() if len(ids) >= 2]
                if len(agreed) == 1:
                    self.shelf_alias = agreed[0]
                    self.log.append({"kind": "shelf_alias", "key": str(agreed[0])[:40]})
                    for rec in self.archive.items:
                        walk_dicts(rec, self.shelf_move)
                    self.__dict__.pop("home_memo", None)
            if self.__dict__.get("shelf_alias") is not None:
                for d in records:
                    self.shelf_move(d)
        except Exception as err:
            self.log.append({"kind": "shelf_lens_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return piece

    @staticmethod
    def shelf_named(d: dict[str, Any]) -> bool:
        """The record already has a category field by its name."""
        return any(isinstance(val, str) and val.strip() and re.search(r"categor|shelf|department|aisle", str(key).lower())
                   and "id" not in str(key).lower() for key, val in d.items())

    def shelf_move(self, d: dict[str, Any]) -> None:
        """K1: the learned category field of one record moves to "category" (last path step)."""
        key = self.__dict__.get("shelf_alias")
        if key is None or not isinstance(d, dict) or not record_id(d) or self.shelf_named(d):
            return
        val = d.get(key)
        if isinstance(val, str) and val.strip():
            names = self.shelf_names()
            label = val.strip() if shelf_norm(val) in names else shelf_last(val) if shelf_norm(shelf_last(val)) in names else val.strip()
            d.pop(key)
            d["category"] = label
            ident = d.get("id") if isinstance(d.get("id"), str) else record_id(d)
            if ident:
                self.archive.__dict__.setdefault("label_of", {}).setdefault(ident, label)

    def named_shelves(self) -> list[str]:
        """SH: the request's "from <name>" category names (shelf_norm), in order, each once."""
        out: list[str] = []
        for hit in re.finditer(r"\bfrom\s+([^;.()\n]{2,80}?)\s*(?=[;.)\n]|$)", str(self.request or "")):
            name = shelf_norm(hit.group(1))
            if name and name not in out:
                out.append(name)
        return out

    def named_apart(self, label: str) -> bool:
        """SH: one half of a joined category name ("tripods" of "Tripods and Camera Supports") is itself another category
        the request names: the two are separate categories, never one to join."""
        named = self.named_shelves()
        parts = [shelf_norm(p) for p in re.split(JOINERS, str(label).lower()) if p.strip()]
        return any(p in named and p != shelf_norm(label) for p in parts)

    def named_map(self, names: list[set[str]]) -> tuple[list[str | None], list[str]]:
        """SH/SHc: (the named category each product takes, the named categories): with two or more, by kind-word fit
        (overlap over union, singular and plural alike), one product per naming line, the better fit first."""
        found = [shelf_norm(h.group(1)) for h in re.finditer(r"\bfrom\s+([^;.()\n]{2,80}?)\s*(?=[;.)\n]|$)", str(self.request or ""))]
        named = [n for n in dict.fromkeys(found) if n]
        mapped: list[str | None] = [None] * len(names)
        if len(named) < 2:
            return mapped, named

        def fit(own: set[str], name: str) -> float:
            words = set(kind_tokens(name))
            hit = sum(1 for w in own if kind_in(w, words))
            return hit / (len(own) + len(words) - hit) if own and words and hit else 0.0
        room = {n: found.count(n) for n in named}
        for score, k, n in sorted(((fit(own, n), k, n) for k, own in enumerate(names) for n in named), key=lambda x: (-x[0], x[1], x[2])):
            if score > 0 and mapped[k] is None and room[n] > 0:
                mapped[k], room[n] = n, room[n] - 1
        return mapped, named

    def named_homes(self, homes: list[set[str]], names: list[set[str]], seen: set[str]) -> list[set[str]]:
        """SH/SHb/SHc: each product's home is its named category (named_map) once a listing showed it; until then empty while
        the store's labels follow the request's names or its own search has not run; never another product's category."""
        mapped, named = self.named_map(names)
        if len(named) < 2:
            return homes

        def key(text: Any) -> str:
            return " ".join(re.sub(r"\s*&\s*|\s+and\s+", " and ", re.sub(r"[,.;:]", " ", shelf_norm(text))).split())
        labels = {key(s): s for s in seen}
        follow = any(key(n) in labels for n in named)
        searched = self.__dict__.get("menu_searched") or set()
        out: list[set[str]] = []
        for k, best in enumerate(homes):
            mine = mapped[k] if k < len(mapped) else None
            others = {key(m) for j, m in enumerate(mapped) if j != k and m and m != mine}
            if mine and key(mine) in labels:
                out.append({s for s in seen if key(s) == key(mine)})
            elif mine and (follow or k not in searched):
                out.append(set())
            else:
                out.append({s for s in best if key(s) not in others})
        return out

    def home_shelves(self) -> list[set[str]]:
        """Map each requested product (or the single product) to the category labels that best match its name, as a
        list of label sets. Labels come from archived dicts that also carry a known listing id under an id-named
        key. Each label's kind tokens are scored against the product's with kind_score, the other products' tokens
        counting as shared; ties at a positive top score keep several labels. Cached in home_memo by archive size;
        an error gives empty sets.
        """
        memo = self.__dict__.get("home_memo")
        if memo and memo[0] == len(self.archive.items):
            return [set(h) for h in memo[1]]
        homes: list[set[str]] = []
        try:
            rows = self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}]
            seen: set[str] = set()

            def visit(d: dict[str, Any]) -> None:
                """Add a dict's category label to the seen set when the dict also holds a known listing id."""
                shelf = self.shelf_of(d)
                if shelf and any(isinstance(v, str) and v in self.archive.ids and "id" in str(key).lower() for key, v in d.items()):
                    seen.add(shelf)
            for rec in self.archive.items:
                walk_dicts(rec, visit)
            names = [set(kind_tokens(row["product"])) for row in rows]
            for k, own in enumerate(names):
                shared = set().union(*(names[j] for j in range(len(names)) if j != k))
                best, top = set(), 0.0
                for shelf in sorted(seen):
                    words = set(kind_tokens(shelf))
                    score = kind_score(own, words, shared)
                    if score > top:
                        best, top = {shelf}, score
                    elif score == top and score > 0:
                        best.add(shelf)
                homes.append(best)
        except Exception:
            homes = [set() for _ in (self.needs.items() or [0])]
        try:
            homes = self.named_homes(homes, names, seen)      # (SH: a sibling category named for another product is no home)
        except Exception as err:
            self.log.append({"kind": "named_homes_error", "error": f"{type(err).__name__}: {err}"[:300]})
        self.home_memo = (len(self.archive.items), [set(h) for h in homes])
        return homes

    def own_shelf(self) -> str:
        """For a single-product request, return the one category label to use when read listings span labels outside
        the product's home labels: the first read label that is a home label, else the first home label. Empty for
        several products, with no home labels, when every read label is a home label, or on error.
        """
        try:
            if self.needs.items():
                return ""
            home = self.home_shelves()[0]
            read = [s.strip().lower() for s in self.archive.shelves_read()]
            if not home or all(s in home for s in read):
                return ""
            mine = [s for s in read if s in home]
            return mine[0] if mine else sorted(home)[0]
        except Exception:
            return ""

    def joined_phrase(self, a: str, b: str, label: str) -> tuple[str, bool] | None:
        """Find request wording that joins two products into one phrase. Returns (phrase, True) when the two halves of
        a two-part category label appear in the request joined by and, or, an ampersand, a slash or a plus.
        Otherwise, for products of different kinds, returns (phrase, False) for the two product names written close
        together with at most three words, a joining word or comma, and no numbers between them, unless both names
        also occur elsewhere in the request. None when no such phrase exists.
        """
        low = re.sub(r"\s+", " ", self.request.lower().replace("’", "'"))
        parts = [p for p in re.split(JOINERS, str(label).lower()) if p.strip()]
        if len(parts) == 2:
            pa, pb = name_pattern(parts[0]), name_pattern(parts[1])
            if pa and pb:
                hit = re.search(r"\b" + pa + r"\s*(?:\band\b|&|\bor\b|/|\+)\s*" + pb, low)
                if hit:
                    return hit.group(0).strip(), True
        ta, tb = kind_tokens(a), kind_tokens(b)
        if not ta or not tb or same_kind(ta[-1], tb[-1]):
            return None
        for x, y in ((a, b), (b, a)):
            px, py = name_pattern(x), name_pattern(y)
            if not px or not py:
                continue
            for hit in re.finditer(r"\b(?:" + px + r")(?P<mid>[^.;:!?]{0,40}?)(?:" + py + r")", low):
                mid = hit.group("mid")
                words = re.findall(r"[a-z0-9]+", mid)
                if len(words) > 3 or not re.search(r"\band\b|&|\bor\b|,|/|\+", mid) or re.search(r"\d|\bx\b", mid):
                    continue
                rest = low[:hit.start()] + " | " + low[hit.end():]
                if all(re.search(r"\b" + name_pattern(z), rest) for z in (a, b)):
                    continue
                return hit.group(0).strip(), False
        return None

    def product_question(self, line: str, rows: list[dict[str, Any]]) -> bool:
        """Tell whether a logged question line concerns a particular product: true when a parenthesised tag in its
        label shares kind tokens with a requested product, or when the asked question names a product. A line not in
        the logged question format is used whole as both label and question.
        """
        hit = re.match(r'-\s*(.*?): asked "(.*?)" -> ', str(line), re.S)
        label, question = (hit.group(1), hit.group(2)) if hit else (str(line), str(line))
        kinds = [kind_tokens(r["product"]) for r in rows]
        for tag in re.findall(r"\(([^()]*)\)", label):
            words = kind_tokens(tag)
            if words and any(kind_in(w, k) for w in words for k in kinds):
                return True
        low = question.lower().replace("’", "'")
        return any(name_pattern(r["product"]) and re.search(r"\b" + name_pattern(r["product"]), low) for r in rows)

    @staticmethod
    def joined_clash(a: dict[str, Any], b: dict[str, Any]) -> bool:
        """Tell whether two product rows set the same property to different values (values compared as lowercase
        alphanumeric words); such a clash keeps the products from being merged.
        """
        def pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
            """Turn a product row's requirements into pairs of property name (the text before the colon, without its tag)
            and value reduced to lowercase alphanumeric words.
            """
            out = []
            for req in row.get("requirements") or []:
                head, _, value = str(req).partition(":")
                out.append((re.split(r"\s*\(", head)[0].strip(), " ".join(re.findall(r"[a-z0-9]+", value.lower()))))
            return out
        return any(same_prop(pa, pb) and va != vb for pa, va in pairs(a) for pb, vb in pairs(b))

    def merge_joined(self, obs: Any) -> str:
        """Merges two requested products sold under one joined category label; returns notes.
        Labels split by a joiner word into two parts are collected into joined_seen.
        At most two merges a call, before any cart, need or product question.
        A pair qualifies with the same quantity, kind words split across the label's
        two sides, no clashing requirements and the request joining them.
        merge_items does the merge.
        """
        seen = self.__dict__.setdefault("joined_seen", {})

        def visit(d: dict[str, Any]) -> None:
            """Record the category, shelf, department or aisle value (60 characters at most) of a dict with an id-like id
            when the value splits into exactly two parts on joiner words.
            """
            if any(isinstance(v, str) and "id" in str(k).lower() and re.fullmatch(ID_LIKE, v) for k, v in d.items()):
                label = next((val.strip() for key, val in d.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                              and re.search(r"categor|shelf|department|aisle", str(key).lower()) and "id" not in str(key).lower()), "")
                if label and len([p for p in re.split(JOINERS, label.lower()) if p.strip()]) == 2:
                    seen.setdefault(label.lower(), label)
        walk_dicts(obs, visit)
        remarks: list[str] = []
        for _ in range(2):
            rows = self.needs.items()
            if len(rows) < 2 or not seen or self.trolley.used or self.trolley.units:
                break
            if self.needs.uncovered or self.needs.indifferent or any(self.product_question(ln, rows) for ln in self.questions_log):
                break
            found = None
            mapped = self.named_map([set(kind_tokens(r["product"])) for r in rows])[0]      # (SHc)
            for low in sorted(seen):
                if self.named_apart(low):
                    continue      # (SH: a half of the name is another category the request names: never joined)
                parts = [p for p in re.split(JOINERS, low) if p.strip()]
                sides = [kind_tokens(p) for p in parts]
                if len(sides) != 2 or not sides[0] or not sides[1]:
                    continue
                whole = sides[0] + sides[1]
                for i in range(len(rows)):
                    for j in range(i + 1, len(rows)):
                        ti, tj = kind_tokens(rows[i]["product"]), kind_tokens(rows[j]["product"])
                        if not ti or not tj or (kinds_within(ti, tj) and kinds_within(tj, ti)) or not kinds_within(ti + tj, whole) \
                                or rows[i]["quantity"] != rows[j]["quantity"]:
                            continue
                        di = [w for w in ti if not kind_in(w, tj)]
                        dj = [w for w in tj if not kind_in(w, ti)]
                        if not any(all(kind_in(w, x) and not kind_in(w, y) for w in di) and all(kind_in(w, y) and not kind_in(w, x) for w in dj)
                                   and any(kind_in(w, x) for w in ti) and any(kind_in(w, y) for w in tj)
                                   for x, y in ((sides[0], sides[1]), (sides[1], sides[0]))):
                            continue
                        if mapped[i] and mapped[j] and mapped[i] != mapped[j]:
                            continue      # (SHc: each matches a different category the request names: two products)
                        if self.joined_clash(rows[i], rows[j]):
                            continue
                        got = self.joined_phrase(rows[i]["product"], rows[j]["product"], seen[low])
                        if got:
                            found = (i, j, seen[low], got[0], got[1])
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
            remarks.append(self.merge_items(*found))
        return " ".join(r for r in remarks if r)

    def merge_items(self, i: int, j: int, label: str, phrase: str, in_order: bool = True) -> str:
        """Merges needs products i and j into one named by the joined phrase.
        Keeps i's quantity and both untagged requirement sets.
        Their tags become the phrase in every needs list,
        reading requirements, hints, references and soft wants,
        tracked references, pending seconds and open pairs.
        Updates the reading items and product; bumps needs.version;
        clears home, menu, quota and walk memos; rescales the wire.
        Logs merge_joined and returns a note for the agent.
        Empty on an item mismatch.
        """
        reading = self.needs.reading
        rows = self.needs.items()
        raw = [r for r in reading.get("items") or [] if isinstance(r, dict) and str(r.get("product") or "").strip()]
        if len(raw) != len(rows) or not (0 <= i < j < len(rows)):
            return ""
        ti, tj = kind_tokens(rows[i]["product"]), kind_tokens(rows[j]["product"])
        whole = kind_tokens(label)
        others = [kind_tokens(r["product"]) for k, r in enumerate(rows) if k not in (i, j)]
        left = len(rows) - 1
        tag_now = phrase if left >= 2 else ""

        def owned(tag: Any) -> bool:
            """Tell whether a tag names the merged products: its kind tokens lie within the label, share a token with
            either product, and do not equal another product's tokens.
            """
            ts = kind_tokens(tag)
            return bool(ts) and kinds_within(ts, whole) and any(kind_in(w, ti + tj) for w in ts) \
                and not any(kinds_within(ts, o) and kinds_within(o, ts) for o in others)

        def retag(text: Any) -> Any:
            """Rewrite the parenthesised tags in the head of a property-colon-value text: tags owned by the merged products
            become the new phrase tag, or vanish when one product remains. Non-string values pass through.
            """
            if not isinstance(text, str):
                return text
            head, sep, value = text.partition(":")

            def put(hit: Any) -> str:
                """Give the replacement for one tag match: the new tag (or nothing) when owned, else the match unchanged."""
                return (f" ({tag_now})" if tag_now else "") if owned(hit.group(1)) else hit.group(0)
            return re.sub(r"\s*\(([^()]*)\)", put, head) + sep + value

        def unique(items: list[Any]) -> list[Any]:
            """Return the items in their original order with repeats removed. Membership is checked against a list, so
            unhashable values work.
            """
            out: list[Any] = []
            for x in items:
                if x not in out:
                    out.append(x)
            return out
        reqs: list[str] = []
        for req in list(rows[i]["requirements"]) + list(rows[j]["requirements"]):
            head, _, value = str(req).partition(":")
            plain = re.sub(r"\s*\([^()]*\)", "", head).strip() + ":" + value
            if plain.lower() not in [r.lower() for r in reqs]:
                reqs.append(plain)
        entry = {"product": phrase, "quantity": rows[i]["quantity"], "requirements": reqs}
        kept = [entry if k == i else r for k, r in enumerate(raw) if k != j]
        needs = self.needs
        needs.written = unique([retag(r) for r in needs.written])
        needs.uncovered = unique([retag(r) for r in needs.uncovered])
        needs.let_go = unique([retag(r) for r in needs.let_go])
        needs.indifferent = unique([retag(r) for r in needs.indifferent])
        if needs.now_reqs is not None:
            needs.now_reqs = unique([retag(r) for r in needs.now_reqs])
        needs.resolved = [retag(r) for r in needs.__dict__.get("resolved") or []]
        for key in ("requirements", "hints", "references", "soft_wants"):
            if isinstance(reading.get(key), list):
                reading[key] = unique([retag(r) for r in reading[key]])
        for tgt in reading.get("reference_targets") if isinstance(reading.get("reference_targets"), list) else []:
            if isinstance(tgt, dict) and owned(str(tgt.get("product") or "")):
                tgt["product"] = phrase
        for ref in self.__dict__.get("refs") or []:
            for key in ("req", "head", "now"):
                if isinstance(ref.get(key), str):
                    ref[key] = retag(ref[key])
            if owned(str(ref.get("tag") or "")):
                ref["tag"] = tag_now
        for key in ("pending_seconds", "open_pair"):
            if isinstance(self.__dict__.get(key), list):
                self.__dict__[key] = [retag(p) for p in self.__dict__[key]]
        if left >= 2:
            reading["items"] = kept
            reading["product"] = " and ".join(str(r.get("product")) for r in kept)
        else:
            reading.pop("items", None)
            reading["product"] = phrase
            reading["quantity"] = rows[i]["quantity"]
        needs.version += 1
        for key in ("home_memo", "menu_memo", "quota_reads", "walked"):
            self.__dict__.pop(key, None)
        self.wire.scale = 1.0 + 0.3 * min(2, max(0, len(needs.items()) - 1))
        self.log.append({"kind": "merge_joined", "label": label, "products": [rows[i]["product"], rows[j]["product"]],
                         "into": phrase, "label_words": in_order, "products_left": max(1, len(needs.items()))})
        return (f"The store sells '{label}' as ONE category: the request asks for one product of it ('{phrase}'), so the order "
                + ("holds one line for it" if left >= 2 else "holds ONE line") + ", not one line for each kind its name joins. "
                "Requirements now: " + needs.text_body())

    def lookup_key(self, tool: str | None) -> str | None:
        """Find the argument name under which a tool takes one listing id: its required free-text argument, or its only
        one. None when no such argument exists or the tool requires any other argument.
        """
        jschema = self.shapes.get(tool or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        needed = [k for k in jschema.get("required") or [] if k in props] if isinstance(jschema.get("required"), list) else []
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
        key = next((k for k in texts if k in needed), None) or (texts[0] if len(texts) == 1 else None)
        return key if key and all(k == key for k in needed) else None

    def lookup_tools(self) -> list[str]:
        """List tools not mapped to search, message, cart, cart view, order or decline that accept a single listing id."""
        taken = {self.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        return [name for name in self.shapes if name not in taken and self.lookup_key(name)]

    def lookup_call(self, ident: str, tool: str | None) -> dict[str, Any] | None:
        """Build arguments for reading one listing with a given tool: details_call for the details tool, else the
        latest logged call of that tool that held a known listing id with the new id put in, else lookup_key. None
        without a tool or key.
        """
        if not tool:
            return None
        if tool == self.role("details"):
            return self.details_call(ident)
        for entry in reversed(self.log):
            for act in (entry.get("actions") or []) if entry.get("kind") == "env" else []:
                args = act.get("args") if act.get("name") == tool else None
                if isinstance(args, dict) and any(isinstance(v, str) and v in self.archive.ids for v in args.values()):
                    return {k: (ident if isinstance(v, str) and v in self.archive.ids else v) for k, v in args.items()}
        key = self.lookup_key(tool)
        return {key: ident} if key else None

    @staticmethod
    def rec_id(d: dict[str, Any]) -> str | None:
        """Return a record's listing id: the id-like value under a key named exactly id, else under any key ending in
        id. None when neither exists.
        """
        for exact in (True, False):
            for key, val in d.items():
                name = str(key).lower()
                if (name == "id" if exact else name.endswith("id")) and isinstance(val, str) and re.fullmatch(ID_LIKE, val):
                    return val
        return None

    @staticmethod
    def asked_ids(call_args: Any) -> set[str]:
        """Collect the id-like strings a tool call asked for: plain string argument values and the strings inside list
        arguments. Arguments that are not a dict give an empty set.
        """
        args = call_args if isinstance(call_args, dict) else {}
        asked = {v for v in args.values() if isinstance(v, str) and re.fullmatch(ID_LIKE, v)}
        asked |= {x for v in args.values() if isinstance(v, list) for x in v if isinstance(x, str) and re.fullmatch(ID_LIKE, x)}
        return asked

    def learn_caps(self, call_name: str, call_args: Any, obs: Any) -> list[str]:
        """Learns tool output from one call; returns cart lines newly out of stock.
        tool_caps counts calls and what records showed: id, stock, map, price, notes.
        Details go to read_with, text_keys, tool_keys and note_keys.
        stock_by keeps stock per id and source with the cart epoch.
        Lookup and row clashes set rows_lag; row changes rows_live.
        Out of stock adds the id to dead; in stock clears it.
        Non-cart rows keep their text and keys.
        Errors are skipped.
        """
        asked = self.asked_ids(call_args)
        name = str(call_name)
        caps = self.__dict__.setdefault("tool_caps", {}).setdefault(name, {"calls": 0, "id": 0, "stock": 0, "props": 0, "notes": 0,
                                                                           "price": 0})
        caps["calls"] += 1
        hit = {"id": False, "stock": False, "props": False, "notes": False, "price": False}
        dead = self.__dict__.setdefault("dead", set())
        dead_by = self.__dict__.setdefault("dead_by", {})
        heard = self.__dict__.setdefault("stock_by", {})
        row_text = self.__dict__.setdefault("row_text", {})
        row_keys = self.__dict__.setdefault("row_keys", set())
        keys = self.__dict__.setdefault("tool_keys", {}).setdefault(name, set())
        note_keys = self.__dict__.setdefault("note_keys", {}).setdefault(name, set())
        carts = {self.role("cart"), self.role("cart_view")} - {None}
        epoch = self.__dict__.get("cart_seq", 0)
        newly: list[str] = []

        def visit(d: dict[str, Any]) -> None:
            """Process one record dict: find its id, read its stock state and update stock_seen, stock_by, rows_lag,
            rows_live, dead and dead_by, noting cart lines that just went dead. For ids the call did not ask for it
            stores row text and keys (outside cart tools) and stops. For asked ids it marks the id, stock, property-map
            and price hits, records read_with, text_keys and tool_keys, and marks as note keys string fields of eight or
            more words, not title-like, whose text differs from the row text or whose key rows never carry.
            """
            ident = self.rec_id(d)
            if not ident:
                return
            state = self.stock_state(d)
            source = "lookup" if ident in asked or call_name in carts else "row"
            if state is not None:
                self.stock_seen = True
                mine = heard.setdefault(ident, {})
                other = mine.get("row" if source == "lookup" else "lookup")
                if other and other[0] == epoch and other[1] != state:
                    self.rows_lag = True
                if source == "row" and mine.get("row") and mine["row"][1] != state:
                    self.rows_live = True
                mine[source] = (epoch, state)
                if state is False:
                    if ident not in dead and ident in self.trolley.units and ident not in newly:
                        newly.append(ident)
                    dead.add(ident)
                    dead_by[ident] = source
                elif source == "lookup" or dead_by.get(ident) != "lookup":
                    dead.discard(ident)
            if ident not in asked:
                if call_name not in carts:
                    row_keys.update(str(k) for k in d)
                    row_text.setdefault(ident, {}).update({str(k): v for k, v in d.items() if isinstance(v, str)})
                return
            hit["id"] = True
            self.__dict__.setdefault("read_with", {}).setdefault(name, set()).add(ident)
            self.__dict__.setdefault("text_keys", {}).setdefault(name, set()).update(
                str(k) for k, v in d.items() if v is None or isinstance(v, str))
            hit["stock"] = hit["stock"] or state is not None
            hit["props"] = hit["props"] or any(isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2
                                               for v in d.values())
            hit["price"] = hit["price"] or any(is_number(v) and re.search(r"price|cost", str(k).lower()) for k, v in d.items())
            keys.update(str(k) for k in d)
            row = row_text.get(ident)
            for k, v in d.items():
                if isinstance(v, str) and len(v.split()) >= 8 and (row.get(str(k)) != v if row is not None else str(k) not in row_keys) \
                        and not re.search(r"title|heading|headline|name|label|caption", str(k).lower()):
                    note_keys.add(str(k))
                    hit["notes"] = True
        if not (isinstance(obs, dict) and obs.get("error")):
            walk_dicts(obs, visit)
        for key, got in hit.items():
            caps[key] += int(got)
        return newly

    def stock_tool(self) -> str | None:
        """Choose the tool for re-reading a listing's stock. Prefers lookup tools that have shown stock (most stock
        hits, then the view role, then details). Else tries the view role, the code-guessed view tool (code_roles,
        cached in role_guess) and the details role, skipping one that returned ids twice without stock. Last, the
        first lookup tool not guessed as cart, cart view, order, decline or message that has never returned an id.
        None when nothing fits.
        """
        caps = self.__dict__.get("tool_caps") or {}
        tools = self.lookup_tools()
        shown = [t for t in tools if (caps.get(t) or {}).get("stock", 0) > 0]
        if shown:
            return max(shown, key=lambda t: (caps[t]["stock"], t == self.role("view"), t == self.role("details"), t))
        guess = self.__dict__.get("role_guess")
        if guess is None:
            try:
                guess = self.code_roles()
            except Exception:
                guess = {}
            self.role_guess = guess
        for tool in (self.role("view"), guess.get("view"), self.role("details")):
            got = caps.get(tool or "") or {}
            if tool and tool in tools and not (got.get("id", 0) >= 2 and got.get("stock", 0) == 0):
                return tool
        busy = {guess.get(r) for r in ("cart", "cart_view", "order", "decline", "message")} - {None}
        return next((t for t in tools if t not in busy and not (caps.get(t) or {}).get("id", 0)), None)

    def notes_tool(self) -> str | None:
        """Return the lookup tool, other than details, whose replies carried note fields the details tool lacks,
        preferring the most note hits; None when there is none.
        """
        caps = self.__dict__.get("tool_caps") or {}
        details = self.role("details")
        dkeys = (self.__dict__.get("tool_keys") or {}).get(details or "", set())
        nkeys = self.__dict__.get("note_keys") or {}
        shown = [t for t in self.lookup_tools() if t != details and set(nkeys.get(t) or set()) - set(dkeys)]
        return max(shown, key=lambda t: ((caps.get(t) or {}).get("notes", 0), t)) if shown else None

    def dead_lines(self) -> set[str]:
        """Return a copy of the set of listing ids last read as out of stock."""
        return set(self.__dict__.get("dead") or set())

    def stale_lines(self) -> list[str]:
        """Return the cart lines whose last stock reading (line_seq) is older than the current cart mark (cart_seq)."""
        seqs = self.__dict__.get("line_seq") or {}
        mark = self.__dict__.get("cart_seq", 0)
        return [i for i in self.trolley.units if seqs.get(i, -1) < mark]

    def dead_swap(self, line: str) -> tuple[str, str]:
        """Sold-out cart line swap.
        (pick, ''): the cheapest listing passing this version's item check,
        on the line's category, not dead, refused or carted, within the cap.
        Else ('', text): up to 3 in-stock, in-budget rows of the category,
        minus ruled-out, pointer, excluded and failing ones.
        Read rows lead by price; for cheapest, a cheaper unread one leads.
        Unread ids go to swap_unread.
        With none, swap_page names the next unread search page.
        """
        dead, ver = self.dead_lines(), str(self.needs.version)
        kind = self.shelf_known(line)
        several = len(self.needs.items()) >= 2
        cap, qty = self.needs.spend_cap, max(1.0, self.trolley.held(line))

        def cost(ident: str) -> float:
            """Return a listing's price as a float, or infinity when its price is unknown."""
            price = self.archive.price(ident)
            return float(price) if is_number(price) else float("inf")
        rest = sum(cost(i) * self.trolley.held(i) for i in self.trolley.units if i != line and cost(i) != float("inf"))
        fits = []
        for key in self.passed:
            ident, _, version = key.rpartition("|")
            if version != ver or ident == line or ident in dead or ident in self.refused_lines or ident in self.trolley.units:
                continue
            if self.field_view(ident)["fails"]:
                continue
            shelf = self.shelf_known(ident)
            if kind and (shelf != kind if several else shelf not in ("", kind)):
                continue
            if is_number(cap) and cost(ident) != float("inf") and cost(ident) * qty + rest > cap:
                continue
            fits.append(ident)
        if fits:
            return min(fits, key=lambda i: (cost(i), i)), ""
        if several and not kind:
            return "", ""
        k = next((n for n, home in enumerate(self.home_shelves()) if kind and kind in home), None) if several else None
        judged = [(prop, plain) for prop, plain in self.code_values(k) if not re.search(
            r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b", plain.lower())]
        failed = {key.split("|")[0] for key, val in self.rulings.items() if val and key.split("|")[1:2] == [ver]}
        pointers = self.ref_pointers()
        rows = []
        for ident, row in self.code_rows().items():
            price = row["price"] if is_number(row["price"]) else cost(ident)
            if ident == line or ident in dead or ident in self.refused_lines or ident in self.trolley.units or row["stock"] is not True:
                continue
            if (kind and row["shelf"] != kind) or price == float("inf") or (is_number(cap) and price * qty + rest > cap):
                continue
            if ident in failed or ident in pointers or self.ref_excluded(row["fields"], k):
                continue
            read = ident in self.archive.read
            if read and self.field_view(ident)["fails"]:
                continue
            if read and (any(self.code_clash(row["fields"], prop, plain) for prop, plain in judged)
                         or not all(self.code_shows(row, prop, plain, set(), True) for prop, plain in judged)):
                continue
            rows.append((0 if read else 1, price, ident))
        rows.sort()
        low = min((x for x in rows if x[0] == 1), key=lambda x: (x[1], x[2]), default=None)
        if low and self.needs.priority() == "cheapest" and all(low[1] < p for r, p, _ in rows if r == 0):
            rows.remove(low)
            rows.insert(0, low)
        self.swap_unread = [i for r, _, i in rows[:3] if r == 1]
        self.swap_page = None
        if not any(r == 0 for r, _, _ in rows):
            try:
                self.swap_page = self.dead_page(k)
            except Exception:
                self.swap_page = None
        return "", ", ".join(f"{i} ({int(p) if float(p).is_integer() else p}{', read: its record shows every stated value' if r == 0 else ''})"
                             for r, p, i in rows[:3])

    def dead_page(self, k: int | None) -> dict[str, Any] | None:
        """Return arguments for the next unread page of the logged search walk that carries the most of product k's
        stated values (fewest pages read breaks ties), with the page argument from page_arg set. None without a page
        argument or a walk that has a next page.
        """
        found = self.page_arg()
        if not found:
            return None
        walks = [w for w in self.__dict__.get("page_book", {}).values() if self.walk_carries(w["args"], k) and self.next_page(w) is not None]
        if not walks:
            return None
        best = max(walks, key=lambda w: (len(self.walk_carries(w["args"], k)), -len(w["read"])))
        return {**best["args"], found[0]: self.next_page(best)}

    def swap_open(self) -> str:
        """Return the first replacement candidate in swap_unread that is unread, not dead and not in the cart, unless a
        swap read already happened (swap_done). Empty string otherwise.
        """
        if self.__dict__.get("swap_done"):
            return ""
        dead = self.dead_lines()
        return next((i for i in self.__dict__.get("swap_unread") or [] if i not in self.archive.read and i not in dead
                     and i not in self.trolley.units), "")

    def dead_note(self, lines: list[str]) -> str:
        """Write the agent note for cart lines that went out of stock, and set event_seen. For each line dead_swap
        either names a verified replacement to put in before re-reading stock and ordering, lists the cheapest in-
        stock rows of its category to check, or asks for another fitting listing. Without a pick, a known next
        search page is added as call arguments.
        """
        parts = []
        self.event_seen = True
        for line in lines:
            self.swap_page, self.swap_unread = None, []
            pick, rows = self.dead_swap(line)
            head = f"{line} is OUT OF STOCK now (its newest record shows it): an order holding it fails. Take it out of the cart"
            if pick:
                least = ", the cheapest such listing" if self.needs.priority() == "cheapest" else ""
                parts.append(f"{head} and put {pick} in (the item check found it meets every requirement now in force{least}), "
                             "re-read its stock and price, then order.")
            elif rows:
                parts.append(f"{head}; no other verified fit of its product is known, so check the cheapest in-stock rows of its "
                             f"category within the budget ({rows}; read rows whose own record lacks or fails a stated requirement "
                             "are left out, and an unread one needs its details read first), put the first that fits in, re-read "
                             "its stock and price, then order.")
            else:
                parts.append(f"{head} and find another listing that fits.")
            page = self.__dict__.get("swap_page")
            if page and not pick:
                parts.append("More rows of its kind: the next unread page of the search led by the stated value, "
                             + json.dumps(page, default=str) + ".")
        return " ".join(parts)

    def refusal_reread(self, obs: Any = None) -> str:
        """After a refused order, return what the shopper's refusal changed, followed by either the sold-out note from
        refusal_gone or, when that finds nothing, the queued stock re-read from refusal_queue. obs is the refused
        order's observation; errors in refusal_gone are logged as refusal_gone_error.
        """
        heard = ""
        try:
            heard, gone = self.refusal_gone(obs)
        except Exception as err:
            gone = ""
            self.log.append({"kind": "refusal_gone_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if gone:
            return heard + gone
        return heard + self.refusal_queue()

    def refusal_gone(self, obs: Any) -> tuple[str, str]:
        """Read the shopper's text in a refused order's observation when the cart holds exactly one line and the last
        action was an order (not muted or covering), and feed it to needs.hear. When that bumps the needs version
        without changing let_go, uncovered, the spend cap or the remaining requirements, the line is taken as sold
        out: marked dead by lookup, logged as refusal_dead, and a dead_note is returned. Returns (heard note, sold-
        out note).
        """
        lines = [i for i in self.trolley.units if self.trolley.held(i) > 0]
        if obs is None or len(lines) != 1 or self.wire.mute or self.__dict__.get("covering") or self.__dict__.get("last_kind") != "order":
            return "", ""
        texts = self.shopper_lines(self.role("order") or "", obs, True)
        if not texts:
            return "", ""
        needs = self.needs
        before = (needs.version, len(needs.let_go), len(needs.uncovered), str(needs.spend_cap), "|".join(needs.props_left()))
        note = needs.hear(texts, None)
        after = (needs.version, len(needs.let_go), len(needs.uncovered), str(needs.spend_cap), "|".join(needs.props_left()))
        heard = (" " + note) if note else ""
        if not note or after[0] == before[0] or after[1:] != before[1:]:
            return heard, ""
        line = lines[0]
        self.__dict__.setdefault("dead", set()).add(line)
        self.__dict__.setdefault("dead_by", {})[line] = "lookup"
        self.log.append({"kind": "refusal_dead", "line": line})
        return heard, (" The refusal says an item became unavailable and the cart held one line, so the code takes " + line
                       + " as sold out (no re-read): " + self.dead_note([line]))

    def refusal_queue(self) -> str:
        """Queue a scripted stock re-read of up to four cart lines with the stock tool after a refused order (purpose
        cart, recheck set). Skips when muted, covering, a queue is already set, fewer than lines plus three moves
        remain or spare budget is under CLOSING. Returns a short note naming the tool, or an empty string.
        """
        tool = self.stock_tool()
        lines = list(self.trolley.units)[:4]
        if not lines or not tool or self.wire.mute or self.__dict__.get("covering") or self.__dict__.get("open_queue") \
                or self.moves_left() < len(lines) + 3 or self.wire.spare() < CLOSING:
            return ""
        self.open_queue = lines
        self.open_recheck, self.open_purpose, self.open_tool = True, "cart", tool
        self.open_say = "The order was refused: I re-read the stock of the cart's lines (a line may have sold out)."
        return " The cart lines' stock is read for you now (" + tool + ")."

    def code_judges_all(self) -> bool:
        """Tell whether code can judge every remaining requirement: plain code values (no links, ranges, signs or
        approximations) across products plus requirements matched by resolved differ references cover their count.
        """
        items = self.needs.items()
        vals = [plain for k in (range(len(items)) if items else [None]) for _, plain in self.code_values(k)]
        plain = [v for v in vals if not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
                                                  v.lower())]
        resolved = self.differ_done(None)
        judged = sum(1 for req in self.needs.props_left() if any(self.ref_match(ref, req) for ref in resolved))
        return len(plain) + judged >= len(list(self.needs.props_left()))

    def differ_done(self, k: int | None) -> list[dict[str, Any]]:
        """Return tracked references with relation differ that are resolved, carry a value, are live and match a
        remaining requirement. For product k of a multi-product request, only those whose tag maps to k or to no
        product are kept.
        """
        rows, left, out = self.needs.items(), self.needs.props_left(), []
        for ref in self.__dict__.get("refs") or []:
            if ref.get("relation") != "differ" or ref.get("status") != "resolved" or not str(ref.get("value") or "").strip() \
                    or not self.ref_live(ref) or not any(self.ref_match(ref, req) for req in left):
                continue
            if rows and k is not None and ref.get("tag") and self.product_of(f"x ({ref['tag']})") not in (k, None):
                continue
            out.append(ref)
        return out

    def last_resort_cart(self) -> str:
        """Once per episode, have code put its best in-stock candidates in the cart and return a note telling the agent
        to order rather than decline. Needs cart and order roles, no mute, moves for every product plus two and
        spare budget of LAST_CALL + 12; a multi-product request that ed_proof shows cannot be met is left alone.
        Picks come from code_picks avoiding dead, refused and out pool entries, one per product, and are put with
        the wire muted via code_put. Returns an empty string unless every pick landed and no cart line is dead;
        logged as last_resort_cart.
        """
        if self.__dict__.get("last_resort_done") or not self.role("cart") or not self.role("order") or self.wire.mute:
            return ""
        rows = self.needs.items()
        n = max(1, len(rows))
        if self.moves_left() < n + 2 or self.wire.spare() < LAST_CALL + 12:
            return ""
        self.last_resort_done = True
        try:
            if rows and self.ed_proof():
                return ""
            avoid = set(self.dead_lines()) | set(self.refused_lines)
            for k in (range(len(rows)) if rows else [None]):
                avoid |= {e["id"] for e in self.ed_pool(k) if e["out"]}
            picks = self.code_picks(avoid)
            if len(picks) < n:
                self.log.append({"kind": "last_resort_cart", "picks": picks, "done": False})
                return ""
            was = self.wire.mute
            self.wire.mute = True
            try:
                bad = self.code_put(picks)
            finally:
                self.wire.mute = was
            done = bad == [] and all(i in self.trolley.units for i, _ in picks) and not set(self.trolley.units) & self.dead_lines()
            self.log.append({"kind": "last_resort_cart", "picks": picks, "done": done, "cart": dict(self.trolley.units)})
            if not done:
                return ""
            return ("Not run yet (no step used). A decline is right only when the records show that nothing can fit, and they do "
                    "not: " + ("every product asked for still has" if rows else "the store still has") + " a candidate in stock, "
                    "within the budget, that no field or check rules out. The code put the best of them in the cart: "
                    + self.trolley.summary() + " Order this cart now; decline only if a record you read shows one of these lines "
                    "failing a requirement now in force.")
        except Exception as err:
            self.log.append({"kind": "last_resort_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def decline_repair(self) -> str:
        """Pre-decline: the code builds a fitting cart and asks to order it.
        Runs after a refused order, at the order reserve or early (several products),
        once per refusal count.
        Needs cart and order roles, 3 moves and time; never while muted.
        Picks come from code_picks minus dead and clashing lines.
        Several products: skipped when ed_proof shows one unmeetable;
        up to 3 item checks re-pick lines with unmet needs.
        Puts are muted and stale lines re-read.
        Logs decline_repair.
        """
        several = len(self.needs.items()) >= 2
        early = several and self.refusal_n == 0 and self.moves_left() > self.order_due()
        if not (self.refusal_n > 0 or self.moves_left() <= self.order_due() or early):
            return ""
        if not self.role("cart") or not self.role("order") or self.moves_left() < 3 or self.wire.spare() <= CLOSING \
                or self.wire.mute or self.__dict__.get("covering") or self.wire.done:
            return ""
        tried = self.__dict__.setdefault("repair_tried", set())
        mark: Any = ("early", self.refusal_n) if early else self.refusal_n
        if mark in tried:
            return ""
        tried.add(mark)
        if several:
            try:
                proof = self.ed_proof()
            except Exception as err:
                proof = []
                self.log.append({"kind": "decline_proof_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if proof:
                self.log.append({"kind": "decline_proof", "products": proof})
                return ""
        dead = self.dead_lines()
        stale = set(self.stale_lines())
        alive = {i for i in self.trolley.units if i not in dead and i not in stale}
        saved = set(self.refused_lines)
        done = False
        ver = str(self.needs.version)

        own = self.code_judges_all()

        def fitting(picks: list[tuple[str, int]]) -> bool:
            """Tell whether every pick passed the item check for this needs version or, when code judges all requirements,
            is read and marked fit by code; false for no picks.
            """
            fit = self.__dict__.get("code_fit") or {}
            return bool(picks) and all(f"{i}|{ver}" in self.passed or (own and i in self.archive.read and fit.get(i)) for i, _ in picks)
        try:
            self.refused_lines.difference_update(alive)
            avoid = set(dead)
            seen: dict[str, dict[str, Any]] = {}
            if several:
                seen = {e["id"]: e for k in range(len(self.needs.items())) for e in self.ed_pool(k)}
                avoid |= {i for i, e in seen.items() if e["clash"]}
            picks = self.code_picks(avoid)
            checks = rounds = 0
            while several and picks and checks < 3 and rounds < 30 and self.wire.spare() > CLOSING + 30:
                unchecked = [i for i, _ in picks if not fitting([(i, 1)])]
                if not unchecked:
                    break
                bare = [i for i in unchecked if not seen.get(i) or not seen[i]["revealed_ok"] or seen[i]["misses"]]
                if bare:
                    avoid.update(bare)
                    avoid.update(i for i, e in seen.items() if not e["read"] and f"{i}|{ver}" not in self.passed
                                 and (not e["revealed_ok"] or e["misses"]))
                    rounds += 1
                    picks = self.code_picks(avoid)
                    continue
                checks += 1
                self.check_item(unchecked[0], "ORDER")
                picks = self.code_picks(avoid)
            if not fitting(picks):
                self.log.append({"kind": "decline_repair", "picks": picks, "done": False})
                return ""
            if self.moves_left() < len(picks) + 2:
                self.log.append({"kind": "decline_repair", "picks": picks, "done": False, "why": "steps"})
                return ""
            was = self.wire.mute
            self.wire.mute = True
            try:
                bad = self.code_put(picks)
                done = bad == [] and all(i in self.trolley.units for i, _ in picks)
                stale_now = [i for i in self.stale_lines() if any(i == p for p, _ in picks)]
                if done and stale_now and self.moves_left() >= len(stale_now) + 2:
                    _, sold = self.code_recheck(stale_now)
                    done = not sold
            finally:
                self.wire.mute = was
            done = done and bool(self.trolley.units) and not set(self.trolley.units) & self.dead_lines()
            self.log.append({"kind": "decline_repair", "picks": picks, "done": done, "cart": dict(self.trolley.units)})
        except Exception as err:
            done = False
            self.log.append({"kind": "decline_repair_error", "error": f"{type(err).__name__}: {err}"[:300]})
        finally:
            if not done:
                self.refused_lines.clear()
                self.refused_lines.update(saved)
        if not done:
            return ""
        several = bool(self.needs.items())
        return ("Not run yet (no step used). A decline is right only when nothing can fit, and the records hold a fitting candidate "
                + ("for every product asked for" if several else "") + " (verified by the item check, or read with every stated "
                "value in its own record), so the code put " + ("them" if several else "it") + " in the cart and read the stock: "
                + self.trolley.summary() + " Order this cart now; decline only if a record you read shows one of these lines "
                "failing a requirement now in force.")

    @staticmethod
    def ed_plain(value: str) -> bool:
        """Tell whether a value is plain enough for code to match literally: no link words, numeric ranges, comparison
        signs or approximation words.
        """
        return not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
                             str(value).lower())

    @staticmethod
    def ed_word(value: str) -> bool:
        """Tell whether a value is a plain word value: a word of three letters or more, no digits, plain, with no
        exclusion words and no reference.
        """
        low = str(value).lower()
        return bool(re.search(r"[a-z]{3,}", low)) and not re.search(r"\d", low) and Errand.ed_plain(low) \
            and not re.search(EXCLUDE_WORDS, low) and not ref_text(low)

    def ed_reqs(self, k: int | None) -> list[tuple[str, str, bool]]:
        """List the remaining requirements of product k (or all of them for one product) as (property name, value
        without its stated or revealed tag, revealed flag), where revealed means the requirement was not written in
        the request.
        """
        rows = self.needs.items()
        out = []
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            if rows and self.product_of(head) not in (k, None):
                continue
            plain = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip()
            out.append((re.split(r"\s*\(", head)[0].strip(), plain, req not in self.needs.written))
        return out

    def ed_pool(self, k: int | None) -> list[dict[str, Any]]:
        """Scores product k's home-label code rows (request's if 1 product);
        ed_memo by archive, needs, passes, rulings, dead/refused lines, cap,
        refs. Entry: id, price, row, read; verified=passed now, not gone;
        gone=dead/sold out/refused; over=>cap at qty; pointer=ref/exclusion;
        ruled=negative ruling now or read+field fails; clash=contradicts
        plain/plus-joined values; misses=plain ones unshown; fit=verified or
        read w/o misses/clash, not ruled/pointer/gone/over; out; revealed_ok.
        """
        rows = self.needs.items()
        ver = str(self.needs.version)
        mark = (k, len(self.archive.items), len(self.archive.read), ver, "|".join(self.needs.props_left()), len(self.passed),
                sum(1 for v in self.rulings.values() if v), tuple(sorted(self.dead_lines())), tuple(sorted(self.refused_lines)),
                str(self.needs.spend_cap), len(self.__dict__.get("refs") or []))
        memo = self.__dict__.setdefault("ed_memo", {})
        if mark in memo:
            return memo[mark]
        if len(memo) > 64:
            memo.clear()
        homes = self.home_shelves()
        if rows:
            home = set(homes[k]) if k is not None and k < len(homes) else set()
        else:
            home = set(self.shelf_labels()) or (set(homes[0]) if homes else set())
        out: list[dict[str, Any]] = []
        if home:
            counts = self.needs.counts()
            qty = counts[k] if rows and k is not None and k < len(counts) else (counts[0] if counts else 1)
            failed = {key.split("|")[0] for key, val in self.rulings.items() if val and key.split("|")[1:2] == [ver]}
            live = {key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == ver}
            cap = float(self.needs.spend_cap) if is_number(self.needs.spend_cap) else None
            judged = [(p, v) for p, v in self.code_values(k if rows else None) if self.ed_plain(v)]
            joins = [(p, parts) for p, v in self.code_values(k if rows else None) if "+" in v
                     for parts in [[x.strip() for x in v.split("+") if x.strip()]] if len(parts) >= 2
                     and all(re.search(r"[a-z]{2,}", x.lower()) for x in parts)]
            shown = [(p, v) for p, v, revealed in self.ed_reqs(k if rows else None) if revealed]
            pointers, dead = self.ref_pointers(), self.dead_lines()
            for ident, row in self.code_rows().items():
                if row["shelf"] not in home:
                    continue
                read = ident in self.archive.read
                cost = row["price"] if row["price"] is not None else self.archive.price(ident)
                price = float(cost) if is_number(cost) else None
                gone = ident in dead or row["stock"] is False or ident in self.refused_lines
                over = cap is not None and price is not None and price * qty > cap + 1e-9
                pointer = ident in pointers or self.ref_excluded(row["fields"], k if rows else None)
                clash = [p for p, v in judged if self.code_clash(row["fields"], p, v)
                         and not any(self.code_seen(v, val) for key, val in row["fields"].items() if same_prop(key, p))]
                for p, parts in joins:
                    named = [val for key, val in row["fields"].items() if same_prop(key, p)]
                    if named and not all(any(self.code_seen(x, val) for val in named) for x in parts):
                        clash.append(p)
                misses = [(p, v) for p, v in judged if not self.code_shows(row, p, v, set(), True)]
                ruled = ident in failed or bool(read and self.field_view(ident)["fails"])
                verified = ident in live and not gone
                fit = (verified or (read and not misses and not clash)) and not (ruled or pointer or gone or over)
                out.append({"id": ident, "price": price, "read": read, "verified": verified, "fit": fit, "gone": gone,
                            "over": over, "pointer": pointer, "ruled": ruled, "clash": clash, "misses": misses, "row": row,
                            "out": bool(gone or over or pointer or ruled or clash),
                            "revealed_ok": all(self.code_shows(row, p, v, set(), True) for p, v in shown)})
        memo[mark] = out
        return out

    def ed_proof(self) -> list[int]:
        """Return the indexes of products the records show cannot be met: some pool entries are read, every read one is
        out, and no unread entry remains in play.
        """
        out = []
        for k in range(len(self.needs.items())):
            pool = self.ed_pool(k)
            reads = [e for e in pool if e["read"]]
            if reads and all(e["out"] for e in reads) and not any(not e["read"] and not e["out"] for e in pool):
                out.append(k)
        return out

    def slack_rows(self) -> list[str]:
        """Multi-product request whose cheapest fitting picks exceed the cap:
        up to three unread listings that fit the money left.
        A product's slack is the cap minus the others' cheapest totals, per unit.
        Candidates are unread, live, unruled rows within the slack,
        ranked by search hits on stated values, then price.
        Picks alternate between products; slack_total keeps the floors.
        Returns [] when a product has no priced fit.
        """
        rows = self.needs.items()
        cap = self.needs.spend_cap
        if len(rows) < 2 or not is_number(cap):
            return []
        try:
            counts = self.needs.counts()
            floors = []
            for k in range(len(rows)):
                prices = [e["price"] for e in self.ed_pool(k) if e["fit"] and e["price"] is not None]
                if not prices:
                    return []
                floors.append(min(prices) * (counts[k] if k < len(counts) else 1))
            if sum(floors) <= float(cap) + 1e-9:
                return []
            book = list(self.__dict__.get("page_book", {}).values())
            hits = set(self.__dict__.get("all_hits") or [])
            pools = []
            for k in range(len(rows)):
                slack = (float(cap) - (sum(floors) - floors[k])) / max(1, counts[k] if k < len(counts) else 1)
                values = [{w for w in re.findall(r"[a-z0-9]{2,}", v.lower()) if w not in LINK_WORDS} for _, v, _ in self.ed_reqs(k)]
                led: dict[str, int] = {i: 1 for i in hits}
                for walk in book:
                    words = [w for text in self.free_text(walk["args"]) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
                    words = set(words[: self.lead_words] if self.lead_words else words)
                    score = sum(1 for vals in values if vals & words)
                    for i in (walk.get("row_shelf") or {}):
                        if score:
                            led[str(i)] = max(led.get(str(i), 0), score)
                pool = sorted((-led.get(e["id"], 0), e["price"], e["id"]) for e in self.ed_pool(k)
                              if not e["read"] and not e["gone"] and not e["pointer"] and not e["ruled"] and not e["clash"]
                              and e["price"] is not None and 0 < slack and e["price"] <= slack + 1e-9)
                pools.append([i for _, _, i in pool])
            out: list[str] = []
            while len(out) < 3 and any(pools):
                for pool in pools:
                    if pool and len(out) < 3:
                        out.append(pool.pop(0))
            if out:
                self.slack_total = sum(floors)
            return out
        except Exception as err:
            self.log.append({"kind": "slack_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

    def verify_problems(self, ident: str) -> list[str] | None:
        """Return the problems from the latest logged item check of a listing when it failed; None when that check
        passed or no check was logged.
        """
        for entry in reversed(self.log):
            if isinstance(entry, dict) and entry.get("kind") == "verify" and entry.get("item") == ident:
                res = entry.get("result")
                if isinstance(res, dict) and res.get("fits") is False:
                    return [str(p) for p in res.get("problems") or [] if str(p).strip()]
                return None
        return None

    def ed_news(self, k: int | None) -> bool:
        """Tell whether anything arose after the request for product k: an order was refused, or an uncovered or let-go
        need applies to k (or to every product).
        """
        if self.refusal_n > 0:
            return True
        rows = self.needs.items()
        return any(not rows or self.product_of(str(req).partition(":")[0]) in (k, None)
                   for req in list(self.needs.uncovered) + list(self.needs.let_go))

    def near_hold(self) -> str:
        """Order the cart directly when a basket placed by the meaning check is still in force: near_cart matches the
        needs version and refusal count, all its lines are in the cart and none is dead or refused, an order tool
        exists, and a move and spare budget remain. Clears near_cart, logs near_order, calls finish_now and returns
        a note that the code ordered instead of declining; empty otherwise, with errors logged.
        """
        try:
            info = self.__dict__.get("near_cart")
            if not info or info.get("ver") != self.needs.version or info.get("refusals") != self.refusal_n or not self.role("order"):
                return ""
            lines = list(info.get("lines") or [])
            if not lines or any(i not in self.trolley.units or i in self.dead_lines() or i in self.refused_lines for i in lines):
                return ""
            if self.moves_left() < 1 or self.wire.done or self.wire.broke or self.wire.spare() < 2:
                return ""
            self.near_cart = None
            self.log.append({"kind": "near_order", "lines": lines})
            was = self.wire.mute
            try:
                self.finish_now()
            finally:
                self.wire.mute = was
            return ("Not run (no step used). A meaning check confirmed that the cart's lines (" + ", ".join(lines) + ") write the "
                    "requested values in other words, and nothing changed since, so the code ordered the cart instead of declining.")
        except Exception as err:
            self.log.append({"kind": "near_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def near_check(self) -> str:
        """Run near_basket once (near_done) when cart and order roles exist, four or more moves and spare budget above
        CLOSING + 30 remain, and the wire is not muted, covering, done or without a model. Errors log near_error and
        give an empty string.
        """
        if self.__dict__.get("near_done") or not self.role("cart") or not self.role("order") or self.moves_left() < 4 \
                or self.wire.spare() <= CLOSING + 30 or self.wire.mute or self.__dict__.get("covering") or self.wire.done \
                or self.wire.no_model:
            return ""
        try:
            return self.near_basket()
        except Exception as err:
            self.log.append({"kind": "near_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def near_basket(self) -> str:
        """Pre-decline: one meaning check of reworded values; may cart a basket.
        Each product takes its cheapest ed_pool fit; else read, in-stock,
        in-budget listings whose only gaps are values written differently
        in same-named fields.
        Keeps the fewest-gap ones, cheapest total within the cap.
        CHECKER judges up to 12 property lines same or not.
        The first confirmed one per product wins (passed, near_ok).
        Puts are muted; stale lines re-read.
        On success sets near_cart and asks to order this cart.
        """
        rows = self.needs.items()
        prods: list[int | None] = list(range(len(rows))) if rows else [None]
        counts = self.needs.counts()
        cap = float(self.needs.spend_cap) if is_number(self.needs.spend_cap) else None
        chosen: dict[int, tuple[str, float, int]] = {}
        pools: dict[int, list[tuple[int, float, str, list[tuple[str, str, str]]]]] = {}
        for n, k in enumerate(prods):
            qty = counts[n] if n < len(counts) else 1
            pool = self.ed_pool(k)
            fits = sorted((e["price"], e["id"]) for e in pool if e["fit"] and e["price"] is not None)
            if fits:
                chosen[n] = (fits[0][1], fits[0][0], qty)
                continue
            if not self.ed_news(k):
                return ""
            reqs = self.ed_reqs(k)
            cands = []
            for e in pool:
                if not e["read"] or e["gone"] or e["over"] or e["pointer"] or e["price"] is None:
                    continue
                lines, ok = [], True
                for prop, value, revealed in reqs:
                    if self.code_shows(e["row"], prop, value, set(), True):
                        continue
                    says = [str(v).strip() for key, v in e["row"]["fields"].items() if same_prop(key, prop) and str(v).strip()]
                    if revealed or not says or not self.ed_word(value) or any(re.search(r"\d", s) for s in says):
                        ok = False
                        break
                    lines.append((prop, value, "; ".join(dict.fromkeys(says))[:120]))
                fails = {str(x).split(":", 1)[0].strip() for x in self.field_view(e["id"])["fails"]} if e["ruled"] else set()
                probs = (self.verify_problems(e["id"]) or []) if e["ruled"] else []
                marks = {w for prop, value, _ in lines for w in prop_words(prop) + re.findall(r"[a-z0-9]+", value.lower())
                         if len(w) >= 3 and w not in LINK_WORDS}
                if ok and lines and all(any(same_prop(f, prop) for prop, _, _ in lines) for f in fails) \
                        and all(any(re.search(r"\b" + re.escape(w), p.lower()) for w in marks) for p in probs):
                    cands.append((len(lines), e["price"], e["id"], lines))
            if not cands:
                return ""
            pools[n] = sorted(cands)[: 6 if len(prods) == 1 else 3]
        if not pools:
            return ""
        least = sum(price * qty for _, price, qty in chosen.values()) + sum(
            min(c[1] for c in pools[n]) * (counts[n] if n < len(counts) else 1) for n in pools)
        if cap is not None and least > cap + 1e-9:
            return ""
        flat = [(n, ident, prop, value, says) for n in sorted(pools) for _, _, ident, lines in pools[n] for prop, value, says in lines][:12]
        text = "\n".join(f"{i + 1}. {ident}: {prop} = '{value[:60]}' | the listing's field says: {says}"
                         for i, (_, ident, prop, value, says) in enumerate(flat))
        ask = ("A shopping assistant is about to decline an order because the listings below do not write a requested value in "
               "the words of the request. THE SHOPPER'S REQUEST (context):\n" + self.request[:1500] + "\n\nEach line: a listing, "
               "a requested property with the requested value, and what the listing's own field says for it.\n" + text
               + "\n\nFor each line: does the listing's value name the requested value in other words (a synonym, another "
               "spelling or wording, or a more exact form of the same thing), or a different value of that property? When "
               "unsure, say false.\nAnswer with ONE JSON object, nothing else: {\"lines\": [{\"n\": 1, \"same\": true}]}")
        self.near_done = True
        answer = self.wire.consult(ask, 600, CHECKER)
        yes: set[int] = set()
        said_lines = answer.get("lines") if isinstance(answer, dict) and isinstance(answer.get("lines"), list) else []
        for entry in said_lines:
            if isinstance(entry, dict) and entry.get("same") is True and is_number(entry.get("n")):
                yes.add(int(entry["n"]) - 1)
        self.log.append({"kind": "near_check", "lines": [f"{i}:{p}={v}|{s}"[:120] for _, i, p, v, s in flat], "same": sorted(yes)})
        for n in sorted(pools):
            for _, price, ident, lines in pools[n]:
                idx = [i for i, row in enumerate(flat) if row[0] == n and row[1] == ident]
                if len(idx) == len(lines) and all(i in yes for i in idx):
                    chosen[n] = (ident, price, counts[n] if n < len(counts) else 1)
                    break
            if n not in chosen:
                return ""
        if cap is not None and sum(price * qty for _, price, qty in chosen.values()) > cap + 1e-9:
            return ""
        picks = [(chosen[n][0], chosen[n][2]) for n in range(len(prods))]
        if self.moves_left() < len(picks) + 2:
            return ""
        ver = str(self.needs.version)
        for n in pools:
            key = f"{chosen[n][0]}|{ver}"
            for memo in [m for m in self.rulings if m == key or m.startswith(key + "|")]:
                self.rulings[memo] = ""
            self.passed.add(key)
            self.__dict__.setdefault("near_ok", set()).add(key)
            self.__dict__.setdefault("pass_price", {})[chosen[n][0]] = self.archive.price(chosen[n][0])
        done = False
        was = self.wire.mute
        self.wire.mute = True
        try:
            bad = self.code_put(picks)
            done = bad == [] and all(i in self.trolley.units for i, _ in picks)
            stale_now = [i for i in self.stale_lines() if any(i == p for p, _ in picks)]
            if done and stale_now and self.moves_left() >= len(stale_now) + 2:
                _, sold = self.code_recheck(stale_now)
                done = not sold
        finally:
            self.wire.mute = was
        done = done and bool(self.trolley.units) and not set(self.trolley.units) & self.dead_lines()
        self.log.append({"kind": "near_basket", "picks": picks, "done": done, "cart": dict(self.trolley.units)})
        if not done:
            return ""
        self.near_cart = {"ver": self.needs.version, "refusals": self.refusal_n, "lines": [i for i, _ in picks]}
        said_by = self.__dict__.setdefault("near_said", {})
        for n, ident, prop, value, says in flat:
            if ident == chosen.get(n, ("",))[0]:
                said_by.setdefault(f"{ident}|{ver}", []).append((prop, value, says))
        said = "; ".join(f"{ident} writes {prop} as '{says}' for '{value[:60]}'" for n, ident, prop, value, says in flat
                         if ident == chosen.get(n, ("",))[0])
        return ("Not run yet (no step used). A meaning check found that " + said + " - the same value in other words - and each "
                "such listing meets every need the shopper revealed, is in stock and within the budget, so the code put the basket "
                "in the cart and read its stock: " + self.trolley.summary() + " Order this cart now; decline only if a record "
                "you read shows one of these lines failing a requirement now in force.")

    def cart_moved(self, before: dict[str, float]) -> None:
        """Update cart tracking after the cart changed, given the cart units from before: bump cart_ver, reset
        since_cart, move cart_seq to the current absorb count when a new line appeared, and clear the stock-read
        mark (line_seq) of lines whose units grew, so stale_lines reports them.
        """
        self.cart_ver = self.__dict__.get("cart_ver", 0) + 1
        self.since_cart = 0
        now = self.trolley.units
        if any(line not in before for line in now):
            self.cart_seq = self.__dict__.get("absorb_n", 0)
        seqs = self.__dict__.setdefault("line_seq", {})
        for line, units in now.items():
            if line in before and units > before[line] + 1e-9:
                seqs.pop(line, None)

    def after_put(self, call_args: dict[str, Any], obs: Any) -> None:
        """After a cart call, note cart_shown when the observation shows the cart, and queue a stock re-read of lines
        put in for the first time whose stock reading is stale. Needs a put call without a fault, a stock tool, four
        or more moves and spare budget of CLOSING, and no mute or covering. Up to four lines go to put_queue with
        put_tool; logged as after_put_read.
        """
        if self.trolley.view_units(obs) is not None:
            self.cart_shown = self.__dict__.get("cart_ver", 0)
        firsts = self.__dict__.setdefault("first_puts", set())
        new = [line for line, count in self.trolley.lines(call_args) if (count is None or count > 0) and line in self.trolley.units
               and line not in firsts]
        if not new or not self.trolley.is_put(call_args) or self.code_fault(obs):
            return
        firsts.update(new)
        stale = set(self.stale_lines())
        new = [line for line in new if line in stale]
        tool = self.stock_tool()
        if not new or not tool or self.wire.mute or self.__dict__.get("covering") or self.moves_left() < 4 or self.wire.spare() < CLOSING:
            return
        queue = list(self.__dict__.get("put_queue") or [])
        self.put_queue = (queue + [line for line in new if line not in queue])[:4]
        self.put_tool = tool
        self.log.append({"kind": "after_put_read", "lines": list(self.put_queue), "tool": tool})

    def mark_fresh(self, obs: Any) -> None:
        """Once stock states have been seen, stamp each cart line whose record in the observation shows a stock state
        with the current absorb count in line_seq, so its stock reading counts as fresh. Lines without a stamp newer
        than cart_seq stay stale.
        """
        if not self.__dict__.get("stock_seen"):
            return
        seqs = self.__dict__.setdefault("line_seq", {})

        def visit(d: dict[str, Any]) -> None:
            """Stamp the dict's listing in line_seq when it is a cart line and the record shows its stock."""
            ident = self.rec_id(d)
            if ident in self.trolley.units and self.stock_state(d) is not None:
                seqs[ident] = self.__dict__.get("absorb_n", 0)
        walk_dicts(obs, visit)

    def cart_gone(self, lines: list[str]) -> set[str]:
        """Return which of the given cart lines are sold out: those already dead, plus any code_recheck finds sold when
        stale lines are re-read (only with a stock tool and enough moves left).
        """
        gone = set(lines) & self.dead_lines()
        stale = [i for i in self.stale_lines() if i in lines and i not in gone]
        if stale and self.stock_tool() and self.moves_left() >= len(stale) + 2:
            _, sold = self.code_recheck(stale)
            gone |= set(sold) | (set(lines) & self.dead_lines())
        return gone

    def batch_tool(self) -> tuple[str, str, int] | None:
        """Find, once, a tool that reads several listings per call and cache it in batch_spec as (tool, array argument,
        ids per call). It needs exactly one array-of-strings argument without enum, nothing else required and two or
        more items allowed; tools mapped to search, message, cart, order or decline roles, or whose name or text
        speaks of carts, removal, purchase or wish lists, are excluded. The limit is maxItems, lowered by an up-to
        figure in the description and capped at five; None when no tool qualifies.
        """
        if "batch_spec" not in self.__dict__:
            spec = None
            try:
                taken = {self.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
                said = {}
                for tdef in self.toolset or []:
                    fn = tdef.get("function") if isinstance(tdef, dict) else None
                    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                        said[fn["name"]] = re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ") + " " + str(fn.get("description") or "")
                for name in list(self.shapes):
                    if re.search(r"\b(?:remov|delet|cancel|clear|discard|releas|refund|purchas|checkout|wish|basket|cart|bag)", said.get(name, name).lower()):
                        taken.add(name)
                for name, jschema in self.shapes.items():
                    props = jschema.get("properties") if isinstance(jschema, dict) else None
                    props = props if isinstance(props, dict) else {}
                    arrays = [key for key, p in props.items() if isinstance(p, dict) and p.get("type") == "array"
                              and isinstance(p.get("items"), dict) and p["items"].get("type") == "string" and not p["items"].get("enum")]
                    needed = jschema.get("required") if isinstance(jschema, dict) and isinstance(jschema.get("required"), list) else []
                    most = props[arrays[0]].get("maxItems") if len(arrays) == 1 else None
                    told = re.search(r"\b(?:up to|at most|no more than|maximum of|max\.?)\s+(\d+)", said.get(name, "").lower())
                    if is_number(most) and told and 2 <= int(told.group(1)) < most:
                        most = int(told.group(1))
                    if name not in taken and len(arrays) == 1 and not [r for r in needed if r not in arrays] and is_number(most) and most >= 2:
                        spec = (name, arrays[0], int(min(most, 5)))
                        break
            except Exception:
                spec = None
            self.batch_spec = spec
        return self.batch_spec

    def batch_ok(self, found: list[Any], ids: list[str]) -> bool:
        """Check that a batch read returned real listing records for the requested ids. Ids count when found in dicts
        that hold a nested property map, or in flat dicts with five or more short strings, the latter only when the
        details tool is not known to return property maps. True when at least half of the ids were found; open_named
        drops the batch tool when its first read fails this check.
        """
        got: set[str] = set()
        wanted = set(ids)
        details = self.role("details")
        dcaps = ((self.__dict__.get("tool_caps") or {}).get(details or "") or {})
        maps_only = bool(details) and details != (self.__dict__.get("batch_spec") or (None,))[0] and (
            dcaps.get("props", 0) > 0 or not dcaps.get("id", 0))

        def visit(d: dict[str, Any]) -> None:
            """Add requested ids found as values of a dict that looks like a listing record (a nested property map, or a
            flat record of short strings when those are accepted).
            """
            nested = any(isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2 for v in d.values())
            flat = not any(isinstance(v, dict) for v in d.values()) and sum(1 for v in d.values() if isinstance(v, str) and 0 < len(v) <= 70) >= 5
            if nested or (flat and not maps_only):
                got.update(v for v in d.values() if isinstance(v, str) and v in wanted)
        try:
            for obs in found:
                walk_dicts(obs, visit)
        except Exception:
            return False
        return bool(got) and 2 * len(got) >= len(wanted)

    @staticmethod
    def title_of(row: dict[str, Any]) -> str:
        """Return a row's title field, else the first string under a title, heading or headline key, else an empty
        string.
        """
        if isinstance(row.get("title"), str):
            return row["title"]
        return next((v for key, v in row.items() if isinstance(v, str) and re.search(r"title|heading|headline", str(key).lower())), "")

    def menu_rows(self, home: set[str], most: int, taken: list[str], qty: int = 1) -> list[str]:
        """Picks up to most unread listings on given category labels to open,
        learning store fields. Candidates: archived label rows not read,
        sold out, taken or refused, within spend cap at given quantity; by
        search hit score, then first seen. If no read listing yet shows all
        stated positive values, the first label row from the logged search
        carrying most of them leads. Rest: first 16 candidates; search
        hits, then row_score rows, then titles least like read/picked ones.
        """
        cap = self.needs.spend_cap
        order: list[str] = []
        titles: dict[str, str] = {}
        sold: set[str] = set()
        known: list[set[str]] = []

        def visit(d: dict[str, Any]) -> None:
            """Sort one row dict on the given labels: read listings add their title words to the known sets used for
            similarity; unread ones are listed in first-seen order with their title and marked sold when sold_out says
            so. Rows on other labels or with unknown ids are ignored.
            """
            ident = self.row_mark(d)
            if ident not in self.archive.ids or self.shelf_of(d) not in home:
                return
            if ident in self.archive.read:
                known.append(set(re.findall(r"[a-z0-9]{3,}", self.title_of(d).lower())))
                return
            if self.sold_out(d):
                sold.add(ident)
            if ident not in order:
                order.append(ident)
                titles[ident] = self.title_of(d)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        hits = sorted(self.__dict__.get("all_hits", []), key=lambda i: -self.__dict__.get("hit_score", {}).get(i, 0))
        cands = [i for i in hits if i in titles] + [i for i in order if i not in hits]
        cands = [i for i in cands if i not in sold and i not in taken and i not in self.refused_lines
                 and not (is_number(cap) and is_number(self.archive.price(i)) and self.archive.price(i) * qty > cap)]
        words = {i: set(re.findall(r"[a-z0-9]{3,}", titles.get(i, "").lower())) for i in cands}
        lead = self.__dict__.get("row_score", {})
        picks: list[str] = []
        try:
            several = bool(self.needs.items())
            k = next((n for n, h in enumerate(self.home_shelves()) if h == home), None) if several else None
            vals = [re.sub(r"\((?:stated|revealed[^)]*)\)", "", r.partition(":")[2]) for r in self.needs.props_left()
                    if not several or self.product_of(r.partition(":")[0]) in (k, None)]
            vals = [v for v in vals if v.strip() and positive_value(v) and not re.search(r"as shown|same as|the page|sold by|redirect|refer", v.lower())]
            facts: dict[str, list[str]] = {}

            def fact_visit(d: dict[str, Any]) -> None:
                """Collect the string and number values in the property maps of read listings on the labels, keyed by listing
                id.
                """
                ident = self.row_mark(d)
                if ident in self.archive.read and self.shelf_of(d) in home:
                    for val in d.values():
                        if isinstance(val, dict):
                            facts.setdefault(ident, []).extend(str(x) for x in val.values() if isinstance(x, (str, int, float)))
            for rec in self.archive.items if vals and most > 0 and (not several or k is not None) else []:
                walk_dicts(rec, fact_visit)
            fit_read = any(all(value_seen(v, " | ".join(texts)) for v in vals) for texts in facts.values())
            if vals and most > 0 and (not several or k is not None) and not fit_read:
                led = sorted((w for w in self.__dict__.get("page_book", {}).values() if self.walk_carries(w["args"], k)),
                             key=lambda w: -len(self.walk_carries(w["args"], k)))
                for walk in led:
                    first = next((i for i in (walk.get("row_shelf") or {}) if i in words), None)
                    if first:
                        picks.append(first)
                        known.append(words[first])
                        break
        except Exception:
            picks = []
        while len(picks) < most:
            pool = [i for i in cands if i not in picks][:16]
            if not pool:
                break
            best = min(pool, key=lambda i: (0 if i in hits else 1 if lead.get(i) else 2,
                                            max((len(words[i] & w) / max(1, len(words[i] | w)) for w in known), default=0.0), cands.index(i)))
            picks.append(best)
            known.append(words[best])
        return picks

    def pre_read(self) -> bool:
        """Queues early batch reads of product fields pre-questions; True if queued.
        Needs batch tool, <=1 question asked, >20 moves (25 in round 2), spare
        budget CLOSING+60. Round 1: 2 menu_rows picks per product w/o rich reads
        (4 if one product), top-up <4; cap 1 batch (2 if many products empty, >25
        moves left); sets pre_read_done, pre_rounds, pre_picked. Round 2:
        alternating picks for products under 3 rich reads, cap 1 batch. Both set
        open_queue, open_recheck, open_say, purpose menu. Errors: pre_read_error.
        """
        try:
            batch = self.batch_tool()
            again = self.__dict__.get("pre_rounds", 0) == 1
            if not batch or (self.__dict__.get("pre_read_done") and not again) or len(self.questions_log) > 1 \
                    or self.moves_left() <= (25 if again else 20) or self.wire.spare() < CLOSING + 60:
                return False
            homes = self.home_shelves()
            qtys = [row["quantity"] for row in self.needs.items()] or [self.needs.quantity()]
            if again:
                self.pre_rounds = 2
                wants: list[list[str]] = []
                taken: list[str] = []
                for k, home in enumerate(homes):
                    have = sum(self.archive.rich_reads(s) for s in home) if home else 4
                    if have < 3:
                        rows = self.menu_rows(home, 4 - have, taken + list(self.__dict__.get("pre_picked") or []), qtys[min(k, len(qtys) - 1)])
                        taken += rows
                        wants.append(rows)
                picks: list[str] = []
                while any(wants):
                    for rows in wants:
                        if rows:
                            picks.append(rows.pop(0))
                picks = picks[: batch[2]]
                if not picks:
                    return False
                self.open_queue = picks
                self.open_recheck = True
                self.open_say = "Some products' first listings showed few fields, so I open a few more before asking."
                self.open_purpose = "menu"
                return True
            empty: list[str] = []
            thin: list[tuple[int, int]] = []
            for k, home in enumerate(homes):
                have = sum(self.archive.rich_reads(s) for s in home) if home else 4
                if have == 0:
                    empty += self.menu_rows(home, 2 if self.needs.items() else 4, empty, qtys[min(k, len(qtys) - 1)])
                elif have < 4:
                    thin.append((k, 4 - have))
            picks = list(empty)
            for k, n in thin:
                picks += self.menu_rows(homes[k], n, picks, qtys[min(k, len(qtys) - 1)])
            picks = picks[: batch[2] * (2 if len(empty) > batch[2] and self.moves_left() > 25 else 1)]
            if not picks:
                return False
            self.pre_read_done = True
            self.pre_rounds = 1
            self.pre_picked = list(picks)
            self.open_queue = picks
            self.open_recheck = True
            self.open_say = "Before asking the shopper anything, I open a few listings of each product asked for to learn their fields."
            self.open_purpose = "menu"
            return True
        except Exception as err:
            self.log.append({"kind": "pre_read_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def lacking_rows(self) -> list[str]:
        """For a multi-product request, return unread listings for products that have no verified fit or audit fit on
        their home labels yet, up to one batch (three without a batch tool). Each listing takes the newest category
        label seen for it; read, refused and sold-out ones are skipped, as are ones over the spend cap at the
        product's quantity. Candidates are ordered by search hit score, then recency, and picks alternate between
        the lacking products. Errors log lacking_error and give [].
        """
        try:
            rows = self.needs.items()
            if not rows:
                return []
            homes = self.home_shelves()
            live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == str(self.needs.version)} - self.refused_lines - self.dead_lines()
            got = {self.shelf_known(i) for i in live | set(self.__dict__.get("audit_fits") or [])} - {""}
            lacking = [k for k, home in enumerate(homes) if home and not home & got]
            if not lacking:
                return []
            batch = self.batch_tool()
            most = batch[2] if batch else 3
            cap = self.needs.spend_cap
            newest: list[tuple[str, str]] = []
            sold: set[str] = set()

            def visit(d: dict[str, Any]) -> None:
                """Record (id, label) for a row dict with a known listing id and a label, and mark it sold when sold_out says
                so.
                """
                ident, shelf = self.row_mark(d), self.shelf_of(d)
                if ident in self.archive.ids and shelf:
                    newest.append((ident, shelf))
                    if self.sold_out(d):
                        sold.add(ident)
            for rec in reversed(self.archive.items):
                walk_dicts(rec, visit)
            where: dict[str, str] = {}
            for ident, shelf in newest:
                where.setdefault(ident, shelf)
            hits = sorted(self.__dict__.get("all_hits", []), key=lambda i: -self.__dict__.get("hit_score", {}).get(i, 0))
            pools: dict[int, list[str]] = {k: [] for k in lacking}
            for ident in dict.fromkeys([i for i in hits if i in where] + [i for i, _ in newest]):
                if ident in self.archive.read or ident in self.refused_lines or ident in sold:
                    continue
                for k in lacking:
                    if where.get(ident) in homes[k]:
                        cost = self.archive.price(ident)
                        if not (is_number(cap) and is_number(cost) and cost * rows[k]["quantity"] > cap):
                            pools[k].append(ident)
                        break
            picks: list[str] = []
            while len(picks) < most and any(pools.values()):
                for k in lacking:
                    if pools[k] and len(picks) < most:
                        picks.append(pools[k].pop(0))
            return picks
        except Exception as err:
            self.log.append({"kind": "lacking_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

    def text_arg(self, the_tool: str) -> str | None:
        """Find the one free-text argument a tool takes: its required non-enum string argument, or its only one when
        none is required. None when there is no such argument or the tool requires any other argument. message_call
        puts the question text there.
        """
        jschema = self.shapes.get(the_tool or "") or {}
        prop_specs = jschema.get("properties") or {}
        req_names = list(jschema.get("required") or [])
        str_params = [fkey for fkey, val in prop_specs.items() if isinstance(val, dict) and (val.get("type") == "string" or "string" in (val.get("type") or [])) and "enum" not in val]
        aim = (next((fkey for fkey in req_names if fkey in str_params and fixed_arg(prop_specs.get(fkey)) is None), None)
               or next((fkey for fkey in req_names if fkey in str_params), None) or (str_params[0] if len(str_params) == 1 else None))
        if not aim or any(fkey != aim and fixed_arg(prop_specs.get(fkey)) is None for fkey in req_names):
            return None
        return aim

    def message_call(self, text_body: str, cid: str) -> dict[str, Any] | None:
        """Build a call proposal that sends a text to the shopper with the message tool under the given call id,
        putting the text in the tool's free-text argument and validating it with ArgCheck. None without a message
        tool, a text argument, or when validation reports faults.
        """
        the_tool = self.role("message")
        aim = self.text_arg(the_tool) if the_tool else None
        if not aim:
            return None
        jschema = self.shapes[the_tool]
        filled = {fkey: fixed_arg((jschema.get("properties") or {}).get(fkey)) for fkey in jschema.get("required") or [] if fkey != aim}
        call_args, faults = ArgCheck.execute({**filled, aim: text_body}, jschema)
        if faults:
            return None
        return {"id": cid, "type": "function", "function": {"name": the_tool, "arguments": json.dumps(call_args)}}

    def product_of(self, text: str) -> int | None:
        """Map a text to the index of the requested product it names (multi-product requests only). The first
        parenthesised tag, then the whole text, is scored by kind tokens against each product with kind_score, the
        other products' tokens counting as shared. Returns the index with a unique positive top score, or None.
        """
        rows = self.needs.items()
        if not rows:
            return None
        names = [set(kind_tokens(row["product"])) for row in rows]
        for piece in re.findall(r"\(([^)]*)\)", str(text))[:1] + [str(text)]:
            words = set(kind_tokens(piece))
            scores = []
            for k, own in enumerate(names):
                shared = set().union(*(names[j] for j in range(len(names)) if j != k))
                scores.append(kind_score(own, words, shared))
            top = max(scores)
            if top > 0 and scores.count(top) == 1:
                return scores.index(top)
            exact = [k for k, own in enumerate(names) if own and len(own) == len(words) and kinds_within(own, words)
                     and kinds_within(words, own)]      # (CSb: singular and plural alike)
            if top > 0 and len(exact) == 1 and scores[exact[0]] == top:
                return exact[0]      # (PO: the product of exactly that name, beside a sibling whose name holds it)
        return None

    def plan_product(self, qitem: dict[str, Any]) -> int | None:
        """Return the product index a planned question is about, judged from its property and else its question text."""
        k = self.product_of(str(qitem.get("property") or ""))
        return k if k is not None else self.product_of(str(qitem.get("question") or ""))

    def quota_first(self, scripted: list[dict[str, Any]], ok: Any = None) -> list[dict[str, Any]]:
        """Multi-product: reorders the plan so the next question hits the neediest
        product. Per product: questions sent/answered (questions_log), uncovered
        needs, open story hints (field-naming, or plain hints > answers). Groups
        by level_of; first group: front a planned question on one, split a
        planned second property of one into its own question, or add an unasked
        quota_pick one passing ok. Kept if no product has answers and item one
        names none, or quota_pick queued a read (quota_wait).
        """
        rows = self.needs.items()
        if not rows:
            return scripted
        asked = [0] * len(rows)
        sent = [0] * len(rows)
        for ln in self.questions_log:
            head = ln.split(":", 1)[0]
            k = self.product_of(head)
            if k is None:
                said = re.search(r'asked "(.*?)" -> ', ln)
                k = self.product_of(said.group(1)) if said else None
            if k is not None:
                sent[k] += 1
                if "[not answered]" not in head:
                    asked[k] += 1
        if max(asked) == 0 and scripted and self.plan_product(scripted[0]) is None:
            return scripted
        found = [0] * len(rows)
        for need in self.needs.uncovered:
            k = self.product_of(str(need).split(":", 1)[0])
            if k is not None:
                found[k] += 1
        try:
            hinted = {self.product_of(str(h).split(":", 1)[0]) for h in self.open_hints()}
        except Exception:
            hinted = set()
        try:
            named, keyless = set(), [0] * len(rows)
            for h in self.open_hints():
                j = self.product_of(str(h).split(":", 1)[0])
                if j is None:
                    continue
                if self.plain_hint(j, str(h).split(":", 1)[0].strip(), hinted=True):
                    keyless[j] += 1
                else:
                    named.add(j)
            hinted = {j for j in range(len(rows)) if j in named or keyless[j] > asked[j]}
        except Exception:
            hinted = set()

        alt_left = {j for j in range(len(rows)) if sent[j] > asked[j] and self.alt_untried(j)}
        fixed = self.ref_fixed()

        def level_of(k: int) -> tuple[int, ...]:
            """Ordering key for a product: later when references fix it while others still lack answers, when it already
            has found needs while others have none, when a product tied to it is unasked, or when no hint points at it;
            then by answers plus found needs, unanswered sends (zero while another wording is untried) and found needs.
            """
            return (1 if k in fixed and any(j not in fixed and not found[j] and asked[j] < 2 for j in range(len(rows))) else 0,
                    1 if found[k] and any(j != k and not found[j] for j in range(len(rows))) else 0,
                    1 if any(t["main"] == k and not asked[t["k"]] for t in self.ties) else 0,
                    1 if asked[k] + found[k] > 0 and any(asked[j] + found[j] == 0 for j in range(len(rows))) else 0,
                    0 if k in hinted else 1, asked[k] + found[k],
                    0 if k in alt_left else sent[k] - asked[k], found[k])
        ready = list(range(len(rows)))
        for level in sorted({level_of(k) for k in ready}):
            due = [k for k in ready if level_of(k) == level]
            for n, qitem in enumerate(scripted):
                if self.plan_product(qitem) in due:
                    return [qitem] + scripted[:n] + scripted[n + 1:]
            for qitem in scripted:
                second = str(qitem.get("second") or "").strip()
                j = self.product_of(second) if second else None
                if j in due:
                    tied = re.search(r"\(([^)]+)\)", second)
                    product = tied.group(1).strip() if tied else rows[j]["product"]
                    return [{"property": second, "question": f"Which {second.split('(')[0].strip()} would you like for the {product}?"}] + scripted
            for k in sorted(due, key=lambda k: (k not in hinted, k)):
                try:
                    pick = self.quota_pick(k, asked[k] == 0 or (found[k] == 0 and asked[k] < 2), ok)
                except Exception:
                    pick = None
                if pick and ok is not None and (not ok(pick.get("property")) or self.asked_for(k, pick.get("property"))):
                    pick = None
                if pick:
                    return [pick] + scripted
                if self.__dict__.get("quota_wait"):
                    return scripted
        return scripted

    def quota_pick(self, k: int, fallback: bool = True, ok: Any = None) -> dict[str, Any] | None:
        """Chooses a question property for product k.
        With no read listings of its category yet, queues a read of up to four rows and returns None.
        Else tries a product_plan pick, then an open story-hint property,
        then ranked properties and menu lines still open.
        Every pick must pass ok and be unasked.
        Returns {property, question} tagged with the product, or None.
        """
        product = self.needs.items()[k]["product"]
        homes = self.home_shelves()
        home = homes[k] if k < len(homes) else set()
        read_home = [s for s in self.archive.shelves_read() if s.strip().lower() in home]
        if not read_home and home and self.__dict__.get("quota_reads", {}).get(k):
            read_home = [s for s in sorted(home) if self.archive.prop_menu(s)[0]][:1]
        if not read_home:
            batch = self.batch_tool()
            unread = self.menu_rows(home, min(4, batch[2]) if batch else 2, [], self.needs.items()[k]["quantity"]) if home else []
            if unread and (self.role("details") or batch) and not self.__dict__.get("quota_reads", {}).get(k) and self.moves_left() > 12:
                self.__dict__.setdefault("quota_reads", {})[k] = True
                self.open_queue = unread
                self.open_recheck = True
                self.open_say = f"Before asking about the {product}, I open a few of its listings to learn their fields."
                self.open_purpose = "quota"
                self.quota_wait = True
            return None
        menu = self.menu_trim(self.archive.prop_menu(read_home[0])[0], self.archive.rich_reads(read_home[0]))

        def good(label: Any) -> bool:
            """Tell whether a property passes the caller's ok filter and was not yet asked for this product."""
            return ok is None or (bool(ok(label)) and not self.asked_for(k, label))
        pick = self.product_plan(k, menu)
        if pick and good(pick.get("property")):
            return pick
        try:
            hinted = self.open_hints(k)
        except Exception:
            hinted = []
        for hint in hinted:
            prop = re.split(r"\s*\(", hint.split(":", 1)[0])[0].strip()
            if prop and not junk_key(prop) and not self.plain_hint(k, prop, hinted=True) and good(f"{prop} ({product})"):
                return {"property": f"{prop} ({product})", "question": f"Which {prop} would you like for the {product}?"}
        if not fallback:
            return None
        try:
            ranked_keys = self.rank_props(k)
        except Exception:
            ranked_keys = []
        for weigh in (False, True):
            for name in ranked_keys:
                line = self.prop_line(k, name)
                prop = line.strip("- ").split(" (in ")[0].strip() if line else ""
                if prop and self.open_prop(k, prop, line, by_samples=weigh) and good(f"{prop} ({product})"):
                    self.log.append({"kind": "quota_ranked_pick", "product": k, "property": prop[:60]})
                    return {"property": f"{prop} ({product})", "question": f"Which {prop} would you like for the {product}?"}
        for weigh in (False, True):
            for line in menu.splitlines():
                prop = line.strip("- ").split(" (in ")[0].strip()
                if prop and self.open_prop(k, prop, line, by_samples=weigh) and good(f"{prop} ({product})"):
                    return {"property": f"{prop} ({product})", "question": f"Which {prop} would you like for the {product}?"}
        return None

    def product_plan(self, k: int, menu: str) -> dict[str, Any] | None:
        """Return product_plan_call's question pick for product k from the given property menu, or None if the planner
        call raises.
        """
        try:
            return self.product_plan_call(k, menu)
        except Exception:
            return None

    def product_plan_call(self, k: int, menu: str) -> dict[str, Any] | None:
        """Asks the planner once per episode which property of product k to ask about.
        Needs a menu and spare budget of CLOSING + 40.
        ASK_PLAN gets the request, requirements, past questions and replies,
        the rule that each question concerns this product, story hints,
        plan notes, the product's menu and the ranked note.
        Returns the first open {property, second, question}, tagged with the product.
        None otherwise.
        """
        if not menu or self.__dict__.get("product_plans", 0) >= 1 or self.wire.spare() < CLOSING + 40:
            return None
        self.product_plans = self.__dict__.get("product_plans", 0) + 1
        product = self.needs.items()[k]["product"]
        remaining = max(1, self.ask_limit() - self.msgs_sent)
        ask_text = ASK_PLAN.replace("QUESTION_COUNT", str(min(3, remaining))) + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
        ask_text += "\n\nREQUIREMENTS SO FAR:\n" + self.needs.text_body()
        if self.questions_log:
            ask_text += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.questions_log)
        ask_text += (f"\n\nTHIS TIME EVERY QUESTION IS ABOUT THE {product.upper()}: every other product asked for already had a "
                     f"question and this one had none. Choose properties of the {product} only, from its own list below, the "
                     "one the story's details point at first.")
        try:
            hinted = self.open_hints(k)
        except Exception:
            hinted = []
        if hinted:
            ask_text += ("\n\nSTORY HINTS (details of the request's story and the property each seems to concern, the likeliest named first; not requirements):\n"
                         + "\n".join(f"- {self.hint_mark(h)}" for h in hinted[:6]))
        ask_text += self.plan_notes(k)
        ask_text += (f"\n\nPROPERTIES IN THE DETAILS OF THE {product.upper()} LISTINGS SEEN SO FAR (name, number of listings, "
                     "example values):\n" + menu)
        try:
            ask_text += self.ranked_note(None, k)
        except Exception as err:
            self.log.append({"kind": "ranked_note_error", "error": f"{type(err).__name__}: {err}"[:300]})
        answer = self.wire.consult(ask_text, 1500, PLANNER[:1])
        self.log.append({"kind": "product_plan", "product": product, "result": answer})
        words = kind_words(product)
        for qitem in (answer or {}).get("questions") or []:
            if not isinstance(qitem, dict):
                continue
            first = str(qitem.get("property") or "").split("(")[0].strip()
            second = str(qitem.get("second") or "").split("(")[0].strip()
            second = "" if re.search(r"none|^-$|n/a", second.lower()) else second
            question = str(qitem.get("question") or "").strip()
            first = "" if self.plain_hint(k, first) else first
            second = "" if second and self.plain_hint(k, second) else second
            if first and self.plan_open(k, first):
                named = bool(question) and bool(words & kind_words(question))
                pair = second if named and second and not strict_prop(first, second) and self.plan_open(k, second) else ""
                return {"property": f"{first} ({product})", "second": f"{pair} ({product})" if pair else "",
                        "question": question if named else f"Which {first} would you like for the {product}?"}
            if second and self.plan_open(k, second):
                return {"property": f"{second} ({product})", "question": f"Which {second} would you like for the {product}?"}
        return None

    def open_prop(self, k: int | None, prop: str, line: str = "", by_samples: bool = False) -> bool:
        """Tell whether a property of product k is still worth asking about. False for junk names, properties an
        answered question already covered, a requirement on the same property that settles it (judged with the menu
        samples when by_samples), a positive value of a similar property already among the samples or field values,
        a word value whose words all appear in the samples, or an indifferent need on the same field. An error
        counts as open.
        """
        try:
            name = str(prop).split("(")[0].strip()
            if not name or junk_key(name):
                return False
            several = bool(self.needs.items())

            def mine(text: str, tagged_only: bool = False) -> bool:
                """Tell whether a text belongs to product k: always for one product; otherwise when product_of maps it to k, or
                to no product unless tagged_only is set.
                """
                if k is None or not several:
                    return True
                j = self.product_of(text)
                return j == k or (j is None and not tagged_only)
            for ln in self.questions_log:
                head = ln.split(":", 1)[0].strip("- ")
                if "[not answered]" not in head and any(mine(part) and self.same_field(k, part, name) for part in head.split(";")):
                    return False
            samples = line.split("):", 1)[1].lower() if "):" in line else ""
            for req in self.needs.props_left():
                head, _, value = str(req).partition(":")
                if not mine(head):
                    continue
                if strict_prop(head, name):
                    if self.settles(str(req), samples if by_samples else ""):
                        return False
                    continue
                if same_prop(head, name):
                    if positive_value(value, head) and value_seen(value, samples or " | ".join(self.field_values(k, name))):
                        return False
                    continue
                words = [w for w in re.findall(r"[a-z]{3,}", value.lower()) if w not in LINK_WORDS]
                if samples and words and not re.search(r"\d", value) and positive_value(value, head) \
                        and all(re.search(r"\b" + re.escape(w), samples) for w in words):
                    return False
            return not any(mine(str(p), tagged_only=True) and self.same_field(k, str(p), name) for p in self.needs.indifferent)
        except Exception:
            return True

    def plan_notes(self, k: int | None) -> str:
        """Build planner prompt notes for product k. The first lists second properties of earlier question pairs whose
        reply left them open (open_pair, last three), keeping only ones for this product that were not asked and
        plan_open still allows. The second lists questions the assistant itself wanted to send (wanted_asks, last
        three) for this product. Empty when there are none or on error.
        """
        try:
            several = bool(self.needs.items())
            asked = {p.strip().lower() for ln in self.questions_log for p in ln.split(":", 1)[0].strip("- ").split(";")}
            opens = []
            for prop in self.__dict__.get("open_pair") or []:
                j = self.product_of(prop) if several else None
                if (k is not None and several and j not in (k, None)) or prop.lower() in asked or not self.plan_open(j, prop):
                    continue
                opens.append(prop)
            wanted = [w for w in self.__dict__.get("wanted_asks") or [] if k is None or not several or self.product_of(w) in (k, None)]
            text = ""
            if opens:
                text += ("\n\nOPEN FROM LAST PAIR (the second property of a planned pair whose reply did not answer it - the shopper "
                         "answered only the first, or only the first was sent - so it is still open):\n"
                         + "\n".join(f"- {p}" for p in opens[-3:]))
            if wanted:
                text += ("\n\nQUESTIONS THE ASSISTANT WANTED TO ASK (it has read the listing records; a property such a question names "
                         "that is still open and is a key below comes first, asked by the key's own name):\n" + "\n".join(f"- {w}" for w in wanted[-3:]))
            return text
        except Exception:
            return ""

    def want_ask(self, raw_args: Any) -> None:
        """Remember the text of a message the assistant wanted to send to the shopper: string argument values from a
        dict or JSON string, joined and cut to 300 characters, go to wanted_asks (earlier copy removed, last three
        kept). Errors are ignored.
        """
        try:
            args = raw_args if isinstance(raw_args, dict) else (json_object(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {})
            text = " ".join(str(v).strip() for v in (args or {}).values() if isinstance(v, str) and v.strip())[:300]
            if text:
                kept = [w for w in self.__dict__.get("wanted_asks") or [] if w.lower() != text.lower()]
                self.wanted_asks = (kept + [text])[-3:]
        except Exception:
            pass

    def drop_wanted(self, label: str) -> None:
        """Remove from wanted_asks each wanted question whose words (stemmed) include every word of a property in an
        asked label, once that property has been asked. The label is split on semicolons and the word and, with its
        not-answered mark removed. Errors are ignored.
        """
        try:
            props = [re.split(r"\s*\(", p)[0] for p in re.split(r";|\band\b", str(label).replace("[not answered]", ""))]
            keep = []
            for text in self.__dict__.get("wanted_asks") or []:
                words = {self.stem(w) for w in re.findall(r"[a-z]+", text.lower())}
                if any(prop_words(p) and all(w in words or self.stem(w) in words for w in prop_words(p)) for p in props):
                    continue
                keep.append(text)
            self.wanted_asks = keep
        except Exception:
            pass

    def names_other(self, question: str, prop: str, menu: str, second: str = "") -> bool:
        """Tell whether a question also names a property other than the one asked: the second property when it differs,
        any other menu key whose words all appear in the question with one outside the asked property and product
        words, or a menu sample value written as a whole phrase in the question. pose_planned then rewrites the
        question plainly. Errors give False.
        """
        try:
            flat = " ".join(re.findall(r"[a-z0-9]+", str(question).lower()))
            said = set(flat.split())
            said |= {w[:-1] for w in said if len(w) > 3 and w.endswith("s") and not w.endswith("ss")}
            own = set(prop_words(prop)) | {"which", "would", "you", "like"}
            for row in self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}]:
                for w in re.findall(r"[a-z0-9]+", str(row.get("product") or "").lower()):
                    own |= {w, w[:-1] if len(w) > 3 and w.endswith("s") else w}

            def named(name: str) -> bool:
                """Tell whether every content word of a property name is in the question and one of them is not part of the
                asked property or the product names.
                """
                words = [w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
                return bool(words) and all(w in said for w in words) and any(w not in own for w in words)
            if second and not strict_prop(second, prop) and named(second):
                return True
            for line in str(menu or "").splitlines():
                key = line.strip().lstrip("-").strip().split(" (in ")[0].strip()
                if not key or key.endswith(":") or strict_prop(key, prop) or junk_key(key):
                    continue
                if named(key):
                    return True
                for sample in (line.split("):", 1)[1] if "):" in line else "").split("|"):
                    toks = re.findall(r"[a-z0-9]+", sample.lower())
                    if any(t not in own and t not in LINK_WORDS for t in toks) and len("".join(toks)) >= 3 \
                            and re.search(r"(?<![a-z0-9])" + re.escape(" ".join(toks)) + r"(?![a-z0-9])", flat):
                        return True
            return False
        except Exception:
            return False

    def page_tools(self) -> list[str]:
        """LA/A2: the store's single-listing page tools: outside the search, message, status, cart and finish roles, declaring
        exactly one argument, a single text or number without fixed values."""
        taken = {self.role(r) for r in ("search", "message", "message_status", "cart", "cart_view", "order", "decline")} - {None}
        out = []
        for name, shape in self.shapes.items():
            if name in taken or not isinstance(shape, dict):
                continue
            props = shape.get("properties") if isinstance(shape.get("properties"), dict) else {}
            if len(props) != 1:
                continue      # (A2b: exactly one declared argument; a second search or browse tool declares more)
            spec = next(iter(props.values())) or {}
            if isinstance(spec, dict) and spec.get("type") in ("string", "integer", "number") and not spec.get("enum"):
                out.append(name)
        return out

    def unpaged(self) -> tuple[str, list[str]] | None:
        """A2: in layout A, the cart lines not yet opened with one of the page tools, for the tool missing the most (each
        line and tool tried once); None when every line was opened with every page tool."""
        seen = self.__dict__.get("page_seen") or {}
        tried = self.__dict__.get("paged_tried") or set()
        safe = {self.role("details"), self.role("view")} - {None}
        caps = self.__dict__.get("tool_caps") or {}
        best: tuple[str, list[str]] | None = None
        for tool in self.page_tools():
            if tool not in safe and not (caps.get(tool) or {}).get("id", 0):
                continue      # (A2b: only the reader's page tools, or one whose answers showed a listing id)
            ids = [str(i) for i in self.trolley.units if str(i) not in (seen.get(tool) or set()) and (tool, str(i)) not in tried]
            if ids and (best is None or len(ids) > len(best[1])):
                best = (tool, ids)
        return best

    def split_pages(self) -> bool:
        """LA: the world is layout A (races 11, 12, 16): two or more tools outside the search, message, status, cart and
        finish roles take one argument (the required one, else the only one declared) that is a single text or number
        without fixed values (a specs page and an offer page, each for one listing id). Its shoppers spend a message naming two or more subjects on a "one at a
        time" line, so no message pairs properties there. Read from the tools' shapes only."""
        try:
            split = len(self.page_tools()) >= 2
        except Exception:
            split = False
        if self.__dict__.get("split_said") != split:
            self.split_said = split
            self.log.append({"kind": "layout_split_pages", "split": split})
        return split

    def pair_room(self, ceiling: int, remaining: int) -> bool:
        """A message may ask two properties: only when the questions may use PAIR_FROM messages or more, never after a
        one-at-a-time or unanswered reply (no_pairs), and - until a reply settled both of a pair - only while another
        message is left after this one (an unproven pair never goes last)."""
        if self.__dict__.get("no_pairs") or ceiling < PAIR_FROM or remaining < 1 or self.split_pages():
            return False      # (LA: layout A, two single-id page tools: one property per message)
        if self.__dict__.get("pairs_heard"):
            return True
        return remaining >= 2 and self.msg_cap is not None

    def must_ask(self, prop: str) -> bool:
        """The property is a must-ask item of the request (the budget, a choice between two wishes, anything to check with
        the shopper first), by its words: such a question goes out alone."""
        words = {w for w in re.findall(r"[a-z]{4,}", str(prop).split("(")[0].lower())}
        if re.search(r"\b(?:budget\w*|spend\w*|pric(?:e[sdy]?|ing)|costs?|afford\w*)\b", " ".join(key_words(str(prop).split("(")[0]))):
            # (X1b: whole words of the property itself - not "Costumes", not its "(product)" tag; X2b: max_budget and
            # maxBudget split into words, "prickly" no price)
            return True
        for item in self.needs.reading.get("ask_first") or []:
            have = set(re.findall(r"[a-z]{4,}", str(item).lower()))
            if words and len(words & have) * 2 >= len(words):
                return True
        return False

    def other_unasked(self, first: str) -> bool:
        """K2: another product than the one `first` belongs to has had no answered question and revealed no need: a
        cross-product pair reaches it in this message, before a same-product pair spends the message on one product."""
        try:
            rows = self.needs.items()
            j0 = self.product_of(first)
            if j0 is None or len(rows) < 2:
                return False
            touched = self.touched_products()
            return any(j != j0 and j not in touched for j in range(len(rows)))
        except Exception:
            return False

    def touched_products(self) -> set[int]:
        """K2: the products with an answered question (by each part of its label, else by its asked text) or a revealed
        need."""
        touched: set[int] = set()
        for ln in self.questions_log:
            head = ln.split(":", 1)[0]
            if "[not answered]" in head:
                continue
            said = re.search(r'asked "(.*?)" -> ', ln)
            for part in head.strip("- ").split(";"):
                k = self.product_of(part)
                if k is None and said and ";" not in head:
                    k = self.product_of(said.group(1))
                if k is not None:
                    touched.add(k)
        for need in self.needs.uncovered:
            k = self.product_of(str(need).split(":", 1)[0])
            if k is not None:
                touched.add(k)
        return touched

    def cross_second(self, first: str, scripted: list[dict[str, Any]], fresh: Any) -> str:
        """For a request of several products: a property of ANOTHER product to ask with `first` - the planner's next pick
        for a product without a revealed need (a product whose need was told gets none), else that product's top
        RANKED open property - tagged with its
        product; '' when none is open."""
        try:
            items = self.needs.items()
            j0 = self.product_of(first)
            if j0 is None or len(items) < 2:
                return ""
            heard = {self.product_of(str(n).split(":", 1)[0]) for n in self.needs.uncovered if n not in self.needs.let_go}
            touched = self.touched_products()
            others = sorted((j for j in range(len(items)) if j != j0 and j not in heard), key=lambda j: (j in touched, j))
            # (E4: a product whose need was told takes no cross second; with none left X2's same-product second applies)

            def tagged(prop: str, j: int) -> str:
                """Return the property with product j's name in parentheses, unless it already carries a tag."""
                return prop if re.search(r"\(([^)]+)\)", prop) else f"{prop} ({items[j]['product']})"
            for j in others:
                for qitem in scripted:
                    prop = str(qitem.get("property") or "").strip()
                    if prop and self.plan_product(qitem) == j and fresh(tagged(prop, j)):
                        return tagged(prop, j)
                for name in self.rank_props(j):
                    if fresh(tagged(name, j)):
                        return tagged(name, j)
        except Exception as err:
            self.log.append({"kind": "cross_second_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return ""

    def cross_text(self, first: str, second: str) -> str:
        """The plain two-product question: 'Which A would you like for the X, and which B for the Y?'."""
        parts = []
        for prop in (first, second):
            tied = re.search(r"\(([^)]+)\)", prop)
            parts.append((prop.split("(")[0].strip(), tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")))
        return f"Which {parts[0][0]} would you like for the {parts[0][1]}, and which {parts[1][0]} for the {parts[1][1]}?"

    def twin_follows(self, first: str) -> bool:
        """X2b: `first` is story-hinted and has a story twin (EC4): after a quiet reply to it alone the twin is asked next,
        so it takes no fallback second."""
        try:
            k = self.product_of(first) if self.needs.items() else None
            return bool(self.hint_mapped(k, first) and self.twin_key(k, first))
        except Exception:
            return False

    def own_second(self, first: str, scripted: list[dict[str, Any]], fresh: Any) -> str:
        """X1: a second property of the SAME product as `first` to ask with it when the planner's own second does not fit:
        the planner's other picks for that product (each property and its second), else the product's RANKED open
        properties, the first that is fresh, no must-ask item and fits the pair (pair_fits); tagged with its product when
        the request has several; '' when none does."""
        try:
            items = self.needs.items()
            k = self.product_of(first) if items else None
            if items and k is None:
                return ""
            picks = []
            for qitem in scripted:
                if not items or self.plan_product(qitem) == k:
                    picks += [str(qitem.get("property") or "").strip(), str(qitem.get("second") or "").strip()]
            ranked = self.rank_props(k)
            picks += ranked
            for prop in picks:
                if not prop or re.search(r"none|^-$|n/a", prop.lower()):
                    continue
                if items and not re.search(r"\(([^)]+)\)", prop) and prop.lower() not in [r.lower() for r in ranked] \
                        and not self.prop_line(k, prop):
                    continue      # (X2b: an untagged pick must be one of this product's own fields)
                label = prop if not items or re.search(r"\(([^)]+)\)", prop) else f"{prop} ({items[k]['product']})"
                name = re.split(r"\s*\(", label)[0].strip()
                if fresh(label) and not self.must_ask(label) and self.pair_fits(first, label) and not alike_score(first, name)[1] \
                        and name.lower() not in [a.lower() for a in self.look_alikes(k, first)]:
                    return label      # (X1b: nor a name that looks like the first's or one of its look-alike fields)
        except Exception as err:
            self.log.append({"kind": "own_second_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return ""

    def pair_fits(self, first: str, second: str) -> bool:
        """Tell whether two properties can share one question message. For several products both must map to the same
        product. They must be different fields with no shared content words (three letters or more, link words
        aside) and no shared field values, and no other open key in the product's cached menu may be a twin of the
        second (twin_of). Errors give False.
        """
        try:
            several = bool(self.needs.items())
            k = self.product_of(first) if several else None
            if several and (k is None or self.product_of(second) != k):
                return False
            if strict_prop(first, second) or same_prop(first, second) or self.same_field(k, first, second):
                return False
            a = {w for w in prop_words(first) if len(w) >= 3 and w not in LINK_WORDS}
            b = {w for w in prop_words(second) if len(w) >= 3 and w not in LINK_WORDS}
            if (a & b) or (self.field_values(k, first) & self.field_values(k, second)):
                return False
            menu = (self.__dict__.get("menu_memo", {}).get(k) or (0, ""))[1]
            for line in menu.splitlines():
                key = line.strip("- ").split(" (in ")[0].strip()
                if key and not strict_prop(key, first) and not strict_prop(key, second) and self.twin_of(k, key, second) \
                        and self.plan_open(k, key):
                    return False
            return True
        except Exception:
            return False

    def queue_second(self, prop: str) -> None:
        """Add a property to pending_seconds unless it is already queued (case-insensitive)."""
        queue = self.__dict__.setdefault("pending_seconds", [])
        if prop and prop.lower() not in [p.lower() for p in queue]:
            queue.append(prop)

    def ask_again(self) -> None:
        """(BR) Asking that stopped for an empty plan starts again once the shopper changed the budget (new listings fit
        and get opened) while messages remain."""
        if self.asking or "ask_off_cap" not in self.__dict__ or self.needs.spend_cap == self.ask_off_cap:
            return
        if self.msgs_sent < self.ask_limit() and self.role("message"):
            self.asking, self.empty_plans, self.plan_wait = True, 1, None      # (BRb: the next empty plan stops it again)
            self.log.append({"kind": "ask_again", "budget": self.needs.spend_cap})
        self.ask_off_cap = self.needs.spend_cap

    def pose_planned(self, urge: bool = False) -> bool:
        """Plans and sends the next question as a scripted message; True when a question went or a read was
        queued. Stops without a message tool, at the ask limit, near the order reserve or under CLOSING + 40.
        Waits while few listings with fields were read, while products lack a home label (menu_search), or
        when nothing new was read since the last question; may queue pre_read or first_reads instead.
        The question is a pending second, else the PLANNER's pick from the request, needs, past questions,
        story hints and menus, filtered by fresh, then cover_top (one product) or quota_first (several).
        It is rewritten plainly when it misses the product or names another property; pair_room may add a
        second property. After the reply it books pair answers and one-at-a-time replies (no_pairs, re-asks,
        spent_asks), keeps unanswered seconds (open_pair, pending_seconds), queues a story twin after a silent
        reply and logs questions_log.
        """
        the_tool = self.role("message")
        ceiling = self.ask_limit()
        if not the_tool or self.msgs_sent >= ceiling:
            self.asking = False
            return False
        prop_list, _ = self.archive.prop_menu()
        sets = self.archive.rich_reads()
        own = self.own_shelf()
        if own:
            prop_list, sets = self.archive.prop_menu(own)[0], self.archive.rich_reads(own)
        pressing = bool(self.needs.reading.get("ask_first")) and not self.questions_log
        thin = sets < ASK_AFTER_SETS and not pressing and not urge
        if thin and (self.wire.tick < ASK_AFTER_STEPS or (self.wire.tick < ASK_AFTER_STEPS + ASK_GRACE_STEPS and self.moves_left() > 20)):
            return False
        if len(self.needs.items()) > 1 and not pressing and not self.questions_log and not self.msgs_sent \
                and self.wire.tick >= ASK_AFTER_STEPS and any(not h for h in self.home_shelves()) and self.menu_search():
            return True
        if len(self.needs.items()) > 1 and not pressing and not urge and not self.questions_log and self.wire.tick < ASK_AFTER_STEPS + 12 \
                and self.moves_left() > 25 and any(not h for h in self.home_shelves()):
            return False
        if self.moves_left() <= max(10, self.order_due()) or self.wire.spare() < CLOSING + 40:
            self.asking = False
            return False
        waiting = self.__dict__.get("plan_wait")
        if waiting is not None and sets <= waiting and not urge:
            return False
        last_sets = self.__dict__.get("asked_at_sets")
        if last_sets is not None and sets <= last_sets and not urge and not pressing \
                and self.moves_left() > self.order_due() + 2 * max(0, ceiling - self.msgs_sent) + 14 \
                and self.wire.tick - self.__dict__.get("asked_at_tick", 0) < 4 \
                and not self.needs.items():
            return False
        if self.pre_read():
            return True
        if not self.questions_log and not self.msgs_sent and not pressing and not urge and self.first_reads():
            return True
        remaining = ceiling - self.msgs_sent
        ask_text = ASK_PLAN.replace("QUESTION_COUNT", str(remaining)) + NEXT_ONE.replace("REMAINING", str(remaining))
        ask_text += "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
        ask_text += "\n\nREQUIREMENTS SO FAR:\n" + self.needs.text_body()
        if self.questions_log:
            ask_text += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.questions_log)
        if self.needs.reading.get("ask_first"):
            ask_text += "\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in self.needs.reading.get("ask_first") or [])
        ask_text += self.plan_notes(None)
        try:
            hinted = self.open_hints()
        except Exception:
            hinted = []
        if hinted:
            ask_text += ("\n\nSTORY HINTS (details of the request's story and the property each seems to concern; not requirements; "
                         "the hidden need usually sits on one of them):\n" + "\n".join(f"- {self.hint_mark(h)}" for h in hinted[:8]))
        shown = self.menu_trim(prop_list, sets)
        if self.needs.items():
            ask_text += "\n\nTHE PRODUCTS ASKED FOR: " + " | ".join(f"{n + 1}) {row['product']} x{row['quantity']}" for n, row in enumerate(self.needs.items()))
            read = self.archive.shelves_read()
            homes = self.home_shelves()
            mine = [s for s in read if s.strip().lower() in set().union(*homes)]
            order = mine
            menus = [(shelf, self.archive.prop_menu(shelf)[0]) for shelf in order[:4]]
            prop_list = "\n".join(f"CATEGORY {shelf}:\n{menu}" for shelf, menu in menus if menu) or prop_list
            shown = "\n".join(f"CATEGORY {shelf}:\n{self.menu_trim(menu, self.archive.rich_reads(shelf))}"
                              for shelf, menu in menus if menu) or shown
        ask_text += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n" + (shown or "(none seen yet)")
        asked_props = {p.strip().lower() for ln in self.questions_log for p in ln.split(":", 1)[0].strip("- ").split(";")}

        def fresh(prop: Any) -> bool:
            """Tell whether a property is still worth asking: not empty, junk, asked, spent, close to an asked one or
            settled, not a plain hint, still open in the plan and present on its product's labels.
            """
            prop = str(prop or "").strip()
            if not prop or prop.lower() in asked_props or junk_key(prop) or prop.lower() in self.__dict__.get("spent_asks", set()):
                return False
            if self.near_asked(prop) or self.settled_prop(prop):
                return False
            j = self.product_of(prop) if self.needs.items() else None
            return not self.plain_hint(j, prop) and self.plan_open(j, prop) and self.shelf_has(j, prop)
        queue = [p for p in self.__dict__.get("pending_seconds") or [] if fresh(p)]
        self.pending_seconds = queue
        pending = queue[0] if queue else ""
        if pending:
            tied = re.search(r"\(([^)]+)\)", pending)
            product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
            text = (self.__dict__.get("pending_text") or {}).pop(pending.lower(), "")
            answer = {"questions": [{"property": pending, "question": text or self.apart_plain(pending, f"Which {pending.split('(')[0].strip()} would you like for the {product}?")}]}
        else:
            try:
                ask_text += self.ranked_note(fresh)
            except Exception as err:
                self.log.append({"kind": "ranked_note_error", "error": f"{type(err).__name__}: {err}"[:300]})
            answer = self.wire.consult(ask_text, 2500, PLANNER)
            if answer is None and self.wire.misses > 0:
                self.plan_wait = sets
                return False
        scripted = [qitem for qitem in (answer or {}).get("questions") or [] if isinstance(qitem, dict) and str(qitem.get("question") or "").strip()]
        kept = []
        for qitem in scripted:
            second = str(qitem.get("second") or "").strip()
            second = "" if re.search(r"none|^-$|n/a", second.lower()) else second
            if fresh(qitem.get("property")):
                kept.append(dict(qitem, second=second if second and fresh(second) else ""))
                if second and self.plain_hint(self.product_of(second) if self.needs.items() else None, second):
                    first = str(qitem.get("property") or "").strip()
                    tied = re.search(r"\(([^)]+)\)", first)
                    product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
                    kept[-1]["question"] = f"Which {first.split('(')[0].strip()} would you like for the {product}?"
            elif second and fresh(second):
                tied = re.search(r"\(([^)]+)\)", second)
                product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
                kept.append({"property": second, "question": f"Which {second.split('(')[0].strip()} would you like for the {product}?"})
        scripted = kept
        if not self.needs.items() and not pending and not self.needs.reading.get("ask_first"):
            scripted = self.cover_top(scripted, fresh, remaining)
        if self.needs.items() and not pending:
            scripted = self.quota_first(scripted, fresh)
            if self.__dict__.pop("quota_wait", False):
                return True
        proposal = None
        split_second = ""
        for thing in scripted[:2]:
            if self.needs.items() and not re.search(r"\(([^)]+)\)", str(thing.get("property") or "")):
                bare = str(thing.get("property") or "").strip()
                j = self.plan_product(thing)
                if j is None and bare:
                    j = next((jj for jj in range(len(self.needs.items())) if any(strict_prop(bare, n) for n in self.rank_props(jj))), None)
                joined = str(self.needs.reading.get("product") or "")
                if j is not None and bare:
                    thing = dict(thing, property=f"{bare} ({self.needs.items()[j]['product']})")
                elif re.search(r"\band\b", joined) and joined.lower() in str(thing.get("question") or "").lower():
                    continue
            if self.needs.items():
                prop = str(thing.get("property") or "")
                tied = re.search(r"\(([^)]+)\)", prop)
                product = tied.group(1).strip() if tied else ""
                words = re.findall(r"[a-z]{3,}", product.lower())
                if product and words and not any(w.rstrip("s") in str(thing.get("question") or "").lower() for w in words):
                    thing = dict(thing, question=f"Which {prop.split('(')[0].strip()} would you like for the {product}?")
            if not pending and not (self.needs.reading.get("ask_first") and not self.questions_log) and self.names_other(str(thing.get("question") or ""), str(thing.get("property") or ""), prop_list,
                                                str(thing.get("second") or "")):
                prop = str(thing.get("property") or "")
                tied = re.search(r"\(([^)]+)\)", prop)
                product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
                thing = dict(thing, question=f"Which {prop.split('(')[0].strip()} would you like for the {product}?")
            if not pending:
                thing = dict(thing, question=self.apart_plain(str(thing.get("property") or ""), str(thing.get("question") or "")))
            proposal = self.message_call(str(thing.get("question")).strip(), f"ask_{self.msgs_sent + 1}_{self.wire.tick}")
            if proposal:
                label = (str(thing.get("property")), str(thing.get("question")).strip())
                second = str(thing.get("second") or "").strip()
                if self.needs.items() and self.pair_room(ceiling, remaining) and not pending and not self.must_ask(label[0]) \
                        and (not (second and not re.search(r"none|^-$|n/a", second.lower()) and self.pair_fits(label[0], second))
                             or self.other_unasked(label[0])):
                    cross = self.cross_second(label[0], scripted, fresh)
                    paired = self.message_call(self.cross_text(label[0], cross), f"ask_{self.msgs_sent + 1}_{self.wire.tick}") if cross else None
                    if paired:
                        proposal, label, second = paired, (label[0] + "; " + cross, self.cross_text(label[0], cross)), ""
                        self.log.append({"kind": "cross_pair", "text": label[1]})
                if ";" not in label[0] and self.pair_room(ceiling, remaining) and not pending and not self.must_ask(label[0]) \
                        and not label[1].startswith("Apart from ") \
                        and not (self.needs.reading.get("ask_first") and not self.questions_log) and not self.twin_follows(label[0]) \
                        and not (second and not re.search(r"none|^-$|n/a", second.lower()) and self.pair_fits(label[0], second)):
                    # (X2: no two-product pair was formed and the planner's own second does not fit: a second property of the
                    # same product joins, so the message names two; a first sent with its "Apart from" clause stays alone;
                    # X2b: nor the first message while something must be learned first, nor a first with a story twin)
                    second = self.own_second(label[0], scripted, fresh) or second
                if second and second.lower() != label[0].lower() and not re.search(r"none|^-$|n/a", second.lower()):
                    if self.pair_room(ceiling, remaining) and self.pair_fits(label[0], second) and not self.must_ask(label[0]) \
                            and not self.must_ask(second):
                        tied = re.search(r"\(([^)]+)\)", label[0])
                        product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
                        pair = (f"Which {label[0].split('(')[0].strip()} would you like for the {product}, "
                                f"and which {second.split('(')[0].strip()}?")
                        proposal = self.message_call(pair, f"ask_{self.msgs_sent + 1}_{self.wire.tick}")
                        if not proposal:
                            continue
                        label = (label[0] + "; " + second, pair)
                    else:
                        prop = label[0].split("(")[0].strip()
                        tied = re.search(r"\(([^)]+)\)", label[0])
                        product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
                        single = self.apart_plain(label[0], f"Which {prop} would you like for the {product}?")
                        proposal = self.message_call(single, f"ask_{self.msgs_sent + 1}_{self.wire.tick}")
                        label = (label[0], single)
                        if not proposal:
                            continue
                        split_second = second
                break
        if not proposal:
            if sets < ASK_AFTER_SETS and self.moves_left() > 12 and not urge:
                self.plan_wait = sets
                return False
            self.empty_plans = self.__dict__.get("empty_plans", 0) + 1
            if self.moves_left() > 12 and self.empty_plans <= 1:
                self.plan_wait = sets
                return False
            self.ask_off_cap = self.needs.spend_cap      # (BR: asking stops for an empty plan at this budget)
            self.asking = False
            return False
        self.asked_at_sets, self.asked_at_tick = sets, self.wire.tick
        self.chat.append({"role": "assistant", "content": f"I will ask the shopper about {label[0]}.", "tool_calls": [proposal]})
        found_before, freed_before = len(self.needs.uncovered), len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0)
        cap_before = str(self.needs.spend_cap)
        self.asking_label = label[0]
        try:
            responses = self.dispatch([proposal], scripted=True)
        finally:
            self.asking_label = ""
        self.post_replies(responses)
        asked_label = label[0]
        both_told = False
        if ";" in label[0] and self.needs.__dict__.get("meta_last") is None:
            told = (len(self.needs.uncovered) - found_before) + (len(self.needs.indifferent)
                                                                  + self.needs.__dict__.get("pinned_heard", 0) - freed_before)
            told += 1 if str(self.needs.spend_cap) != cap_before else 0
            both_told = told >= 2
            if told >= 1:
                self.pairs_heard = True
            self.log.append({"kind": "pair_reply", "settled": told, "proven": bool(self.__dict__.get("pairs_heard"))})
        if pending and label[0].split(";")[0].strip().lower() == pending.lower():
            self.pending_seconds = [p for p in self.__dict__.get("pending_seconds") or [] if p.lower() != pending.lower()]
        open_note = ""
        heard_meta = self.needs.__dict__.pop("meta_last", None)

        def plain(text: Any) -> str:
            """Reduce a text to its lowercase alphanumeric words, for comparing a question with a meta reply."""
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
        if heard_meta is not None and plain(label[1]) and plain(label[1]) in plain(heard_meta):
            self.no_pairs = True
            parts = [p.strip() for p in label[0].split(";") if p.strip()] or [label[0]]
            first = parts[0]
            tied = re.search(r"\(([^)]+)\)", first)
            product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
            again = self.__dict__.setdefault("reasked", set())
            plain_q = f"Which {first.split('(')[0].strip()} would you like for the {product}?"
            alone = len(parts) == 1
            alt = self.collide_alt(first) if alone else ""
            if alt:
                again.add(first.lower())
                self.__dict__.setdefault("spent_asks", set()).add(first.lower())
                self.pending_seconds = [alt] + [p for p in self.__dict__.get("pending_seconds") or [] if p.lower() != alt.lower()]
                open_note = f" [{first}: not answered, its story hint's alternative {alt} asked next]"
            elif first.lower() not in again and plain(plain_q) != plain(label[1]) and not (alone and self.holds_plain(label[1], plain_q)):
                again.add(first.lower())
                self.pending_seconds = [first] + [p for p in self.__dict__.get("pending_seconds") or [] if p.lower() != first.lower()]
                open_note = f" [{first}: not answered, asked again alone next]"
            else:
                redo = self.distinct_ask(first, label[1]) if first.lower() not in again else None
                if redo:
                    again.update({first.lower(), redo[0].lower()})
                    self.__dict__.setdefault("pending_text", {})[redo[0].lower()] = redo[1]
                    self.pending_seconds = [redo[0]] + [p for p in self.__dict__.get("pending_seconds") or [] if p.lower() != redo[0].lower()]
                    itself = re.split(r"\s*\(", redo[0])[0].strip().lower() == re.split(r"\s*\(", first)[0].strip().lower()
                    if not itself:
                        self.__dict__.setdefault("spent_asks", set()).add(first.lower())
                    open_note = f" [{first}: not answered, asked again " + ("by its values next]" if itself else f"as {redo[0]} next]")
                else:
                    self.__dict__.setdefault("spent_asks", set()).add(first.lower())
                    open_note = f" [{first}: not answered, not asked again]"
            pairs = self.__dict__.setdefault("open_pair", [])
            for second in parts[1:] + ([split_second] if split_second else []):
                if second.lower() not in [p.lower() for p in pairs]:
                    pairs.append(second)
            label = (" and ".join(parts) + " [not answered] [one at a time]", label[1])
        elif ";" in label[0] and len(self.needs.uncovered) > found_before:
            first, second = [p.strip() for p in label[0].split(";", 1)]
            found_text = " ".join(self.needs.uncovered[found_before:]).lower()

            def heard_of(prop: str) -> bool:
                # (X1b: the property's own words, its product's name words aside, against the need without its "(product)"
                # tag: "line weight (Fishing Lines)" is not heard in "color (Fishing Lines): red")
                tied = re.search(r"\(([^)]+)\)", prop)
                product = tied.group(1) if tied else str(self.needs.reading.get("product") or "")
                own = {w.rstrip("s") for w in key_words(product)}
                plain = re.findall(r"[a-z]{4,}", prop.split("(")[0].lower())
                words = [w for w in plain if w.rstrip("s") not in own] or plain      # (X2b: "glass" kept whole)
                several = bool(self.needs.items())
                for need in self.needs.uncovered[found_before:]:
                    head = str(need).split(":", 1)[0].lower()
                    if any(w in re.sub(r"\([^)]*\)", " ", head) for w in words) and \
                            (not several or self.product_of(head) in (None, self.product_of(prop))):
                        return True
                return False

            def reopen(prop: str) -> None:
                if self.pair_room(self.ask_limit(), self.ask_limit() - self.msgs_sent):
                    pairs = self.__dict__.setdefault("open_pair", [])
                    if prop.lower() not in [p.lower() for p in pairs]:
                        pairs.append(prop)
                else:
                    self.queue_second(prop)
            if not both_told and heard_of(second) and not heard_of(first):
                # (the reply told the pair's second property: the first is still unasked)
                reopen(first)
                open_note = f" [{first}: not answered, still open]"
                label = (second, label[1])
            else:
                if not both_told and not heard_of(second) and heard_of(first):
                    reopen(second)
                    open_note = f" [{second}: not answered, still open]"
                    label = (first, label[1])
                elif not both_told and not heard_of(second):
                    open_note = " [the need told matches neither property by name: both stay asked]"
                else:
                    label = (first, label[1])
        elif ";" in label[0] and not both_told and len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0) > freed_before:
            # (K3: a "don't mind" to a pair: a shopper takes up a wish of theirs before anything else when a message names
            # one, so neither property is likely a wish. Both stay asked - the label keeps the pair - and the second is
            # not queued again; it is not recorded as "don't mind" either, since the reply did not name it)
            second = label[0].split(";", 1)[1].strip()
            self.needs.indifferent = [p for p in self.needs.indifferent if p.strip().lower() not in (second.lower(), second.split("(")[0].strip().lower())]
            open_note = f" [{second}: asked with it; the reply named no wish, so most likely neither is one]"
        elif ";" in label[0] and len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0) == freed_before:
            self.no_pairs = True
            pairs = self.__dict__.setdefault("open_pair", [])
            for part in (p.strip() for p in label[0].split(";")):
                if part and part.lower() not in [p.lower() for p in pairs]:
                    pairs.append(part)
            label = (label[0].replace(";", " and") + " [not answered]", label[1])
        elif split_second:
            found_text = " ".join(self.needs.uncovered[found_before:]).lower()
            pairs = self.__dict__.setdefault("open_pair", [])
            if not any(w in found_text for w in re.findall(r"[a-z]{4,}", split_second.split("(")[0].lower())) \
                    and split_second.lower() not in [p.lower() for p in pairs]:
                pairs.append(split_second)
        if heard_meta is None and len(self.needs.uncovered) == found_before:
            self.__dict__.setdefault("quiet_asks", []).append(asked_label.split(";")[0].strip())
            twin = self.quiet_reask(asked_label) if ";" not in asked_label else ""
            if twin:
                self.pending_seconds = [twin] + [p for p in self.__dict__.get("pending_seconds") or [] if p.lower() != twin.lower()]
                open_note += f" [{twin}: its story twin, asked next]"
        for _, text_body in responses[:1]:
            self.questions_log.append(f"- {label[0]}: asked \"{label[1]}\" -> {text_body.split(chr(10) + '[status]')[0][:400]}{open_note}")
        self.drop_wanted(asked_label)
        self.asking = self.msgs_sent < ceiling
        return True

    def reply_back(self, text_body: str) -> bool:
        """Answer a question from the shopper by sending the given text as a scripted message and posting the replies;
        True when sent. Refuses without a message tool, at the ask limit, with four or fewer moves or spare under
        CLOSING, or while asking with one message left and plenty of moves and budget.
        """
        if not self.role("message") or self.msgs_sent >= self.ask_limit() or self.moves_left() <= 4 or self.wire.spare() < CLOSING \
                or (self.asking and self.msgs_sent >= self.ask_limit() - 1 and self.moves_left() > 10 and self.wire.spare() >= CLOSING + 40):
            return False
        proposal = self.message_call(text_body, f"reply_{self.wire.tick}")
        if not proposal:
            return False
        self.chat.append({"role": "assistant", "content": "I will answer the shopper's question.", "tool_calls": [proposal]})
        self.post_replies(self.dispatch([proposal], scripted=True))
        return True

    def post_replies(self, responses: list[tuple[str, str]]) -> None:
        """Append tool replies to the chat as tool messages under their call ids, with the status line added to the
        last one.
        """
        for idx, (cid, text_body) in enumerate(responses):
            if idx == len(responses) - 1:
                text_body += "\n" + self.status_line()
            self.chat.append({"role": "tool", "tool_call_id": cid, "content": text_body})

    def plain_args(self, call_name: str, text_body: str, kind: str = "") -> dict[str, Any]:
        """Fill a tool's required arguments from its schema with a given text: an enum takes the first value matching
        the decline or order words for that kind (else its first value), a string the text cut to maxLength, a
        boolean True and a number the larger of 1 and its minimum. A tool with one free-text property also gets the
        text there. Returns the argument dict.
        """
        jschema = self.shapes.get(call_name) or {}
        prop_specs = jschema.get("properties") or {}
        call_args: dict[str, Any] = {}
        words = DECLINE_WORDS if kind == "decline" else ORDER_WORDS if kind == "order" else ""
        for key in jschema.get("required") or []:
            pspec = prop_specs.get(key) or {}
            if pspec.get("enum"):
                fit = [val for val in pspec["enum"] if words and isinstance(val, str) and re.search(words, val.lower())]
                call_args[key] = fit[0] if fit else pspec["enum"][0]
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
        """Replace free-text arguments of a decline call with the SORRY text (cut to each field's maxLength) when the
        request forbids naming listings in a decline or the text names a known listing id; enum fields are kept.
        Unchanged when the reading sets decline_no_names to False.
        """
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

    def code_reading(self) -> dict[str, Any]:
        """Read the request by code alone, without a model, into a needs reading marked code_reading: spend cap and
        currency, in-stock wording, a cheapest priority, a message limit and a step limit (kept only from
        FIRST_ORDER_BY + 3), and the code-guessed tool roles. Two or more products found by code_items give items
        with quantities and requirements; otherwise one product (a matched label or code_label), a count from
        code_count and requirements from code_needs.
        """
        text_body = self.request
        low = text_body.lower()
        reading: dict[str, Any] = {"code_reading": True, "requirements": [], "items": [], "quantity": 1, "rules": [],
                                   "soft_wants": [], "ask_first": [], "hints": [], "references": [], "priority_detail": ""}
        reading["price_cap"], reading["currency"] = self.code_budget(text_body)
        reading["in_stock_only"] = bool(re.search(r"\bin[- ]stock\b|\bon hand\b|\bon the shelf\b|\bsold[- ]out\b|\bback[- ]?order|"
                                                  r"\bout of stock\b|\bavailable now\b", low))
        reading["priority"] = "cheapest" if re.search(r"\bcheapest\b|\blowest[- ](?:price|priced|cost)\b|\bleast expensive\b|"
                                                      r"\blowest\b[^.]{0,20}\bprice\b|\bsmallest (?:total|price)\b", low) else "none"
        said = self.code_limit(low, r"messages?|questions?|notes?|repl(?:y|ies)|texts?")
        if said is not None:
            reading["message_budget"] = said
        moves = self.code_limit(low, r"steps?|moves?|actions?|(?:tool )?calls?|turns?")
        if moves is not None and moves >= FIRST_ORDER_BY + 3:
            reading["step_budget"] = moves
        roles = self.code_roles()
        reading["roles"] = roles
        items = self.code_items(roles.get("search"))
        if len(items) >= 2:
            reading["items"] = [{key: row[key] for key in ("product", "quantity", "requirements")} for row in items]
            reading["product"] = " and ".join(row["product"] for row in items)
            reading["requirements"] = [req for row in items for req in row["requirements"]]
        else:
            reading["product"] = items[0]["label"] if items else self.code_label(roles.get("search"), kind_words(text_body))
            reading["quantity"] = items[0]["quantity"] if items else self.code_count(low)
            reading["requirements"] = [f"{prop}: {value}" for prop, value in self.code_needs(low)]
        return reading

    @staticmethod
    def code_budget(text_body: str) -> tuple[Any, Any]:
        """Finds the spend cap in a request text: (amount, unit) or (None, None).
        Each number is scored: a currency sign or money word before it +3,
        a limit word before or a limit phrase after +2, order or total wording +1.
        Numbers counting messages, steps, units or items are skipped.
        The best score of 2+ wins, the larger figure on ties.
        The unit comes from the next two words.
        """
        best: tuple[int, float] = (0, 0.0)
        found: tuple[Any, Any] = (None, None)
        for sentence in re.split(r"(?<=[.!?;])\s+", str(text_body).lower()):
            whole = 1 if re.search(r"\b(?:order|total|whole|lot|sum|bill|altogether|overall|everything)\b", sentence) else 0
            for m in re.finditer(r"([$\u20ac\u00a3]\s*)?(\d[\d,]*(?:\.\d+)?)", sentence):
                after = sentence[m.end():]
                if re.match(r"\s*(?:x\b|\u00d7|%|:\s*\d|-\s*\d|/\s*\d|(?:[a-z-]+\s+){0,2}?(?:messages?|questions?|notes?|repl\w*|texts?|"
                            r"steps?|moves?|actions?|(?:tool )?calls?|turns?|units?|items?|pieces?|packs?|pairs?|lots?|sets?|of\b|times?)\b)", after):
                    continue
                if not m.group(1) and re.search(r"[\d.,:/-]\s*$|\bx\s*$", sentence[:m.start()]):
                    continue
                before = sentence[max(0, m.start() - 60): m.start()]
                score = whole
                if m.group(1) or re.search(r"\b(?:spend\w*|budget\w*|afford\w*|pay\w*|price[ds]?|costs?|cash|money)\b(?:\W+[\w']+){0,4}\W*$", before):
                    score += 3
                if re.search(r"\b(?:under|below|at most|no more than|not more than|up to|within|max\w*|less than|cheaper than|"
                             r"not (?:more|over|above)|limit\w*|reach\w*|go to|to|at|cap\w*|past|exceed\w*|beyond|go over)\W+(?:[\w']+\W+)?$", before):
                    score += 2
                if re.match(r"\s*(?:[a-z.()]+\s*){0,3}?(?:or (?:lower|less|under|below|cheaper)|at (?:the (?:very )?)?most|max\w*|tops\b|"
                            r",?\s*(?:and )?not (?:one|a) \w+ (?:over|more|above))", after):
                    score += 2
                figure = float(m.group(2).replace(",", ""))
                if score >= 2 and (score, figure) > best:
                    best = (score, figure)
                    unit = re.match(r"\s*((?:[a-z][a-z.]*\s*){0,2})", after)
                    words = [w.strip(".") for w in (unit.group(1).split() if unit else []) if w.strip(".") not in LINK_WORDS
                             and not re.fullmatch(r"or|and|not|tops|max\w*|only|just|is|was", w.strip("."))]
                    sign = (m.group(1) or "").strip()
                    found = (int(figure) if figure.is_integer() else figure, " ".join(words) or sign or None)
        return found

    @staticmethod
    def code_limit(low: str, nouns: str) -> int | None:
        """Return a limit stated in lowercase text for the given nouns, written as digits or number words up to twenty,
        when limit wording precedes it (at most, up to, within, you have and similar) or follows it (in total, max,
        allowed). None when the text states none.
        """
        figures = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                   "twelve": 12, "fifteen": 15, "twenty": 20}
        for m in re.finditer(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty)\s+(?:[a-z-]+\s+){0,2}?(?:"
                             + nouns + r")\b", low):
            before, after = low[max(0, m.start() - 60): m.start()], low[m.end(): m.end() + 40]
            if re.search(r"\b(?:at most|no more than|not more than|up to|only|max\w*|use|using|take|send|allow\w*|receive|get|total of|"
                         r"limit\w*|inside|within|under|fewer than|less than|you've|you have|have|keep\w*|stick to|go past|past|"
                         r"exceed\w*|beyond)\W+(?:[\w']+\W+){0,3}$", before) \
                    or re.match(r"\W*(?:[\w']+\W+){0,3}?(?:in total|total|no more|at most|max\w*|tops|or fewer|or less|allowed|limit)\b", after):
                return int(m.group(1)) if m.group(1).isdigit() else figures[m.group(1)]
        return None

    @staticmethod
    def code_count(low: str) -> int:
        """Read a unit count such as three units or two copies from lowercase request text; 1 when absent or outside 1
        to 99.
        """
        figures = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        said = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:units?|of (?:them|those|these)|copies)\b", low)
        if not said:
            return 1
        n = int(said.group(1)) if said.group(1).isdigit() else figures[said.group(1)]
        return n if 1 <= n <= 99 else 1

    @staticmethod
    def code_needs(seg: str) -> list[tuple[str, str]]:
        """Extract (property, value) pairs from lowercase request text written as the property is, must be or should be
        a value, after a clause start or after where or whose. Values end at punctuation or a linking word. A pair
        is dropped when the value refers to another listing, either side runs past four words, the property holds a
        pronoun or price, budget, order or stock word, or the value starts with an article, pronoun, preposition or
        filler word.
        """
        out: list[tuple[str, str]] = []
        verbs = r"(?:is|are|has to be|have to be|must be|should be|needs to be|need to be|can be)"
        stop = r"(?=\s*(?:[,.;:!?()]|$)|\s+(?:so|since|as|because|for|to|which|that|nothing|otherwise|but|when|while|if|then|though|please|and (?:the|its|their|a|an|one|two|three|\d))\b)"
        pattern = (r"(?:\b(?:where|whose)\s+(?:the\s+|its\s+|their\s+)?|(?:\b(?:and|with|but)\s+|[,;:]\s*|^|[.!?]\s+)(?:the|its|their)\s+)"
                   r"([a-z][a-z'&/ -]{1,32}?)\s+" + verbs + r"\s+(?:either\s+|just\s+)?([^,.;:!?()]{1,60}?)" + stop)
        for m in re.finditer(pattern, seg):
            prop, value = m.group(1).strip(), m.group(2).strip()
            if ref_text(value) or ref_text(seg[m.end(): m.end() + 60]) and re.search(r"\b(?:page|listing|shown|sold by)\b", seg[m.end(): m.end() + 60]):
                continue
            if not value or len(value.split()) > 4 or len(prop.split()) > 4:
                continue
            if re.search(r"\b(?:i|you|we|it|he|she|they|my|your|our|there|this|that|which|what|who|me|us|them|one|ones|price|budget|cost|total|order|stock)\b", prop):
                continue
            if re.match(r"(?:a|an|the|my|your|our|his|her|their|me|us|for|to|in|on|at|of|see|go|get|be|have|only|too|very|so|not|no|what|how|it|all|"
                        r"mostly|always|never|just|really|still|also|from|about|like|one|ones)\b", value):
                continue
            out.append((prop, value))
        return out

    def code_items(self, tool: str | None) -> list[dict[str, Any]]:
        """Find several products in the request by matching labels from the search tool's largest enum (four or more
        values) against the request words, plural s ignored, where a count (digits or a number word up to ten,
        optional x) shortly precedes the match. Each counted label (first match, quantity 1 to 99) becomes {product,
        label, quantity, requirements}, with requirements from code_needs over the text up to the next product (the
        last product up to its sentence end) tagged with the product. Returned in request order.
        """
        props = (self.shapes.get(tool or "") or {}).get("properties")
        lists = [spec["enum"] for spec in (props.values() if isinstance(props, dict) else []) if isinstance(spec, dict)
                 and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 4]
        labels = [str(label) for label in max(lists, key=len)] if lists else []
        low = self.request.lower()
        toks = [(m.group(0).rstrip("s"), m.start(), m.end()) for m in re.finditer(r"[a-z]{3,}", low) if m.group(0) not in LINK_WORDS]
        words = [t[0] for t in toks]
        figures = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        counted: dict[str, tuple[int, int, int]] = {}
        for label in labels:
            own = [w.rstrip("s") for w in re.findall(r"[a-z]{3,}", label.lower()) if w not in LINK_WORDS]
            if not own:
                continue
            for i in range(len(words) - len(own) + 1):
                if words[i: i + len(own)] != own:
                    continue
                start, end = toks[i][1], toks[i + len(own) - 1][2]
                if len(re.findall(r"[a-z0-9]+", low[start:end])) > 2 * len(own):
                    continue
                count = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s+|\u00d7\s*)?(?:(?:of|[a-z'&-]{3,})\s+){0,4}$",
                                  low[max(0, start - 56): start])
                if count and label not in counted:
                    n = int(count.group(1)) if count.group(1).isdigit() else figures[count.group(1)]
                    if 1 <= n <= 99:
                        counted[label] = (start, end, n)
                    break
        ordered = sorted(counted.items(), key=lambda kv: kv[1][0])
        out = []
        for j, (label, (start, end, n)) in enumerate(ordered):
            stop = ordered[j + 1][1][0] if j + 1 < len(ordered) else len(low)
            seg = low[end: stop]
            seg = re.split(r"(?<=[.!?])\s", seg)[0] if j + 1 == len(ordered) else seg
            name = label.lower()
            out.append({"product": name, "label": label, "quantity": n,
                        "requirements": [f"{prop} ({name}): {value}" for prop, value in self.code_needs(seg)]})
        return out

    def code_roles(self) -> dict[str, str]:
        """Guesses role tools from names, descriptions, schemas; no model.
        Order/decline: required enum with both word kinds -> both; else
        names with such words; else arg-free tools saying final near them.
        Cart/basket/bag/trolley: cart if required/array args, else cart view.
        Rest by cue: search (find/look/query; not shopper messaging), message
        (send/reply/chat + shopper word/messaging), details (details/specs/attributes),
        view (offer/price/stock). Message/details/view: exactly 1 required
        free-text arg; search: free-text or enum. Returns {role: tool name}.
        """
        said: dict[str, str] = {}
        for tdef in self.toolset:
            fn = tdef["function"]
            said[fn["name"]] = re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ").lower() + " | " \
                + str(fn.get("description") or "").lower()

        def props(name: str) -> dict[str, Any]:
            """Return a tool's schema properties, or an empty dict when absent."""
            found = (self.shapes.get(name) or {}).get("properties")
            return found if isinstance(found, dict) else {}

        def needed(name: str) -> list[str]:
            """List a tool's required argument names that also appear in its properties."""
            found = (self.shapes.get(name) or {}).get("required")
            return [key for key in found if key in props(name)] if isinstance(found, list) else []

        def texts(name: str) -> list[str]:
            """List a tool's free-text arguments: string type without enum."""
            return [key for key, spec in props(name).items() if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")]
        roles: dict[str, str] = {}
        for name in said:
            for key in needed(name):
                vals = [str(v).lower() for v in (props(name)[key].get("enum") or [])] if isinstance(props(name)[key], dict) else []
                if any(re.search(ORDER_WORDS, v) for v in vals) and any(re.search(DECLINE_WORDS, v) for v in vals):
                    roles.setdefault("order", name)
                    roles.setdefault("decline", name)
        for name in said:
            head = said[name].split(" | ")[0]
            if name in roles.values():
                continue
            if "decline" not in roles and re.search(DECLINE_WORDS, head):
                roles["decline"] = name
            elif "order" not in roles and re.search(ORDER_WORDS, head):
                roles["order"] = name
        for name in said:
            if name in roles.values() or needed(name):
                continue
            if "decline" not in roles and re.search(r"\bfinal\b", said[name]) and re.search(DECLINE_WORDS, said[name]):
                roles["decline"] = name
            elif "order" not in roles and re.search(r"\bfinal\b", said[name]) and re.search(ORDER_WORDS, said[name]):
                roles["order"] = name
        basket = r"\b(?:cart|basket|bag|trolley)\b"
        for name in said:
            if name in roles.values() or not re.search(basket, said[name]):
                continue
            if needed(name) or any(isinstance(spec, dict) and spec.get("type") == "array" for spec in props(name).values()):
                roles.setdefault("cart", name)
            elif not needed(name):
                roles.setdefault("cart_view", name)
        person = r"\b(?:shoppers?|customers?|buyers?|clients?|users?)\b"
        speak = r"\b(?:send\w*|repl\w*|notes?|tell\w*|messag\w*|chat\w*|writ\w*|talk\w*)\b"
        cues = [("search", r"\b(?:search\w*|find\w*|look\w*|locat\w*|hunt\w*|brows\w*|quer\w*)\b", False),
                ("message", speak, True),
                ("details", r"\b(?:details?|records?|spec\w*|propert\w*|attribut\w*|facts?|full)\b", True),
                ("view", r"\b(?:offer|price|stock|availab\w*)\b", True)]
        for role, cue, one_text in cues:
            for name in said:
                if role in roles or name in roles.values() or not re.search(cue, said[name]):
                    continue
                if role == "search" and re.search(person, said[name]) and re.search(speak, said[name]):
                    continue
                if role == "message" and not (re.search(person, said[name]) or re.search(r"\b(?:messag\w*|chat\w*)\b", said[name])):
                    continue
                if one_text and not (len(needed(name)) == 1 and needed(name)[0] in texts(name)):
                    continue
                if not one_text and not (texts(name) or any(isinstance(spec, dict) and spec.get("enum") for spec in props(name).values())):
                    continue
                roles[role] = name
        return roles

    def code_label(self, tool: str | None, want: set[str]) -> str:
        """Pick the label from the search tool's largest enum (four or more values) that shares the most kind words
        with the wanted words, ties going to the label with the larger share of its own words matched. Empty when no
        label overlaps.
        """
        props = (self.shapes.get(tool or "") or {}).get("properties")
        lists = [spec["enum"] for spec in (props.values() if isinstance(props, dict) else []) if isinstance(spec, dict)
                 and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 4]
        best, top = "", (0, 0.0)
        for label in max(lists, key=len) if lists else []:
            words = kind_words(label)
            hit = len(words & want)
            if hit and (hit, hit / max(1, len(words))) > top:
                best, top = str(label), (hit, hit / max(1, len(words)))
        return best

    def code_values(self, k: int | None) -> list[tuple[str, str]]:
        """Return (property, value) pairs of the remaining requirements of product k (all of them for one product) that
        code can match: the value without its stated or revealed tag, positive or short (five words or fewer, no
        exclusion words), and not a reference.
        """
        rows = self.needs.items()
        out = []
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            if rows and self.product_of(head) not in (k, None):
                continue
            plain = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip()
            short = len(re.findall(r"[a-z0-9]+", plain.lower())) <= 5 and not re.search(EXCLUDE_WORDS, plain.lower())
            if (positive_value(value, head) or short) and plain and not ref_text(plain):
                out.append((re.split(r"\s*\(", head)[0].strip(), plain))
        return out

    def code_queries(self, k: int | None) -> list[dict[str, Any]]:
        """Up to 3 code searches for product k: first firm stated value + kind
        word; category label from the tool's largest enum; shorter: longest
        value word + kind word. No firm values: product name or label words.
        Text, category, mode args come from the schema; mode value: as the
        tool description pairs it, else first (text) or last (category).
        A price-ceiling arg gets the per-unit spend cap. Queries cut to
        the text arg's maxLength; missing required args filled. ArgCheck
        validates calls; faults retry without price bound; duplicates dropped.
        """
        tool = self.role("search")
        jschema = self.shapes.get(tool or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        needed = [key for key in jschema.get("required") or [] if key in props]
        rows = self.needs.items()
        product = str(rows[k]["product"] if rows and k is not None and k < len(rows) else self.needs.reading.get("product") or "")
        label = self.code_label(tool, kind_words(product) or kind_words(self.request))
        told = next((str(tdef["function"].get("description") or "") for tdef in self.toolset if tdef["function"]["name"] == tool), "")
        parts = [part.lower() for part in re.split(r"[.;]", told)]
        tok = self.token_key()
        texts = [key for key, spec in props.items() if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum") and key != tok]
        text_key = next((key for key in texts if re.search(r"^q$|query|text|keyword|needle|term|word|phrase|string|search", key.lower())),
                        texts[0] if texts else None)
        enums = {key: spec["enum"] for key, spec in props.items() if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and spec["enum"]}
        shelf_key = max((key for key in enums if len(enums[key]) >= 4), key=lambda key: len(enums[key]), default=None)
        modes = [key for key in enums if key != shelf_key and len(enums[key]) <= 4]
        mode_key = next((key for key in modes if key in needed), modes[0] if modes else None)

        def mode_for(arg: str, last: bool) -> Any:
            """Chooses a value for the search tool's short mode enum when calling with argument arg: the first enum value
            that one sentence of the tool description names together with that argument. Falls back to the last enum value
            when last is set (shelf calls) and to the first one otherwise (text calls).
            """
            vals = enums.get(mode_key or "", [])
            for val in vals:
                if any(re.search(r"\b" + re.escape(str(val).lower()) + r"\b", part) and re.search(r"\b" + re.escape(arg.lower()) + r"\b", part)
                       for part in parts):
                    return val
            return vals[-1] if last else vals[0]

        def price_word(key: str) -> bool:
            """Tells whether a numeric search argument looks like a price ceiling: its name suggests ceil, max, price, cap,
            budget, upper or most, or a description sentence naming the key also talks about price or cost.
            """
            return bool(re.search(r"ceil|max|price|cap$|budget|upper|most", key.lower())) or any(
                key.lower() in part and re.search(r"\bpric|\bcost", part) for part in parts)
        cap_key = next((key for key, spec in props.items() if isinstance(spec, dict) and spec.get("type") in ("integer", "number")
                        and not spec.get("enum") and price_word(key) and not re.search(r"page|first|offset|skip|count|size|num", key.lower())), None)
        cap = self.needs.spend_cap
        qty = (self.needs.counts()[k] if k is not None and k < len(self.needs.counts()) else self.needs.quantity()) or 1
        bound = {cap_key: (int(cap // qty) if props[cap_key].get("type") == "integer" else round(cap / qty, 2))} \
            if cap_key and is_number(cap) and cap > 0 else {}
        cap_len = props.get(text_key, {}).get("maxLength") if text_key and isinstance(props.get(text_key), dict) else None
        names = [w for w in re.findall(r"[a-z0-9]+", product.lower()) if len(w) > 2 and w not in LINK_WORDS]
        kind = names[-1] if names else ""
        firm = [value for prop, value in self.code_values(k) if not re.search(r"\bor\b|/|,|\b(?:" + VALUE_LINKS + r")\b", value.lower())
                and not self.unsearchable(prop, value)]
        queries: list[str] = []
        if firm:
            vw = [w for w in re.findall(r"[a-z0-9]+", firm[0].lower()) if w not in LINK_WORDS]
            if vw:
                queries.append(" ".join(dict.fromkeys(vw + [kind])))
                if len(vw) >= 2:
                    queries.append(" ".join(dict.fromkeys([max(vw, key=len), kind])))
        if not queries:
            queries.append(" ".join(dict.fromkeys(names[:1] + names[-1:])) if names else " ".join(sorted(kind_words(label))[:2]))
        if is_number(cap_len):
            fitted = []
            for query in queries:
                while len(query) > cap_len and " " in query:
                    query = query.rsplit(" ", 1)[0]
                fitted.append(query[: int(cap_len)])
            queries = fitted
        calls: list[dict[str, Any]] = []
        if text_key and queries[0]:
            calls.append({text_key: queries[0], **({mode_key: mode_for(text_key, False)} if mode_key else {}), **bound})
        if label and shelf_key:
            calls.append({shelf_key: label, **({mode_key: mode_for(shelf_key, True)} if mode_key else {}), **bound})
        if text_key and len(queries) > 1:
            calls.append({text_key: queries[1], **({mode_key: mode_for(text_key, False)} if mode_key else {}), **bound})
        out: list[dict[str, Any]] = []
        for call_args in calls:
            for key in needed:
                spec = props[key] if isinstance(props[key], dict) else {}
                if key in call_args:
                    continue
                if spec.get("enum"):
                    call_args[key] = spec["enum"][0]
                elif spec.get("type") in ("integer", "number"):
                    call_args[key] = spec.get("default", spec.get("minimum", 1))
                elif spec.get("type") == "string":
                    call_args[key] = queries[0] or label
            fixed, faults = ArgCheck.execute(call_args, jschema)
            if faults and bound:
                fixed, faults = ArgCheck.execute({key: val for key, val in call_args.items() if key not in bound}, jschema)
            if not faults and fixed not in out:
                out.append(fixed)
        return out

    def name_lift(self) -> None:
        """Product-name words store fields use as values become written
        requirements. Per unlifted product (or single item): field maps of rows
        on its home shelves (own shelf if single); at 2+ maps, more than last
        try, asks name_values for <=2 (key, phrase) pairs open requirements lack.
        Each pair joins needs.written, the reading's and item's requirement lists
        and now_reqs; bumps needs.version; logs name_lift. Product done once
        something lifted or 8 rows seen; errors logged, never raised.
        """
        try:
            rows = self.needs.items()
            products = list(range(len(rows))) if rows else [None]
            done = self.__dict__.setdefault("name_lifted", set())
            tried = self.__dict__.setdefault("name_tried", {})
            if len(done) >= len(products):
                return
            homes = self.home_shelves()
            recs = self.ch_rows()
            for k in products:
                tag = k if k is not None else -1
                if tag in done:
                    continue
                if rows:
                    home = set(homes[k]) if k < len(homes) else set()
                else:
                    own = self.own_shelf()
                    home = {own} if own else (set(homes[0]) if homes else set())
                if not home:
                    continue
                mine = [r["fields"] for r in recs.values() if r["fields"] and r["shelf"] in home]
                if len(mine) < 2 or len(mine) <= tried.get(tag, 0):
                    continue
                tried[tag] = len(mine)
                product = str(rows[k]["product"] if rows else self.needs.reading.get("product") or "")
                held = " | ".join(re.sub(r"\([^)]*\)", " ", str(r)) for r in self.needs.props_left())
                lifted = name_values(product, home, mine, held)[:2]
                if lifted or len(mine) >= 8:
                    done.add(tag)
                for key, phrase in lifted:
                    req = f"{key} ({product}): {phrase}" if rows else f"{key}: {phrase}"
                    self.needs.written.append(req)
                    self.needs.reading["requirements"] = list(self.needs.written)
                    items = self.needs.reading.get("items")
                    if rows and isinstance(items, list) and k < len(items) and isinstance(items[k], dict) \
                            and isinstance(items[k].get("requirements"), list):
                        items[k]["requirements"].append(req)
                    if self.needs.now_reqs is not None:
                        self.needs.now_reqs.append(req)
                    self.needs.version += 1
                    self.log.append({"kind": "name_lift", "product": k, "requirement": req})
        except Exception as err:
            self.log.append({"kind": "name_lift_error", "error": f"{type(err).__name__}: {err}"[:300]})

    def unsearchable(self, prop: Any, value: Any) -> bool:
        """Says whether a requirement value cannot be reached by search: true when the value holds a model code and
        either the property is about fit or compatibility (fit, compatible, works, suits) or one of its codes is in
        the dead_codes set learned from earlier searches.
        """
        codes = MODEL_CODE.findall(str(value or ""))
        if not codes:
            return False
        if re.search(r"\bfit|compatib|\bworks?\b|\bsuits?\b", str(prop or "").lower()):
            return True
        dead = self.__dict__.get("dead_codes") or set()
        return any(" ".join(code_tokens(c)) in dead for c in codes)

    def learn_dead(self, call_args: dict[str, Any]) -> None:
        """Remembers the model codes of a search query so later searches do not repeat them. It takes the free-text
        values of the call, finds model codes in them and keeps each code whose tokens make up all but at most two of
        the query's tokens. New codes are added to the dead_codes set on self and logged as dead_codes with the query;
        any error is swallowed.
        """
        try:
            text = " ".join(self.free_text(call_args))
            toks = code_tokens(text)
            got = set()
            for c in MODEL_CODE.findall(text):
                mine = code_tokens(c)
                if mine and len(toks) - len(mine) <= 2:
                    got.add(" ".join(mine))
            if got:
                dead = self.__dict__.setdefault("dead_codes", set())
                fresh = got - dead
                dead |= got
                if fresh:
                    self.log.append({"kind": "dead_codes", "codes": sorted(fresh)[:6], "query": text[:80]})
        except Exception:
            pass

    def dead_code_guard(self, call_name: str, call_args: dict[str, Any]) -> str:
        """Stops a search call that repeats a model code already known to find nothing. It only acts on the search role
        when dead_codes is non-empty: if the call's free-text tokens contain a dead code's tokens in sequence, it logs
        dead_code_held and returns a not-run note telling the model to search the item's kind with a stated value and
        to read each row's text for model-family fit rules instead. Returns an empty string when the call may run or
        on any error.
        """
        try:
            dead = self.__dict__.get("dead_codes") or set()
            if call_name != self.role("search") or not dead:
                return ""
            toks = code_tokens(" ".join(self.free_text(call_args)))
            for code in sorted(dead):
                want = code.split()
                if want and any(toks[i: i + len(want)] == want for i in range(len(toks) - len(want) + 1)):
                    self.log.append({"kind": "dead_code_held", "code": code})
                    return (f"Not run (no step used): a search holding '{code}' found nothing before - this store's search does not "
                            "reach model codes or model families. Search the item's kind with a stated value in the item's own "
                            "words, then read each row's text: a row saying it fits every model of a family covers each model "
                            "of that family unless it names that model as an exception; the code checks those texts for you.")
            return ""
        except Exception:
            return ""

    def fit_targets(self) -> list[tuple[int | None, str]]:
        """Lists the products that must fit a specific model code. For each open property requirement whose value (with
        the stated or revealed tags removed) does not refer to another listing, it takes the first code-like model
        code without a wildcard run, maps the requirement to its product index when the request has several items, and
        keeps one (product index, code) pair per product. Requirements whose product cannot be resolved are skipped.
        """
        rows = self.needs.items()
        out: list[tuple[int | None, str]] = []
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            plain = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", value)
            if ref_text(plain):
                continue
            codes = [c for c in MODEL_CODE.findall(plain) if code_like(c) and not re.search(r"[xX*#?]{2}", c)]
            if not codes:
                continue
            k = self.product_of(head) if rows else None
            if rows and k is None:
                continue
            if not any(kk == k for kk, _ in out):
                out.append((k, codes[0]))
        return out

    def fit_rows(self, k: int | None, model: str, home: set[str]) -> list[tuple[str, bool]]:
        """Returns the rows on the given home shelves whose own text carries a model-family rule that covers the target
        model, cheapest first, as (id, shows) pairs. Rows that are dead, refused, out of stock or over the spend cap
        at the product's quantity are left out. The shows flag is true when the row's words contain every plain, code-
        free requirement value of that product.
        """
        rows = self.ch_rows()
        counts = self.needs.counts()
        qty = (counts[k] if k is not None and k < len(counts) else self.needs.quantity()) or 1
        cap = self.needs.spend_cap
        dead = set(self.dead_lines()) | set(self.refused_lines)
        values = [kind_words(plain) for _, plain in self.code_values(k) if not MODEL_CODE.search(plain)]
        out = []
        for ident, row in rows.items():
            if row["shelf"] not in home or ident in dead or row["stock"] is False:
                continue
            price = row["price"]
            if is_number(cap) and is_number(price) and price * qty > cap:
                continue
            if family_fit(row.get("text") or "", model) is not True:
                continue
            shows = all(v <= row["words"] for v in values if v)
            out.append((price if is_number(price) else float("inf"), ident, shows))
        return [(i, shows) for _, i, shows in sorted(out)]

    def fit_walk(self) -> bool:
        """Reads every page of model-code searches to judge rows' fit rules.
        Runs when a requirement names a model code and rows carry family rules.
        Idle when the wire is done, an order was tried, or time is short.
        Per product, fit_state tracks queries, pages, done and home shelf.
        walk_call sends the next query or page within FIT_PAGES and FIT_TOTAL,
        keeping moves for questions and the order.
        Without a home shelf it adopts one with 3 of 5 walked rows.
        Posts a fit note.
        When done, vl_launch opens up to FIT_OPENS fitting rows.
        True on a move.
        """
        try:
            if self.wire.done or self.wire.broke or self.wire.mute or self.orders_tried or self.refusal_n or not self.role("search"):
                return False
            if self.wire.spare() < CLOSING + 60:
                return False
            targets = self.fit_targets()
            found = self.page_arg()
            if not targets or not found:
                return False
            homes = self.home_shelves()
            rows_now = self.ch_rows()
            several = bool(self.needs.items())
            state = self.__dict__.setdefault("fit_state", {})
            spent = sum(st.get("pages", 0) for st in state.values())
            floor = self.order_due() + 2 * max(0, self.ask_limit() - self.msgs_sent) + 4
            book = self.__dict__.get("page_book", {})
            if not any(family_rules(r.get("text") or "") for r in rows_now.values()):
                return False
            for k, model in targets:
                st = state.setdefault(str(k), {"key": None, "tried": [], "pages": 0, "done": False, "ver": None, "told": None,
                                               "home": []})
                if several:
                    home = set(homes[k]) if k is not None and k < len(homes) else set()
                else:
                    home = set(homes[0]) if homes else set()
                home = home or set(st["home"])
                if home and not any(r["shelf"] in home and family_rules(r.get("text") or "") for r in rows_now.values()):
                    continue
                name = self.needs.items()[k]["product"] if several else str(self.needs.reading.get("product") or "item")
                if not st["done"]:
                    if st["pages"] >= FIT_PAGES or spent >= FIT_TOTAL or self.moves_left() <= floor + 1:
                        st["done"] = True
                    else:
                        walk = book.get(st["key"]) if st["key"] else None
                        if walk is not None and not walk.get("total") and len(walk.get("read") or ()) >= 1:
                            walk, st["key"] = None, None
                        if walk is None:
                            pick = None
                            for args in self.code_queries(k):
                                key = json.dumps({a: v for a, v in args.items() if a != found[0]}, sort_keys=True, default=str)
                                if key in st["tried"]:
                                    continue
                                pick = (args, key)
                                break
                            if pick is None:
                                st["done"] = True
                            else:
                                args, key = pick
                                st["tried"].append(key)
                                st["key"] = key
                                if key not in book:
                                    st["pages"] += 1
                                    self.log.append({"kind": "fit_walk", "product": k, "model": model, "query": " ".join(self.free_text(args))[:80]})
                                    self.walk_call(args, f"The {name} must fit model {model}, and this store gives fit only in each row's "
                                                         "own text, not in a searchable field: I read every page of this search so that "
                                                         "each row's fit can be checked.")
                                    return True
                                walk = book.get(key)
                        if walk is not None and not st["done"]:
                            nxt = self.next_page(walk)
                            if nxt is not None:
                                st["pages"] += 1
                                query = " ".join(self.free_text(walk["args"]))
                                self.walk_call({**walk["args"], found[0]: nxt},
                                               f"I read the next page of '{query}' to check every row's fit for model {model}.")
                                return True
                            st["done"] = True
                if not home:
                    shelves = [s for s in ((book.get(st["key"]) or {}).get("row_shelf") or {}).values() if s] if st["key"] else []
                    top = max(set(shelves), key=shelves.count) if shelves else ""
                    if not top or shelves.count(top) * 5 < 3 * len(shelves):
                        continue
                    st["home"], home = [top], {top}
                    self.log.append({"kind": "fit_home", "product": k, "shelf": top[:60], "rows": len(shelves)})
                fits = self.fit_rows(k, model, home)
                sig = (self.needs.version, tuple(i for i, _ in fits[:6]))
                if st["told"] != sig and fits:
                    st["told"] = sig
                    self.fit_note(k, name, model, fits)
                asked_out = not self.asking or self.msgs_sent >= self.ask_limit() or self.moves_left() <= floor + 10
                if not st["done"] or not asked_out or st["ver"] == self.needs.version:
                    continue
                st["ver"] = self.needs.version
                queue = [i for i, shows in sorted(fits, key=lambda f: not f[1]) if i not in self.archive.read][:FIT_OPENS]
                if queue and self.moves_left() - len(queue) > floor:
                    self.log.append({"kind": "fit_open", "product": k, "model": model, "ids": queue})
                    self.vl_launch(queue, home, floor, f"I open the cheapest rows whose own text says they fit model {model}.")
                    return True
            return False
        except Exception as err:
            self.log.append({"kind": "fit_walk_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def fit_note(self, k: int | None, name: str, model: str, fits: list[tuple[str, bool]]) -> None:
        """Appends a fit-rows note to the last tool message in the chat: how many rows of the category say in their own
        text that they fit the model, the six cheapest with price and a read mark, and a reminder to take the cheapest
        one that meets every other requirement in force. Logs the count and the first ids; errors are ignored.
        """
        try:
            read = self.archive.read
            items = [f"{i} ({self.archive.price(i):g}{', read' if i in read else ''})" if is_number(self.archive.price(i)) else i
                     for i, _ in fits[:6]]
            text_body = (f"[fit rows] {name}: {len(fits)} row(s) of its category say in their own text that they fit {model} (a "
                         f"model-family rule covering it, without excepting it); cheapest first: " + ", ".join(items) + ". A row "
                         f"naming another family, or excepting {model}, does not fit. Of these, take the cheapest whose own record "
                         "meets every other requirement now in force.")
            if self.chat and self.chat[-1].get("role") == "tool":
                self.chat[-1]["content"] = str(self.chat[-1]["content"]) + "\n" + text_body
            self.log.append({"kind": "fit_rows", "product": k, "model": model, "n": len(fits), "first": [i for i, _ in fits[:4]]})
        except Exception:
            pass

    def code_rows(self) -> dict[str, dict[str, Any]]:
        """Builds a per-listing summary from every stored observation for the code-only finish. For each row whose id is
        a known archive id it merges the shelf, a title or name field, the first price, cost or amount number
        (skipping counts and totals), the stock state from key_stock, the set of kind words, up to 6000 characters of
        joined text and the flat values of any nested field dicts. Returns a dict of id to shelf, title, price, stock,
        words, fields and text.
        """
        out: dict[str, dict[str, Any]] = {}

        def visit(d: dict[str, Any]) -> None:
            """Folds one observed dict into the summary when its row id is a known archive id. It updates the shelf and title
            when present, takes the first price-like number whose key does not look like a count or total, sets stock from
            any stock key, adds the scalar values of the dict (and of its nested dicts) to the word set and the capped
            text, and copies nested dict fields as strings.
            """
            ident = self.row_mark(d)
            if ident not in self.archive.ids:
                return
            info = out.setdefault(ident, {"shelf": "", "title": "", "price": None, "stock": None, "words": set(), "fields": {}, "text": ""})
            info["shelf"] = self.shelf_of(d) or info["shelf"]
            info["title"] = next((str(val) for key, val in d.items() if isinstance(val, str) and re.search(r"title|name", str(key).lower())
                                  and "id" not in str(key).lower()), "") or info["title"]
            for key, val in d.items():
                name = str(key).lower()
                if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost|amount", name) \
                        and not re.search(r"count|qty|quant|total|units|line|due|payable|sum", name):
                    info["price"] = float(val)
                    break
            for key, val in d.items():
                state = self.key_stock(key, val)
                if state is not None:
                    info["stock"] = state
            pieces = [str(v) for v in d.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
            pieces += [str(x) for v in d.values() if isinstance(v, dict) for x in v.values() if isinstance(x, (str, int, float))]
            info["words"] |= kind_words(" ".join(pieces))
            if len(info["text"]) < 6000:
                info["text"] = (info["text"] + " " + " ".join(pieces))[:6000]
            for v in d.values():
                if isinstance(v, dict):
                    info["fields"].update({str(key): str(x) for key, x in v.items() if isinstance(x, (str, int, float)) and not isinstance(x, bool)})
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return out

    @staticmethod
    def code_shows(row: dict[str, Any], prop: str, value: str, hit_words: set[str], read: bool) -> bool:
        """Decides whether a row shows a required value: fields named like the property must carry it (code_seen);
        without such fields the title must carry it; failing that, an unread row passes only when all the value's kind
        words were among the words of searches that returned it.
        """
        named = [val for key, val in row["fields"].items() if same_prop(key, prop)]
        if named:
            return any(Errand.code_seen(value, val) for val in named)
        if Errand.code_seen(value, row.get("title") or ""):
            return True
        own = kind_words(value)
        return not read and bool(own) and own <= hit_words

    @staticmethod
    def code_seen(value: str, text: str) -> bool:
        """Checks whether a text carries a required value. The value is split into alternatives on or, slash, comma and
        semicolon; an alternative matches when each of its words (lightly stemmed) and each of its numbers appears in
        the text. Otherwise it matches only when the value holds model codes and a family-fit rule in the text covers
        every one of them.
        """
        pool = str(text or "").lower()
        if not pool.strip():
            return False
        body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", str(value).lower())
        for alt in re.split(r"\s*(?:\bor\b|/|,|;)\s*", body):
            words = [w for w in re.findall(r"[a-z]{2,}", alt) if w not in LINK_WORDS]
            figs = re.findall(r"\d+(?:\.\d+)?", alt)
            if (words or figs) and all(re.search(r"\b" + re.escape(w[:-1] if len(w) > 4 else w), pool) for w in words) \
                    and all(re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d.])", pool) for f in figs):
                return True
        codes = [c for c in re.findall(r"[A-Za-z]{1,8}[\s-]?\d{2,}[A-Za-z0-9-]*", str(value)) if code_like(c)]
        return bool(codes) and all(family_fit(text, c) for c in codes)

    def code_picks(self, avoid: set[str] | None = None, keep: set[str] | None = None) -> list[tuple[str, int]]:
        """Code-only finish: (id, quantity) per product, [] when one has no candidate.
        Pool: home-shelf rows minus failed, refused, avoided, dead, pointer,
        sold-out, over-cap and clashing rows.
        After a decline audit, unlisted rows stay out unless code alone judges them.
        Rank: kept lines, passes, fit, price (cheapest), read state, stock,
        field quality, word score.
        Pass 1: best per pool; over the cap, pass 2 by fit and price.
        Sets code_fit for the picks.
        Returns the picks.
        """
        rows = self.needs.items()
        counts = self.needs.counts()
        homes = self.home_shelves() if rows else [set(self.shelf_labels()) or (self.home_shelves() or [set()])[0]]
        info = self.code_rows()
        ver = str(self.needs.version)
        failed = {key.split("|")[0] for key, val in self.rulings.items() if val and key.split("|")[1:2] == [ver]}
        live = {key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == ver}
        reviewed = self.__dict__.get("audit_ids")
        if self.__dict__.get("audit_done") and (reviewed is None or len(self.archive.read) <= self.__dict__.get("audit_reads", -1)):
            reviewed = set(self.archive.read)
        answered = next((isinstance(e.get("result"), dict) for e in reversed(self.log) if isinstance(e, dict) and e.get("kind") == "audit"), False)
        audited = (set(reviewed or set()) - set(self.__dict__.get("audit_fits") or []) - live) if answered else set()
        own_all = bool(audited) and self.code_judges_all()
        cap = float(self.needs.spend_cap) if is_number(self.needs.spend_cap) else None

        def value_rank(row: dict[str, Any], prop: str, value: str) -> tuple[int, int]:
            """Rates how well a read row's fields show one required value as a (key match, value match) pair, lower being
            better: 0 for a field named exactly like the property and 1 for a similar name, then 0 for an exact value, 1
            for a matching part and 3 otherwise. Gives (2, 2) when only the title shows the value and (2, 3) when nothing
            does.
            """
            want = {" ".join(ch_words(alt)) for alt in re.split(r"\s*(?:\bor\b|/)\s*", value.lower()) if ch_words(alt)}
            best = (2, 3)
            for key, val in row["fields"].items():
                if not same_prop(key, prop) or not Errand.code_seen(value, val):
                    continue
                tier = 0 if " ".join(ch_words(val)) in want else 1 if any(" ".join(ch_words(p)) in want for p in ch_parts(val)) else 3
                best = min(best, (0 if prop_key(key) == prop_key(prop) else 1, tier))
            if best == (2, 3) and Errand.code_seen(value, row.get("title") or ""):
                best = (2, 2)
            return best
        asked = kind_words(self.request)
        hits = self.__dict__.get("code_hits") or {}
        cheapest = self.needs.priority() == "cheapest"
        avoid, keep = set(avoid or ()), set(keep or ())
        avoid |= self.dead_lines()
        avoid |= self.ref_pointers()
        pools: list[list[tuple]] = []
        for k, home in enumerate(homes):
            qty = counts[k] if k < len(counts) else 1
            want = kind_words(rows[k]["product"] if rows else self.needs.reading.get("product") or "")
            vals: set[str] = set()
            stated: list[tuple[str, str]] = []
            judged: list[tuple[str, str]] = []
            for prop, plain in self.code_values(k if rows else None):
                vals |= kind_words(plain)
                if re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b", plain.lower()):
                    continue
                judged.append((prop, plain))
                stated.append((prop, plain))
            pool = []
            apart = [(ref["prop"], [str(v) for v in [ref.get("value")] if str(v or "").strip()])
                     for ref in self.differ_done(k if rows else None)]
            for ident, row in info.items():
                if ident in failed or ident in self.refused_lines or ident in avoid or row["stock"] is False:
                    continue
                if (home and row["shelf"] not in home) or (not home and not (want & row["words"])):
                    continue
                if any(self.code_clash(row["fields"], prop, value) for prop, value in stated):
                    continue
                if self.ref_excluded(row["fields"], k if rows else None):
                    continue
                cost = row["price"] if row["price"] is not None else self.archive.price(ident)
                if cap is not None and is_number(cost) and cost * qty > cap:
                    continue
                fits = all(self.code_shows(row, prop, value, hits.get(ident) or set(), ident in self.archive.read) for prop, value in judged)
                fits = fits and all(any(same_prop(key, prop) and str(val).strip() and not any(ch_writes(v, val) for v in bad)
                                        for key, val in row["fields"].items()) for prop, bad in apart)
                shows = [value_rank(row, prop, value) for prop, value in judged] if ident in self.archive.read else []
                if ident in audited and not (own_all and fits and (judged or apart) and all(max(s) <= 1 for s in shows)):
                    continue
                score = 3 * len(vals & row["words"]) + 2 * len(want & row["words"]) + len(asked & row["words"]) \
                    + 4 * len(vals & (hits.get(ident) or set()))
                price = cost if is_number(cost) else float("inf")
                quality = (sum(s[0] for s in shows), sum(s[1] for s in shows)) if shows else (9, 9)
                pool.append((0 if ident in keep else 1, 0 if ident in live else 1, not fits, cap is not None and not is_number(cost),
                             price if cheapest else 0.0, 0 if ident in self.archive.read else 1, row["stock"] is not True,
                             (quality, -score), price, ident))
            if not pool:
                return []
            pools.append(sorted(pool))
        for rank in ((lambda e: e), (lambda e: (e[2], e[3], e[8], e[-1]))):
            picks: list[tuple[str, int]] = []
            total, chosen = 0.0, []
            for k, pool in enumerate(pools):
                entry = next((e for e in sorted(pool, key=rank) if all(e[-1] != p for p, _ in picks)
                              and not (e[-1] in self.archive.read and self.field_view(e[-1])["fails"])), None)
                if entry is None:
                    return []
                picks.append((entry[-1], counts[k] if k < len(counts) else 1))
                chosen.append(entry)
                total += (entry[8] if entry[8] != float("inf") else 0.0) * picks[-1][1]
            if cap is None or total <= cap:
                self.code_fit = {e[-1]: not e[2] for e in chosen}
                return picks
        return []

    @staticmethod
    def code_clash(fields: dict[str, str], prop: str, value: str) -> bool:
        """True when some field keys name the property and none of their values matches the required value."""
        named = [val for key, val in fields.items() if same_prop(key, prop)]
        return bool(named) and not any(value_seen(value, val) for val in named)

    def cart_form(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None, list[str]]:
        """Reads the cart tool's schema into the pieces needed to build cart calls. For each required enum argument it
        picks a put value (the one learned from a working cart call, else a value with add-like words such as add or
        put, else the first) and a drop value (a remove-like value, or the other one of two). Returns the schema, the
        put and drop argument maps, the string argument most likely to hold the listing id, and the numeric non-enum
        arguments.
        """
        jschema = self.shapes.get(self.role("cart") or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        known = json_object(self.trolley.put_form) if self.trolley.put_form else None

        def words(v: Any) -> set[str]:
            """Splits a value into its set of lowercase words, breaking camelCase first."""
            return set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(v)).lower()))
        put: dict[str, Any] = {}
        drop: dict[str, Any] = {}
        for key in jschema.get("required") or []:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            vals = spec.get("enum") if isinstance(spec.get("enum"), list) else []
            if not vals:
                continue
            puts = [v for v in vals if words(v) & PUT_WORDS and not words(v) & DROP_WORDS]
            takes = [v for v in vals if words(v) & DROP_WORDS]
            put[key] = known[key] if isinstance(known, dict) and known.get(key) in vals else (puts[0] if puts else vals[0])
            others = [v for v in vals if v != put[key]]
            drop[key] = next((v for v in takes if v != put[key]), others[0] if len(vals) == 2 else None)
        texts = [key for key, spec in props.items() if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")]
        id_key = next((key for key in texts if re.search(r"id|item|product|listing|sku|target|which|number", key.lower())),
                      texts[0] if texts else None)
        counts = [key for key, spec in props.items() if isinstance(spec, dict) and spec.get("type") in ("integer", "number") and not spec.get("enum")]
        return jschema, put, drop, id_key, counts

    def cart_args(self, picks: list[tuple[str, int]]) -> list[dict[str, Any]]:
        """Builds the cart calls that put the picked listings in at their quantities. When the trolley knows a line-array
        form, it makes one call with an array of id and count entries; otherwise one call per pick with the id
        argument and, when a numeric argument exists, the first one set to the quantity, each on top of the put enum
        values. Every call is validated by ArgCheck, and any fault returns an empty list.
        """
        jschema, base, _, id_key, counts = self.cart_form()
        spec_lines = self.trolley.line_spec()
        calls: list[dict[str, Any]] = []
        if spec_lines:
            arr, id_field, count_field = spec_lines
            calls.append({**base, arr: [{id_field: ident, count_field: qty} for ident, qty in picks]})
        else:
            if not id_key:
                return []
            for ident, qty in picks:
                call_args = {**base, id_key: ident}
                if counts:
                    call_args[counts[0]] = qty
                calls.append(call_args)
        out = []
        for call_args in calls:
            fixed, faults = ArgCheck.execute(call_args, jschema)
            if faults:
                return []
            out.append(fixed)
        return out

    def cart_drops(self, lines: list[str]) -> list[dict[str, Any]] | None:
        """Builds the cart calls that take unwanted lines out. Returns an empty list for no lines or a line-array cart,
        and None when there is no id argument or no safe removal form. With enum actions it uses the drop values with
        the id and the held count; without them it sets the first count argument to zero when its minimum allows that.
        Any ArgCheck fault also returns None.
        """
        if not lines or self.trolley.line_spec():
            return []
        jschema, put, drop, id_key, counts = self.cart_form()
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        if not id_key:
            return None
        out = []
        for ident in lines:
            if put:
                if any(drop.get(key) is None for key in put):
                    return None
                call_args = {**drop, id_key: ident}
                if counts:
                    call_args[counts[0]] = max(1, int(self.trolley.held(ident) or 1))
            elif counts and (not is_number((props.get(counts[0]) or {}).get("minimum")) or props[counts[0]]["minimum"] <= 0):
                call_args = {id_key: ident, counts[0]: 0}
            else:
                return None
            fixed, faults = ArgCheck.execute(call_args, jschema)
            if faults:
                return None
            out.append(fixed)
        return out

    def code_replies(self, n: int) -> list[Any]:
        """Returns up to n observations from the latest env entry in the log: the reply's observations list, or its
        single observation.
        """
        entry = next((e for e in reversed(self.log) if isinstance(e, dict) and e.get("kind") == "env"), None)
        resp = (entry or {}).get("reply")
        if not isinstance(resp, dict):
            return []
        obs_list = resp.get("observations") if isinstance(resp.get("observations"), list) else [resp.get("observation")]
        return list(obs_list)[:n]

    @staticmethod
    def code_fault(obs: Any) -> bool:
        """Says whether an observation reports a problem: true when some dict in its first few nesting levels has a key
        matching FAULT_KEYS (error, fault, refusal, invalid and the like) with a non-empty value.
        """
        found = []

        def visit(d: dict[str, Any]) -> None:
            """Collects a dict when one of its fault-like keys holds a non-empty value."""
            if any(re.search(FAULT_KEYS, str(key).lower()) and val not in (None, False, "", [], {}) for key, val in d.items()):
                found.append(d)
        walk_dicts(obs, visit, 0, 2)
        return bool(found)

    def code_search(self, products: int, more: bool = False) -> None:
        """Code-built searches for the code-only finish, no model. more unset:
        all products' ranked code_queries interleaved, trimmed to moves
        left - reserve, 1 + CODE_PAGES rounds (2 more if cheapest asked); more
        set: one round of code_more pages. Each round via send_chunk; unless
        search is known to match any word, query kind words -> code_hits
        per hit. Lead queries' next pages -> code_more; stops early if all
        picks fit (cheapest not asked) or no calls fit. Sets code_searched.
        """
        search = self.role("search")
        texts = [key for key, spec in ((self.shapes.get(search or "") or {}).get("properties") or {}).items()
                 if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")]
        reserve = 2 + 2 * products
        if more:
            lead = list(self.__dict__.get("code_more") or [])
            calls = lead[: max(0, self.moves_left() - reserve - 2 * products)]
            rounds = 1
        else:
            lists = [self.code_queries(k if self.needs.items() else None) for k in range(products)]
            calls = []
            for rank in range(max((len(q) for q in lists), default=0)):
                calls += [q[rank] for q in lists if rank < len(q) and q[rank] not in calls]
            calls = calls[: max(0, self.moves_left() - reserve)]
            lead = [q[0] for q in lists if q]
            rounds = 1 + CODE_PAGES + (2 if self.needs.priority() == "cheapest" else 0)
        if not calls:
            return
        self.code_searched = True
        for rnd in range(rounds):
            self.send_chunk([(f"code_search_{n}_{self.wire.tick}", search, call_args) for n, call_args in enumerate(calls)])
            replies = self.code_replies(len(calls))
            if self.__dict__.get("word_kind") != "any":
                hits = self.__dict__.setdefault("code_hits", {})
                for call_args, obs in zip(calls, replies):
                    said = " ".join(str(call_args.get(key) or "") for key in texts)
                    if not said.strip():
                        continue
                    words = kind_words(said)

                    def visit(d: dict[str, Any]) -> None:
                        """Adds the query's kind words to code_hits for a returned row whose id is a known archive id."""
                        ident = self.row_mark(d)
                        if ident in self.archive.ids:
                            hits.setdefault(ident, set()).update(words)
                    walk_dicts(obs, visit)
            following = []
            for call_args, obs in zip(calls, replies):
                if call_args in lead:
                    nxt = self.code_next(call_args, obs)
                    if nxt:
                        following.append(nxt)
            self.code_more = following
            if rnd == rounds - 1:
                break
            picks = self.code_picks()
            fit = self.__dict__.get("code_fit") or {}
            if self.needs.priority() != "cheapest" and picks and all(fit.get(ident) for ident, _ in picks):
                break
            lead = following
            calls = following[: max(0, self.moves_left() - reserve - 3 * products)]
            if not calls:
                break

    def code_next(self, call_args: dict[str, Any], obs: Any) -> dict[str, Any] | None:
        """Works out the arguments for the next results page of a search call, or None when there is none. It needs a
        paging argument (page_arg) and a list of row dicts in the reply, and stops when every next, cursor or more
        marker in the reply is empty. In token mode it puts the first non-empty marker in the token argument;
        otherwise it adds one page or one page size to the current value and stops past the schema maximum or, without
        markers, once the reported total is covered. The new arguments are validated against the search schema.
        """
        spec = self.page_arg()
        if not spec:
            return None
        key, by_page, low, high = spec
        pages: list[dict[str, Any]] = []

        def visit(d: dict[str, Any]) -> None:
            """Keeps a dict holding a non-empty list made only of dicts, which marks a page of result rows."""
            if any(isinstance(v, list) and v and all(isinstance(x, dict) for x in v) for v in d.values()):
                pages.append(d)
        walk_dicts(obs, visit, 0, 3)
        if not pages:
            return None
        page = pages[0]
        rows = max(len(v) for v in page.values() if isinstance(v, list))
        marks = [v for k, v in page.items() if re.search(r"next|onward|cursor|bookmark|token|continu|more", str(k).lower())]
        total = next((v for k, v in page.items() if is_number(v) and not isinstance(v, bool)
                      and re.search(r"count|total|matched|found|hits", str(k).lower())), None)
        if marks and all(m in (None, False, "", 0) for m in marks):
            return None
        if by_page == "token":
            tok = next((m for m in marks if isinstance(m, str) and m.strip()), None)
            nxt_args = {**call_args, key: tok} if tok else None
        else:
            cur = call_args.get(key, low)
            cur = cur if is_number(cur) else low
            nxt = cur + 1 if by_page else cur + rows
            if (high is not None and nxt > high) or (not marks and (total is None or ((cur - low + 1) * rows if by_page else cur + rows) >= total)):
                return None
            nxt_args = {**call_args, key: int(nxt) if float(nxt).is_integer() else nxt}
        if not nxt_args:
            return None
        fixed, faults = ArgCheck.execute(nxt_args, self.shapes.get(self.role("search") or "") or {})
        return None if faults else fixed

    def code_read(self, picks: list[tuple[str, int]], avoid: set[str], keep: set[str] | None) -> list[tuple[str, int]]:
        """Opens unread picks so their records can be checked, for up to CODE_READS rounds. Each round reads them with
        the batch tool in one call (two or more unread) or with one details call each, then reruns code_picks with the
        same avoid and keep sets. It stops when the picks no longer change, nothing is unread, no read tool exists, or
        moves or seconds run short, and returns the latest picks.
        """
        for _ in range(CODE_READS):
            unread = [ident for ident, _ in picks if ident not in self.archive.read]
            batch = self.batch_tool()
            reads = 1 if batch and len(unread) >= 2 else len(unread)
            if not unread or not (batch or self.role("details")) or self.moves_left() < reads + 2 * len(picks) + 3 or self.wire.spare() <= 8:
                break
            if batch and len(unread) >= 2:
                slab = [(f"code_read_{self.wire.tick}", batch[0], {batch[1]: unread[: batch[2]]})]
            else:
                slab = [(f"code_read_{n}_{self.wire.tick}", self.role("details"), call_args)
                        for n, call_args in enumerate(self.details_call(ident) for ident in unread) if call_args]
            if not slab:
                break
            self.send_chunk(slab)
            fresh = self.code_picks(avoid, keep)
            if fresh == picks:
                break
            picks = fresh
        return picks

    def code_put(self, picks: list[tuple[str, int]]) -> list[str] | None:
        """Makes the cart hold exactly the picks. It removes lines that are not picks and puts picks whose held count
        differs (or all picks with a line-array cart), sending the calls in one chunk. Returns None when no valid cart
        call can be built, moves are short or replies are missing. Otherwise it returns the sorted ids that failed:
        named in a faulted reply, or absent from the cart as shown after the calls (fetched with the cart_view role
        when no reply showed it). Failed ids are taken out of the trolley's units.
        """
        cart = self.role("cart")
        want = {ident for ident, _ in picks}
        spec_lines = self.trolley.line_spec()
        extra = [line for line in self.trolley.units if line not in want]
        todo = list(picks) if spec_lines else [(ident, qty) for ident, qty in picks if abs(self.trolley.held(ident) - qty) > 1e-9]
        drops = self.cart_drops(extra)
        puts = self.cart_args(todo) if todo else []
        if drops is None or (todo and not puts):
            return None
        calls = drops + puts
        if not calls:
            return []
        if self.moves_left() < len(calls) + 1:
            return None
        self.send_chunk([(f"code_cart_{n}_{self.wire.tick}", cart, call_args) for n, call_args in enumerate(calls)])
        replies = self.code_replies(len(calls))
        bad: set[str] = set()
        shown = None
        for call_args, obs in zip(calls, replies):
            seen = self.trolley.view_units(obs)
            if seen is not None:
                shown = seen
            if self.code_fault(obs):
                named = [ident for ident in want if ident in as_text(obs)]
                lines = [line for line, _ in self.trolley.lines(call_args) if line in want]
                bad.update(named or lines or (list(want) if spec_lines else []))
        if len(replies) < len(calls):
            return None
        if shown is None and not bad and self.role("cart_view") and self.moves_left() >= 3:
            self.send_chunk([(f"code_view_{self.wire.tick}", self.role("cart_view"), {})])
            view = self.code_replies(1)
            shown = self.trolley.view_units(view[0]) if view else None
            if view and shown is None:
                bad.update(ident for ident in want if ident not in self.trolley.units)
        if shown is not None:
            bad.update(ident for ident in want if ident not in shown)
        for ident in bad:
            self.trolley.units.pop(ident, None)
        return sorted(bad)

    def code_recheck(self, lines: list[str]) -> tuple[bool, set[str]]:
        """Re-reads the given lines with the stock tool (or the view or details role) to confirm stock before ordering.
        Returns (False, empty set) when there is nothing to read, no tool, or too few moves or seconds; otherwise it
        sends one lookup per line and returns True with the ids whose rebuilt row shows them out of stock.
        """
        tool = self.stock_tool() or self.role("view") or self.role("details")
        if not lines or not tool or self.moves_left() < len(lines) + 2 or self.wire.spare() < 8:
            return False, set()
        slab = []
        for n, ident in enumerate(lines):
            call_args = self.lookup_call(ident, tool)
            if call_args:
                slab.append((f"code_check_{n}_{self.wire.tick}", tool, call_args))
        if not slab:
            return False, set()
        self.send_chunk(slab)
        info = self.code_rows()
        return True, {ident for ident in lines if (info.get(ident) or {}).get("stock") is False}

    def code_news(self, lines: list[str]) -> None:
        """Scans shopper lines not seen before, plus those still pending in needs, for a new budget figure. A positive
        figure from code_budget becomes needs.spend_cap and is logged as code_news; seen lines are kept in news_read.
        finish_now calls it on each reply so a changed budget applies to the next code-built basket.
        """
        waiting = [ln for lns, _ in self.needs.__dict__.get("pending") or [] for ln in lns]
        seen = self.__dict__.setdefault("news_read", set())
        for line in list(lines) + waiting:
            if line in seen:
                continue
            seen.add(line)
            figure, _ = self.code_budget(line)
            if is_number(figure) and figure > 0:
                self.needs.spend_cap = figure
                self.log.append({"kind": "code_news", "budget": figure})

    def code_cart(self, avoid: set[str] | None = None, keep: set[str] | None = None) -> bool:
        """Builds and fills the cart by code alone; True when it holds checked lines.
        Needs a cart role, two moves and four seconds; model calls are muted.
        Up to four rounds: picks, one code_search when they do not fit,
        reads, puts and a stock re-check.
        Failed or sold-out lines join the avoid set and the round repeats.
        Logs code_finish, code_swap or code_finish_error.
        False when nothing can be ordered.
        """
        cart, search = self.role("cart"), self.role("search")
        if not cart or self.wire.done or self.wire.broke or self.moves_left() < 2 or self.wire.spare() < 4:
            return False
        self.wire.mute = True
        try:
            avoid = set(avoid or ())
            products = max(1, len(self.needs.items()))
            for _ in range(4):
                picks = self.code_picks(avoid, keep)
                unfit = not picks or not all(self.__dict__.get("code_fit", {}).get(ident) for ident, _ in picks)
                if search and not self.__dict__.get("code_searched") and self.moves_left() >= 4 and unfit:
                    self.code_search(products)
                    picks = self.code_picks(avoid, keep)
                elif search and self.__dict__.get("code_more") and self.moves_left() >= 6 + 2 * products and (
                        unfit or self.needs.priority() == "cheapest"):
                    self.code_search(products, True)
                    picks = self.code_picks(avoid, keep)
                picks = self.code_read(picks, avoid, keep)
                if not picks:
                    self.log.append({"kind": "code_finish", "picks": [], "result": "nothing to order"})
                    return False
                bad = self.code_put(picks)
                if bad is None:
                    self.log.append({"kind": "code_finish", "picks": picks, "result": "no cart call"})
                    return False
                if not bad:
                    _, sold = self.code_recheck([ident for ident, _ in picks]) if self.moves_left() >= len(picks) + 3 else (False, set())
                    bad = sorted(sold)
                if not bad:
                    self.log.append({"kind": "code_finish", "picks": picks, "cart": dict(self.trolley.units)})
                    return bool(self.trolley.units)
                avoid |= set(bad)
                self.log.append({"kind": "code_swap", "lines": bad})
            return False
        except Exception as err:
            self.log.append({"kind": "code_finish_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def finish_now(self) -> None:
        """Closing call when the normal loop must stop; up to CODE_FINISHES tries.
        First turn with cart lines: gone lines join the avoid set.
        Orders the cart as is, else after code_cart fills it;
        later turns reorder if allowed, or decline with the standard apology.
        Orders follow any due basket statement.
        Absorbs replies and budget news of non-final calls.
        An unfinished order re-checks stock and repeats, or retries without sold lines.
        Out of turns: the last order if allowed, else a decline.
        Logs forced_finish.
        """
        if self.wire.done or self.wire.broke or self.moves_left() <= 0 or self.wire.spare() < 2:
            return
        call_rank, decl = self.role("order"), self.role("decline")
        avoid: set[str] = set()
        keep: set[str] | None = None
        move: dict[str, Any] | None = None
        kind = None
        for turn in range(CODE_FINISHES):
            if self.wire.done or self.wire.broke or self.moves_left() <= 0 or self.wire.spare() < 2:
                return
            if move is None:
                gone: set[str] = set()
                if call_rank and turn == 0 and self.trolley.units:
                    try:
                        self.wire.mute = True
                        lines0 = list(self.trolley.units)
                        gone = self.cart_gone(lines0)
                        if gone:
                            avoid |= gone
                            keep = set(lines0) - gone
                            self.log.append({"kind": "code_dead", "lines": sorted(gone)})
                    except Exception as err:
                        self.log.append({"kind": "cart_gone_error", "error": f"{type(err).__name__}: {err}"[:300]})
                        gone = set()
                if call_rank and turn == 0 and not gone and self.trolley.used and (self.trolley.units or self.trolley.view_units(self.trolley.last_view) is None):
                    move = {"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")}
                elif call_rank and self.code_cart(avoid, keep):
                    move = {"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")}
                elif call_rank and turn > 0 and self.again_ok(avoid):
                    move = {"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")}
                elif decl:
                    move = {"name": decl, "args": self.plain_args(decl, SORRY, "decline")}
                else:
                    return
            if self.call_kind(move["name"], move["args"]) == "order" and self.statement_due() and self.wire.spare() > 4:
                try:
                    self.state_gate(None, True)
                except Exception as err:
                    self.log.append({"kind": "statement_error", "error": f"{type(err).__name__}: {err}"[:300]})
                if self.wire.done or self.wire.broke or self.moves_left() <= 0:
                    return
            resp = self.wire.act([move])
            self.log.append({"kind": "forced_finish", "action": move, "reply": resp})
            kind = self.call_kind(move["name"], move["args"])
            if not resp or resp.get("finished") or self.wire.done or turn == CODE_FINISHES - 1 or self.moves_left() <= 1:
                break
            self.wire.mute = True
            lines = list(self.trolley.units)
            obs = resp.get("observation")
            self.absorb(move["name"], move["args"], obs, resp, True)
            self.code_news(self.shopper_lines(move["name"], obs, True))
            move = None
            if kind != "order" or not lines:
                continue
            read, sold = self.code_recheck(lines)
            if not read:
                continue
            self.refused_lines.difference_update(set(lines) - sold)
            info = self.code_rows()
            total = sum(((info.get(line) or {}).get("price") or 0.0) * self.trolley.held(line) for line in lines)
            cap = self.needs.spend_cap
            if not sold and not (is_number(cap) and total > cap):
                move = {"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")}
            elif is_number(cap) and total > cap:
                avoid, keep = set(sold), None
            else:
                avoid, keep = set(sold), set(lines) - sold
        if not self.wire.done and resp and not resp.get("finished") and decl and kind != "decline" and self.moves_left() > 0:
            again = False
            try:
                again = bool(call_rank) and kind == "order" and self.again_ok(avoid)
            except Exception as err:
                self.log.append({"kind": "again_error", "error": f"{type(err).__name__}: {err}"[:300]})
            move = ({"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")} if again
                    else {"name": decl, "args": self.plain_args(decl, SORRY, "decline")})
            if again and self.statement_due() and self.moves_left() >= 2 and self.wire.spare() > 4:
                try:
                    self.state_gate(None, True)
                except Exception as err:
                    self.log.append({"kind": "statement_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if self.wire.done or self.wire.broke or self.moves_left() <= 0:
                return
            self.log.append({"kind": "forced_finish", "action": move, "reply": self.wire.act([move])})

    def again_ok(self, avoid: set[str]) -> bool:
        """True when the cart holds lines, none of them is avoided or dead, and code_picks still finds a basket without
        those avoided lines.
        """
        units = set(self.trolley.units)
        gone = set(avoid) | self.dead_lines()
        return bool(units) and not units & gone and bool(self.code_picks(gone))

    def catch_up(self) -> None:
        """Brings late model results into the chat while time remains. If the request was first read by code and a model
        reply has arrived since, it asks the READER models to read the request again once and, when the answer has a
        roles map, uses that reading and notes the new plan overview. It also hears the three latest pending shopper
        lines once a reply came after they were queued. Notes go on the last user or tool message (or a new user
        message) with the status line. Skipped when muted or under CLOSING seconds; errors are logged.
        """
        try:
            if self.wire.mute or self.wire.spare() < CLOSING:
                return
            notes = []
            if self.needs.reading.get("code_reading") and not self.__dict__.get("reread") \
                    and self.wire.last_reply > self.__dict__.get("read_at", self.wire.last_reply) and self.wire.spare() > CLOSING + 60:
                self.reread = True
                msg = self.wire.query_model([{"role": "user", "content": self.read_content}], READER, 3000)
                reading = json_object((msg or {}).get("content") or "") or {}
                self.log.append({"kind": "re_read", "ok": isinstance(reading.get("roles"), dict)})
                if isinstance(reading.get("roles"), dict):
                    self.use_reading(reading, True)
                    notes.append("[plan updated] the models answer again and read the request; this reading replaces the first one:\n"
                                 + self.needs.overview(self.step_cap, self.msg_cap))
            waiting = self.needs.__dict__.get("pending") or []
            if waiting and self.wire.last_reply > self.needs.__dict__.get("pending_at", 0.0) and self.wire.spare() >= CLOSING:
                self.needs.pending = []
                for lns, qtext in waiting[-3:]:
                    said = self.needs.hear(lns, qtext)
                    if said:
                        notes.append("[the shopper's earlier lines, read now] " + said)
            if notes:
                text_body = "\n".join(notes) + "\n" + self.status_line()
                last = self.chat[-1] if self.chat else None
                if last and last.get("role") in ("user", "tool") and isinstance(last.get("content"), str):
                    last["content"] += "\n" + text_body
                else:
                    self.chat.append({"role": "user", "content": text_body})
        except Exception as err:
            self.log.append({"kind": "catch_up_error", "error": f"{type(err).__name__}: {err}"[:300]})

    def refused(self, call_name: str, obs: Any, resp: dict[str, Any]) -> bool:
        """Says whether a finish was refused: the observation has a refusal key, or an order call did not finish the
        episode and its error is neither invalid_args nor unknown_tool.
        """
        if isinstance(obs, dict) and "refusal" in obs:
            return True
        return call_name == self.role("order") and not resp.get("finished") and not (isinstance(obs, dict) and obs.get("error") in ("invalid_args", "unknown_tool"))

    def notes_for(self, call_name: str, obs: Any, refused: bool) -> str:
        """Remarks added to a tool result after a call.
        A budget_exceeded error caps msg_cap at the messages sent.
        A refused decline resets the audit flags: look again and order what fits.
        A refused order counts the refusal, resets order_vetted and marks the cart lines
        refused; its sentences go to refusal_texts with advice to order again.
        Other results with several keys and lists of strings pass on as shopper
        messages, unless they repeat the refusal already handled.
        """
        remarks = []
        if isinstance(obs, dict) and obs.get("error") == "budget_exceeded":
            if call_name == self.role("message") or self.msg_cap is not None:
                self.msg_cap = self.msgs_sent if self.msg_cap is None else min(self.msg_cap, self.msgs_sent)
            remarks.append("That budget is used up; do not call this tool again.")
        if refused and call_name == self.role("decline") and getattr(self, "last_kind", None) == "decline":
            self.audit_done = False
            self.decline_vetted = False
            self.event_seen = True
            remarks.append("THE DECLINE WAS REFUSED and the episode continues: the store does not accept that nothing fits. Read the refusal, apply what it says, look again at the closest candidates (their own properties, stock and price against the requirements now), and order the one that fits; decline again only if each one truly fails.")
        elif refused:
            self.refusal_n += 1
            self.order_vetted = False
            self.event_seen = True
            reread = ""
            try:
                reread = self.refusal_reread(obs)
            except Exception as err:
                self.log.append({"kind": "refusal_reread_error", "error": f"{type(err).__name__}: {err}"[:300]})
            self.refused_lines.update(self.trolley.units)
            said: list[str] = []

            def gather(piece: Any, lvl: int = 0) -> None:
                """Walks a refusal observation up to six levels deep, at most 20 items per list, and collects every string of
                four or more words into said. These sentences become refusal_texts, which lets a later result that only
                repeats the refusal be recognised.
                """
                if lvl > 6:
                    return
                if isinstance(piece, dict):
                    for val in piece.values():
                        gather(val, lvl + 1)
                elif isinstance(piece, list):
                    for val in piece[:20]:
                        gather(val, lvl + 1)
                elif isinstance(piece, str) and len(piece.split()) >= 4:
                    said.append(piece)
            gather(obs)
            self.refusal_texts = {" ".join(str(s).lower().split()) for s in said}
            remarks.append("THE ORDER WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (take out any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again." + reread)
            if self.states_basket() and self.stated_sig is not None:
                if self.msgs_left() <= 0:
                    remarks.append("No message is left for a new basket statement. If the stated basket can still be bought (every "
                                   "line in stock, the total inside the budget) and still fits the request as changed, order it again "
                                   "as it stands; otherwise put the best fit in and order it. Never decline or empty the cart because "
                                   "no statement is possible: a decline is right only when nothing fits.")
                else:
                    remarks.append("When the stated basket can still be bought (every line in stock, the total inside the budget) and is "
                                   "still the right choice under the change, order it again as it stands; when the cart must change, change it, and the code states "
                                   "the new basket to the shopper before the next order.")
        elif isinstance(obs, dict) and len(obs) >= 2 and call_name != self.role("message"):
            lns: list[str] = []
            for piece in obs.values():
                if isinstance(piece, list) and piece and all(isinstance(val, str) for val in piece):
                    long_strings(piece, lns)
            seen = (self.__dict__.pop("refusal_texts", None) or set()) if lns else set()
            if lns and seen and all(any(" ".join(str(s).lower().split()) in t for t in seen) for s in lns):
                remarks.append("This result repeats the message the refused finish already carried (handled above); it says nothing "
                               "about this result.")
            elif lns:
                self.event_seen = True
                remarks.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(lns[:4]))
        return " ".join(remarks)

    @staticmethod
    def shelf_of(row: dict[str, Any]) -> str:
        """Returns a row's category in lowercase from the first category, shelf, department or aisle field of 1 to 60
        characters that is not an id field, or an empty string.
        """
        return next((val.strip().lower() for key, val in row.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                     and re.search(r"categor|shelf|department|aisle", str(key).lower()) and "id" not in str(key).lower()), "")

    @staticmethod
    def key_stock(key: Any, val: Any) -> bool | None:
        """Reads one field as a stock signal. Returns None when the key is not about stock, availability or on-hand
        counts, or the value is None. Otherwise it returns False for out-of-stock signs (a true out or sold flag, a
        false or zero stock value, words such as out, sold, none or unavailable, a zero on-hand count, or backorder,
        preorder or discontinued text) and True for anything else.
        """
        name = str(key).lower()
        if val is None or not re.search(r"stock|avail|on.?hand", name):
            return None
        gone = False
        if re.search(r"stock|avail", name):
            if re.search(r"out|sold", name):
                gone = val is True
            else:
                gone = val is False or (is_number(val) and val <= 0) or (
                    isinstance(val, str) and bool(re.search(r"\b(?:out|sold|none|unavailable|not|no|false)\b", val.lower())))
        if re.search(r"on.?hand", name) and (val is False or (is_number(val) and val <= 0)):
            gone = True
        if isinstance(val, str) and re.search(r"back.?order|pre.?order|discontinu", val.lower()):
            gone = True
        return not gone

    @staticmethod
    def stock_state(row: dict[str, Any]) -> bool | None:
        """Combines a row's stock signals: True when every one says in stock, False when any says out, None when there
        are none.
        """
        states = [s for s in (Errand.key_stock(key, val) for key, val in row.items()) if s is not None]
        return all(states) if states else None

    @staticmethod
    def sold_out(row: dict[str, Any]) -> bool:
        """True when any field of the row signals out of stock."""
        return any(Errand.key_stock(key, val) is False for key, val in row.items())

    def shelf_labels(self) -> set[str]:
        """Finds the category labels the episode is shopping on. It counts the categories of read rows, once per id and
        skipping ids of foreign references, and returns the most frequent ones for a single-item request or when
        nothing was counted. With several products it takes, for each product's home shelves, the most counted of
        those that were read (or the whole set when none was), then adds the most counted other shelves while some
        product has no home, until there is one per product. Falls back to the most frequent shelves on any error.
        """
        tally: dict[str, int] = {}
        counted: set[str] = set()
        skip = self.ref_foreign()

        def visit(d: dict[str, Any]) -> None:
            """Counts a dict's category once when an id field names a read listing that was not counted yet and is not a
            foreign reference.
            """
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.archive.read), None)
            shelf = self.shelf_of(d) if ident and ident not in counted and ident not in skip else ""
            if shelf:
                counted.add(ident)
                tally[shelf] = tally.get(shelf, 0) + 1
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        top = max(tally.values(), default=0)
        best = {shelf for shelf, n in tally.items() if n == top}
        if not self.needs.items() or not tally:
            return best
        try:
            homes = self.home_shelves()
            picked: set[str] = set()
            for home in homes:
                read = [s for s in home if s in tally]
                most = max((tally[s] for s in read), default=0)
                picked |= {s for s in read if tally[s] == most} if read else set(home)
            for shelf in sorted(tally, key=lambda s: -tally[s]):
                if all(homes) or len(picked) >= len(homes):
                    break
                picked.add(shelf)
            return picked or best
        except Exception:
            return best

    @staticmethod
    def row_mark(row: dict[str, Any]) -> str:
        """Returns a row's id: the value of a key named exactly id, else of the first key ending in id (string or
        integer), else the whole row as sorted JSON.
        """
        for exact in (True, False):
            for key, val in row.items():
                name = str(key).lower()
                if (name == "id" if exact else name.endswith("id")) and isinstance(val, (str, int)) and not isinstance(val, bool):
                    return str(val)
        return json.dumps(row, sort_keys=True, default=str)

    def search_keys(self, call_args: dict[str, Any]) -> tuple[list[tuple], tuple, bool, set]:
        """Splits a search call's arguments into the parts used to learn how the store's search behaves. Returns the
        free-text arguments as (key, word tuple) pairs, kept only when every word is plain alphanumeric; the sorted
        enum mode values; whether this is a first page (empty token, or page or offset at its minimum); and the set of
        paging keys.
        """
        props = self.shapes.get(self.role("search") or "")
        props = props.get("properties") if isinstance(props, dict) else None
        props = props if isinstance(props, dict) else {}

        def spec(key: str) -> dict[str, Any]:
            """Returns the search schema entry of a key, or an empty dict."""
            pspec = props.get(key)
            return pspec if isinstance(pspec, dict) else {}
        texts, mode, first, paging = [], [], True, set()
        tok = self.token_key()
        for key, val in call_args.items():
            name = str(key).lower()
            if key == tok and isinstance(val, str):
                paging.add(key)
                first = first and not val.strip()
            elif isinstance(val, str) and spec(key).get("enum"):
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
        """Learns whether the store's search needs all words or any word. On first-page searches it compares each free-
        text argument with earlier queries in the same mode and with the same other arguments: when one query extends
        the other and the totals differ, more words giving more results means any, otherwise all. A strict word subset
        with fewer results for the larger set also points to all. Each query is kept in word_seen; returns word_kind.
        """
        texts, mode, first, paging = self.search_keys(call_args)
        if not first:
            return self.word_kind
        total = max([val for val in obs.values() if isinstance(val, int) and not isinstance(val, bool)] + [len(rows)])
        for key, words in texts:
            ctx = (key, mode)
            others = tuple(sorted((k2, str(v2)) for k2, v2 in call_args.items() if k2 != key and k2 not in paging))
            subset = False
            for seen_ctx, seen_others, seen_words, seen_total in self.word_seen:
                if seen_ctx != ctx or seen_others != others or seen_total == total:
                    continue
                if len(words) > len(seen_words) and words[: len(seen_words)] == seen_words:
                    self.word_kind = "any" if total > seen_total else "all"
                elif len(seen_words) > len(words) and seen_words[: len(words)] == words:
                    self.word_kind = "any" if total < seen_total else "all"
                else:
                    cut = self.lead_words or None
                    mine, theirs = set(words[:cut]), set(seen_words[:cut])
                    subset = subset or (mine > theirs and total < seen_total) or (theirs > mine and seen_total < total)
            if subset and not self.word_kind:
                self.word_kind = "all"
            self.word_seen.append((ctx, others, words, total))
        return self.word_kind

    def lead_rule(self, call_args: dict[str, Any], obs: dict[str, Any], rows: list) -> int:
        """Learns how many leading words of a query the store's search actually uses. For first-page searches it compares
        the query with earlier ones in the same context: a shared prefix of k words with a different total or row
        order proves that more than k words count, while identical results from queries agreeing on all of the shorter
        one mark k as a possible limit. Sets lead_words and lead_ctx to a limit just above the proven bound, or zero,
        and returns lead_words.
        """
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

    def token_key(self) -> str | None:
        """Returns the search tool's string paging argument, a non-enum key named like cursor, token, bookmark, next,
        page or offset, or None.
        """
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        for key, pspec in (props if isinstance(props, dict) else {}).items():
            if isinstance(pspec, dict) and pspec.get("type") == "string" and not pspec.get("enum") \
                    and re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(key).lower()):
                return key
        return None

    @staticmethod
    def next_token(obs: Any) -> str | None:
        """Returns the first non-empty string in an observation whose key looks like a next-page marker (next, cursor,
        token, bookmark, more), or None.
        """
        if not isinstance(obs, dict):
            return None
        for key, val in obs.items():
            if isinstance(val, str) and val.strip() and re.search(r"next|onward|cursor|bookmark|token|continu|more", str(key).lower()):
                return val
        return None

    def page_arg(self) -> tuple[str, Any, Any, float | None] | None:
        """Describes how the search tool pages. Returns (key, by_page, low, high) for the first numeric page or offset
        argument, where by_page tells page numbers from offsets, low comes from the schema minimum or default and high
        from its maximum. Falls back to (token key, token, empty string, None) for cursor paging, or None.
        """
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
        tok = self.token_key()
        return (tok, "token", "", None) if tok else None

    @staticmethod
    def stem(word: str) -> str:
        """Drops a plural s from words longer than three letters, leaving words ending in ss alone."""
        return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word

    def shelf_rows(self) -> dict[str, tuple[float, str, str]]:
        """Collects priced, in-stock rows on the current shelf labels for the cheapest sweep, as id to (price, title,
        maker). Refused rows and rows an observation shows sold out are dropped. Prices are replaced by the archive's
        latest price, and rows over the spend cap at the requested quantity are removed. kind_queries, the ceiling
        sweep and the sweep's closing summary of the cheapest rows read from it.
        """
        shelves, cap, qty = self.shelf_labels(), self.needs.spend_cap, self.needs.quantity()
        found: dict[str, tuple[float, str, str]] = {}
        if not shelves:
            return found

        def visit(d: dict[str, Any]) -> None:
            """Reads one observed dict: for a known listing on a current shelf that is not refused, it records the price, a
            title (the title field or another name-like field that is not a category, brand or id) and the maker words, or
            removes the listing when the dict shows it sold out.
            """
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.archive.ids), None)
            if not ident or ident in self.refused_lines or self.shelf_of(d) not in shelves:
                return
            if self.sold_out(d):
                found.pop(ident, None)
                return
            cost = next((val for key, val in d.items() if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower())), None)
            title = d.get("title") if isinstance(d.get("title"), str) and 0 < len(d["title"]) <= 300 else next(
                (val for key, val in d.items() if isinstance(val, str) and 0 < len(val) <= 300 and re.search(r"title|heading|headline|name", str(key).lower())
                 and not re.search(r"categor|shelf|brand|make|manufactur|vendor|shop|seller|store|id", str(key).lower())), "")
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
        """Turns the current shelf labels into short search texts. For each label it splits the words into runs broken by
        linking words or symbols and yields the first two words together, the last word of each run, then the other
        words. Returns them in order without duplicates. sweep_next uses these texts as one of its rotating sources of
        sweep searches, beside the title-word queries.
        """
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
        """Derives search texts from the titles of in-budget shelf rows. It takes single words, or adjacent word pairs
        when pairs is set, of three or more letters that are not linking or maker words and not already all in the
        shelf label, and counts the titles holding each. Returns those found in two or more titles, ordered by the
        cheapest title price, then by frequency.
        """
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
        """Picks the next search call of the cheapest-item shelf sweep.
        Nothing without a paging argument, shelf labels or booked searches.
        A ready ceiling sweep call comes first.
        Else the next page of the largest id-ordered listing, then queries from
        title word pairs, title words and shelf words on the latest search template.
        Calls already tried are skipped.
        """
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
                """True when a template search argument stays as is: non-strings, enum values, shelf labels and the paging key."""
                return not isinstance(v, str) or bool((props.get(k) or {}).get("enum")) or v.strip().lower() in shelves or k == key
            texts = [k for k, v in args.items() if not fixed(k, v)]
            text_key = self.lead_ctx[0] if self.lead_ctx and self.lead_ctx[0] in texts else max(texts, key=lambda k: len(args[k].split()), default=None)
            template = {k: v for k, v in args.items() if k != text_key and fixed(k, v)} if text_key else None
        done = {tuple(self.stem(w) for w in re.findall(r"[a-z0-9]{2,}", " ".join(self.free_text(w["args"])).lower())[:2])
                for w in book.values() if low in w["read"] and w.get("id_order") is not True}

        def query(text: str) -> dict[str, Any] | None:
            """Builds a sweep search from the template with the given text, or None when there is no template, the text's
            first two stemmed words were already searched, or the same call was tried.
            """
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
        ceil, ended = self.ceiling_sweep(tried)
        if ceil is not None:
            return ceil, True
        if ended:
            return None, False
        for pick in [0] + [1 + (turn + shift) % 3 for shift in range(3)]:
            picks = [a for a in sources[pick]() if json.dumps(a, sort_keys=True, default=str) not in tried]
            if picks:
                return picks[0], pick == 0
        return None, False

    def sweep_step(self) -> bool:
        """Takes one step of the cheapest shelf sweep when the cheapest is asked, some requirement was let go and none is
        left open. It stops after SWEEP_READS steps or when moves or seconds run short, keeping room for the first
        order. It sends the call from sweep_next as a scripted search with a short explanation, records it in
        sweep_tried, and advances sweep_turn unless it was a listing page. When no further step is possible, it
        appends the six cheapest in-stock rows within budget to the last reply with advice to cart and order the
        lowest-priced one. Returns True when a step ran.
        """
        nd = self.needs
        if nd.priority() != "cheapest" or not nd.let_go or nd.props_left() or not self.role("search") or self.wire.done:
            return False
        keep = 3 + (1 if self.trolley.units else 0)
        if self.orders_tried == 0:
            keep = max(keep + 3, FIRST_ORDER_BY)
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

    def free_text(self, call_args: dict[str, Any]) -> list[str]:
        """Returns the free-text values of a search call: string arguments that are neither enum values nor the paging
        token.
        """
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        props = props if isinstance(props, dict) else {}
        tok = self.token_key()
        return [val for key, val in call_args.items() if isinstance(val, str) and not (props.get(key) or {}).get("enum") and key != tok]

    def book_page(self, call_args: dict[str, Any], rows: list, total: int, obs: Any = None) -> None:
        """Records a search result page in page_book, keyed by the call's arguments without the paging key. It notes the
        page read (number, offset or token, plus the next token in token mode), the largest page size and total, the
        shelves shown, each row's shelf, the first page's row ids and, when page one holds every match, all rows. With
        five or more rows it tracks whether the store returns rows in id order; one unsorted page clears that flag for
        good.
        """
        found = self.page_arg()
        if not found or not rows:
            return
        key, mode, low, _ = found
        at = call_args.get(key, low)
        if mode == "token":
            at = at.strip() if isinstance(at, str) else ""
        elif not is_number(at):
            return
        base = {k: v for k, v in call_args.items() if k != key}
        walk = self.__dict__.setdefault("page_book", {}).setdefault(
            json.dumps(base, sort_keys=True, default=str), {"args": base, "read": set(), "size": 0, "total": 0, "shelves": set()})
        walk["read"].add(at)
        if mode == "token":
            walk.setdefault("tokens", {})[at] = self.next_token(obs)
        walk["size"] = max(walk["size"], len(rows))
        walk["total"] = max(walk["total"], total)
        walk["shelves"] |= {self.shelf_of(r) for r in rows if isinstance(r, dict)} - {""}
        shown = walk.setdefault("row_shelf", {})
        for r in rows:
            if isinstance(r, dict):
                shown.setdefault(self.row_mark(r), self.shelf_of(r))
        if at == low:
            walk["first"] = tuple(self.row_mark(r) for r in rows if isinstance(r, dict))
        if at == low and total <= len(rows):
            walk["every"] = [r for r in rows if isinstance(r, dict)]
        marks = [self.row_mark(r) for r in rows if isinstance(r, dict)]
        if len(marks) >= 5:
            walk["id_order"] = walk.get("id_order", True) is not False and (marks == sorted(marks) or marks == sorted(
                marks, key=lambda m: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", m)]))

    def next_page(self, walk: dict[str, Any]) -> Any:
        """Returns the next unread page value of a booked search, or None. In token mode it is a saved next token not yet
        read; otherwise it steps from the lowest read value by one page or one page size and stops at the schema
        maximum or once the reported total is covered. The fit walk, cheapest walk, shelf sweep and pre-decline reads
        all step their searches with it.
        """
        key, by_page, low, high = self.page_arg()
        if by_page == "token":
            for nxt in (walk.get("tokens") or {}).values():
                if isinstance(nxt, str) and nxt not in walk["read"]:
                    return nxt
            return None
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
        """Counts the open non-reference requirements whose value words share a prefix with a word of the search call's
        free text, looking only at the leading words the store uses.
        """
        def near(a: str, b: str) -> bool:
            """True when two words of three or more letters start one with the other."""
            return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))
        vals = [re.findall(r"[a-z0-9]{3,}", req.partition(":")[2].lower()) for req in self.needs.props_left() if not ref_text(req.partition(":")[2])]
        words = [w for text in self.free_text(call_args) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
        words = words[: self.lead_words] if self.lead_words else words
        return sum(1 for value_words in vals if any(near(w, v) for w in words for v in value_words))

    def row_hits(self, rows: list, words: list[str] | None = None) -> list[str]:
        """Returns the unread known rows of a result worth opening: within the cap at the requested quantity, not sold
        out, and on the current shelves when both are known; their shelf goes into hit_shelf. When the request has
        references, unread rows whose seller or brand words were all in the query and among pointed_words are also
        added to ref_hits.
        """
        cap, qty = self.needs.spend_cap, self.needs.quantity()
        shelves = self.shelf_labels()
        hits = []
        asked = {w for t in words or [] for w in re.findall(r"[a-z0-9]+", str(t).lower())} if self.needs.reading.get("references") else set()
        pointers = self.__dict__.setdefault("ref_hits", [])
        for r in rows:
            if not isinstance(r, dict):
                continue
            ident = self.row_mark(r)
            makers = [set(re.findall(r"[a-z0-9]+", val.lower())) for k2, val in r.items() if isinstance(val, str) and val.strip()
                      and re.search(r"shop|seller|store|vendor|brand|make|manufactur", str(k2).lower()) and "id" not in str(k2).lower()]
            if asked and ident in self.archive.ids and ident not in self.archive.read \
                    and any(m and m <= asked and m <= self.pointed_words() for m in makers):
                if ident not in pointers:
                    pointers.append(ident)
            cost = next((val for k2, val in r.items() if is_number(val) and re.search(r"price|cost", str(k2).lower())), None)
            shelf = self.shelf_of(r)
            if ident in self.archive.ids and ident not in self.archive.read and not (is_number(cap) and is_number(cost) and cost * qty > cap) \
                    and not self.sold_out(r) and not (shelves and shelf and shelf not in shelves):
                hits.append(ident)
                self.__dict__.setdefault("hit_shelf", {})[ident] = shelf
        return hits

    def pointed_words(self) -> set[str]:
        """Gathers the words that may point to a referenced seller: the words of the request's references plus the words
        of read rows' text fields. Each row's own seller, shop or brand words are left out of its text words, and
        title-like fields are skipped. row_hits uses the set to tell when a listed seller was named in a reference or
        in another listing. The result is memoised on the number of stored observations; an error keeps the words
        gathered so far.
        """
        memo = self.__dict__.get("pointed_memo")
        if memo and memo[0] == len(self.archive.items):
            return memo[1]
        words: set[str] = set()
        try:
            for ref in self.needs.reading.get("references") or []:
                words |= set(re.findall(r"[a-z0-9]+", str(ref).lower()))

            def visit(d: dict[str, Any]) -> None:
                """Handles one read row: short values under seller, shop, brand or maker keys (not note or text keys) form the
                row's own seller words, and other non-title strings are its text. The text words, minus the row's own seller
                words, join the pointed set. Rows not yet read are ignored, and seller values over 40 characters count as text
                rather than as the row's own seller.
                """
                if self.row_mark(d) not in self.archive.read:
                    return
                own: set[str] = set()
                texts: list[str] = []
                for key, val in d.items():
                    if not isinstance(val, str):
                        continue
                    low = str(key).lower()
                    if re.search(r"shop|seller|store|vendor|brand|make|manufactur", low) and "id" not in low and len(val) <= 40 \
                            and not re.search(r"note|text|desc|comment|detail|info|message|section", low):
                        own |= set(re.findall(r"[a-z0-9]+", val.lower()))
                    elif not re.search(r"title|heading|headline|name|label|caption", low):
                        texts.append(val.lower())
                for text in texts:
                    words.update(set(re.findall(r"[a-z0-9]+", text)) - own)
            for rec in self.archive.items:
                walk_dicts(rec, visit)
        except Exception:
            pass
        self.pointed_memo = (len(self.archive.items), words)
        return words

    def open_pages(self, most: int = 2) -> list[dict[str, Any]]:
        """Suggests up to most next-page searches worth reading: booked searches that still have an unread page, lie on
        the current shelves when both are known, and carry at least one open requirement value. They are ordered by
        value score, then by smaller total, and returned as call arguments with the paging key set. Without a paging
        argument the list is empty.
        """
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
                picks.append((-score, walk["total"], {**walk["args"], found[0]: nxt}))
        picks.sort(key=lambda p: (p[0], p[1]))
        return [args for _, _, args in picks[:most]]

    def value_probe(self, raw: list[str]) -> tuple[str, str]:
        """Spots a query whose leading words name a required property instead of its value, and proposes a probe search.
        It returns empty strings for any-word stores, when a leading word already matches a value or the product, or
        when an equivalent query was run. Otherwise it returns the property name and a probe text: the requirement's
        first value (from its first digit on, if any) followed by up to three words of the product.
        """
        if self.word_kind == "any":
            return "", ""
        lead = [w for w in re.findall(r"[a-z0-9]+", " ".join(raw).lower()) if len(w) >= 2][: self.lead_words or 2]
        product = " ".join(re.findall(r"[a-z0-9]+", str(self.needs.reading.get("product") or "").lower())[:3])

        def near(a: str, b: str) -> bool:
            """Matches two tokens: with digits, equal once leading zeros go, or one number starting the other; otherwise a
            shared prefix of three or more letters.
            """
            if any(c.isdigit() for c in a + b):
                return a.lstrip("0") == b.lstrip("0") or (a.isdigit() and b.startswith(a) and not b[len(a):len(a) + 1].isdigit())
            return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))
        reqs = [(req.partition(":")[0].strip(), re.split(r"[,;]", re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]))[0].strip().lower())
                for req in self.needs.props_left() if ":" in req and not ref_text(req.partition(":")[2])]
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

    def ran_query(self, text: str) -> bool:
        """True when some booked search has the same leading words as the text, compared stemmed and sorted over the
        store's lead width or three words.
        """
        width = self.lead_words or 3
        mine = sorted(self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[:width])
        for walk in self.__dict__.get("page_book", {}).values():
            ran = sorted(self.stem(w) for w in re.findall(r"[a-z0-9]+", " ".join(self.free_text(walk["args"])).lower())[:width])
            if ran == mine:
                return True
        return False

    def search_template(self) -> tuple[dict[str, Any], str] | None:
        """Returns a reusable free-text search shape from the latest booked search that is not a category listing and has
        exactly one free-text argument, as (other arguments, text key). Returns None when no booked search qualifies.
        Line searches and combined-value searches put their own text into this shape.
        """
        for walk in reversed(list(self.__dict__.get("page_book", {}).values())):
            args = walk["args"]
            if self.cg_listing(args):
                continue
            texts = self.free_text(args)
            keys = [k for k, v in args.items() if isinstance(v, str) and v in texts and v.strip()]
            if len(texts) == 1 and len(keys) == 1:
                return {k: v for k, v in args.items() if k != keys[0]}, keys[0]
        return None

    def word_trap(self, raw: list[str]) -> tuple[str, str]:
        """Detects a query word that narrows an all-words search for the wrong reason.
        Skipped for any-word stores.
        A leading word that matches a property name, not a value, is flagged.
        If the other words still carry a value, builds a replacement query
        and returns a note with it, unless it already ran.
        Returns empty strings otherwise.
        """
        try:
            if self.word_kind == "any":
                return "", ""
            toks = [t.strip("\"'()[]{},.;!?|").lower() for t in " ".join(raw).split()]
            toks = [t for t in toks if re.search(r"[a-z0-9]", t)]
            width = self.lead_words or (max(3, len(toks)) if self.word_kind == "all" else 3)
            toks = toks[:width]

            def near(a: str, b: str) -> bool:
                """Compares two tokens: numbers when equal without leading zeros, words when one starts the other with at least
                three letters.
                """
                if any(c.isdigit() for c in a + b):
                    return a.lstrip("0") == b.lstrip("0")
                return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))

            def parts(t: str) -> list[str]:
                """Splits a token into its alphanumeric pieces of two or more characters."""
                return [p for p in re.findall(r"[a-z0-9]+", t) if len(p) >= 2]
            reqs = [(re.split(r"\s*\(", req.partition(":")[0])[0].lower(), req.partition(":")[0],
                     re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()) for req in self.needs.props_left()
                    if ":" in req and not ref_text(req.partition(":")[2])]
            vals = {w for _, _, v in reqs for w in re.findall(r"[a-z0-9]{2,}", v)}
            products = [r["product"] for r in self.needs.items()] or [str(self.needs.reading.get("product") or "")]
            kinds = set(re.findall(r"[a-z0-9]{3,}", " ".join(products + sorted(self.shelf_labels())).lower())) - LINK_WORDS
            names = {w for n, _, _ in reqs for w in re.findall(r"[a-z]{3,}", n)} - LINK_WORDS
            dropped = {w for old in self.needs.let_go for w in re.findall(r"[a-z]{3,}", old.partition(":")[2].lower())} - LINK_WORDS
            flagged, why = [], ""
            for t in toks:
                ps = parts(t)
                if not ps or any(near(p, k) for p in ps for k in vals | kinds):
                    continue
                if any(near(p, n) for p in ps for n in names):
                    flagged.append(t)
                    why = why or "name"
                elif any(near(p, d) for p in ps for d in dropped):
                    flagged.append(t)
                    why = why or "dropped"
            if not flagged:
                return "", ""
            rest = [t for t in toks if t not in flagged]
            held = [(tag, v) for _, tag, v in reqs if any(near(p, w) for t in rest for p in parts(t) for w in re.findall(r"[a-z0-9]{2,}", v))]
            if not held:
                return "", ""
            if not any(near(p, k) for t in rest for p in parts(t) for k in kinds):
                k = self.product_of(held[0][0]) if self.needs.items() else None
                name = products[k] if k is not None and k < len(products) else products[0]
                kind = [w for w in re.findall(r"[a-z]{3,}", name.lower()) if w not in LINK_WORDS][-1:]
                rest += kind
            query = " ".join(rest[:width])
            if not query or self.ran_query(query):
                return "", ""
            note = (f"This query's word '{flagged[0]}' is a word of a required property's NAME, not of a required value: every word is "
                    "required here, so it keeps only the listings whose fields or title repeat that name and misses those whose field "
                    "holds just the value." if why == "name" else
                    f"This query's word '{flagged[0]}' belongs only to a requirement the shopper DROPPED: it still keeps only the "
                    "listings carrying it, though they no longer need to.")
            return note + f" Also search '{query}' and read every page of it.", query
        except Exception as err:
            self.log.append({"kind": "trap_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return "", ""

    def combined_queries(self) -> list[dict[str, Any]]:
        """Builds up to two searches that join several required values in one query, for products whose shelf has no
        passed listing yet (or, for one item, while the pick is unverified). From each open requirement of that
        product it takes the first alternative of a plain value of one or two words, skipping numbers, negations and
        page or reference wording. With at least two such values it packs them into the store's lead width (three
        words by default), adds the product's kind word when room remains, and puts the text on the search template
        unless that query already ran.
        """
        template = self.search_template()
        if not template:
            return []
        rows = self.needs.items()
        width = self.lead_words or 3
        live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == str(self.needs.version)} - self.refused_lines
        got = {self.shelf_known(i) for i in live} - {""}
        homes = self.home_shelves() if rows else []
        out: list[dict[str, Any]] = []
        for k in (range(len(rows)) if rows else [None]):
            if rows and (k >= len(homes) or not homes[k] or homes[k] & got):
                continue
            if not rows and not self.unverified():
                continue
            vals: list[list[str]] = []
            for req in self.needs.props_left():
                head, _, value = req.partition(":")
                if rows and self.product_of(head) not in (k, None):
                    continue
                value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip().lower()
                if not value or re.search(r"\d|as shown|same as|page|redirect|refer|\bnot\b|\bno\b|without|except|anything|other than", value):
                    continue
                first = re.split(r"[,;/]|\bor\b|\band\b", value)[0]
                words = [w for w in re.findall(r"[a-z]{2,}", first) if w not in LINK_WORDS]
                if 1 <= len(words) <= 2:
                    vals.append(words)
            if len(vals) < 2:
                continue
            query: list[str] = []
            for words in vals:
                if len(query) + len(words) <= width:
                    query += words
            product = rows[k]["product"] if rows else str(self.needs.reading.get("product") or "")
            kind = [w for w in re.findall(r"[a-z]{3,}", product.lower()) if w not in LINK_WORDS][-1:]
            if kind and len(query) < width and kind[0] not in query:
                query += kind
            text = " ".join(query)
            if len(query) >= 2 and not self.ran_query(text):
                out.append({**template[0], template[1]: text})
        return out[:2]

    def arbitrate(self, thing: str, ruling: str) -> str:
        """Settles a clash between the decline audit, which found the item fits, and an earlier failed check. It reruns
        check_item for the order with a note giving both readings and how to weigh them. A new failure text is
        returned; when the second check passes, the stored rulings for the item at this needs version are cleared and
        an empty string is returned. On error the original ruling is kept.
        """
        try:
            reason = next((str((r.get("result") or {}).get("reason") or "") for r in reversed(self.log)
                           if isinstance(r, dict) and r.get("kind") == "audit" and isinstance(r.get("result"), dict)), "")
            note = ("\n\nTWO READINGS DISAGREE ABOUT THIS ITEM. The decline audit, reading the same records, found that it meets every "
                    "requirement" + (f" ({reason[:400]})" if reason else "") + "; an earlier check found: " + ruling[:500]
                    + "\nDecide from the records which reading is right: a field that states the required value in other words, or a "
                    "shade, qualifier or spelling of the same named value, meets the requirement; a field that states a different "
                    "value fails it.")
            again = self.check_item(thing, "ORDER", second=note)
            if again:
                return again
            key = f"{thing}|{self.needs.version}"
            for memo in [k for k in self.rulings if k == key or k.startswith(key + "|")]:
                self.rulings[memo] = ""
            return ""
        except Exception as err:
            self.log.append({"kind": "arbitrate_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ruling

    def unrefuted(self) -> list[str]:
        """Returns one read listing per product that nothing has ruled out, or an empty list. Candidates must sit on the
        product's home shelf and have a price; rows failed at this needs version, rejected by the decline audit,
        refused, sold out, already picked, reference pointers or over the cap at the product's quantity are skipped.
        Listings passed at this version come first, then the cheapest. It gives up when some product lacks a home
        shelf or a candidate, or when the total breaks the spend cap.
        """
        try:
            rows = self.needs.items()
            homes = self.home_shelves() if rows else [set(self.shelf_labels())]
            if not homes or any(not home for home in homes):
                return []
            ver = str(self.needs.version)
            failed = {k.split("|")[0] for k, v in self.rulings.items() if v and k.split("|")[1:2] == [ver]}
            live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == ver}
            reviewed = self.__dict__.get("audit_ids")
            if self.__dict__.get("audit_done") and (reviewed is None or len(self.archive.read) <= self.__dict__.get("audit_reads", -1)):
                reviewed = set(self.archive.read)
            reviewed = set(reviewed or set())
            audited = reviewed - set(self.__dict__.get("audit_fits") or [])
            sold: dict[str, bool] = {}

            def visit(d: dict[str, Any]) -> None:
                """Records, for each known listing in a dict with a stock or availability key, whether that dict shows it sold
                out.
                """
                ident = self.row_mark(d)
                if ident in self.archive.ids and any(re.search(r"stock|avail", str(key).lower()) for key in d):
                    sold[ident] = self.sold_out(d)
            for rec in self.archive.items:
                walk_dicts(rec, visit)
            cap = self.needs.spend_cap
            picks: list[str] = []
            total = 0.0
            pointers = self.ref_pointers()
            for k, home in enumerate(homes):
                qty = rows[k]["quantity"] if rows else self.needs.quantity()
                pool = []
                for ident in sorted(self.archive.read):
                    if ident in failed or ident in audited or ident in self.refused_lines or sold.get(ident) or ident in picks \
                            or ident in pointers or self.shelf_known(ident) not in home:
                        continue
                    cost = self.archive.price(ident)
                    if not is_number(cost) or (is_number(cap) and cost * qty > cap):
                        continue
                    pool.append((ident not in live, cost, ident))
                if not pool:
                    return []
                _, cost, ident = min(pool)
                picks.append(ident)
                total += cost * qty
            if is_number(cap) and total > cap:
                return []
            return picks
        except Exception as err:
            self.log.append({"kind": "unrefuted_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

    def fit_hold(self) -> str:
        """Holds back a decline once when the decline audit named possible fits and the cart is empty. For up to two
        audit fits that were read it runs check_item for the order, arbitrating a failure when time allows, and keeps
        those that pass at the current needs version. With several products every home shelf must already hold a
        passed listing. Returns a note naming the passing listings and telling the model to cart, re-read and order
        them, or an empty string; sets fit_held.
        """
        try:
            if self.__dict__.get("fit_held") or self.trolley.units or self.moves_left() <= 4 or self.wire.spare() < CLOSING + 30:
                return ""
            fits = [i for i in self.__dict__.get("audit_fits") or [] if i in self.archive.ids and i not in self.refused_lines]
            if not fits:
                return ""
            self.fit_held = True
            good = []
            for thing in fits[:2]:
                if thing not in self.archive.read:
                    continue
                ruling = self.check_item(thing, "ORDER")
                if ruling and self.wire.spare() > CLOSING + 30:
                    ruling = self.arbitrate(thing, ruling)
                if not ruling and f"{thing}|{self.needs.version}" in self.passed:
                    good.append(thing)
                    if not self.needs.items():
                        break
            if not good:
                return ""
            if self.needs.items():
                live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == str(self.needs.version)} - self.refused_lines
                got = {self.shelf_known(i) for i in live} - {""}
                if any(home and not home & got for home in self.home_shelves()):
                    return ""
            return ("The decline audit listed " + ", ".join(good) + " as a possible fit, and the independent item check, reading "
                    + ("its" if len(good) == 1 else "their") + " full record, found " + ("it meets" if len(good) == 1 else "they meet")
                    + " every requirement now in force. Weigh " + ("it" if len(good) == 1 else "them") + " before declining: if it fits, put "
                    + ("a verified line of every product" if self.needs.items() else good[0]) + " in the cart, re-read its stock and price, "
                    "and order; decline if you find that it fails a requirement in force.")
        except Exception as err:
            self.log.append({"kind": "fit_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def id_store(self) -> bool:
        """True when at least two booked searches came back in id order and none came back out of order."""
        orders = [w.get("id_order") for w in self.__dict__.get("page_book", {}).values() if "id_order" in w]
        return orders.count(True) >= 2 and False not in orders

    def product_index(self, thing: str) -> int | None:
        """Maps a listing to the index of the product whose home shelves hold its shelf; None for a single-item request
        or when no home matches.
        """
        if not self.needs.items():
            return None
        shelf = self.shelf_known(thing)
        for k, home in enumerate(self.home_shelves()):
            if shelf and shelf in home:
                return k
        return None

    def walk_home(self, k: int | None) -> set[str]:
        """Returns the home shelves of product k (the first one when k is None), falling back to the current shelf labels
        for a single item.
        """
        homes = self.home_shelves()
        home = set(homes[k if k is not None else 0]) if homes and (k or 0) < len(homes) else set()
        return home or (set(self.shelf_labels()) if k is None else set())

    @staticmethod
    def fig_amount(text: Any) -> tuple[str, str] | None:
        """Parses a value part that starts with a whole number and a word into (number, word); None when the part holds a
        bound phrase such as at least or up to, or does not start that way.
        """
        low = str(text).lower()
        if re.search(CH_BOUND, low):
            return None
        found = re.match(r"\s*(\d+)\s+([a-z]+)", low)
        return (found.group(1), found.group(2)) if found else None

    @staticmethod
    def amount_part(text: Any) -> bool:
        """True when a value part holds a digit and either a bound phrase or a leading number followed by a word."""
        low = str(text).lower()
        return bool(re.search(r"\d", low)) and bool(re.search(CH_BOUND, low) or re.match(r"\s*\d+(?:[.,]\d+)?\s+[a-z]", low))

    def walk_carries(self, args: dict[str, Any], k: int | None, alts: bool = False) -> set[Any]:
        """Tells which open requirements a search call carries in its leading words. For each requirement (of product k
        when there are several) it splits the value into alternatives: a number-and-unit alternative counts when both
        appear among the query tokens, any other non-amount alternative when a query word shares a prefix with one of
        its words. Returns requirement indexes, or (index, alternative) pairs when alts is set.
        """
        def near(a: str, b: str) -> bool:
            """Prefix match between a query word and a value word, both at least three letters long."""
            return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))
        several = bool(self.needs.items()) and k is not None
        words = [w for text in self.free_text(args) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
        words = words[: self.lead_words] if self.lead_words else words
        toks = [w for text in self.free_text(args) for w in re.findall(r"[a-z0-9]+", text.lower())]
        toks = toks[: self.lead_words] if self.lead_words else toks
        out: set[Any] = set()
        for n, req in enumerate(self.needs.props_left()):
            if several and self.product_of(req.partition(":")[0]) not in (k, None):
                continue
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()
            for a, part in enumerate(re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", value)):
                fig = self.fig_amount(part)
                if fig:
                    if fig[0] in toks and any(w == fig[1] or near(w, fig[1]) for w in toks):
                        out.add((n, a) if alts else n)
                    continue
                if not self.amount_part(part) and any(near(w, v) for w in words for v in re.findall(r"[a-z0-9]{3,}", part)):
                    out.add((n, a) if alts else n)
        return out

    def walk_score(self, args: dict[str, Any], k: int | None) -> int:
        """Counts the open requirements a search call carries for product k."""
        return len(self.walk_carries(args, k))

    def walk_options(self, k: int | None) -> list[tuple[int, str, dict[str, Any]]]:
        """Booked searches worth paging for product k in the cheapest walk, as
        (pages left, key, walk) sorted by pages, then requirements carried.
        Each needs store id order, a read first page, an unread page, rows
        on k's home shelves, a requirement. Skips ones covered by a fully
        read id-ordered same-mode search or needing > 2x WALK_PAGES pages;
        on equal total and first page, keeps fewest pages left. All-words
        store: prunes options a broader or narrower one makes redundant.
        """
        home = self.walk_home(k)
        if not home:
            return []
        cut = self.lead_words or None

        def lead(walk: dict[str, Any]) -> set[str]:
            """Returns the stemmed leading words of a booked search's free text, cut at the store's lead width."""
            return {self.stem(w) for w in re.findall(r"[a-z0-9]+", " ".join(self.free_text(walk["args"])).lower())[:cut]}
        label = {self.stem(w) for s in home for w in re.findall(r"[a-z0-9]{2,}", s)}
        several = bool(self.needs.items()) and k is not None
        values = {self.stem(w) for req in self.needs.props_left() if not several or self.product_of(req.partition(":")[0]) in (k, None)
                  for w in re.findall(r"[a-z0-9]{2,}", req.partition(":")[2].lower())}
        book = self.__dict__.get("page_book", {})
        done = [(lead(w), self.walk_carries(w["args"], k, alts=True), self.vl_mode(w["args"])) for w in book.values()
                if w.get("id_order") is True and w.get("first") and self.next_page(w) is None and lead(w)
                and len(w.get("row_shelf") or {}) >= w["total"]]
        best: dict[tuple, tuple[int, str, dict[str, Any]]] = {}
        for key, walk in book.items():
            if walk.get("id_order") is not True or not walk.get("first") or self.next_page(walk) is None:
                continue
            shown = walk.get("row_shelf") or {}
            carries = self.walk_carries(walk["args"], k)
            if not any(s in home for s in shown.values()) or not carries:
                continue
            words = lead(walk)
            pairs = self.walk_carries(walk["args"], k, alts=True)
            how = self.vl_mode(walk["args"])
            if any(mode == how and (seen <= words or (seen <= values | label and pairs <= got)) for seen, got, mode in done):
                continue
            pages = -(-walk["total"] // max(1, walk["size"])) - len(walk["read"])
            twin = (walk["total"], walk["first"])
            if 0 < pages <= 2 * WALK_PAGES and (twin not in best or pages < best[twin][0]):
                best[twin] = (pages, key, walk)
        opts = list(best.values())
        if self.all_words():
            def nested(p: tuple, o: tuple) -> bool:
                """Option o is redundant next to p when p's words are a strict subset of o's with some extra word that is not a
                value or label word (and p fits WALK_PAGES), or o's are a strict subset of p's with only such words extra.
                """
                a, b = lead(p[2]), lead(o[2])
                return (a < b and not (b - a) <= values | label and p[0] <= WALK_PAGES) or (b < a and (a - b) <= values | label)
            opts = [o for o in opts if not any(nested(p, o) for p in opts if p is not o)]
        return sorted(opts, key=lambda p: (p[0], -self.walk_score(p[2]["args"], k), p[1]))

    def walk_plan(self, budget: int, only: set | None = None) -> list[tuple[int | None, str, int]]:
        """Plans the cheapest walk within a page budget, only when the cheapest is asked, search and paging exist and the
        store lists results in id order. For each product not yet walked (optionally limited to only) it takes the
        best walk_options entry, then gives pages to products with the fewest pages first, at most WALK_PAGES each,
        until the budget is spent. Returns (product, search key, pages) tuples.
        """
        if budget <= 0 or self.needs.priority() != "cheapest" or not self.role("search") or not self.page_arg() or not self.id_store():
            return []
        done = self.__dict__.get("walked", set())
        wants = []
        for k in (list(range(len(self.needs.items()))) or [None]):
            if k in done or (only is not None and k not in only):
                continue
            opts = self.walk_options(k)
            if opts:
                wants.append((opts[0][0], -1 if k is None else k, k, opts[0][1]))
        plan, left = [], budget
        for pages, _, k, key in sorted(wants, key=lambda p: (p[0], p[1])):
            n = min(pages, WALK_PAGES, left)
            if n <= 0:
                break
            plan.append((k, key, n))
            left -= n
        return plan

    def walk_pick(self, k: int | None) -> tuple[str, float] | None:
        """Returns the cheapest current pick for product k as (id, price): a cart line of that product that is not dead,
        or else a listing passed at the current needs version that is neither refused nor dead. None when no candidate
        has a price.
        """
        gone = self.dead_lines()
        cands = [i for i in self.trolley.units if (k is None or self.product_index(i) == k) and i not in gone]
        if not cands:
            cands = [key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == str(self.needs.version)]
            cands = [i for i in cands if i not in self.refused_lines and i not in gone and (k is None or self.product_index(i) == k)]
        priced = [(i, self.archive.price(i)) for i in cands]
        priced = [(i, float(p)) for i, p in priced if is_number(p)]
        return min(priced, key=lambda ip: (ip[1], ip[0])) if priced else None

    def line_query(self, k: int | None) -> dict[str, Any] | None:
        """Builds a search for the product line of product k's current pick: the maker's first word plus the first title
        word after the maker that is not a maker, linking, shelf, requirement or product word. Without a maker it
        hands over to title_line. Returns None without a pick, a search template or a usable word, or when the query
        already ran.
        """
        pick = self.walk_pick(k)
        template = self.search_template()
        if not pick or not template:
            return None
        rec: dict[str, Any] = {}

        def visit(d: dict[str, Any]) -> None:
            """Keeps the observed dict that belongs to the pick and has a title as the pick's record."""
            nonlocal rec
            if self.row_mark(d) == pick[0] and self.title_of(d):
                rec = d
        for item in self.archive.items:
            walk_dicts(item, visit)
        maker = next((v for key, v in rec.items() if isinstance(v, str) and v.strip() and re.search(r"brand|make|manufactur", str(key).lower())
                      and "id" not in str(key).lower()), "")
        maker_words = re.findall(r"[a-z0-9]+", maker.lower())
        if not maker_words:
            return self.title_line(k, rec, template)
        avoid = set(maker_words) | LINK_WORDS | {w for s in self.walk_home(k) for w in re.findall(r"[a-z]+", s)}
        avoid |= {w for req in self.needs.props_left() for w in re.findall(r"[a-z]+", req.lower())}
        avoid |= {w for r in (self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}]) for w in re.findall(r"[a-z]+", r["product"].lower())}
        title_words = re.findall(r"[a-z][a-z'-]{3,}", self.title_of(rec).lower())
        after = title_words[title_words.index(maker_words[0]) + 1:] if maker_words[0] in title_words else title_words
        words = [w for w in after if w.strip("'-") not in avoid and "'" not in w]
        if not words:
            return None
        text = f"{maker_words[0]} {words[0]}"
        if self.ran_query(text):
            return None
        return {**template[0], template[1]: text}

    def walk_rows(self, k: int | None, keys: list[str], most: int) -> list[str]:
        """Lists up to most unread listings from walked searches to open next for product k, cheapest first. Rows come
        from the id-ordered searches that carry the most requirements and from the searches named in keys, limited to
        the product's home shelves. A listing is kept when it has a price, is not read, refused or sold out, fits the
        cap at the product's quantity and costs less than the current pick (walk_pick, or else the cheapest read row
        among them).
        """
        home = self.walk_home(k)
        book = self.__dict__.get("page_book", {})
        top = max([len(self.walk_carries(w["args"], k)) for w in book.values() if w.get("id_order") is True] or [0])
        ids: set[str] = set()
        for key, walk in book.items():
            got = self.walk_carries(walk["args"], k) if walk.get("id_order") is True else set()
            if (got and len(got) == top) or key in keys:
                ids |= {i for i, s in (walk.get("row_shelf") or {}).items() if s in home}
        pick = self.walk_pick(k)
        if not pick:
            seen = sorted((float(self.archive.price(i)), i) for i in ids if i in self.archive.read and is_number(self.archive.price(i)))
            pick = (seen[0][1], seen[0][0]) if seen else None
        qty = self.needs.items()[k]["quantity"] if k is not None and k < len(self.needs.items()) else self.needs.quantity()
        cap = self.needs.spend_cap
        sold: dict[str, bool] = {}

        def visit(d: dict[str, Any]) -> None:
            """Records the sold-out state of any dict whose id field names one of the collected listings."""
            if any(isinstance(v, str) and v in ids and "id" in str(key).lower() for key, v in d.items()):
                sold[self.row_mark(d)] = self.sold_out(d)
        for item in self.archive.items:
            walk_dicts(item, visit)
        out = []
        for ident in ids:
            cost = self.archive.price(ident)
            if ident in self.archive.read or ident in self.refused_lines or sold.get(ident) or not is_number(cost):
                continue
            if (is_number(cap) and cost * qty > cap) or (pick and cost >= pick[1]):
                continue
            out.append((cost, ident))
        return [ident for _, ident in sorted(out)[:most]]

    def page_digest(self, obs: Any) -> str:
        """Summarises a search reply in one line for the chat: up to twelve row dicts (four or more keys and a known
        listing id) shown as id, price or a question mark, a sold-out mark, a title cut to 70 characters and the
        category.
        """
        rows: list[dict[str, Any]] = []

        def visit(d: dict[str, Any]) -> None:
            """Collects a dict with four or more keys whose id field names a known listing."""
            if len(d) >= 4 and any(isinstance(v, str) and v in self.archive.ids and "id" in str(key).lower() for key, v in d.items()):
                rows.append(d)
        walk_dicts(obs, visit, ceiling=3)
        parts = []
        for r in rows[:12]:
            cost = next((v for key, v in r.items() if is_number(v) and re.search(r"price|cost", str(key).lower())), None)
            parts.append(f"{self.row_mark(r)} {cost if cost is not None else '?'}{' SOLD OUT' if self.sold_out(r) else ''} "
                         f"'{self.title_of(r)[:70]}' [{self.shelf_of(r)}]")
        return "Page rows (id, price, title, category): " + ("; ".join(parts) if parts else "none")

    def walk_call(self, args: dict[str, Any], say: str) -> bool:
        """Sends one scripted search call with a short assistant explanation and puts its result in the chat. When the
        call stored a new observation, the tool message carries a page_digest of it plus any note that mentions the
        shopper, a meaning or a refusal; otherwise the raw reply text, or a not-run message when nothing came back.
        Returns True when a new observation was stored.
        """
        proposal = {"id": f"walk_{len(self.chat)}_{self.wire.tick}", "type": "function",
                    "function": {"name": self.role("search"), "arguments": json.dumps(args)}}
        self.chat.append({"role": "assistant", "content": say, "tool_calls": [proposal]})
        before = len(self.archive.items)
        self.covering = True
        try:
            responses = self.dispatch([proposal], scripted=True)
        finally:
            self.covering = False
        if not responses:
            self.chat.append({"role": "tool", "tool_call_id": proposal["id"], "content": "Not run."})
            return False
        cid, text_body = responses[0]
        got = len(self.archive.items) > before
        body = self.page_digest(self.archive.items[-1]) if got else text_body
        note = text_body.split("\n[note] ", 1)[1] if "\n[note] " in text_body else ""
        if got and note and re.search(r"shopper|Meaning:|REFUSED", note):
            body += "\n[note] " + note
        self.chat.append({"role": "tool", "tool_call_id": cid, "content": body})
        return got

    def run_walk(self, plan: list[tuple[int | None, str, int]], floor: int, ceil: list | None = None, repair: bool = False) -> bool:
        """Cheapest walk: planned pages of id-ordered searches, then planned products'
        pick-line searches (<= 2 more pages if total fits in 3), then ceiling-walks
        ceil products, each while moves > floor and time left. If any ran: cheapest
        unread in-stock rows below pick prices + ceiling rows, interleaved to batch
        size (CEIL_OPENS if ceiling rows, else 2), trimmed to moves left, queued
        for open_named. Clears walked-shelf rank checks, resets order_vetted,
        adds walk summary to last tool message, logs it. True if any call ran.
        """
        found = self.page_arg()
        if not (plan or ceil) or not found or not self.role("search"):
            return False
        key = found[0]
        book = self.__dict__.get("page_book", {})
        walked = self.__dict__.setdefault("walked", set())
        report: list[str] = []
        ran = False
        for k, wkey, n in plan:
            walked.add(k)
            walk = book.get(wkey)
            if not walk:
                continue
            query = " ".join(self.free_text(walk["args"]))
            got = 0
            for _ in range(n):
                nxt = self.next_page(walk)
                if nxt is None or self.moves_left() <= floor + 1 or self.wire.done or self.wire.broke or self.wire.spare() < CLOSING + 30:
                    break
                if not self.walk_call({**walk["args"], key: nxt}, "The cheapest is asked for and this store lists search results in a fixed "
                                      f"id order, not by price: I read the next page of '{query}'."):
                    break
                got += 1
                ran = True
            report.append(f"'{query}': {got} more page(s) read, {len(walk.get('row_shelf') or {})} of its {walk['total']} matches seen")
        lines: dict[Any, str] = {}
        for k, _, _ in plan:
            if self.moves_left() <= floor + 2 or self.wire.done or self.wire.spare() < CLOSING + 30:
                break
            args = self.line_query(k)
            if args and self.walk_call(args, "I also search the product line of the pick (its maker and the word naming its line): a "
                                       "line's other sizes and colours are separate listings that may cost less."):
                ran = True
                lines[k] = json.dumps({a: v for a, v in args.items() if a != key}, sort_keys=True, default=str)
                line = self.__dict__.get("page_book", {}).get(lines[k])
                more = 0
                while line and more < 2 and line["total"] <= 3 * max(1, line["size"]) and self.next_page(line) is not None \
                        and self.moves_left() > floor + 3 and not self.wire.done and self.wire.spare() > CLOSING + 30:
                    if not self.walk_call({**line["args"], key: self.next_page(line)}, "I read the next page of the line search."):
                        break
                    more += 1
                report.append(f"the pick's line '{' '.join(self.free_text(args))}'")
        ceil_rows: dict[Any, list[str]] = {}
        for k in dict.fromkeys(ceil or []):
            if self.moves_left() <= (floor if repair else max(floor, self.order_due() + 2)) + 1 or self.wire.done \
                    or self.wire.spare() < CLOSING + 30:
                break
            got_rows = self.ceiling_walk(k, floor, repair)
            if got_rows:
                ran = True
                report.append(got_rows[0])
                ceil_rows[k] = list(got_rows[1])
        if not ran:
            return False
        batch = self.batch_tool()
        most = batch[2] if batch else (CEIL_OPENS if ceil_rows else 2)
        pools = [self.walk_rows(k, [lines.get(k, "")], most) for k, _, _ in plan] + [rows for rows in ceil_rows.values()]
        queue: list[str] = []
        while len(queue) < most and any(pools):
            for pool in pools:
                while pool and pool[0] in queue:
                    pool.pop(0)
                if pool and len(queue) < most:
                    queue.append(pool.pop(0))
        while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= self.moves_left() - floor:
            queue.pop()
        homes = set().union(*(self.walk_home(k) for k in [k for k, _, _ in plan] + list(ceil_rows)))
        self.rank_checked = {mark for mark in self.rank_checked if self.shelf_known(mark.split("|")[0]) not in homes}
        self.order_vetted = False
        summary = ("[cheapest walk] " + "; ".join(report) + ". "
                   + ("The cheapest unread in-stock rows within the budget these searches showed, below each pick's price, are opened "
                      "next: " + ", ".join(f"{i} ({self.archive.price(i):g})" for i in queue) + ". " if queue else
                      "No unread in-stock row within the budget below the pick's price was shown. ")
                   + "For each product, choose the lowest-priced listing whose own fields meet every requirement now in force; the "
                   "rank check runs again on the cart call and the order.")
        if queue:
            self.open_queue = queue
            self.open_recheck = True
            self.open_say = "The cheapest is asked for: I open the cheapest unread rows the walk showed."
            self.open_purpose, self.open_floor = "walk", floor
            self.open_named()
        if self.chat and self.chat[-1].get("role") == "tool":
            self.chat[-1]["content"] = str(self.chat[-1]["content"]) + "\n" + summary + ("" if queue else "\n" + self.status_line())
        self.log.append({"kind": "walk", "plan": [[k, n] for k, _, n in plan], "ceiling": list(ceil_rows), "opened": queue})
        return True

    def decline_fits(self) -> set[str]:
        """Returns the listings that count as fits before a decline: those passed at the current needs version plus the
        decline audit's fits not failed at this version, minus refused lines.
        """
        ver = str(self.needs.version)
        live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == ver}
        failed = {k.split("|")[0] for k, v in self.rulings.items() if v and k.split("|")[1:2] == [ver]}
        return (live | (set(self.__dict__.get("audit_fits") or []) - failed)) - self.refused_lines

    def decline_lacking(self) -> list[int | None]:
        """Lists the products whose home shelves hold none of the decline fits. For a single item it returns [None] when
        no fit sits on a current shelf label, else an empty list.
        """
        got = {self.shelf_known(i) for i in self.decline_fits()} - {""}
        if not self.needs.items():
            return [] if set(self.shelf_labels()) & got else [None]
        return [k for k, home in enumerate(self.home_shelves()) if home and not home & got]

    def decline_walks(self) -> list[str]:
        """Picks up to two booked searches to page through before a decline, one per product that lacks a fit. Options
        from walk_options must carry at least one stated requirement (not a page, seller or same-as reference) and are
        ranked by how many they carry, then by score, pages left and key.
        """
        reqs = self.needs.props_left()

        def stated(args: dict[str, Any], k: int | None) -> int:
            """Counts the requirements a search carries for product k, leaving out reference-style ones such as as shown,
            same as or sold by.
            """
            return sum(1 for n in self.walk_carries(args, k) if n < len(reqs)
                       and not re.search(r"as shown|same as|the page|sold by|redirect|refer", reqs[n].lower()))
        keys: list[str] = []
        for k in self.decline_lacking():
            opts = sorted((o for o in self.walk_options(k) if stated(o[2]["args"], k)),
                          key=lambda p: (-stated(p[2]["args"], k), -self.walk_score(p[2]["args"], k), p[0], p[1]))
            for _, key, _ in opts:
                if key not in keys:
                    keys.append(key)
                    break
        return keys[:2]

    def row_pools(self, homes: list[tuple[set[str], int]], ids: set[str] | None, below: list[float] | None = None) -> list[list[str]]:
        """Builds, for each (home shelves, quantity) pair, a cheapest-first list of unread listings to open. A listing
        qualifies when its shelf is in the home set, it is not refused or sold out, it has a price within the cap at
        that quantity, it is in ids when ids is given, and it costs less than the matching entry of below when that
        list is given.
        """
        cap = self.needs.spend_cap
        shelf: dict[str, str] = {}
        sold: dict[str, bool] = {}

        def visit(d: dict[str, Any]) -> None:
            """Records the shelf of each known listing that shows one, and its sold-out state when the dict has a stock or
            availability key.
            """
            ident = self.row_mark(d)
            if ident in self.archive.ids and self.shelf_of(d):
                shelf[ident] = self.shelf_of(d)
                if any(re.search(r"stock|avail", str(key).lower()) for key in d):
                    sold[ident] = self.sold_out(d)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        pools = []
        for n, (home, qty) in enumerate(homes):
            pool = []
            for ident, where in shelf.items():
                cost = self.archive.price(ident)
                if where not in home or ident in self.archive.read or ident in self.refused_lines or sold.get(ident) \
                        or (ids is not None and ident not in ids) or not is_number(cost) or (is_number(cap) and cost * qty > cap) \
                        or (below is not None and not cost < below[n]):
                    continue
                pool.append((float(cost), ident))
            pools.append([ident for _, ident in sorted(pool)])
        return pools

    def over_budget_rows(self) -> list[list[str]]:
        """Finds cheaper rows when the fitting basket breaks the budget. For a multi-product request with a numeric cap,
        outside cheapest mode, it sums the cheapest decline fit of every product at its quantity. If that total is
        over the cap, it returns per product the unread in-budget rows priced below that product's pick, the products
        with the largest spend first; otherwise an empty list.
        """
        try:
            rows, cap = self.needs.items(), self.needs.spend_cap
            if not rows or self.needs.priority() == "cheapest" or not is_number(cap):
                return []
            homes = self.home_shelves()
            fits = self.decline_fits()
            picks: list[float] = []
            shares: list[float] = []
            total = 0.0
            for k, home in enumerate(homes):
                priced = [float(self.archive.price(i)) for i in fits if self.shelf_known(i) in home and is_number(self.archive.price(i))]
                if not home or not priced:
                    return []
                picks.append(min(priced))
                shares.append(min(priced) * rows[k]["quantity"])
                total += min(priced) * rows[k]["quantity"]
            if total <= cap:
                return []
            pools = self.row_pools([(home, rows[k]["quantity"]) for k, home in enumerate(homes)], None, picks)
            pools = [pools[k] for k in sorted(range(len(pools)), key=lambda k: -shares[k])]
            return pools if any(pools) else []
        except Exception as err:
            self.log.append({"kind": "over_budget_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

    def cart_spent(self) -> bool:
        """True when the cart cannot be ordered as is: a line is dead, or the archive prices of the lines times their
        counts add up to more than the spend cap. Returns False for an empty cart, no numeric cap, an unpriced line or
        an error. decline_reads uses it to allow a last look before declining even when the cart holds lines, since
        such a cart cannot be bought anyway.
        """
        try:
            units = dict(self.trolley.units)
            if not units:
                return False
            if set(units) & self.dead_lines():
                return True
            cap = self.needs.spend_cap
            if not is_number(cap):
                return False
            total = 0.0
            for ident, n in units.items():
                price = self.archive.price(ident)
                if not is_number(price):
                    return False
                total += float(price) * float(n)
            return total > float(cap) + 1e-6
        except Exception:
            return False

    def decline_reads(self) -> str:
        """Prepares a last look before a decline, once per episode, when time allows, a read tool exists and the cart is
        empty or over budget. It picks booked searches to page through (decline_walks, up to three pages, only in id-
        ordered stores) and cheaper rows for an over-budget basket (over_budget_rows). When either exists it stores
        decline_queue and returns a note telling the model what the code reads and opens and to buy what fits,
        declining only when each one fails; otherwise it returns an empty string.
        """
        try:
            batch = self.batch_tool()
            if self.__dict__.get("decline_read_done") or self.wire.spare() < CLOSING + 60 or not (self.role("details") or batch) \
                    or (self.trolley.units and not self.cart_spent()):
                return ""
            opens = 1 if batch else 2
            room = min(3, self.moves_left() - opens - 6)
            keys = self.decline_walks() if room > 0 and self.page_arg() and self.role("search") and self.id_store() else []
            cheaper = self.over_budget_rows() if self.moves_left() > opens + 6 else []
            if not keys and not cheaper:
                return ""
            self.decline_read_done = True
            self.decline_queue = (keys, room, 6)
            said = []
            if keys:
                book = self.__dict__.get("page_book", {})
                said.append("searches that carry required values matched more listings than were read ("
                            + "; ".join(f"'{' '.join(self.free_text(book[key]['args']))}'" for key in keys) + "), and this store lists "
                            "results in a fixed id order, not by fit or price, so fitting listings sit on unread pages as often as on "
                            f"read ones: the code reads up to {room} of those pages for you now and opens the cheapest in-stock rows "
                            "within the budget they show")
            if cheaper:
                said.append("the listings found for every product together cost more than the budget: the cheapest unread "
                            "in-stock rows priced below each product's pick are opened for you too")
            return ("Before a decline, " + "; ".join(said) + ". Judge those records against the requirements now in force by "
                    "meaning; buy what fits (" + ("one line per product, at its quantity" if self.needs.items() else "the item")
                    + "), and decline only if each one fails.")
        except Exception as err:
            self.log.append({"kind": "decline_reads_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def run_decline_reads(self) -> bool:
        """Executes the queued pre-decline reads. It pops decline_queue and reads up to the allowed number of next pages
        of the chosen searches through walk_call. From the rows those searches showed, it builds pools of unread, in-
        stock, in-budget listings on the shelves of products lacking a fit, and adds the over-budget pools. The
        cheapest rows, interleaved up to the batch size (or two) and trimmed to the moves left, are queued for
        open_named; when there are none, a note says so on the last tool message. Returns True when a page was read or
        rows were queued.
        """
        keys, most_pages, floor = self.__dict__.pop("decline_queue")
        found = self.page_arg()
        book = self.__dict__.get("page_book", {})
        batch = self.batch_tool()
        opens = 1 if batch else 2
        read = 0
        walked: list[str] = []
        for key in keys if found else []:
            walk = book.get(key)
            if not walk:
                continue
            walked.append(key)
            query = " ".join(self.free_text(walk["args"]))
            while read < most_pages and self.moves_left() > floor + opens and not self.wire.done and not self.wire.broke \
                    and self.wire.spare() > CLOSING + 30:
                nxt = self.next_page(walk)
                if nxt is None or not self.walk_call({**walk["args"], found[0]: nxt}, "Before declining, I read the next page of "
                                                      f"'{query}': this store lists results in a fixed id order, so fits may sit on unread pages."):
                    break
                read += 1
        seen: set[str] = set()
        for key in walked:
            seen |= set((book.get(key) or {}).get("row_shelf") or {})
        rows = self.needs.items()
        homes = [(self.walk_home(k), rows[k]["quantity"] if k is not None and k < len(rows) else self.needs.quantity())
                 for k in self.decline_lacking()]
        pools = (self.row_pools(homes, seen) if walked else []) + self.over_budget_rows()
        most = batch[2] if batch else 2
        queue: list[str] = []
        while len(queue) < most and any(pools):
            for pool in pools:
                while pool and pool[0] in queue:
                    pool.pop(0)
                if pool and len(queue) < most:
                    queue.append(pool.pop(0))
        while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= self.moves_left() - floor:
            queue.pop()
        self.log.append({"kind": "decline_reads", "pages": read, "opened": queue})
        if queue and not self.wire.done:
            self.open_queue = queue
            self.open_say = "Before declining, I open the cheapest unread in-budget rows those pages and picks show."
            self.open_purpose, self.open_floor = "decline", floor
            self.open_named()
        elif self.chat and self.chat[-1].get("role") == "tool":
            self.chat[-1]["content"] = str(self.chat[-1]["content"]) + "\nNo unread in-stock row within the budget was shown there.\n" + self.status_line()
        return read > 0 or bool(queue)

    def ref_register(self) -> None:
        """Opens reference trackers for the request's written requirements: each requirement whose value refers to
        another listing, or a listed reference whose value uses differ-from wording, goes to ref_add with the
        reading's reference targets and the request text.
        """
        reading = self.needs.reading
        infos = reading.get("reference_targets") if isinstance(reading.get("reference_targets"), list) else []
        infos = [t for t in infos if isinstance(t, dict)]
        listed = {str(r) for r in reading.get("references") or []} if isinstance(reading.get("references"), list) else set()
        for req in list(self.needs.written):
            value = str(req).partition(":")[2]
            if ref_text(value) or (str(req) in listed and re.search(DIFFER_WORDS, value.lower())):
                self.ref_add(str(req), infos, self.request, "request")

    def ref_untracked(self, refs: list[str], tracked: list[dict[str, Any]]) -> list[str]:
        """Returns the reference requirements that no tracker matches with a resolved status or a chain; on error it
        returns all of them.
        """
        try:
            return [r for r in refs if not any(self.ref_match(t, r) and (t.get("status") == "resolved" or t.get("chain")) for t in tracked)]
        except Exception:
            return refs

    def ref_take(self) -> None:
        """Brings the reference trackers up to date with the needs. A resolved tracker whose requirement now appears in
        another wording gets its own current text swapped back in (ref_swap). Reference news queued from shopper
        replies is drained and each item is opened with ref_add; each new tracker adds its requirement to the
        reading's references list. Errors are logged.
        """
        try:
            for ref in self.refs:
                if ref["status"] == "resolved" and any(self.ref_match(ref, r) and r != ref["now"] for r in self.needs.props_left()):
                    self.ref_swap(ref, ref["now"])
            news = list(self.needs.__dict__.get("ref_news") or [])
            self.needs.ref_news = []
            for item in news:
                ref = self.ref_add(str(item.get("need") or ""), list(item.get("infos") or []), str(item.get("line") or ""), "reply")
                if ref:
                    refs = self.needs.reading.get("references")
                    if not isinstance(refs, list):
                        refs = self.needs.reading["references"] = []
                    if ref["req"] not in refs:
                        refs.append(ref["req"])
        except Exception as err:
            self.log.append({"kind": "ref_take_error", "error": f"{type(err).__name__}: {err}"[:300]})

    def ref_add(self, req: str, infos: list[dict[str, Any]], source: str, origin: str) -> dict[str, Any] | None:
        """Opens a tracker for a requirement that points at another listing (a named seller's).
        Parses the property and product tag; skips junk and already tracked ones.
        Resolves the target (seller, kind, details) with ref_target.
        The relation (same, within, differ, counterpart) comes from the value and wording.
        The tracker starts open with empty counters and is logged.
        Returns it, or None when skipped.
        """
        head, _, value = str(req).partition(":")
        prop = re.split(r"\s*\(", head)[0].strip()
        tags = re.findall(r"\(([^)]*)\)", head)
        tag = tags[-1].strip() if tags else ""
        if not prop or junk_key(prop):
            return None
        for old in self.refs:
            if old["req"] == req or (same_prop(old["prop"], prop) and kind_words(old["tag"]) == kind_words(tag)):
                return None
        info = None
        for cand in infos:
            if same_prop(str(cand.get("property") or ""), prop) and (not tag or not cand.get("product") or not self.needs.items()
                                                                    or kind_words(cand.get("product")) & kind_words(tag)):
                info = cand
                break
        tgt = self.ref_target(info, value, source, prop)
        if not tgt:
            self.log.append({"kind": "ref_skip", "req": str(req)[:200]})
            return None
        rel, tol, unit = ref_relation(value)
        if rel == "same" and isinstance(info, dict) and str(info.get("relation") or "").lower() == "within" and is_number(info.get("tolerance")):
            rel, tol, unit = "within", float(info["tolerance"]), str(info.get("unit") or "")
        told = str(info.get("relation") or "").lower() if isinstance(info, dict) else ""
        if rel == "same" and told == "counterpart":
            rel = "counterpart"
        span = ref_span(prop, source).lower()
        cut = span.find(str(tgt.get("seller") or "").lower()) if str(tgt.get("seller") or "").strip() else -1
        if rel == "same" and not re.search(SAME_WORDS, span[:cut] if cut > 0 else span) and (
                re.search(r"differ|not the same|exclu|apart|other than|unlike", told)
                or (cut > 0 and re.search(DIFFER_WORDS, span[:cut]))):
            rel, tol, unit = "differ", None, ""
        ref = {"req": str(req), "head": head.strip(), "prop": prop, "tag": tag, "relation": rel, "tol": tol, "unit": unit,
               "target": tgt, "first": dict(tgt), "chain": [], "seen": {self.ref_key(tgt)}, "judged": set(), "queries": [],
               "pages": 0, "hops": 0, "tries": 0, "fails": 0, "status": "open", "value": "", "goes_with": [], "now": "",
               "origin": origin, "unindexed": False}
        self.refs.append(ref)
        self.log.append({"kind": "ref_open", "req": str(req)[:200], "target": tgt, "relation": rel, "origin": origin})
        return ref

    def ref_target(self, info: Any, value: str, source: str, prop: str) -> dict[str, Any] | None:
        """Resolves which listing a reference requirement points to, as seller, kind and details. It prefers the model's
        target info when its seller passes ref_seller_ok, else what ref_parse finds in the request span or the value.
        Details are deduplicated, kept only when they share words with the source text, and capped at six. Returns
        None when no valid seller is found.
        """
        seller, kind, details = "", "", []
        if isinstance(info, dict):
            seller = str(info.get("seller") or "").strip()
            kind = str(info.get("kind") or "").strip()
            raw = info.get("details") if isinstance(info.get("details"), list) else []
            details = [str(d).strip() for d in raw if isinstance(d, (str, int, float)) and not isinstance(d, bool) and str(d).strip()]
        parsed = None
        for text in (ref_span(prop, source), value):
            got = ref_parse(text, lambda name: self.ref_seller_ok(name, source))
            if got and self.ref_seller_ok(got["seller"], source):
                parsed = got
                break
        if not self.ref_seller_ok(seller, source):
            if not parsed:
                return None
            seller, kind, details = parsed["seller"], kind or parsed["kind"], details or parsed["details"]
        elif not details and parsed and ref_name(parsed["seller"]) == ref_name(seller):
            details = parsed["details"]
        said = set(ref_toks(source))
        details = [d for d in dict.fromkeys(details) if set(ref_toks(d)) & said][:6]
        return {"seller": seller, "kind": (kind or parsed["kind"] if parsed else kind)[:80], "details": details}

    def ref_seller_ok(self, seller: Any, source: Any) -> bool:
        """Checks a seller name for a reference: at least three letters, named as a whole word (optionally with a
        possessive s) in the source text, and not made only of product kind words or search enum words. ref_target
        uses it to accept the model's seller or a parsed one.
        """
        name = str(seller or "").strip()
        if len(re.sub(r"[^A-Za-z]", "", name)) < 3:
            return False
        if not re.search(r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?:'s)?(?![a-z0-9])", str(source or "").lower()):
            return False
        kinds: set[str] = set()
        for row in self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}]:
            kinds |= kind_words(row["product"])
        for pspec in ((self.shapes.get(self.role("search") or "") or {}).get("properties") or {}).values():
            if isinstance(pspec, dict) and isinstance(pspec.get("enum"), list):
                kinds |= {w for v in pspec["enum"] for w in kind_words(v)}
        words = kind_words(name)
        return bool(words) and not words <= kinds

    @staticmethod
    def ref_key(tgt: dict[str, Any]) -> tuple:
        """Builds an identity key for a reference target: the normalised seller name plus the sorted tokens of its
        details.
        """
        return (ref_name(tgt.get("seller") or ""), tuple(sorted({x for d in tgt.get("details") or [] for x in ref_toks(d)})))

    @staticmethod
    def ref_desc_of(tgt: dict[str, Any]) -> str:
        """Describes a reference target in words as the kind sold by the seller, with its details in parentheses when
        present.
        """
        return (f"the {tgt.get('kind') or 'listing'} sold by {tgt.get('seller')}"
                + (" (" + "; ".join(tgt.get("details") or []) + ")" if tgt.get("details") else ""))

    def ref_desc(self, ref: dict[str, Any]) -> str:
        """Describes the target listing of a reference tracker."""
        return self.ref_desc_of(ref["target"])

    def ref_index(self) -> dict[str, dict[str, Any]]:
        """Merges every stored observation into one dict per known listing id, using dicts with at least four keys or a
        nested dict of two or more fields. Notes in ref_seen_at the latest observation index of each listing and
        memoises the result on the number of stored observations. A listing seen in several observations gets the
        fields of all of them merged, later ones winning.
        """
        memo = self.__dict__.get("ref_memo")
        if memo and memo[0] == len(self.archive.items):
            return memo[1]
        rows: dict[str, dict[str, Any]] = {}
        seen_at: dict[str, int] = {}
        for n, rec in enumerate(self.archive.items):
            def visit(d: dict[str, Any]) -> None:
                """Merges a dict into its listing's entry when its id is known and it looks like a record, noting the observation
                index.
                """
                ident = self.row_mark(d)
                if ident in self.archive.ids and (len(d) >= 4 or any(isinstance(v, dict) and len(v) >= 2 for v in d.values())):
                    rows.setdefault(ident, {}).update(d)
                    seen_at[ident] = n
            walk_dicts(rec, visit)
        self.ref_memo = (len(self.archive.items), rows)
        self.ref_seen_at = seen_at
        return rows

    def ref_sellers(self, row: dict[str, Any]) -> set[str]:
        """Returns the normalised seller names of a row: short string values under seller-like keys or keys learned by
        seller_keys, leaving out id fields and note or description-like keys.
        """
        learned = self.seller_keys()
        return {ref_name(v) for k, v in row.items() if isinstance(v, str) and v.strip() and len(v) <= 60
                and (re.search(SELLER_KEYS, str(k).lower()) or str(k) in learned) and "id" not in str(k).lower()
                and not re.search(r"note|text|desc|comment|detail|info|message|section", str(k).lower())} - {""}

    def seller_keys(self) -> set[str]:
        """Learns which row keys hold seller names in this store beyond the usual seller-like names. A key counts when
        its short string value is a tracked reference seller, or a capitalised name of up to three words that also
        turns up in row notes as a capitalised run and is not a nested field value. Title, category, text, url and id
        keys are ignored. The result is memoised on the observation count and the tracked seller names; errors return
        an empty set.
        """
        try:
            names = {ref_name(t.get("seller")) for r in self.__dict__.get("refs") or [] for t in (r.get("target") or {}, r.get("first") or {})
                     if isinstance(t, dict)} - {""}
            mark = (len(self.archive.items), tuple(sorted(names)))
            memo = self.__dict__.get("seller_memo")
            if memo and memo[0] == mark:
                return memo[1]
            rows = self.ref_index()
            noted: set[str] = set()
            vocab: set[str] = set()
            for row in rows.values():
                for v in row.values():
                    if isinstance(v, dict):
                        vocab.update(ref_name(x) for x in v.values() if isinstance(x, str))
                for sentence in re.split(r"(?<=[.!?;])\s+", self.ref_texts(row, desc=False)):
                    run: list[str] = []
                    for word in re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)[1:] + [""]:
                        if word[:1].isupper() and re.search(r"[a-z]", word):
                            run.append(word)
                            continue
                        for n in range(len(run)):
                            for j in range(n + 1, min(len(run), n + 3) + 1):
                                noted.add(ref_name(" ".join(run[n:j])))
                        run = []
            noted = {w for w in noted if len(w) >= 4} - vocab
            keys: set[str] = set()
            for row in rows.values():
                for k, v in row.items():
                    low = str(k).lower()
                    if isinstance(v, str) and 0 < len(v) <= 60 and "id" not in low and not re.search(
                            r"title|heading|name|label|caption|categor|shelf|department|aisle|note|text|desc|comment|detail|info|"
                            r"message|section|url|image|summary|overview", low) and (
                            ref_name(v) in names or (v.strip()[:1].isupper() and len(v.split()) <= 3 and ref_name(v) in noted)):
                        keys.add(str(k))
            self.seller_memo = (mark, keys)
            return keys
        except Exception:
            return set()

    @staticmethod
    def ref_texts(row: dict[str, Any], desc: bool = True) -> str:
        """Joins a row's free-text strings of three or more words, skipping id, title, name, category, url and image
        fields. Note-like fields are used first; description, overview, summary or about fields are used only when no
        notes exist and desc is set.
        """
        notes, other = [], []
        for k, v in row.items():
            name = str(k).lower()
            if not isinstance(v, str) or len(v.split()) < 3 or name == "id" or name.endswith("id") \
                    or re.search(r"title|heading|headline|name|label|caption|categor|shelf|department|url|image", name):
                continue
            (other if re.search(r"desc|overview|summary|about", name) else notes).append(v)
        return "\n".join(notes or (other if desc else []))

    def ref_parts(self, row: dict[str, Any]) -> tuple[set[str], set[str]]:
        """Splits a row into key tokens and value tokens for reference matching. Values are the title, the notes with
        negated phrases removed and the nested field values (or the flat non-meta fields when no nested dict exists);
        keys are those fields' names with camelCase and underscores split.
        """
        notes = re.sub(r"\b(?:not|no|never|without|nor)\b(?:\W+[A-Za-z0-9.:/&'-]+){1,3}", " ", self.ref_texts(row, desc=False), flags=re.I)
        keys, vals = [], [self.title_of(row), notes]
        maps = [v for v in row.values() if isinstance(v, dict)]
        for fields in maps or [{k: v for k, v in row.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)
                                and not re.search(r"id$|title|shop|seller|brand|price|amount|desc|overview|note", str(k).lower())}]:
            for k, x in fields.items():
                keys.append(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(k)).replace("_", " "))
                if isinstance(x, (str, int, float)) and not isinstance(x, bool):
                    vals.append(str(x))
        return set(ref_toks(" ".join(keys))), set(ref_toks(" | ".join(vals)))

    def ref_shows(self, row: dict[str, Any], detail: str, parts: tuple[set[str], set[str]] | None = None) -> bool:
        """Checks whether a row shows a reference detail. The detail's word tokens (three or more letters, no digits, not
        field names) must meet the row's value tokens, and each figure with a unit must equal one of the same-unit
        figures in the title or nested fields whenever such figures exist.
        """
        keys, vals = parts or self.ref_parts(row)
        want = {t for t in ref_toks(detail) if len(t) >= 3 and not re.search(r"\d", t)} - keys
        if want and not want & vals:
            return False
        figs = ref_figs(detail)
        if figs:
            mine = ref_figs(" | ".join([self.title_of(row)] + [str(x) for v in row.values() if isinstance(v, dict)
                                                               for x in v.values() if isinstance(x, (str, int, float))]))
            for fig, unit in figs:
                same = [f for f, u in mine if u == unit]
                if same and not any(abs(f - fig) < 1e-9 for f in same):
                    return False
        return True

    @staticmethod
    def ref_field(row: dict[str, Any], prop: str) -> str:
        """Returns the first non-empty nested field value whose key names the property, trying a strict key match first
        and a looser same-property match second; an empty string when none is found. Only values inside nested field
        dicts count; booleans and blank values are skipped.
        """
        for strict in (True, False):
            for val in row.values():
                if isinstance(val, dict):
                    for k, x in val.items():
                        if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip() \
                                and (strict_prop(k, prop) if strict else same_prop(k, prop)):
                            return str(x).strip()
        return ""

    def ref_ids_add(self, ids: list[str]) -> None:
        """Adds the given listing ids to self.ref_ids, the set of listings opened while following a shopper's
        reference to another listing.
        """
        self.ref_ids = set(self.__dict__.get("ref_ids") or set()) | set(ids)

    def ref_foreign(self) -> set[str]:
        """Returns the ids opened for references that sit on a shelf outside every home shelf of the request's products.
        Starts from self.ref_ids, skips ids already in the cart (trolley units) and keeps ids whose indexed row has a
        shelf that no product's home shelves include. Returns an empty set when no reference ids exist, when any product
        has no home shelves, or on any error.
        """
        try:
            ids = self.__dict__.get("ref_ids") or set()
            if not ids:
                return set()
            shelves = self.home_shelves()
            if not shelves or not all(shelves):
                return set()
            homes = set().union(*shelves)
            rows = self.ref_index()
            mine = set()
            for k in self.trolley.units:
                mine.add(k)
            return {i for i in ids if i not in mine and self.shelf_of(rows.get(i) or {}) and self.shelf_of(rows.get(i) or {}) not in homes}
        except Exception:
            return set()

    def ref_cands(self, ref: dict[str, Any]) -> list[str]:
        """Ranks the indexed listings that may be the listing a reference points at, best first.
        Keeps rows whose sellers include the target seller and counts the target details fully covered by the title and
        description-like fields. Sorting favours a shelf matching the target kind (worth two details) plus covered
        details, then the higher ref_seen_at mark. Stores the per-id covered count in ref['shows'] and returns the ids.
        """
        tgt = ref["target"]
        name = ref_name(tgt["seller"])
        kinds = kind_words(tgt["kind"])
        rows = self.ref_index()
        seen_at = self.__dict__.get("ref_seen_at") or {}
        scored = []
        for ident, row in rows.items():
            if name not in self.ref_sellers(row):
                continue
            hay = set(ref_toks(self.title_of(row) + " " + " ".join(str(v) for k, v in row.items()
                                                                  if isinstance(v, str) and re.search(r"desc|overview|summary", str(k).lower()))))
            shown = sum(1 for d in tgt["details"] if set(ref_toks(d)) and set(ref_toks(d)) <= hay)
            scored.append((-(2 if kinds & kind_words(self.shelf_of(row)) else 0) - shown, -seen_at.get(ident, 0), ident, shown))
        scored.sort()
        ref["shows"] = {ident: shown for _, _, ident, shown in scored}
        return [ident for _, _, ident, _ in scored]

    def ref_modes(self, needed: set[str], labels: set[str]) -> tuple[list[tuple[dict[str, Any], str]], set[str]]:
        """Derives search templates from the walks recorded in self.page_book. A walk with exactly one free-text argument
        gives a template: its other arguments (shelf-label values dropped unless the search schema requires that key)
        plus the free-text key, and walks sharing a template are grouped. A group is marked blind when two of its queries
        with no words in common returned the same first results (two or more). Returns the other templates, ranked by
        distinct query word sets and then first appearance, together with the set of blind group keys.
        """
        groups: dict[str, dict[str, Any]] = {}
        for n, walk in enumerate(self.__dict__.get("page_book", {}).values()):
            args = walk.get("args") if isinstance(walk.get("args"), dict) else {}
            texts = self.free_text(args)
            keys = [k for k, v in args.items() if isinstance(v, str) and v in texts and v.strip()]
            if len(texts) != 1 or len(keys) != 1:
                continue
            other = {k: v for k, v in args.items() if k != keys[0] and (k in needed or not (isinstance(v, str) and v.strip().lower() in labels))}
            group = groups.setdefault(json.dumps(other, sort_keys=True, default=str), {"tpl": (other, keys[0]), "seen": [], "at": n})
            group["seen"].append(({self.stem(w) for w in re.findall(r"[a-z0-9]+", texts[0].lower())}, tuple(walk.get("first") or ())))
        ranked, blind = [], set()
        for mode, group in groups.items():
            seen = group["seen"]
            if any(len(f1) >= 2 and f1 == f2 and not (w1 & w2) for i, (w1, f1) in enumerate(seen) for w2, f2 in seen[i + 1:]):
                blind.add(mode)
                continue
            ranked.append((-len({tuple(sorted(w)) for w, _ in seen}), group["at"], group["tpl"]))
        ranked.sort(key=lambda x: (x[0], x[1]))
        return [tpl for _, _, tpl in ranked], blind

    def ref_template(self) -> tuple[dict[str, Any], str] | None:
        """Picks the search arguments for looking up a referenced listing.
        Returns (fixed arguments, free-text key) or None.
        First the top template from ref_modes.
        Else the latest search call with one free-text argument, paging dropped.
        Shelf-label values are removed unless the schema requires that key.
        Falls back to the schema's single free-text property.
        """
        jschema = self.shapes.get(self.role("search") or "") or {}
        needed = set(jschema.get("required") or []) if isinstance(jschema.get("required"), list) else set()
        labels = {self.shelf_of(r) for r in self.ref_index().values()} - {""}
        modes, blind = self.ref_modes(needed, labels)
        tpl = modes[0] if modes else None
        paging = self.page_arg()
        for entry in reversed(self.log) if not tpl else []:
            for act in reversed(entry.get("actions") or []) if entry.get("kind") == "env" else []:
                args = act.get("args") if act.get("name") == self.role("search") else None
                if not isinstance(args, dict):
                    continue
                texts = [k for k, v in args.items() if isinstance(v, str) and v.strip() and v in self.free_text(args)]
                if len(texts) == 1:
                    other = {k: v for k, v in args.items() if k != texts[0] and not (paging and k == paging[0])}
                    kept = {k: v for k, v in other.items() if k in needed or not (isinstance(v, str) and v.strip().lower() in labels)}
                    if json.dumps(kept, sort_keys=True, default=str) in blind:
                        continue
                    tpl = (other, texts[0])
                    break
            if tpl:
                break
        if tpl:
            other, key = tpl
            return {k: v for k, v in other.items() if k in needed or not (isinstance(v, str) and v.strip().lower() in labels)}, key
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        tok = self.token_key()
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum") and k != tok]
        if len(texts) == 1 and not (needed - {texts[0]}):
            return {}, texts[0]
        return None

    def ref_words(self, ref: dict[str, Any]) -> list[str]:
        """Chooses search words for a reference target from its details.
        Order: letter-digit tokens, numbers of 2+ digits, then plain words.
        Skips link words, seller-name words, units after a number and field names.
        Plain words go longest first.
        A multi-word detail's last word leads before the first hop.
        Duplicates are removed.
        """
        keys: set[str] = set()
        for row in self.ref_index().values():
            for val in row.values():
                if isinstance(val, dict):
                    keys |= {w for k in val for w in re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(k)).lower())}
        keys |= {w for req in self.needs.written + self.needs.uncovered for w in re.findall(r"[a-z]+", re.split(r"\s*\(", str(req).partition(":")[0])[0].lower())}
        seller = set(re.findall(r"[a-z0-9]+", str(ref["target"]["seller"]).lower()))
        joined, figures, words = [], [], []
        for d in ref["target"]["details"]:
            prev = ""
            for whole in str(d).lower().split():
                unit = bool(re.fullmatch(r"\d+(?:[.,]\d+)?", prev))
                prev = whole
                for w in re.findall(r"[a-z0-9]+", whole):
                    if w in LINK_WORDS or w in seller or (unit and not re.search(r"\d", w)):
                        continue
                    if re.search(r"\d", w) and re.search(r"[a-z]", w):
                        joined.append(w)
                    elif w.isdigit():
                        if len(w) >= 2:
                            figures.append(w)
                    elif len(w) >= 3 and w == whole.strip(".,;:!?()[]{}\"'") and w not in keys and w.rstrip("s") not in keys:
                        words.append(w)
        words.sort(key=len, reverse=True)
        lasts = []
        skeys = self.store_keys()
        split = [[w for w in re.findall(r"[a-z0-9]+", str(d).lower()) if w not in LINK_WORDS] for d in ref["target"]["details"]]
        pairs = ref.get("hops", 0) == 0 and len(split) >= 1 and all(len(parts) >= 2 for parts in split)
        for parts in split:
            if len(parts) >= 2 and parts[-1] in words and (pairs or any(" ".join(parts[:j]) in skeys for j in range(1, len(parts)))):
                lasts.append(parts[-1])
        words = [w for w in lasts if w in words] + [w for w in words if w not in lasts]
        return list(dict.fromkeys(joined + figures + words))

    def store_keys(self) -> set[str]:
        """Collects the field names used in stored observations: for each nested dict holding two or more string or
        number values, every key is split at camelCase, lowercased and reduced to space-joined alphanumeric words.
        Walks every record in self.archive.items and memoizes the set in self.store_keys_memo against the archive
        length, so it is rebuilt only after new records arrive.
        """
        mark = len(self.archive.items)
        memo = self.__dict__.get("store_keys_memo")
        if memo and memo[0] == mark:
            return memo[1]
        keys: set[str] = set()

        def visit(d: dict[str, Any]) -> None:
            """Adds the normalised key names of each child dict of d that holds at least two string or number values to the
            enclosing key set.
            """
            for v in d.values():
                if isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2:
                    for k in v:
                        name = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(k)).lower()))
                        if name:
                            keys.add(name)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        self.store_keys_memo = (mark, keys)
        return keys

    def ref_plans(self, ref: dict[str, Any]) -> list[str]:
        """Builds the next search queries for a reference, leaving out queries already in ref['queries']. For an
        unindexed target (search on the seller name showed none of its rows) it pairs the top detail word with the last
        kind word, cut to the lead-word width; otherwise it tries the seller name with the top detail word (when the
        seller name is shorter than that width) and then the seller name alone.
        """
        width = max(2, self.lead_words or 3)
        seller = " ".join(re.findall(r"[A-Za-z0-9]+", re.sub(r"'s\b", "", str(ref["target"]["seller"]))))
        words = self.ref_words(ref)
        kind = [w for w in re.findall(r"[a-z]{3,}", str(ref["target"]["kind"]).lower()) if w not in LINK_WORDS][-1:]
        if ref.get("unindexed"):
            plans = [" ".join((words[:1] + kind)[:width])] if words or kind else []
        else:
            plans = ([seller + " " + words[0]] if words and len(seller.split()) < width else []) + [seller]
        return [p for p in plans if p.strip() and p.lower() not in ref["queries"]]

    def ref_page(self, ref: dict[str, Any]) -> dict[str, Any] | None:
        """Returns the arguments for the next result page of the latest search in self.page_book whose free text names
        the reference's seller and that has a next page. Returns None when the store has no paging argument or the
        reference has already used three pages.
        """
        found = self.page_arg()
        if not found or ref["pages"] >= 3:
            return None
        name = ref_name(ref["target"]["seller"])
        for walk in reversed(list(self.__dict__.get("page_book", {}).values())):
            if not name or name not in ref_name(" ".join(self.free_text(walk["args"]))):
                continue
            nxt = self.next_page(walk)
            if nxt is not None:
                return {**walk["args"], found[0]: nxt}
        return None

    def ref_room(self) -> bool:
        """Decides whether another step may go to following references. Refuses when every open reference is a blind
        'differ' one (no chain) and REF_DIFFER_BLIND steps are spent; caps reference steps at REF_STEPS of the step cap
        (at least 8), or REF_DIFFER_STEPS (at least 6) when all open ones are 'differ'. It also needs time beyond the
        closing reserve, unspent budget and a live wire, and refuses while ref_waits holds. Otherwise it needs moves
        left above ref_floor, or a late allowance from ref_hop_room.
        """
        blind = [r for r in self.__dict__.get("refs") or [] if r.get("status") == "open"]
        if blind and all(r.get("relation") == "differ" and not r.get("chain") for r in blind) \
                and self.__dict__.get("ref_steps", 0) >= REF_DIFFER_BLIND:
            return False
        cap = max(8, int(REF_STEPS * self.step_cap))
        floor = self.ref_floor()
        opens = [r for r in self.__dict__.get("refs") or [] if r.get("status") == "open"]
        if opens and all(r.get("relation") == "differ" for r in opens):
            cap = max(6, int(REF_DIFFER_STEPS * self.step_cap))
        if not (self.__dict__.get("ref_steps", 0) < cap and self.wire.spare() > CLOSING + 60 and not self.wire.spent_out()
                and not self.wire.done and not self.wire.broke):
            return False
        if self.ref_waits(floor):
            return False
        return self.moves_left() > floor or self.ref_hop_room()

    def ref_floor(self) -> int:
        """Number of moves kept back from reference work: ten past the first-order point (a quarter of the step cap plus
        one, held between FIRST_ORDER_BY and FIRST_ORDER_DUE), or order_due plus 7, whichever is larger.
        """
        return max(max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, self.step_cap // 4 + 1)) + 10, self.order_due() + 7)

    def ref_product(self, ref: dict[str, Any]) -> int | None:
        """Returns the index of the requested product that the reference's tag names, or None when the request has no
        product rows or the reference has no tag.
        """
        return self.product_of(f"x ({ref['tag']})") if self.needs.items() and ref.get("tag") else None

    def ref_waits(self, floor: int | None = None) -> bool:
        """Tells whether reference work should hold so that other products can find candidates first. Applies only when
        the request has product rows, every open reference is a 'differ' one with a chain, and more than the floor plus
        4 moves remain. Products owned by open references are left out; it holds when another product has no fitting
        entry in its ed_pool (two or more such products when some reference has no product). Logs ref_wait when the
        lacking list changes, and logs errors and returns False.
        """
        rows = self.needs.items()
        opens = [r for r in self.__dict__.get("refs") or [] if r.get("status") == "open"]
        if not rows or not opens or not all(r.get("relation") == "differ" and r.get("chain") for r in opens):
            return False
        if self.moves_left() <= (self.ref_floor() if floor is None else floor) + 4:
            return False
        try:
            owners = {self.ref_product(r) for r in opens}
            lacking = [k for k in range(len(rows)) if not any(e["fit"] for e in self.ed_pool(k))]
        except Exception as err:
            self.log.append({"kind": "ref_wait_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False
        known = owners - {None}
        rest = [k for k in lacking if k not in known]
        waits = len(rest) >= 2 if None in owners else bool(rest)
        if waits and self.__dict__.get("ref_wait_said") != tuple(rest):
            self.ref_wait_said = tuple(rest)
            self.log.append({"kind": "ref_wait", "lacking": rest})
        return waits

    def ref_hop_room(self) -> bool:
        """Grants reference steps past the reserve floor to an open reference that has hopped at least once and has a
        full target (seller, kind and details). A grant lasts three reference steps from ref['late_from']. A new grant
        starts only for such a reference with no queries yet whose other products all have a verified fit
        (ed_others_verified), and is logged as ref_late. Returns False when moves left are within order_due plus 2.
        """
        if self.moves_left() <= self.order_due() + 2:
            return False
        steps = self.__dict__.get("ref_steps", 0)
        for ref in self.__dict__.get("refs") or []:
            tgt = ref.get("target") or {}
            if ref.get("status") != "open" or ref.get("hops", 0) < 1 or not str(tgt.get("seller") or "").strip() \
                    or not str(tgt.get("kind") or "").strip() or not tgt.get("details"):
                continue
            start = ref.get("late_from")
            if start is not None:
                if steps - start < 3:
                    return True
                continue
            if ref.get("queries") or not self.ed_others_verified(ref):
                continue
            ref["late_from"] = steps
            self.log.append({"kind": "ref_late", "prop": ref.get("prop"), "to": str(tgt.get("seller"))[:80]})
            return True
        return False

    def ed_others_verified(self, ref: dict[str, Any]) -> bool:
        """Checks that every product other than the reference's own has a verified, fitting entry in its ed_pool.
        True with no product rows; when the reference maps to no product, at most one product may lack one. Returns
        False if the pool lookup fails.
        """
        rows = self.needs.items()
        if not rows:
            return True
        try:
            bare = [k for k in range(len(rows)) if not any(e["verified"] and e["fit"] for e in self.ed_pool(k))]
        except Exception:
            return False
        own = self.ref_product(ref)
        return not [k for k in bare if k != own] if own is not None else len(bare) <= 1

    def ref_calls_ok(self) -> bool:
        """Allows another reference model check while fewer than REF_CALLS were made, time beyond the closing reserve
        plus 40 is left, and spend is under 80% of the scaled spend cap.
        """
        return (self.__dict__.get("ref_calls", 0) < REF_CALLS and self.wire.spare() > CLOSING + 40 and not self.wire.spent_out()
                and self.wire.spend < 0.8 * SPEND_CAP * self.wire.__dict__.get("scale", 1.0))

    def ref_match(self, ref: dict[str, Any], req: Any) -> bool:
        """Tells whether a requirement line is the one a reference stands for: it equals ref['req'] or ref['now'], or
        its value still points to another listing (or uses differ wording for a 'differ' reference) under the same
        property name. When the requirement carries a product tag in parentheses, its kind words must overlap the
        reference's tag.
        """
        head, _, value = str(req).partition(":")
        if str(req) in (ref["req"], ref.get("now")):
            return True
        pointing = ref_text(value) or (ref.get("relation") == "differ" and re.search(DIFFER_WORDS, value.lower()))
        if not pointing or not same_prop(re.split(r"\s*\(", head)[0], ref["prop"]):
            return False
        tags = re.findall(r"\(([^)]*)\)", head)
        return not tags or not ref["tag"] or bool(kind_words(tags[-1]) & kind_words(ref["tag"]))

    def ref_live(self, ref: dict[str, Any]) -> bool:
        """Reports whether a reference still matters: true when an open requirement matches it or shares its property,
        false when its property is only found among the requirements the shopper let go.
        """
        left = self.needs.props_left()
        if any(self.ref_match(ref, r) or same_prop(re.split(r"\s*\(", str(r).partition(":")[0])[0], ref["prop"]) for r in left):
            return True
        return not any(same_prop(re.split(r"\s*\(", str(r).partition(":")[0])[0], ref["prop"]) for r in self.needs.let_go)

    def ref_step(self) -> bool:
        """Runs one round of reference following when references exist and the wire is still running, and returns
        whether it acted. On an exception it logs ref_error, marks every open reference 'stuck' with a reason and
        returns False.
        """
        if not self.__dict__.get("refs") or self.wire.done or self.wire.broke:
            return False
        try:
            return self.ref_turn()
        except Exception as err:
            self.log.append({"kind": "ref_error", "error": f"{type(err).__name__}: {err}"[:300]})
            for ref in self.refs:
                if ref["status"] == "open":
                    ref["status"], ref["why"] = "stuck", "the code's resolver failed"
            return False

    def ref_turn(self) -> bool:
        """Reference-resolver round. Drops dead open refs (ref_dropped); ref_judge
        rules on read candidates, fewest hops first; True if it spent a step.
        With room (ref_room), pools unread unjudged candidates per ref that
        searched, lacks plans, has few candidates or a detailed top one;
        interleaves them into a batch-size queue. Full queue -> ref_read;
        else, given a search template, next planned query or result page for
        refs needing one. Refs out of candidates, plans and pages: not found.
        """
        opens = []
        for ref in self.refs:
            if ref["status"] == "open" and not self.ref_live(ref):
                ref["status"] = "dropped"
                self.log.append({"kind": "ref_dropped", "req": ref["req"][:200]})
            if ref["status"] == "open":
                opens.append(ref)
        if not opens:
            return False
        acted = False
        tick0 = self.wire.tick
        for ref in sorted(opens, key=lambda r: r["hops"]):
            acted = self.ref_judge(ref) or acted
        if self.wire.tick > tick0:
            return True
        opens = [r for r in opens if r["status"] == "open"]
        if not opens or not self.ref_room():
            return acted
        batch = self.batch_tool()
        most = batch[2] if batch else 2
        pools = []
        for ref in opens:
            pool = [i for i in self.ref_cands(ref) if i not in self.archive.read and i not in ref["judged"]]
            shows = ref.get("shows") or {}
            ready = bool(ref["queries"]) or not self.ref_plans(ref) or len(pool) <= most or any(shows.get(i) for i in pool[:1])
            pools.append(pool if ready else [])
        tpl = self.ref_template()
        empty = [r for r, pool in zip(opens, pools) if not pool and self.ref_plans(r)] if tpl else []
        queue: list[str] = []
        while len(queue) < most and any(pools):
            for pool in pools:
                while pool and pool[0] in queue:
                    pool.pop(0)
                if pool and len(queue) < most:
                    queue.append(pool.pop(0))
        needy = empty if queue and len(queue) < most else []
        if queue and not needy:
            return self.ref_read(queue, opens) or acted
        if not tpl:
            return acted
        for ref in needy or sorted(opens, key=lambda r: (r["hops"], len(r["queries"]))):
            for plan in self.ref_plans(ref):
                ref["queries"].append(plan.lower())
                if self.ran_query(plan):
                    return True
                return self.ref_search(ref, {**tpl[0], tpl[1]: plan}) or acted
            args = self.ref_page(ref)
            if args:
                ref["pages"] += 1
                return self.ref_search(ref, args) or acted
        for ref in opens:
            if not [i for i in self.ref_cands(ref) if i not in ref["judged"]] and not self.ref_plans(ref) and not self.ref_page(ref):
                self.ref_stop(ref, "the listing it points at was not found in the store's search")
                acted = True
        return acted

    def ref_read(self, queue: list[str], opens: list[dict[str, Any]]) -> bool:
        """Opens the queued candidate listings for open references, through the batch read tool when two or more are
        queued, and posts the replies. On the first batch read it checks the result with batch_ok and either marks the
        batch tool proven or drops it (batch_dropped). Adds the spent steps to ref_steps, records the ids in ref_ids,
        logs ref_read and, unless the batch tool was just dropped, marks ids that stayed unread as judged. Returns
        whether a step was spent.
        """
        batch = self.batch_tool() if len(queue) >= 2 else None
        before = self.wire.tick
        heads = ", ".join(dict.fromkeys(r["head"] for r in opens))
        responses = self.open_reads(queue, f"The request takes {heads} from another listing: I open the listings that may be it.", batch)
        dropped = False
        if batch and responses and not self.__dict__.get("batch_proven"):
            if self.batch_ok(self.__dict__.get("last_reads") or [], queue):
                self.batch_proven = True
            else:
                self.batch_spec, dropped = None, True
                self.log.append({"kind": "batch_dropped", "tool": batch[0]})
        if responses:
            self.post_replies(responses)
        self.ref_steps = self.__dict__.get("ref_steps", 0) + (self.wire.tick - before)
        self.ref_ids_add(queue)
        self.log.append({"kind": "ref_read", "ids": queue})
        if not dropped:
            unread = [i for i in queue if i not in self.archive.read]
            for ref in opens:
                ref["judged"].update(unread)
        return self.wire.tick > before

    def ref_search(self, ref: dict[str, Any], args: dict[str, Any]) -> bool:
        """Sends one search call with the given arguments to find a referenced listing and adds the spent steps to
        ref_steps. When the results hold indexed rows but none from the target seller, although the query was the
        seller name itself, it marks the reference 'unindexed' so later plans search by detail words. Logs ref_search
        with the query and row count and returns whether a step was spent.
        """
        before = self.wire.tick
        got = self.walk_call(args, f"The request takes {ref['head']} from another listing ({self.ref_desc(ref)}): I search for it.")
        self.ref_steps = self.__dict__.get("ref_steps", 0) + (self.wire.tick - before)
        rows: list[dict[str, Any]] = []
        if got:
            walk_dicts(self.archive.items[-1], lambda d: rows.append(d) if self.row_mark(d) in self.archive.ids and len(d) >= 4 else None, ceiling=3)
        name = ref_name(ref["target"]["seller"])
        text = " ".join(self.free_text(args))
        if rows and not any(name in self.ref_sellers(r) for r in rows) and ref_name(text) == name:
            ref["unindexed"] = True
        self.log.append({"kind": "ref_search", "prop": ref["prop"], "query": text[:120], "rows": len(rows)})
        return self.wire.tick > before

    def ref_judge(self, ref: dict[str, Any]) -> bool:
        """Decides a reference from read, unjudged candidates. Detail loose if 4+
        read seller listings miss it; rank: all non-loose details shown, then
        fully covered count. Short top one, room, more candidates/plans/pages:
        wait. Else may first read <=2 promising listings' notes; stop a
        'differ' ref with notes unreadable in time; code-resolve if only one
        candidate covers all details, notes plain (ref_plain); stop if out of
        model checks; or model-check <=4 picks, apply it. True if acted.
        """
        tgt = ref["target"]
        cands = [i for i in self.ref_cands(ref) if i in self.archive.read and i not in ref["judged"]]
        if not cands:
            return False
        rows = self.ref_index()
        seller_read = [i for i in self.ref_cands(ref) if i in self.archive.read]
        parts = {i: self.ref_parts(rows.get(i) or {}) for i in set(cands) | set(seller_read)}
        loose = ref.setdefault("loose", set())

        def shows(i: str, d: str) -> bool:
            """Tells whether candidate listing i shows detail d, using ref_shows on its indexed row and parsed parts."""
            return self.ref_shows(rows.get(i) or {}, d, parts[i])
        if len(seller_read) >= 4:
            loose |= {d for d in tgt["details"] if not any(shows(i, d) for i in seller_read)}

        def words_ok(i: str) -> bool:
            """True when candidate i shows every target detail that is not marked loose."""
            return all(shows(i, d) for d in tgt["details"] if d not in loose)
        full = {i: sum(1 for d in tgt["details"] if set(ref_toks(d)) and set(ref_toks(d)) <= parts[i][0] | parts[i][1]) for i in cands}
        ranked = sorted(cands, key=lambda i: (not words_ok(i), -full[i], cands.index(i)))
        def fig_clash(i: str) -> bool:
            row = rows.get(i) or {}
            mine = ref_figs(" | ".join([self.title_of(row)] + [str(x) for v in row.values() if isinstance(v, dict)
                                                               for x in v.values() if isinstance(x, (str, int, float))]))
            for d in tgt["details"]:
                for fig, unit in ref_figs(d):
                    same = [f for f, u in mine if u == unit]
                    if same and not any(abs(f - fig) < 1e-9 for f in same):
                        return True      # (RFb: its own title or fields write another figure: the other page cannot help)
            return False
        if not words_ok(ranked[0]) and self.split_pages() and self.ref_room():
            facts = self.ref_facts_missing(ref, [i for i in ranked if not words_ok(i) and not fig_clash(i)])
            if facts and self.ref_facts_read(ref, facts):
                return True       # (RF: layout A shows facts and notes on two pages: the other page is read first)
        if not words_ok(ranked[0]) and ((self.ref_room() or self.ref_waits()) and ([i for i in self.ref_cands(ref) if i not in self.archive.read
                                                                                    and i not in ref["judged"]] or self.ref_plans(ref) or self.ref_page(ref))):
            return False
        pick = [i for i in ranked if words_ok(i)][:4] or ranked[:4]
        exact = [i for i in pick if tgt["details"] and full[i] == len(tgt["details"])]
        missing = self.ref_notes_missing(ref, exact if len(exact) == 1 else ([i for i in pick if words_ok(i)][:2] or pick[:1]))
        if missing and self.ref_room() and self.ref_notes_read(ref, missing):
            return True
        if missing and self.ref_waits():
            return False
        if missing and not self.ref_room() and ref.get("relation") == "differ":
            ref["judged"].update(pick)
            self.ref_stop(ref, "the notes of the listing it points at could not be read in the steps left")
            return True
        if len(exact) == 1 and self.ref_plain(ref, exact[0]):
            ref["judged"].update(pick)
            self.ref_ids_add(pick)
            self.ref_resolve(ref, exact[0], self.ref_field(rows.get(exact[0]) or {}, ref["prop"]), [])
            return True
        if not self.ref_calls_ok():
            ref["judged"].update(pick)
            self.ref_stop(ref, "no hop checks are left")
            return True
        self.ref_apply(ref, pick, self.ref_ask(ref, pick, ""))
        if ref["status"] == "open" and ref["target"] is tgt:
            ref["judged"].update(i for i in cands if not words_ok(i))
        return True

    def ref_plain(self, ref: dict[str, Any], ident: str) -> bool:
        """Tells whether a listing's own field for the reference property can be taken without a model check. It needs a
        field value, notes (non-description text) that neither name the property nor share tokens with the value, and
        no notes word (past each sentence's first) that is another seller's name or a capitalised name of four or more
        letters absent from the listing's sellers, keys, title, shelf and field values.
        """
        row = self.ref_index().get(ident) or {}
        val = self.ref_field(row, ref["prop"])
        notes = self.ref_texts(row, desc=False)
        if not val or self.ref_names_prop(ref["prop"], notes) or set(ref_toks(val)) & set(ref_toks(notes)):
            return False
        own = self.ref_sellers(row)
        elsewhere = {s for r in self.ref_index().values() for s in self.ref_sellers(r)} - own
        keys, _ = self.ref_parts(row)
        fields = keys | set(ref_toks(" | ".join([self.title_of(row), self.shelf_of(row)] + [str(x) for v in row.values() if isinstance(v, dict)
                                                                                           for x in v.values() if isinstance(x, (str, int, float))])))
        for sentence in re.split(r"(?<=[.!?;])\s+", notes):
            for word in re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)[1:]:
                low = ref_name(word)
                if low in elsewhere or (word[:1].isupper() and len(low) >= 4 and low not in own and low not in fields):
                    return False
        return True

    def ref_notes_tool(self) -> str | None:
        """Picks the tool that reads a listing's notes while following a reference. Uses notes_tool when known and
        returns None when the details tool already carries note keys. Otherwise it prefers a lookup tool, other than
        details, whose result keys add text keys beyond the details tool and row keys, the most-called first; then the
        view or stock tool when it is among the lookup tools and its caps record shows no id count.
        """
        tool = self.notes_tool()
        if tool:
            return tool
        details = self.role("details")
        if ((self.__dict__.get("note_keys") or {}).get(details or "")):
            return None
        tkeys = self.__dict__.get("tool_keys") or {}
        texts = self.__dict__.get("text_keys") or {}
        dkeys = set(tkeys.get(details or "") or set())
        rows = set(self.__dict__.get("row_keys") or set())
        caps = self.__dict__.get("tool_caps") or {}
        tools = [t for t in self.lookup_tools() if t != details]
        extra = [t for t in tools if (set(tkeys.get(t) or set()) - dkeys - rows) & set(texts.get(t) or set())]
        if extra:
            return max(extra, key=lambda t: ((caps.get(t) or {}).get("calls", 0), t))
        for tool in (self.role("view"), self.stock_tool()):
            if tool in tools and not (caps.get(tool) or {}).get("id", 0):
                return tool
        return None

    def ref_facts_missing(self, ref: dict[str, Any], ids: list[str]) -> tuple[str, list[str]] | None:
        """(RF) In layout A: a page tool that these candidates (at most two) were never read with, and those of them not
        tried with it for this reference yet; None when every one was read with every page tool."""
        seen = self.__dict__.get("read_with") or {}
        pages = self.__dict__.get("page_seen") or {}
        tried = ref.setdefault("faced", set())
        safe = {self.role("details"), self.role("view")} - {None, self.stock_tool()}
        for tool in self.page_tools():
            if tool not in safe:
                continue      # (RFb: only the details / view page tools)
            need = [i for i in dict.fromkeys(ids) if i not in (seen.get(tool) or set()) and i not in (pages.get(tool) or set())
                    and (tool, i) not in tried]
            if need and self.lookup_call(need[0], tool):
                return tool, need[:1]      # (RFb: one candidate a round)
        return None

    def ref_facts_read(self, ref: dict[str, Any], got: tuple[str, list[str]]) -> bool:
        """(RF) One scripted read of candidates through their other page tool; True when a step went out."""
        tool, need = got
        ref.setdefault("faced", set()).update((tool, i) for i in need)
        before = self.wire.tick
        responses = self.open_reads(need, f"The request takes {ref['head']} from another listing: I open the other page of the "
                                    f"listing that may be it ({tool}), since its details may sit there.", None, tool)
        if responses:
            self.post_replies(responses)
        self.ref_steps = self.__dict__.get("ref_steps", 0) + (self.wire.tick - before)
        self.ref_ids_add(need)
        self.log.append({"kind": "ref_facts", "ids": need, "tool": tool})
        return self.wire.tick > before

    def ref_notes_missing(self, ref: dict[str, Any], ids: list[str]) -> list[str]:
        """Lists up to two of the given ids whose notes were neither read with the notes tool nor tried for this
        reference. Returns [] when there is no notes tool or no ids.
        """
        tool = self.ref_notes_tool()
        if not tool or not ids:
            return []
        seen = set((self.__dict__.get("read_with") or {}).get(tool) or set())
        tried = ref.setdefault("noted", set())
        return [i for i in dict.fromkeys(ids) if i not in seen and i not in tried][:2]

    def ref_notes_read(self, ref: dict[str, Any], need: list[str]) -> bool:
        """Reads the notes of the given candidate listings with the notes tool, since a note may send the reader on to
        another listing. Marks them in ref['noted'] and ref_ids, posts the replies, adds the spent steps to ref_steps
        and logs ref_notes. Returns whether a step was spent.
        """
        tool = self.ref_notes_tool()
        if not tool or not need:
            return False
        ref.setdefault("noted", set()).update(need)
        before = self.wire.tick
        responses = self.open_reads(need, f"The request takes {ref['head']} from another listing: I read the notes of the listing "
                                    f"that may be it ({tool}), since a note may send the reader on.", None, tool)
        if responses:
            self.post_replies(responses)
        self.ref_steps = self.__dict__.get("ref_steps", 0) + (self.wire.tick - before)
        self.ref_ids_add(need)
        self.log.append({"kind": "ref_notes", "ids": need, "tool": tool})
        return self.wire.tick > before

    def ref_pointers(self) -> set[str]:
        """Collects listing ids on the chains of live 'differ' references (open, stuck or resolved) that belong to the
        reference's own product: listings whose value the shopper's value must differ from.
        """
        out: set[str] = set()
        for ref in self.__dict__.get("refs") or []:
            if ref.get("relation") == "differ" and ref.get("status") in ("open", "stuck", "resolved") and self.ref_live(ref):
                out.update(i for i in ref.get("chain") or [] if self.ref_own_product(ref, i))
        return out

    def ref_own_product(self, ref: dict[str, Any], ident: str) -> bool:
        """Tells whether a listing belongs to the reference's product: true with no product rows, when its shelf is
        unknown or maps to no product, when the reference has no tag, or when shelf and tag map to the same product.
        """
        if not self.needs.items():
            return True
        shelf = self.shelf_known(ident)
        k = self.product_of(f"x ({shelf})") if shelf else None
        return k is None or not ref.get("tag") or k == self.product_of(f"x ({ref['tag']})")

    def ref_pointer_note(self, thing: str) -> str:
        """Returns a note saying that a listing is the one a live 'differ' reference points at (or one its notes lead
        through), so its value is the very one the shopper's value must differ from and it cannot meet that requirement.
        Returns an empty string when no such reference covers it.
        """
        for ref in self.__dict__.get("refs") or []:
            if ref.get("relation") == "differ" and thing in (ref.get("chain") or []) and self.ref_live(ref) and self.ref_own_product(ref, thing):
                what = f" ({ref['value']})" if ref.get("status") == "resolved" and ref.get("value") else ""
                return (f"{thing} is the listing the shopper's reference for {ref['head']} points at, or one its notes lead "
                        f"through, so its {ref['prop']} is the very value{what} the shopper's {ref['prop']} must differ from: it cannot "
                        "meet that requirement")
        return ""

    def ref_excluded(self, fields: dict[str, str], k: int | None) -> bool:
        """Tells whether a candidate's fields hold the value that a resolved, live 'differ' reference rules out.
        References tagged for a product other than k are skipped; a field counts when its key names the reference
        property and its normalised value equals the resolved value.
        """
        rows = self.needs.items()
        for ref in self.__dict__.get("refs") or []:
            if ref.get("relation") != "differ" or ref.get("status") != "resolved" or not self.ref_live(ref):
                continue
            if rows and ref.get("tag") and self.product_of(f"x ({ref['tag']})") not in (k, None):
                continue
            bad = {" ".join(ref_toks(v)) for v in [ref.get("value")] if str(v or "").strip()}
            for key, val in (fields or {}).items():
                if same_prop(key, ref["prop"]) and " ".join(ref_toks(val)) in bad:
                    return True
        return False

    def own_redirects(self, thing: str) -> list[tuple[str, str, str]]:
        """Sentences in a listing's own notes that seem to send an open requirement's
        property to another listing: a sentence naming the property and, after it,
        another seller's name or a foreign capitalised name of 4+ letters,
        not negated within five words before.
        Words before 'by <Name>' are blanked for the property check.
        Returns (requirement, property, sentence) triples, or [].
        """
        try:
            rows = self.ref_index()
            row = rows.get(thing) or {}
            notes = self.ref_texts(row, desc=False)
            if not notes:
                return []
            own = self.ref_sellers(row)
            elsewhere = {s for r in rows.values() for s in self.ref_sellers(r)} - own
            keys, _ = self.ref_parts(row)
            fields = keys | set(ref_toks(" | ".join([self.title_of(row), self.shelf_of(row)] + [str(x) for v in row.values() if isinstance(v, dict)
                                                                                               for x in v.values() if isinstance(x, (str, int, float))])))
            several = self.needs.items()
            k = self.product_of(f"x ({self.shelf_of(row)})") if several and self.shelf_of(row) else None
            out = []
            for req in self.needs.props_left():
                head, _, _ = str(req).partition(":")
                if several and k is not None and self.product_of(head) not in (k, None):
                    continue
                prop = re.split(r"\s*\(", head)[0].strip()
                if not prop:
                    continue
                for sentence in re.split(r"(?<=[.!?;])\s+", notes):
                    words = list(re.finditer(r"[A-Za-z][A-Za-z'-]*", sentence))
                    seen = re.sub(r"[A-Za-z0-9]+(?=\s+by\s+[A-Z])", lambda h: " " * len(h.group(0)), sentence)
                    first_prop = next((m.start() for m in words if prop_named(prop, seen[:m.end()])), None)
                    if first_prop is None:
                        continue
                    for n, m in enumerate(words[1:], 1):
                        word = m.group(0)
                        low = ref_name(word)
                        if m.start() <= first_prop or not low:
                            continue
                        other = low in elsewhere or (word[:1].isupper() and len(low) >= 4 and low not in own
                                                     and not set(ref_toks(word)) & fields and low not in fields)
                        negated = any(w.group(0).lower() in ("not", "nor", "never", "no", "neither") for w in words[max(0, n - 5): n])
                        if other and not negated:
                            out.append((str(req), prop, sentence.strip()))
                            break
                    else:
                        continue
                    break
            return out
        except Exception:
            return []

    def notes_remark(self, call_name: str, call_args: dict[str, Any]) -> str:
        """Builds a remark for a tool reply about listings whose own notes may redirect a requirement elsewhere.
        Skips search, cart, cart view, message and finishing calls. For each asked id that is in the cart or passed the
        item check under the current needs version, and was not told yet, it runs own_redirects; on a hit it withdraws
        the pass so the check reruns, logs own_redirect, and says the listing's own field does not count, asking for a
        swap when the item is in the cart.
        """
        if call_name in (self.role("search"), self.role("cart"), self.role("cart_view"), self.role("message")) or call_name in self.finishers():
            return ""
        ver = str(self.needs.version)
        parts = []
        for ident in sorted(self.asked_ids(call_args)):
            key = f"{ident}|{ver}"
            if ident not in self.trolley.units and key not in self.passed:
                continue
            told = self.__dict__.setdefault("own_told", set())
            own = self.own_redirects(ident) if key not in told else []
            if not own:
                continue
            told.add(key)
            self.passed.discard(key)
            self.log.append({"kind": "own_redirect", "id": ident, "props": [p for _, p, _ in own]})
            what = "; ".join(f"{p} (\"{s[:160]}\")" for _, p, s in own)
            parts.append(f"{ident}'s own notes may send {what} to another listing: if they do, its own field does not count for "
                         + "; ".join(f"'{r}'" for r, _, _ in own) + " (the value comes from the other listing), so it meets "
                         + ("them" if len(own) > 1 else "it") + " only when that listing's record shows it; the item check runs again "
                         "before it is ordered" + (". If they do, take it out of the cart and put another verified fit in."
                                                   if ident in self.trolley.units else "."))
        return " ".join(parts)

    def ref_phrase(self, line: str) -> tuple[str, int, int, dict[str, Any]] | None:
        """Cuts the reference phrase out of a shopper line that points to another listing. Parses the line with
        ref_parse, requires an accepted seller name and finds where it is written. The phrase starts at the first
        differ, reference or pointing word of the clause before the seller name and ends at the sentence end, after a
        trailing '(or ...)' aside, or before a comma followed by a vocative or pronoun. Returns the phrase, its start
        and end offsets and the parsed target, or None.
        """
        text = str(line or "")
        if not ref_text(text):
            return None
        parsed = ref_parse(text, lambda name: self.ref_seller_ok(name, text))
        if not parsed or not self.ref_seller_ok(parsed["seller"], text):
            return None
        hit = re.search(r"(?<![A-Za-z'])" + re.escape(parsed["seller"]) + r"(?:'s)?(?![A-Za-z])", text)
        if not hit:
            return None
        head = text[:hit.start()]
        cut = max([head.rfind(c) for c in (",", ";", "!", "?")] + [m.start() for m in re.finditer(r"[.](?:\s|$)", head)] + [-1])
        lead = head[cut + 1:]
        marks = [m.start() for pat in (DIFFER_WORDS, REF_WORDS, POINT_WORDS) for m in re.finditer(pat, lead.lower())]
        if not marks:
            return None
        start = cut + 1 + min(marks)
        tail = text[hit.end():]
        aside = re.search(r"\(\s*or\b[^)]*\)", tail)
        stop = re.search(r"[.!?;](?:\s|$)", tail)
        if aside and (not stop or aside.start() < stop.start()):
            end = hit.end() + aside.end()
        else:
            end = hit.end() + (stop.start() if stop else len(tail))
        junk = re.search(r",\s*(?:(?:and|but|so|or)\s+)?(?:mate|matey|friend|sir|madam|love|pal|buddy|lad|lass|dear|captain|skipper"
                         r"|i|i'm|i'll|i've|i'd|we|you|that's|it's)\b", tail, re.I)
        if junk and hit.end() + junk.start() < end:
            end = hit.end() + junk.start()
        phrase = text[start:end].strip(" ,;")
        if not phrase:
            return None
        return phrase, start, end, parsed

    def ref_line_words(self, qtext: str | None, outside: str) -> list[str]:
        """Finds words of the question text that the line outside the reference phrase names as a property, the last
        tier ref_fix uses to place a reference line. Tries runs of three, two and then one question words, skipping runs
        that hold a link word, a word under three letters or a word of the product kinds (the product rows, or the single
        product read from the request). Keeps each run that prop_named finds in the outside text; returns [] when there
        is no question text.
        """
        if not qtext:
            return []
        kinds: set[str] = set()
        for row in self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}]:
            kinds |= set(re.findall(r"[a-z]+", str(row["product"]).lower()))
        q = re.findall(r"[a-z0-9]+", str(qtext).lower())
        out = []
        for n in (3, 2, 1):
            for i in range(len(q) - n + 1):
                run = q[i: i + n]
                if any(w in LINK_WORDS or len(w) < 3 or w in kinds or w.rstrip("s") in kinds for w in run):
                    continue
                if prop_named(" ".join(run), outside):
                    out.append(" ".join(run))
        return out

    def ref_fix(self, answer: dict[str, Any], lines: list[str], qtext: str | None) -> list[str]:
        """Fixes model reads of shopper lines that copy a property from another listing.
        Each ref_phrase hit is masked; its property is found by tier
        (model refs, need names, asked labels, store keys, question words),
        else ref_line_unplaced is logged.
        Several products: the tag comes from the line, needs or question.
        Adds one '<head>: <phrase>' need and drops needs only on the referenced listing.
        Cleans dropped and no_preference entries.
        Sets answer['references'] and answer['ref_needs'].
        Logs ref_line.
        """
        masked = []
        for line in lines:
            got = self.ref_phrase(line)
            if not got:
                masked.append(line)
                continue
            phrase, start, end, parsed = got
            outside, inside = line[:start] + " " + line[end:], line[start:end]
            masked.append(outside)
            needs = [n for n in answer.get("needs") or [] if isinstance(n, str) and ":" in n] if isinstance(answer.get("needs"), list) else []
            refs = [r for r in answer.get("references") or [] if isinstance(r, dict)] if isinstance(answer.get("references"), list) else []

            def name(req: Any) -> str:
                """Returns a requirement's property name: the text before the colon, without any tag in parentheses."""
                return re.split(r"\s*\(", str(req).partition(":")[0])[0].strip()

            def inside_only(prop: str) -> bool:
                """True when the property is named inside the reference phrase and not in the rest of the line."""
                return bool(prop) and prop_named(prop, inside) and not prop_named(prop, outside)

            def detail(req: str) -> bool:
                """Tells whether a requirement only describes the referenced listing: its property is named only inside the
                phrase, or all its value tokens appear inside the phrase and none outside it.
                """
                vals = set(ref_toks(str(req).partition(":")[2]))
                return inside_only(name(req)) or (bool(vals) and vals <= set(ref_toks(inside)) and not vals & set(ref_toks(outside)))
            asked = [name(a) for a in str(self.__dict__.get("asking_label") or "").split(";") if name(a)]
            tiers = [[str(r.get("property") or "") for r in refs],
                     [name(n) for n in needs if ref_text(n.partition(":")[2]) or re.search(DIFFER_WORDS, n.partition(":")[2].lower())],
                     [name(n) for n in needs],
                     asked,
                     sorted(self.store_keys()) + [name(r) for r in self.needs.written],
                     self.ref_line_words(qtext, outside)]
            prop = ""
            for tier in tiers:
                named = [p for p in tier if p and prop_named(p, outside)]
                if named:
                    prop = max(named, key=lambda p: (prop_named(p, outside, exact=True), len(re.findall(r"[a-z0-9]+", p.lower())), len(p)))
                    break
            if not prop:
                prop = next((p for p in tiers[0] + tiers[1] + (asked if len(asked) == 1 else []) if p and not inside_only(p)), "")
            if not prop:
                self.log.append({"kind": "ref_line_unplaced", "line": line[:200]})
                continue
            several = self.needs.items()
            tag = ""
            if several:
                k = self.product_of(outside)
                if k is None:
                    tags = [t for n in needs if same_prop(name(n), prop) for t in re.findall(r"\(([^)]*)\)", n.partition(":")[0])]
                    k = self.product_of(f"x ({tags[-1]})") if tags else None
                if k is None and qtext:
                    k = self.product_of(qtext)
                tag = several[k]["product"] if k is not None else ""

            def same_item(req: Any) -> bool:
                """True unless a product tag is set and the requirement's own tag shares no kind words with it."""
                tags = re.findall(r"\(([^)]*)\)", str(req).partition(":")[0])
                return not several or not tag or not tags or bool(kind_words(tags[-1]) & kind_words(tag))
            mine = [n for n in needs if same_prop(name(n), prop) and same_item(n)]
            rel, tol, unit = ref_relation(phrase)
            keep = next((n for n in mine if (ref_text(n.partition(":")[2]) or re.search(DIFFER_WORDS, n.partition(":")[2].lower()))
                         and ref_relation(n.partition(":")[2])[0] == rel and ref_name(parsed["seller"]) in ref_name(n.partition(":")[2])), None)
            head = mine[0].partition(":")[0].strip() if mine else (f"{prop} ({tag})" if tag else prop)
            if several and tag and not re.findall(r"\(([^)]*)\)", head):
                head = f"{head} ({tag})"
            need = keep or f"{head}: {phrase}"
            details = [n for n in needs if n not in mine and detail(n)]
            spot = needs.index(mine[0]) if mine else len(needs)
            rest = [n for n in needs if n not in mine and n not in details]
            answer["needs"] = rest[:spot] + [need] + rest[spot:]
            now = answer.get("requirements_now")
            if isinstance(now, list):
                kept = []
                for req in now:
                    if not isinstance(req, str):
                        continue
                    if req not in self.needs.written and (req in details or detail(req)):
                        continue
                    if same_prop(name(req), prop) and same_item(req) and req not in self.needs.written:
                        continue
                    kept.append(req)
                answer["requirements_now"] = kept + ([need] if need not in kept else [])
            if isinstance(answer.get("dropped"), list):
                answer["dropped"] = [d for d in answer["dropped"] if not (isinstance(d, str) and (inside_only(name(d)) or detail(d)))]
            if isinstance(answer.get("no_preference"), list):
                answer["no_preference"] = [p for p in answer["no_preference"] if not (isinstance(p, str) and (inside_only(name(p)) or same_prop(name(p), prop)))]
            entry = next((r for r in refs if same_prop(str(r.get("property") or ""), prop)), None)
            if entry is None:
                entry = {"property": prop}
                refs.append(entry)
            if not self.ref_seller_ok(entry.get("seller"), line) or ref_name(entry.get("seller")) != ref_name(parsed["seller"]):
                entry.update(seller=parsed["seller"], kind=parsed["kind"], details=list(parsed["details"]))
            elif not entry.get("details"):
                entry["details"] = list(parsed["details"])
            entry.setdefault("kind", parsed["kind"])
            if rel != "same" or not entry.get("relation"):
                entry.update(relation=rel, tolerance=tol, unit=unit)
            answer["references"] = refs
            answer["ref_needs"] = list(answer.get("ref_needs") or []) + [need]
            self.log.append({"kind": "ref_line", "prop": prop, "tag": tag, "relation": rel, "need": need[:200], "kept": bool(keep),
                             "details_out": [d[:80] for d in details]})
        return masked

    @staticmethod
    def ref_names_prop(prop: str, text: str) -> bool:
        """Tells whether text mentions a property: a word of three or more letters from the property name appears, or
        shares its first four letters with a text word. True when the property has no such words.
        """
        words = [w for w in prop_words(prop) if len(w) >= 3]
        toks = set(re.findall(r"[a-z]+", str(text).lower()))
        return not words or any(t == w or (len(w) >= 4 and len(t) >= 4 and t[:4] == w[:4]) for t in toks for w in words)

    def ref_card(self, ident: str) -> str:
        """Formats one indexed listing as a text card for the reference model check: id, sellers (short strings under
        seller-like or learned seller keys, leaving out id and note-like keys), category shelf, title, up to 40 scalar
        facts from each nested dict, and notes, each cut to length and the card to 5000 characters.
        """
        row = self.ref_index().get(ident) or {}
        facts: dict[str, str] = {}
        for val in row.values():
            if isinstance(val, dict):
                for k, x in list(val.items())[:40]:
                    if isinstance(x, (str, int, float)) and not isinstance(x, bool):
                        facts[str(k)] = str(x)[:90]
        learned = self.seller_keys()
        sellers = sorted({str(v) for k, v in row.items() if isinstance(v, str) and v.strip() and len(v) <= 60 and "id" not in str(k).lower()
                          and (re.search(SELLER_KEYS, str(k).lower()) or str(k) in learned)
                          and not re.search(r"note|text|desc|comment|detail|info|message|section", str(k).lower())})
        card = {"id": ident, "sold by": " / ".join(sellers), "category": self.shelf_of(row), "title": self.title_of(row)[:160],
                "facts": facts, "notes": self.ref_texts(row)[:900]}
        return as_text(card, 5000)

    def ref_ask(self, ref: dict[str, Any], pick: list[str], reason: str) -> dict[str, Any] | None:
        """Sends the REF_HOP check to the model for the picked candidates, filling in the reference's property and
        target description and adding each pick's card. A reason given for an unusable earlier answer is appended so the
        model answers again. Counts the call in ref_calls, logs ref_hop and returns the answer, or None when no checks
        are left.
        """
        if not self.ref_calls_ok():
            return None
        self.ref_calls = self.__dict__.get("ref_calls", 0) + 1
        ask_text = REF_HOP.replace("PROPERTY", ref["prop"]).replace("DESCRIPTION", self.ref_desc(ref))
        ask_text += "\n\nRECORDS:\n" + "\n".join(self.ref_card(i) for i in pick)
        if reason:
            ask_text += "\n\nYOUR LAST ANSWER COULD NOT BE USED: " + reason + " Answer again from the records."
        answer = self.wire.consult(ask_text, 3000, CHECKER[:1])
        self.log.append({"kind": "ref_hop", "prop": ref["prop"], "ids": pick, "answer": answer})
        return answer

    def ref_apply(self, ref: dict[str, Any], pick: list[str], answer: Any, again: bool = True) -> None:
        """Acts on the model's answer about which picked listing is the referenced one.
        A non-dict answer is a failure; a second one stops the reference.
        A target outside the pick, or missing details, marks the pick judged.
        Else ref_validate checks it: a problem earns one retry with the reason.
        A valid answer stops at a target already on the chain, follows a go_to,
        or resolves the value.
        """
        if not isinstance(answer, dict):
            ref["fails"] += 1
            if ref["fails"] >= 2:
                ref["judged"].update(pick)
                self.ref_stop(ref, "the hop check gave no answer")
            return
        target = answer.get("target")
        target = str(target).strip() if isinstance(target, (str, int)) and not isinstance(target, bool) else ""
        missing = [str(x) for x in answer.get("missing") or [] if str(x).strip()] if isinstance(answer.get("missing"), list) else []
        if target not in pick or missing:
            ref["judged"].update(pick)
            self.ref_ids_add(pick)
            if target in pick:
                self.log.append({"kind": "ref_partial", "prop": ref["prop"], "target": target, "missing": missing[:4]})
            return
        where, problem, go, value, pairs = self.ref_validate(ref, target, answer, pick)
        if problem and again and self.ref_calls_ok():
            self.ref_apply(ref, pick, self.ref_ask(ref, pick, problem), again=False)
            return
        self.ref_ids_add(pick)
        if problem:
            self.log.append({"kind": "ref_invalid", "prop": ref["prop"], "target": target, "problem": problem[:300]})
            if where == "target":
                ref["judged"].add(target)
                return
            ref["judged"].update(pick)
            ref["chain"].append(target)
            self.ref_stop(ref, problem)
            return
        ref["judged"].update(pick)
        if target in ref["chain"]:
            self.ref_stop(ref, "its notes lead back to a listing already followed")
        elif go:
            self.ref_go(ref, target, go)
        else:
            self.ref_resolve(ref, target, value, pairs)

    def ref_validate(self, ref: dict[str, Any], target: str, answer: dict[str, Any], pick: list[str] | None = None
                     ) -> tuple[str, str, dict[str, Any] | None, str, list[str]]:
        """Checks a model answer about a reference target against its record: (where, problem, go, value, pairs).
        'target' when the record misses a detail or another pick covers more; 'answer' when the answer is unusable.
        A go_to must name a seller written in the notes beside the property, and not the target itself.
        Its details are kept when half their tokens appear in the notes.
        Else the value must appear in the record and goes_with values in the notes.
        For a counterpart, a pairing rule found in the notes replaces them.
        """
        row = self.ref_index().get(target) or {}
        keys, vals = self.ref_parts(row)
        for d in [d for d in ref["target"]["details"] if d not in (ref.get("loose") or set())]:
            if not self.ref_shows(row, d, (keys, vals)):
                return "target", f"{target} does not show '{d}', so it is not the listing described.", None, "", []

        def shown(ident: str) -> int:
            """Counts the target details whose tokens all appear in a listing's parsed keys and values."""
            k, v = self.ref_parts(self.ref_index().get(ident) or {})
            return sum(1 for d in ref["target"]["details"] if set(ref_toks(d)) and set(ref_toks(d)) <= k | v)
        best = max(pick or [target], key=shown)
        if shown(best) > shown(target):
            return "target", f"{best} shows more of the details named than {target}.", None, "", []
        notes = self.ref_texts(row)
        ntoks = set(ref_toks(notes))
        go = answer.get("go_to") if isinstance(answer.get("go_to"), dict) else None
        if go and str(go.get("seller") or "").strip():
            seller = str(go["seller"]).strip()
            pattern = r"(?<![a-z0-9])" + re.escape(seller.lower()) + r"(?:'s)?(?![a-z0-9])"
            if not re.search(pattern, notes.lower()):
                return "answer", f"the go_to seller '{seller}' is not written in {target}'s notes.", None, "", []
            if not any(re.search(pattern, s.lower()) and self.ref_names_prop(ref["prop"], s) for s in re.split(r"(?<=[.!?])\s+", notes)):
                return "answer", f"the note naming {seller} is about another property, not {ref['prop']}.", None, "", []
            raw = go.get("details") if isinstance(go.get("details"), list) else []
            dets = [str(d).strip() for d in raw if isinstance(d, (str, int, float)) and not isinstance(d, bool) and str(d).strip()]
            dets = [d for d in dets if set(ref_toks(d)) and 2 * len(set(ref_toks(d)) & ntoks) >= len(set(ref_toks(d)))][:6]
            nxt = {"seller": seller, "kind": str(go.get("kind") or "").strip()[:80], "details": dets}
            if ref_name(seller) in self.ref_sellers(row) and self.ref_key(nxt) == self.ref_key(ref["target"]):
                return "answer", f"the go_to is {target} itself.", None, "", []
            return "", "", nxt, "", []
        value = str(answer.get("value") or "").strip()
        if not value or not set(ref_toks(value)) or not set(ref_toks(value)) <= vals | keys:
            return "answer", f"the value '{value}' is not written in {target}'s record.", None, "", []
        raw = answer.get("goes_with") if isinstance(answer.get("goes_with"), list) else []
        pairs = [str(p).strip() for p in raw if isinstance(p, (str, int, float)) and not isinstance(p, bool) and str(p).strip()]
        pairs = [p for p in dict.fromkeys(pairs) if set(ref_toks(p)) and set(ref_toks(p)) <= ntoks and set(ref_toks(p)) != set(ref_toks(value))][:3]
        if ref.get("relation") == "counterpart":
            rule = rule_partners(notes, value)
            if rule:
                pairs = rule[:3]
        return "", "", None, value, pairs

    def ref_go(self, ref: dict[str, Any], target: str, go: dict[str, Any]) -> None:
        """Moves a reference on to the listing the target's notes send the reader to. Puts the target on the chain and
        stops when the new target was followed before or REF_HOPS would be reached; otherwise counts the hop and resets
        the target, judged set, queries, pages and failure counts. Logs ref_go and tells the model the code follows the
        notes.
        """
        ref["chain"].append(target)
        key = self.ref_key(go)
        if key in ref["seen"]:
            self.ref_stop(ref, "its notes lead back to a listing already followed")
            return
        if ref["hops"] + 1 >= REF_HOPS:
            self.ref_stop(ref, f"more than {REF_HOPS} listings in a row send the reader elsewhere")
            return
        ref["seen"].add(key)
        ref["hops"] += 1
        ref.update(target=go, judged=set(), queries=[], pages=0, tries=0, fails=0, unindexed=False, loose=set())
        self.log.append({"kind": "ref_go", "prop": ref["prop"], "from": target, "to": go})
        self.ref_tell(f"{target} is the listing named for {ref['head']}, and its notes send the reader on to {self.ref_desc(ref)}: the "
                      "code follows them.")

    def ref_value(self, ref: dict[str, Any]) -> str:
        """Turns a resolved reference into requirement value text by relation. 'within' with a numeric tolerance gives
        a range around the number with its unit; 'at most' and 'at least' prefix the value; 'counterpart' joins the
        paired values with 'or' (the value itself when none differ); 'differ' gives 'not <value>'; any other relation
        keeps the value as read.
        """
        value, pairs = str(ref["value"]), list(ref.get("goes_with") or [])
        if ref["relation"] == "within" and is_number(ref.get("tol")):
            hit = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)", value.lower().replace(",", ""))
            if hit:
                base, tol = float(hit.group(1)), float(ref["tol"])
                unit = hit.group(2) or str(ref.get("unit") or "")
                return f"{round(base - tol, 6):g} to {round(base + tol, 6):g} {unit}".strip() + f" (within {tol:g} of {value})"
        if ref["relation"] in ("at most", "at least"):
            return f"{ref['relation']} {value}"
        if ref["relation"] == "counterpart":
            partners = [p for p in pairs if p.lower() != value.lower()]
            return " or ".join(partners) if partners else value
        if ref["relation"] == "differ":
            return f"not {value}"
        return value

    def ref_resolve(self, ref: dict[str, Any], target: str, value: str, pairs: list[str]) -> None:
        """Marks a reference resolved with the value read on the target listing, or stops it when that value is empty.
        Puts the target on the chain, stores value and goes_with, builds '<head>: <value text>', swaps it into the needs
        with ref_swap, records it in needs.resolved and bumps needs.version. Logs ref_resolved (and ref_counterpart) and
        tells the model the outcome, worded for pairings, counterparts and 'differ' relations.
        """
        if not str(value or "").strip():
            self.ref_stop(ref, f"{target} shows no value for it")
            return
        ref["chain"].append(target)
        ref.update(status="resolved", value=str(value).strip(), goes_with=list(pairs))
        new_req = f"{ref['head']}: {self.ref_value(ref)}"
        ref["now"] = new_req
        self.ref_swap(ref, new_req)
        self.needs.resolved.append(new_req)
        self.needs.version += 1
        self.log.append({"kind": "ref_resolved", "req": ref["req"][:200], "now": new_req, "chain": list(ref["chain"])})
        via = f", reached from {ref['chain'][0]} through the listings' notes" if len(ref["chain"]) > 1 else ""
        pair = (f"; its notes pair {' and '.join(pairs)} with it, which is not the same value and does not meet it" if pairs else "")
        if ref["relation"] == "counterpart":
            pair = (f"; the store's pairing rule there pairs {ref['value']} with {' and '.join(pairs)}, and the shopper wants the "
                    f"counterpart, so {' or '.join(pairs)} meets it and {ref['value']} does not" if pairs else
                    "; no pairing rule there names that value, so the page's own value stands")
            self.log.append({"kind": "ref_counterpart", "prop": ref["prop"], "value": ref["value"], "partners": list(pairs)})
        if ref["relation"] == "differ":
            pair = (f"; its notes pair {' and '.join(pairs)} with it" if pairs else "") + (
                ". The shopper's value MUST DIFFER from that one: any other value meets it, that value does not")
        self.ref_tell(f"RESOLVED: {ref['head']} is given by another listing, and {target} shows {ref['value']}{via}{pair}. The "
                      f"requirement is now: {new_req}.")

    def ref_swap(self, ref: dict[str, Any], new_req: str) -> None:
        """Replaces the requirement a reference stood for with the resolved one.
        Matches by ref_match, or by the property and the first target's seller.
        Updates the written and uncovered lists and now_reqs.
        Adds the new requirement to uncovered when nothing matched.
        Mirrors written into reading['requirements'].
        Rewrites the per-product reading rows of the same product.
        """
        nd = self.needs
        seller = ref_name(ref["first"]["seller"])

        def mine(req: Any) -> bool:
            """Tells whether a requirement belongs to this reference: ref_match accepts it, or it names the reference
            property and its value mentions the first target's seller, with any product tag fitting the reference tag.
            """
            if self.ref_match(ref, req):
                return True
            head, _, value = str(req).partition(":")
            tags = re.findall(r"\(([^)]*)\)", head)
            if tags and ref["tag"] and not kind_words(tags[-1]) & kind_words(ref["tag"]):
                return False
            return bool(seller) and seller in ref_name(value) and same_prop(re.split(r"\s*\(", head)[0], ref["prop"])
        hit = False
        for name in ("written", "uncovered"):
            old = list(getattr(nd, name))
            new = [new_req if mine(r) else r for r in old]
            hit = hit or new != old
            setattr(nd, name, list(dict.fromkeys(new)))
        if nd.now_reqs is not None:
            new = [new_req if mine(r) else r for r in nd.now_reqs]
            nd.now_reqs = list(dict.fromkeys(new + ([] if new_req in new else [new_req])))
        if not hit:
            nd.uncovered.append(new_req)
        if isinstance(nd.reading.get("requirements"), list):
            nd.reading["requirements"] = list(nd.written)
        value = new_req.partition(":")[2].strip()
        rows = [row for row in nd.reading.get("items") or [] if isinstance(row, dict) and isinstance(row.get("requirements"), list)]
        for row in rows:
            if len(rows) >= 2 and ref["tag"] and not (kind_words(row.get("product") or "") & kind_words(ref["tag"])):
                continue
            row["requirements"] = [f"{str(r).partition(':')[0].strip()}: {value}" if mine(r) else r for r in row["requirements"]]

    def ref_stop(self, ref: dict[str, Any], why: str) -> None:
        """Marks an open reference 'stuck' with a reason, logs ref_stuck with its chain, and tells the model the value
        stays unknown and must never be the reason to decline or reject a listing.
        """
        if ref["status"] != "open":
            return
        ref["status"], ref["why"] = "stuck", why
        self.log.append({"kind": "ref_stuck", "req": ref["req"][:200], "why": why[:300], "chain": list(ref["chain"])})
        self.ref_tell(f"NOT RESOLVED: {ref['head']} ({why}). Its value stays unknown: if a listing's notes you read show where it "
                      "ends, use that value; otherwise never decline or reject a listing over it.")

    def ref_tell(self, text: str) -> None:
        """Appends a '[reference]' message to the last chat turn when it is a tool or user text turn past the opening
        two; otherwise queues the text in self.ref_told for the next ref_line.
        """
        if self.chat and self.chat[-1].get("role") in ("tool", "user") and isinstance(self.chat[-1].get("content"), str) and len(self.chat) > 2:
            self.chat[-1]["content"] += "\n[reference] " + text
        else:
            self.ref_told = list(self.__dict__.get("ref_told") or []) + [text]

    def ref_line(self) -> str:
        """Builds the '[references]' status text for the chat: resolved requirements with the listing they were read
        on (and a must-differ mark), open ones with the current target description, stuck ones as unknown. Appends and
        clears the queued ref_told messages, and returns an empty string on error.
        """
        try:
            parts = []
            for ref in self.__dict__.get("refs") or []:
                if ref["status"] == "resolved":
                    parts.append(f"{ref['now']} (read on {ref['chain'][-1]}" + (", must differ from it" if ref.get("relation") == "differ" else "") + ")")
                elif ref["status"] == "open":
                    parts.append(f"{ref['head']}: the code is following it (now: {self.ref_desc(ref)})")
                elif ref["status"] == "stuck":
                    parts.append(f"{ref['head']}: not resolved, value unknown (never decline or reject a listing over it)")
            told = self.__dict__.pop("ref_told", None)
            return ("\n[references] " + "; ".join(parts) if parts else "") + ("".join("\n[reference] " + t for t in told) if told else "")
        except Exception:
            return ""

    def ref_note(self, thing: str = "") -> str:
        """Builds a prompt section on requirements given by reference to another listing, used in item checks.
        Resolved references state the value read on the last chain listing, where the chain started when it hopped, any
        pairings, and for 'counterpart' or 'differ' relations what meets the requirement and what does not. Open or
        stuck ones say the code did not resolve them and how to use a record that is that listing; their last two chain
        ids (other than thing) are gathered and up to two archive records of them are appended. Returns an empty string
        when nothing applies or on error.
        """
        try:
            lines, ends = [], []
            for ref in self.__dict__.get("refs") or []:
                if ref["status"] == "resolved":
                    via = f" (reached from {ref['chain'][0]} through the listings' notes)" if len(ref["chain"]) > 1 else ""
                    pair = (f"; its notes pair {' and '.join(ref['goes_with'])} with it, which is not the same value and does not meet it"
                            if ref["goes_with"] else "")
                    if ref.get("relation") == "counterpart":
                        pair = (f"; the store's pairing rule there pairs {ref['value']} with {' and '.join(ref['goes_with'])}: the "
                                f"shopper wants that counterpart, NOT {ref['value']} itself" if ref["goes_with"] else "")
                        lines.append(f"- {ref['now']}: the counterpart of the value shown on {ref['chain'][-1]}{via}{pair}.")
                        continue
                    if ref.get("relation") == "differ":
                        pair = (f"; its notes pair {' and '.join(ref['goes_with'])} with it" if ref["goes_with"] else "")
                        lines.append(f"- {ref['now']}: the shopper's {ref['prop']} MUST DIFFER from the value shown on {ref['chain'][-1]} "
                                     f"({ref['value']}){via}{pair}: any other value meets it, that value does not; the listings of its "
                                     f"chain ({', '.join(ref['chain'])}) carry it.")
                    else:
                        lines.append(f"- {ref['now']}: the value shown on {ref['chain'][-1]}{via}{pair}.")
                elif ref["status"] in ("open", "stuck"):
                    lines.append(f"- {ref['head']}: NOT RESOLVED by the code (" + ("its chain so far: " + " -> ".join(ref["chain"]) if ref["chain"]
                                 else "the listing it points at was not found") + "). When a record below is that listing, or the last "
                                 "listing its chain reached, and writes its own value for it without sending the reader elsewhere, that "
                                 "value is the requirement; otherwise its value is unknown, so never fail an item on it alone.")
                    ends += [i for i in ref["chain"][-2:] if i != thing]
            if not lines:
                return ""
            text = ("\n\nREQUIREMENTS GIVEN BY REFERENCE TO ANOTHER LISTING (the code followed that listing's notes in the store's "
                    "records):\n" + "\n".join(lines))
            recs = [r for i in dict.fromkeys(reversed(ends)) for r in self.archive.about(i, 1)]
            if recs:
                text += "\nRECORDS OF THE LAST LISTINGS ITS CHAIN REACHED:\n" + "\n".join(recs[:2])
            return text
        except Exception:
            return ""

    def ceil_learn(self, call_args: dict[str, Any], rows: list, total: int, obs: Any) -> None:
        """Learns price-bound search from a reply for cheapest-item ceiling walk.
        ceil_seen: result total per non-paging args. Numbered pages: full page, empty
        next/more keys, total says more rows -> page_cap = rows reached. If exactly one
        numeric bound arg (no paging/size/count/lower-bound), one arg = all rows'
        single shelf, all rows priced: ceil_mode = signature, other-args template minus
        values all rows share, fixed mode, shelf/bound keys, free-text word count, if
        bound is integer. A row priced over bound instead flags signature in ceil_bad.
        """
        found = self.page_arg()
        pkey = found[0] if found else None
        base = {k: v for k, v in call_args.items() if k != pkey}
        self.__dict__.setdefault("ceil_seen", {})[json.dumps(base, sort_keys=True, default=str)] = int(total)
        dicts = [r for r in rows if isinstance(r, dict)]
        if found and found[1] is True and dicts and isinstance(obs, dict):
            at = call_args.get(pkey, found[2])
            nexts = [v for key, v in obs.items() if re.search(r"next|more|onward|continu|cursor", str(key).lower()) and not isinstance(v, (list, dict))]
            size = max(len(dicts), self.page_size())
            if is_number(at) and nexts and all(v is None or v is False or v == "" for v in nexts) and len(dicts) >= size:
                reach = int((at - found[2] + 1) * size)
                if total > reach:
                    self.page_cap = min(self.__dict__.get("page_cap") or reach, reach)
        texts, mode, first, paging = self.search_keys(call_args)
        bounds = [k for k, v in call_args.items() if is_number(v) and not isinstance(v, bool) and k != pkey and k not in paging and v > 1
                  and not re.search(r"page|size|count|result|min|low|from|start|offset", str(k).lower())]
        if len(bounds) != 1 or not dicts:
            return
        bkey = bounds[0]
        cap = float(call_args[bkey])
        labels = {self.shelf_of(r) for r in dicts}
        shelf_keys = [k for k, v in call_args.items() if isinstance(v, str) and k != pkey and v.strip() and v.strip().lower() in labels]
        skey = shelf_keys[0] if len(shelf_keys) == 1 else None
        fixed = {k: v for k, v in mode if k != skey}
        sig = json.dumps([sorted(fixed.items()), bkey], default=str)
        prices = [self.row_cost(r) for r in dicts]
        bad = self.__dict__.setdefault("ceil_bad", set())
        if any(p is not None and p > cap + 1e-9 for p in prices):
            bad.add(sig)
            return
        if sig in bad or skey is None or len(labels) != 1 or "" in labels or any(p is None for p in prices):
            return

        def every(val: str) -> bool:
            """Tells whether every row of the reply holds the given string, ignoring case, in some field whose key is not
            category-like.
            """
            low = val.strip().lower()
            return all(any(isinstance(x, str) and x.strip().lower() == low for kk, x in r.items()
                           if not re.search(r"categor|shelf|department|aisle", str(kk).lower())) for r in dicts)
        tmpl = {k: v for k, v in base.items() if k not in (bkey, skey) and not (isinstance(v, str) and v.strip() and k not in fixed and every(v))}
        words = sum(len(v.split()) for k, v in tmpl.items() if isinstance(v, str) and k not in fixed)
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties") or {}
        old = self.__dict__.get("ceil_mode")
        if old and old["sig"] not in bad and old["words"] < words:
            return
        self.ceil_mode = {"sig": sig, "tmpl": tmpl, "mode": fixed, "shelf": skey, "bound": bkey, "words": words,
                          "int": ((props.get(bkey) or {}) if isinstance(props, dict) else {}).get("type") == "integer"}

    @staticmethod
    def row_cost(row: dict[str, Any]) -> float | None:
        """Returns the first numeric value under a price or cost key in a row, as a float, or None."""
        cost = next((v for key, v in row.items() if is_number(v) and not isinstance(v, bool) and re.search(r"price|cost", str(key).lower())), None)
        return float(cost) if cost is not None else None

    def ceil_spec(self) -> dict[str, Any] | None:
        """Returns the learned price-bound search mode (ceil_mode) unless its signature was marked bad, else None."""
        spec = self.__dict__.get("ceil_mode")
        return spec if spec and spec["sig"] not in self.__dict__.get("ceil_bad", set()) else None

    def cg_listing(self, args: dict[str, Any]) -> bool:
        """Tells whether search arguments form a price-bound search under the learned mode: every fixed mode argument
        matches, or, with no fixed arguments, the bound key is present. False without a usable mode.
        """
        spec = self.ceil_spec()
        if not spec or not isinstance(args, dict):
            return False
        if spec["mode"]:
            return all(args.get(k) == v for k, v in spec["mode"].items())
        return spec["bound"] in args

    def page_size(self) -> int:
        """Largest result-page size seen across the recorded search walks, or 0 when there are none."""
        return max([w.get("size", 0) for w in self.__dict__.get("page_book", {}).values()] or [0])

    def visible_depth(self) -> int | None:
        """Number of results the store's paging can reach, from the paging argument's first and last allowed values and
        the page size (page count times size for page numbers, offset span plus one page for offsets). Falls back to the
        learned page_cap when paging uses tokens or has no known last value.
        """
        found = self.page_arg()
        size = self.page_size()
        if found and size > 0 and found[1] != "token" and found[3] is not None:
            return int((found[3] - found[2] + 1) * size) if found[1] else int(found[3] - found[2] + size)
        return self.__dict__.get("page_cap")

    def raw_label(self, labels: set[str]) -> str:
        """Returns the shelf label as the store writes it for a set of lowercased labels. Over all archived records it
        counts dicts that hold a known listing id under an id key and a short category, shelf, department or aisle value
        whose lowercase form is in labels, and returns the most frequent spelling. Empty string when nothing matches.
        """
        if not labels:
            return ""
        tally: dict[str, int] = {}

        def visit(d: dict[str, Any]) -> None:
            """Counts the raw category-like value of a dict that carries a known listing id under an id key, when its
            lowercase form is one of the wanted labels.
            """
            if not any(isinstance(v, str) and v in self.archive.ids and "id" in str(key).lower() for key, v in d.items()):
                return
            raw = next((val.strip() for key, val in d.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                        and re.search(r"categor|shelf|department|aisle", str(key).lower()) and "id" not in str(key).lower()), "")
            if raw and raw.lower() in labels:
                tally[raw] = tally.get(raw, 0) + 1
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return max(sorted(tally), key=lambda s: tally[s], default="")

    def below(self, price: Any) -> float | int | None:
        """Returns the bound just under a price for the next ceiling search: one less for a whole-number price, the
        integer part of a fractional price when the bound is an integer, else 0.01 less. None for a non-number or when
        the result is zero or below.
        """
        if not is_number(price):
            return None
        spec = self.ceil_spec() or {}
        val = float(price)
        if spec.get("int") or val.is_integer():
            out: float | int = int(val) - 1 if val.is_integer() else int(val)
        else:
            out = round(val - 0.01, 2)
        return out if out > 0 else None

    def ceil_mid(self, a: float, b: float) -> float | int:
        """Midpoint of two bounds for the ceiling search: floor division when the bound is an integer or both values are
        whole, otherwise rounded to two decimals.
        """
        spec = self.ceil_spec() or {}
        if spec.get("int") or (float(a).is_integer() and float(b).is_integer()):
            return (int(a) + int(b)) // 2
        return round((float(a) + float(b)) / 2, 2)

    def ceil_args(self, spec: dict[str, Any], label: str, bound: float | int) -> dict[str, Any]:
        """Builds ceiling-search arguments from the learned template, with the shelf key set to label and the bound key
        to bound (as an int when whole).
        """
        whole = spec.get("int") or float(bound).is_integer()
        return {**spec["tmpl"], spec["shelf"]: label, spec["bound"]: int(bound) if whole else bound}

    def cg_products(self) -> list[int | None]:
        """Returns the request's product indexes, or [None] when it has no product rows."""
        return list(range(len(self.needs.items()))) or [None]

    def ceiling_ready(self, k: int | None) -> bool:
        """Tells whether the price-ceiling walk can run for product k: the request wants the cheapest item, k was not
        walked yet, a usable bound mode and a paging argument exist, and the store's raw label for k's home shelves is
        known. False for k None when the request has product rows.
        """
        if self.needs.items() and k is None:
            return False
        return (self.needs.priority() == "cheapest" and k not in self.__dict__.get("ceil_walked", set()) and self.ceil_spec() is not None
                and self.page_arg() is not None and bool(self.raw_label(self.walk_home(k))))

    def ceil_aim(self, k: int | None) -> float | None:
        """Returns the price to beat for product k: the lower of the walk's current pick cost and any stored aim in
        ceil_aims. With neither, and a numeric bound requirement on k, it takes the cheapest known price among unread
        home-shelf rows and read home-shelf listings that are not dead, refused or out of stock. None when no price is
        known.
        """
        pick = self.walk_pick(k)
        aim = (self.__dict__.get("ceil_aims") or {}).get(k)
        costs = ([pick[1]] if pick else []) + ([float(aim)] if is_number(aim) else [])
        if not costs and self.bound_need(k):
            home = self.walk_home(k)
            seen = [c for c, i in self.cheap_unread(self.home_rows(home), k)] + [
                float(self.archive.price(i)) for i in self.archive.read if self.shelf_known(i) in home and is_number(self.archive.price(i))
                and i not in self.dead_lines() and i not in self.refused_lines and self.last_stock(i) is not False]
            costs = [min(seen)] if seen else []
        return min(costs) if costs else None

    def bound_need(self, k: int | None) -> bool:
        """Tells whether product k has an open requirement whose value holds a digit and a bound word (CH_BOUND).
        False with no product rows or no k.
        """
        if not self.needs.items() or k is None:
            return False
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            if self.product_of(head) == k and re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
                return True
        return False

    def home_rows(self, home: set[str]) -> set[str]:
        """Collects the ids of archived rows that sit on one of the given home shelves. Walks every record in the archive
        and keeps a dict's row id when it is a known listing id and its shelf is in home; ceil_aim uses the result to
        find unread rows of a product's category.
        """
        out: set[str] = set()

        def visit(d: dict[str, Any]) -> None:
            """Adds a dict's row id when it is a known listing id and its shelf is a home shelf."""
            ident = self.row_mark(d)
            if ident in self.archive.ids and self.shelf_of(d) in home:
                out.add(ident)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return out

    def ceil_clash(self, home: set[str], bound: float) -> bool:
        """Tells whether any archived row on a home shelf, holding a known listing id under an id key, is priced at or
        under the bound and shown in stock. An empty price-filter reply despite such a row means the filter does not hold,
        so callers mark its mode bad. Scanning stops at the first hit.
        """
        hit = False

        def visit(d: dict[str, Any]) -> None:
            """Sets the hit flag for a home-shelf dict with a known listing id whose cost is within the bound and whose stock
            state reads in stock; does nothing once a hit is found.
            """
            nonlocal hit
            if hit or self.shelf_of(d) not in home or not any(isinstance(v, str) and v in self.archive.ids and "id" in str(key).lower() for key, v in d.items()):
                return
            cost = self.row_cost(d)
            hit = cost is not None and cost <= bound and self.stock_state(d) is True
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return hit

    def cheap_unread(self, ids: Any, k: int | None, below: float | None = None) -> list[tuple[float, str]]:
        """Lists unread candidates among the given ids as (price, id) pairs, cheapest first. Skips ids already read or
        refused, shown sold out in an archived row, or lacking a numeric price. Also drops ids whose price times the
        wanted quantity (product k's, else the request's) exceeds the spend cap, and ids at or above the optional below
        price.
        """
        ids = set(ids)
        if not ids:
            return []
        qty = self.needs.items()[k]["quantity"] if k is not None and k < len(self.needs.items()) else self.needs.quantity()
        cap = self.needs.spend_cap
        sold: dict[str, bool] = {}

        def visit(d: dict[str, Any]) -> None:
            """Records whether a dict holding one of the ids under an id key shows the item as sold out, when its stock
            state is known.
            """
            if any(isinstance(v, str) and v in ids and "id" in str(key).lower() for key, v in d.items()):
                state = self.stock_state(d)
                if state is not None:
                    sold[self.row_mark(d)] = not state
        for item in self.archive.items:
            walk_dicts(item, visit)
        out = []
        for ident in ids:
            cost = self.archive.price(ident)
            if ident in self.archive.read or ident in self.refused_lines or sold.get(ident) or not is_number(cost):
                continue
            if (is_number(cap) and is_number(qty) and cost * qty > cap) or (below is not None and cost >= below):
                continue
            out.append((float(cost), ident))
        return sorted(out)

    def all_words(self) -> bool:
        """True when the store's search needs every query word to match: word_kind is 'all', or lead words were learned
        and word_kind is not 'any'.
        """
        return self.word_kind == "all" or (bool(self.lead_words) and self.word_kind != "any")

    def valued_ids(self, k: int | None) -> set[str]:
        """Collects home-shelf ids for product k from search walks that came back in id order, are not price-bound and
        whose arguments carry k's wanted values (walk_carries). These rows lead the ranking after a ceiling walk. Empty
        unless the store's search matches all words.
        """
        if not self.all_words():
            return set()
        home = self.walk_home(k)
        out: set[str] = set()
        for walk in self.__dict__.get("page_book", {}).values():
            if walk.get("id_order") is True and not self.cg_listing(walk["args"]) and self.walk_carries(walk["args"], k):
                out |= {i for i, s in (walk.get("row_shelf") or {}).items() if s in home}
        return out

    def valued_complete(self, k: int | None) -> bool:
        """Tells whether a finished value search already covers product k. Looks for a walk in id order, not price-bound,
        with first results, no next page and a shelf recorded for every row of its total, that reached k's home shelves
        and carries one of k's open requirements whose value is a single option (no list or alternatives). Applies only
        when search matches all words; False on error.
        """
        try:
            if not self.all_words():
                return False
            home = self.walk_home(k)
            several = bool(self.needs.items()) and k is not None
            reqs = self.needs.props_left()
            lone = set()
            for n, req in enumerate(reqs):
                if several and self.product_of(req.partition(":")[0]) not in (k, None):
                    continue
                value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()
                if len([p for p in re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", value) if p.strip()]) == 1:
                    lone.add(n)
            for walk in self.__dict__.get("page_book", {}).values():
                if walk.get("id_order") is not True or self.cg_listing(walk["args"]) or not walk.get("first") \
                        or self.next_page(walk) is not None or len(walk.get("row_shelf") or {}) < walk["total"]:
                    continue
                if any(s in home for s in (walk.get("row_shelf") or {}).values()) and self.walk_carries(walk["args"], k) & lone:
                    return True
        except Exception:
            return False
        return False

    def row_carries(self, ident: str, k: int | None) -> bool:
        """Tells whether an archived listing's text carries one of product k's open requirement values. Reference
        values are skipped; the rest are split at commas, slashes, 'or', 'either' and 'nor' into amounts (a figure with a
        unit) and word groups (stems of three or more characters, unless the part is an amount). Gathers the string
        fields of every archived dict holding the id under an id key and returns True when a figure with its unit appears
        there, or all words of a group are among its stems. False when nothing is wanted.
        """
        several = bool(self.needs.items()) and k is not None
        wants = []
        figs: list[tuple[str, str]] = []
        for req in self.needs.props_left():
            if several and self.product_of(req.partition(":")[0]) not in (k, None):
                continue
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()
            if ref_text(value):
                continue
            for part in re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", value):
                fig = self.fig_amount(part)
                if fig:
                    figs.append(fig)
                    continue
                words = [self.stem(w) for w in re.findall(r"[a-z0-9]{3,}", part)] if not self.amount_part(part) else []
                if words:
                    wants.append(words)
        if not wants and not figs:
            return False
        text: list[str] = []

        def visit(d: dict[str, Any]) -> None:
            """Adds the lowercased string fields of a dict that holds the id under an id key to the gathered text."""
            if any(isinstance(v, str) and v == ident and "id" in str(key).lower() for key, v in d.items()):
                text.extend(str(v).lower() for v in d.values() if isinstance(v, str))
        for item in self.archive.items:
            walk_dicts(item, visit)
        have = {self.stem(w) for w in re.findall(r"[a-z0-9]{3,}", " ".join(text))}
        body = " ".join(text)
        if any(re.search(r"(?<![\d.,])" + re.escape(fig) + r"\s*-?\s*" + re.escape(self.stem(unit)), body) for fig, unit in figs):
            return True
        return any(all(w in have for w in words) for words in wants)

    def vl_room(self) -> tuple[int, int]:
        """Returns the messages still allowed to the shopper and the moves left after the order reserve and two moves per
        remaining message. Messages count as zero when asking is off or there is no message tool.
        """
        left = max(0, self.ask_limit() - self.msgs_sent) if self.asking and self.role("message") else 0
        return left, self.moves_left() - self.order_due() - 2 * left

    def vl_values(self, k: int | None) -> list[list[list[str]]]:
        """Collects search words for product k's code-checked requirement values: one list of alternatives per
        requirement, each alternative a word list. Skips unsearchable and reference values; a value with 'or', 'either'
        or a slash splits into up to three alternatives, otherwise only its part before a comma, 'and', '&' or '+' is
        used. Parts with exclusion words, or amounts without a figure, are dropped; link words, possessives, decimals and
        words of the requirement name are removed, though a figure's unit stays. Repeated entries are skipped.
        """
        out: list[list[list[str]]] = []
        for head, value in self.code_values(k):
            if self.unsearchable(head, value):
                continue
            low = re.sub(r"\([^)]*\)", " ", str(value).lower().replace("\u2019", "'")).strip()
            if not low or ref_text(low):
                continue
            if re.search(r"\bor\b|\beither\b|/", low):
                pieces = [p for p in re.split(r"[,;/]|\bor\b|\beither\b", low) if p.strip()][:3]
            else:
                pieces = [re.split(r"[,;&+]|\band\b", low)[0]]
            name = set(re.findall(r"[a-z0-9]+", str(head).lower()))
            alts: list[list[str]] = []
            for piece in pieces:
                fig = self.fig_amount(piece)
                if re.search(EXCLUDE_WORDS, piece) or (self.amount_part(piece) and not fig):
                    continue
                piece = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", re.sub(r"'s\b", "", piece))
                words = [w for w in re.findall(r"\d+(?:[.,]\d+)+|[a-z0-9]+", piece)
                         if (len(w) >= 2 or (fig and w == fig[0])) and not re.search(r"[.,]", w) and w not in LINK_WORDS
                         and (w not in name or (fig and w == fig[1]))]
                if words and words not in alts:
                    alts.append(words)
            if alts and alts not in out:
                out.append(alts)
        return out

    def vl_words(self, k: int | None, label: str) -> list[str]:
        """Returns the category label words to add to value searches. Unless the store is known not to match label words
        (label_indexed is False), every label word of two or more characters is used. Otherwise it keeps label words
        found in at least half of the home-shelf titles in the archive, falling back to the last word (over two letters)
        of product k's name.
        """
        words = list(dict.fromkeys(w for w in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", label.lower().replace("’", "'")))
                                   if len(w) >= 2 and w not in LINK_WORDS))
        if self.__dict__.get("label_indexed") is not False:
            return words
        home = self.walk_home(k)
        titles: list[set[str]] = []

        def visit(d: dict[str, Any]) -> None:
            """Adds the stemmed title words of a home-shelf dict with a known listing id to the gathered title sets."""
            if self.shelf_of(d) in home and self.row_mark(d) in self.archive.ids:
                titles.append({self.stem(w) for w in re.findall(r"[a-z0-9]+", self.title_of(d).lower())})
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        kept = [w for w in words if titles and 2 * sum(1 for t in titles if self.stem(w) in t) >= len(titles)]
        if kept:
            return kept
        rows = self.needs.items()
        product = str(rows[k]["product"] if rows and k is not None and k < len(rows) else self.needs.reading.get("product") or "")
        return [w for w in re.findall(r"[a-z0-9]+", product.lower()) if len(w) > 2 and w not in LINK_WORDS][-1:]

    def vl_queries(self, k: int | None, label: str, cap_len: Any) -> tuple[list[str], list[str]]:
        """Builds value-search queries for product k, returned as (combined, single) lists, from its requirement values and
        the label words; each query skips repeated stems and is cut to the lead-word count and the free-text maxLength.
        Cheapest requests try every alternative of a value, others only the first. With two or more values the combined
        queries vary the first value that has alternatives and keep the rest at their first one, and the single queries
        use the first value alone; with one value its queries come back first and the second list is empty.
        """
        vals = self.vl_values(k)
        if not vals:
            return [], []
        label_words = self.vl_words(k, label)
        many = self.needs.priority() == "cheapest"

        def build(parts: list[list[str]]) -> str:
            """Joins the word parts and then the label words into one query, dropping any word whose stem already appeared.
            Keeps only the first lead_words words when that count is learned, and when cap_len is a number trims whole words
            from the end, then characters, until the text fits.
            """
            seen: set[str] = set()
            out: list[str] = []
            for w in [w for p in parts for w in p] + label_words:
                if self.stem(w) not in seen:
                    seen.add(self.stem(w))
                    out.append(w)
            text = " ".join(out[: self.lead_words] if self.lead_words else out)
            if is_number(cap_len):
                while len(text) > cap_len and " " in text:
                    text = text.rsplit(" ", 1)[0]
                text = text[: int(cap_len)]
            return text

        def alts(v: list[list[str]]) -> list[list[str]]:
            """Returns every alternative of a value when the cheapest item is wanted, else only the first."""
            return v if many else v[:1]
        lone = [build([a]) for a in alts(vals[0])]
        if len(vals) < 2:
            return list(dict.fromkeys(f for f in lone if f)), []
        j = next((n for n, v in enumerate(vals) if len(alts(v)) > 1), 0)
        both = [build([a if n == j else v[0] for n, v in enumerate(vals)]) for a in alts(vals[j])]
        return list(dict.fromkeys(f for f in both if f)), list(dict.fromkeys(f for f in lone if f))

    def vl_next(self, first: list[str], second: list[str], tkey: str, mode: dict[str, Any]) -> str | None:
        """Picks the next value-search query, going through the combined list and then the single list. Returns the
        first query not yet searched, or an empty string once every query of a list was searched and one of them
        matched. Returns None when every query was searched without matches, and '' when both lists are empty.
        """
        for forms in (first, second):
            if not forms:
                continue
            totals = [self.vl_seen(t, tkey, mode) for t in forms]
            if None in totals:
                return forms[totals.index(None)]
            if any(totals):
                return ""
        return None if first or second else ""

    def vl_mode(self, args: dict[str, Any]) -> tuple:
        """Returns the sorted (key, value) pairs of string arguments whose search schema property is an enum of four or
        fewer options, used as the fixed search mode.
        """
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        props = props if isinstance(props, dict) else {}
        return tuple(sorted((key, val) for key, val in args.items() if isinstance(val, str) and isinstance(props.get(key), dict)
                            and isinstance(props[key].get("enum"), list) and len(props[key]["enum"]) <= 4))

    def vl_set(self, text: str) -> frozenset:
        """Stemmed word set of a query, limited to the lead-word count when that is learned, for comparing queries."""
        return frozenset(self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[: self.lead_words or None])

    def vl_seen(self, text: str, tkey: str, mode: dict[str, Any]) -> int | None:
        """Returns the result total recorded in ceil_seen for an earlier search whose free-text key holds the same stemmed
        word set as text and whose mode arguments all match. Lets value searches skip queries the store already answered.
        None when no such search was recorded.
        """
        mine = self.vl_set(text)
        for key, total in (self.__dict__.get("ceil_seen") or {}).items():
            try:
                args = json.loads(key)
            except Exception:
                continue
            if isinstance(args, dict) and isinstance(args.get(tkey), str) and all(args.get(m) == v for m, v in mode.items()) \
                    and self.vl_set(args[tkey]) == mine:
                return int(total)
        return None

    def vl_indexed(self, label: str) -> None:
        """Decides once, from the latest archive record, whether the store's search matches category label words that
        the rows themselves do not carry. Counts rows with a known id on the label's shelf whose non-category string
        fields miss some label word; two or more such rows set label_indexed to True. Does nothing when it is already
        decided or the archive is empty.
        """
        if self.__dict__.get("label_indexed") is not None or not self.archive.items:
            return
        words = {self.stem(w) for w in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", label.lower())) if len(w) >= 2 and w not in LINK_WORDS}
        hits = 0

        def visit(d: dict[str, Any]) -> None:
            """Counts a dict on the label's shelf with a known listing id when its non-category string fields miss at least
            one label word.
            """
            nonlocal hits
            if self.row_mark(d) not in self.archive.ids or self.shelf_of(d) != label.strip().lower():
                return
            other = {self.stem(w) for key, v in d.items() if isinstance(v, str) and not re.search(r"categor|shelf|department|aisle", str(key).lower())
                     for w in re.findall(r"[a-z0-9]+", v.lower())}
            hits += bool(words - other)
        walk_dicts(self.archive.items[-1], visit, ceiling=3)
        if hits >= 2:
            self.label_indexed = True

    def vl_queue(self, k: int | None, ids: Any, floor: int, most: int) -> list[str]:
        """Picks the ids to open after a value search: unread candidates cheapest first (below the walk's current pick
        when the cheapest is wanted), up to the batch size and the room left, which is CEIL_OPENS per product for
        cheapest requests and most otherwise. Trims the queue to fit the moves above floor and, for cheapest requests,
        counts the opens per product in valued_opens.
        """
        cheapest = self.needs.priority() == "cheapest"
        pick = self.walk_pick(k) if cheapest else None
        batch = self.batch_tool()
        opened = self.__dict__.setdefault("valued_opens", {})
        room = CEIL_OPENS - opened.get(k, 0) if cheapest else most
        queue = [i for _, i in self.cheap_unread(ids, k, pick[1] if pick else None)][:max(0, min(batch[2] if batch else most, room))]
        while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= self.moves_left() - floor:
            queue.pop()
        if cheapest:
            opened[k] = opened.get(k, 0) + len(queue)
        return queue

    def vl_launch(self, queue: list[str], homes: set[str], floor: int, say: str) -> None:
        """Starts reading queued listings as a walk: clears rank checks for listings on the given home shelves, resets
        order_vetted, sets the open queue, recheck flag, message and floor, and calls open_named.
        """
        self.rank_checked = {mark for mark in self.rank_checked if self.shelf_known(mark.split("|")[0]) not in homes}
        self.order_vetted = False
        self.open_queue = queue
        self.open_recheck = True
        self.open_say = say
        self.open_purpose, self.open_floor = "walk", floor
        self.open_named()

    def vl_search(self) -> bool:
        """Runs one value search: the product's requirement values joined with the category's raw label, so the
        reply lists that category's listings with those values. Needs more than four moves. For each product
        under LABEL_TRIES, with unsettled values and a known label, a past free-text search (not price-bound)
        is the template (free-text key, enum modes and required arguments kept). It builds queries (vl_queries)
        and picks one not yet run (vl_next); when all ran empty but a walk with the values exists, it retries
        once without label_indexed. Arguments pass ArgCheck and walk_call; the result total is noted on the reply.
        A search showing every home-shelf match records the walk for cheapest requests and waits for vl_flush
        or queues the cheapest unread rows; other requests queue up to LABEL_OPENS rows.
        Logs label_search, launches any queue, returns True after one search;
        values that cannot be searched are marked settled.
        """
        _, room = self.vl_room()
        if room <= 4:
            return False
        tool = self.role("search")
        jschema = self.shapes.get(tool or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        found = self.page_arg()
        tries = self.__dict__.setdefault("label_tries", {})
        settled = self.__dict__.setdefault("label_done", {})
        sets = self.__dict__.setdefault("label_sets", {})
        products = self.cg_products()
        for k in products:
            if tries.get(k, 0) >= LABEL_TRIES:
                continue
            sig = json.dumps(self.vl_values(k))
            if sig in settled.setdefault(str(k), []):
                continue
            home = self.walk_home(k)
            label = self.raw_label(home)
            if not label:
                continue
            tmpl, tkey = None, ""
            for call in self.code_queries(k):
                keys = [key for key, val in call.items() if isinstance(val, str) and val.strip() and val in self.free_text(call)]
                if len(keys) == 1 and not self.cg_listing(call):
                    tmpl, tkey = call, keys[0]
                    break
            if not tmpl:
                settled[str(k)].append(sig)
                continue
            mode = {key: val for key, val in tmpl.items() if isinstance(props.get(key), dict) and props[key].get("enum")}
            tmpl = {key: val for key, val in tmpl.items() if key == tkey or key in mode or key in (jschema.get("required") or [])}
            cap_len = props[tkey].get("maxLength") if isinstance(props.get(tkey), dict) else None
            first, second = self.vl_queries(k, label, cap_len)
            sets.setdefault(str(k), set()).update(self.vl_set(q) for q in first + second)
            todo = self.vl_next(first, second, tkey, mode)
            if todo is None:
                todo = ""
                carried = any(self.walk_carries(w["args"], k) and any(s in home for s in (w.get("row_shelf") or {}).values())
                              for w in self.__dict__.get("page_book", {}).values())
                if carried and self.__dict__.get("label_indexed") is None:
                    self.label_indexed = False
                    first, second = self.vl_queries(k, label, cap_len)
                    sets[str(k)].update(self.vl_set(q) for q in first + second)
                    todo = self.vl_next(first, second, tkey, mode) or ""
            if not todo:
                settled[str(k)].append(sig)
                continue
            args, faults = ArgCheck.execute({**tmpl, tkey: todo}, jschema)
            if faults or not isinstance(args, dict):
                settled[str(k)].append(sig)
                continue
            tries[k] = tries.get(k, 0) + 1
            vals = self.vl_values(k)
            self.walk_call(args, f"I search the required value{'s' if len(vals) > 1 else ''} with the category's name as this store's "
                                 f"rows write it ('{todo}'): a keyword search matches listing values, titles and categories, so this "
                                 "lists the category's listings that carry the value.")
            total = self.vl_seen(todo, tkey, mode)
            self.vl_indexed(label)
            key = json.dumps({a: v for a, v in args.items() if not found or a != found[0]}, sort_keys=True, default=str)
            walk = self.__dict__.get("page_book", {}).get(key)
            every = bool(walk) and len(walk.get("row_shelf") or {}) >= walk["total"] and any(s in home for s in (walk.get("row_shelf") or {}).values())
            floor = self.order_due() + 2
            queue: list[str] = []
            note = "" if total is None else (f"[value search] '{todo}': " + ("no listing matched." if not total else f"{total} match(es)"
                                                + (" - every one is shown here." if every else ".")))
            if every and self.needs.priority() == "cheapest" and found and walk.get("id_order") is not False:
                self.__dict__.setdefault("valued", set()).add(key)
                self.__dict__.setdefault("valued_keys", {}).setdefault(str(k), []).append(key)
                if len(products) > 1:
                    self.__dict__.setdefault("valued_wait", {})[k] = True
                else:
                    queue = self.vl_queue(k, [i for i, s in (walk.get("row_shelf") or {}).items() if s in home], floor, CEIL_OPENS)
            elif every and self.needs.priority() != "cheapest" and todo in (first or second):
                queue = self.vl_queue(k, [i for i, s in (walk.get("row_shelf") or {}).items() if s in home], floor, LABEL_OPENS)
            if queue:
                note += (" Its unread in-stock rows within the budget on the requested category" + (" below the pick" if self.walk_pick(k) and
                         self.needs.priority() == "cheapest" else "") + ", cheapest first, are opened next: "
                         + ", ".join(f"{i} ({self.archive.price(i):g})" for i in queue) + ".")
            if note and self.chat and self.chat[-1].get("role") == "tool":
                self.chat[-1]["content"] = str(self.chat[-1]["content"]) + "\n" + note + ("" if queue else "\n" + self.status_line())
            self.log.append({"kind": "label_search", "product": k, "query": todo, "total": total, "every": every, "opened": queue})
            if queue:
                self.vl_launch(queue, home, floor, "I open the cheapest unread rows of that search: every listing of the category "
                                                   "carrying the required values is among its matches.")
            return True
        return False

    def vl_flush(self, floor: int) -> bool:
        """Opens the cheapest unread rows of the value searches held back (valued_wait), then clears it.
        Each product takes home-shelf ids from its value walks, below its current pick,
        two per product when moves allow, within CEIL_OPENS.
        Pools are interleaved, trimmed to the moves left and launched as a walk.
        A note tells the model to pick the lowest-priced fit per product.
        Logs value_flush; True when anything was opened.
        """
        wait = self.__dict__.setdefault("valued_wait", {})
        ks = [k for k, on in wait.items() if on]
        wait.clear()
        if not ks:
            return False
        book = self.__dict__.get("page_book", {})
        keys = self.__dict__.get("valued_keys") or {}
        opened = self.__dict__.setdefault("valued_opens", {})
        batch = self.batch_tool()
        per = 2 if self.moves_left() - floor - 1 >= 2 * len(ks) else 1
        pools: list[list[str]] = []
        for k in ks:
            home = self.walk_home(k)
            ids = {i for key in keys.get(str(k), []) if key in book and self.walk_carries(book[key]["args"], k)
                   for i, s in (book[key].get("row_shelf") or {}).items() if s in home}
            pick = self.walk_pick(k)
            pools.append([i for _, i in self.cheap_unread(ids, k, pick[1] if pick else None)][:max(0, min(per, CEIL_OPENS - opened.get(k, 0)))])
        owner: dict[str, Any] = {}
        queue: list[str] = []
        most = batch[2] if batch else per * len(ks)
        while len(queue) < most and any(pools):
            for k, pool in zip(ks, pools):
                while pool and pool[0] in queue:
                    pool.pop(0)
                if pool and len(queue) < most:
                    owner[pool[0]] = k
                    queue.append(pool.pop(0))
        while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= self.moves_left() - floor:
            queue.pop()
        if not queue:
            return False
        for i in queue:
            opened[owner[i]] = opened.get(owner[i], 0) + 1
        homes = set().union(*(self.walk_home(k) for k in ks))
        self.vl_launch(queue, homes, floor, "The cheapest is asked for: I open the cheapest unread rows each product's value search showed.")
        summary = ("[value searches read] The cheapest unread in-stock rows within the budget that each product's value search showed, "
                   "below its pick, were opened: " + ", ".join(f"{i} ({self.archive.price(i):g})" for i in queue) + ". For each product, "
                   "choose the lowest-priced listing whose own fields meet every requirement now in force; a fit that costs more needs "
                   "every cheaper candidate opened and rejected first.")
        if self.chat and self.chat[-1].get("role") == "tool":
            self.chat[-1]["content"] = str(self.chat[-1]["content"]) + "\n" + summary
        self.log.append({"kind": "value_flush", "products": ks, "opened": queue})
        return True

    def product_steps(self) -> str:
        """Builds a '[product steps]' status line for cheapest requests with several products: per product, the value
        pages read (valued_by) and whether it has a pick, plus the first product without a pick as next. Empty string
        for single-product or non-cheapest requests, or on error.
        """
        try:
            rows = self.needs.items()
            if len(rows) < 2 or self.needs.priority() != "cheapest":
                return ""
            used = self.__dict__.get("valued_by") or {}
            parts, nxt = [], ""
            for k, row in enumerate(rows):
                pick = self.walk_pick(k)
                parts.append(f"{row['product']}: {used.get(k, 0)} value page(s) read, pick {'yes' if pick else 'no'}")
                nxt = nxt or ("" if pick else str(row["product"]))
            return "\n[product steps] " + "; ".join(parts) + (f"; next: {nxt}" if nxt else "") + "."
        except Exception:
            return ""

    def value_pages(self) -> bool:
        """Chooses the next value-search step; True when it acted. Holds back while an open reference has room;
        stops after an order try or a refusal, at the wire's end, when time is short or with no search tool.
        First a new value search (vl_search). Then, for cheapest requests in stores that page results with a
        search that is not any-word, it picks an unfinished id-ordered walk that carries a product's values and
        reached its home shelf, skipping walks past the paging reach and ones a complete walk already covers.
        Pages are capped per product, in total and by a fair share, above a reserve; long or multi-product reads
        wait while messages can still go.
        The option with the fewest pages read for its product, then the best walk, goes to read_value_search;
        with nothing left, held-back products open through vl_flush.
        Errors are logged as value_pages_error.
        """
        try:
            if any(r.get("status") == "open" for r in self.__dict__.get("refs") or []) and self.ref_room():
                return False
            if self.orders_tried or self.refusal_n or self.wire.done or self.wire.broke or self.wire.spare() < CLOSING + 60 \
                    or not self.role("search"):
                return False
            if self.vl_search():
                return True
            if self.needs.priority() != "cheapest" or self.word_kind == "any":
                return False
            found = self.page_arg()
            if not found:
                return False
            book = self.__dict__.get("page_book", {})
            done = self.__dict__.setdefault("valued", set())
            used = self.__dict__.setdefault("valued_by", {})
            opened = self.__dict__.setdefault("valued_opens", {})
            wait = self.__dict__.setdefault("valued_wait", {})
            sets = self.__dict__.get("label_sets") or {}
            depth = self.visible_depth()
            size = max(1, self.page_size())
            per = max(VALUE_PAGES, -(-depth // size) - 1) if depth else VALUE_PAGES
            products = self.cg_products()
            several = len(products) > 1
            cap = max(VALUE_TOTAL, VALUE_TOTAL_PER * len(products))
            opens = 1 if self.batch_tool() else (2 if several else CEIL_OPENS)
            floor = self.order_due() + 2
            known = self.all_words()
            left, room = self.vl_room()
            share = max(4, room // max(1, len([k for k in products if self.walk_home(k) and not used.get(k)])))
            options, pending = [], False
            for n, k in enumerate(products):
                home = self.walk_home(k)
                if not home:
                    continue
                mine = sets.get(str(k)) or set()
                several_k = bool(self.needs.items()) and k is not None
                values = {self.stem(w) for req in self.needs.props_left() if not several_k or self.product_of(req.partition(":")[0]) in (k, None)
                          for w in re.findall(r"[a-z0-9]{2,}", req.partition(":")[2].lower())}
                label = {self.stem(w) for s in home for w in re.findall(r"[a-z0-9]{2,}", s)}
                whole = [(self.vl_set(" ".join(self.free_text(w["args"]))), self.walk_carries(w["args"], k, alts=True), self.vl_mode(w["args"]))
                         for w in book.values() if w.get("id_order") is True and w.get("first") and self.next_page(w) is None
                         and not self.cg_listing(w["args"]) and len(w.get("row_shelf") or {}) >= w["total"]]
                cands = []
                for key, walk in book.items():
                    if key in done or walk.get("id_order") is not True or not walk.get("first") or self.next_page(walk) is None \
                            or self.cg_listing(walk["args"]) or not self.walk_carries(walk["args"], k) \
                            or not any(s in home for s in (walk.get("row_shelf") or {}).values()):
                        continue
                    words = self.vl_set(" ".join(self.free_text(walk["args"])))
                    composed = words in mine
                    if not known and (depth is None or walk["total"] > depth):
                        continue
                    if known and depth is not None and walk["total"] > depth and not composed:
                        continue
                    pairs = self.walk_carries(walk["args"], k, alts=True)
                    how = self.vl_mode(walk["args"])
                    if known and any(mode == how and (seen <= words or (seen <= values | label and pairs <= got)) for seen, got, mode in whole):
                        continue
                    reach = walk["total"] if depth is None else min(walk["total"], depth)
                    pages = -(-reach // max(1, walk["size"])) - len(walk["read"])
                    allowed = min(per - used.get(k, 0), cap - sum(used.values()))
                    if several:
                        allowed = min(allowed, share - used.get(k, 0) - opened.get(k, 0) - (0 if opened.get(k) else opens))
                    take = pages if pages <= allowed else (allowed if composed else 0)
                    if take <= 0 or self.moves_left() - take - opens <= floor:
                        continue
                    if (take > 2 or several) and left and room > take + opens + 4:
                        pending = True
                        continue
                    cands.append((take, -self.walk_score(walk["args"], k), key, words))
                if known:
                    cands = [c for c in cands if not any((o[3] < c[3] and not (c[3] - o[3]) <= values | label)
                                                         or (c[3] < o[3] and (o[3] - c[3]) <= values | label) for o in cands)]
                options += [(used.get(k, 0), take, score, n, key, k) for take, score, key, _ in cands]
            if options:
                _, take, _, _, key, k = min(options, key=lambda o: o[:5])
                done.add(key)
                return self.read_value_search(k, key, floor, take, several)
            if wait and not pending and not (left and room > opens * len(wait) + 4):
                return self.vl_flush(floor)
            return False
        except Exception as err:
            self.log.append({"kind": "value_pages_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def read_value_search(self, k: int | None, key: str, floor: int, most: int = VALUE_PAGES, defer: bool = False) -> bool:
        """Reads further pages of one value-search walk for product k (id order, not price).
        Reads pages while moves, time and wire allow; counts them in valued_by.
        Stops when a call fails or the page does not advance.
        Records the walk and tells the reply how many matches were seen.
        With defer it waits for vl_flush; else it opens the cheapest unread rows.
        Logs value_pages; True when a page was read.
        """
        found = self.page_arg()
        walk = self.__dict__.get("page_book", {}).get(key)
        if not found or not walk:
            return False
        query = " ".join(self.free_text(walk["args"]))
        used = self.__dict__.setdefault("valued_by", {})
        known = self.all_words()
        say = ("The cheapest is asked for, and this store's keyword search needs every word and lists results in a fixed id order, not "
               f"by price: I read the next page of '{query}' so that every listing carrying these words is seen." if known else
               "The cheapest is asked for, and this store lists search results in a fixed id order, not by price: I read the next page "
               f"of '{query}' so that every listing it matched is seen.")
        got, tried = 0, False
        while self.next_page(walk) is not None and got < most and self.moves_left() > floor + 1 and not self.wire.done \
                and not self.wire.broke and self.wire.spare() >= CLOSING + 30:
            tried = True
            used[k] = used.get(k, 0) + 1
            nxt = self.next_page(walk)
            if not self.walk_call({**walk["args"], found[0]: nxt}, say):
                break
            got += 1
            if self.next_page(walk) == nxt:
                break
        if not got:
            return tried
        self.__dict__.setdefault("valued_keys", {}).setdefault(str(k), []).append(key)
        seen = len(walk.get("row_shelf") or {})
        complete = self.next_page(walk) is None and seen >= walk["total"]
        home = self.walk_home(k)
        head = (f"[value search read] '{query}': {seen} of its {walk['total']} matches seen"
                + ((" - every listing carrying these words is among them" if known else " - every listing it matched is among them") if complete
                   else " - the rest cannot be reached, so a cheaper fit may lie past the reachable rows" if self.next_page(walk) is None
                   else "") + ". ")
        queue: list[str] = []
        if defer:
            self.__dict__.setdefault("valued_wait", {})[k] = True
            summary = head + ("Its cheapest unread in-stock rows within the budget are opened once every product's value search is read. "
                              "Choose the lowest-priced listing whose own fields meet every requirement now in force; a fit that costs "
                              "more needs every cheaper candidate opened and rejected first.")
        else:
            pick = self.walk_pick(k)
            queue = self.vl_queue(k, [i for i, s in (walk.get("row_shelf") or {}).items() if s in home], floor, CEIL_OPENS)
            summary = (head + ("Its cheapest unread in-stock rows within the budget on the requested category" + (" below the pick" if pick else "")
                               + " are opened next: " + ", ".join(f"{i} ({self.archive.price(i):g})" for i in queue) + ". " if queue else
                               "No unread in-stock row within the budget on the requested category is left among them. ")
                       + "Choose the lowest-priced listing whose own fields meet every requirement now in force; a fit that costs more "
                       "needs every cheaper candidate opened and rejected first.")
            if queue:
                self.vl_launch(queue, home, floor, "The cheapest is asked for: I open the cheapest unread rows of that search.")
        if self.chat and self.chat[-1].get("role") == "tool":
            self.chat[-1]["content"] = str(self.chat[-1]["content"]) + "\n" + summary + ("" if queue else "\n" + self.status_line())
        self.log.append({"kind": "value_pages", "product": k, "query": query, "pages": got, "complete": complete, "opened": queue,
                         "defer": defer})
        return True

    def ceiling_walk(self, k: int | None, floor: int, repair: bool = False) -> tuple[str, list[str]] | None:
        """Lists product k's category below the price to beat (ceil_aim) with the store's price filter, for
        cheapest requests; needs a usable bound mode, paging, the raw category label and an aim (ceil_walked,
        ceil_aimed). From just under the aim it sends up to CEIL_CALLS bounded searches (cached ones reused)
        while moves, time and wire allow. A row above the bound, or an empty reply while a cheaper in-stock home
        row is known, marks the mode bad. An empty reply moves the bound halfway up to the last non-empty one; a
        result the paging and calls can cover is read to the end; else the bound drops toward the cheapest row.
        The unread, in-stock, in-budget home ids under the aim are ranked: complete value searches first, then
        rows carrying the values, then the rest.
        Logs ceiling_walk and returns a report line with up to CEIL_OPENS ids,
        or None when it could not run.
        """
        spec, found = self.ceil_spec(), self.page_arg()
        home = self.walk_home(k)
        label = self.raw_label(home)
        aim = self.ceil_aim(k)
        if not spec or not found or not label or aim is None:
            return None
        self.__dict__.setdefault("ceil_walked", set()).add(k)
        pick_now = self.walk_pick(k)
        self.__dict__.setdefault("ceil_aimed", {})[k] = (float(aim), pick_now[0] if pick_now else "")
        bound = self.below(aim)
        if bound is None:
            return None
        floor = max(floor, self.order_due() + 2) if not self.orders_tried and not repair else floor + (2 if repair else 0)
        depth = self.visible_depth()
        book = self.__dict__.setdefault("page_book", {})
        keys: list[str] = []
        calls, top, complete, ran, last = 0, None, False, False, bound
        what = "the pick" if pick_now or is_number((self.__dict__.get("ceil_aims") or {}).get(k)) else "the cheapest row seen"
        say = f"The cheapest is asked for: I list '{label}' priced below {what} ({aim:g}) with the store's price filter."
        while calls < CEIL_CALLS and self.moves_left() > floor + 1 and not self.wire.done and not self.wire.broke \
                and self.wire.spare() >= CLOSING + 30:
            args = self.ceil_args(spec, label, bound)
            key = json.dumps(args, sort_keys=True, default=str)
            self.__dict__.setdefault("ceil_keys", set()).add(key)
            calls += 1
            reuse = key in book and bool(book[key].get("first")) and key in (self.__dict__.get("ceil_seen") or {})
            if not reuse and not self.walk_call(args, say):
                break
            ran, last = True, bound
            keys.append(key)
            walk = book.get(key)
            if walk and any(is_number(self.archive.price(i)) and self.archive.price(i) > bound for i in (walk.get("row_shelf") or {})):
                self.__dict__.setdefault("ceil_bad", set()).add(spec["sig"])
            if self.ceil_spec() is None:
                break
            total = self.__dict__.get("ceil_seen", {}).get(key)
            if total is None:
                break
            if not total or not walk:
                if self.ceil_clash(home, bound):
                    self.__dict__.setdefault("ceil_bad", set()).add(spec["sig"])
                    break
                if top is None:
                    complete = True
                    break
                nb = self.ceil_mid(bound, top)
                if not bound < nb < top:
                    break
                bound = nb
                continue
            pages = -(-total // max(1, walk["size"]))
            if (depth is None or total <= depth) and pages - len(walk["read"]) <= CEIL_CALLS - calls:
                while self.next_page(walk) is not None and calls < CEIL_CALLS and self.moves_left() > floor + 1 \
                        and not self.wire.done and not self.wire.broke and self.wire.spare() >= CLOSING + 30:
                    calls += 1
                    nxt = self.next_page(walk)
                    if not self.walk_call({**args, found[0]: nxt}, "I read the next page of that price-filter listing.") \
                            or self.next_page(walk) == nxt:
                        break
                complete = self.next_page(walk) is None
                break
            top = bound
            costs = [float(c) for c in (self.archive.price(i) for i in (walk.get("row_shelf") or {})) if is_number(c)]
            low = min(costs) if costs else 0.0
            fill = max(1, CEIL_CALLS - calls) * max(1, walk["size"])
            nb = min(self.ceil_mid(low, bound), self.ceil_mid(low, low + (bound - low) * 1.8 * fill / max(total, 1)))
            if not nb < bound:
                break
            bound = nb
        if not ran:
            return None
        ids: set[str] = set()
        for key in keys:
            ids |= {i for i, s in (book.get(key, {}).get("row_shelf") or {}).items() if s in home}
        cands = [i for _, i in self.cheap_unread(ids, k, aim)]
        valued = self.valued_ids(k)
        lead = [i for i in cands if i in valued]
        rest = [i for i in cands if i not in valued]
        texty = [i for i in rest if self.row_carries(i, k)]
        bare = [] if self.valued_complete(k) else [i for i in rest if i not in texty]
        ranked = lead + texty + bare
        if complete and not ids:
            report = f"the price filter on '{label}' below {aim:g}: no listing of it costs less than {what}"
        elif complete:
            report = (f"the price filter on '{label}': every listing priced up to {last:g} seen ({len(ids)}, {len(cands)} of them unread, "
                      "in stock and within the budget)")
        else:
            report = f"the price filter on '{label}' below {aim:g}: {len(ids)} of the cheaper listings seen ({len(cands)} unread, in stock and within the budget)"
        self.log.append({"kind": "ceiling_walk", "product": k, "label": label, "aim": aim, "calls": calls, "complete": complete,
                         "rows": len(ids), "ranked": ranked[:CEIL_OPENS]})
        return report, ranked[:CEIL_OPENS]

    def ceiling_sweep(self, tried: set) -> tuple[dict[str, Any] | None, bool]:
        """Plans the price-filter sweep for a single-product request: the next bounded search and whether the sweep is
        over. Uses the learned bound mode with the shelves' raw label and a bound just under the cheapest price in
        shelf_rows. Returns those arguments when that search is not in tried, the next page when one exists and was not
        tried, or (None, done) where done means nothing cheaper can exist: no bound under the cheapest price, an empty
        result with no cheaper in-stock row known, or every row of the search seen. (None, False) when it cannot run.
        """
        spec, found = self.ceil_spec(), self.page_arg()
        if not spec or not found or self.needs.items():
            return None, False
        rows = self.shelf_rows()
        shelves = self.shelf_labels()
        label = self.raw_label(shelves)
        if not rows or not label:
            return None, False
        bound = self.below(min(c for c, _, _ in rows.values()))
        if bound is None:
            return None, True
        args = self.ceil_args(spec, label, bound)
        key = json.dumps(args, sort_keys=True, default=str)
        if key not in tried:
            self.__dict__.setdefault("ceil_keys", set()).add(key)
            return args, False
        walk = self.__dict__.get("page_book", {}).get(key)
        if not walk:
            return None, self.__dict__.get("ceil_seen", {}).get(key) == 0 and not self.ceil_clash(set(shelves), bound)
        nxt = self.next_page(walk)
        if nxt is not None:
            more = {**args, found[0]: nxt}
            return (more, False) if json.dumps(more, sort_keys=True, default=str) not in tried else (None, False)
        return None, len(walk.get("row_shelf") or {}) >= walk["total"]

    def value_guard(self, call_name: str, call_args: dict[str, Any]) -> str:
        """Holds back a details call for cheapest requests when the store's search needs every word and the episode is
        not in its reserve. When the one listing named is unread, not in the cart, not opened by name and not among the
        rows that value searches showed, it lists up to six unread rows from those searches that cost less than the
        listing (and the pick), and returns a 'Not run' message asking to open them first. Speaks once per product and
        needs version (vguard_told); returns an empty string otherwise, when references exist, or on error.
        """
        try:
            if call_name != self.role("details") or self.needs.priority() != "cheapest" or not self.all_words() or self.in_reserve():
                return ""
            named = [v for v in call_args.values() if isinstance(v, str) and v in self.archive.ids]
            if len(named) != 1 or named[0] in self.archive.read or named[0] in self.trolley.units:
                return ""
            ident = named[0]
            if ident in self.named_open() or self.__dict__.get("refs") or self.needs.reading.get("references"):
                return ""
            k = self.product_index(ident)
            if self.needs.items() and k is None:
                return ""
            mark = (k, self.needs.version)
            told = self.__dict__.setdefault("vguard_told", set())
            if mark in told:
                return ""
            valued = self.valued_ids(k)
            if not valued or ident in valued:
                return ""
            pick = self.walk_pick(k)
            mine = self.archive.price(ident)
            costs = ([pick[1]] if pick else []) + ([float(mine)] if is_number(mine) else [])
            limit = min(costs) if costs else None
            rows = self.cheap_unread(valued, k, limit)[:6] if limit is not None else []
            if not rows:
                return ""
            told.add(mark)
            return ("Not run (no step used). The cheapest is asked for, and this store's keyword search needs every word, so the rows "
                    f"its searches led by a required value showed carry those words somewhere in their records; {ident} is not among "
                    "them, while these rows of theirs cost less" + (" than it and the pick" if pick else " than it") + " and are unread: "
                    + ", ".join(f"{i} ({c:g})" for c, i in rows) + ". Open them first, cheapest first; send this call again if you "
                    "still want it.")
        except Exception as err:
            self.log.append({"kind": "value_guard_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def title_line(self, k: int | None, rec: dict[str, Any], template: tuple[dict[str, Any], str]) -> dict[str, Any] | None:
        """Builds a search from a listing's title for product k: the first title word of three or more letters that is
        not a link word, home-shelf word, requirement word or product word (compared by stem), plus the next title word
        when it passes too. Returns the template's fixed arguments with that text as the free-text value, or None when
        no word fits or that query already ran.
        """
        avoid = set(LINK_WORDS) | {w for s in self.walk_home(k) for w in re.findall(r"[a-z]+", s)}
        avoid |= {w for req in self.needs.props_left() for w in re.findall(r"[a-z0-9]+", req.lower())}
        avoid |= {w for r in (self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}])
                  for w in re.findall(r"[a-z]+", str(r.get("product") or "").lower())}
        stems = {self.stem(w) for w in avoid}
        toks = re.findall(r"[a-z0-9]+", self.title_of(rec).lower())
        for n, tok in enumerate(toks):
            if re.fullmatch(r"[a-z]{3,}", tok) and tok not in avoid and self.stem(tok) not in stems:
                nxt = toks[n + 1] if n + 1 < len(toks) else ""
                text = tok + (f" {nxt}" if nxt and nxt not in avoid and self.stem(nxt) not in stems else "")
                return None if self.ran_query(text) else {**template[0], template[1]: text}
        return None

    def rearm_walk(self) -> bool | None:
        """Restarts the price-ceiling walk for a product whose aimed pick has since died or been refused, for cheapest
        requests with time left and a search tool. Needs a usable bound mode and paging, a current pick that costs more
        than the old aim and enough moves; clears the stored aim, logs ceil_rearm and runs the walk in repair mode.
        Returns None when nothing was rearmed.
        """
        if self.needs.priority() != "cheapest" or self.wire.done or self.wire.spare() < CLOSING + 60 or not self.role("search"):
            return None
        aimed = self.__dict__.get("ceil_aimed") or {}
        gone = self.dead_lines() | self.refused_lines
        opens = 1 if self.batch_tool() else 2
        for k, (aim, ident) in list(aimed.items()):
            if not ident or ident not in gone or self.ceil_spec() is None or self.page_arg() is None:
                continue
            pick = self.walk_pick(k)
            if not pick or pick[1] <= aim + 1e-9 or self.moves_left() < FIRST_ORDER_BY + 3 + opens:
                continue
            aimed.pop(k, None)
            (self.__dict__.get("ceil_aims") or {}).pop(k, None)
            self.log.append({"kind": "ceil_rearm", "product": k, "dead": ident, "aim": pick[1]})
            return self.run_walk([], FIRST_ORDER_BY, [k], repair=True)
        return None

    def walk_step(self) -> bool:
        """Runs one step of the code's own walk through listings; True when it acted.
        Serves pending decline reads, a queued walk, then a rearmed ceiling walk.
        For cheapest requests before any order, runs a price-ceiling walk for the
        best-placed product, or probes the price filter with ceil_seed.
        When the moves left no longer cover the plan, sets walk_early and cuts it.
        Errors are logged as walk_error.
        """
        try:
            if self.__dict__.get("decline_queue"):
                return self.run_decline_reads()
            queued = self.__dict__.pop("walk_queue", None)
            if queued:
                return self.run_walk(*queued)
            again = self.rearm_walk()
            if again is not None:
                return again
            if self.orders_tried or self.needs.priority() != "cheapest" or self.wire.spare() < CLOSING + 60 or self.wire.done:
                return False
            if self.unfound_product():
                return False
            opens = 1 if self.batch_tool() else 2
            spare = self.moves_left()
            ceil = [k for k in self.cg_products() if (self.walk_pick(k) or self.bound_need(k)) and self.ceiling_ready(k)
                    and self.ceil_aim(k) is not None]
            if ceil and spare > self.order_due() + 2 + CEIL_CALLS + opens:
                used = self.__dict__.get("valued_by") or {}
                ceil.sort(key=lambda k: (bool(self.walk_pick(k)), used.get(k, 0), -1 if k is None else k))
                return self.run_walk([], self.order_due() + 2, ceil[:1])
            if not ceil and spare > self.order_due() + 2 + CEIL_CALLS + opens and self.ceil_seed():
                return True
            plan = self.walk_plan(WALK_TOTAL)
            if not plan:
                return False
            pages = sum(n for _, _, n in plan) + sum(1 for k, _, _ in plan if self.walk_pick(k))
            if spare > self.order_due() + 2 + pages + opens:
                return False
            plan = self.walk_plan(min(WALK_TOTAL, spare - self.order_due() - 1 - opens))
            if not plan:
                return False
            self.walk_early = True
            return self.run_walk(plan, self.order_due())
        except Exception as err:
            self.log.append({"kind": "walk_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def seed_args(self, label: str, bound: Any) -> dict[str, Any] | None:
        """Builds a first price-filter search from the search tool's schema, before any bound mode is learned. Needs
        paging, a category-like key (an enum option equal to the label, or a plain string set to the label) and a
        numeric bound key named like a ceiling, max, price, budget or upper limit but not like a paging or count key.
        Every other enum key with two to four options must take an option the tool description names next to the
        category key, and every required key must be filled; the result is checked with ArgCheck. Returns the
        arguments, or None when any part is missing.
        """
        tool = self.role("search")
        jschema = self.shapes.get(tool or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        found = self.page_arg()
        if not props or not found or not is_number(bound):
            return None
        told = next((str(tdef["function"].get("description") or "") for tdef in self.toolset if tdef["function"]["name"] == tool), "")
        parts = [part.lower() for part in re.split(r"[.;]", told)]
        tok = self.token_key()
        shelf_key, shelf_val = None, None
        for key, spec in props.items():
            if not isinstance(spec, dict) or key in (tok, found[0]) or not re.search(r"categor|shelf|department|aisle", str(key).lower()):
                continue
            if isinstance(spec.get("enum"), list):
                hit = next((v for v in spec["enum"] if isinstance(v, str) and v.strip().lower() == label.strip().lower()), None)
                if hit is not None:
                    shelf_key, shelf_val = key, hit
                    break
            elif spec.get("type") == "string":
                shelf_key, shelf_val = key, label
                break
        bound_key = next((key for key, spec in props.items() if isinstance(spec, dict) and spec.get("type") in ("integer", "number")
                          and not spec.get("enum") and key != found[0] and re.search(r"ceil|max|price|cap$|budget|upper|most", str(key).lower())
                          and not re.search(r"page|first|offset|skip|count|size|num", str(key).lower())), None)
        if not shelf_key or not bound_key:
            return None
        args: dict[str, Any] = {shelf_key: shelf_val, bound_key: int(bound) if props[bound_key].get("type") == "integer" or float(bound).is_integer() else bound}
        for key, spec in props.items():
            if key == shelf_key or not isinstance(spec, dict) or not isinstance(spec.get("enum"), list) or not 2 <= len(spec["enum"]) <= 4:
                continue
            hit = next((v for v in spec["enum"] if any(re.search(r"\b" + re.escape(str(v).lower()) + r"\b", part)
                                                       and re.search(r"\b" + re.escape(shelf_key.lower()) + r"\b", part) for part in parts)), None)
            if hit is None:
                return None
            args[key] = hit
        for key in jschema.get("required") or []:
            if key not in args:
                return None
        fixed, faults = ArgCheck.execute(args, jschema)
        return fixed if not faults and isinstance(fixed, dict) else None

    def ceil_seed(self) -> bool:
        """Probes whether the store's price filter works when no bound mode is known yet, for cheapest requests. For the
        first product not yet seeded that has a pick and a known raw label, it builds arguments with seed_args just under
        the pick's price, skips them when their signature is marked bad or that search already ran, and otherwise sends
        the search so ceil_learn can learn the mode from the reply. Logs ceil_seed and, when no mode was learned, adds the
        status line to the reply. Returns True after a search, False otherwise or on error.
        """
        try:
            if self.ceil_spec() is not None or self.__dict__.get("ceil_mode") or self.needs.priority() != "cheapest":
                return False
            seeded = self.__dict__.setdefault("ceil_seeded", set())
            found = self.page_arg()
            for k in self.cg_products():
                if k in seeded or (self.needs.items() and k is None):
                    continue
                pick = self.walk_pick(k)
                label = self.raw_label(self.walk_home(k))
                if not pick or not label:
                    continue
                seeded.add(k)
                args = self.seed_args(label, self.below(pick[1]))
                if not args:
                    return False
                mode = dict(self.search_keys(args)[1])
                sig = json.dumps([sorted((m, v) for m, v in mode.items() if not (isinstance(args.get(m), str) and args[m] == label)),
                                  next((a for a in args if is_number(args[a]) and a != (found[0] if found else None)), "")], default=str)
                key = json.dumps({a: v for a, v in args.items() if not found or a != found[0]}, sort_keys=True, default=str)
                if sig in self.__dict__.get("ceil_bad", set()) or key in (self.__dict__.get("ceil_seen") or {}):
                    continue
                self.__dict__.setdefault("ceil_seed_keys", set()).add(key)
                self.walk_call(args, f"The cheapest is asked for: I list '{label}' priced below the pick ({pick[1]:g}) to see whether this "
                                     "store's price filter holds.")
                self.log.append({"kind": "ceil_seed", "product": k, "args": args, "learned": self.ceil_spec() is not None})
                if self.chat and self.chat[-1].get("role") == "tool" and self.ceil_spec() is None:
                    self.chat[-1]["content"] = str(self.chat[-1]["content"]) + "\n" + self.status_line()
                return True
            return False
        except Exception as err:
            self.log.append({"kind": "ceil_seed_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def unfound_product(self) -> bool:
        """Tells whether a product of a multi-product request still lacks coverage: it has no home shelves or none of
        them was read yet, or items passed the check under the current needs version while unverified() holds. False for
        single-product requests or on error.
        """
        if not self.needs.items():
            return False
        try:
            read = {s.strip().lower() for s in self.archive.shelves_read()}
            if any(not home or not home & read for home in self.home_shelves()):
                return True
            live = [k for k in self.passed if k.rpartition("|")[2] == str(self.needs.version)]
            return bool(live) and self.unverified()
        except Exception:
            return False

    def walk_hold(self, call_args: dict[str, Any] | None) -> str:
        """Holds back a cart or order call for cheapest requests while cheaper fits may sit unread.
        Plans a price-ceiling walk where ready (the lowest line price as the aim)
        and a walk over remaining search pages within half the moves left.
        When either applies, stores walk_queue and returns a 'Not run yet' note.
        Else (a walk queued, time short or a late refusal) returns ''.
        Errors are logged as walk_hold_error.
        """
        try:
            if self.needs.priority() != "cheapest" or self.__dict__.get("walk_queue") or self.wire.spare() < CLOSING + 60 \
                    or (self.refusal_n and self.moves_left() <= 15):
                return ""
            ids = [i for i, c in self.trolley.lines(call_args) if c is None or c > 0] if call_args is not None else list(self.trolley.units)
            only = {self.product_index(i) for i in ids if i}
            floor = FIRST_ORDER_BY + 1 if self.orders_tried == 0 else 4
            half = (self.moves_left() - floor - 1) // 2
            room = half - (1 if self.batch_tool() else 2)
            ceil = []
            aims = self.__dict__.setdefault("ceil_aims", {})
            for k in sorted(only, key=lambda x: -1 if x is None else x):
                if room >= 2 and self.ceiling_ready(k):
                    ceil.append(k)
                    room -= CEIL_CALLS
                    costs = [float(p) for p in (self.archive.price(i) for i in ids if i and self.product_index(i) == k) if is_number(p)]
                    if costs:
                        aims[k] = min(costs + ([float(aims[k])] if is_number(aims.get(k)) else []))
            plan = self.walk_plan(min(WALK_TOTAL, room), only)
            if not plan and not ceil:
                return ""
            self.walk_queue = (plan, max(floor, self.moves_left() - half - 1), ceil)
            book = self.__dict__.get("page_book", {})
            queries = [" ".join(self.free_text(book[key]["args"])) for _, key, _ in plan if key in book]
            parts = (["reads the remaining pages of " + ", ".join(f"'{q}'" for q in queries)] if queries else []) + (
                ["lists the category's listings priced below the pick with the store's price filter"] if ceil else [])
            return ("Not run yet (no step used). The cheapest is asked for, and this store lists search results in a fixed id order, "
                    "not by price, so a cheaper fit may sit on pages never read: the code " + " and ".join(parts)
                    + " for you now and opens the cheapest unread rows below the pick. Then send the call again, or the same call "
                    "with a cheaper fit.")
        except Exception as err:
            self.log.append({"kind": "walk_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def search_note(self, call_args: dict[str, Any], obs: Any) -> str:
        """Learns from a search reply and returns remarks for the model.
        Finds the rows and the total; an empty reply is a dead search.
        Logs the page, the price-bound mode and the rows' best value score.
        Remarks: lead words only, one-letter words, property-name queries,
        re-spelled queries with the same rows.
        A complete all-words search records its hits and names rows to open.
        A partial one explains paging and narrowing.
        Warns of rows over the call's price limit.
        Returns the remarks text.
        """
        if not isinstance(obs, dict):
            return ""

        def rows_of(piece: dict[str, Any]) -> list | None:
            """Returns the first non-empty list of dicts among a dict's values, else the first such list (even empty), else
            None.
            """
            lists = [val for val in piece.values() if isinstance(val, list) and all(isinstance(r, dict) for r in val)]
            return next((val for val in lists if val), lists[0] if lists else None)
        rows = rows_of(obs)
        if rows is None:
            obs = next((val for val in obs.values() if isinstance(val, dict) and rows_of(val) is not None), None)
            if obs is None:
                return ""
            rows = rows_of(obs) or []
        totals = [val for val in obs.values() if isinstance(val, int) and not isinstance(val, bool) and val > len(rows)]
        if not rows:
            self.learn_dead(call_args)
        self.book_page(call_args, rows, max(totals or [len(rows)]), obs)
        try:
            self.ceil_learn(call_args, rows, max(totals or [len(rows)]), obs)
        except Exception as err:
            self.log.append({"kind": "ceil_learn_error", "error": f"{type(err).__name__}: {err}"[:300]})
        score = self.value_score(call_args) if rows else 0
        if score:
            lead_rows = self.__dict__.setdefault("row_score", {})
            for r in rows:
                if isinstance(r, dict):
                    lead_rows[self.row_mark(r)] = max(lead_rows.get(self.row_mark(r), 0), score)
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
        raw = [val for key, val in call_args.items() if isinstance(val, str) and key not in dict(mode) and key not in paging_keys]
        if first and raw:
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
            elif not probe and self.__dict__.get("trap_told", 0) < 2 and self.moves_left() > self.order_due() + 3:
                trap, _ = self.word_trap(raw)
                if trap:
                    self.trap_told = self.__dict__.get("trap_told", 0) + 1
                    remarks.append(trap)
            marks = tuple(self.row_mark(r) for r in rows)
            others = tuple(sorted((k2, str(v2)) for k2, v2 in call_args.items() if k2 not in paging_keys and not (isinstance(v2, str) and k2 not in dict(mode))))
            sig = (mode, others, max(totals or [len(rows)]), marks)
            book = self.__dict__.setdefault("page_sigs", {})
            query = " ".join(raw)
            if len(set(marks)) >= 3 and sig in book and book[sig] != query:
                remarks.append(f"Same total and rows as your earlier search '{book[sig]}': the store read both alike, so re-spelling "
                               "it shows nothing new. Read its next page, or change its leading words.")
            book.setdefault(sig, query)
            hits = []
            told = bool((kind == "all" or self.lead_words) and not totals and rows)
            if told or (kind != "any" and not totals and rows):
                held = None if told else list(self.__dict__.get("ref_hits") or [])
                self.ref_hits = []
                hits = self.row_hits(rows, raw)
                score = self.value_score(call_args)
                ranks = self.__dict__.setdefault("hit_score", {})
                ranks.update({i: max(ranks.get(i, 0), score) for i in hits})
                found = self.__dict__.setdefault("all_hits", [])
                found += [i for i in hits if i not in found]
                if held is not None:
                    self.ref_hits = held
                if told and hits and len(texts) and len(rows) <= 12:
                    remarks.append("Every word is required here and every match is shown: each row carries the words the store read "
                                   "somewhere in its record, often in fields its title does not show. Open the in-budget ones before "
                                   "searching on: " + ", ".join(hits[:6]) + ".")
                pointers = [i for i in self.__dict__.pop("ref_hits", []) if i not in hits][:6] if told else []
                if pointers:
                    remarks.append("The request gives a value by pointing at another listing, and these rows are sold by the shop or brand "
                                   "this search names: open them whatever their price, stock or category (a listing that gives a value "
                                   "is read, not bought), and follow what its notes say: " + ", ".join(pointers) + ".")
            if (kind == "all" or self.lead_words) and not self.__dict__.get("hits_back"):
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
        """Collects shopper text from a tool reply. For a message call it takes the reply string, or the strings that
        long_strings gathers from a dict reply without an error; for other calls it gathers from lists of strings, and,
        when the call was refused, also takes any string field of four or more words.
        """
        lns: list[str] = []
        if call_name == self.role("message"):
            if isinstance(obs, str):
                lns = [obs]
            elif isinstance(obs, dict) and "error" not in obs:
                long_strings([val for val in obs.values() if isinstance(val, (str, list))], lns)
        elif isinstance(obs, dict):
            for piece in obs.values():
                if isinstance(piece, list) and piece and all(isinstance(val, str) for val in piece):
                    long_strings(piece, lns)
                elif isinstance(piece, str) and len(piece.split()) >= 4 and refused:
                    lns.append(piece)
        return lns

    def screen(self, proposal: dict[str, Any], permitted: set[str], scripted: bool, batch: int, queued_messages: int) -> tuple[str | None, str, dict[str, Any]]:
        """Pre-run call screen: (refusal text or None, tool name, checked args).
        Refuses tools unknown or not allowed now (messages during asking wait
        in want_ask), non-JSON-object or ArgCheck-failing args, a finisher
        with others, messages over cap or into the basket-statement reserve,
        unchanged cart views, exact recent repeats (not finishers, messages,
        cart calls). Unscripted calls also face late_hold, value_guard,
        dead_code_guard; cart calls screen_cart. Refusals say no step used.
        """
        func = proposal.get("function") or {}
        call_name, raw_args = func.get("name"), func.get("arguments")
        if call_name not in self.shapes:
            return f"Not run (no step used): '{call_name}' is not one of the tools. Use one of: {', '.join(sorted(permitted))}.", call_name, {}
        if not scripted and call_name not in permitted:
            if call_name == self.role("message") and self.asking:
                self.want_ask(raw_args)
                return "Not run (no step used): questions to the shopper are planned and sent for you after you have opened a few listings' details.", call_name, {}
            return "Not run (no step used): too few steps are left for this; only these tools can be used now: " + ", ".join(sorted(permitted)) + ".", call_name, {}
        call_args = raw_args if isinstance(raw_args, dict) else (json_object(raw_args) if (raw_args or "").strip() else {})
        if call_args is None:
            return "Not run (no step used): the arguments were not a valid JSON object.", call_name, {}
        call_args, faults = ArgCheck.execute(call_args, self.shapes[call_name])
        if faults:
            return "Not run (no step used): invalid arguments: " + "; ".join(faults[:6]), call_name, call_args
        if call_name in self.finishers() and batch > 1:
            return "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls.", call_name, call_args
        if call_name == self.role("message") and self.msg_cap is not None and self.msgs_sent + queued_messages >= self.msg_cap:
            return "Not run (no step used): the message budget is used up. Do not message the shopper again.", call_name, call_args
        if not scripted and call_name == self.role("message") and self.states_basket() and self.msgs_sent + queued_messages + 1 > self.ask_limit():
            return ("Not run (no step used): the messages left are kept for the basket statement, which the code sends to the "
                    "shopper right before the order (and again if a refused order makes the cart change). Do not message the "
                    "shopper; go on with the cart and the order."), call_name, call_args
        if not scripted and call_name == self.role("cart_view") and call_name != self.role("cart") and not self.__dict__.get("batch_cart") \
                and self.trolley.units and self.__dict__.get("cart_shown", -1) == self.__dict__.get("cart_ver", -2) \
                and not self.__dict__.get("since_cart", 1):
            told = self.__dict__.setdefault("view_told", set())
            if self.cart_shown not in told:
                told.add(self.cart_shown)
                return ("Not run (no step used): the last cart result already shows the basket - " + self.trolley.summary()
                        + " A cart view now shows nothing new; to re-check a line's stock, read its record with "
                        + (self.stock_tool() or "the tool that shows stock") + "."), call_name, call_args
        sig = call_name + as_text(call_args)
        if call_name not in self.finishers() and call_name not in (self.role("message"), self.role("cart")) and sig in self.last_calls:
            return "Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else.", call_name, call_args
        held = "" if scripted else self.late_hold(call_name, call_args)
        held = held or ("" if scripted else self.value_guard(call_name, call_args))
        held = held or ("" if scripted else self.dead_code_guard(call_name, call_args))
        if held:
            return held, call_name, call_args
        if call_name == self.role("cart"):
            refusal = self.screen_cart(call_args, sig)
            if refusal:
                return refusal, call_name, call_args
        return None, call_name, call_args

    def screen_cart(self, call_args: dict[str, Any], sig: str) -> str:
        """Cart call screen: refusal text or '' to allow.
        Refuses changes once the stated basket locks, and puts of differ-reference targets.
        Warns once before removing an item-checked listing.
        Puts: warn once if held; defer to walk_hold; item-check unchecked lines;
        the rank check queues up to 2 better rows once;
        for cheapest requests, names a cheaper verified fit once.
        Last, unit counts per line are checked, once per call.
        Returns the refusal text or ''.
        """
        if self.stated_lock():
            return ("Not run (no step used): the basket was stated to the shopper and no message is left to state a changed one, "
                    "and the stated basket can still be bought (in stock, inside the budget): leave the cart as it is and "
                    "order it.")
        thing = self.trolley.aim(call_args)
        point = self.ref_pointer_note(thing) if thing and self.trolley.is_put(call_args) else ""
        pointed = self.__dict__.setdefault("pointer_told", {})
        if point and pointed.get(thing, 0) < 2:
            pointed[thing] = pointed.get(thing, 0) + 1
            return f"Not run (no step used): {point}. Choose another listing."
        mark = f"{sig}#{self.__dict__.get('puts', {}).get(thing, 0)}"
        mine = self.archive.price(thing) if thing else None
        cheaper_put = any(is_number(mine) and is_number(self.archive.price(p)) and self.archive.price(p) < mine
                          for p in self.__dict__.get("batch_puts", []) if p != thing)
        swap = self.needs.priority() == "cheapest" and (cheaper_put or bool(self.cheaper_fit(thing)))
        moved = thing in self.__dict__.get("pass_price", {}) and self.__dict__["pass_price"][thing] != mine
        if thing and not self.trolley.is_put(call_args) and f"{thing}|{self.needs.version}" in self.passed and not swap and not moved \
                and self.refusal_n == self.refusals_seen.get(thing, self.refusal_n) and mark not in self.warned:
            self.warned.add(mark)
            least = " It is also the cheapest verified fit." if self.needs.priority() == "cheapest" and not self.cheaper_fit(thing) else ""
            return (f"Not run (no step used): an independent check found {thing} meets every requirement now in force, reading each "
                    "property under the value rules (where this store writes the value itself in a field, only those words meet it); the category label alone decides the item kind, whatever object its title or description "
                    f"names.{least} Nothing has changed since. If a change or a refusal rules it out, one of its fields clearly fails "
                    "a requirement, or you are replacing it with a cheaper listing, send the same call again.")
        if not self.trolley.is_put(call_args) or not thing:
            return ""
        if self.trolley.held(thing) > 0 and self.trolley.adds is not False and sig not in self.warned:
            self.warned.add(sig)
            return (f"Not run (no step used): the cart already holds {thing} x{int(self.trolley.held(thing))}, and this cart tool adds units to a held line, "
                    f"so this call would raise it. To order it, leave the cart as it is; to change the units, take the difference out or in. If you really mean to add, send the same call again.")
        walk = "" if point else self.walk_hold(call_args)
        if walk:
            return walk
        for other, count in self.trolley.lines(call_args)[1:]:
            if (count is None or count > 0) and self.moves_left() > 3 and f"{other}|{self.needs.version}" not in self.checked_adds:
                self.checked_adds.add(f"{other}|{self.needs.version}")
                ruling = self.check_item(other, "ADD TO THE CART")
                if ruling:
                    return "Not run (no step used). " + ruling + " If you are certain it fits, send the same call again."
        if self.moves_left() > 3 and f"{thing}|{self.needs.version}" not in self.checked_adds:
            self.checked_adds.add(f"{thing}|{self.needs.version}")
            ruling = self.check_item(thing, "ADD TO THE CART")
            if ruling:
                return "Not run (no step used). " + ruling + " If you are certain it fits, send the same call again."
        for line_id, count in self.trolley.lines(call_args) if self.moves_left() > 3 else []:
            if count is not None and count <= 0:
                continue
            better = self.rank_check(line_id)
            if better:
                cheaper = self.rank_verify(line_id)
                unread = [x for x in self.__dict__.get("rank_better") or [] if x not in self.archive.read
                          and x not in self.dead_lines()][:2]
                if unread and not cheaper and not self.__dict__.get("open_queue") and not self.__dict__.get("rank_opened") \
                        and self.moves_left() > 6 and (self.role("details") or self.batch_tool()):
                    self.rank_opened = True
                    self.open_queue = unread
                    self.open_say = "I open the listings the rank check named as cheaper fits, so they can be checked."
                return "Not run (no step used). " + better.replace("before ordering", "before putting it in the cart").replace(
                    "order the current pick", "send the same call again") + (
                    f" The independent item check found that {cheaper} meets every requirement now in force and costs less than "
                    f"{line_id}: put {cheaper} in instead." if cheaper else "")
        for line_id, count in self.trolley.lines(call_args) if self.needs.priority() == "cheapest" else []:
            if count is not None and count <= 0:
                continue
            cheaper = self.cheaper_fit(line_id)
            told = self.__dict__.setdefault("cheaper_told", {})
            if cheaper and cheaper not in self.trolley.units and not told.get(("cart", line_id, cheaper)):
                told[("cart", line_id, cheaper)] = 1
                return (f"Not run (no step used). The independent item check found that {cheaper} meets every requirement now in "
                        f"force, and it costs less than {line_id}. The cheapest fit is asked for and the check's reading stands over "
                        f"yours: put {cheaper} in instead. If a record shows {cheaper} failing a requirement now in force, send the "
                        "same call again.")
        need_n = self.needs.quantity()
        count = self.trolley.count_arg(call_args)
        aim = need_n - self.trolley.held(thing) if self.trolley.adds is not False else need_n
        if self.needs.items():
            odd = [(i, int(c)) for i, c in self.trolley.lines(call_args) if c is not None and c > 0 and int(c) not in self.needs.counts()]
            if odd and sig not in self.warned:
                self.warned.add(sig)
                return (f"Not run (no step used): the products asked for need {self.needs.counts()} units (one line each), but this call "
                        f"asks for {odd}. If you are sure, send the same call again.")
            return ""
        if count is not None and abs(count - aim) > 1e-9 and sig not in self.warned:
            self.warned.add(sig)
            return f"Not run (no step used): the request needs a quantity of {need_n} of one listing, but this call asks for {int(count)}. If you are sure, send the same call again."
        return ""

    def screen_finish(self, cid: str, call_name: str, call_args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """Screens an order or decline call: returns (refusal text or None, arguments to send). A locked stated
        basket goes at once. An order is held with a 'Not run yet' note for an empty cart, a cart over the budget,
        sold-out lines (swapped by the code on the third try), a cheaper verified fit (swapped in on the third try),
        a pending walk_hold, layout-A cart lines missing a page, lines whose stock was not read since the last cart
        change, and once for count faults, item and rank checks and a checklist; a due statement goes to state_gate.
        A decline runs a chain of one-time steps: near_hold, keeping verified lines, opening cheaper rows that may fit
        the budget together, unread rows of products with no verified fit, listings the checks named, decline_reads,
        the decline review, fit_hold, unrefuted candidates, the unresolved-reference note, decline_repair, near_check
        and last_resort_cart;
        if all pass it sends scrubbed decline arguments.
        """
        if self.call_kind(call_name, call_args) == "order" and self.stated_lock():
            self.log.append({"kind": "stated_lock", "cart": dict(self.trolley.units)})
            return None, call_args
        if self.call_kind(call_name, call_args) == "order" and not self.trolley.units and self.role("cart"):
            empty = self.__dict__.setdefault("empty_told", {})
            mark = self.__dict__.get("cart_seq", 0)
            if empty.get(mark, 0) < 3:
                empty[mark] = empty.get(mark, 0) + 1
                return ("Not run yet (no step used). The cart is empty, so an order now would fail and spend a step. Put the "
                        "chosen listing" + ("s, one line per product at its quantity," if self.needs.items() else " at its quantity")
                        + " in the cart first (send the cart call again if it was held and you still mean it), then order."), call_args
        if self.call_kind(call_name, call_args) == "order" and self.trolley.units and self.moves_left() > 4:
            over = self.cart_over_cap()
            told_over = self.__dict__.setdefault("over_told", {})
            mark_over = (self.__dict__.get("cart_seq", 0), str(self.needs.spend_cap))
            if over and told_over.get(mark_over, 0) < 2:
                told_over[mark_over] = told_over.get(mark_over, 0) + 1
                self.log.append({"kind": "over_cap_hold", "total": over[0], "cap": over[1]})
                fig = lambda x: f"{int(x)}" if float(x).is_integer() else f"{float(x):.2f}"
                return (f"Not run yet (no step used). At the newest prices the cart totals {fig(over[0])}, over the budget in "
                        f"force ({fig(over[1])}), so this order would fail. Put in the cart the cheapest listings that meet every "
                        "requirement within that budget (the fits read so far, or a search for more), then order; decline only "
                        "when the records show none within it."), call_args
        if self.call_kind(call_name, call_args) == "order" and self.moves_left() > 2:
            dead = sorted(set(self.trolley.units) & self.dead_lines())
            mark = (tuple(dead), self.__dict__.get("cart_seq", 0))
            told = self.__dict__.setdefault("dead_told", {})
            if dead and told.get(mark, 0) < 2:
                told[mark] = told.get(mark, 0) + 1
                return "Not run yet (no step used). " + self.dead_note(dead), call_args
            if dead and told.get(mark, 0) == 2 and self.role("cart") and self.moves_left() >= 6 and self.wire.spare() > CLOSING:
                told[mark] = 3
                was = self.wire.mute
                try:
                    done = self.code_cart(set(dead) | self.dead_lines(), set(self.trolley.units) - set(dead))
                except Exception as err:
                    done = False
                    self.log.append({"kind": "dead_swap_error", "error": f"{type(err).__name__}: {err}"[:300]})
                finally:
                    self.wire.mute = was
                self.log.append({"kind": "dead_swap", "lines": dead, "done": done, "cart": dict(self.trolley.units)})
                if done and self.trolley.units and not set(self.trolley.units) & self.dead_lines():
                    return ("Not run yet (no step used). The cart still held " + ", ".join(dead) + ", which is sold out, so the code "
                            "took it out and put a candidate in instead, keeping the other lines, and read the lines' stock: "
                            + self.trolley.summary() + " Check this cart against the request, then order it."), call_args
        if self.call_kind(call_name, call_args) == "order" and self.moves_left() > 3:
            thing = self.chosen()
            cheaper = self.rank_verify(thing) if thing else ""
            for line in ([] if cheaper else [i for i in self.trolley.units if i != thing]):
                cheaper = self.cheaper_fit(line)
                if cheaper:
                    thing = line
                    break
            told = self.__dict__.setdefault("cheaper_told", {})
            if cheaper and told.get((thing, cheaper), 0) >= 2 and not told.get(("swap", thing, cheaper)) and thing in self.trolley.units \
                    and self.moves_left() > 6 and not self.stated_lock() and (not self.states_basket() or self.msgs_left() > 0) \
                    and self.last_stock(cheaper) is not False and self.role("cart") and not self.wire.mute:
                told[("swap", thing, cheaper)] = 1
                picks = [((cheaper if i == thing else i), int(round(float(u)))) for i, u in self.trolley.units.items()]
                was = self.wire.mute
                self.wire.mute = True
                try:
                    bad = self.code_put(picks)
                finally:
                    self.wire.mute = was
                self.log.append({"kind": "cheaper_swap", "from": thing, "to": cheaper, "bad": bad, "cart": dict(self.trolley.units)})
                if bad == [] and cheaper in self.trolley.units and thing not in self.trolley.units:
                    return (f"Not run yet (no step used). The independent item check found that {cheaper} meets every requirement and "
                            f"costs less than {thing}; the cheapest fit is asked for, so the code put {cheaper} in the cart in its place: "
                            + self.trolley.summary() + " Re-read its stock and price, and order it."), call_args
            if cheaper and told.get((thing, cheaper), 0) < 2:
                told[(thing, cheaper)] = told.get((thing, cheaper), 0) + 1
                return (f"Not run yet (no step used). The independent item check found that {cheaper} meets every requirement, and "
                        f"it costs less than {thing}. The cheapest fit is asked for and the check's reading stands over yours: take "
                        f"{thing} out of the cart, put {cheaper} in, re-read its stock and price, and order it."), call_args
        if self.call_kind(call_name, call_args) == "order" and self.trolley.units:
            walk = self.walk_hold(None)
            if walk:
                return walk, call_args
        if self.call_kind(call_name, call_args) == "order" and self.trolley.units and self.split_pages():
            paged = self.unpaged()
            if paged and self.moves_left() > len(paged[1][:4]) + 1 and self.wire.spare() > CLOSING \
                    and self.lookup_call(paged[1][0], paged[0]):      # (A2b: only a read the code can build)
                # (A2: layout A judges an order on both pages of every line: the missing pages are opened first)
                tool, ids = paged[0], paged[1][:4]
                self.__dict__.setdefault("paged_tried", set()).update((tool, i) for i in ids)
                self.open_queue = ids
                self.open_recheck = True
                self.open_purpose = "pages"
                self.open_tool = tool
                self.open_say = "Before ordering, I open each cart line with every page tool the store has."
                return ("Not run yet (no step used). This store shows a listing on two pages, and an order is judged on both: "
                        "these cart lines were never opened with " + tool + ": " + ", ".join(ids) + ". They are read for you now; "
                        "then order if every line still fits (in stock, within the budget, nothing on its pages against a "
                        "requirement), else replace the line."), call_args
        reader = self.stock_tool() if self.call_kind(call_name, call_args) == "order" else None
        if reader and self.moves_left() > len(self.trolley.units) + 2 and self.wire.spare() > CLOSING:
            stale = self.stale_lines()
            mark = (tuple(sorted(self.trolley.units)), self.__dict__.get("cart_seq", 0))
            if stale and self.__dict__.get("recheck_mark") != mark:
                self.recheck_mark = mark
                self.open_queue = stale[:4]
                self.open_recheck = True
                self.open_purpose = "cart"
                self.open_tool = reader
                return ("Not run yet (no step used). Stock and prices can change without any notice, and these cart lines were not "
                        "read since the last cart change: " + ", ".join(stale[:4]) + f". Their stock is read for you now ({reader}, the "
                        "tool that shows stock); then order if each is still in stock and the total is within the budget, else replace "
                        "the line."), call_args
        if self.call_kind(call_name, call_args) == "order" and not self.order_vetted and self.moves_left() > 2 and self.wire.spare() > CLOSING:
            self.order_vetted = True
            faults = self.trolley.faults(self.needs.quantity(), self.needs.counts())
            if faults and self.moves_left() > 3:
                return "Not run yet (no step used). The cart as counted is not ready: " + "; ".join(faults) + ". Fix it, then order.", call_args
            lines = list(self.trolley.units) if self.needs.items() else ([self.chosen()] if self.chosen() else [])
            for thing in lines:
                ruling = self.check_item(thing, "ORDER")
                if ruling:
                    return "Not run yet (no step used). " + ruling, call_args
            for thing in lines:
                better = self.rank_check(thing)
                if better:
                    cheaper = self.rank_verify(thing)
                    return "Not run yet (no step used). " + better + (
                        f" The independent item check found that {cheaper} meets every requirement now in force and costs less "
                        f"than {thing}: take {thing} out, put {cheaper} in, re-read its stock and price, and order." if cheaper else ""), call_args
            return ("Not run yet (no step used). Pre-order check: (1) the cart holds exactly one line " + ("per product asked for, each at its quantity" if self.needs.items() else "the chosen item at the required quantity") + ", and nothing else; "
                    "(2) after your last cart change you re-read this item's stock and price with a tool that shows them, and it is in stock and price times quantity is within the current budget; "
                    "(3) its own properties meet every stated requirement, every need the shopper revealed and every change announced; "
                    "(4) if the cheapest or best was asked for, no better fitting item exists among everything you could find; "
                    "(5) every procedural rule in the request has been followed. If anything is unchecked, check it now; otherwise call the order tool again."), call_args
        if self.call_kind(call_name, call_args) == "decline":
            near_note = self.near_hold()
            if near_note:
                return near_note, call_args
            passed_lines = [i for i in self.trolley.units if f"{i}|{self.needs.version}" in self.passed and i not in self.refused_lines
                            and i not in self.dead_lines()]
            if passed_lines and self.moves_left() > 1 and not self.__dict__.get("decline_kept"):
                self.decline_kept = True
                return ("Not run yet (no step used). The cart holds " + ", ".join(passed_lines) + ", which the independent check "
                        "found meets every requirement now in force. A decline would throw that away: re-read its stock and price and "
                        "order it" + (" with every product's line in the cart" if self.needs.items() else "") + ". Decline only if a "
                        "change since then clearly rules it out."), call_args
            slack = self.slack_rows() if not self.__dict__.get("slack_done") and (self.role("details") or self.batch_tool()) else []
            spread = self.batch_tool()
            reads = -(-len(slack) // spread[2]) if spread and len(slack) >= 2 else len(slack)
            if slack and self.moves_left() > reads + 4 and self.wire.spare() > CLOSING + 15:
                self.slack_done = True
                self.open_queue = slack
                self.open_say = "Before declining, I open cheaper rows that may let every product fit the budget together."
                total = self.__dict__.get("slack_total")
                return ("Not run yet (no step used). Every product has a fitting listing, but the cheapest fits together cost "
                        + (f"{total:g}" if is_number(total) else "more than the budget") + f", over the budget of {self.needs.spend_cap}. "
                        "Rows priced within what the budget leaves each product after the other products' cheapest fits were never "
                        "opened, so their records are read for you now: " + ", ".join(slack) + ". Judge each against its own "
                        "product's requirements under the value rules (where this store writes the value itself in a field, only "
                        "those words meet it); when fits within the budget together cover every product, put every product's line "
                        "in the cart and order; decline only if no such set exists."), call_args
            batch = self.batch_tool()
            fill = self.lacking_rows() if not self.__dict__.get("decline_fill") and (self.role("details") or batch) else []
            reads = -(-len(fill) // batch[2]) if batch and len(fill) >= 2 else len(fill)
            if fill and self.moves_left() > reads + 3 and self.wire.spare() > CLOSING + 15:
                self.decline_fill = True
                self.open_queue = fill
                self.open_say = "Before declining, I open unread listings of the products that have no verified fit yet."
                return ("Not run yet (no step used). Some products asked for have no verified fit yet, and listings of their "
                        "categories within the budget were never opened, so their records are read for you now, before any "
                        "decline: " + ", ".join(fill) + ". Judge each against its own product's requirements "
                        "under the value rules (where this store writes the value itself in a field, only those words meet it); when "
                        "every product has a fitting listing, put every product's line in the cart and order; decline only if "
                        "a product truly has none."), call_args
            queue = self.named_unread()[: batch[2] if batch else 3]
            reads = -(-len(queue) // batch[2]) if batch and len(queue) >= 2 else len(queue)
            if queue and not self.__dict__.get("decline_opens") and (self.role("details") or batch) and self.moves_left() > reads + 3 \
                    and self.wire.spare() > CLOSING + 15:
                self.decline_opens = True
                self.open_queue = queue
                return ("Not run yet (no step used). Listings the checks named were never opened, so their records are read for you now, "
                        "before any decline: " + ", ".join(queue) + ". Judge each against the requirements under the value rules (a field of the "
                        "property's own name decides, and where this store writes the value itself there only those words meet it; "
                        "when none exists, an alternative field, the options list or the title may show the "
                        "value), buy one that fits, and decline only if each one fails."), call_args
            pages_note = self.decline_reads()
            if pages_note:
                return "Not run yet (no step used). " + pages_note, call_args
            reread = len(self.archive.read) > self.__dict__.get("audit_reads", 10 ** 9) and self.__dict__.get("audits", 0) < 2
            if reread and self.decline_vetted and self.moves_left() > 3 and self.wire.spare() > CLOSING + 30:
                audit_note = self.review_decline()
                if audit_note:
                    return "Not run yet (no step used). " + audit_note, call_args
            if not self.decline_vetted and self.moves_left() > 3 and self.wire.spare() > CLOSING + 30:
                self.decline_vetted = True
                audit_note = self.review_decline()
                if audit_note:
                    return "Not run yet (no step used). " + audit_note, call_args
                return ("Not run yet (no step used). Before declining, be sure: you searched in several ways (different words, filters, pages), you checked every candidate's own properties, "
                        "and each one fails a stated requirement, a revealed need, the budget or stock. Any key spelling of a property's field counts. "
                        "If something does fit, buy it instead; if nothing fits, call the decline tool again."), call_args
            fit_note = self.fit_hold()
            if fit_note:
                return "Not run yet (no step used). " + fit_note, call_args
            best = self.unrefuted() if not self.__dict__.get("best_held") and self.moves_left() > 2 else []      # (UB: was > 4)
            for _ in range(3):
                unchecked = [i for i in best if f"{i}|{self.needs.version}" not in self.passed]
                if not unchecked or self.wire.spare() < CLOSING + 30:
                    break
                self.check_item(unchecked[0], "ORDER")
                best = self.unrefuted()
            missing = [i for i in best if i not in self.trolley.units]
            extra = [i for i in self.trolley.units if i not in best]
            late_ok = bool(best) and all(f"{i}|{self.needs.version}" in self.passed for i in best) and not self.ties \
                and not re.search(r"must (?:work with|fit)|compatible with|\bfits? (?:the|my|your|our|this|that)\b", self.request, re.I)
            pages = 0
            if late_ok and self.split_pages():
                seen_pages = self.__dict__.get("page_seen") or {}
                pages = sum(1 for t in self.page_tools() for i in best if i not in (seen_pages.get(t) or set()))
            # (UB: verified picks need their missing puts and the order; UBb: stray lines out, layout A's page reads)
            need = len(missing) + len(extra) + pages + 1 if late_ok else len(best) + 3
            if best and self.moves_left() > need and self.wire.spare() > (LAST_CALL + 12 if late_ok else CLOSING + 20):
                self.best_held = True
                several = bool(self.needs.items())
                return ("Not run yet (no step used). A decline is right only when no listing can meet the requirements, and the "
                        "records hold " + ("a candidate for every product asked for" if several else "a candidate") + " that is in "
                        "stock, within the budget and that no check ruled out: " + ", ".join(best) + ". Unless a record shows one of "
                        "them failing a requirement now in force, put " + ("one line of each product, at its quantity," if several
                        else "it") + " in the cart, re-read stock and price, and order; decline only if each candidate clearly "
                        "fails."), call_args
            soft = [r for r in self.__dict__.get("refs") or [] if r.get("relation") == "differ" and r.get("status") in ("open", "stuck")
                    and self.ref_live(r)]
            if soft and self.moves_left() > 3 and not self.__dict__.get("ref_decline_held"):
                self.ref_decline_held = True
                heads = ", ".join(dict.fromkeys(r["head"] for r in soft))
                return ("Not run yet (no step used). " + heads + " is given by another listing that the code could not resolve, so its "
                        "value is unknown: it never rules a listing out and never justifies a decline. Put the listing that meets every "
                        "other requirement (in stock, within the budget) in the cart" + (", one line per product" if self.needs.items() else "")
                        + ", re-read its stock and price, and order; decline only if no listing meets the other requirements."), call_args
            repair = self.decline_repair()
            if repair:
                return repair, call_args
            near = self.near_check()
            if near:
                return near, call_args
            last = self.last_resort_cart()
            if last:
                return last, call_args
            return None, self.scrub_decline(call_args, call_name)
        if self.call_kind(call_name, call_args) == "order" and self.statement_due():
            held = self.state_gate(cid)
            if held:
                return held, call_args
        return None, call_args

    def dispatch(self, proposals: list[dict[str, Any]], scripted: bool = False) -> list[tuple[str, str]]:
        """Screens, sends a turn's tool calls; returns (call id, reply text)
        pairs in order. Notes listings the batch carts; runs screen per call
        (after a held cart change, later cart calls refused); counts queued
        messages. A leading finisher also passes screen_finish; held, it ends
        the turn. Else calls past moves left - 1 are refused, keeping the last
        step for order/decline. Sent in chunks of <= 16 via send_chunk, cut to
        moves left; a pending basket-statement note precedes its call's reply.
        """
        responses: list[tuple[str, str]] = []
        self.begin_batch()
        permitted = {tdef["function"]["name"] for tdef in self.offered()}
        to_send: list[tuple[str, str, dict[str, Any]]] = []
        queued_messages = 0
        cart_held = False
        self.batch_puts = []
        self.batch_cart = any((p.get("function") or {}).get("name") == self.role("cart") for p in proposals if isinstance(p, dict))
        for proposal in proposals:
            func = proposal.get("function") or {}
            if func.get("name") == self.role("cart"):
                raw = func.get("arguments")
                put_args = raw if isinstance(raw, dict) else (json_object(raw) if isinstance(raw, str) and raw.strip() else {})
                if isinstance(put_args, dict) and self.trolley.is_put(put_args) and self.trolley.aim(put_args):
                    self.batch_puts.append(self.trolley.aim(put_args))
        for proposal in proposals:
            if cart_held and proposal["function"]["name"] == self.role("cart"):
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
        for begin in range(0, len(to_send), 16):
            slab = to_send[begin: begin + 16]
            if self.wire.done or self.wire.broke or self.moves_left() <= 0:
                responses += [(cid, "Not run: the episode is over or no steps are left.") for cid, _, _ in slab]
                continue
            slab = slab[: self.moves_left()]
            responses += self.send_chunk(slab)
        call_rank = {c["id"]: i for i, c in enumerate(proposals)}
        responses.sort(key=lambda req: call_rank.get(req[0], 0))
        stated = self.__dict__.pop("stmt_note", None)
        if stated:
            responses = [(cid, stated[1] + "\n" + text_body) if cid == stated[0] else (cid, text_body) for cid, text_body in responses]
        return responses

    def send_chunk(self, slab: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str]]:
        """Sends one chunk of calls to the world as a single act request, logs it as an env entry, and turns each
        observation into reply text with absorb (the last one also learns whether the episode finished). Returns a
        failure text for every call when no reply came, and a 'Not run' text for calls past the returned observations.
        """
        acts = [{"name": call_name, "args": call_args} for _, call_name, call_args in slab]
        prior = self.wire.tick
        resp = self.wire.act(acts)
        self.log.append({"kind": "env", "step": prior + 1, "actions": acts, "reply": resp})
        if resp is None:
            return [(cid, "The call failed (network or server). Try again.") for cid, _, _ in slab]
        obs_list = resp.get("observations") if isinstance(resp.get("observations"), list) else [resp.get("observation")]
        collected = []
        for idx, (cid, call_name, call_args) in enumerate(slab):
            if idx >= len(obs_list):
                collected.append((cid, "Not run: the episode finished before this call."))
                continue
            collected.append((cid, self.absorb(call_name, call_args, obs_list[idx], resp, idx == len(obs_list) - 1)))
        return collected

    def absorb(self, call_name: str, call_args: dict[str, Any], obs: Any, resp: dict[str, Any], last: bool) -> str:
        """Tool observation to model reply: a clipped record plus remarks.
        Archives records; tracks messages, orders, calls and lookups.
        Learns message counts, tool caps, stock times and read ids.
        Remarks: sold-outs, own-note redirects, cart changes and views,
        notes, searches (search_note, merge_joined), an early decline review.
        Shopper lines go to the needs; references to ref_take.
        Flags a finished episode's last line.
        Returns the reply text.
        Used for every tool call.
        """
        msg_tool, basket_tool = self.role("message"), self.role("cart")
        if call_name not in (msg_tool, basket_tool, self.role("cart_view"), self.role("order"), self.role("decline"),
                             self.role("message_status")):
            try:
                turned, hit = text_records(obs)
            except Exception:
                turned, hit = obs, False
            if hit:
                obs = turned
                self.log.append({"kind": "text_records", "tool": call_name})
        if call_name != msg_tool:
            try:
                obs = id_texts(obs)
            except Exception:
                pass
        shown, obs = obs, self.shelf_lens(lensed(obs))
        if call_name in self.page_tools() and not self.code_fault(obs):
            # (A2: the listing ids each page tool opened; A2b: not a call the store refused)
            for val in (call_args or {}).values():
                if isinstance(val, (str, int)) and not isinstance(val, bool):
                    self.__dict__.setdefault("page_seen", {}).setdefault(call_name, set()).add(str(val))
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
        self.absorb_n = self.__dict__.get("absorb_n", 0) + 1
        self.last_obs = obs
        try:
            if call_name and call_name == self.status_tool():
                self.learn_msgs(obs, True)
            elif call_name == msg_tool:
                self.learn_msgs(obs, False)
        except Exception as err:
            self.log.append({"kind": "msg_count_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if call_name != basket_tool and not self.__dict__.get("covering"):
            self.since_cart = self.__dict__.get("since_cart", 0) + 1
        try:
            gone_now = self.learn_caps(call_name, call_args, obs)
        except Exception as err:
            gone_now = []
            self.log.append({"kind": "caps_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if self.trolley.units:
            fresh: set[str] = set()
            stock_rule = bool(self.__dict__.get("stock_seen"))
            asked = self.asked_ids(call_args)
            rows_ok = bool(self.__dict__.get("rows_live")) and not self.__dict__.get("rows_lag")
            lookups_only = not rows_ok and call_name not in (basket_tool, self.role("cart_view"))

            def seen_line(d: dict[str, Any]) -> None:
                """Adds the cart line ids found under id keys in one dict of the reply to the freshly seen set. When the store
                shows stock, only dicts with a stock state count, and in lookup-only replies only lines the call asked about;
                otherwise only dicts that hold nested dicts count.
                """
                hits = {v for k, v in d.items() if isinstance(v, str) and v in self.trolley.units and "id" in str(k).lower()}
                if not hits:
                    return
                if stock_rule:
                    if self.stock_state(d) is None or (lookups_only and not hits & asked):
                        return
                elif not any(isinstance(v, dict) for v in d.values()):
                    return
                fresh.update(v for k, v in d.items() if isinstance(v, str) and v in self.trolley.units)
            walk_dicts(obs, seen_line)
            for line_id in fresh:
                self.__dict__.setdefault("line_read", {})[line_id] = self.wire.tick
                self.__dict__.setdefault("line_seq", {})[line_id] = self.absorb_n
        if call_name == self.role("details") and not (isinstance(obs, dict) and obs.get("error")):
            self.archive.read.update(val for val in call_args.values() if isinstance(val, str) and val in self.archive.ids)
        looked = bool(self.role("details")) and call_name not in (self.role("details"), self.role("search")) and call_name in self.lookup_tools()
        named_ids = {val for val in call_args.values() if isinstance(val, str)}
        if self.moves_left() < self.order_due() and (call_name in (self.role("details"), self.role("search")) or looked) \
                and not self.__dict__.get("covering") \
                and not (call_name == self.role("details") and named_ids & set(self.trolley.units)) \
                and not (looked and named_ids & (set(self.trolley.units) | {k.rpartition("|")[0] for k in self.passed})):
            self.late_n = self.__dict__.get("late_n", 0) + 1
        remarks = []
        if gone_now:
            remarks.append("[stock] " + self.dead_note(gone_now))
        try:
            redirected = self.notes_remark(call_name, call_args)
            if redirected:
                remarks.append("[notes] " + redirected)
        except Exception as err:
            self.log.append({"kind": "notes_remark_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if call_name == basket_tool:
            self.order_vetted = False
            self.trolley.changed_at = self.wire.tick
            held_before = dict(self.trolley.units)
            remarks.append(self.trolley.booked(call_args, obs))
            try:
                self.cart_moved(held_before)
                self.mark_fresh(obs)
                self.after_put(call_args, obs)
            except Exception as err:
                self.log.append({"kind": "after_put_error", "error": f"{type(err).__name__}: {err}"[:300]})
            put_item = self.trolley.aim(call_args)
            if put_item and self.trolley.is_put(call_args):
                self.refusals_seen[put_item] = self.refusal_n
                puts = self.__dict__.setdefault("puts", {})
                puts[put_item] = puts.get(put_item, 0) + 1
            put_units = self.trolley.count_arg(call_args)
            if put_item and self.trolley.is_put(call_args) and f"{put_item}|{self.needs.version}" in self.passed \
                    and (put_units is None or put_units > 0) and self.trolley.held(put_item) > 0 and put_item not in self.dead_lines():
                remarks.append(f"An independent check found {put_item} meets every requirement now in force, reading each property "
                               "under the value rules (where this store writes the value itself in a field, only those words meet it); keep it unless a new change rules it out.")
        if call_name == self.role("cart_view"):
            self.trolley.viewed(obs, self.wire.tick)
            remarks.append(self.trolley.summary())
        refused = self.refused(call_name, obs, resp if last else {"finished": False})
        if refused and self.last_kind == "order":
            self.cart_seq = self.absorb_n
        remarks.append(self.notes_for(call_name, obs, refused))
        if call_name == self.role("search"):
            remarks.append(self.search_note(call_args, obs))
            self.__dict__.get("named_pages", set()).discard(json.dumps(call_args, sort_keys=True, default=str))
            try:
                remarks.append(self.merge_joined(obs))
            except Exception as err:
                self.log.append({"kind": "merge_joined_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if (last and not self.__dict__.get("early_audit") and call_name not in (basket_tool, msg_tool) and call_name not in self.finishers()
                and FIRST_ORDER_BY < self.moves_left() <= self.step_cap // 3 + 2 and not self.trolley.units
                and not any(key.rpartition("|")[2] == str(self.needs.version) for key in self.passed)):
            self.early_audit = True
            ahead = self.review_decline(early=True)
            if ahead:
                remarks.append("Nothing verified fits yet. " + ahead)
        qtext = " ".join(str(val) for val in call_args.values() if isinstance(val, str)) if call_name == msg_tool else None
        if qtext and self.__dict__.get("stating"):
            qtext = "(the basket stated to the shopper before the order; it asks about no property) " + qtext
        caught = self.shopper_lines(call_name, obs, refused)
        if caught:
            remarks.append(self.needs.hear(caught, qtext))
            self.ref_take()
        text_body = as_text(clipped(shown, SEARCH_TEXT if call_name == self.role("search") else RECORD_TEXT))
        if last and resp.get("finished"):
            text_body += " [the episode is finished]"
        summary = " ".join(qty_n for qty_n in remarks if qty_n)
        return text_body + ("\n[note] " + summary if summary else "")

    def tool_calls(self, msg: dict[str, Any], seq_no: int) -> tuple[list[dict[str, Any]], int]:
        """Normalises the tool calls of a model message into proposals with an id (call_<n> when missing), type
        'function', name and arguments as a JSON string, skipping entries that are not dicts. Returns the proposals and
        the updated call counter.
        """
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
        """Runs one shopping episode and returns its log. Reads the request, logs the plan and role map, and
        opens the chat with the rules (no basket-statement wording when states_basket is false) and an opener
        with the request, the opening observation, the needs overview and the status line. For up to
        TURN_LIMIT turns it stops when the wire is done or broken, steps or time run out, or a due statement
        no longer fits. Each turn the code's own steps go first: catch-up, message probe, due replies, named
        opens, sweep, fit walk, value pages, planned questions, references, tie checks and the price walk.
        Else the actor model is asked for tool calls (models rotate after STUCK_LIMIT idle turns); missing
        replies, refusals and call-less replies (up to IDLE_LIMIT) end the loop or get a reminder.
        Binding calls wait for planned questions; the rest are dispatched and posted.
        At the end finish_now runs if the episode is open; steps, cost and seconds are logged.
        """
        if self.wire.done:
            return self.log
        self.read_request()
        self.log.append({"kind": "plan", "plan": self.needs.reading, "roles": self.role_map})
        opener_obs = self.pview.get("opening")
        opener = "SHOPPER REQUEST:\n" + self.request
        if opener_obs is not None:
            opener += "\n\nOPENING OBSERVATION:\n" + as_text(opener_obs)
            try:
                self.archive.add(json.loads(json.dumps(opener_obs, default=str)))      # (K1d: a copy, so relabelling leaves the opening as it came)
            except (TypeError, ValueError):
                self.archive.add(opener_obs)
        overview = self.needs.overview(self.step_cap, self.msg_cap)
        if overview:
            opener += "\n\n" + overview
        opener += "\n\n" + self.status_line()
        rules_text = ACT_RULES if self.states_basket() else ACT_RULES.replace(NO_STATEMENT[0], NO_STATEMENT[1])
        self.chat = [{"role": "system", "content": rules_text}, {"role": "user", "content": opener}]
        idle_n = fails = seq_no = 0
        turn_ok, missed = monotonic(), 0
        for _ in range(TURN_LIMIT):
            if self.wire.done or self.wire.broke or self.moves_left() <= 0:
                break
            if self.wire.spare() < LAST_CALL:
                break
            if self.states_basket() and self.moves_left() <= 2 and self.orders_tried == 0 and self.statement_due():
                break
            self.catch_up()
            if self.msg_probe():
                continue
            if self.needs.reply_due:
                text_body, self.needs.reply_due = self.needs.reply_due, ""
                if self.reply_back(text_body):
                    continue
            if self.open_named():
                continue
            if (not self.asking or self.__dict__.get("empty_plans")) and self.sweep_step():
                continue
            self.name_lift()
            if self.fit_walk():
                continue
            if self.value_pages():
                continue
            self.ask_again()
            if self.asking and self.pose_planned():
                continue
            if self.ref_step():
                continue
            if self.tie_step():
                continue
            if self.walk_step():
                continue
            model_list = ACTOR if self.stuck < STUCK_LIMIT else ACTOR[1:] + ACTOR[:1]
            msg = self.wire.query_model(self.chat, model_list, 2500, self.toolset)
            if msg is None:
                missed += 1
                if not self.wire.usable(model_list, True) or (monotonic() - turn_ok > DOWN_AFTER and (
                        missed >= 2 or (self.wire.down() and len(self.wire.call_models) >= 2))):
                    break
                if self.wire.call_fail in ("empty", "refused"):
                    fails += 1
                    if fails >= 2:
                        break
                continue
            turn_ok, missed = monotonic(), 0
            proposals, seq_no = self.tool_calls(msg, seq_no)
            content = msg.get("content") or ""
            if not isinstance(content, str):
                content = as_text(content)
            binding_moves = {self.role("cart"), self.role("order"), self.role("decline")} - {None}
            if self.asking and any(c["function"]["name"] in binding_moves for c in proposals) and self.pose_planned(urge=True):
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
        self.log.append({"kind": "end", "finished": self.wire.done, "steps": self.wire.tick,
                         "cost": round(self.wire.spend, 5), "seconds": round(monotonic() - self.wire.t0, 1)})
        return self.log


def agent_main(pdata: dict[str, Any]) -> list[dict[str, Any]]:
    """Entry point: builds an Errand from the episode payload, runs it and returns the episode log. On a crash it logs
    the error, still tries to close the episode with finish_now (logging a second crash if that fails) and returns
    the log.
    """
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
