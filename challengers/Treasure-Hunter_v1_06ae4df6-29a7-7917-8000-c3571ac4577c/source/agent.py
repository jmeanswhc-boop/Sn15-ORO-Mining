from __future__ import annotations

import json
import re
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------------------------------------------------------- models
ACTOR = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-haiku-5.5"]
READER = ["anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"]        # reads the request and the tools once
CHECKER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]  # item checks and the rank check
AUDITOR = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]  # the decline audit
LISTENER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]  # shopper lines
PLANNER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]   # the question plan (the hidden need hangs on it)

# ---------------------------------------------------------------------------------------------------------------- clock
DEADLINE = 265.0          # every call ends before this many seconds
CLOSING = 70.0            # with less time than this left, the errand heads for its finish
LAST_CALL = 28.0          # with less than this left, the code sends the finish itself
CLOCK_STEP = 30           # (v605e) the status line's time left, once shown, in steps of this many seconds
CH_SYNONYM_MIN = 4        # (v605e F3) read listings of a category carrying a field before a synonym of a value it never writes may meet it
VIEW_CHARS = 16000        # longest observation text shown to the actor
SEARCH_TEXT = 240         # longest text of one field on a search page shown to the actor
RECORD_TEXT = 700         # longest text of one field in any other observation shown to the actor
SOFT_SPEND = 0.03         # model spend (USD) after which a role's cheaper fallback model is used
SPEND_CAP = 0.045         # model spend after which the errand heads for its finish (a run's key pays for every task)
TURN_LIMIT = 80
PACE_GAP = 0.15           # least gap in seconds between two HTTP requests of the errand, model and world calls alike (TA)
SLOW_GAP = 1.0            # the gap for SLOW_FOR seconds after a request was refused for load (429 or 503): one bucket per sandbox
SLOW_FOR = 30.0
DOWN_AFTER = 45.0         # the models count as gone after this many seconds without a model reply (and two failed tries)
MODEL_TRIES = 2           # attempts per model of one call after a load or transport failure, with a growing pause between
WORLD_TRIES = 5           # attempts of one world call after a load or transport failure
CODE_FINISHES = 4         # finishing calls the code sends at most (a refused order is repaired and sent again)
CODE_PAGES = 3            # further result pages the code reads of each product's best search
CODE_READS = 8            # rounds of record reads of the code's picks at most (one record per product a round)
JOINERS = r"\s+and\s+|\s*&\s*|\s*/\s*|\s+\+\s+"   # (CA) how a store joins two kinds in one category name
PUT_WORDS = {"add", "put", "hold", "set", "insert", "place", "keep", "include", "in", "plus", "increase", "append"}
DROP_WORDS = {"remove", "drop", "release", "delete", "take", "out", "clear", "discard", "minus", "decrease", "subtract", "off", "unhold"}
FAULT_KEYS = r"error|fault|refus|reject|denied|invalid|fail"
MSG_GUESS = 5             # (R9b) messages assumed allowed when neither the request nor a status tool tells
FIRST_ORDER_BY = 7        # with this many steps left after a refusal, only cart, view and finishing tools remain
FIRST_ORDER_DUE = 10      # at most this many steps kept for the first order and one repair (a quarter of the steps, at least FIRST_ORDER_BY)
IDLE_LIMIT = 3            # replies without a tool call before the code finishes
STUCK_LIMIT = 4           # replies in a row where nothing was sent before the backup model takes a turn
SWEEP_READS = 8           # at most this many shelf slices the code reads itself when a change leaves the whole shelf acceptable
WALK_PAGES = 6            # under a cheapest rule in an id-ordered store: at most this many further pages read per product
WALK_TOTAL = 8            # and at most this many in all
VALUE_PAGES = 4           # (CG2) an id-ordered all-words search led by a required value: at most this many further pages read at once per product
VALUE_TOTAL = 8           # and at most this many in all
VALUE_TOTAL_PER = 5       # (EB3) several products: at least this many value pages in all per product
LABEL_TRIES = 4           # (EB1) composed value + category-label searches of one product at most
LABEL_OPENS = 3           # (EB7) no cheapest rule: rows of a one-page all-values composed search opened at most
CEIL_CALLS = 4            # (CG3) price-filter listing calls of one product's ceiling walk at most (lowering the limit and reading pages together)
CEIL_OPENS = 3            # and at most this many of its rows opened (a pick and two backups)
LINK_WORDS = {"and", "or", "the", "for", "with", "of", "a", "an", "in", "on", "to", "by", "from", "at", "as", "is", "are", "be",
              "this", "that", "it", "its", "your", "our", "you", "all", "any", "not", "no"}
ASK_AFTER_SETS = 2        # listing detail records read before the first planned question
ASK_GRACE_STEPS = 3       # while fewer records were read and over 20 steps remain, this many more steps for them
ASK_AFTER_STEPS = 5       # or this many steps used

# ---------------------------------------------------------------------------------------------------------------- prompts
ACT_RULES = """You are a careful shopping assistant working for one shopper in a simulated store; the supplied tools are your only way to act. The errand scores 1 or 0, and it scores 1 only when the finish is right in every respect: the ordered item meets every requirement the shopper wrote, every hidden need the shopper reveals when asked, and every change the shopper announces along the way; the quantity and the budget are right; the cheapest (or best) item is chosen when the request asks for that; and every rule of procedure in the request is obeyed. When nothing in the store meets everything, the right finish is a decline, done exactly as the request says.

HOW THE STORE WORKS
- Each tool call costs one step, failed and invalid calls included, and steps are limited. Thinking costs nothing.
- One reply may carry several tool calls when none needs another's result (for example opening several listings' details); they run in order. An order or a decline always goes alone.
- A details tool that takes a LIST of listing ids shows all their full records for ONE step (up to its item limit): when you open two or more listings, use it.
- Tool names, argument names and result shapes are different in every store. Read each tool's description and JSON schema and use exactly the names and allowed values given. Learn from the first results which tool shows which facts (price, stock, detailed properties).
- Early on, learn which category labels the store puts on listings of the requested kind. When neighbouring labels exist, the right one is the label that matches the request's wording as a whole; search and buy inside it.
- Searches differ from store to store; learn this one's from its first results. Rows may come best match first or in a fixed id order; a keyword search may need every word (adding a word shrinks the total) or take any word (adding a word grows the total, and listings matching more of the words rank higher); only a few rows or pages may be visible; a filter may work in one mode only (a keyword mode may ignore the category or price filter, a filter mode may ignore the words). Compare a reported total with the rows you can see: when it is larger, the fitting or cheapest listing may sit on a later page or beyond the visible ones. When every word is required, narrow with more exact words (or the category and price filters with a lower price limit) until the total fits on the visible pages. Keep queries short (one to three words) with the most distinctive words FIRST: some stores read only the first word or two of a query and ignore the rest, so a long query may search only its opening words (compare a longer query's total and rows with a shorter one that starts the same way; the notes report what they show). When every word is required, a required value with the item kind ("blue chair", the value first) lists exactly the listings that carry it. When any word matches, common words (the item kind, a unit) match most of the shelf and bury the rest: a required value's own distinctive word or figure alone ("herringbone", "30m", "0.4") lists just the listings that carry it, so search it alone and in the listings' other spellings ("30 m", "3 atm"), read every page of such a search when its total fits on them, and put the item kind with the values together only to rank the closest matches first. A keyword search may also match category names and property values. Notes after each search say what the totals so far show about this store's search.

THE SHOPPER AND HIDDEN NEEDS
- The shopper holds needs they did not write down and reveals one only when a message asks about that property. The FIRST thing a message asks is answered most reliably, so a message asks about one property, or two at most with the likelier first: a short, plain, friendly question that names the property (one or two example values are fine). No instructions, no lists, labels, positions or rules.
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

HOW TO JUDGE A LISTING (this decides the score)
- The grader reads each requirement from the listing's field for that property, by the value's own words as this store writes them in that field. A field whose name says the same property counts whatever its spelling: separators, a prefix, a shorter or longer name, or a usual alternative name ("fit" for a pants fit type, "neckline" for a collar type, "dial size" for a watch dial size, "color family" for a colour). When a field carrying the property's own name exists, it decides, and so does a field of that name with one more leading word (pants length for length, clothing material for material): when either one writes the value, the requirement is met. Two fields whose names differ only by separators (size class, size_class) are one field: the requirement is met when either one writes the value. Otherwise such an alternative field decides; when no field names the property at all, the options list or the title may show the value. The description is never evidence. (One exception: a model number, a part series, or the models and items a listing works with, fits or is compatible with may be read from its description, overview or notes, where stores write them.)
- Values match by their words, as this store writes them: case, separators, singular and plural and spelling variants are the same; a more specific value that contains the required one meets it ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060); one element of a list is enough. A synonym, an abbreviation, a related word or a shade ("tech wood" for technical wood, "win11" for windows 11, "embroidered" for embroidery, "solid" for plain, ivory or cream for off-white) meets a value only when no listing of the category writes the value itself in that field: when other listings write "matte" as their finish, a "satin" finish does not meet matte. A value joined with another by a plus ("wool + nylon") does not meet one of its parts when the field also carries the other part alone ("nylon"). FIELD EVIDENCE lines, when given, are computed from this store's records and say which case holds. A different value of the property does not (quartz is not mechanical, rosewood is not maple), nor does a word that only looks alike. A value that claims to cover every value of the property ("suitable for all occasions", "all seasons", "universal") names none of them and does not meet a required value; a list that also names the required value still meets it. A field that names another value of the property does not meet the required value just because the two often go together or one tends to imply the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); only a shade usually sold as the required colour, or a more specific value that contains it, counts.
- A field of a different property is not the requirement's property (a lining material is not the material, a length is not a style, a strap length is not a strap type): it never stands in for a property whose own field the category's listings carry.
- Only what the shopper stated or revealed counts. Never rule out a listing for something the shopper did not ask for: an age group, gender, size, occasion, condition, quality or use you guess from the story or the description. The story is context; if it seems to matter, ask the shopper about that property instead of assuming. Your own questions are not requirements: only the shopper's answers are.
- Numbers are compared as amounts after converting units (4.5 cm is 45 mm); limits include the edge; a range counts only when all of it meets the limit. When no field names the property and the title is the evidence, an exact required value is met when the title shows its words and figures in the same order, even with more detail after them (9' x 12' 4" shows 9' x 12'; 9' x 13' does not).
- Figures in a title ("55 x 94 Inch", "117 x 228 cm") do not say which dimension each one is: never rule a listing out, or skip opening it, for the figures in its title; read the fields of its details.
- Respect exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
- The requested kind of product is met by the listing's category alone: a listing on the requested category IS that kind for the grader even when its title or description names another object (a container, a case, a refill, a set, an accessory), so never skip, reject or demote it for what it seems to be; judge it only on the requirements, price and stock. No property has to repeat the category name. When the store labels categories, the item must sit in the one whose label matches the request's wording as a whole (a request for "A or B" matches a category "A & B", not a narrower neighbour "B"). It must be in stock when the request wants things on hand, and price times quantity must be within the budget (equal is fine). A listing's price is in the budget's unit unless its record names another (45 against a budget of 1000 US cents is 45 cents); never rescale either.
- A decline is rarely the right finish. When no listing meets the strict wording but one meets each requirement under these rules (a more specific value, another name of the same field, or a synonym or shade where no listing of the category writes the value itself in that field), that listing is the fit; decline only when the search really found nothing that meets every requirement under these rules.

CHEAPEST OR BEST
- For the cheapest, the order must be the lowest-priced item among ALL items meeting every requirement; one fitting item is not enough. When the keyword search needs every word and matches property values, the value-and-kind search lists the candidates that carry the value: read all its pages and open them by price; a category listing with a price limit shows the whole category's prices but not its values: use it to see which cheaper rows exist (lower the limit until the total fits), then check those rows' properties. Search each required value with the item kind as a short query, the value first ("blue chair", "30m watch"), and read every page of it when its total fits on the readable pages (when it does not, add a second required value or use a narrower kind word); do the same with the kind's other names ("stool", "bench") and with the value alone; when a value cannot be searched (a figure the store drops), also search the product line of each fit or near-fit you find, its maker with one distinctive word of its title as two words, every page: a line's other sizes and colours are separate listings that share its property values and may cost less; sort the visible candidates by price yourself and open their details from the cheapest up; the first whose property fields meet everything, with every cheaper one opened and rejected, is the answer. Keep the next fitting ones as backups.
- For the best by some measure (most, largest, longest, highest...), compare ALL fitting items on that measure the same way and take the extreme. An item stating no value for the measure cannot win.

FINISHING
- Keep a ranked list of verified fits with at least one backup: the first pick may sell out at checkout.
- Before every order attempt: the cart holds exactly one line per product asked for (one line when one product is asked), each the chosen item at its required quantity, and nothing else; after your last cart change you have re-read the chosen item's stock and price, and it is in stock and within budget.
- Decline only after a thorough search finds nothing that meets every requirement, following the request's decline rule exactly. When the request says not to offer substitutes, the decline remark names, describes or points to no listing: no ids, titles or prices.
- Budget your steps: the first order attempt goes out while at least seven steps remain, so a refusal can still be handled. With 7 steps left only cart, stock-check and finishing tools remain. Decide once; do not swap the cart back and forth.
- Every reply carries at least one tool call. Keep the text short: the requirements now, the verified fits with price and stock, the next move."""

READ_REQUEST = """Below are a shopper's request and a store's tool list. Answer with ONE compact JSON object (no indentation), nothing else, holding these keys:

"product": the kind of product asked for, in the request's words (for several products, all of them joined with "and").
"items": one entry per DIFFERENT product the request asks to buy, in its order: {"product": its kind in the request's words, "quantity": units of one listing of it (as for "quantity" below), "requirements": its firm requirements (as for "requirements" below)}. One entry when one product is asked.
"quantity": units of one listing to order (an integer; 1 when the request does not ask for several). Something sold as one item, such as a pair of shoes or a set, is quantity 1. Only a plain request to buy several units sets it ("I need three of them", "order 2 packs"); how many people will use or share it, or how many old ones there were, is story and leaves it at 1.
"requirements": a list of strings, one per firm requirement on the product (for several products: every product's, each written "property (product): constraint"), each "property: constraint" with the shopper's exact value, limit, exclusion or closed list. Leave out price, stock, quantity, budgets and mere wishes, and do not restate the product kind or category: words that belong to the category's name (a season, an occasion or an audience in it) are covered by the category, not a separate requirement. A requirement is a property of the product with the value the request says it must have ("the pattern has to be herringbone", "with a mechanical movement", "at least 4 drawers", "the occasion has to be party"). The story around it is never a requirement: whom it is for, an event it will be worn or used at ("to wear at his wedding", "for the staff party"), where, when or how it will be used, how many people share it, or that it replaces an old one ("a new watch" asks for a replacement, not for a condition). An occasion, use or audience is a requirement only when the request makes it the item's own property ("the occasion has to be business", "a blazer made for business"). A value the request gives with an abbreviation or another name for it ("polyvinyl chloride (PVC)", "a USB-C, also called type-c, port") is a closed list of both: "material: polyvinyl chloride or pvc". A requirement's value is words or figures the request itself writes as the item's value; a quality, reason or wish the story gives (natural and breathable, slim enough for a case, works with what I already have, covers the whole drop) is a hint, never a requirement.
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
"references": requirements whose value the request gives only by pointing at ANOTHER listing or page ("the same colour as the jumpsuit sold by X", "within 5 of the figure on the page for Y", "a colour other than the one on Z's page", "the counterpart of what the page of X's kettle shows", "a value that pairs well with the one on X's lamp page", "no lower than the figure on X's page"), each "property (product): as shown on <how the request names that listing>" ("property (product): different from the one shown on <...>" when the value must differ from that listing's, "property (product): the counterpart of the one shown on <...>" when it must be the value the store pairs with that listing's, "property (product): at least / at most the one shown on <...>" for a bound); also put each in "requirements" in that form. Empty when none.
"reference_targets": for each entry of "references", the listing it points at: {"property": its property, "product": its product, "seller": the shop or brand name the request gives for that listing, exactly as written, "kind": the kind of product that listing is, "details": [each other value the request names for that listing, short and in the request's words ("blue", "4gb", "net weight 167.262186 g")], "relation": "same" (the same as, matching), "differ" (the value must NOT be that listing's: set apart from, different from, other than, not the one, anything but), "partner" (the value must be the one the store PAIRS with that listing's value, never that value itself: the counterpart of it, compatible with it, going well together with it, its match or partner), "within", "at most" (no more than, no higher than) or "at least" (no less than, no lower than) (how the value must relate to that listing's), "tolerance": the number for "within", else null, "unit": its unit, else ""}. Words that only say where to look (that a page shows it, or that the page may send the reader on to another listing) are no details. Empty when none.
"bundles": when the request says some products must fit, work with or be compatible with ANOTHER product it asks for (a main item and its accessories or parts), one entry per such accessory: {"part": the accessory product as written in "items", "main": the main product as written in "items", "model": the model number, series or name the request gives for the main item, else ""}; [] when none.
"decline_no_names": true when the request's decline rule forbids offering, naming or suggesting substitutes or alternatives; false when it asks you to name something; null when it gives no rule.
"roles": an object naming the tool for each role, with the exact tool names from the list, or null when no tool fits: "search", "details", "view", "message", "message_status", "cart", "cart_view", "order", "decline". "message" sends a message to the shopper; "message_status" reports how many messages may be sent, were sent or are left; "cart" adds to or takes from the cart; "order" buys or checks out; "decline" ends the errand without buying.
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
"references": for each need in "needs" whose value the shopper gives only by pointing at another listing or page, the listing it points at: {"property": its property, "seller": the shop or brand name the line gives for that listing, exactly as written, "kind": the kind of product that listing is, "details": [each other value the line names for that listing, short and in the line's words], "relation": "same", "differ" (the value must NOT be that listing's: set apart from, different from, other than, not the one, anything but), "partner" (the value the store PAIRS with that listing's value, never that value itself: the counterpart of it, compatible with it, going well together with it), "within", "at most" (no more than, no higher than) or "at least" (no less than, no lower than), "tolerance": the number for "within", else null, "unit": its unit, else ""}; words that only say where to look are no details; [] when none.
One line can carry several changes; report each. A need on another property, even a related one (a seat height beside an overall height, a strap width beside a width), is a new requirement that replaces nothing: an earlier requirement is dropped or replaced only when the line says so about that same property. A replaced requirement is both dropped (the old) and a new need (the new). When the shopper says which of two wishes matters more, the preferred one is a need. A stated budget is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and kindly: folksy, figurative or roundabout words that name a value mean the shopper firmly wants that value; a value is excluded only when the shopper clearly says to avoid it. Words that only stress how firmly the shopper holds a value (only, nothing but, nothing short of, strictly, stays, sticking with, locked to, holding out for) are not part of it: write just the value words ("material: jute", "color: white"), never "<value> only"; only a shopper who also rules out mixes or other values in so many words adds an exclusion. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit. A new budget replaces the old one whether higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference or anything works, put the property asked about in no_preference (when the message asked about two properties, only the FIRST one). A reply that only says the shopper did not follow or understand the question ("that was rather dense", "you've lost me", "could you put it another way") is this shopper's way of having no preference when the question named the product the property belongs to: then put the property asked about in no_preference too; when several products are asked for and the question did not say which one, the property is still open (put nothing in no_preference). In a request for several products, write each need "property (product): constraint", the product taken from the line (a line about "toe style of the women's boots" gives "toe style (women's boots): ..."). The assistant's own question is never a requirement: only what the shopper says is; values named as examples in the question do not become needs. A "don't mind" answer frees only the property the assistant's message asked about; it never drops or changes a need the shopper stated earlier, and when the message asked nothing (it only told something), such an answer changes nothing. A need given by pointing at another listing keeps the line's own pointing words as its value ("finish: unlike the one shown on Y's kettle page"); the values the line names for that other listing describe that listing, not the product: they are no needs, drops or freed properties."""

READ_RULES = """- Read each requirement from the listing's field for that property, by the value's own words as this store writes them in that field: a field whose name says the same property counts whatever its spelling (separators, a prefix, a shorter or longer name, or a usual alternative name such as "fit" for a pants fit type, "neckline" for a collar type, "color family" for a colour). When a field carrying the property's own name exists, it decides, and so does a field of that name with one more leading word (pants length for length, clothing material for material): when either one writes the value, the requirement is met. Two fields whose names differ only by separators (size class, size_class) are one field: the requirement is met when either one writes the value. Otherwise such an alternative field decides; only when no field names the property may the options list or the title show the value. The description is never evidence. (One exception: a model number, a part series, or the models and items a listing works with, fits or is compatible with may be read from its description, overview or notes, where stores write them.)
- Values match by their words, as this store writes them: case, separators, singular and plural and spelling variants are the same; a more specific value that contains the required one meets it ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060); one element of a list is enough. A synonym, an abbreviation, a related word or a shade ("tech wood" for technical wood, "win11" for windows 11, "embroidered" for embroidery, "solid" for plain, ivory or cream for off-white) meets a value only when no listing of the category writes the value itself in that field: when other listings write "matte" as their finish, a "satin" finish does not meet matte. A value joined with another by a plus ("wool + nylon") does not meet one of its parts when the field also carries the other part alone ("nylon"). FIELD EVIDENCE lines, when given, are computed from this store's records and say which case holds. A different value of the property does not (quartz is not mechanical, rosewood is not maple), nor a word that only looks alike. A value that claims to cover every value of the property ("suitable for all occasions", "all seasons", "universal") names none of them and does not meet a required value; a list that also names the required value still meets it. A field that names another value of the property does not meet the required value just because the two often go together or one tends to imply the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); only a shade usually sold as the required colour, or a more specific value that contains it, counts.
- A field of a different property does not count (a lining material is not the material, a length is not a style, a strap length is not a strap type): it never stands in for a property whose own field the category's listings carry.
- Numbers are compared as amounts after converting units (4.5 cm is 45 mm); limits include the edge; a range counts only when all of it meets the limit. When no field names the property and the title is the evidence, an exact required value is met when the title shows its words and figures in the same order, even with more detail after them (9' x 12' 4" shows 9' x 12'; 9' x 13' does not).
- A listing's price is in the same unit as the request's budget unless its record names another currency or unit (a price of 45 against a budget of 1000 US cents is 45 cents, well within it); never rescale a price or the budget.
- Only the requirements listed count: never fail a listing for an age group, gender, size, occasion, condition, season, quality or use the requirements do not name.
- The requested kind is met by the category label alone: a listing on the requested category is of that kind even when its title names another object (a container, a case, a refill, a set, an accessory); judge it only on the requirements, price and stock."""

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
1. then properties the listings' details below actually carry. Read the story first (STORY HINTS below, when given, name its details and the property each concerns: they come first): the hidden need is usually the property its concrete details concern without naming the value. What the user will touch or press, see up close or in photos, hear, measure or fit it against, read on the label, or how it was made points at that part or aspect of the item and its property: tags, labels or the fine print point to the material, fabric or ingredients; photos and looks to the pattern, colour or the visible part; a body part or a space it must fit to that part's shape or size; buying for someone else, or someone barely known, to whom it is made for (gender, age group); how it must feel to the hand, skin or foot first to the material or fabric; where, when or for what activity it is worn or used (a gym class, the office, outdoors) to its style or type, the property whose values name such uses; something done up or adjusted again and again to its closure or mechanism; what does the work inside it to its core component. Weather, a job, a schedule or a place that is not where the item is worn or used are background. Ask first about the property the story's details point at, by the name the listings use for it. Whom the item is made for is a candidate like any other: ask it when the story points at someone else, or when nothing else is hinted. Then prefer properties describing what the product is like (pattern, material, shape, type or style variant, closure, length, function) over logistics (warranty, condition, pack size, variant or size lists, years).
Never pick a property the request already settles (its requirements, the price, stock, quantity), never repeat one already asked, never spend two questions on near-duplicates (two kinds of style, two spellings of one property), and never ask about brand, maker, a product or model name, price, stock or delivery. Ask only about a property that appears as a key in the listings' details below, by the name the listings use for it; a property under STORY HINTS that no key below matches may be asked by its plain name, unless the hint is marked [no key] (enough of its product's listings were read without that field): then ask the key below whose values differ the way its detail describes, or leave it. Each step-1 question is one short, plain, friendly sentence (at most 15 words) asking which <property> they would like for the <item kind>, naming the property by the listings' own name for it (example values alone never reveal a need), optionally followed by one or two example values. No lead-in, no status report, no instructions or lists, and never a catch-all key such as options, variants, a description, a title or a name. Each question asks about ONE property of ONE product (the step-0 choice between two wishes aside): never two properties in one question and never properties of two products, since a reply reveals at most one need and a question naming two subjects may get no answer at all. In "second", give the next most promising property of the SAME product (not another name for the first), or leave it empty: the assistant may add it to its last message.
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
"goes_with": the values the target's text says go together with, pair with, match or count the same as that value: a sentence saying the target's value goes together with, pairs with or matches another value, or a store rule listing such pairs of values for the PROPERTY, names them; give the partner of the target's own value ([] when none).
A listing the text names only as what the target is not ("not the green one from X"), or as a version, size or colour sold elsewhere, is no go_to; nor is one it names for another property. A record that does not show the described details is not the target, whatever its text says."""

DECLINE_WORDS = r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without"
ORDER_WORDS = r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed"
SORRY = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
ID_LIKE = r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}"
CODE_LIKE = r"[A-Za-z0-9_\-]+"


# ================================================================================================================ helpers
def as_text(piece: Any, ceiling: int = VIEW_CHARS) -> str:
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


def junk_key(key: Any) -> bool:
    """A field or planned property that is not a product property (brand, shop, category, title, text, price, stock,
    ids): judged on its words, the planner's "(product)" part left out."""
    part = re.split(r"\s*\(", str(key))[0]
    words = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower()))
    return bool(re.search(r"\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|"
                          r"heading|listing|description|overview|price|cost|amount|availability|sku|url|aisle|shelf|section|"
                          r"maker|note|notes|about|blurb|orderable|ref)\b", words)
                or re.fullmatch(r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)", words))


def prop_key(name: Any) -> str:
    """A property name reduced for comparing: its "(product)" part left out, camelCase split, letters only, each word
    without a plural s, joined ("toeStyle" and "toe styles" both give "toestyle")."""
    part = re.split(r"\s*\(", str(name))[0]
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower())
    return "".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words)


def edits_within(a: str, b: str, most: int) -> bool:
    """The edit distance of two short strings is at most `most`."""
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
    """Two property names for the same property: equal once reduced (prop_key); a spelling variant (the same first three
    letters, at most 1 edit apart, 2 for names of 6+ letters: flavor and flavour); or one a 5+ letter start of the other
    (sleeve and sleeve length)."""
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
    """The words of a property name as prop_key reduces them (its "(product)" part left out, camelCase split, plural s off)."""
    part = re.split(r"\s*\(", str(name))[0]
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]


def spelled_alike(lo: str, hi: str) -> bool:
    """hi is lo with one or two letters added, each a vowel or a doubled letter (color, colour; jewelry, jewellery), or lo
    with two neighbouring letters swapped (center, centre)."""
    gap = len(hi) - len(lo)
    if gap == 0:
        diff = [i for i in range(len(lo)) if lo[i] != hi[i]]
        return len(diff) == 2 and diff[1] == diff[0] + 1 and lo[diff[0]] == hi[diff[1]] and lo[diff[1]] == hi[diff[0]]
    if gap not in (1, 2):
        return False

    def soft(word: str, i: int) -> bool:
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
    """Two property names for the same property, strictly (for choosing questions): equal once reduced (prop_key); a
    spelling variant of 4+ letters with the same first three letters (spelled_alike); or the same words plus one trailing
    token of 1-3 letters, a unit or code tag (height, height cm). A longer name that starts with the other (display,
    display size; protein, protein source) or a changed letter (shade, shape; collar, colour) is another property."""
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
           r"or lower|minimum|maximum|between|within|under|over|above|below|than)\b"      # (CH) a limit's words
CJ_COMPARE = (r"\b(?:falls?\s+short\s+of|short\s+of|nothing\s+(?:less|other)\s+than|nothing\s+short\s+of|nothing\s+but|"
              r"no\s+(?:less|worse)\s+than|at\s+least|(?:less|more|better|worse)\s+than|exactly|only|just)\b")   # (CJ2)
CH_SKIP = {"and", "the", "a", "an", "of"}
LITERAL_VALUES: set[str] = set()   # (CJ2) word values the store writes that a shopper's line gave after a comparison; per errand
# (v605d: module state, cleared in Errand.__init__ - one errand runs per process at a time; agent_main is synchronous)


def ch_words(text: Any) -> list[str]:
    """CH: the words and figures of a value in order: lower-case letter and digit runs (a hyphen, slash or plus separates
    them: 'mid-heel' gives mid, heel), a few link words left out; negations stay ('not available', 'no warranty')."""
    return [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text or "").lower()) if w not in CH_SKIP]


def ch_word_in(word: str, pool: list[str]) -> bool:
    """CH: a value's word is among a field value's words: the same, a singular or plural spelling of it (same_kind), or,
    for words of 6+ letters with the same first two, one letter apart (a misspelling: polyster, polyester)."""
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
    """CH: the alternatives of a required value (split at 'or'; a comma list is one alternative, all of it wanted), each as
    its words; the '(stated)' and '(revealed ...)' marks, quotes and emphasis words (only, just) left out."""
    plain = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", str(value or "").lower()).replace('"', " ")
    plain = re.sub(r"\b(?:only|just|exactly|strictly)\b", " ", plain)
    return [ws for ws in (ch_words(part) for part in re.split(r"\bor\b", plain)) if ws]


def ch_writes(value: Any, text: Any) -> bool:
    """CH: a field value writes a required value's own words (ch_match)."""
    return ch_match(ch_alts(value), ch_words(text))


def ch_match(alts: list[list[str]], pool: list[str]) -> bool:
    """CH: a field value's words (pool) write a required value's alternatives (ch_alts): every word and figure of one of
    them is among its words (containment counts: 'forest green' writes green), or the alternative's words run together
    equal a run of its words run together (a spacing variant: '18 k' and '18k', 'wi-fi' and 'wifi')."""
    if not pool:
        return False
    for ws in alts:
        if all(ch_word_in(w, pool) for w in ws):
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
    """CH: a field value shares anything with a required value: a word (ch_word_in), a figure, or a word's first three
    letters starting one of its words or the reverse (an abbreviation: 'win10' and windows)."""
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
    """CH: the elements of a field value: the parts of a list (split at commas, semicolons and bars), else the value."""
    low = str(text or "").strip().lower()
    return [p.strip() for p in re.split(r"[,;|]", low) if p.strip()] or ([low] if low else [])


def ch_sides(part: Any) -> list[str]:
    """CH: the sides of an element joined by '+' ('a + b'); [] when it is not joined (a 'with' may name a trim or a part,
    so it joins nothing here)."""
    sides = [s.strip() for s in re.split(r"\s*\+\s*", str(part)) if s.strip()]
    return sides if len(sides) >= 2 else []


def figure_written(fig: str, said: str) -> bool:
    """CI: the request writes this figure as a number (not inside a longer one) that is a decimal, or is followed by a
    symbol, the end or a unit of at most three letters ('6.5 m', '12"', '20 MP', '5 kg'), never by a longer word ('3
    kids')."""
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
    """A requirement's value that names what the item is. A yes/no value is one; a description is not (4+ content words in
    one of its alternatives or clauses: 'suits the newer kind of board'; a list of short alternatives is a value); an exclusion is not (no, not, without, nothing,
    never, neither/nor, except, avoid, anything but, other than) unless a value comes before it ('cotton, nothing in
    leather' is cotton) or it negates the property's own name ('machine washable: not machine washable' is a no). Bounds
    are no exclusions: 'no more than', a comparative before 'than' ('no thicker than') or in a value with a figure ('0.39
    inches, no thicker'). An exclusion or a description leaves the property's own value unknown."""
    low = str(value).lower().strip()
    if re.fullmatch(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*", low):
        return True
    if LITERAL_VALUES and " ".join(re.findall(r"[a-z0-9]+", re.sub(r"\((?:stated|revealed[^)]*)\)", " ", low))) in LITERAL_VALUES:
        return True      # (CJ2: a value the store writes, given by the shopper after a comparison: 'not available')

    def content(text: str) -> list[str]:
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
    """One of a requirement value's words (3+ letters, link and comparison words left out; a longer word by its stem) or
    figure runs (4:3, 1920 x 1080, 11) is among a field's sample values."""
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
    """The words of a product or category name that say what it is: 3+ letters, link words left out, plural s off."""
    return {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", str(text).lower()) if w not in LINK_WORDS}


def kind_tokens(text: Any) -> list[str]:
    """CA: the words of a product or category name that say what it is, as written and in order: 3+ letters, link
    words and a possessive 's left out ("women's flats" gives women, flats)."""
    return [w for w in re.findall(r"[a-z]{3,}", re.sub(r"'s\b", "", str(text).lower().replace("’", "'"))) if w not in LINK_WORDS]


def plural_forms(word: Any) -> set[str]:
    """CA: the spellings one kind word can take in the singular and the plural, itself among them: +s, +es, y/ies,
    f or fe/ves, and the same back ('cookie' and 'cookies', 'lens' and 'lenses', 'battery' and 'batteries')."""
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
    """CA: two kind words name one kind: equal, or one is a singular or plural spelling of the other (a symmetric test,
    so no one-way stem has to agree)."""
    return a == b or str(b) in plural_forms(a) or str(a) in plural_forms(b)


def kind_in(word: Any, pool: Any) -> bool:
    """CA: a kind word has its kind among the words of `pool`."""
    return any(same_kind(word, other) for other in pool)


def kinds_within(words: Any, pool: Any) -> bool:
    """CA: every kind word of `words` has its kind among the words of `pool`."""
    return all(kind_in(w, pool) for w in words)


def kind_score(own: Any, words: Any, shared: Any) -> float:
    """CC: how well a name's kind words `own` are matched by `words`, singular and plural alike (same_kind), a word that
    the other products' names share counting half."""
    return sum(0.5 if kind_in(w, shared) else 1.0 for w in own if kind_in(w, words))


def name_pattern(name: Any) -> str:
    """CA: a pattern matching a product name's kind words in order in lower-case text, each in any of its singular or
    plural spellings and with a possessive ('battery chargers' matches 'battery charger' and 'batteries chargers');
    '' for a name without one."""
    pats = []
    for w in kind_tokens(name):
        forms = sorted(plural_forms(w), key=lambda x: (-len(x), x))
        pats.append(r"(?:" + "|".join(re.escape(x) for x in forms) + r")(?:'s\b|'|\b)")
    return r"[\s'-]+".join(pats)


VALUE_LINKS = r"at least|at most|no more than|no less than|more than|less than|up to|or more|or less|or larger|or smaller|" \
              r"or higher|or lower|minimum|maximum|between|within|under|over|above|below|exactly|only|either|neither|nor|than|plus"


def one_prop_hint(hint: str) -> str:
    """A hint naming alternative properties ("colour / style (chair): detail") keeps the first as its property and the
    others in its detail ("colour (chair): detail (or style)")."""
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
    """The word a hint's detail starts with when it stands alone there (a comma, a bracket, a link word or the end follows)
    and the request writes it directly before the product's own name ('a sexy jumpsuit or romper' for the detail 'sexy,
    for a dinner' of the product 'jumpsuit or romper'); '' otherwise."""
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
    """The value words of a requirement that the request never writes ([] when it writes each, or for a reference): words
    of 3+ letters (a plural or a word sharing all but its last 1-3 letters counts as written) and figure runs (4:3, 1920 x
    1080, never a lone figure: the request may spell it) in that order; comparison and link words do not count; a yes/no
    value is judged by its property's name."""
    req = str(req)
    head, _, value = req.partition(":")
    low = value.strip().lower()
    if not low or req in refs or re.search(r"as shown|same as|the page|listing|sold by|redirect|refer", low) or ref_text(low) \
            or any(low == str(r).partition(":")[2].strip().lower() for r in refs):
        return []      # (R9a: a value that points at another listing, or a listed reference whatever its product tag)
    if re.fullmatch(r"\W*(?:yes|no|true|false|required|included|needed)\W*", low):
        low = re.split(r"\s*\(", head)[0].lower()
    body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", low)
    said = str(request).lower()
    figs = re.findall(r"\d+(?:\.\d+)?", body)
    if figs and all(figure_written(f, said) for f in figs):
        return []        # (CI: every figure of the value is written in the request; a unit spelt out differently is no guess)
    said_words = set(re.findall(r"[a-z]+", said))
    said_stems = {s[:-1] if s.endswith("s") and len(s) > 3 else s for s in said_words}
    missing = []
    content = [w for w in re.findall(r"[a-z]{3,}", body) if w not in LINK_WORDS]
    for w in content:
        stem = w[:-2] if w.endswith("es") and len(w) > 4 else w[:-1] if w.endswith("s") and len(w) > 3 else w
        # (a plural or inflection, a 1-edit spelling variant (grey, gray), the start of a written compound (water,
        # waterproof), or a stem sharing all but its last 3 letters (zipper, zips) counts as written)
        if not any(s == w or (s.startswith(stem) and len(s) <= len(stem) + 3) or (stem.startswith(s) and len(s) >= max(4, len(stem) - 2))
                   or (len(w) >= 4 and s.startswith(w)) or (len(w) >= 4 and len(s) >= 4 and s[0] == w[0] and edits_within(s, w, 1))
                   for s in said_words) \
                and not any(len(s) >= 3 and stem.startswith(s) and len(s) >= len(stem) - 3 for s in said_stems):
            missing.append(w)
    if len(content) <= 2 and any(len(w) >= 4 and w not in missing for w in content):
        return []      # (a short value whose core word the request writes is written: 'genuine leather' for 'real leather')
    for run in re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body):
        figs = re.findall(r"\d+(?:\.\d+)?", run)
        if len(figs) >= 2 and not re.search(r"(?<![\d.])" + r"\D{0,4}".join(re.escape(f) for f in figs) + r"(?!\d)", said):
            missing.append(run)
    return missing


REF_WORDS = r"\bas shown\b|\bshown on\b|\bsame as\b|\bpage\b|\bsold by\b|\bredirect|\brefer(?:s|red|ring)?\b|\blisting points\b|\bpoints you\b"
REF_STEPS = 0.4           # the code's own reference reads and searches use at most this share of the steps (and at least 8)
REF_CALLS = 12            # at most this many reference hop calls per errand
REF_HOPS = 8              # at most this many listings followed for one reference
# (CE2) relation words that make a reference an exclusion: the value must differ from that listing's ('set apart from'
# is kept beside 'apart from': it opens the pointing phrase at its verb, so the need reads 'set apart from ...', not
# 'apart from ...', which reads as 'besides')
DIFFER_WORDS = (r"\bset apart from\b|\bapart from\b|\bdistinct from\b|\bdifferent (?:from|to|than)\b|\bdiffer(?:s|ent|ing)?\s+from\b"
                r"|\bother than\b|\bunlike\b|\bnot the same as\b|\banything but\b|\bnot the one\b|\bnever the one\b")
# (R9a) relation words that make a reference a pairing: the value must be the one the store pairs with that listing's own
# value (its counterpart), never that value itself
MATE_WORDS = (r"\bcounterparts?\b|\bcompatible with\b|\bgo(?:es|ing)?\s+(?:well\s+)?together\b|\bpair(?:s|ed|ing)?\s+(?:well\s+)?with\b"
              r"|\bpartners?\b|\bplay(?:s|ing)?\s+nicely\b|\bmatching partner\b|\bworks?\s+well\s+with\b")
# (R9a) bound words: the value must reach (at least) or stay within (at most) that listing's figure
LEAST_WORDS = (r"\bat least\b|\bno (?:less|lower|fewer|smaller) than\b|\bnot\s+(?:any\s+)?(?:less|lower|fewer|smaller)\s+than\b"
               r"|\bat (?:the )?minimum\b|\bor (?:more|higher|larger|greater|above)\b")
MOST_WORDS = (r"\bat most\b|\bno (?:more|higher|greater|larger|bigger) than\b|\bnot\s+(?:any\s+)?(?:more|higher|greater|larger|bigger)\s+than\b"
              r"|\bup to\b|\bat (?:the )?maximum\b|\bor (?:less|lower|smaller|fewer|below)\b")
# (CE2b) words that open the pointing part of a reference phrase (with REF_WORDS and DIFFER_WORDS)
POINT_WORDS = (r"\bwithin\b|\bat (?:most|least)\b|\bno (?:more|less|lower|higher) than\b|\bsame as\b|\bas shown\b|\bthe value\b|\bthe one\b"
               r"|\bcounterparts?\b|\bcompatible with\b|\bwhatever\b|\bnot\s+any\s+(?:less|more|lower|higher)\b|\bmatching\b")
REF_DIFFER_STEPS = 0.3    # (CE) with only differ references open, the code's reference steps stop at this share (an exclusion of one value)
REF_DIFFER_BLIND = 5      # (CE) ... and at this many while no listing of any open differ reference's chain was found yet
# (CE1) field names that name the listing's seller (generic shop words; the store's own seller fields are learned, seller_keys)
SELLER_KEYS = r"shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by"


def ref_text(value: Any) -> bool:
    """A requirement value given by pointing at another listing or page ('as shown on the listing of the lamp sold by X')."""
    return bool(re.search(REF_WORDS, str(value or "").lower()))


def ref_toks(text: Any) -> list[str]:
    """Comparable tokens of a text: lowercase words and figures, a figure joined to the short unit word right after it
    ('12 v', '12-v' and '12v' alike, '300 cd/m2' gives 300cd and m2), a plural s dropped, link words and lone letters left out."""
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
    """(figure, unit word) pairs a text writes: a figure and the word right after it ('12 v', '1.35v', '24 inches' gives
    24 and inch; '1066/1333mhz' gives each figure the unit); plural forms reduced as ref_toks does."""
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
    """A seller name reduced for comparing: lowercase letters and figures only, a possessive 's left out."""
    return re.sub(r"[^a-z0-9]", "", re.sub(r"'s\b", "", str(text or "").lower()))


def ref_relation(value: Any) -> tuple[str, float | None, str]:
    """(relation, tolerance, unit) a reference requirement's words set: 'differ' (CE2: a differ word such as 'different
    from', 'other than' or 'unlike' - read before any bound word, so a bound word followed by a differ phrase is a
    differ), 'within' with its figure and unit ('within 5 units above or below the one shown...'), 'at most', 'at least',
    else 'same'."""
    low = str(value or "").lower()
    if re.search(DIFFER_WORDS, low):
        return "differ", None, ""
    if re.search(MATE_WORDS, low):
        return "partner", None, ""       # (R9a: the counterpart the store pairs with that listing's value)
    hit = re.search(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?", low)
    if hit:
        return "within", float(hit.group(1)), (hit.group(2) or "").strip()
    if re.search(MOST_WORDS, low):
        return "at most", None, ""
    if re.search(LEAST_WORDS, low):
        return "at least", None, ""
    return "same", None, ""


MATE_VERB = r"(?:^|\s+)(?:goes|go|going)\s+(?:well\s+)?together\s+with\s+|(?:^|\s+)pairs?\s+(?:well\s+)?with\s+|\s+(?:is|are)\s+paired\s+with\s+"
SEND_ON = (r"\b(?:go(?:es)?\s+by|taken?(?:\s+(?:it|that|this|the\s+(?:information|value|detail)))?\s+from|(?:be\s+)?read\s+from|"
           r"refer(?:red)?\s+to|looks?\s+to|check|use|comes?\s+from|follow|see)\b")
MATE_LEAD = {"a", "an", "the", "that", "this", "these", "those", "its", "their", "our", "which", "it", "they", "one", "ones", "also"}


def mate_phrase(side: str, prop: Any, tail: bool) -> str:
    """R9a: the value words of one side of a pairing sentence: determiners and pronouns before it left out, the
    property's own words after it left out when other words remain; `tail` keeps the last four words (a left side)."""
    words = [w.strip(".,;:'") for w in re.findall(r"[a-z0-9][a-z0-9.&+/'-]*", str(side or "").lower())]
    words = [w for w in words if w]
    if tail:
        words = words[-4:]
    while words and words[0] in MATE_LEAD:
        words.pop(0)
    own = [w for w in re.findall(r"[a-z0-9]+", re.split(r"\s*\(", str(prop or ""))[0].lower()) if w not in LINK_WORDS]
    if own and len(words) > len(own) and words[-len(own):] == own:
        words = words[:-len(own)]
    if words and words[-1] in ("one", "ones", "value", "version"):
        words = words[:-1]
    return " ".join(words).strip(" .,;:")


def rule_mates(text: Any, value: Any, prop: Any, own: bool) -> list[str]:
    """R9a: the values a text pairs with `value` ('A goes together with B', 'pairs with', a store rule listing several
    pairs): B when one side writes the value (or, with `own`, when that side only points at the record's own value -
    'its', 'that', 'which' or the property's name alone), A when the other side does. At most three, the value itself never."""
    vt = set(ref_toks(value))
    if not vt:
        return []
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n", str(text or "")):
        if not own and not (re.search(r"\brule\b", sentence, re.I) or prop_named(prop, sentence)):
            continue      # (another record's pairing counts only as a store rule or when it names the property)
        body = sentence.split(":", 1)[1] if re.search(r"\brule\b[^:]{0,80}:", sentence, re.I) else sentence
        for clause in re.split(r",\s*(?:and\s+)?|;\s*|\s+and\s+(?=[^,]{0,40}?\b(?:goes|go|pairs?)\b)", body):
            hit = re.search(MATE_VERB, clause, re.I)
            if not hit:
                continue
            left, right = clause[:hit.start()], clause[hit.end():]
            left_toks = set(ref_toks(left))
            right_toks = set(ref_toks(right))
            lone = not (left_toks - set(ref_toks(prop))) or bool(re.match(r"\s*(?:its|it|that|this|which|they)\b", left.strip(), re.I))
            if vt <= left_toks or (own and lone and not vt <= right_toks):
                mate = mate_phrase(right, prop, False)
            elif vt <= right_toks:
                mate = mate_phrase(left, prop, True)
            else:
                continue
            if mate and set(ref_toks(mate)) and set(ref_toks(mate)) != vt and len(mate.split()) <= 5 and mate not in out:
                out.append(mate)
    return out[:3]


def ref_parse(text: Any, ok: Any = None) -> dict[str, Any] | None:
    """{seller, kind, details} read from a reference's own words ('the kettle sold by Abcdef that is steel, 2l and new';
    'the Ghijkl mugs & cups with matte finish, ...'): the seller is the first capitalised name (a capital and
    3+ lowercase letters, a possessive 's allowed) that does not open the text; the kind is the words naming the product
    before 'sold by' or right after the name; the details are the parts after them, split at commas, 'and', 'with', 'in',
    'at' and 'that is/has/are/comes as'. Text from '(or ...' on is left out. None without a name. With `ok`, the seller is
    the first such name `ok` accepts (BW5: a capitalised product or category word before it is passed over)."""
    try:
        body = re.sub(r"\(\s*or\b[^()]{0,25}\)", " ", str(text or ""))      # (a short aside such as '(or ...)' mid-phrase)
        body = re.split(r"\(\s*or\b|\s+shows?\b|\s*[\u2014\u2013]+\s*or\b|\s+-+\s+or\b", body)[0]
        hit = next((m for m in re.finditer(r"(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])", body) if m.start() > 0
                    and (ok is None or ok(m.group(1)))), None)
        if not hit:
            return None
        before, after, kind = body[:hit.start()], body[hit.end():], ""
        sold = re.search(r".*(?:\bthe|\bfor|\ba|\ban)\s+(.*?)\s+(?:sold|made|offered|listed)\s+by\s*$", before, re.I | re.S)
        if sold:
            kind = sold.group(1)
        else:
            lead = re.match(r"\s*([^,;(]*?)(?=\s+(?:with|that|which|in|at|having|from|of|for|sold)\b|[,;]|$)", after, re.I)
            if lead:
                kind, after = lead.group(1), after[lead.end():]
        parts = re.split(r",|;|\band\b|\bwith\b|\bthat\s+(?:is|are|has|have|comes?(?:\s+as)?)\b|\bwhich\b|\bin\b|\bat\b", after, flags=re.I)
        details = []
        for part in parts:
            words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.%:/&'-]*", part) if w.lower() not in LINK_WORDS]
            if words:
                details.append(" ".join(words))
        return {"seller": hit.group(1), "kind": kind.strip()[:80], "details": details[:6]}
    except Exception:
        return None


def prop_named(prop: Any, text: Any, exact: bool = False) -> bool:
    """CE2b: a text names a property: the property's words (its "(product)" part left out, camelCase and separators split)
    appear as one run of the text's words, each the same word or (unless exact) its plural."""
    part = re.sub(r"([a-z])([A-Z])", r"\1 \2", re.split(r"\s*\(", str(prop or ""))[0])
    words = re.findall(r"[a-z0-9]+", part.lower())
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
    if not words or len(" ".join(words)) < 3:
        return False

    def same(a: str, b: str) -> bool:
        return a == b or (not exact and ((len(b) >= 3 and a in (b + "s", b + "es")) or (len(a) >= 3 and b in (a + "s", a + "es"))))
    n = len(words)
    return any(all(same(t, w) for t, w in zip(toks[i: i + n], words)) for i in range(len(toks) - n + 1))


def ref_span(prop: Any, source: Any) -> str:
    """The words of `source` right after its first mention of the property, to the end of that sentence (or to '(or')."""
    try:
        words = [w for w in re.findall(r"[a-z0-9]+", re.split(r"\s*\(", str(prop))[0].lower()) if w not in LINK_WORDS]
        hit = re.search(r"\b" + r"\W+".join(map(re.escape, words)) + r"\b", str(source).lower()) if words else None
        if not hit:
            return ""
        return re.split(r"\(\s*or\b|(?<=[.!?])\s", str(source)[hit.end(): hit.end() + 500])[0]
    except Exception:
        return ""


def lensed(piece: Any) -> Any:
    """A copy of an observation whose listing-like records also carry the plain names the code reads (price, in_stock,
    title) when they lack them: each from the one field whose name and value type fit (worlds rename them)."""
    try:
        piece = json.loads(json.dumps(piece))
    except (TypeError, ValueError):
        return piece

    def visit(d: dict[str, Any]) -> None:
        if not any("id" in str(key).lower() for key in d) and len(d) >= 3:
            # (R9c: a record whose listing id sits under a reference-like name - a listing's ref, code, sku or handle -
            # also carries it as id, the name the code reads; never a model, part, colour, postal or category code)
            refs = [key for key in d if isinstance(d[key], str) and re.fullmatch(ID_LIKE, d[key]) and " " not in d[key]
                    and re.search(r"(?:^|[_\s-])(?:ref|reference|identifier|uid|handle|sku|code|key)$", str(key).lower())
                    and not re.search(r"model|part|colou?r|zip|postal|categor|shelf|aisle|brand|maker|seller|shop|currency|lang|next|page|token|cursor|after", str(key).lower())]
            if len(refs) == 1:
                d["id"] = d[refs[0]]
        if not any(isinstance(val, str) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, val) for key, val in d.items()):
            return
        names = {str(key).lower(): key for key in d}
        if not any(re.search(r"price|cost", low) and is_number(d[key]) for low, key in names.items()):
            picks = [key for low, key in names.items() if is_number(d[key]) and re.search(r"amount|fee|charge|pay|rate|value", low)
                     and not re.search(r"total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|size|weight|width|height|length|found|match", low)]
            if len(picks) == 1:
                d["price"] = d[picks[0]]
        if not any(re.search(r"stock|avail", low) for low in names):
            picks = [key for low, key in names.items() if isinstance(d[key], bool)
                     and re.search(r"hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store", low)]
            if len(picks) == 1:
                d["in_stock"] = d[picks[0]]
        if not any(re.search(r"title|heading", low) for low in names):
            picks = [key for low, key in names.items() if isinstance(d[key], str) and re.search(r"headline|name|label|caption", low)
                     and not re.search(r"shop|seller|store|vendor|brand|make|categor|shelf|department|id", low)]
            if len(picks) == 1:
                d["title"] = d[picks[0]]
    walk_dicts(piece, visit)
    return piece


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
    """Model calls and world calls through the sandbox proxy, both bounded by the errand's clock."""

    def __init__(self, bind: dict[str, Any]):
        self.t0 = monotonic()
        self.bind = dict(bind)
        self.model_list = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
        self.world = ProxyClient(timeout=95, max_retries=1, retry_delay=1)      # (TA3: the retries are act's own)
        self.spend = 0.0
        self.tick = 0
        self.done = False
        self.broke = False
        self.log: list[dict[str, Any]] = []
        self.last_send = 0.0           # (TA) when the errand's newest HTTP request went out
        self.slow_until = 0.0          # requests keep SLOW_GAP apart until then (a 429 or 503 came)
        self.model_resume = 0.0        # the next model request waits until then (the pause after a failed attempt)
        self.last_reply = self.t0      # when a model last answered
        self.misses = 0                # model attempts failed since then
        self.no_model = False          # the run key pays for no more model calls (402, a key limit)
        self.banned: set[str] = set()  # models the proxy refused for this errand (401, 403, 404)
        self.tool_banned: set[str] = set()   # models no endpoint serves with tools (a 404 about tools)
        self.spare_models: list[str] = []    # allowed models a 403 listed: the errand's fallback ladder
        self.call_models: set[str] = set()   # the models the newest call tried
        self.mute = False              # the code finishes alone: no model call goes out
        self.call_fail = ""            # how the newest call that got no message failed: "empty", "refused" or "" (failed())

    def spare(self) -> float:
        return DEADLINE - (monotonic() - self.t0)

    def spent_out(self) -> bool:
        return self.spend >= SPEND_CAP * self.__dict__.get("scale", 1.0)

    def send(self, client: Any, path: str, body: dict[str, Any]) -> Any:
        """One HTTP request through the proxy, paced (TA1): at least PACE_GAP after the errand's previous request, SLOW_GAP
        for SLOW_FOR seconds after a refusal for load - the sandbox's episodes share one request bucket; a model request
        also waits until model_resume (the pause a failed model attempt set)."""
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

    def pause(self, seed: str, tries: int, status: Any) -> float:
        """The wait before trying again (TA2): from 2 s after a 429 (1 s after any other failure), doubled each time, at
        most 8 s, spread over 75-125% by a figure taken from the request's own text and the step, never from the clock, so
        errands refused together do not come back together."""
        mix = (len(seed) * 7 + sum(ord(ch) for ch in seed[-64:]) + 31 * tries + 13 * self.tick) % 101 / 100.0
        return min(8.0, (2.0 if status == 429 else 1.0) * 2 ** tries) * (0.75 + 0.5 * mix)

    def down(self, at: float | None = None) -> bool:
        """No model can be called, or none answered for DOWN_AFTER seconds (by `at`) while the newest attempts failed."""
        return self.no_model or (self.misses > 0 and (monotonic() if at is None else at) - self.last_reply > DOWN_AFTER)

    def usable(self, model_list: list[str], tools: bool = False) -> bool:
        return not self.no_model and not self.mute and any(
            mdl not in self.banned and not (tools and mdl in self.tool_banned) for mdl in list(model_list) + self.spare_models)

    @staticmethod
    def plain(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The messages with each content one plain string and no cache marks: the simpler body sent once after a 400."""
        out = []
        for msg in msgs:
            msg = dict(msg)
            if isinstance(msg.get("content"), list):
                msg["content"] = "\n\n".join(str(part.get("text") or "") for part in msg["content"] if isinstance(part, dict))
            out.append(msg)
        return out

    def query_model(self, msgs: list[dict[str, Any]], model_list: list[str], token_cap: int,
            toolset: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        # (TA2: a load or transport failure is tried again after a growing pause, a 400 once with plain contents, a model
        # the proxy refuses is dropped for the errand, a spent key stops the model calls; never the whole list in ms)
        self.call_fail = ""
        self.call_models = set()
        if self.mute or self.no_model:
            return None
        seed = ""
        kinds: set[str] = set()        # how the attempts failed: "empty" (no content), "hard" (a refusal), "soft" (load, network)
        refused_here: set[str] = set()   # models refused outright in this call: not tried again in it
        ladder = list(model_list) + [mdl for mdl in self.spare_models if mdl not in model_list]
        lone = len({mdl for mdl in ladder if mdl not in self.banned}) <= 1
        for pos, mdl in enumerate(ladder):
            if self.no_model:
                return self.failed(kinds)
            if mdl in self.banned or mdl in refused_here or (toolset and mdl in self.tool_banned):
                continue
            if "sonnet" in mdl:
                cheaper_next = pos + 1 < len(model_list) and "sonnet" not in model_list[pos + 1]
                # (a projected cost: prompt characters at about 2.3 a token, plus the answer)
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
                        self.tool_banned.add(mdl)          # (no endpoint takes tools: the model still reads)
                    elif not (status == 403 and re.search(r"moderat|flagged", said)):
                        self.banned.add(mdl)
                    listed = re.search(r'allowed_models"?\s*:\s*\[([^\]]*)', body) if status == 403 else None
                    if listed and not self.spare_models:
                        # (the proxy names the models this run may call: the first two become the errand's fallback ladder)
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
                    continue        # (the plain body goes at once: a refused body is no load)
                if isinstance(status, int) and 400 <= status < 500 and status not in (408, 425, 429):
                    kinds.add("hard")
                    refused_here.add(mdl)
                    break
                kinds.add("soft")
                if monotonic() - started > 20:
                    # (a hung or slow attempt: the next model, not the same wait again; once the models are down the call
                    # ends here only when two models failed so, or it has no other)
                    if self.down() and (lone or len(self.call_models) >= 2):
                        return self.failed(kinds)
                    break
                if not seed:
                    seed = json.dumps(msgs[-1:], default=str)[-600:]
                # (the pause grows with the attempts failed since the last model reply, across calls: 2, 4, 8, 8 ... s; the
                # next model request waits it out - send() - so a world call after a failed call is not held up)
                wait = self.pause(seed + mdl, min(self.misses - 1, 6), status)
                if self.spare() - wait < LAST_CALL - 10:
                    return self.failed(kinds)
                self.model_resume = monotonic() + wait
                if self.down(self.model_resume) and (lone or len(self.call_models) >= 2):
                    return self.failed(kinds)       # (gone for DOWN_AFTER: one attempt per model and two models per call)
        return self.failed(kinds)

    def failed(self, kinds: set[str]) -> None:
        """Ends a call that got no message: "empty" when a model answered without content, "refused" when every attempt
        was refused outright (a 4xx), "" when load or the network lost it (TA4 reads this)."""
        self.call_fail = "empty" if "empty" in kinds else "refused" if kinds == {"hard"} else ""
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

    def consult(self, ask_text: str, token_cap: int, model_list: list[str]) -> dict[str, Any] | None:
        msg = self.query_model([{"role": "user", "content": ask_text}], model_list, token_cap)
        return json_object((msg or {}).get("content") or "")

    def act(self, acts: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Sends one or more actions as consecutive steps; resyncs the step number when the world asks for another. After a
        refusal for load, a lost connection or a server error the same envelope goes again after a growing pause (TA3):
        the world answers a step it already ran from its record, so a repeat never runs it twice."""
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
            answer = self.send(self.world, "/environment/call", env_body)
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
                        # (TA3: a step of this errand ran but its reply was lost: the next number is free, or the world
                        # names the one it expects)
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


# ================================================================================================================ records
class Records:
    """Every observation the store returned, and the views the checks and the question plan read from them."""

    def __init__(self):
        self.items: list[Any] = []
        self.ids: set[str] = set()
        self.read: set[str] = set()     # ids whose details (a record with a nested property map) were seen
        self.label_of: dict[str, str] = {}     # (CC) listing id -> the category label a record of that id carried

    def add(self, obs: Any) -> None:
        self.items.append(obs)

        def visit(d: dict[str, Any]) -> None:
            for key, piece in d.items():
                if isinstance(piece, str) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, piece):
                    self.ids.add(piece)
            # (a record carrying a property map of 2+ fields is a read record, whichever tool showed it)
            ident = next((piece for key, piece in d.items() if isinstance(piece, str) and str(key).lower() in ("id", "_id")
                          or (isinstance(piece, str) and str(key).lower().endswith("id") and re.fullmatch(ID_LIKE, piece))), None)
            if ident and any(isinstance(val, dict) and sum(1 for v in val.values() if isinstance(v, (str, int, float))) >= 2
                             for val in d.values()):
                self.read.add(ident)
            own = next((piece.strip() for key, piece in d.items() if isinstance(piece, str) and 0 < len(piece.strip()) <= 60
                        and re.search(r"categor|shelf|department|aisle", str(key).lower()) and "id" not in str(key).lower()), "")
            if ident and own:
                # (CC: a listing's category label, for its records that carry none - a spec sheet may hold only the id, the
                # title and the property map while its search row and offer carry the label)
                self.__dict__.setdefault("label_of", {}).setdefault(ident, own)
        walk_dicts(obs, visit)

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

    def cards(self, char_cap: int = 30000, cheaper_than: str = "", skip: set[str] | None = None, first: Any = ()) -> list[str]:
        """One line per listing seen: its search row and its details merged (the details win), DETAILS listings first,
        each value clipped; within a character budget. With cheaper_than, only that listing and the ones priced below
        it (or with no price seen) are kept. Listings in `first` come before all others (EA1: they keep their card when the
        budget is short)."""
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower()
                          and re.fullmatch(ID_LIKE, val) and val in self.ids), None)
            # (EA1: a record of 3 keys or fewer counts when it carries a property map of 2+ fields - a spec sheet may be the
            # id, the title and the map alone - the rule add() uses to mark a read)
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
            # (v605e: the cards do not all fit - a read listing its record shows sold out goes after the other read ones)
            def gone(i: str) -> bool:
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
        """CC: a record's category label as written: its own category-like field, else the label another record of the
        same listing id carried ('' when neither)."""
        own = next((val.strip() for key, val in d.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                    and re.search(r"categor|shelf|department|aisle", str(key).lower()) and "id" not in str(key).lower()), "")
        if own:
            return own
        labels = self.__dict__.get("label_of") or {}
        return next((labels[val] for key, val in d.items() if isinstance(val, str) and val in labels and "id" in str(key).lower()), "")

    def rich_reads(self, shelf: str = "") -> int:
        """Listings read whose record carries a property map of 3+ fields (in flat records, 5+ short text fields); with
        `shelf`, only records of that category."""
        rich: set[str] = set()

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.read), None)
            if not ident:
                return
            if shelf and self.label_for(d).strip().lower() != shelf.strip().lower():
                return        # (CC: a record without a label of its own takes its listing's)
            maps = [val for val in d.values() if isinstance(val, dict)]
            if any(len(val) >= 3 for val in maps) or (not maps and sum(1 for val in d.values() if isinstance(val, str) and 0 < len(val) <= 70) >= 5):
                rich.add(ident)
        for rec in self.items:
            walk_dicts(rec, visit)
        return len(rich)

    def shelves_read(self) -> list[str]:
        """The category labels of the records read, most read first."""
        tally: dict[str, int] = {}

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.read), None)
            shelf = self.label_for(d) if ident else ""      # (CC: a spec sheet without a label takes its listing's)
            if ident and shelf and any(isinstance(val, dict) for val in d.values()):
                tally[shelf] = tally.get(shelf, 0) + 1
        for rec in self.items:
            walk_dicts(rec, visit)
        return sorted(tally, key=lambda s: -tally[s])

    def prop_menu(self, shelf: str = "") -> tuple[str, int]:
        """Property names on detail-like records (5+ short string fields), with counts and sample values; with `shelf`,
        only records of that category. Fields that are not product properties (brand, shop, category, title, text,
        price, stock, ids) are left out: questions about them reveal nothing."""
        tallies: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        sets = 0
        mapped, row_tally = 0, {}

        def on_shelf(d: dict[str, Any]) -> bool:
            return any(isinstance(val, str) and val.strip().lower() == shelf.lower() and re.search(r"categor|shelf|department|aisle", str(key).lower())
                       for key, val in d.items()) or self.label_for(d).strip().lower() == shelf.strip().lower()     # (CC)

        def seek(piece: Any, inside: bool, lvl: int) -> None:
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
            nonlocal sets, mapped
            brief_vals = [(fkey, val) for fkey, val in d.items() if isinstance(val, str) and 0 < len(val) <= 70 and not junk_key(fkey)]
            if len(brief_vals) < 5 and not (shelf and len(brief_vals) >= 2 and not any(isinstance(v, dict) for v in d.values())):
                return
            sets += 1
            # (CC: a short record carrying a listing id is a listing row; one without is a property map or a full record)
            rowlike = len(brief_vals) < 5 and any(isinstance(val, str) and "id" in str(fkey).lower() for fkey, val in d.items())
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
            # (CC: once the category's records carry their properties, a field only its listing rows carry - the shop, a
            # summary - is a listing field, not a property)
            for fkey in [fkey for fkey in tallies if row_tally.get(fkey, 0) >= tallies[fkey]]:
                tallies.pop(fkey)
        top_props = sorted(tallies, key=lambda fkey: -tallies[fkey])[:28]
        return "\n".join(f"- {fkey} (in {tallies[fkey]} listings): " + " | ".join(examples.get(fkey, [])) for fkey in top_props), sets

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
        self.resolved: list[str] = []          # requirements whose value the code read on a referenced listing (BT6)
        self.ref_news: list[dict[str, Any]] = []   # needs the shopper gave by pointing at a listing, for the resolver (BT1)

    def items(self) -> list[dict[str, Any]]:
        """The products the request asks for, when it asks for two or more different ones (else [])."""
        rows = []
        for row in self.reading.get("items") or []:
            if isinstance(row, dict) and str(row.get("product") or "").strip():
                qty = row.get("quantity")
                reqs = row.get("requirements") if isinstance(row.get("requirements"), list) else []
                rows.append({"product": str(row["product"]).strip(), "quantity": qty if isinstance(qty, int) and not isinstance(qty, bool) and qty >= 1 else 1,
                             "requirements": [str(r) for r in reqs if r and ":" in str(r)]})
        return rows if len(rows) >= 2 else []

    def quantity(self) -> int:
        if self.items():
            return 1      # (several products: each line has its own quantity; filters use one unit)
        need_n = self.reading.get("quantity")
        return need_n if isinstance(need_n, int) and not isinstance(need_n, bool) and need_n >= 1 else 1

    def counts(self) -> list[int]:
        """The units each cart line must hold: one entry per product."""
        return [row["quantity"] for row in self.items()] or [self.quantity()]

    def items_text(self) -> str:
        rows = self.items()
        if not rows:
            return ""
        return ("THE REQUEST ASKS FOR SEVERAL PRODUCTS; the order holds exactly one line per product, at its quantity, and nothing "
                "else; each listing is judged only against the requirements of the product it is (its category decides), and the "
                "cheapest or best rule applies within each product: " + " | ".join(
                    f"{n + 1}) {row['product']} x{row['quantity']}" for n, row in enumerate(rows)))

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
        collected += f"; quantity {self.quantity()}" if not self.items() else "; " + self.items_text()
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
        """need -> the same need under the property name the shopper's line uses, when the listener's name is not a phrase
        of the line but each of its words begins the matching word of one run of the line ("pack type" -> "packaging type")."""
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

    def drop_named(self, old_need: str, lns: list[str]) -> bool:
        """May a dropped requirement go: True unless it is a stated or revealed one and no 1-3 word phrase of the shopper's
        lines names its property (same_prop), so a line about another property with a similar name never drops it. CJ1: a
        phrase naming a field of the product's category (strict_prop) whose values carry the requirement's value names it
        too ('drop dietary needs' for 'sugar content: low sugar' where the store writes low sugar under dietary needs); a
        revealed need also goes when the reply gives its property a new value (a replacement)."""
        try:
            name = str(old_need).split(":", 1)[0]
            tag = re.findall(r"\(([^)]*)\)", name)

            def fits_tag(w: str) -> bool:
                return same_prop(w.split(":", 1)[0], name) and (not tag or re.findall(r"\(([^)]*)\)", w.split(":", 1)[0]) in ([], tag))
            stated = [w for w in self.written if fits_tag(w)]
            if not stated:
                # (the reply's own new needs are in the ledger already when its drops are read: they are not the old value)
                fresh = [n for n in self.__dict__.get("fresh_needs") or [] if isinstance(n, str) and ":" in n]
                stated = [w for w in self.uncovered if fits_tag(w) and w not in self.written and w not in fresh]
                if not stated:
                    return True
                if any(fits_tag(n) and n != old_need for n in fresh):
                    return True       # (a revealed value replaced by a new one of the same property)
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
                        continue      # (a value given by pointing at another listing has no field values of its own)
                    fields = hook(head) or {}
                    others = [r.split(":", 1)[0] for r in self.written + self.uncovered if isinstance(r, str) and ":" in r
                              and not same_prop(r.split(":", 1)[0], name)]
                    for p in phrases:
                        if any(strict_prop(o, p) for o in others):
                            continue      # (the phrase names another requirement in force: it speaks of that one)
                        if any(strict_prop(key, p) and value_seen(value, " | ".join(vals)) for key, vals in fields.items()):
                            self.wire.log.append({"kind": "drop_by_field", "need": w[:200], "phrase": p})
                            return True
            return False
        except Exception:
            return True

    def literal_values(self, answer: dict[str, Any]) -> None:
        """CJ2: a new need whose value, its comparison words off, is a value the store writes in the field of that
        property ('falls short of not available' where listings write 'not available') becomes "property: <that value>",
        in the needs and in requirements_now, and positive_value() takes it as a value from then on."""
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
        """A property a 'don't mind' must not free: a stated requirement in force has a positive value the request itself
        writes, and it is the same property by name or the asked name ends in its name ('general style' for 'style')."""
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
        """CD: one of the needs is written by the lines themselves - the words of its property and of its value all appear
        in them (a reveal names both; a figurative word of a one-at-a-time reply names no property)."""
        said = " ".join(str(ln) for ln in lns).lower()
        for need in needs if isinstance(needs, list) else []:
            head, _, value = str(need).partition(":")
            names = [w for w in prop_words(head) if len(w) >= 3]
            values = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in LINK_WORDS]
            if names and values and all(w in said for w in names + values):
                return True
        return False

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
        answer = self.wire.consult(ask_text, 2500, LISTENER)
        if not answer:
            if self.wire.mute or self.wire.misses > 0:
                # (TA4: the listener's call was lost to load or the network, or the code finishes alone: the lines stay
                # unread and are read once a model answers again - Errand.catch_up - or by the code's finish)
                self.caught.difference_update(new_lines)
                self.exchanges.pop()
                self.pending = list(self.__dict__.get("pending") or []) + [(list(new_lines), qtext)]
                self.pending_at = monotonic()
            return ""
        try:
            if answer.get("meta_only") is True and qtext and not self.valued(answer.get("needs"), new_lines):
                # (CD: the line only asks for one question at a time - it answers nothing: no need, no freed or dropped
                # property; the property asked stays open and goes out again alone)
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
        if renames:
            # (a revealed need keeps the property name the shopper's line uses: the listener shortened "packaging type" to the
            # stated "pack type", which replaced the stated requirement)
            answer["needs"] = [renames.get(n, n) if isinstance(n, str) else n for n in answer.get("needs") or []]
            if isinstance(answer.get("requirements_now"), list):
                answer["requirements_now"] = [renames.get(r, r) if isinstance(r, str) else r for r in answer["requirements_now"]]
            self.wire.log.append({"kind": "need_renamed", "names": list(renames.values())[:4]})
        masked = new_lines
        hook = self.__dict__.get("ref_hook")
        if hook:
            # (CE2b: a line pointing at another listing carries one need of the property it names outside its pointing
            # phrase; that listing's facts are no needs, drops or freed properties - the lines with the phrase blanked
            # are what the drop and restore tests read)
            try:
                masked = hook(answer, new_lines, qtext) or new_lines
            except Exception as err:
                self.wire.log.append({"kind": "ref_line_error", "error": f"{type(err).__name__}: {err}"[:300]})
                masked = new_lines
        remarks = []
        # (a requirement stands on the first reading or on the shopper's own words: one whose property and value
        # words appear in neither came from the story and is left out)
        said = (so_far + " " + " ".join(self.exchanges) + " " + (qtext or "")).lower()

        def grounded(req: str) -> bool:
            words = [w for w in re.findall(r"[a-z]+", req.lower()) if len(w) >= 3]
            return not words or any(w in said for w in words)

        try:
            self.literal_values(answer)
        except Exception as err:
            self.wire.log.append({"kind": "literal_value_error", "error": f"{type(err).__name__}: {err}"[:300]})
        self.fresh_needs = [n for n in answer.get("needs") or [] if isinstance(n, str)]     # (CJ1: for drop_named)
        now_list = answer.get("requirements_now")
        if isinstance(now_list, list) and all(isinstance(req, str) for req in now_list):
            self.now_reqs = [req for req in now_list if req.strip() and grounded(req)]
        for new_need in answer.get("needs") or []:
            if isinstance(new_need, str) and new_need and new_need not in self.uncovered and grounded(new_need):
                self.uncovered.append(new_need)
                remarks.append(f"new requirement: {new_need}")
        try:
            # (BT1: a need the shopper gives by pointing at another listing goes to the code's reference resolver)
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
                    # (a stated value stays; the reply still counts as a "don't mind" for the question's bookkeeping)
                    self.pinned_heard = self.__dict__.get("pinned_heard", 0) + 1
                    continue
                self.indifferent.append(prop_name)
        widened = False
        for old_need in answer.get("dropped") or []:
            if isinstance(old_need, str) and old_need and old_need not in self.let_go \
                    and not (is_number(answer.get("new_budget"))
                             and re.search(r"\b(?:prices?|budget|costs?|spend\w*|cents?|dollars?)\b", old_need.lower())):
                if not self.drop_named(old_need, masked):
                    # (a stated requirement the shopper's lines do not name stays in force: a reply about another property
                    # with a similar name dropped it)
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
            # (a need the shopper revealed stays in force until the shopper drops it: a rebuilt list that lost it,
            # after a "don't mind" to some other message, gets it back)
            for need in self.uncovered:
                prop_name = need.split(":", 1)[0].strip().lower()
                kept = any(prop_name and prop_name in req.lower() for req in self.now_reqs)
                released = any(prop_name and prop_name in old.lower() for old in self.let_go)
                if prop_name and not kept and not released:
                    self.now_reqs.append(need)
            for need in self.written:
                # (a stated requirement stays in force too unless it was dropped: a rebuilt list that lost it, with no
                # requirement of that property left, gets it back)
                try:
                    if need in self.let_go or need in self.now_reqs or self.drop_named(need, masked):
                        continue      # (a line that names the property is the listener's to read)
                    name, tag = need.split(":", 1)[0], re.findall(r"\(([^)]*)\)", need.split(":", 1)[0])
                    if any(same_prop(name, old.split(":", 1)[0]) and re.findall(r"\(([^)]*)\)", old.split(":", 1)[0]) == tag for old in self.let_go):
                        continue
                    if not any(same_prop(name, req.split(":", 1)[0]) and re.findall(r"\(([^)]*)\)", req.split(":", 1)[0]) == tag
                               for req in self.now_reqs if isinstance(req, str)):
                        self.now_reqs.append(need)
                except Exception:
                    continue
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
class CartBook:
    """The cart as the code knows it: which form of the cart tool puts units in (the first accepted call's enum
    values), the units of each line counted from the calls and corrected by every cart view, and whether the tool
    adds units to a held line or sets them."""

    def __init__(self, the_tool: str | None, jschema: dict[str, Any]):
        self.the_tool = the_tool
        self.prop_specs = (jschema or {}).get("properties") or {}
        self.put_form: str | None = None
        self.units: dict[str, float] = {}
        self.adds: bool | None = None          # True: a put on a held line adds; False: it sets; None: not seen yet
        self.used = False
        self.changed_at = 0
        self.viewed_at = -1
        self.last_view: Any = None
        self.pending: dict[str, float] = {}    # line -> units the code expects after the last call, if it adds / sets

    def form(self, call_args: dict[str, Any]) -> str:
        return as_text({fkey: val for fkey, val in call_args.items() if isinstance(self.prop_specs.get(fkey), dict) and "enum" in self.prop_specs[fkey]})

    def line_spec(self) -> tuple[str, str, str] | None:
        """(array argument, id field, count field) when the cart tool takes a list of line objects."""
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
        """The (listing, units) lines a cart call names: every line of an array argument, else the one flat line."""
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
        """The listing a cart call puts in (the first line of an array argument that has units; else the first line)."""
        found = self.lines(call_args)
        put = [ln for ln in found if ln[1] is None or ln[1] > 0]
        return (put or found)[0][0] if found else None

    def aim_flat(self, call_args: dict[str, Any]) -> str | None:
        """The listing a flat cart call names: a one-word string argument that is not an enum."""
        for key, piece in call_args.items():
            pspec = self.prop_specs.get(key) if isinstance(self.prop_specs.get(key), dict) else {}
            if isinstance(piece, str) and "enum" not in pspec and piece.strip() and " " not in piece.strip():
                return piece
        return None

    def count_arg(self, call_args: dict[str, Any]) -> float | None:
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
        return self.put_form is None or self.form(call_args) == self.put_form

    def held(self, thing: str) -> float:
        return self.units.get(thing, 0.0)

    def booked(self, call_args: dict[str, Any], obs: Any) -> str:
        """Books a cart call the world answered; returns a note on what the cart holds now."""
        self.used = True
        if isinstance(obs, dict) and obs.get("error"):
            return ""
        if self.put_form is None:
            self.put_form = self.form(call_args)
        if self.line_spec() and isinstance(call_args.get(self.line_spec()[0]), list):
            # (a list of lines states the whole basket: the lines it leaves out are gone; a later view corrects this)
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
                self.pending = {thing: count}         # the units if the tool sets instead of adding
            else:
                self.units[thing] = count if count > 0 else 0.0
        else:
            self.units[thing] = max(0.0, prior - count) if count > 0 else 0.0
        if self.units.get(thing, 0) <= 0:
            self.units.pop(thing, None)
        seen = self.view_units(obs)
        if seen is not None:
            self.learn(seen)
        elif (count <= 0 or not self.is_put(call_args)) and self.shows_empty(obs, thing):
            # (v605e: a take-out whose answer lists the cart's lines and lists none - the line taken out is gone; another line
            # still counted stays until a view shows the cart, since such a list may hold only what changed)
            self.units.pop(thing, None)
            self.pending.pop(thing, None)
        return self.summary()

    def shows_empty(self, obs: Any, thing: str) -> bool:
        """v605e: the answer to taking out `thing` shows an empty cart - a list that is empty, at least one number and every
        number 0, and no other line the code counts named anywhere in it (an acknowledgment with no list or no number shows
        nothing; the answer may name the line taken out)."""
        if not isinstance(obs, dict) or thing not in self.units:
            return False
        lists = [v for v in obs.values() if isinstance(v, list)]
        nums = [v for v in obs.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        text_body = as_text(obs)
        return bool(lists) and all(not v for v in lists) and bool(nums) and all(v == 0 for v in nums) \
            and not any(other in text_body for other in self.units if other != thing)

    def view_units(self, obs: Any) -> dict[str, float] | None:
        """Line -> units read from a cart-like observation: dicts holding a known or one-word id string and one
        positive integer field (the units). None when the observation shows no such rows."""
        rows: dict[str, float] = {}
        known = set(self.units) | set(self.pending)

        def visit(d: dict[str, Any]) -> None:
            ids = [val for val in d.values() if isinstance(val, str) and (val in known or re.fullmatch(ID_LIKE, val)) and " " not in val]
            pairs = [(fkey, val) for fkey, val in d.items() if isinstance(val, int) and not isinstance(val, bool) and 0 < val < 1000
                     and str(fkey) != "price"]
            if len(pairs) > 1:
                # (a row holding a unit price and a count: the count-named field is the units)
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

    def faults(self, need_n: int, counts: list[int] | None = None) -> list[str]:
        """What keeps the cart from being exactly one line at the wanted units (with `counts`, one line per product)."""
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
        turn_cap = self.pview.get("max_calls_per_turn")
        self.turn_cap = int(turn_cap) if is_number(turn_cap) and turn_cap >= 1 else 16     # (R9g: the task's calls per turn)
        self.wire.done = bool(self.pview.get("finished"))
        self.archive = Records()
        self.needs = Needs(self.wire, self.request)
        self.needs.ref_hook = self.ref_fix          # (CE2b: the shopper's pointing lines, read by the code)
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
        self.passed: set[str] = set()              # item|version the independent check found meeting everything
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
        self.refs: list[dict[str, Any]] = []        # requirements given by reference, followed by the code (BT)
        LITERAL_VALUES.clear()                      # (CJ2: word values are learned per errand)
        self.needs.field_hook = self.store_fields   # (CJ: the store's fields of a product's category, for drops and literals)

    # ----------------------------------------------------------------------------------------------- small reads
    @property
    def log(self) -> list[dict[str, Any]]:
        return self.wire.log

    def moves_left(self) -> int:
        return self.step_cap - self.wire.tick

    def role(self, call_name: str) -> str | None:
        return self.role_map.get(call_name)

    def ask_limit(self) -> int:
        """R9b: the questions allowed - the message allowance (the request's, the status tool's or MSG_GUESS) less the one
        message kept for stating the basket before the order."""
        cap = self.msg_cap if self.msg_cap is not None else MSG_GUESS
        keep = 1 if self.role("order") and self.role("message") and cap >= 2 else 0
        if keep and cap >= 4 and self.all_revealed():
            keep = 2      # (every product drew a need: a second message stays free for a statement after a refused order)
        return max(0, cap - keep)

    def all_revealed(self) -> bool:
        """R9b: every product asked for has drawn at least one hidden need from the shopper (still in force)."""
        found = [n for n in self.needs.uncovered if isinstance(n, str) and n not in self.needs.let_go]
        if not found:
            return False
        rows = self.needs.items()
        if not rows:
            return True
        return len({self.product_of(n) for n in found} - {None}) >= len(rows)

    def call_kind(self, call_name: str, call_args: dict[str, Any] | None) -> str | None:
        """'order', 'decline' or None for a call. When one tool does both, its mode value tells which."""
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
        return {self.role("order"), self.role("decline")} - {None}

    def clock_text(self) -> str:
        """v605e: the time left as the actor sees it - nothing while more than half of the errand's time remains, then rounded
        down to CLOCK_STEP seconds (exact seconds differ on every run and re-sample the actor's answers). Text only: every
        time-based decision reads wire.spare() itself."""
        left = max(0.0, self.wire.spare() - LAST_CALL)
        if left > (DEADLINE - LAST_CALL) / 2:
            return ""
        step = int(left // CLOCK_STEP) * CLOCK_STEP
        return f"about {step} s left" if step > 0 else f"under {CLOCK_STEP} s left"

    def status_line(self) -> str:
        spare = self.moves_left()
        pieces = [f"[status] steps used {self.wire.tick} of {self.step_cap}, {spare} left",
                 f"messages sent {self.msgs_sent}" + (f" of {self.msg_cap}" if self.msg_cap is not None else "")]
        clock = self.clock_text()
        if clock:
            pieces.append(clock)
        text_body = "; ".join(pieces) + ".\n[requirements now] " + self.needs.text_body() + self.ref_line() + self.product_steps() + self.fit_note_line()
        usable = [tdef["function"]["name"] for tdef in self.offered()]
        if len(usable) < len(self.toolset):
            text_body += "\n[tools usable now] " + ", ".join(usable) + " (any other tool call is refused without using a step" + (
                "; questions to the shopper are planned and sent for you" if self.asking else "") + ")."
        if self.trolley.used:
            text_body += "\n" + self.trolley.summary()
        if spare <= 2:
            text_body += (" LAST STEPS: order the cart now if it holds a candidate for " + ("every product asked for" if self.needs.items()
                          else "the item") + "; otherwise put the best candidate in (a verified fit, else the cheapest in-stock, "
                          "in-budget listing whose record shows no requirement failing) and order. Decline only when the records "
                          "show that nothing can fit.")
        elif spare <= 5 or self.wire.spare() < CLOSING:
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

    # ----------------------------------------------------------------------------------------------- setup
    def read_request(self) -> None:
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
                break       # (TA5: the readers go round again only while load or the network lost every call, until DOWN_AFTER)
            msg = self.wire.query_model([{"role": "user", "content": content}], [mdl], 3000)
            reading = json_object((msg or {}).get("content") or "") or {}
            if isinstance(reading.get("roles"), dict):
                break
            self.wire.log.append({"kind": "setup_retry", "model": mdl})
        if not isinstance(reading.get("roles"), dict):
            # (TA5: no model could read the request - the plan comes from its own words and the tools' schemas, so the
            # errand still shops, and still finishes by code if the models stay away; a model reading replaces it once a
            # model answers again - catch_up)
            try:
                reading = {**self.code_reading(), **{key: val for key, val in reading.items() if key != "roles"}}
            except Exception as err:
                self.wire.log.append({"kind": "code_reading_error", "error": f"{type(err).__name__}: {err}"[:300]})
                reading = {}
            self.read_at = monotonic()
            self.wire.log.append({"kind": "code_reading", "roles": reading.get("roles"), "product": reading.get("product"),
                                  "price_cap": reading.get("price_cap"), "items": len(reading.get("items") or [])})
        self.use_reading(reading)

    def use_reading(self, reading: dict[str, Any], again: bool = False) -> None:
        """Applies a reading of the request to the errand's plan; `again` (TA4: a model reading replacing the code's) keeps
        the cart tool the cart was used with, a budget the shopper changed since, and restarts the step limit from the
        world's own."""
        if not again:
            self.steps_given = self.step_cap
        old_cart = self.role("cart")
        role_map = reading.get("roles") if isinstance(reading.get("roles"), dict) else {}
        self.role_map = {fkey: val for fkey, val in role_map.items() if isinstance(val, str) and val in self.shapes}
        if again and self.trolley.used and old_cart:
            self.role_map["cart"] = old_cart
        if not (again and self.trolley.used):
            self.trolley = CartBook(self.role("cart"), self.shapes.get(self.role("cart") or "", {}))
        self.needs.sweepable = bool(self.role("search")) and self.page_arg() is not None
        spend_cap = reading.get("message_budget")
        if is_number(spend_cap) and spend_cap >= 0:
            self.msg_cap = int(spend_cap)
        self.needs.reading = reading
        try:
            # (several products need reads and checks per product: the spend at which the errand heads for its finish grows
            # 30% for each product past the first, at most two more)
            self.wire.scale = 1.0 + 0.3 * min(2, max(0, len(self.needs.items()) - 1))
        except Exception:
            self.wire.scale = 1.0
        named = [str(req) for req in reading.get("requirements") or [] if req]
        # (a requirement names its property; a wish without one is story or a soft want, never a requirement)
        self.needs.written = [req for req in named if ":" in req and req.split(":", 1)[0].strip() and req.split(":", 1)[1].strip()]
        loose = [req for req in named if req not in self.needs.written]
        if loose:
            reading["soft_wants"] = [str(w) for w in reading.get("soft_wants") or [] if w] + loose
            reading["requirements"] = list(self.needs.written)
        try:
            self.story_values(reading)
        except Exception as err:
            self.wire.log.append({"kind": "story_values_error", "error": f"{type(err).__name__}: {err}"[:300]})
        try:
            self.ref_register()
        except Exception as err:
            self.wire.log.append({"kind": "ref_register_error", "error": f"{type(err).__name__}: {err}"[:300]})
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
            self.needs.version += 1       # (checks made under the code's plan no longer count)

    def story_values(self, reading: dict[str, Any]) -> None:
        """A requirement whose value words the request never writes is a guess from the story: it leaves the requirements
        (and its product's entry) and joins the hints, at most half of them and never the sole one; references are exempt.
        The hints become a list of strings."""
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
        """The reader's hints that are written values (lead_value: 'style (jumpsuit or romper): sexy, for a dinner' when the
        request says 'a sexy jumpsuit or romper'): each becomes the stated requirement 'style (jumpsuit or romper): sexy'
        unless a requirement of that property (or carrying that word) is in force for its product. Returns those hints."""
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
        """The reader's story hints whose property is still open (not asked, not a requirement in force, not one the
        shopper does not mind); with k, only those of product k (or naming no product). EC5: a hint whose name sits inside
        an answered property's name, or holds it (roast, coffee roast), was asked (asked_inside). EC3: a hint whose own
        property drew no reveal comes back first under its '(or X)' alternative (hint_alt), once."""
        out, alts = [], []
        for hint in self.needs.reading.get("hints") or []:
            prop = str(hint).split(":", 1)[0].strip()
            if not prop:
                continue
            j = self.product_of(prop) if self.needs.items() else None
            if k is not None and j not in (k, None):
                continue
            # (a hint is the story's pointer at the property: a requirement of its name settles it only by a positive value
            # its listings' sample values carry, when its field was seen)
            if self.open_prop(j if self.needs.items() else None, prop, self.prop_line(j, prop), by_samples=True) \
                    and not self.asked_inside(j if self.needs.items() else None, prop):
                out.append(str(hint))      # (EC5: nor one whose subject was asked under a name holding its own)
                continue
            alt = self.hint_alt(j if self.needs.items() else None, str(hint), prop)
            if alt:
                alts.append(alt)
        return alts + out

    def prop_line(self, k: int | None, prop: str) -> str:
        """The question-menu line (name, count, sample values) of the field named like `prop` (strict_prop) among product
        k's own records (one product: its own category's, else all); '' when none. The menu is kept until a record arrives."""
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
        """The sample values of product k's field named like `prop` (empty when it was not seen)."""
        line = self.prop_line(k, prop)
        values = line.split("):", 1)[1].split("|") if "):" in line else []
        return {" ".join(re.findall(r"[a-z0-9]+", v.lower())) for v in values} - {""}     # (round-neck and round neck alike)

    def same_field(self, k: int | None, a: str, b: str) -> bool:
        """Two names for one field of product k: the same property by strict name, or related names (same_prop: collar
        and collar type, sleeve and sleeves) whose fields share a sample value (the same property under two keys)."""
        if strict_prop(a, b):
            return True
        try:
            return same_prop(a, b) and bool(self.field_values(k, a) & self.field_values(k, b))
        except Exception:
            return False

    def hinted_prop(self, k: int | None, prop: str) -> bool:
        """A story hint names this property (strict_prop), for product k or for no product."""
        name = str(prop).split("(")[0].strip()
        for hint in self.needs.reading.get("hints") or []:
            head = str(hint).split(":", 1)[0]
            if strict_prop(head, name) and (k is None or not self.needs.items() or self.product_of(head) in (k, None)):
                return True
        return False

    def product_reads(self, k: int | None) -> int:
        """Records with facts read on product k's own category (one product: its own category, else all)."""
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
        """A story hint's property that no field of product k's records carries (strict name) while 2+ of its records were
        read: no longer asked by that plain name (EC1: at 2 reads, so the planner maps the story onto a shown key)."""
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            if not name or not (hinted or self.hinted_prop(k, name)) or self.prop_line(k, name) or self.product_reads(k) < 2:
                return False
            menu = (self.__dict__.get("menu_memo", {}).get(k) or (0, ""))[1]
            # (a related key name - 'closure' for 'closure type' - is the hint's own field under another name)
            return not any(same_prop(line.strip("- ").split(" (in ")[0].strip(), name) for line in menu.splitlines())
        except Exception:
            return False

    def hint_mark(self, hint: str) -> str:
        """A hint as the planner reads it: marked [no key] when plain_hint holds for it."""
        try:
            prop = str(hint).split(":", 1)[0].strip()
            k = self.product_of(prop) if self.needs.items() else None
            return f"{hint} [no key]" if self.plain_hint(k, prop, hinted=True) else str(hint)
        except Exception:
            return str(hint)

    @staticmethod
    def hint_alts_of(detail: str) -> list[str]:
        """The '(or X, Y)' alternatives a hint's detail keeps (one_prop_hint writes them there)."""
        return [a.strip() for m in re.findall(r"\(or ([^()]+)\)", str(detail)) for a in re.split(r",|\bor\b|/", m) if a.strip()]

    @staticmethod
    def name_inside(a: Any, b: Any) -> bool:
        """EC4/EC5: the words of one property name (3+ letters, no link word) all among the other's: roast and coffee roast,
        fit and pants fit type - one subject under a plain and a qualified key."""
        wa = {w for w in prop_words(a) if len(w) >= 3 and w not in LINK_WORDS}
        wb = {w for w in prop_words(b) if len(w) >= 3 and w not in LINK_WORDS}
        return bool(wa) and bool(wb) and (wa <= wb or wb <= wa)

    def asked_inside(self, k: int | None, head: str) -> bool:
        """EC5: an answered question of product k (its property's tag, else its asked text names the product) asked a
        property whose name holds the hint's name or sits inside it: the hint's subject was asked under another key."""
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
        """EC3: a story hint whose own property was asked and drew no reveal (a no-preference or confused reply, logged in
        quiet_asks) reopens once under the first of its '(or X)' alternatives that is a key of product k's records
        (prop_line), while that key is still open and not asked: 'X (product): detail'. '' otherwise."""
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
        """EC4/EC5: a story hint of product k (or naming no product) points at `prop`: its property or one of its '(or X)'
        alternatives names it (strict_prop), or its text writes every word of the property's name."""
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
        """EC4: two property names that may stand for one subject: not the same name (strict_prop), and a word of one
        matches a word of the other (equal, or one a 4+ letter start of the other: pack, packaging) where that word leads
        its own name (any word of a one-word name, else any but the last): battery type and battery technology, pack type
        and packaging type, material and clothing material; plug type and bulb type share only their last word."""
        if strict_prop(a, b):
            return False

        def words(name: Any) -> tuple[list[str], list[str]]:
            ws = [w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
            return ws, (ws[:-1] if len(ws) > 1 else ws)

        def match(x: str, y: str) -> bool:
            return x == y or (min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x)))
        wa, ha = words(a)
        wb, hb = words(b)
        return any(match(x, y) for x in ha for y in wb) or any(match(x, y) for x in wa for y in hb)

    def twin_of(self, k: int | None, a: str, b: str) -> bool:
        """EC4: `a` may stand for the same subject as `b` for product k: name_twin, or the same field under a related name
        (same_field), never the same name."""
        try:
            return not strict_prop(a, b) and (self.name_twin(a, b) or self.same_field(k, a, b))
        except Exception:
            return False

    def twin_key(self, k: int | None, prop: str) -> str:
        """EC4: the story twin of `prop` for product k: a key of its records (the menu's most common first), open and not
        asked, that twin_of links to it (never one whose name sits inside the other's: the same subject) and that the
        story names - a word of its own that the property's name lacks (3+ letters, no link word, no word of the
        product's name) in one of the product's story hints or in the request, or a story hint maps it (hint_mapped);
        '' when none."""
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
                    continue          # (one name inside the other - roast, coffee roast - is the same subject: its reply stands)
                words = [w for w in prop_words(key) if len(w) >= 3 and w not in LINK_WORDS and w not in mine and not kind_in(w, own)]
                if any(re.search(r"\b" + re.escape(w), story) for w in words) or self.hint_mapped(k, key):
                    return key
        except Exception:
            return ""
        return ""

    def values_form(self, k: int | None, prop: str, product: str) -> str:
        """EC4: 'For the <product>, would you like A, B or C?' from 2-3 sample values of product k's field `prop` that no
        other key of its records carries (one value each, with a letter, at most 30 characters); '' with fewer than 2."""
        try:
            line = self.prop_line(k, prop)
            if "):" not in line:
                return ""
            menu = (self.__dict__.get("menu_memo", {}).get(k) or (0, ""))[1]

            def norm(v: str) -> str:
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

    def distinct_ask(self, prop: str) -> tuple[str, str] | None:
        """EC4 (i): the one re-ask of a property whose plain one-property question drew a one-at-a-time reply (its name
        stands for a second subject of the shopper's too): (the label, the text) - the property's own values form first
        (the collision is symmetric: its twin's name would draw the same reply, while its values name it alone; in the
        traced collisions the planner's property was the hidden one: mints packaging, the scales' battery size), else the
        story twin (twin_key) by its own values, else by name; at most once per product; None when neither is there."""
        try:
            several = bool(self.needs.items())
            k = self.product_of(prop) if several else None
            done = self.__dict__.setdefault("twin_asks", set())
            if (several and k is None) or (k if several else -1) in done:
                return None
            product = self.needs.items()[k]["product"] if several else str(self.needs.reading.get("product") or "item")
            target = re.split(r"\s*\(", str(prop))[0].strip()
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
        """(v605f FC) a message of the questions log asked `prop` (strict_prop) for product k - its property's tag, else its
        asked text names the product (one product: any message) - answered or not."""
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
        """(v605f FC) the first '(or X)' alternative of a story hint of product k whose own property is `prop` (same_field or
        name_inside) that is a key of the product's records (prop_line), open (plan_open), not spent and not asked; ''."""
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
        """(v605f FC1, EC3 after a one-at-a-time reply) the property asked drew a one-at-a-time reply - its message named a second
        subject of the shopper's too, and the reply class is fixed by the subjects a message names: the '(or X)' alternative
        of the story hint whose property it is (alt_key) gives the next question's label, 'X (product)'; once per product
        (twin_asks, shared with distinct_ask and quiet_reask); '' when there is none."""
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
        """(v605f FC2) a question text that already writes every word of the plain form 'Which X would you like for the P?' (the
        property's and the product's words; stems): the plain form names the same subjects again, so it is no re-ask."""
        frame = {"which", "would", "you", "like", "for", "the"}
        have = {self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())}
        need = [self.stem(w) for w in re.findall(r"[a-z0-9]+", str(plain_q).lower()) if w not in frame and w not in LINK_WORDS]
        return bool(need) and all(w in have for w in need)

    def alt_untried(self, k: int) -> bool:
        """(v605f FC4) product k holds a story hint whose property was asked and whose '(or X)' alternative is still untried."""
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
        """EC4 (ii): after a no-preference or confused reply to a property a story hint maps (hint_mapped), the product's
        story twin (twin_key) as the next question's label, once per product (shared with distinct_ask), unless one of
        the product's hints reopened under its own alternative (EC3) or, with several products, another product had no
        message yet (coverage first); '' when there is none."""
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
        """EC7: once 4+ of a product's records were read, its question menu leaves out a key seen on only 1 of them."""
        if reads < 4 or not menu:
            return menu
        kept = [ln for ln in menu.splitlines() if not re.search(r"\(in [01] listings?\)", ln)]
        return "\n".join(kept) if kept else menu

    def first_reads(self) -> bool:
        """EC2 (MC6): before the first planned message, in a store without a list-read tool (pre_read covers one), each
        product whose category rows were seen but which has fewer than 2 records with facts read gets the top unread
        rows of its category read by the code (menu_rows: in stock, within the budget; at most 2 per product, the
        products with the fewest reads first), once, within the steps before the order's reserve (order_due() + 2 per
        message left + 8). True when the reads are queued."""
        try:
            if self.__dict__.get("first_reads_done") or (self.batch_tool() and self.ref_batch()) or not self.role("details"):
                return False      # (R9c: a list tool that shows only offer pages leaves the spec sheets to these reads)
            self.first_reads_done = True
            rows = self.needs.items()
            homes = self.home_shelves()
            qtys = [row["quantity"] for row in rows] or [self.needs.quantity()]
            room = self.moves_left() - self.order_due() - 2 * max(0, self.ask_limit() - self.msgs_sent) - 8
            wants: list[tuple[int, int, list[str]]] = []
            taken: list[str] = []
            for k in range(len(rows) or 1):
                home = homes[k] if k < len(homes) else set()
                have = self.product_reads(k if rows else None)
                want = 3 if room >= 3 * max(1, len(rows)) else 2      # (R9i: a third record per product when the steps allow)
                if home and have < want and room > 0:
                    picks = self.menu_rows(home, want - have, taken, qtys[min(k, len(qtys) - 1)])
                    taken += picks
                    if picks:
                        wants.append((have, k, picks))
            queue = [picks[n] for n in range(3) for _, _, picks in sorted(wants) if n < len(picks)][: max(0, room)]
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

    def plan_open(self, k: int | None, prop: str) -> bool:
        """open_prop for a planner's pick; when a story hint names the property too, a requirement of its name settles it
        only by a value among the field's sample values (the hint and the pick are two signals that it is still unknown)."""
        try:
            hinted = self.hinted_prop(k, prop)
            return self.open_prop(k, prop, self.prop_line(k, prop) if hinted else "", by_samples=hinted)
        except Exception:
            return self.open_prop(k, prop)

    def settles(self, req: str, samples: str = "") -> bool:
        """A requirement in force settles its property for the questions only by a positive stated value: a reference does;
        an exclusion or a description (positive_value), or a value neither the request nor the shopper's lines write
        (story_guess), does not; with `samples` (the field's sample values), neither does a word value none of whose words
        they carry."""
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

    # ----------------------------------------------------------------------------------------------- tools on offer
    def order_due(self) -> int:
        """Steps kept back for the first order attempt and the repair a refusal needs. CF: a third of the errand's steps
        (at most 16: 13 at 40 steps, 12 at 37), never below the base rule (a quarter, at least FIRST_ORDER_BY and at most
        FIRST_ORDER_DUE: smaller worlds keep what they had) nor above half the steps; once a shopper change, a refusal or a
        sold-out record was seen this errand, at least 8 plus one step per product (a repair needs a re-read, a put and the
        order per line). An early first finish leaves room to repair a refusal."""
        base = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, self.step_cap // 4 + 1))
        due = max(base, min(16, self.step_cap // 3))
        # (v605d M7: a refusal, a change or a sold-out CART LINE - static sold-out search rows are no event)
        if self.__dict__.get("event_seen") or (self.__dict__.get("dead") and set(self.trolley.units) & self.__dict__["dead"]):
            try:
                due = max(due, 8 + max(1, len(self.needs.items())))
            except Exception:
                due = max(due, 9)
        return max(base, min(due, self.step_cap // 2))

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
            name_set = {call_rank, decl, trolley, self.role("cart_view"), self.role("view") or self.role("details"), self.stock_tool()}
            if spare > 3 and self.swap_open():
                name_set.add(self.role("details"))     # (CF7: the sold-out note's unread row may be read once)
            # (a check that names listings to open, nothing verified yet, or a widening change: a few late reads stay)
            late = self.__dict__.get("late_n", 0) < 3
            if late and ((self.unverified() and spare > 5) or (self.named_open() and spare >= min(FIRST_ORDER_BY + 2, self.order_due()))):
                name_set.add(self.role("details"))
            if late and (self.widened_late() or self.__dict__.get("named_pages") or (self.unverified() and spare > 5)) and spare > 4:
                name_set.add(self.role("search"))
            if spare >= 7 and self.backup_rows():
                name_set.add(self.role("details"))     # (ED3: the cheapest backups' two reads)
        toolset = self.toolset
        if name_set is not None and call_rank and trolley:
            trimmed = [tdef for tdef in self.toolset if tdef["function"]["name"] in name_set]
            toolset = trimmed or toolset
        if self.role("message"):
            # (R9b: the code sends every message - the planned questions and the basket statement - so the one kept for the
            # statement is never spent by the actor)
            toolset = [tdef for tdef in toolset if tdef["function"]["name"] != self.role("message")]
        return toolset

    def in_reserve(self) -> bool:
        """The steps kept for the first order (or for the repair after a refusal) have begun."""
        spare = self.moves_left()
        return bool(self.role("order") and self.role("cart")) and spare > 3 and (
            (spare <= self.order_due() and self.orders_tried == 0) or (spare <= FIRST_ORDER_BY and self.refusal_n > 0))

    def unverified(self) -> bool:
        """No listing the item check passed under the current requirements (outside a refused cart); in a several-product
        request, also while some product with a known category has none of its own."""
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
        """Listings a check named that were seen, are not read yet and were not in a refused cart."""
        return {i for i in self.__dict__.get("named", set()) if i in self.archive.ids} - self.archive.read - self.refused_lines

    def widened_late(self) -> bool:
        """The newest change widened the acceptable set, and it came within the steps the order keeps."""
        return self.needs.widened_at == self.needs.version and self.wire.tick - self.needs.widened_tick <= self.order_due()

    def begin_batch(self) -> None:
        self.late_q = 0
        self.late_ids = set()

    def late_hold(self, call_name: str, call_args: dict[str, Any]) -> str:
        """In the reserve, a details read of a cart line or a verified fit (the stock re-check) always goes; any other read goes only while
        fewer than 3 late reads were made or queued in this reply, and only for a listing a check named, while nothing is
        verified, or (a search) right after a widening change."""
        details, search = self.role("details"), self.role("search")
        if call_name not in (details, search) and call_name in self.lookup_tools() and details:
            # (CG5: any other single-listing lookup counts in the reserve as a details read does - the stock re-read of a cart
            # line or a verified fit always goes; absorb tallies details and search reads only, so this one is tallied here)
            # (v605d M8: absorb tallies it once it ran - the details screen above queues it for this reply)
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
            # (ED3: beyond the 3 late reads, the cheapest unread rows below the cheapest verified fit - 2 reads in all)
            backups = [i for i in self.backup_rows() if i not in self.__dict__.get("late_ids", set())][: 2 - self.__dict__.get("backup_n", 0)]
            if named & set(backups):
                allowed = True
                self.backup_n = self.__dict__.get("backup_n", 0) + 1
        opened = self.swap_open() if not allowed and call_name == details else ""
        if opened and opened in named and spare > 2:
            allowed = True                # (CF7: the cheapest unread row a sold-out note listed - one details read)
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
        """ED3 (v509c: the next-cheapest is verified before checkout): under a cheapest rule, once a sold-out record or event
        was seen (event_seen, dead) or a check named listings still unread, the cheapest unread in-stock rows of each
        product's category priced below its cheapest verified fit (ed_pool), cheapest first - the reserve keeps 2 details
        reads beyond its 3 late ones for them. [] once those 2 were used."""
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

    # ----------------------------------------------------------------------------------------------- checks
    def field_names(self) -> str:
        """The property field names on the listing details read so far, spellings that differ only by separators
        merged, with how many listings carry each (the main field of a property is the one most listings carry)."""
        prop_list, _ = self.archive.prop_menu()
        if not prop_list:
            return ""
        return ("\n\nFIELD NAMES ON THIS STORE'S LISTINGS (separator-only spellings merged; how many listings read carry each):\n"
                + prop_list)

    def check_item(self, thing: str, aim_text: str, second: str = "") -> str:
        if self.wire.spare() < CLOSING + 20:
            return ""
        key = f"{thing}|{self.needs.version}"
        if key in self.__dict__.get("near_ok", set()):
            return ""     # (ED6: a meaning check confirmed its wording of each value it does not write, under these requirements)
        refs = [str(r) for r in self.needs.reading.get("references") or [] if r] if isinstance(self.needs.reading.get("references"), list) else []
        tracked = list(self.__dict__.get("refs") or [])
        own = self.own_redirects(thing)       # (CE6: the check runs again once such notes appear)
        memo = key + (f"|{len(self.archive.read)}" if refs or any(r.get("status") != "resolved" for r in tracked) else "") + ("|second" if second else "") \
            + (f"|own{len(own)}" if own else "")
        if memo in self.rulings:
            return self.rulings[memo]
        pointer = self.ref_pointer_note(thing)
        if pointer:
            # (CE5: under differ a listing of the reference's chain carries the excluded value by definition - no model call)
            ruling = f"An independent check says {thing} does NOT meet every requirement: {pointer}. Choose another listing."
            self.rulings[memo] = ruling
            self.log.append({"kind": "verify", "item": thing, "result": {"item": thing, "fits": False, "problems": ["reference chain"]}})
            return ruling
        archive = self.archive.about(thing)
        if not archive:
            return ""
        evidence = self.field_view(thing)
        if evidence["fails"]:
            # (CH2: the store writes the value itself in the field of the property's own name and this listing's value
            # shares nothing with it, or joins it with a value the field carries alone - no model call)
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
                f"{prop} from, refer to, go by or look to another listing), the grader takes {prop} from that listing, so this item's own "
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
        """CH: code_rows, kept until a record arrives."""
        memo = self.__dict__.get("ch_rows_memo")
        if memo and memo[0] == len(self.archive.items):
            return memo[1]
        rows = self.code_rows()
        self.ch_rows_memo = (len(self.archive.items), rows)
        return rows

    def store_fields(self, label: Any = "") -> dict[str, list[str]]:
        """CJ: the property fields of the read listings of the category of the product `label` names (its "(product)" tag;
        a one-product request: its own category), each with the values they show; every read listing's when no category is
        known."""
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
        """CH: the listing's own notes may send this property to another listing (CE's own_redirects when present; else a
        record of it with redirect words and the property's name), so its own field may not decide."""
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
        """CH: the FIELD EVIDENCE of one listing (field_facts), kept per listing, records and requirements."""
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
        # (v605f FE: a meaning check confirmed this listing's wording of these values: no plain failure, and its lines say so)
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
        """CH: for each requirement in force on the listing's product that names what the item is (no limit with a figure,
        no reference), computed from the records of the read listings of its category: the store's field for it (the one
        named like the property, else the one whose values write the value), how many listings carry it and how many other
        listings write the value itself there, and what this listing shows. "lines" are those facts; "rules" the
        category's part alone; "fails" the plain failures the code decides alone - the field of the property's own name
        shares no word, abbreviation start or figure with a value other listings of the category write there, or its only
        matching value joins the value with another one the field also carries alone."""
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
        table: dict[str, list[tuple[str, str, list[str]]]] = {}      # field -> (listing, element, its words)
        for i, r in peers.items():
            for key, val in r["fields"].items():
                if str(val).strip() and not junk_key(key):
                    carry[key] = carry.get(key, 0) + 1
                    elems = ch_parts(val)
                    elems += [str(val).strip().lower()] if len(elems) > 1 else []      # (the whole list too: 'a, b' wanted together)
                    table.setdefault(key, []).extend((i, p, ch_words(p)) for p in elems)

        def clip(text: Any, n: int = 70) -> str:
            text = " ".join(str(text).split())
            return text if len(text) <= n else text[: n - 3] + "..."
        for req in self.needs.props_left()[:12]:
            head, _, raw = str(req).partition(":")
            value = " ".join(re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw).split())
            prop = re.split(r"\s*\(", head)[0].strip()
            if not prop or not value or ref_text(value) or re.search(r"as shown|same as|the page|sold by|redirect|refer", value.lower()):
                continue
            if re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
                # (a limit with a figure is compared as an amount; v605f FG: where 2+ read listings of the category carry a field
                # named for the property, a listing with no field sharing a word of that name does not show it: 'capacity' for a
                # volume; 'calories' for calories per serving may be the property under a shorter name)
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
                continue        # (an amount with its unit word, '15 minutes': compared as an amount, not by its words)
            j = self.product_of(head) if several else None
            if several and j is not None and j != k:
                continue
            positive = positive_value(value, head)
            if not positive and not re.search(EXCLUDE_WORDS, value.lower()):
                continue        # (a description: no field holds it)
            alts = ch_alts(value)
            if not alts:
                continue
            named = sorted((key for key in carry if strict_prop(key, prop)), key=lambda x: -carry[x])
            by_name = bool(named)
            if not positive and by_name:
                flat = " ".join(re.findall(r"[a-z0-9]+", value.lower()))
                if any(" ".join(words) == flat for key in named for _, _, words in table.get(key, [])):
                    positive = True          # (CJ2: the store writes this value itself in the property's field: 'not available')
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
                # (v605d M9: an exclusion - the listings writing the value it excludes, its negation word left out)
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
            # (no read listing writes it: listings of the category whose titles or texts name it show the store's own word)
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
                # (v605e F3: a field few read listings carry does not show yet how this store writes the value)
                out["rules"].append(base + " Its values seen: " + " | ".join(samples) + f"; none writes it, but too few read listings "
                                    f"carry this field to tell whether this store writes '{clip(value)}' itself: a value of the same "
                                    "meaning is not shown to meet it yet.")
            else:
                out["rules"].append(base + " Its values seen: " + " | ".join(samples) + "; none writes it, so a value of the same "
                                    "meaning may meet it.")
            mine = [(key, str(me["fields"][key])) for key in named if str(me["fields"].get(key, "")).strip()]
            shown = "; ".join(f"{key}: '{clip(v, 90)}'" for key, v in mine)
            # (EA2: this listing's key of the property's words after one leading qualifier word - 'pants length' for length -
            # is the same property's field here: while one writes or touches the value, no plain failure; the item check decides)
            own_words = prop_words(prop)
            qualified = [(key, str(v)) for key, v in me["fields"].items() if own_words and key not in named and str(v).strip()
                         and not junk_key(key) and len(prop_words(key)) == len(own_words) + 1 and prop_words(key)[1:] == own_words]
            qual_writes = [(key, v) for key, v in qualified if ch_writes(value, v)]
            qual_near = bool(qual_writes) or any(ch_touches(value, v) for _, v in qualified)
            qual_note = (f" Its {qual_writes[0][0]} '{clip(qual_writes[0][1], 60)}' writes it (a qualified key of the same property: "
                         "a field named for the property with one more leading word is the same property's field under its fuller "
                         "name), so it meets it."
                         if qual_writes else "")      # (v605f FB: decisive, not 'judge both fields')
            if self.ch_redirected(thing, prop):
                out["lines"].append(base + " This listing's own notes may send this property to another listing, so its own "
                                    "field may not decide.")
                continue
            if not positive:
                out["lines"].append(base + " This listing: " + (shown or "no such field") + " (an exclusion: the listing must not "
                                    "carry the excluded value).")
                continue
            # (review: never over a field of the listing's own that shares the property's name and does not write the value -
            # 'waist' for a waist type: that field may be the property's own under a shorter name, and the title loses to it)
            kin_words = {w for w in prop_words(prop) if len(w) >= 3 and w not in LINK_WORDS}
            kin = [key for key, v in me["fields"].items() if str(v).strip() and not junk_key(key) and set(prop_words(key)) & kin_words
                   and not ch_writes(value, v)]
            title_says = not by_name and not kin and ch_writes(value, me.get("title") or "")
            if not mine and qual_writes:
                # (v605f FB: no field of the plain name, and the qualified one writes the value)
                out["lines"].append(base + " This listing has no such field." + qual_note)
                continue
            if not mine and title_says:
                # (v605e F2: no field is named for the property and this listing has no stand-in field: its title counts)
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
                sides = ch_sides(part)
                if not sides or not any(ch_match(alts, ch_words(s)) for s in sides):
                    return []
                return [s for s in sides if not ch_match(alts, ch_words(s)) and " ".join(ch_words(s)) in alone]
            clean = [p for p in parts if ch_match(alts, ch_words(p)) and not knocked(p)]
            joined = [(p, knocked(p)) for p in parts if ch_match(alts, ch_words(p)) and knocked(p)]
            if clean:
                out["lines"].append(base + f" This listing: {shown} - it writes '{clip(value)}': it meets it.")
            elif qual_writes:
                # (v605f FB: the qualified field writes the value - the requirement is met whatever the plain-named field says)
                out["lines"].append(base + f" This listing: {shown}." + qual_note)
            elif title_says:
                # (v605e F2: the field stands in for the property only by its values - no field is named for it - so where it
                # does not write the value, the title decides: READ_RULES' title rule)
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
                # (v605e F3)
                out["lines"].append(base + f" This listing: {shown}. No read listing of this category writes '{clip(value)}' in "
                                    "this field (its values: " + " | ".join(samples) + f"), but only {on} read listing(s) carry it: too "
                                    f"few to tell whether this store writes '{clip(value)}' itself, so a value of the same meaning is "
                                    "not shown to meet it yet.")
            else:
                out["lines"].append(base + f" This listing: {shown}. No read listing of this category writes '{clip(value)}' in "
                                    "this field (its values: " + " | ".join(samples) + "), so a value of the same meaning may meet it.")
        return out

    def rank_check(self, thing: str) -> str:
        """For a cheapest or best request: listings in the records that may fit and beat the chosen one."""
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
        # (a line that was in the cart when a checkout was refused is taken as gone: it is never offered as better)
        dead = self.dead_lines()
        better = [str(x) for x in (answer or {}).get("better") or [] if isinstance(x, (str, int)) and str(x) != thing
                  and str(x) not in self.refused_lines and str(x) not in dead][:4]
        better = [x for x in better if not self.field_view(x)["fails"]]     # (CH2: the code's own field evidence rules it out)
        self.rank_better = list(better)       # (v605f FD)
        self.__dict__.setdefault("named", set()).update(better)
        if not better:
            return ""
        return (f"A rank check thinks these listings may also fit and beat {thing} on the {what}: " + ", ".join(better)
                + (f" ({answer.get('reason')})" if answer.get("reason") else "")
                + ". Check them (details, stock, price) before ordering; if each fails, order the current pick.")

    def review_decline(self, early: bool = False) -> str:
        """The decline audit; an early one (nothing verified late in the episode) does not count against the two
        audits a decline may trigger."""
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
        ask_text += self.ref_note()
        ask_text += "\n\nLISTINGS SEEN (one card each; DETAILS = its full record was read, ROW = seen only in search rows):\n" + "\n".join(cards)
        if fresh:
            ask_text += ("\n\nOPENED SINCE THE LAST CHECK (their DETAILS cards above are new: judge each of them first, field by field, "
                         "by the reading rules): " + ", ".join(fresh))
        ask_text += self.field_names()
        answer = self.wire.consult(ask_text, 6000, AUDITOR)
        self.log.append({"kind": "audit", "result": answer})
        conforms = [str(x) for x in (answer or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        conforms = [x for x in conforms if not self.field_view(x)["fails"]]     # (CH2)
        self.audit_fits = conforms
        unopened = [str(x) for x in (answer or {}).get("unopened") or [] if isinstance(x, (str, int))
                    and str(x) not in self.archive.read and str(x) in self.archive.ids][:6]
        shelves, cap, qty, hit_shelf = self.shelf_labels(), self.needs.spend_cap, self.needs.quantity(), self.__dict__.get("hit_shelf", {})
        hits = [i for i in sorted(self.__dict__.get("all_hits", []), key=lambda i: -self.__dict__.get("hit_score", {}).get(i, 0)) if i not in self.archive.read
                and not (shelves and hit_shelf.get(i) and hit_shelf.get(i) not in shelves)
                and not (is_number(cap) and is_number(self.archive.price(i)) and self.archive.price(i) * qty > cap)]
        try:
            slack = [i for i in self.slack_rows() if i not in self.archive.read]      # (ED5)
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
        """One scripted reply reading the listings in `queue` (in lists of ids with `batch`, else one call each of `reader`,
        by default the details tool); its responses ([] when no call could be built)."""
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
        """Sends the details reads a code check queued (scripted, one reply; in lists of ids when the store has a tool that
        reads several listings, checked on its first use), then, unless they were re-reads or reads for the question
        menu, the decline audit over the new records."""
        queue = self.__dict__.pop("open_queue", None)
        put_read = False
        if not queue and self.__dict__.get("put_queue") and not self.wire.done and not self.wire.broke:
            # (CB4: the code's own stock read of what was just put in the cart, once no other queued read waits)
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
            # (CF8: reads for the question menu, a product's quota or the cheapest walk never cut into the steps kept for the
            # order while a candidate exists - a cart line or a verified fit)
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
            # (CB3: a cart line that a stock-bearing record showed since the newest cart change is not read again; CB4: nor is
            # a listing the code queued after its put that has left the cart since)
            stale = set(self.stale_lines())
            queue = [i for i in queue if i in stale or (i not in self.trolley.units and not put_read)]
            if not queue:
                return False
        batch = (self.batch_tool() if purpose == "cart" else self.ref_batch()) if len(queue) >= 2 else None
        if batch and purpose == "cart" and not self.__dict__.get("batch_proven"):
            batch = None      # (the list tool's first, unproven use is never on the cart's own lines: they are read one by one)
        if batch and reader and not ((self.__dict__.get("tool_caps") or {}).get(batch[0]) or {}).get("stock"):
            batch = None      # (CB: a stock re-read goes through a list read only when its answers showed stock)
        responses = self.open_reads(queue, say, batch, reader)
        if batch and responses and not self.__dict__.get("batch_proven"):
            if self.batch_ok(self.__dict__.get("last_reads") or [], queue):
                self.batch_proven = True
            else:
                # (the list tool did not return the listings' records: it is not used again, and the ids are read one by one
                # - only as many as this queue stood for before list reads (none for the optional menu reads, 2 for quota and
                # walk opens, 3 otherwise), and only while the steps the order, the walk or a decline keeps stay free)
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
        """(v605f) what the newest record carrying the listing's stock says: True in stock, False sold out, None unknown."""
        state: bool | None = None

        def visit(d: dict[str, Any]) -> None:
            nonlocal state
            if self.row_mark(d) == ident:
                got = self.stock_state(d)
                if got is not None:
                    state = got
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return state

    def rank_verify(self, thing: str) -> str:
        """(v605f FD) under a cheapest rule, after a rank check named listings that beat `thing`: each named listing already read,
        not sold out as last read, not refused, within the budget for its units and with no plain field failure gets the item
        check at once (no step; at most 2), so that a pass counts for the cheaper-fit holds; returns cheaper_fit(thing)."""
        try:
            if self.needs.priority() != "cheapest":
                return ""
            ver = str(self.needs.version)
            cap = self.needs.spend_cap
            done = 0
            details = self.role("details")
            unread = [x for x in self.__dict__.get("rank_better") or [] if x in self.archive.ids and x not in self.archive.read
                      and x not in self.refused_lines and x not in self.dead_lines()][:2]
            if unread and details and self.moves_left() > len(unread) + 6 and self.wire.spare() > CLOSING + 40:
                # (R9e: listings the rank check named that nobody opened are read by the code, outside the chat, so the item
                # check can judge them - a cheaper fit then counts for the cheaper-fit holds and swap)
                for x in unread:
                    args = self.lookup_call(x, details)
                    if args:
                        self.quiet_call(details, args)
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

    # ----------------------------------------------------------------------------------------------- R9f: bundles
    def part_map(self) -> dict[int, int]:
        """R9f: accessory product -> the main product it must fit, from the reading's bundles (several products only)."""
        rows = self.needs.items()
        out: dict[int, int] = {}
        bundles = self.needs.reading.get("bundles") if isinstance(self.needs.reading.get("bundles"), list) else []
        for entry in [e for e in bundles if isinstance(e, dict)] if rows else []:
            part = self.product_of(f"x ({entry.get('part')})")
            main = self.product_of(f"x ({entry.get('main')})")
            if part is not None and main is not None and part != main:
                out[part] = main
        return out

    def lead_pick(self, main: int, extra: Any = ()) -> str | None:
        """R9f: the listing that stands for the main product: a line of the cart call being judged (`extra`), its cart line,
        else its cheapest listing the item check passed under the current requirements."""
        for line in list(extra or ()) + list(self.trolley.units):
            if self.product_index(line) == main:
                return line
        best: tuple[str, float] | None = None
        for key in self.passed:
            ident, _, version = key.rpartition("|")
            if version != str(self.needs.version) or ident in self.dead_lines() or self.product_index(ident) != main:
                continue
            cost = self.archive.price(ident)
            if best is None or (cost is not None and cost < best[1]):
                best = (ident, cost if cost is not None else float("inf"))
        if best:
            return best[0]
        codes = self.main_codes(main)
        hits = self.hunt_rows(main, codes) if codes else []
        return hits[0][1] if hits else None      # (R9f: before any check, the cheapest row naming the requested model)

    def listing_text(self, ident: str) -> str:
        """R9f: every text a listing's records and rows show (fields, property values, description, notes), lowercased."""
        row = self.ref_index().get(ident) or {}
        bits = [str(v) for v in row.values() if isinstance(v, str)]
        bits += [f"{k} {x}" for v in row.values() if isinstance(v, dict) for k, x in v.items() if isinstance(x, (str, int, float))]
        return " ".join(bits).lower()

    def fit_names(self, ident: str) -> list[str]:
        """R9f: the names an accessory must carry to fit listing `ident`: the codes its own text gives after a model, series
        or part word (a property of that name too), and the names after 'also sold as'."""
        row = self.ref_index().get(ident) or {}
        text = " ".join(str(v) for v in row.values() if isinstance(v, str))
        out = [m.group(1) for m in re.finditer(r"\b(?:model(?:\s+(?:no|number))?|series|part(?:\s+(?:no|number))?)\b\.?\s*[:#]?\s*"
                                               r"([A-Za-z]{1,6}[\s-]?\d[\w-]{0,10})", text, re.I)]
        out += [m.group(1) for m in re.finditer(r"\balso sold as\s+([^.;,\n]{3,24})", text, re.I)]
        for v in row.values():
            if isinstance(v, dict):
                out += [str(x) for k, x in v.items() if isinstance(x, str) and re.search(r"\bmodel|\bseries", str(k).lower())
                        and re.fullmatch(r"[A-Za-z]{1,6}[\s-]?\d[\w-]{0,10}", x.strip())]
        return list(dict.fromkeys(t.strip(" .") for t in out if 2 <= len(t.strip(" .")) <= 24))[:6]

    def main_codes(self, main: int) -> list[str]:
        """R9f: the model or series codes the request gives for a main product (the reading's bundles), code-shaped only."""
        out: list[str] = []
        for entry in self.needs.reading.get("bundles") or []:
            if isinstance(entry, dict) and self.product_of(f"x ({entry.get('main')})") == main:
                out += re.findall(r"[A-Za-z]{1,6}[\s-]?\d[\w-]{0,10}", str(entry.get("model") or ""))
        return list(dict.fromkeys(c.strip(" .-") for c in out if len(c.strip(" .-")) >= 3))[:3]

    def hunt_rows(self, k: int, toks: list[str]) -> list[tuple[float, str]]:
        """R9f: (price, id) of the rows seen on product k's categories whose own text names one of `toks`, in stock as last
        seen and within the budget for k's units, cheapest first."""
        homes = self.home_shelves()
        home = homes[k] if k < len(homes) else set()
        rows = self.needs.items()
        qty = rows[k]["quantity"] if k < len(rows) else 1
        cap = self.needs.spend_cap
        pats = [self.fit_regex(t) for t in toks if self.fit_regex(t)]
        out = []
        info = self.code_rows()
        label_of = self.archive.__dict__.get("label_of") or {}
        dead = self.dead_lines()
        for ident, row in info.items():
            shelf = row.get("shelf") or str(label_of.get(ident) or "").strip().lower()
            if not home or shelf not in home or ident in dead or ident in self.refused_lines or row.get("stock") is False:
                continue
            cost = row.get("price")
            if not is_number(cost) or (is_number(cap) and cost * qty > cap):
                continue
            text = self.listing_text(ident)
            if any(re.search(pt, text) for pt in pats):
                out.append((float(cost), ident))
        return sorted(out)

    def fit_step(self) -> bool:
        """R9f: one scripted step for a bundle (a main item and the accessories that must fit it), before the actor's turn.
        Main: its model code is often written only in descriptions, which searches do not index - the code reads its
        category's broadest search to the end, page by page, and opens the two cheapest rows that name the code.
        Accessories, once the main pick's names are known: a search of each series-shaped name with the accessory's
        category words (else the category's broadest search), read to the end, then its two cheapest rows naming a name
        are opened. True when a step went out."""
        parts = self.part_map()
        if not parts or not self.role("search") or self.wire.done or self.wire.broke or self.wire.spent_out() \
                or self.moves_left() <= self.order_due() + 6 or self.wire.spare() < CLOSING + 60:
            return False
        state = self.__dict__.setdefault("hunts", {})
        jobs: list[tuple[str, int, list[str]]] = [("main", m, self.main_codes(m)) for m in sorted(set(parts.values()))]
        for k, main in sorted(parts.items()):
            pick = self.lead_pick(main)
            names = [t for t in (self.fit_names(pick) if pick else []) if self.fit_regex(t)]
            names.sort(key=lambda t: 0 if re.fullmatch(r"[A-Za-z]{1,4}\d{1,3}", t) else 1)    # (series names first: titles carry them)
            jobs.append(("part", k, names))
        key_arg = self.page_arg()
        for purpose, k, toks in jobs:
            if not toks:
                continue
            st = state.setdefault((purpose, k), {"tried": [], "pages": 0, "opened": False, "walks": []})
            if st["opened"]:
                continue
            label = self.ask_name(k, self.needs.items()[k]["product"])
            kind = " ".join(kind_tokens(label)[:3]) or label
            tpl = self.search_template()
            # (1) a search of each name with the category's words (an accessory's series name is in its title)
            if purpose == "part" and tpl:
                for tok in toks[:2]:
                    query = f"{tok} {kind}".strip()
                    if query.lower() in st["tried"]:
                        continue
                    st["tried"].append(query.lower())
                    args = {**tpl[0], tpl[1]: query}
                    st["walks"].append(json.dumps({a: v for a, v in args.items() if not key_arg or a != key_arg[0]}, sort_keys=True, default=str))
                    if self.walk_call(args, f"The {kind} must fit the main item: I search its name {tok}."):
                        return True
            # (2) the walks to read to their end: the name searches; for a main item (or names no search found), the category's
            # own search by its label, else the broadest search that showed its rows
            book = self.__dict__.get("page_book", {})
            homes = self.home_shelves()
            home = homes[k] if k < len(homes) else set()

            def walk_of(text: str) -> dict[str, Any] | None:
                return next((w for w in book.values() if " ".join(self.free_text(w["args"])).strip().lower() == text.strip().lower()
                             and not self.cg_listing(w["args"])), None)
            walks = [w for w in (walk_of(q) for q in st["tried"] if not q.startswith("label ")) if w and w.get("total")]
            if purpose == "main" or not walks or not self.hunt_rows(k, toks):
                if tpl and f"label {kind}" not in st["tried"]:
                    st["tried"].append(f"label {kind}")
                    if self.walk_call({**tpl[0], tpl[1]: label}, f"I list the {label} to find the one named {toks[0]}."):
                        return True
                own = walk_of(label)
                broad = sorted([w for w in book.values() if home and (w.get("shelves") or set()) & home and not self.cg_listing(w["args"])],
                               key=lambda w: -w.get("total", 0))
                walks = [w for w in ([own] if own and own.get("total") else []) + broad[:1] if w] or walks
            cap = self.pg(WALK_TOTAL) + 2
            for walk in walks[:2]:
                nxt = self.next_page(walk) if key_arg else None
                if nxt is not None and st["pages"] < cap and self.moves_left() > self.order_due() + 6:
                    st["pages"] += 1
                    if self.walk_call({**walk["args"], key_arg[0]: nxt}, f"I read the next page for {kind} naming {toks[0]}."):
                        return True
            if not walks:
                continue          # (no search to read yet: the hunt waits for one)
            # (3) the cheapest rows that name it are opened (spec sheets or records), once
            st["opened"] = True
            found = [i for _, i in self.hunt_rows(k, toks) if i not in self.archive.read][:3]
            self.log.append({"kind": "hunt", "purpose": purpose, "product": k, "names": toks, "pages": st["pages"], "open": found})
            if found and (self.role("details") or self.batch_tool()):
                self.open_queue = found
                self.open_recheck = True
                self.open_say = f"I open the cheapest {kind} that name {toks[0]}."
                self.open_purpose = "hunt"
                opened = self.open_named()
                return opened
        return False

    def fit_note_line(self) -> str:
        """R9f: the status line's note on accessories that must fit a main item (once the main pick's names are known)."""
        try:
            parts = self.part_map()
            rows = self.needs.items()
            notes = []
            for k, main in sorted(parts.items()):
                pick = self.lead_pick(main)
                toks = self.fit_names(pick) if pick else []
                if toks and k < len(rows):
                    notes.append(f"the {rows[k]['product']} must name one of {', '.join(toks)} (the names of the main item {pick}) "
                                 "in its title or text")
            for main in sorted(set(parts.values())):
                codes = self.main_codes(main)
                hits = self.hunt_rows(main, codes)[:4] if codes else []
                if hits and main < len(rows):
                    notes.append(f"the {rows[main]['product']}'s model {', '.join(codes)} is written in the text of " + ", ".join(
                        f"{i} ({c:g})" for c, i in hits) + " (cheapest first; check their other requirements)")
            return ("\n[fits the main item] " + "; ".join(notes) + ".") if notes else ""
        except Exception:
            return ""

    @staticmethod
    def fit_regex(token: str) -> str:
        """R9f: a token as a whole name in lowercased text, its letter and figure runs joined by any separator ('ab2' in
        '[ab2-12345]', never in 'ab20' or 'abx2')."""
        runs = re.findall(r"[a-z]+|\d+", str(token).lower())
        return r"(?<![a-z0-9])" + r"[\s_-]*".join(map(re.escape, runs)) + r"(?![0-9])" if runs else ""

    def fits_lead(self, ident: str, extra: Any = ()) -> bool | None:
        """R9f: whether an accessory listing fits the main item its product goes with - True when its own text names one of
        the main pick's tie tokens, False when it names none; None for a listing of no accessory product, or while the
        main pick or its tokens are unknown."""
        try:
            parts = self.part_map()
            k = self.product_index(ident) if parts else None
            if k is None or k not in parts:
                return None
            main = self.lead_pick(parts[k], extra)
            toks = [t for t in (self.fit_names(main) if main else []) if self.fit_regex(t)]
            if not toks:
                return None
            text = self.listing_text(ident)
            return any(re.search(self.fit_regex(t), text) for t in toks)
        except Exception as err:
            self.log.append({"kind": "fits_main_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return None

    def fit_hold_text(self, lines: list[str]) -> str:
        """R9f: a hold for accessory lines that do not fit the main item (twice per line and main pick), else ''."""
        told = self.__dict__.setdefault("fit_told", {})
        for line in lines:
            if self.fits_lead(line, lines) is not False:
                continue
            k = self.product_index(line)
            main = self.lead_pick(self.part_map()[k], lines) if k is not None else None
            if not main or told.get((line, main), 0) >= 2:
                continue
            told[(line, main)] = told.get((line, main), 0) + 1
            toks = self.fit_names(main)
            kind = self.needs.items()[k]["product"] if k is not None and k < len(self.needs.items()) else "accessory"
            return (f"Not run (no step used): {line} must fit the main item {main}, but its title, fields and texts name none of "
                    f"{main}'s model, series or other names ({', '.join(toks)}), so it does not fit. Choose a {kind} whose title or "
                    f"text names one of them (a short search such as '{toks[-1]} {kind}' lists them), the cheapest such one that "
                    "meets its other requirements.")
        return ""

    def cheaper_fit(self, thing: str) -> str:
        """Under a cheapest rule: the lowest-priced listing the item check passed under the current requirements that
        costs less than the chosen one (a line of a refused cart is taken as gone)."""
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
                continue      # (several products: only a listing of the same product can replace this one)
            cost = self.archive.price(ident)
            if cost is not None and cost < mine and (best is None or cost < best[1]):
                if self.field_view(ident)["fails"]:
                    continue      # (v605d M3, CH2: records read since its check show a plain failure of its own field)
                if self.fits_lead(ident) is False:
                    continue      # (R9f: an accessory that does not fit the main item is no fit)
                best = (ident, cost)
        return best[0] if best else ""

    def shelf_known(self, thing: str) -> str:
        """The category label of a listing, from any record that carries its id ('' when unknown)."""
        found = ""

        def visit(d: dict[str, Any]) -> None:
            nonlocal found
            if any(isinstance(val, str) and val == thing and "id" in str(key).lower() for key, val in d.items()):
                found = self.shelf_of(d) or found
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return found or str((self.archive.__dict__.get("label_of") or {}).get(thing) or "").strip().lower()     # (CC)

    def chosen(self) -> str | None:
        """The line the order would buy: the one booked line, else the newest booked one."""
        if self.trolley.units:
            return list(self.trolley.units)[-1]
        return None

    # ----------------------------------------------------------------------------------------------- products and lists
    def home_shelves(self) -> list[set[str]]:
        """For each product asked for (one entry for a one-product request): the category labels seen on listings (rows or
        records) whose kind words best match the product's (a word several products share counts half); an empty set when
        none matches."""
        memo = self.__dict__.get("home_memo")
        if memo and memo[0] == len(self.archive.items):
            return [set(h) for h in memo[1]]
        homes: list[set[str]] = []
        try:
            rows = self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}]
            seen: set[str] = set()

            def visit(d: dict[str, Any]) -> None:
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
                    score = kind_score(own, words, shared)      # (CC: 'battery' and 'batteries' alike)
                    if score > top:
                        best, top = {shelf}, score
                    elif score == top and score > 0:
                        best.add(shelf)
                homes.append(best)
        except Exception:
            homes = [set() for _ in (self.needs.items() or [0])]
        self.home_memo = (len(self.archive.items), [set(h) for h in homes])
        return homes

    def own_shelf(self) -> str:
        """For a one-product request: the product's own category label when listings of another category were read too
        (a referenced listing, a stray row); '' when only its own category was read or no label matches the product."""
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

    # ----------------------------------------------------------------------------------------------- CA joined categories
    def joined_phrase(self, a: str, b: str, label: str) -> tuple[str, bool] | None:
        """CA: (the request's own words for two products one store category joins, whether they are the category's own
        name): the category's name in order (either joiner: 'women's flats and heels'); else the two names joined by
        'and', '&', 'or', '/', '+' or a comma within four words with no figure between ('flats and heels') - this looser
        form only when the two names end in different kinds (a men's watch and a women's watch, an indoor and an outdoor
        light are two products of one kind) and the request does not name both again outside the joined phrase ('a table
        and a chair: the table in oak, the chair with armrests' describes two). None when the request writes neither
        (two products named apart, each with its own quantity, stay two)."""
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
                    continue      # (each is named again on its own: the request treats them as two)
                return hit.group(0).strip(), False
        return None

    def product_question(self, line: str, rows: list[dict[str, Any]]) -> bool:
        """CA: a logged question (questions_log) was about one of the products: its property label carries a tag of a
        product's kind, or the question names a product."""
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
        """CA: two products' requirements set one property to different values (the request then wrote each its own)."""
        def pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
            out = []
            for req in row.get("requirements") or []:
                head, _, value = str(req).partition(":")
                out.append((re.split(r"\s*\(", head)[0].strip(), " ".join(re.findall(r"[a-z0-9]+", value.lower()))))
            return out
        return any(same_prop(pa, pb) and va != vb for pa, va in pairs(a) for pb, vb in pairs(b))

    def merge_joined(self, obs: Any) -> str:
        """CA (F1): the reading split ONE store category named as two kinds joined ('A and B', 'A & B', 'A/B', 'A + B')
        into two products. The labels on the rows of each search are kept; while the cart was never used, no question
        about a product went out (product_question) and the shopper's replies set or freed no requirement, products i
        and j merge when a label splits at one joiner into sides A and B, both products' kind words lie inside the label,
        the words one has and the other lacks sit on different sides (each product touching its own side), their
        quantities are equal, their requirements do not set one property two ways, and the request writes the label's
        words in order or the two names joined (joined_phrase). Kind words are compared by same_kind. Returns the remark
        for the actor ('' when nothing merged)."""
        seen = self.__dict__.setdefault("joined_seen", {})

        def visit(d: dict[str, Any]) -> None:
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
            for low in sorted(seen):
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
        """CA: products i and j of the reading become the one product `phrase` (merge_joined): its requirements the union
        of theirs, their "(product)" tags rewritten everywhere (to the phrase while several products remain, dropped when
        one is left); the per-product caches (home_memo, menu_memo, quota_reads, walked: keyed by product or by its
        index) are dropped and the requirement version moves."""
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
            ts = kind_tokens(tag)
            return bool(ts) and kinds_within(ts, whole) and any(kind_in(w, ti + tj) for w in ts) \
                and not any(kinds_within(ts, o) and kinds_within(o, ts) for o in others)

        def retag(text: Any) -> Any:
            if not isinstance(text, str):
                return text
            head, sep, value = text.partition(":")

            def put(hit: Any) -> str:
                return (f" ({tag_now})" if tag_now else "") if owned(hit.group(1)) else hit.group(0)
            return re.sub(r"\s*\(([^()]*)\)", put, head) + sep + value

        def unique(items: list[Any]) -> list[Any]:
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

    # ----------------------------------------------------------------------------------------------- CB lookup tools and stock
    def lookup_key(self, tool: str | None) -> str | None:
        """CB: the one text argument of a single-listing lookup tool (its one required non-choice string, else its only
        one); None for a tool with other required arguments or none (searches, carts, list reads, views)."""
        jschema = self.shapes.get(tool or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        needed = [k for k in jschema.get("required") or [] if k in props] if isinstance(jschema.get("required"), list) else []
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
        key = next((k for k in texts if k in needed), None) or (texts[0] if len(texts) == 1 else None)
        return key if key and all(k == key for k in needed) else None

    def lookup_tools(self) -> list[str]:
        """CB: the store's single-listing lookup tools: neither the search, message, cart, cart view nor a finishing tool."""
        taken = {self.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        return [name for name in self.shapes if name not in taken and self.lookup_key(name)]

    def lookup_call(self, ident: str, tool: str | None) -> dict[str, Any] | None:
        """CB: arguments for reading one listing with `tool` (details_call for any lookup tool): an earlier call's
        arguments with the listing id swapped in, else the tool's one id text argument."""
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
        """CB: a record's listing id (a key named id, else one ending in id), when it is id-like text."""
        for exact in (True, False):
            for key, val in d.items():
                name = str(key).lower()
                if (name == "id" if exact else name.endswith("id")) and isinstance(val, str) and re.fullmatch(ID_LIKE, val):
                    return val
        return None

    @staticmethod
    def asked_ids(call_args: Any) -> set[str]:
        """CB: the id-like texts a call names (its text arguments and the texts of its list arguments)."""
        args = call_args if isinstance(call_args, dict) else {}
        asked = {v for v in args.values() if isinstance(v, str) and re.fullmatch(ID_LIKE, v)}
        asked |= {x for v in args.values() if isinstance(v, list) for x in v if isinstance(x, str) and re.fullmatch(ID_LIKE, x)}
        return asked

    def learn_caps(self, call_name: str, call_args: Any, obs: Any) -> list[str]:
        """CB: what each tool's answers carry, learned from use (tool_caps): per tool, the answers whose records hold the
        listing id the call named together with a stock field (stock_state), a property map (2+ scalar fields), notes or
        a price. Notes are a long free text (8+ words) the same listing's search row lacks or writes otherwise (with no
        row of it seen: under a key no row carries); their keys are kept per tool (note_keys), with every key a tool's
        answers carried (tool_keys). Each stock record is booked as a row (a listing the call did not name) or a lookup
        (one it named, or a cart reply); rows are learned to show the live stock (rows_live) once a row shows a listing's
        stock changed from what an earlier row of it showed, and to lag (rows_lag) once a row and a lookup of one listing
        disagree with no stock-relevant cart change between. dead: the listings whose newest stock record shows them
        sold out - a lookup's sold-out is cleared only by a later lookup. Returns the cart lines found sold out now."""
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
            self.__dict__.setdefault("read_with", {}).setdefault(name, set()).add(ident)      # (CE3: listings each tool read)
            # (CE3: the keys whose value was text or empty on a record the call named - where a notes field can be)
            self.__dict__.setdefault("text_keys", {}).setdefault(name, set()).update(
                str(k) for k, v in d.items() if v is None or isinstance(v, str))
            hit["stock"] = hit["stock"] or state is not None
            hit["props"] = hit["props"] or any(isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2
                                               for v in d.values())
            hit["price"] = hit["price"] or any(is_number(v) and re.search(r"price|cost", str(k).lower()) for k, v in d.items())
            keys.update(str(k) for k in d)
            row = row_text.get(ident)
            for k, v in d.items():
                titled = bool(re.search(r"title|name|headline|heading|label|caption", str(k).lower()))
                if not titled and isinstance(v, str) and len(v.split()) >= 8 and (
                        row.get(str(k)) != v if row is not None else str(k) not in row_keys):
                    # (R9c: a title written longer or shorter than on the search row is no note)
                    note_keys.add(str(k))
                    hit["notes"] = True
        if not (isinstance(obs, dict) and obs.get("error")):
            walk_dicts(obs, visit)
        for key, got in hit.items():
            caps[key] += int(got)
        return newly

    def stock_tool(self) -> str | None:
        """CB: the tool that re-reads a listing's stock: the single-listing lookup tool whose answers carried the listing's
        stock most often; before any did, the view tool, the code's own guess of it from the tools' descriptions, then
        the details tool - each only while it has not answered twice for a listing without a stock field; then a lookup
        tool that has not answered for a listing yet and that the guess gives no cart, message or finishing role (the
        read after a put probes it). None when no lookup tool is left that could show stock."""
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
        """CB: the lookup tool whose answers carry notes (learn_caps) under a key the details tool's answers never
        carried; None when no tool has shown such notes (the details tool shows them itself, or none were seen)."""
        caps = self.__dict__.get("tool_caps") or {}
        details = self.role("details")
        dkeys = (self.__dict__.get("tool_keys") or {}).get(details or "", set())
        nkeys = self.__dict__.get("note_keys") or {}
        shown = [t for t in self.lookup_tools() if t != details and set(nkeys.get(t) or set()) - set(dkeys)]
        return max(shown, key=lambda t: ((caps.get(t) or {}).get("notes", 0), t)) if shown else None

    def dead_lines(self) -> set[str]:
        """CB: listings whose newest record with a stock field shows them sold out."""
        return set(self.__dict__.get("dead") or set())

    def stale_lines(self) -> list[str]:
        """CB: cart lines with no stock-bearing record (CB3) since the newest stock-relevant cart change."""
        seqs = self.__dict__.get("line_seq") or {}
        mark = self.__dict__.get("cart_seq", 0)
        return [i for i in self.trolley.units if seqs.get(i, -1) < mark]

    def dead_swap(self, line: str) -> tuple[str, str]:
        """CB: (listing, '') to put in for a sold-out cart line: a listing the item check passed under the current
        requirements, of the line's category (with several products: known to be of it), not sold out, refused or in
        the cart, and keeping the cart's total (the other lines included) within the budget - the cheapest first; else
        ('', rows): the cheapest in-stock rows of its category that keep the total within the budget, at most three."""
        dead, ver = self.dead_lines(), str(self.needs.version)
        kind = self.shelf_known(line)
        several = len(self.needs.items()) >= 2
        cap, qty = self.needs.spend_cap, max(1.0, self.trolley.held(line))

        def cost(ident: str) -> float:
            price = self.archive.price(ident)
            return float(price) if is_number(price) else float("inf")
        rest = sum(cost(i) * self.trolley.held(i) for i in self.trolley.units if i != line and cost(i) != float("inf"))
        fits = []
        for key in self.passed:
            ident, _, version = key.rpartition("|")
            if version != ver or ident == line or ident in dead or ident in self.refused_lines or ident in self.trolley.units:
                continue
            if self.field_view(ident)["fails"]:
                continue      # (v605d M4, CH2: records read since its check show a plain failure of its own field)
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
        # (CF7: a row the item check failed, or a read row whose own record lacks or fails a stated value, is no replacement;
        # read rows showing every stated value come first, then unread rows, the cheapest first)
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
                continue      # (v605d M4, CE5: a pointer, or a listing showing the value a differ reference sets apart)
            read = ident in self.archive.read
            if read and self.field_view(ident)["fails"]:
                continue      # (v605d M4, CH2)
            if read and (any(self.code_clash(row["fields"], prop, plain) for prop, plain in judged)
                         or not all(self.code_shows(row, prop, plain, set(), True) for prop, plain in judged)):
                continue
            rows.append((0 if read else 1, price, ident))
        rows.sort()
        low = min((x for x in rows if x[0] == 1), key=lambda x: (x[1], x[2]), default=None)
        if low and self.needs.priority() == "cheapest" and all(low[1] < p for r, p, _ in rows if r == 0):
            rows.remove(low)      # (v605e F4: under a cheapest rule the cheapest unread row, below every listed read row, leads)
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
        """CF7: the next unread page of the search whose leading words carry a stated value of product k (the walk carrying
        the most), as a call's arguments; None when no such search has an unread page."""
        found = self.page_arg()
        if not found:
            return None
        walks = [w for w in self.__dict__.get("page_book", {}).values() if self.walk_carries(w["args"], k) and self.next_page(w) is not None]
        if not walks:
            return None
        best = max(walks, key=lambda w: (len(self.walk_carries(w["args"], k)), -len(w["read"])))
        return {**best["args"], found[0]: self.next_page(best)}

    def swap_open(self) -> str:
        """CF7: the cheapest unread row the newest sold-out note listed, while its one details read in the reserve is unused."""
        if self.__dict__.get("swap_done"):
            return ""
        dead = self.dead_lines()
        return next((i for i in self.__dict__.get("swap_unread") or [] if i not in self.archive.read and i not in dead
                     and i not in self.trolley.units), "")

    def dead_note(self, lines: list[str]) -> str:
        """CB: the remark for cart lines shown sold out, naming what to put in instead."""
        parts = []
        self.event_seen = True            # (CF1: a sold-out record was seen this errand)
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
        """CF4: right after a refused order the cart lines' stock is read at once by the code (the stock tool, as the cart's
        re-read, one scripted reply): a refusal often means a line sold out and its text need not name the line; the read is
        the re-read the next order needs anyway (CB3). Not while the code itself finishes (muted) or another read waits.
        ED3: a refused cart of one line whose refusal text the listener reads now as an item becoming unavailable - the
        needs' version moved while no requirement, revealed need or budget changed - marks that line sold out with no
        re-read (a later lookup still clears it); the listener's note leads the remark either way."""
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
        """ED3: (the listener's note on the refusal's text, the sold-out remark) for a refused one-line cart; ('', '') when
        the refusal carries no text, or the cart holds several lines (a store may sell only the first line)."""
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
        """CF4: the cart lines' stock re-read queued after a refusal ('' when it cannot go)."""
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
        """CF5: every requirement in force is a plain value the code's own evidence can judge (code_values, without a bound,
        a range or an approximation): then a read record showing every stated value is a fit by the code's evidence too."""
        items = self.needs.items()
        vals = [plain for k in (range(len(items)) if items else [None]) for _, plain in self.code_values(k)]
        plain = [v for v in vals if not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
                                                  v.lower())]
        # (EA4: a resolved differ reference's requirement, 'P: not V', is judged by the code too - its field and value are known)
        resolved = self.differ_done(None)
        judged = sum(1 for req in self.needs.props_left() if any(self.ref_match(ref, req) for ref in resolved))
        return len(plain) + judged >= len(list(self.needs.props_left()))

    def differ_done(self, k: int | None) -> list[dict[str, Any]]:
        """EA4: the resolved differ references still in force (a requirement in force is theirs), for product k (all of
        them when k is None or the reference names no product)."""
        rows, left, out = self.needs.items(), self.needs.props_left(), []
        for ref in self.__dict__.get("refs") or []:
            if ref.get("relation") != "differ" or ref.get("status") != "resolved" or not str(ref.get("value") or "").strip() \
                    or not self.ref_live(ref) or not any(self.ref_match(ref, req) for req in left):
                continue
            if rows and k is not None and ref.get("tag") and self.product_of(f"x ({ref['tag']})") not in (k, None):
                continue
            out.append(ref)
        return out

    def decline_repair(self) -> str:
        """CF5 (v605_plan 3.4): a decline after a refused order, or within the steps kept for the order, while the records hold
        a fitting candidate for every product - one the item check passed under the requirements now in force, or a read one
        whose own record shows every stated value (code_picks, code_fit) - is held once per refusal count: the code makes
        the cart hold exactly those picks (code_put: other lines and sold-out lines out, a line read in stock since the
        refusal back among the candidates; no search, no other pick), reads the stock of lines not read since, and the
        actor is told to order it. '' lets the decline go: TA6's rule stays the one
        authority (nothing that can fit - the decline), so a task whose right answer is a decline stays declinable."""
        several = len(self.needs.items()) >= 2
        early = several and self.refusal_n == 0 and self.moves_left() > self.order_due()     # (ED4: before the reserve too)
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
            # (ED4, MULTI_ITEM_PLAN 2.6: a product the records prove has no fit lets the decline go)
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
            # (a pick counts when the item check passed it now; the code's own evidence counts only when the code can see
            # every requirement in force - no bound, range or exclusion it cannot judge - and the pick's record was read)
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
                # (ED4: an unverified pick gets the item check first, as unrefuted does - a failing one drops out; a read
                # pick the code's own evidence already fits needs none; one whose own record (fields, else title) does not
                # show every plain value and every need the shopper revealed is passed over unchecked)
                unchecked = [i for i, _ in picks if not fitting([(i, 1)])]
                if not unchecked:
                    break
                bare = [i for i in unchecked if not seen.get(i) or not seen[i]["revealed_ok"] or seen[i]["misses"]]
                if bare:
                    avoid.update(bare)
                    # (v605e review: with them every unread listing just as bare - not passed, its row not showing every plain
                    # value or revealed need - goes at once: one by one, each round re-ran code_picks over every row)
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
            # (only the picks found here go in - no search and no other pick: the cart calls, then one stock read of the
            # lines no stock-bearing record showed since, while the order's step stays free)
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
        """ED: a value the code's own evidence can test word by word (no bound, range, '+' join or approximation)."""
        return not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
                             str(value).lower())

    @staticmethod
    def ed_word(value: str) -> bool:
        """ED6: a word value - letters, no figure, bound, exclusion or reference."""
        low = str(value).lower()
        return bool(re.search(r"[a-z]{3,}", low)) and not re.search(r"\d", low) and Errand.ed_plain(low) \
            and not re.search(EXCLUDE_WORDS, low) and not ref_text(low)

    def ed_reqs(self, k: int | None) -> list[tuple[str, str, bool]]:
        """ED6: (property, value, revealed) of each requirement in force on product k (all for one product)."""
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
        """ED (v605e group D): the listings seen on product k's category (one product: k None), from the records alone, as
        code_picks reads them: price, read, verified (the item check passed it under the requirements now in force), fit
        (verified, or read with its own record showing every plain stated or revealed value - code_shows - and no field of
        one clashing; never sold out, refused, ruled out by a check or its own CH field evidence, a CE5 pointer, or over
        the budget for its units), the requirements its record does not show (misses) and the ones it clashes on.
        Memoised per records, requirements, checks, stock and budget."""
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
            # (a value joined with '+' names parts that must all be there: a field of the property missing one clashes)
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
                # (a field that writes the value word for word never clashes - value_seen skips words of 2 letters)
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
        """ED4 (MC5, MULTI_ITEM_PLAN 2.6): the products the records prove have no fit - some listing of it was read and every
        read one fails a firm need on its own record (a value its field contradicts, a check's ruling or CH field
        evidence, a CE5 pointer) or is sold out, refused or over the budget for its units."""
        out = []
        for k in range(len(self.needs.items())):
            reads = [e for e in self.ed_pool(k) if e["read"]]
            if reads and all(e["out"] for e in reads):
                out.append(k)
        return out

    def slack_rows(self) -> list[str]:
        """ED5 (v509c: decline_ok false + open these rows, in code): several products, each with a fit in the records
        (ed_pool), but the cheapest fits together pass the budget: per product, the unread in-stock rows of its category
        priced at or below its slack - the budget less the other products' cheapest fits, per unit - the rows of searches
        whose leading words carry the most of its required values first (the all-words hits count one), then the
        cheapest; at most 3 in all, taken in turn from each product. [] otherwise."""
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
                    # (how many of the product's values the search's leading words carry - a word of 2 letters too)
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
        """v605e review (ED6): the problems the newest item check of `ident` named when it failed it; None when its newest
        check passed it or none is logged."""
        for entry in reversed(self.log):
            if isinstance(entry, dict) and entry.get("kind") == "verify" and entry.get("item") == ident:
                res = entry.get("result")
                if isinstance(res, dict) and res.get("fits") is False:
                    return [str(p) for p in res.get("problems") or [] if str(p).strip()]
                return None
        return None

    def ed_news(self, k: int | None) -> bool:
        """ED6: product k had news this errand - a refusal, or a requirement revealed or dropped for it (or for no product
        in particular)."""
        if self.refusal_n > 0:
            return True
        rows = self.needs.items()
        return any(not rows or self.product_of(str(req).partition(":")[0]) in (k, None)
                   for req in list(self.needs.uncovered) + list(self.needs.let_go))

    def near_hold(self) -> str:
        """(v605f FE) a decline while near_basket's meaning-confirmed basket is still in the cart (same requirements, no refusal
        since, no line sold out or refused): the code orders the cart itself (finish_now) instead; '' lets the decline go on."""
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
        """ED6 (v509c G5/V3a near_basket): once per errand, before a decline that follows news (a reveal, a drop or a
        refusal), after every other hold: per product with no fit, up to 6 (one product) or 3 read listings - in stock,
        within the budget, meeting every revealed need on their own fields (code_shows), whose every unshown requirement
        is a stated word value (no figure, bound, exclusion or reference) that a field of the property writes in words -
        go to ONE cheap meaning call; a basket every line of which is a fit or confirmed, within the budget together, is
        put in by the code (its confirmed listings count as verified: rulings cleared as arbitrate does) and the actor is
        told to order. '' lets the decline go."""
        if self.__dict__.get("near_done") or not self.role("cart") or not self.role("order") or self.moves_left() < 4 \
                or self.wire.spare() <= CLOSING + 30 or self.wire.mute or self.__dict__.get("covering") or self.wire.done \
                or self.wire.spent_out() or self.wire.no_model:
            return ""
        try:
            return self.near_basket()
        except Exception as err:
            self.log.append({"kind": "near_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def near_basket(self) -> str:
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
                    cands.append((len(lines), e["price"], e["id"], lines))      # (CH fails only on the lines' own properties)
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
        # (v605f FE: the basket stands - a later decline while it is in the cart is replaced by the code's own order)
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
        """CB3/CB6, after a cart call: the cart's version moves (cart_ver; the actor's calls since it are counted from 0);
        a listing new to the cart marks every line for a stock re-read (cart_seq); a line whose units grew needs its
        own; a removal or a smaller count needs none (taking a line out sells nothing out)."""
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
        """CB4: after a cart call - a reply that lists the lines makes a cart view right after it redundant (cart_shown);
        a listing put in for the first time has its stock read at once by the code (one scripted read with the stock
        tool): a listing can sell out right after it is put in, and the read is the pre-order re-read as well."""
        if self.trolley.view_units(obs) is not None:
            self.cart_shown = self.__dict__.get("cart_ver", 0)
        firsts = self.__dict__.setdefault("first_puts", set())
        new = [line for line, count in self.trolley.lines(call_args) if (count is None or count > 0) and line in self.trolley.units
               and line not in firsts]
        if not new or not self.trolley.is_put(call_args) or self.code_fault(obs):
            return
        firsts.update(new)
        stale = set(self.stale_lines())
        new = [line for line in new if line in stale]      # (a cart reply that showed a line's stock was its read)
        tool = self.stock_tool()
        if not new or not tool or self.wire.mute or self.__dict__.get("covering") or self.moves_left() < 4 or self.wire.spare() < CLOSING:
            return
        # (its own queue, apart from open_queue: the other code paths that queue reads never inherit its purpose or tool;
        # open_named runs it once no other queued read is waiting)
        queue = list(self.__dict__.get("put_queue") or [])
        self.put_queue = (queue + [line for line in new if line not in queue])[:4]
        self.put_tool = tool
        self.log.append({"kind": "after_put_read", "lines": list(self.put_queue), "tool": tool})

    def mark_fresh(self, obs: Any) -> None:
        """CB3: a cart reply that itself shows a line's stock (a record with the line's id and a stock field) is that line's
        stock read after this change."""
        if not self.__dict__.get("stock_seen"):
            return
        seqs = self.__dict__.setdefault("line_seq", {})

        def visit(d: dict[str, Any]) -> None:
            ident = self.rec_id(d)
            if ident in self.trolley.units and self.stock_state(d) is not None:
                seqs[ident] = self.__dict__.get("absorb_n", 0)
        walk_dicts(obs, visit)

    def cart_gone(self, lines: list[str]) -> set[str]:
        """CB5: before the code's own first order: the cart lines shown sold out, after one stock re-read (stock tool) of
        the lines not read since the newest cart change, when steps allow."""
        gone = set(lines) & self.dead_lines()
        stale = [i for i in self.stale_lines() if i in lines and i not in gone]
        if stale and self.stock_tool() and self.moves_left() >= len(stale) + 2:
            _, sold = self.code_recheck(stale)
            gone |= set(sold) | (set(lines) & self.dead_lines())
        return gone

    def batch_tool(self) -> tuple[str, str, int] | None:
        """(tool, argument, most ids) of a tool that reads a list of listings, found from its schema alone: one array
        argument of plain strings with an item limit of 2+, no other required argument, and no other role (search,
        message, cart, cart view, order, decline). None when the store has none, or once its first use failed."""
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
                    # (a read only: a tool whose name or description says it changes the cart or a list is never a list read)
                    if re.search(r"\b(?:remov|delet|cancel|clear|discard|releas|refund|purchas|checkout|wish|basket|cart|bag)", said.get(name, name).lower()):
                        taken.add(name)
                for name, jschema in self.shapes.items():
                    props = jschema.get("properties") if isinstance(jschema, dict) else None
                    props = props if isinstance(props, dict) else {}
                    arrays = [key for key, p in props.items() if isinstance(p, dict) and p.get("type") == "array"
                              and isinstance(p.get("items"), dict) and p["items"].get("type") == "string" and not p["items"].get("enum")]
                    needed = jschema.get("required") if isinstance(jschema, dict) and isinstance(jschema.get("required"), list) else []
                    most = props[arrays[0]].get("maxItems") if len(arrays) == 1 else None
                    if name not in taken and len(arrays) == 1 and not [r for r in needed if r not in arrays] and is_number(most) and most >= 2:
                        spec = (name, arrays[0], int(min(most, 5)))     # (R9c: list tools refuse more than 5 ids in practice)
                        break
            except Exception:
                spec = None
            self.batch_spec = spec
        return self.batch_spec

    @staticmethod
    def batch_ok(found: list[Any], ids: list[str]) -> bool:
        """A list read returned the listings' records: most of the asked ids sit in a record that carries a property map (or,
        in flat records, 5+ short text fields)."""
        got: set[str] = set()
        wanted = set(ids)

        def visit(d: dict[str, Any]) -> None:
            nested = any(isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2 for v in d.values())
            flat = not any(isinstance(v, dict) for v in d.values()) and sum(1 for v in d.values() if isinstance(v, str) and 0 < len(v) <= 70) >= 5
            if nested or flat:      # (a flat record of 5+ short fields counts too, as in rich_reads)
                got.update(v for v in d.values() if isinstance(v, str) and v in wanted)
        try:
            for obs in found:
                walk_dicts(obs, visit)
        except Exception:
            return False
        return bool(got) and 2 * len(got) >= len(wanted)

    @staticmethod
    def title_of(row: dict[str, Any]) -> str:
        if isinstance(row.get("title"), str):
            return row["title"]
        return next((v for key, v in row.items() if isinstance(v, str) and re.search(r"title|heading|headline", str(key).lower())), "")

    def menu_rows(self, home: set[str], most: int, taken: list[str], qty: int = 1) -> list[str]:
        """Up to `most` unread rows of the categories `home`: in stock, within the budget for `qty` units as last priced,
        not in a refused cart or `taken`; while no record read there shows every stated value of the product in its
        fields, first the first such row, in result order, of the search whose leading words carry the most of its values;
        then the all-words hits, then rows a search led by a required value showed (the likely fits), then the rest, and
        within each, rows whose titles share the fewest words with each other and with the titles read there (their
        fields cover more of the shelf)."""
        cap = self.needs.spend_cap
        order: list[str] = []
        titles: dict[str, str] = {}
        sold: set[str] = set()
        known: list[set[str]] = []

        def visit(d: dict[str, Any]) -> None:
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
            # (no likely fit read yet: the search carrying the product's values shows its likeliest fits first, and its first
            # such row comes before the picks for title variety)
            several = bool(self.needs.items())
            k = next((n for n, h in enumerate(self.home_shelves()) if h == home), None) if several else None
            vals = [re.sub(r"\((?:stated|revealed[^)]*)\)", "", r.partition(":")[2]) for r in self.needs.props_left()
                    if not several or self.product_of(r.partition(":")[0]) in (k, None)]
            vals = [v for v in vals if v.strip() and positive_value(v) and not re.search(r"as shown|same as|the page|sold by|redirect|refer", v.lower())]
            facts: dict[str, list[str]] = {}

            def fact_visit(d: dict[str, Any]) -> None:
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
        """Before the first planned question, with a tool that reads a list of listings: one list read of the unread
        rows of each product whose own category has too few records read (2 rows for a product with none, 4 for a lone
        product with none, up to 4 records in all for one with fewer), so the question menu holds each product's own
        fields; then, once more while over 25 steps remain, one list read topping up to 4 each product whose category
        still has fewer than 3 records with facts (its first reads were thin). True when the reads are queued."""
        try:
            batch = self.ref_batch()      # (R9c: a list of offer pages adds no fields to the menu: first_reads reads spec sheets)
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
                want = 4 if self.needs.items() else 6      # (R9i: more records, a better field count - live top-4 51% at 4, 57% at 8)
                if have == 0:
                    empty += self.menu_rows(home, want, empty, qtys[min(k, len(qtys) - 1)])
                elif have < want:
                    thin.append((k, want - have))
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
        """In a several-product request: unread rows of each product with no verified fit (no listing the item check passed,
        and none the last decline audit named as a possible fit, on its own category) - in stock and within the budget for
        its units as last seen; the all-words hits first, then the newest rows, taken in turn from each such product - as
        many as one list read takes (else 3)."""
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

    def product_of(self, text: str) -> int | None:
        """Which requested product a property label or question names: the one its first "(product)" tag names, else the
        one the whole text names, by shared kind words (a word several products share counts half); None when none is
        named or two tie (a paired question naming both products is about neither alone)."""
        rows = self.needs.items()
        if not rows:
            return None
        names = [set(kind_tokens(row["product"])) for row in rows]
        for piece in re.findall(r"\(([^)]*)\)", str(text))[:1] + [str(text)]:
            words = set(kind_tokens(piece))
            scores = []
            for k, own in enumerate(names):
                shared = set().union(*(names[j] for j in range(len(names)) if j != k))
                scores.append(kind_score(own, words, shared))      # (CC: singular and plural alike)
            top = max(scores)
            if top > 0 and scores.count(top) == 1:
                return scores.index(top)
        return None

    def plan_product(self, qitem: dict[str, Any]) -> int | None:
        """The product a planned question is about: its property's "(product)" tag, else its question's words."""
        k = self.product_of(str(qitem.get("property") or ""))
        return k if k is not None else self.product_of(str(qitem.get("question") or ""))

    def field_rank(self, k: int | None, counts: bool = False) -> list[Any]:
        """R9i: product k's properties, most listings first - the question menu's fields of its own category with their
        counts, names that are one field merged (same_prop: the larger count stands), what is not a product property left
        out. Live Races 5-8: the hidden property was among the four most common fields 51% of the time from 4 records,
        57% from 8, 64% from the whole shelf."""
        rows = self.needs.items()
        if rows and k is not None:
            homes = self.home_shelves()
            home = sorted(homes[k], key=lambda s: -self.archive.rich_reads(s)) if k < len(homes) and homes[k] else []
            shelf = home[0] if home else ""
        else:
            shelf = self.own_shelf()
        menu, sets = self.archive.prop_menu(shelf) if shelf or not rows else ("", 0)
        self.rank_sets = sets
        merged: list[list[Any]] = []
        for name, count in re.findall(r"^- (.+?) \(in (\d+) listings\)", menu, re.M):
            name = name.strip()
            if not name or junk_key(name) or re.search(r"\b(?:condition|options?|variants?|models?|warranty|years?|asin|upc|ean|"
                                                       r"includes?|package|certifications?)\b", name.lower()):
                continue          # (bookkeeping and logistics fields: never a hidden need's own property in the live races)
            twin = next((m for m in merged if same_prop(m[0], name)), None)
            if twin:
                twin[1] = max(twin[1], int(count))
            else:
                merged.append([name, int(count)])
        merged.sort(key=lambda m: -m[1])
        return [(m[0], m[1]) for m in merged] if counts else [m[0] for m in merged]

    def written_edit(self, k: int | None, drop: str = "", add: tuple[str, str] | None = None) -> None:
        """R9k: takes a stated requirement out of (drop), or puts one into (add: the requirement and its product entry's
        form), every list the plan keeps: the stated list, the rebuilt list, the reading's list and product k's entry."""
        nd = self.needs
        lists = [nd.written] + ([nd.now_reqs] if nd.now_reqs is not None else []) + (
            [nd.reading["requirements"]] if isinstance(nd.reading.get("requirements"), list) else [])
        rows = [r for r in nd.reading.get("items") or [] if isinstance(r, dict) and isinstance(r.get("requirements"), list)]
        row = rows[k] if k is not None and k < len(rows) else (rows[0] if len(rows) == 1 else None)
        if drop:
            for lst in lists:
                while drop in lst:
                    lst.remove(drop)
            head, _, value = drop.partition(":")
            prop, val = re.split(r"\s*\(", head)[0].strip().lower(), value.strip().lower()
            if row is not None:
                row["requirements"] = [x for x in row["requirements"] if not (re.split(r"\s*\(", str(x).partition(":")[0])[0].strip().lower() == prop
                                                                             and str(x).partition(":")[2].strip().lower() == val)]
        if add:
            for lst in lists:
                if add[0] not in lst:
                    lst.append(add[0])
            if row is not None and add[1] not in row["requirements"]:
                row["requirements"].append(add[1])
        nd.version += 1

    def name_guards(self) -> None:
        """R9k: once a product's category is known and two of its records were read (once per product): (a) a stated
        requirement whose value words are all words of the category's own label is dropped - the category covers it (a
        shelf named after a style gives no requirement of that style); (b) an accessory's requirement that it works with or fits the main
        item (its value naming the main item's kind) is dropped - the fit check reads the main item's names; (c) words of
        the request's name for the product that the category label lacks, and that a field of the category's records
        carries as a value, become a requirement of that field (a product named with a type word its shelf's name
        lacks: the field whose values write that word)."""
        rows = self.needs.items()
        state = self.__dict__.setdefault("guard_state", {})
        homes = self.home_shelves()
        parts = self.part_map() if rows else {}
        info: dict[str, dict[str, Any]] | None = None
        for k in range(len(rows) or 1):
            kk = None if not rows else k
            home = homes[k] if k < len(homes) else set()
            reads = sum(self.archive.rich_reads(s) for s in home) if home else 0
            if not rows and not reads:
                read = self.archive.shelves_read()
                home = {read[0].strip().lower()} if read else set()     # (one product whose matched category has no record
                # read - its listings sit on a category its name shares no word with: the category of the records read most)
                reads = sum(self.archive.rich_reads(s) for s in home) if home else 0
            st = state.setdefault(k, {"first": False, "reads": 0, "tries": 0, "placed": False})
            if not home or reads < 2 or (st["first"] and (st["placed"] or st["tries"] >= 4 or reads <= st["reads"])):
                continue
            st["reads"], st["tries"] = reads, st["tries"] + 1
            first, st["first"] = not st["first"], True
            label = max(sorted(home), key=lambda s: self.archive.rich_reads(s))
            lwords = kind_words(label)
            phrase = rows[k]["product"] if rows else str(self.needs.reading.get("product") or "")
            mine = [r for r in self.needs.written if not rows or self.product_of(str(r).partition(":")[0]) == k]
            main_words = kind_words(rows[parts[k]]["product"]) if rows and k in parts else set()
            for r in (mine if first else []):
                head, _, value = str(r).partition(":")
                vwords = kind_words(value)
                covered = bool(vwords) and vwords <= lwords and not re.search(r"\d", value)
                tied = bool(main_words) and bool(vwords & main_words) and bool(
                    re.search(r"compatib|\bworks?\b|working|\bfits?\b|fitting|goes with|suits?\b", (head + " " + value).lower()))
                if covered or tied:
                    self.written_edit(kk, drop=str(r))
                    self.log.append({"kind": "name_guard_drop", "req": str(r)[:200], "label": label, "why": "category" if covered else "fit"})
            have = kind_words(" ".join(str(r) for r in self.needs.written if not rows or self.product_of(str(r).partition(":")[0]) == k))
            ltoks = kind_tokens(label)
            extra = [w for w in kind_tokens(phrase) if not kind_in(w, ltoks) and w.rstrip("s") not in lwords
                     and w.rstrip("s") not in have]      # (singular and plural spellings of a label word are the label's)
            if not extra:
                st["placed"] = True
                continue
            info = info if info is not None else self.code_rows()
            recs = [row for row in info.values() if row.get("shelf") in home and row.get("fields")]

            whole = [w.rstrip("s") for w in kind_tokens(phrase)]
            full_hits: dict[str, int] = {}
            tight: dict[str, float] = {}

            def carriers(words: list[str]) -> dict[str, int]:
                cnt: dict[str, int] = {}
                for row in recs:
                    for f, v in row["fields"].items():
                        name = re.sub(r"[_.]+", " ", str(f)).strip().lower()
                        if junk_key(name) or re.search(r"\b(?:condition|options?|variants?|models?|warranty|title|name)\b", name):
                            continue
                        vals = [x.rstrip("s") for x in re.findall(r"[a-z0-9]+", str(v).lower())]
                        if all(w.rstrip("s") in vals for w in words):
                            cnt[name] = cnt.get(name, 0) + 1
                            tight[name] = max(tight.get(name, 0.0), len(words) / max(1, len(vals)))     # (the words ARE the value)
                            if all(w in vals for w in whole):
                                full_hits[name] = full_hits.get(name, 0) + 1     # (the value writes the whole product name)
                return cnt
            used: set[str] = set()
            for words in [extra] + ([[w] for w in extra] if len(extra) > 1 else []):
                if any(w in used for w in words):
                    continue
                cnt = carriers(words)
                if not cnt:
                    continue
                field = max(cnt, key=lambda f: (round(tight.get(f, 0.0), 2), cnt[f], len(f)))      # (a field whose value is
                # those words before one that lists them among others; then the most records; then the more specific name)
                if any(same_prop(field, re.split(r"\s*\(", str(r).partition(":")[0])[0]) for r in self.needs.written
                       if not rows or self.product_of(str(r).partition(":")[0]) == k):
                    continue      # (a requirement of that field is in force already)
                value = " ".join(kind_tokens(phrase)) if full_hits.get(field) else " ".join(words)
                req = f"{field} ({rows[k]['product']}): {value}" if rows else f"{field}: {value}"
                self.written_edit(kk, add=(req, f"{field}: {value}"))
                used.update(words)
                st["placed"] = True
                self.log.append({"kind": "name_guard_add", "req": req, "label": label, "records": cnt[field]})

    def seek_products(self, lacking: list[int]) -> bool:
        """R9j: before the first question, a product the store was never searched for is searched by the code (its name from
        the request, once), and one whose category is known but none of whose listings was read gets its first records read
        (the menu reads run again for it, once); True when a step went out."""
        stage = self.__dict__.setdefault("seek_stage", {})
        rows = self.needs.items()
        homes = self.home_shelves()
        tpl = self.search_template()
        for k in lacking:
            if k >= len(rows):
                continue
            known = k < len(homes) and bool(homes[k])
            if not known and stage.get(k, 0) < 1 and tpl:
                stage[k] = 1
                phrase = rows[k]["product"]
                self.log.append({"kind": "seek", "product": k, "query": phrase})
                if self.walk_call({**tpl[0], tpl[1]: phrase}, f"I look up the {phrase} first, so every product gets its questions."):
                    return True
            if known and stage.get(k, 0) < 2:
                stage[k] = 2
                self.first_reads_done = False
                self.pre_read_done = False
                if self.pre_read() or self.first_reads():
                    return True
        return False

    def coverage_first(self, scripted: list[dict[str, Any]], ok: Any) -> list[dict[str, Any]]:
        """R9i (veto): the planner's question stands unless its field is rare on the product's records read (under a quarter
        of them, with 3 or more read) and no story hint points at it; then the product's most common open field goes
        instead. Live Races 5-8: a hidden need's field sat on a median 75% of its shelf's records and on under a quarter of
        4 sampled records only 5% of the time; the planner's misses on the pack were rare fields,
        its hits often common but not the most common ones."""
        rows = self.needs.items()
        first = scripted[0] if scripted else None
        prop = str((first or {}).get("property") or "").strip()
        if rows:
            k = self.product_of(prop) if prop else None
            if k is None:
                return scripted
            product = rows[k]["product"]
        else:
            k, product = None, str(self.needs.reading.get("product") or "item")
        name = re.split(r"\s*\(", prop)[0].strip()

        def label(p: str) -> str:
            return f"{p} ({product})" if rows else p

        def stated(p: str) -> bool:
            return any(same_prop(re.split(r"\s*\(", str(r).partition(":")[0])[0], p) and (not rows or self.product_of(str(r).partition(":")[0]) in (k, None))
                       for r in self.needs.props_left())
        every = self.field_rank(k, True)
        sets = int(self.__dict__.get("rank_sets") or 0)
        ranked = [p for p, _ in every if (ok is None or ok(label(p))) and not stated(p)]
        if not ranked or sets < 3:
            return scripted
        if name and (self.hint_mapped(k, name) or next((c for p, c in every if same_prop(name, p)), 0) * 4 >= sets):
            return scripted       # (a story-hinted or common field: the planner's pick stands)
        pick = {"property": label(ranked[0]), "second": "", "question": f"Which {ranked[0]} would you like for the {product}?"}
        self.log.append({"kind": "coverage_veto", "planner": prop, "pick": ranked[0], "records": sets})
        return [pick] + scripted

    def ask_name(self, k: int | None, fallback: str) -> str:
        """R9h: how a question names product k - the category label its listings carry (the shopper names each product by it),
        of its labels the one sharing the most words with the request's product phrase and adding the fewest; else the
        phrase itself."""
        try:
            rows = self.needs.items()
            homes = self.home_shelves()
            if k is None or not rows or k >= len(rows) or k >= len(homes) or not homes[k]:
                return fallback
            want = {w.rstrip("s") for w in kind_tokens(rows[k]["product"])}

            def fit(label: str) -> tuple[int, int, int]:
                have = {w.rstrip("s") for w in kind_tokens(label)}
                return len(have & want), -len(have - want), -len(label)
            best = max(sorted(homes[k]), key=fit)
            return str(best).strip().lower() or fallback
        except Exception:
            return fallback

    def label_question(self, prop: str, text: str) -> str:
        """R9h: with several products, a question names its product by the product's full category label. Live races: a
        question naming the hidden property drew the reveal 72-100% when every word of the shopper's product name was in
        it, 38-67% with some of them and 0-31% with none."""
        rows = self.needs.items()
        if not rows or ";" in prop:
            return text
        k = self.product_of(prop) if "(" in prop else self.product_of(text)
        if k is None:
            return text
        tied = re.search(r"\(([^)]+)\)", prop)
        phrase = (tied.group(1).strip() if tied else rows[k]["product"]).lower()
        label = self.ask_name(k, phrase)
        name = re.split(r"\s*\(", prop)[0].strip()

        def stems(t: str) -> set[str]:
            return {w.rstrip("s") for w in kind_tokens(t)}
        # (the request's name and the category label both, unless one holds the other's words: the shopper may go by either)
        called = phrase if stems(label) <= stems(phrase) else label if stems(phrase) <= stems(label) else f"{phrase} ({label})"
        need = stems(called)
        if not name or (need and need <= stems(text) and len(text.split()) <= 24):
            return text
        return f"Which {name} would you like for the {called}?"

    def quota_first(self, scripted: list[dict[str, Any]], ok: Any = None) -> list[dict[str, Any]]:
        """CC: every message goes to a product due: the fewest answered questions and revealed needs together, then the fewest
        revealed needs (a question counts for the product its property's tag names, else the one its asked text names; one
        the shopper did not answer, '[not answered]', counts for none); a product with a revealed need waits until every
        product without one had 2 answered questions (EC5/MC2: instead a product with a revealed need goes after every
        product with no answered question; among the rest one an open story hint names goes first, whatever its reveals,
        then the fewest answered and revealed; EC6/MC1: then the fewest messages sent to it that drew no answer, then
        fewer revealed). The planner's first question for a product due goes first, else a
        planned second of one, else quota_pick's question (a product an open story hint names first, then the request's
        order; its most-common-field fallback only for a product never asked yet); with none, the next products in line.
        While nothing was asked, a first planned question that names no product stands."""
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
                sent[k] += 1          # (EC6: every message to the product, '[not answered]' ones too)
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
        unasked = [j for j in range(len(rows)) if not asked[j]]
        try:
            # (EC5: a product counts as story-hinted while one of its open hints can be asked by its key, or while it has
            # more open [no key] hints - asked through the planner's mapping onto a shown key - than answered questions)
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

        def level_of(k: int) -> tuple[int, int, int, int, int]:
            # (R9h: a product that revealed a need goes after every product that has not; then the fewest messages sent to
            # it, answered or not - every product gets one before any gets a second; then a product an open story hint
            # names; then the fewest answered and revealed)
            return (1 if found[k] else 0, sent[k], 0 if k in hinted else 1, asked[k] + found[k],
                    0 if k in alt_left else sent[k] - asked[k])
        ready = list(range(len(rows)))
        for level in sorted({level_of(k) for k in ready}):
            due = [k for k in ready if level_of(k) == level]
            for n, qitem in enumerate(scripted):
                if self.plan_product(qitem) in due:
                    return [qitem] + scripted[:n] + scripted[n + 1:]     # (the planner's order among the products due)
            for qitem in scripted:
                second = str(qitem.get("second") or "").strip()
                j = self.product_of(second) if second else None
                if j in due:
                    # (a planned second of a product due goes out on its own: the planner's next pick for it)
                    tied = re.search(r"\(([^)]+)\)", second)
                    product = tied.group(1).strip() if tied else rows[j]["product"]
                    return [{"property": second, "question": f"Which {second.split('(')[0].strip()} would you like for the {product}?"}] + scripted
            for k in sorted(due, key=lambda k: (k not in hinted, k)):
                try:
                    pick = self.quota_pick(k, asked[k] == 0, ok)
                except Exception:
                    pick = None
                if pick and ok is not None and (not ok(pick.get("property")) or self.asked_for(k, pick.get("property"))):
                    pick = None       # (v605f FC3: the code's own pick passes the planner's freshness test too)
                if pick:
                    return [pick] + scripted
                if self.__dict__.get("quota_wait"):
                    return scripted        # (its rows are opened first; the question follows them)
        return scripted

    def quota_pick(self, k: int, fallback: bool = True, ok: Any = None) -> dict[str, Any] | None:
        """A question for product k, which had none: with none of its records read, the code reads a few of its search rows
        first (and asks after them); else one planner call restricted to that product's fields chooses, else the product's
        most common field that is still open (open_prop)."""
        product = self.needs.items()[k]["product"]
        homes = self.home_shelves()
        home = homes[k] if k < len(homes) else set()
        read_home = [s for s in self.archive.shelves_read() if s.strip().lower() in home]
        if not read_home and home and self.__dict__.get("quota_reads", {}).get(k):
            # (CC: its rows were opened and still no read record of its category counts: its category's menu as it stands)
            read_home = [s for s in sorted(home) if self.archive.prop_menu(s)[0]][:1]
        if not read_home:
            # (no record of this product read yet: the code reads a few of its search rows before asking about it)
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
        menu = self.menu_trim(self.archive.prop_menu(read_home[0])[0], self.archive.rich_reads(read_home[0]))     # (EC7)

        def good(label: Any) -> bool:
            # (v605f FC3: never a property spent, asked before for this product (answered or not) or not fresh)
            return ok is None or (bool(ok(label)) and not self.asked_for(k, label))
        pick = self.product_plan(k, menu)
        if pick and good(pick.get("property")):
            return pick
        try:
            hinted = self.open_hints(k)
        except Exception:
            hinted = []
        for hint in hinted:
            # (a story hint still open for this product comes before its most common field)
            prop = re.split(r"\s*\(", hint.split(":", 1)[0])[0].strip()
            if prop and not junk_key(prop) and not self.plain_hint(k, prop, hinted=True) and good(f"{prop} ({product})"):
                return {"property": f"{prop} ({product})", "question": f"Which {prop} would you like for the {product}?"}
        if not fallback:
            return None       # (CC: a product asked before gets a planned or hinted question only, never the code's field pick)
        for weigh in (False, True):
            # (the fields no requirement settles first; then, when none is left, one whose requirement of its name has a
            # value none of its sample values carry: story words written in the property's place)
            for line in menu.splitlines():
                prop = line.strip("- ").split(" (in ")[0].strip()
                if prop and self.open_prop(k, prop, line, by_samples=weigh) and good(f"{prop} ({product})"):
                    return {"property": f"{prop} ({product})", "question": f"Which {prop} would you like for the {product}?"}
        return None

    def product_plan(self, k: int, menu: str) -> dict[str, Any] | None:
        """One planner call (haiku only; at most 1 per errand) restricted to product k: the request, the ledger, the
        questions so far and that product's own field list. Its first property still open, as that product's question."""
        try:
            return self.product_plan_call(k, menu)
        except Exception:
            return None

    def product_plan_call(self, k: int, menu: str) -> dict[str, Any] | None:
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
        """A property that may still hide a need of product k (None: the one product, or any): not asked about it yet (a
        paired question nobody answered leaves both open), not settled by a requirement in force for it (one of the same
        property by strict name with a positive stated value (settles; with by_samples and its menu line, a value among its
        sample values), or, given its menu line, one whose positive word value its sample values carry), not one the
        shopper does not mind (strict name)."""
        try:
            name = str(prop).split("(")[0].strip()
            if not name or junk_key(name):
                return False
            several = bool(self.needs.items())

            def mine(text: str, tagged_only: bool = False) -> bool:
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
                    # (only a positive stated value settles the property: 'closure: no laces, ties or buckles' leaves it open)
                    if self.settles(str(req), samples if by_samples else ""):
                        return False
                    continue
                if same_prop(head, name):
                    # (a related name - collar type and collar, sleeve length and sleeves, material and material filter - is
                    # the same property under another key when one of the requirement's values is among this field's sample
                    # values; display size and display resolution, protein and protein source share none)
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
        """For the planner (k: only product k's): the seconds of paired questions whose reply did not mention them (still
        open: the shopper answered the first only), and the questions the assistant itself wanted to send."""
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
                         "answered only the first, or only the first was sent - so it is still open; each was the plan's next pick):\n"
                         + "\n".join(f"- {p}" for p in opens[-3:]))
            if wanted:
                text += ("\n\nQUESTIONS THE ASSISTANT WANTED TO ASK (it has read the listing records; a property such a question names "
                         "that is still open and is a key below comes first, asked by the key's own name):\n" + "\n".join(f"- {w}" for w in wanted[-3:]))
            return text
        except Exception:
            return ""

    def want_ask(self, raw_args: Any) -> None:
        """Keeps the text of an actor's question the code refused while it plans the questions (the last 3 distinct)."""
        try:
            args = raw_args if isinstance(raw_args, dict) else (json_object(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {})
            text = " ".join(str(v).strip() for v in (args or {}).values() if isinstance(v, str) and v.strip())[:300]
            if text:
                kept = [w for w in self.__dict__.get("wanted_asks") or [] if w.lower() != text.lower()]
                self.wanted_asks = (kept + [text])[-3:]
        except Exception:
            pass

    def drop_wanted(self, label: str) -> None:
        """A wanted question whose words name a property just asked (every word of its name) is no longer offered."""
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
        """CD: a planned question's text may name a subject besides its property: every word of another listed field's name
        (or of the planner's second), or one of another field's sample values given as an example - outside the words of
        the property and of the products' names."""
        try:
            flat = " ".join(re.findall(r"[a-z0-9]+", str(question).lower()))
            said = set(flat.split())
            said |= {w[:-1] for w in said if len(w) > 3 and w.endswith("s") and not w.endswith("ss")}
            own = set(prop_words(prop)) | {"which", "would", "you", "like"}
            for row in self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}]:
                for w in re.findall(r"[a-z0-9]+", str(row.get("product") or "").lower()):
                    own |= {w, w[:-1] if len(w) > 3 and w.endswith("s") else w}

            def named(name: str) -> bool:
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

    def pair_fits(self, first: str, second: str) -> bool:
        """CD: a second property may join the first in the last message: of the same product (one product, or the same
        "(product)"), and no near-twin of the first - not the same or a related name (strict_prop, same_prop,
        same_field), no shared name word, no shared sample values."""
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
            # (EC4: nor a second that may stand for another open key of the product - twin_of - which the reply would count
            # as one more subject: packaging type while pack type is open)
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
        """Queues the unasked second property of a pair; it goes out on its own as the next question."""
        queue = self.__dict__.setdefault("pending_seconds", [])
        if prop and prop.lower() not in [p.lower() for p in queue]:
            queue.append(prop)

    def pose_planned(self, urge: bool = False) -> bool:
        """Sends the next planned question, one at a time: each question is chosen after the answers to the earlier
        ones were read, so a 'don't mind' or a revealed need steers the next pick. Questions start once two listings'
        details were read (or at once when something must be learned from the shopper first)."""
        the_tool = self.role("message")
        ceiling = self.ask_limit()
        if not the_tool or self.msgs_sent >= ceiling:
            self.asking = False
            return False
        if urge:
            self.urged = True     # (R9b: once the actor wants to commit, the remaining questions go out back to back)
        urge = urge or bool(self.__dict__.get("urged"))
        prop_list, _ = self.archive.prop_menu()
        sets = self.archive.rich_reads()
        own = self.own_shelf()
        if own:
            # (one product, and listings of another category were read too: only its own category's records make the menu
            # and start the questions, so a referenced listing of another kind neither lends its fields nor fires them)
            prop_list, sets = self.archive.prop_menu(own)[0], self.archive.rich_reads(own)
        pressing = bool(self.needs.reading.get("ask_first")) and not self.questions_log
        thin = sets < ASK_AFTER_SETS and not pressing and not urge
        if thin and (self.wire.tick < ASK_AFTER_STEPS or (self.wire.tick < ASK_AFTER_STEPS + ASK_GRACE_STEPS and self.moves_left() > 20)):
            return False
        if self.moves_left() <= max(10, self.order_due()) or self.wire.spare() < CLOSING + 40:
            self.asking = False       # (CF2: no question once the steps kept for the order begin)
            return False
        waiting = self.__dict__.get("plan_wait")
        if waiting is not None and sets <= waiting and not urge:
            return False
        last_sets = self.__dict__.get("asked_at_sets")
        if last_sets is not None and sets <= last_sets and not urge and not pressing \
                and self.moves_left() > self.order_due() + 2 * max(0, ceiling - self.msgs_sent) + 14 \
                and self.wire.tick - self.__dict__.get("asked_at_tick", 0) < 4 \
                and not self.needs.items():
            # (EC2/MC6: several products - no such wait while the steps hold the +8 margin, which the +14 test above implies)
            # (the next question waits for another listing record, so it is chosen from more of the store)
            return False
        if self.pre_read():
            return True       # (the reads for the question menu run first, at once; the question follows them)
        if not self.questions_log and not self.msgs_sent and not pressing and not urge and self.first_reads():
            return True       # (EC2: each product's first 2 records are read before the first message; the question follows)
        if self.needs.items() and not urge and not pressing and not self.questions_log:
            # (R9b: several products - the first question waits until every product's category was seen and one of its
            # listings' fields read, so the plan can reach each product's own properties)
            try:
                homes = self.home_shelves()
                lacking = [k for k in range(len(self.needs.items())) if k >= len(homes) or not homes[k] or self.product_reads(k) < 1]
            except Exception:
                lacking = []
            if lacking and self.moves_left() > self.order_due() + 2 * max(0, ceiling - self.msgs_sent) + 14 \
                    and self.wire.spare() > CLOSING + 90 and self.__dict__.get("ask_waits", 0) < 12:
                try:
                    if self.seek_products(lacking):
                        return True       # (R9j: the code looks the product up, or reads its first records, itself)
                except Exception as err:
                    self.log.append({"kind": "seek_error", "error": f"{type(err).__name__}: {err}"[:300]})
                self.ask_waits = self.__dict__.get("ask_waits", 0) + 1
                if self.ask_waits == 1:
                    self.log.append({"kind": "ask_wait", "lacking": lacking})
                return False
        if self.quota_step():
            return True       # (R9b: the message allowance is read once, before the first question)
        ceiling = self.ask_limit()
        if self.msgs_sent >= ceiling:
            self.asking = False
            return False
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
        shown = self.menu_trim(prop_list, sets)      # (EC7: the planner's menu; prop_list stays whole for the CD2 test)
        if self.needs.items():
            ask_text += "\n\nTHE PRODUCTS ASKED FOR: " + " | ".join(f"{n + 1}) {row['product']} x{row['quantity']}" for n, row in enumerate(self.needs.items()))
            read = self.archive.shelves_read()
            homes = self.home_shelves()
            mine = [s for s in read if s.strip().lower() in set().union(*homes)]
            # (the products' own categories first: a referenced listing's category lends no fields while each product has one)
            order = (mine or read) if all(homes) else mine + [s for s in read if s not in mine]
            menus = [(shelf, self.archive.prop_menu(shelf)[0]) for shelf in order[:4]]
            prop_list = "\n".join(f"CATEGORY {shelf}:\n{menu}" for shelf, menu in menus if menu) or prop_list
            shown = "\n".join(f"CATEGORY {shelf}:\n{self.menu_trim(menu, self.archive.rich_reads(shelf))}"
                              for shelf, menu in menus if menu) or shown
        ask_text += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n" + (shown or "(none seen yet)")
        asked_props = {p.strip().lower() for ln in self.questions_log for p in ln.split(":", 1)[0].strip("- ").split(";")}

        def fresh(prop: Any) -> bool:
            # (a property already asked is never asked again: a 'don't mind' or a confused reply stays the answer; nor
            # is a requirement in force or a property the shopper does not mind, by name for the same product; nor a
            # story hint's plain name once its product's records show no field of that name)
            prop = str(prop or "").strip()
            if not prop or prop.lower() in asked_props or junk_key(prop) or prop.lower() in self.__dict__.get("spent_asks", set()):
                return False      # (CD: nor one whose plain question drew a one-at-a-time reply)
            j = self.product_of(prop) if self.needs.items() else None
            return not self.plain_hint(j, prop) and self.plan_open(j, prop)
        # (the second property of an earlier pair, which its reply did not settle, goes out next on its own; it stays
        # queued until it is sent, and one that is no longer open leaves the queue)
        queue = [p for p in self.__dict__.get("pending_seconds") or [] if fresh(p)]
        self.pending_seconds = queue
        pending = queue[0] if queue else ""
        if pending:
            tied = re.search(r"\(([^)]+)\)", pending)
            product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
            text = (self.__dict__.get("pending_text") or {}).pop(pending.lower(), "")      # (EC4: a distinct re-ask's own text)
            answer = {"questions": [{"property": pending, "question": text or f"Which {pending.split('(')[0].strip()} would you like for the {product}?"}]}
        else:
            answer = self.wire.consult(ask_text, 2500, PLANNER)
            if answer is None and self.wire.misses > 0:
                # (TA4: the planner's call was lost to load or the network: no plan is no evidence - planned again once
                # another listing is opened, not counted as an empty plan)
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
                    # (its second is a story hint's plain name no field carries: the question asks its first alone)
                    first = str(qitem.get("property") or "").strip()
                    tied = re.search(r"\(([^)]+)\)", first)
                    product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
                    kept[-1]["question"] = f"Which {first.split('(')[0].strip()} would you like for the {product}?"
            elif second and fresh(second):
                # (its first property was asked or is settled: the plan's unasked second goes out on its own)
                tied = re.search(r"\(([^)]+)\)", second)
                product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
                kept.append({"property": second, "question": f"Which {second.split('(')[0].strip()} would you like for the {product}?"})
        scripted = kept
        if self.needs.items() and not pending:      # (CD: a queued re-ask goes first; the rotation after it)
            scripted = self.quota_first(scripted, fresh)
            if self.__dict__.pop("quota_wait", False):
                return True       # (the reads queued for an unread product run first, at once; the question follows)
        if not pending and not (pressing and not self.questions_log):
            try:
                scripted = self.coverage_first(scripted, fresh)      # (R9i)
            except Exception as err:
                self.log.append({"kind": "coverage_first_error", "error": f"{type(err).__name__}: {err}"[:300]})
        proposal = None
        split_second = ""
        for thing in scripted[:2]:
            if self.needs.items():
                # (several products: a question names the product its property belongs to, else the reply is a confused one)
                prop = str(thing.get("property") or "")
                tied = re.search(r"\(([^)]+)\)", prop)
                product = tied.group(1).strip() if tied else ""
                words = re.findall(r"[a-z]{3,}", product.lower())
                if product and words and not any(w.rstrip("s") in str(thing.get("question") or "").lower() for w in words):
                    thing = dict(thing, question=f"Which {prop.split('(')[0].strip()} would you like for the {product}?")
            if not pending and not (self.needs.reading.get("ask_first") and not self.questions_log) and self.names_other(str(thing.get("question") or ""), str(thing.get("property") or ""), prop_list,
                                                str(thing.get("second") or "")):
                # (CD: the planner's text may name another subject - another field's name or one of its sample values as an
                # example: the property goes out alone in the plain form, the property and its product)
                prop = str(thing.get("property") or "")
                tied = re.search(r"\(([^)]+)\)", prop)
                product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
                thing = dict(thing, question=f"Which {prop.split('(')[0].strip()} would you like for the {product}?")
            proposal = self.message_call(str(thing.get("question")).strip(), f"ask_{self.msgs_sent + 1}_{self.wire.tick}")
            if proposal:
                label = (str(thing.get("property")), str(thing.get("question")).strip())
                second = str(thing.get("second") or "").strip()
                if second and second.lower() != label[0].lower() and not re.search(r"none|^-$|n/a", second.lower()):
                    if remaining == 1 and not self.__dict__.get("no_pairs") and self.pair_fits(label[0], second) \
                            and self.hint_mapped(self.product_of(label[0]) if self.needs.items() else None, label[0]):
                        # (EC5: only a story-hinted first property takes a second; a menu filler goes out single)
                        # (CD: only the last message may carry a second property - of the same product, no near-twin of the
                        # first, and while no reply asked for one question at a time; the code writes the pair plainly)
                        tied = re.search(r"\(([^)]+)\)", label[0])
                        product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
                        pair = (f"Which {label[0].split('(')[0].strip()} would you like for the {product}, "
                                f"and which {second.split('(')[0].strip()}?")
                        proposal = self.message_call(pair, f"ask_{self.msgs_sent + 1}_{self.wire.tick}")
                        if not proposal:
                            continue
                        label = (label[0] + "; " + second, pair)
                    else:
                        # (with few messages left, one property per question: a refused pair would cost too much)
                        prop = label[0].split("(")[0].strip()
                        tied = re.search(r"\(([^)]+)\)", label[0])
                        product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
                        single = f"Which {prop} would you like for the {product}?"
                        proposal = self.message_call(single, f"ask_{self.msgs_sent + 1}_{self.wire.tick}")
                        label = (label[0], single)
                        if not proposal:
                            continue
                        split_second = second
                break
        if not proposal:
            if sets < ASK_AFTER_SETS and self.moves_left() > 12 and not urge:
                self.plan_wait = sets     # (planned again once another listing is opened)
                return False
            self.empty_plans = self.__dict__.get("empty_plans", 0) + 1
            if self.moves_left() > 12 and self.empty_plans <= 1:
                # (messages and steps remain: planned again once another listing is opened, instead of never)
                self.plan_wait = sets
                return False
            self.asking = False
            return False
        try:
            worded = self.label_question(label[0], label[1])
            if worded != label[1]:
                better = self.message_call(worded, f"ask_{self.msgs_sent + 1}_{self.wire.tick}")
                if better:
                    proposal, label = better, (label[0], worded)
        except Exception as err:
            self.log.append({"kind": "label_question_error", "error": f"{type(err).__name__}: {err}"[:300]})
        self.asked_at_sets, self.asked_at_tick = sets, self.wire.tick
        self.chat.append({"role": "assistant", "content": f"I will ask the shopper about {label[0]}.", "tool_calls": [proposal]})
        found_before, freed_before = len(self.needs.uncovered), len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0)
        self.asking_label = label[0]       # (CE2b: the property asked, while its reply is read)
        try:
            responses = self.dispatch([proposal], scripted=True)
        finally:
            self.asking_label = ""
        self.post_replies(responses)
        asked_label = label[0]
        if pending and label[0].split(";")[0].strip().lower() == pending.lower():
            self.pending_seconds = [p for p in self.__dict__.get("pending_seconds") or [] if p.lower() != pending.lower()]
        open_note = ""
        heard_meta = self.needs.__dict__.pop("meta_last", None)

        def plain(text: Any) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
        if heard_meta is not None and plain(label[1]) and plain(label[1]) in plain(heard_meta):
            # (CD: the reply only asked for one question at a time - nothing was answered: pairs are off; the first property
            # goes out again alone in the plain form as the next message (once, and not when that very text drew the
            # reply: such a property is not planned again), and a second stays open for the planner)
            self.no_pairs = True
            parts = [p.strip() for p in label[0].split(";") if p.strip()] or [label[0]]
            first = parts[0]
            tied = re.search(r"\(([^)]+)\)", first)
            product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
            again = self.__dict__.setdefault("reasked", set())
            plain_q = f"Which {first.split('(')[0].strip()} would you like for the {product}?"
            alone = len(parts) == 1      # (a pair's one-at-a-time reply may come from its two subjects: its first goes out alone)
            alt = self.collide_alt(first) if alone else ""
            if alt:
                # (v605f FC1: the story hint's '(or X)' alternative goes next, before any re-ask of the same subject)
                again.add(first.lower())
                self.__dict__.setdefault("spent_asks", set()).add(first.lower())
                self.pending_seconds = [alt] + [p for p in self.__dict__.get("pending_seconds") or [] if p.lower() != alt.lower()]
                open_note = f" [{first}: not answered, its story hint's alternative {alt} asked next]"
            elif first.lower() not in again and plain(plain_q) != plain(label[1]) and not (alone and self.holds_plain(label[1], plain_q)):
                # (v605f FC2: not when the text sent already wrote the plain form's words - the same subjects again)
                again.add(first.lower())
                self.pending_seconds = [first] + [p for p in self.__dict__.get("pending_seconds") or [] if p.lower() != first.lower()]
                open_note = f" [{first}: not answered, asked again alone next]"
            else:
                # (EC4/MC3: the plain text itself drew it - the property's name stands for a second subject too: once per
                # product, the next message asks the story twin, else the property by its own values; then it is spent)
                redo = self.distinct_ask(first) if first.lower() not in again else None
                if redo:
                    again.update({first.lower(), redo[0].lower()})      # (a one-at-a-time reply to the re-ask spends it)
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
            # (a reply reveals at most one need: after a reveal the second property is still unasked)
            second = label[0].split(";", 1)[1].strip()
            found_text = " ".join(self.needs.uncovered[found_before:]).lower()
            if not any(w in found_text for w in re.findall(r"[a-z]{4,}", second.split("(")[0].lower())):
                self.queue_second(second)
                open_note = f" [{second}: not answered, still open]"
            label = (label[0].split(";")[0].strip(), label[1])
        elif ";" in label[0] and len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0) > freed_before:
            # (a "don't mind" to a pair settles its first property only; the reply did not mention the second, which stays
            # open and is listed for the planner to rank - a shopper may well have answered it too)
            second = label[0].split(";", 1)[1].strip()
            label = (label[0].split(";")[0].strip(), label[1])
            self.needs.indifferent = [p for p in self.needs.indifferent if p.strip().lower() not in (second.lower(), second.split("(")[0].strip().lower())]
            pairs = self.__dict__.setdefault("open_pair", [])
            if second.lower() not in [p.lower() for p in pairs]:
                pairs.append(second)
            open_note = f" [{second}: not answered, still open]"
        elif ";" in label[0] and len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0) == freed_before:
            # (a paired question the reply did not answer: pairs are off, and both properties stay open)
            self.no_pairs = True
            pairs = self.__dict__.setdefault("open_pair", [])
            for part in (p.strip() for p in label[0].split(";")):
                if part and part.lower() not in [p.lower() for p in pairs]:
                    pairs.append(part)        # (CD: both stay open and are listed for the planner)
            label = (label[0].replace(";", " and") + " [not answered]", label[1])
        elif split_second:
            # (the planner's second, split off with few messages left, stays open and is listed for the planner to rank
            # next, unless the reply settled it)
            found_text = " ".join(self.needs.uncovered[found_before:]).lower()
            pairs = self.__dict__.setdefault("open_pair", [])
            if not any(w in found_text for w in re.findall(r"[a-z]{4,}", split_second.split("(")[0].lower())) \
                    and split_second.lower() not in [p.lower() for p in pairs]:
                pairs.append(split_second)
        if heard_meta is None and len(self.needs.uncovered) == found_before:
            # (EC3: the property asked drew no reveal - a no-preference or confused reply: a story hint of it may reopen
            # under its own alternative; EC4 (ii): after such a reply to a hint-mapped property, its story twin goes next)
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

    # ----------------------------------------------------------------------------------------------- R9b: messages, statement
    def quota_tool(self) -> str | None:
        """R9b: the tool that reports the message allowance: the reading's message_status role, else a tool with no
        required argument whose name or description speaks of messages (chat, a quota, an allowance, usage) and of how many
        are allowed, spent or left; never a tool that holds another role."""
        others = {self.role(r) for r in ("search", "details", "view", "message", "cart", "cart_view", "order", "decline")} - {None}
        tool = self.role("message_status")
        if tool and tool not in others and not (self.shapes.get(tool) or {}).get("required"):
            return tool
        for tdef in self.toolset:
            fn = tdef["function"]
            tname = str(fn.get("name") or "")
            if tname in others or (self.shapes.get(tname) or {}).get("required"):
                continue
            blurb = re.sub(r"([a-z])([A-Z])", r"\1 \2", tname).replace("_", " ").lower() + " " + str(fn.get("description") or "").lower()
            if re.search(r"\bmessag|\bchat|\bquota\b|\ballowance\b|\busage\b", blurb) \
                    and re.search(r"\bhow many\b|\bleft\b|\bremain|\bspent\b|\bused\b|\bquota\b|\ballow|\busage\b|\blimit|\bcount", blurb):
                return tname
        return None

    @staticmethod
    def quota_from(obs: Any, sent: int) -> int | None:
        """R9b: the message allowance a status answer shows: a count of all messages allowed (allowed, limit, budget, quota,
        total, max), else those left (left, remaining, available) plus those spent (spent, used, sent, so far - the code's
        own count when no such field is shown). None when it shows neither."""
        if not isinstance(obs, (dict, list)) or (isinstance(obs, dict) and obs.get("error")):
            return None
        tally: dict[str, int] = {}

        def visit(d: dict[str, Any]) -> None:
            for key, val in d.items():
                if not isinstance(val, int) or isinstance(val, bool) or not 0 <= val <= 60:
                    continue
                low = str(key).lower()
                if re.search(r"so.?far|spent|used|sent|consumed", low):
                    tally.setdefault("used", val)
                elif re.search(r"left|remain|availab|unused", low):
                    tally.setdefault("left", val)
                elif re.search(r"allow|limit|max|cap|total|budget|quota|permit", low):
                    tally.setdefault("total", val)
        walk_dicts(obs, visit)
        if "total" in tally:
            return tally["total"]
        if "left" in tally:
            return tally["left"] + tally.get("used", sent)
        return None

    def quota_step(self) -> bool:
        """R9b: one scripted read of the status tool before the first question, while the allowance is unknown; True when a
        step went out."""
        tool = self.quota_tool()
        if not tool or self.msg_cap is not None or self.__dict__.get("quota_read") or self.wire.done or self.wire.broke \
                or self.moves_left() <= max(12, self.order_due() + 2) or self.wire.spare() < CLOSING + 40:
            return False
        self.quota_read = True
        args, faults = ArgCheck.execute({}, self.shapes.get(tool) or {})
        if faults:
            return False
        proposal = {"id": f"quota_{self.wire.tick}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.chat.append({"role": "assistant", "content": "I check how many messages I may send the shopper.", "tool_calls": [proposal]})
        before = self.wire.tick
        self.post_replies(self.dispatch([proposal], scripted=True))
        return self.wire.tick > before

    def basket_mark(self) -> tuple:
        """R9b: the cart as last changed: the count of answered cart calls and the lines with their units."""
        return self.__dict__.get("basket_ver", 0), tuple(sorted((k, float(v)) for k, v in self.trolley.units.items()))

    def basket_owed(self) -> bool:
        """R9b: an order needs the basket stated to the shopper after its last change: the cart holds lines, a message is
        left, no statement was made since the newest cart call (nor did one fail), and two steps remain."""
        if not self.role("message") or not self.trolley.units or self.__dict__.get("state_failed"):
            return False
        cap = self.msg_cap if self.msg_cap is not None else MSG_GUESS
        return self.msgs_sent < cap and self.__dict__.get("stated_mark") != self.basket_mark() \
            and self.moves_left() >= 2 and self.wire.spare() > 8

    def basket_words(self, short: bool = False) -> str:
        """R9b: the statement - each cart line with its units, a short title, its id and unit price, the total, and a
        request to go ahead (`short`: ids, units and prices only)."""
        info = self.code_rows()
        parts, total = [], 0.0
        for ident, units in self.trolley.units.items():
            row = info.get(ident) or {}
            title = "" if short else " ".join(str(row.get("title") or "").split()[:7])[:60].strip()
            price = row.get("price")
            n = int(units) if float(units).is_integer() else units
            parts.append(f"{n} x {title + ' ' if title else ''}({ident})" + (f" at {price:g} each" if is_number(price) else ""))
            total += float(price) * float(units) if is_number(price) else 0.0
        text = ("Here is the basket I am about to order for you: " if not short else "Basket to order: ") + "; ".join(parts) + "."
        if total > 0:
            text += f" Total {total:g}."
        return text + " Shall I go ahead and place this order?"

    def quiet_call(self, name: str, args: dict[str, Any]) -> str | None:
        """R9b: one call sent by the code outside the chat (booked like any call); the text the actor would read, or None
        when the world gave no answer."""
        resp = self.wire.act([{"name": name, "args": args}])
        self.log.append({"kind": "quiet_call", "action": {"name": name, "args": args}, "reply": resp})
        if not resp:
            return None
        return self.absorb(name, args, resp.get("observation"), resp, True)

    def state_gate(self, code: bool = False) -> str:
        """R9b: before an order, the basket statement when it is due: one message sent by the code. When no call went out
        since the newest cart call, one cart view (or stock read) goes first, so a change that cart call sets off is seen
        before the statement. Returns a hold text when either answer changed something (the order waits; with `code` the
        code's own finish goes on regardless), else ''."""
        if self.basket_owed() is False:
            return ""
        def state() -> tuple:
            return (self.needs.version, self.needs.spend_cap, self.refusal_n,
                    frozenset(self.dead_lines() & set(self.trolley.units)), self.basket_mark())
        before = state()
        looked: list[str] = []
        if not self.__dict__.get("calls_since_cart") and self.moves_left() >= 3:
            look = self.role("cart_view")
            args = ArgCheck.execute({}, self.shapes.get(look) or {})[0] if look and not (self.shapes.get(look) or {}).get("required") else None
            if args is None and self.stock_tool():
                look = self.stock_tool()
                args = self.lookup_call(next(iter(self.trolley.units)), look)
            if look and args is not None:
                seen = self.quiet_call(look, args)
                if seen:
                    looked.append(seen)
                if state() != before and not code:
                    return ("Not run yet (no step used). Before the order the code read the cart once more, and something changed: "
                            + " ".join(looked)[-1500:] + " Apply it and fix the cart, then order.")
                if self.basket_owed() is False:
                    return ""
        text = self.basket_words()
        proposal = self.message_call(text, f"state_{self.wire.tick}") or self.message_call(self.basket_words(True), f"state_{self.wire.tick}")
        if not proposal:
            self.state_failed = True
            return ""
        tool = self.role("message")
        args = json.loads(proposal["function"]["arguments"])
        self.stating = str(next(iter(args.values()), text))
        try:
            said = self.quiet_call(tool, args)
        finally:
            self.stating = ""
        last = self.log[-1].get("reply") if self.log and self.log[-1].get("kind") == "quiet_call" else None
        obs = last.get("observation") if isinstance(last, dict) else None
        if said is None or (isinstance(obs, dict) and obs.get("error")):
            self.state_failed = True
            return ""
        self.stated_mark = self.basket_mark()
        self.log.append({"kind": "basket_stated", "text": text[:600], "lines": dict(self.trolley.units)})
        if state()[:4] != before[:4] and not code:
            return ("Not run yet (no step used). Before the order the code stated the basket to the shopper, and the answer "
                    "brought a change: " + said[-1500:] + " Apply it (fix the cart if a line no longer fits), then order.")
        self.state_note = "[the basket was stated to the shopper before this order] " + said[-600:]
        return ""

    def reply_back(self, text_body: str) -> bool:
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
        for idx, (cid, text_body) in enumerate(responses):
            if idx == len(responses) - 1:
                text_body += "\n" + self.status_line()
            self.chat.append({"role": "tool", "tool_call_id": cid, "content": text_body})

    # ----------------------------------------------------------------------------------------------- finishing
    def plain_args(self, call_name: str, text_body: str, kind: str = "") -> dict[str, Any]:
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

    # ----------------------------------------------------------------------------------------------- code-only (TA)
    def code_reading(self) -> dict[str, Any]:
        """TA5: a plan from the request's own words and the tools' schemas, for an errand whose request no model could
        read: the role of each tool, the budget (code_budget), stock, a cheapest rule, message and step limits
        (code_limit), and the products: the shelf labels the request writes with a count when it names two or more,
        else the label sharing the most words with the request, each with its units and its requirements (code_needs)."""
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
            # (one product: the shelf label written with a count, else the one sharing the most words with the request)
            reading["product"] = items[0]["label"] if items else self.code_label(roles.get("search"), kind_words(text_body))
            reading["quantity"] = items[0]["quantity"] if items else self.code_count(low)
            reading["requirements"] = [f"{prop}: {value}" for prop, value in self.code_needs(low)]
        return reading

    @staticmethod
    def code_budget(text_body: str) -> tuple[Any, Any]:
        """TA5: (figure, unit) of the spending limit a text writes, else (None, None). Each figure that is no count of
        messages, steps or units and no part of a run of figures is scored by the words around it: a spending word or a
        currency sign just before it 3, a limit cue just before it ('under', 'up to', 'at most', 'to', 'at') 2, a limit
        cue just after it ('or lower', 'at most', 'and not a cent more') 2, a sentence about the order as a whole
        1; the best score of 2 or more wins (the larger figure on a tie); its unit is the one or two words after it."""
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
                    continue          # (part of a run of figures: 1920 x 1080, 4000:1)
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
        """TA5: a limit the request sets on messages or steps (`nouns`): a count of them with a limit cue just before it
        ('at most', 'no more than', 'up to', 'only', 'send', 'use', 'take', 'receive', 'allow', 'inside', 'within', 'total
        of', "you've") or just after it ('in total', 'no more', 'at most', 'max'); a count without one ('my 3 step
        routine', 'with 2 notes of rose') is story."""
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
        """TA5: the units of one listing a one-product request asks for: a count before 'units', 'of them', 'copies'; else 1."""
        figures = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        said = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:units?|of (?:them|those|these)|copies)\b", low)
        if not said:
            return 1
        n = int(said.group(1)) if said.group(1).isdigit() else figures[said.group(1)]
        return n if 1 <= n <= 99 else 1

    @staticmethod
    def code_needs(seg: str) -> list[tuple[str, str]]:
        """TA5: (property, value) of each requirement a stretch of the request writes as 'where/whose (the) P is V', '(and)
        the P has to / must / should be V'; the value ends at a comma or a joining word ('so', 'since', 'and the') and
        has at most four words; a property naming a person or a value that reads as story (it starts with a preposition, a
        pronoun or a verb such as 'for', 'my', 'see') or points at another listing is left out."""
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
        """TA5: the products a request names with a count, in its order: a shelf label of the search tool whose words the
        request writes in a row (link words and a plural s aside: 'bowls and plates' for a label 'Bowl & Plates'),
        after a count ('3 x', 'two', '2 pairs of', at most four words between, none a unit); each with its units and the
        requirements written after it up to the next one (or the sentence's end); [] when no label comes with a count."""
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
                # (no unit between the count and the label: '16 gb of memory' counts no memory)
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
        """TA5: the tool for each role, guessed from the tools' own names, descriptions and argument schemas: a tool whose
        required choice holds an order word and a decline word finishes both ways; finishing words in a tool's name; a
        cart-like tool with arguments puts in, one without views; then the search tool (a search word and a free text or
        a choice; not one that sends to the shopper), the message tool (a person and a send or reply word, one required
        text), the details and the view tools. No tool takes two roles but the order and decline pair."""
        said: dict[str, str] = {}
        for tdef in self.toolset:
            fn = tdef["function"]
            said[fn["name"]] = re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ").lower() + " | " \
                + str(fn.get("description") or "").lower()

        def props(name: str) -> dict[str, Any]:
            found = (self.shapes.get(name) or {}).get("properties")
            return found if isinstance(found, dict) else {}

        def needed(name: str) -> list[str]:
            found = (self.shapes.get(name) or {}).get("required")
            return [key for key in found if key in props(name)] if isinstance(found, list) else []

        def texts(name: str) -> list[str]:
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
                    continue        # (a tool that sends to the shopper is no search, whatever it says it finds out)
                if role == "message" and not (re.search(person, said[name]) or re.search(r"\b(?:messag\w*|chat\w*)\b", said[name])):
                    continue
                if one_text and not (len(needed(name)) == 1 and needed(name)[0] in texts(name)):
                    continue
                if not one_text and not (texts(name) or any(isinstance(spec, dict) and spec.get("enum") for spec in props(name).values())):
                    continue
                roles[role] = name
        return roles

    def code_label(self, tool: str | None, want: set[str]) -> str:
        """TA: the shelf label, from the search tool's longest list of choices, sharing the most words with `want` (the
        larger share of the label's own words breaking a tie); '' when none shares a word."""
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
        """TA6: (property, value) of each stated or revealed requirement in force for product k (all for one product) whose
        value names what the item is (positive_value, or a name of at most five words without an exclusion, such as a
        colour of several words or a model name), without the '(stated)' marks."""
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
        """TA6: search calls built from the search tool's schema alone, best first: a keyword search of the first stated
        value with the product's kind word (value first: a store may read only the first words), the listing of the
        shelf whose label shares the most words with the product (when the tool lists shelves) within the budget (a
        numeric ceiling argument), and the value's longest word with the kind word (for a store that reads two words); a
        keyword search of the product's first and last words when no value is stated. A choice argument takes the value
        its description pairs with the argument used (else its first value for the words, its last for the shelf)."""
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
            vals = enums.get(mode_key or "", [])
            for val in vals:
                if any(re.search(r"\b" + re.escape(str(val).lower()) + r"\b", part) and re.search(r"\b" + re.escape(arg.lower()) + r"\b", part)
                       for part in parts):
                    return val
            return vals[-1] if last else vals[0]

        def price_word(key: str) -> bool:
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
        firm = [value for _, value in self.code_values(k) if not re.search(r"\bor\b|/|,|\b(?:" + VALUE_LINKS + r")\b", value.lower())]
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

    def code_rows(self) -> dict[str, dict[str, Any]]:
        """TA6: every listing seen, by id - its category label, its title, its newest price and stock as the records show
        them, the words of all its fields, and its property fields."""
        out: dict[str, dict[str, Any]] = {}

        def visit(d: dict[str, Any]) -> None:
            ident = self.row_mark(d)
            if ident not in self.archive.ids:
                return
            info = out.setdefault(ident, {"shelf": "", "title": "", "price": None, "stock": None, "words": set(), "fields": {}})
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
            for v in d.values():
                if isinstance(v, dict):
                    info["fields"].update({str(key): str(x) for key, x in v.items() if isinstance(x, (str, int, float)) and not isinstance(x, bool)})
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return out

    @staticmethod
    def code_shows(row: dict[str, Any], prop: str, value: str, hit_words: set[str], read: bool) -> bool:
        """TA6: a listing's own evidence of a stated value (code_seen): its field of the property carries it; with no such
        field, its title does; for a listing whose record was not read yet, a keyword search holding every word of the
        value that found it is a hint too (its words may sit in a field of another name)."""
        named = [val for key, val in row["fields"].items() if same_prop(key, prop)]
        if named:
            return any(Errand.code_seen(value, val) for val in named)
        if Errand.code_seen(value, row.get("title") or ""):
            return True
        own = kind_words(value)
        return not read and bool(own) and own <= hit_words

    @staticmethod
    def code_seen(value: str, text: str) -> bool:
        """TA6: every word and figure of one of a value's alternatives (split at 'or', '/', ',') is in `text` (a longer
        word by its stem; a figure not inside a longer one)."""
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
        return False

    def code_picks(self, avoid: set[str] | None = None, keep: set[str] | None = None) -> list[tuple[str, int]]:
        """TA6: (listing, units) for each product asked for, from the records alone: a line in `keep` first, then a listing
        the item check passed, then one whose own evidence shows every stated value (code_shows), a read one before an
        unread one (under a cheapest rule the cheaper first), shown in stock, the one sharing the most words with the
        requirements, its kind and the request, the cheaper. Each pick is not shown out of stock, not ruled out (by a
        check, an answered decline audit, a refused checkout, `avoid`, or its own field of a stated property), within
        the budget, on the product's category (or, while none is known, naming the product's kind). When the picks
        together pass the budget, the cheapest fitting listing of each product is taken instead. [] when some product
        has none or even those pass the budget: the records then hold nothing that can fit."""
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
        # (a listing the decline audit reviewed without naming it is ruled out - only when the audit answered: one whose
        # model call failed, or that ran muted, is no evidence)
        answered = next((isinstance(e.get("result"), dict) for e in reversed(self.log) if isinstance(e, dict) and e.get("kind") == "audit"), False)
        audited = (set(reviewed or set()) - set(self.__dict__.get("audit_fits") or []) - live) if answered else set()
        # (EA4: while the code judges every requirement in force, a read listing its own evidence shows fitting every one is
        # not ruled out by the audit's silence; a check that failed it still rules it out)
        own_all = bool(audited) and self.code_judges_all()
        cap = float(self.needs.spend_cap) if is_number(self.needs.spend_cap) else None

        def value_rank(row: dict[str, Any], prop: str, value: str) -> tuple[int, int]:
            # (EA5: how a read record shows a value - in the field named like the request's property (0) or another name of
            # it (1), or no field (2); by whole-value equality (0), a list part (1), the title only (2), containment (3))
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
        avoid |= self.dead_lines()       # (CB5: a listing whose newest stock record shows it sold out)
        avoid |= self.ref_pointers()     # (CE5: a listing a differ reference's chain went through)
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
                    continue          # (a bound or a range: no word or figure of it decides alone)
                judged.append((prop, plain))
                stated.append((prop, plain))
            pool = []
            # (EA4: this product's resolved differ references - a fit shows a value of the property that is not the excluded one)
            apart = [(ref["prop"], [str(v) for v in [ref.get("value")] + list(ref.get("goes_with") or []) if str(v or "").strip()])
                     for ref in self.differ_done(k if rows else None)]
            for ident, row in info.items():
                if ident in failed or ident in self.refused_lines or ident in avoid or row["stock"] is False:
                    continue
                if (home and row["shelf"] not in home) or (not home and not (want & row["words"])):
                    continue
                if any(self.code_clash(row["fields"], prop, value) for prop, value in stated):
                    continue
                if self.ref_excluded(row["fields"], k if rows else None):
                    continue          # (CE5: its field shows the value a differ reference sets apart)
                cost = row["price"] if row["price"] is not None else self.archive.price(ident)
                if cap is not None and is_number(cost) and cost * qty > cap:
                    continue
                fits = all(self.code_shows(row, prop, value, hits.get(ident) or set(), ident in self.archive.read) for prop, value in judged)
                fits = fits and all(any(same_prop(key, prop) and str(val).strip() and not any(ch_writes(v, val) for v in bad)
                                        for key, val in row["fields"].items()) for prop, bad in apart)
                shows = [value_rank(row, prop, value) for prop, value in judged] if ident in self.archive.read else []
                if ident in audited and not (own_all and fits and (judged or apart) and all(max(s) <= 1 for s in shows)):
                    continue          # (an answered decline audit reviewed it without naming it; its silence yields only to the
                    # value itself or a list part in a field of the property - never a title, a longer word or a phrase holding it,
                    # nor a product with no requirement the code can see)
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
                # (v605d M4, CH2: a read listing whose field evidence shows a plain failure is no pick - judged lazily, at selection)
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
        """TA6: a read record shows a stated requirement failing: it has fields of the requirement's property and none of
        them carries a word or figure of the required value (the code's own evidence; a record with no such field never
        clashes)."""
        named = [val for key, val in fields.items() if same_prop(key, prop)]
        return bool(named) and not any(value_seen(value, val) for val in named)

    def cart_form(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None, list[str]]:
        """TA6: the cart tool's schema read for the code's own calls: (schema, the required choices' put values, their
        take-out values (None where the tool has none), the id argument, the units arguments). A choice's put value is
        the one the actor's first accepted cart call used, else the value whose words (split at non-letters and capitals)
        hold a put word; its take-out value the one holding a take-out word (else, of two values, the other one)."""
        jschema = self.shapes.get(self.role("cart") or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        known = json_object(self.trolley.put_form) if self.trolley.put_form else None

        def words(v: Any) -> set[str]:
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
        """TA6: the cart calls that put `picks` in, from the cart tool's schema: one call naming every line when the tool
        takes a list of line objects (it states the whole cart), else one call per line (its id argument, its units
        argument, and the put value of each required choice); [] when the schema allows none."""
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
        """TA6: the cart calls that take `lines` out: the take-out value of each required choice with the line's units,
        else (a tool that sets the units) the line at 0 units; [] for a tool that takes a list of lines (its put call
        states the whole cart) or no line; None when the schema allows no take-out."""
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
        """TA6: the observations of the newest world calls the code sent (one per action; R9g: a batch the per-turn limit
        split spans that many requests, read in order)."""
        cap = max(1, int(self.__dict__.get("turn_cap") or 16))
        parts = max(1, -(-n // cap))
        got: list[Any] = []
        for entry in reversed(self.log):
            if not (isinstance(entry, dict) and entry.get("kind") == "env"):
                continue
            resp = entry.get("reply")
            if not isinstance(resp, dict):
                break
            obs_list = resp.get("observations") if isinstance(resp.get("observations"), list) else [resp.get("observation")]
            got = list(obs_list) + got
            parts -= 1
            if len(got) >= n or parts <= 0:
                break
        return got[:n]

    @staticmethod
    def code_fault(obs: Any) -> bool:
        """TA6: a reply that refuses the call: a key naming an error, fault, refusal, rejection or invalid input with a value."""
        found = []

        def visit(d: dict[str, Any]) -> None:
            if any(re.search(FAULT_KEYS, str(key).lower()) and val not in (None, False, "", [], {}) for key, val in d.items()):
                found.append(d)
        walk_dicts(obs, visit, 0, 2)
        return bool(found)

    def code_search(self, products: int, more: bool = False) -> None:
        """TA6: the code's searches (code_queries) for every product, the best query of each first, as far as the steps
        allow; then, while some product has no fitting row yet or a cheapest rule holds, the next pages of each product's
        best query (CODE_PAGES rounds at most; the pages still unread are kept in code_more). `more`: one round of those
        kept pages only (a later round found no fitting row left). The rows a keyword search returned are booked with
        its words (code_hits) unless the store's keyword search is known to need only one word."""
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
            rounds = 1 + self.pg(CODE_PAGES) + (2 if self.needs.priority() == "cheapest" else 0)
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
            # (further pages keep the steps that reading the picks needs: about three records a product)
            calls = following[: max(0, self.moves_left() - reserve - 3 * products)]
            if not calls:
                break

    def code_next(self, call_args: dict[str, Any], obs: Any) -> dict[str, Any] | None:
        """TA6: the call for the next page of a search's results, from the search tool's paging argument and what the page
        says (a next marker, else its match count against the rows shown so far); None on the last page."""
        spec = self.page_arg()
        if not spec:
            return None
        key, by_page, low, high = spec
        pages: list[dict[str, Any]] = []

        def visit(d: dict[str, Any]) -> None:
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
        """TA6: the picks' own records are read while steps allow (the list-read tool for two or more, else the details
        tool), and the picks made again from what they show - up to CODE_READS rounds, keeping the steps the cart, a stock
        re-check, the order and one repair need."""
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
        """TA6: makes the cart hold exactly `picks` (lines that are not picks taken out, picks not yet held at their units
        put in) and checks the cart's replies: a pick whose call was refused, or that the cart's replies (or one cart view
        when they show no lines) do not show, is returned to be swapped. None when the calls cannot be made."""
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
            return None           # (the world did not answer: no evidence on the picks)
        if shown is None and not bad and self.role("cart_view") and self.moves_left() >= 3:
            self.send_chunk([(f"code_view_{self.wire.tick}", self.role("cart_view"), {})])
            view = self.code_replies(1)
            shown = self.trolley.view_units(view[0]) if view else None
            if view and shown is None:
                # (a view that names none of the lines empties the cart as counted: those picks are not in it)
                bad.update(ident for ident in want if ident not in self.trolley.units)
        if shown is not None:
            bad.update(ident for ident in want if ident not in shown)
        for ident in bad:
            self.trolley.units.pop(ident, None)
        return sorted(bad)

    def code_recheck(self, lines: list[str]) -> tuple[bool, set[str]]:
        """TA6: (read, sold out): each line's stock is read again when steps allow (the view tool, else the details tool,
        one call per line) - a listing can sell out while it sits in the cart - and the lines shown out of stock now."""
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
        """TA6: shopper lines no model read (a refusal's, or ones whose listener call failed) are read by the code for a
        new budget: a line's spending figure (code_budget) becomes the budget."""
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
        """TA6: the errand ends with no cart to order (the models gave out, or the clock or the steps ran short, or a
        checkout was refused): rather than decline, the code shops by itself with no model call - it searches when the
        records hold no fitting candidate, reads its picks while steps allow, makes the cart hold one line per product
        (code_picks, `keep` lines first, never `avoid`), confirms it from the replies, re-reads the lines' stock, and swaps
        a refused or sold-out line (four rounds at most). True when the cart holds the picks; False when the records hold
        nothing that can fit or no call can be made (the caller then declines)."""
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
                    # (no fitting row is left, or a cheaper one may sit further on: the next pages of the best searches)
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
        """Ends the errand: the cart as it stands is ordered; an empty one the code fills first (TA6, code_cart); a decline
        only when that finds nothing that can fit. A refused finish is booked like any refusal, the lines' stock read
        again, a sold-out line swapped and the order sent again (CODE_FINISHES calls at most); the last word is still the
        decline when the store never took the order."""
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
                    # (CB5: the cart's lines not read since the newest cart change get one stock re-read; a sold-out line is
                    # swapped by the code before the first order instead of being ordered)
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
                    # (TA6: the cart was empty or a line was gone - the code shopped by itself, so the errand orders)
                    move = {"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")}
                elif call_rank and turn > 0 and self.again_ok(avoid):
                    # (CF6: after a refusal the repair could not be made, no cart line is known sold out and the records still
                    # hold candidates - the same cart goes again rather than a decline)
                    move = {"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")}
                elif decl:
                    move = {"name": decl, "args": self.plain_args(decl, SORRY, "decline")}
                else:
                    return
            if self.call_kind(move["name"], move["args"]) == "order":
                try:
                    self.state_gate(code=True)      # (R9b: the basket is stated before the code's own order too)
                except Exception as err:
                    self.log.append({"kind": "state_gate_error", "error": f"{type(err).__name__}: {err}"[:300]})
                if self.wire.done or self.moves_left() <= 0:
                    return
            resp = self.wire.act([move])
            self.log.append({"kind": "forced_finish", "action": move, "reply": resp})
            kind = self.call_kind(move["name"], move["args"])
            if not resp or resp.get("finished") or self.wire.done or turn == CODE_FINISHES - 1 or self.moves_left() <= 1:
                break
            # (TA6: the store refused the finish - booked like any refusal: the lines that were in the cart, the records,
            # the shopper's lines; a budget in them is read by the code)
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
                continue          # (the lines stay refused: the code fills the cart with others)
            self.refused_lines.difference_update(set(lines) - sold)
            info = self.code_rows()
            total = sum(((info.get(line) or {}).get("price") or 0.0) * self.trolley.held(line) for line in lines)
            cap = self.needs.spend_cap
            if not sold and not (is_number(cap) and total > cap):
                # (no line sold out and the cart is within the budget: the refusal was a change of mind the store has now
                # told - the same cart goes again)
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
            # (CF6: a refused order goes again while nothing in its cart is known sold out and something can fit)
            move = ({"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")} if again
                    else {"name": decl, "args": self.plain_args(decl, SORRY, "decline")})
            self.log.append({"kind": "forced_finish", "action": move, "reply": self.wire.act([move])})

    def again_ok(self, avoid: set[str]) -> bool:
        """CF6: a refused cart may go again as it stands: it holds lines, none known sold out or to be avoided, and the records
        hold a candidate for every product (code_picks; a decline is right only when nothing can fit)."""
        units = set(self.trolley.units)
        gone = set(avoid) | self.dead_lines()
        return bool(units) and not units & gone and bool(self.code_picks(gone))

    def catch_up(self) -> None:
        """TA4: once a model answers again after an outage: a plan that came from the code is replaced by a model reading
        of the request (once), and shopper lines whose listener call failed are read now; the actor is told of both."""
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

    # ----------------------------------------------------------------------------------------------- observations
    def refused(self, call_name: str, obs: Any, resp: dict[str, Any]) -> bool:
        """An order call whose episode did not finish, or any result holding a refusal field."""
        if isinstance(obs, dict) and "refusal" in obs:
            return True
        return call_name == self.role("order") and not resp.get("finished") and not (isinstance(obs, dict) and obs.get("error") in ("invalid_args", "unknown_tool"))

    def notes_for(self, call_name: str, obs: Any, refused: bool) -> str:
        remarks = []
        if isinstance(obs, dict) and obs.get("error") == "budget_exceeded":
            if call_name == self.role("message"):
                self.msgs_sent = max(0, self.msgs_sent - 1)      # (R9b: the refused message did not go out)
            if call_name == self.role("message") or self.msg_cap is not None:
                self.msg_cap = self.msgs_sent if self.msg_cap is None else min(self.msg_cap, self.msgs_sent)
            remarks.append("That budget is used up; do not call this tool again.")
        if refused and call_name == self.role("decline") and getattr(self, "last_kind", None) == "decline":
            # (the store turned the decline down: it holds that something can still be bought - the audit runs again)
            self.audit_done = False
            self.decline_vetted = False
            self.event_seen = True        # (CF1)
            remarks.append("THE DECLINE WAS REFUSED and the episode continues: the store does not accept that nothing fits. Read the refusal, apply what it says, look again at the closest candidates (their own properties, stock and price against the requirements now), and order the one that fits; decline again only if each one truly fails.")
        elif refused:
            self.refusal_n += 1
            self.order_vetted = False
            self.event_seen = True        # (CF1: the order's reserve now holds a repair)
            reread = ""
            try:
                reread = self.refusal_reread(obs)
            except Exception as err:
                self.log.append({"kind": "refusal_reread_error", "error": f"{type(err).__name__}: {err}"[:300]})
            self.refused_lines.update(self.trolley.units)
            said: list[str] = []

            def gather(piece: Any, lvl: int = 0) -> None:
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
            self.refusal_texts = {" ".join(str(s).lower().split()) for s in said}       # (v605f FI)
            remarks.append("THE ORDER WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (take out any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again." + reread)
        elif isinstance(obs, dict) and len(obs) >= 2 and call_name != self.role("message"):
            lns: list[str] = []
            for piece in obs.values():
                if isinstance(piece, list) and piece and all(isinstance(val, str) for val in piece):
                    long_strings(piece, lns)
            seen = (self.__dict__.pop("refusal_texts", None) or set()) if lns else set()
            if lns and seen and all(any(" ".join(str(s).lower().split()) in t for t in seen) for s in lns):
                # (v605f FI: a finish-time push shows on the next call too - the refusal already carried it)
                remarks.append("This result repeats the message the refused finish already carried (handled above); it says nothing "
                               "about this result.")
            elif lns:
                self.event_seen = True    # (CF1)
                remarks.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(lns[:4]))
        return " ".join(remarks)

    @staticmethod
    def shelf_of(row: dict[str, Any]) -> str:
        """The category label a record carries ('' when none)."""
        return next((val.strip().lower() for key, val in row.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                     and re.search(r"categor|shelf|department|aisle", str(key).lower()) and "id" not in str(key).lower()), "")

    @staticmethod
    def key_stock(key: Any, val: Any) -> bool | None:
        """CB: what one field says of stock - True (can be bought now), False (cannot), None (not a stock field, or no
        value): a stock or availability field (one named 'out'/'sold' says False when true; another says False when it
        is false, zero or below, or words such as out, sold, none, not), an on-hand field false or zero, and back-order,
        pre-order or discontinued text all say False. The one stock reader of the code (code_rows, sold_out, CB)."""
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
        """CB: a record's stock as its fields say it (key_stock): False when any says it cannot be bought now, True when
        one says it can, None when it has no stock field."""
        states = [s for s in (Errand.key_stock(key, val) for key, val in row.items()) if s is not None]
        return all(states) if states else None

    @staticmethod
    def sold_out(row: dict[str, Any]) -> bool:
        """A record whose stock field says it cannot be bought now (key_stock)."""
        return any(Errand.key_stock(key, val) is False for key, val in row.items())

    def shelf_labels(self) -> set[str]:
        """The category label(s) most of the listings read so far sit on; empty while none is known."""
        tally: dict[str, int] = {}
        counted: set[str] = set()
        skip = self.ref_foreign()

        def visit(d: dict[str, Any]) -> None:
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
            # (several products: each product's own most-read label, or its matching label seen in rows when none of its
            # records was read; the most-read labels fill in for a product no label matches)
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
            subset = False
            for seen_ctx, seen_others, seen_words, seen_total in self.word_seen:
                if seen_ctx != ctx or seen_others != others or seen_total == total:
                    continue
                if len(words) > len(seen_words) and words[: len(seen_words)] == seen_words:
                    self.word_kind = "any" if total > seen_total else "all"
                elif len(seen_words) > len(words) and seen_words[: len(words)] == words:
                    self.word_kind = "any" if total < seen_total else "all"
                else:
                    # (EB5: a strict word-subset pair in another order - the words the store reads, when learned - whose
                    # superset matched fewer listings; it settles 'all' only while no prefix pair spoke)
                    cut = self.lead_words or None
                    mine, theirs = set(words[:cut]), set(seen_words[:cut])
                    subset = subset or (mine > theirs and total < seen_total) or (theirs > mine and seen_total < total)
            if subset and not self.word_kind:
                self.word_kind = "all"
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

    def token_key(self) -> str | None:
        """The search tool's string paging argument (a continuation token handed back by the results), if any."""
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        for key, pspec in (props if isinstance(props, dict) else {}).items():
            if isinstance(pspec, dict) and pspec.get("type") == "string" and not pspec.get("enum") \
                    and re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(key).lower()):
                return key
        return None

    @staticmethod
    def next_token(obs: Any) -> str | None:
        """The continuation token a result carries for its next page (None on the last page)."""
        if not isinstance(obs, dict):
            return None
        for key, val in obs.items():
            if isinstance(val, str) and val.strip() and re.search(r"next|onward|cursor|bookmark|token|continu|more", str(key).lower()):
                return val
        return None

    def page_arg(self) -> tuple[str, Any, Any, float | None] | None:
        """The search tool's paging argument from its schema: (name, counts whole pages? or "token", first value, highest
        or None). A token argument's first page is the empty token."""
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
                return not isinstance(v, str) or bool((props.get(k) or {}).get("enum")) or v.strip().lower() in shelves or k == key
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
        ceil, ended = self.ceiling_sweep(tried)
        if ceil is not None:
            return ceil, True      # (CG3: the price-filter listing below the cheapest in-stock row seen comes first)
        if ended:
            return None, False     # (and when it shows nothing cheaper in stock, the sweep is done)
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

    def free_text(self, call_args: dict[str, Any]) -> list[str]:
        """The free-text argument values of a search call (enum-valued arguments left out)."""
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        props = props if isinstance(props, dict) else {}
        tok = self.token_key()
        return [val for key, val in call_args.items() if isinstance(val, str) and not (props.get(key) or {}).get("enum") and key != tok]

    def book_page(self, call_args: dict[str, Any], rows: list, total: int, obs: Any = None) -> None:
        """Books a search page in its walk (the same call with only the paging argument changed)."""
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
            walk["every"] = [r for r in rows if isinstance(r, dict)]     # (a first page showing every match)
        marks = [self.row_mark(r) for r in rows if isinstance(r, dict)]
        if len(marks) >= 5:
            # (5+ rows in ascending id order: a fixed listing order, not a ranking)
            walk["id_order"] = walk.get("id_order", True) is not False and (marks == sorted(marks) or marks == sorted(
                marks, key=lambda m: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", m)]))

    def next_page(self, walk: dict[str, Any]) -> Any:
        """The first unread paging value of a walk that can still show unseen rows, else None."""
        key, by_page, low, high = self.page_arg()
        if by_page == "token":
            # (the token of the deepest page read whose next page is still unread)
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
        """How many property requirements in force have a value word among a search's leading query words."""
        def near(a: str, b: str) -> bool:
            return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))
        vals = [re.findall(r"[a-z0-9]{3,}", req.partition(":")[2].lower()) for req in self.needs.props_left() if not ref_text(req.partition(":")[2])]
        words = [w for text in self.free_text(call_args) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
        words = words[: self.lead_words] if self.lead_words else words
        return sum(1 for value_words in vals if any(near(w, v) for w in words for v in value_words))

    def row_hits(self, rows: list, words: list[str] | None = None) -> list[str]:
        """Search rows not read yet that are in stock, within the budget and (when it is known) on the requested category.
        With the query's words, when the request gives a value by reference: unread rows sold by the shop or brand the
        query names, when the reference or a read record of another seller names it too (pointed_words), are kept apart
        (ref_hits), whatever their price, stock or category - such a listing is read, not bought."""
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
        """Words that name another listing's seller: the request's references, and the text fields (not titles, not seller
        fields) of read records, less each record's own seller words (a redirect chain names the next seller there, often in
        the seller's notes)."""
        memo = self.__dict__.get("pointed_memo")
        if memo and memo[0] == len(self.archive.items):
            return memo[1]
        words: set[str] = set()
        try:
            for ref in self.needs.reading.get("references") or []:
                words |= set(re.findall(r"[a-z0-9]+", str(ref).lower()))

            def visit(d: dict[str, Any]) -> None:
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
                        own |= set(re.findall(r"[a-z0-9]+", val.lower()))      # (a seller name; a seller's notes are text)
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
        """Ready arguments for the next unread page of each search whose leading query words carry a required value and
        whose total is larger than the pages read: the ones carrying more required values first, then the narrower."""
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
        """(property, value-first query) when a keyword query's leading words hold a word of a required property's NAME
        and no word of any required value or of the product: a store's keyword index holds listing values, titles and
        categories, so such a query keeps only the listings that repeat the name. The query offered is the value from
        its first figure (comparison words dropped), else the whole value, then the product as the request words it."""
        if self.word_kind == "any":
            return "", ""
        lead = [w for w in re.findall(r"[a-z0-9]+", " ".join(raw).lower()) if len(w) >= 2][: self.lead_words or 2]
        product = " ".join(re.findall(r"[a-z0-9]+", str(self.needs.reading.get("product") or "").lower())[:3])

        def near(a: str, b: str) -> bool:
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

    # ----------------------------------------------------------------------------------------------- cheapest walk
    def ran_query(self, text: str) -> bool:
        """A search whose leading words (as many as the store reads, stems, any order) are those of `text` was run."""
        width = self.lead_words or 3
        mine = sorted(self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[:width])
        for walk in self.__dict__.get("page_book", {}).values():
            ran = sorted(self.stem(w) for w in re.findall(r"[a-z0-9]+", " ".join(self.free_text(walk["args"])).lower())[:width])
            if ran == mine:
                return True
        return False

    def search_template(self) -> tuple[dict[str, Any], str] | None:
        """(the other arguments, the text argument) of the newest search with one free-text argument, for a new query."""
        for walk in reversed(list(self.__dict__.get("page_book", {}).values())):
            args = walk["args"]
            if self.cg_listing(args):
                continue      # (CG: a price-filter listing call is no keyword search to vary)
            texts = self.free_text(args)
            keys = [k for k, v in args.items() if isinstance(v, str) and v in texts and v.strip()]
            if len(texts) == 1 and len(keys) == 1:
                return {k: v for k, v in args.items() if k != keys[0]}, keys[0]
        return None

    def word_trap(self, raw: list[str]) -> tuple[str, str]:
        """(note, query) when a keyword query that needs every word leads with a word that narrows it wrongly: a word of a
        required property's NAME that no required value, product or category word carries ("round TOE boot": listings
        whose field holds just the value are left out), or a word that only a dropped requirement's value carries. The
        query offered leaves that word out (the product's kind word added), when such a search was not run and it keeps a
        required value."""
        try:
            if self.word_kind == "any":
                return "", ""
            toks = [t.strip("\"'()[]{},.;!?|").lower() for t in " ".join(raw).split()]
            toks = [t for t in toks if re.search(r"[a-z0-9]", t)]
            # (EB6: while the word count is unknown in a store that needs every word, every token is read)
            width = self.lead_words or (max(3, len(toks)) if self.word_kind == "all" else 3)
            toks = toks[:width]

            def near(a: str, b: str) -> bool:
                if any(c.isdigit() for c in a + b):
                    return a.lstrip("0") == b.lstrip("0")
                return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))

            def parts(t: str) -> list[str]:
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
        """Up to 2 ready searches, each combining the word values of the requirements in force for a product with no verified
        fit (a closed list by its first value; figures, references and exclusions left out), the product's kind word last
        when the store reads it too; only when 2+ values combine and no search with those leading words was run."""
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
        """The decline audit named a listing a possible fit that the item check failed: one more item check sees both
        readings; when it passes, the listing counts as verified (its earlier failing rulings are cleared)."""
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
        """For a decline: one read listing per product asked for (one for a one-product request), on its own category, in
        stock and within the budget as last seen, that no item check failed under the requirements now in force - a
        verified one first, else the cheapest; [] when some product has none or the picks together pass the budget."""
        try:
            rows = self.needs.items()
            homes = self.home_shelves() if rows else [set(self.shelf_labels())]
            if not homes or any(not home for home in homes):
                return []
            ver = str(self.needs.version)
            failed = {k.split("|")[0] for k, v in self.rulings.items() if v and k.split("|")[1:2] == [ver]}
            live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == ver}
            # (a listing the decline audit reviewed without naming it a possible fit counts as ruled out)
            reviewed = self.__dict__.get("audit_ids")
            if self.__dict__.get("audit_done") and (reviewed is None or len(self.archive.read) <= self.__dict__.get("audit_reads", -1)):
                reviewed = set(self.archive.read)     # (the last audit saw every record read so far)
            reviewed = set(reviewed or set())
            audited = reviewed - set(self.__dict__.get("audit_fits") or [])
            sold: dict[str, bool] = {}

            def visit(d: dict[str, Any]) -> None:
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
        """Once, for a decline with an empty cart after the decline audit listed possible fits: the item check on the first
        two whose details records were read; when one passes (for several products: when verified listings then cover every product), the
        decline is held and the actor told to buy it."""
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
                    continue      # (judged on its full record, never on a search row)
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
        """The store lists search results in a fixed id order: two or more searches showed 5+ rows in ascending id order,
        and none showed another order."""
        orders = [w.get("id_order") for w in self.__dict__.get("page_book", {}).values() if "id_order" in w]
        return orders.count(True) >= 2 and False not in orders

    def product_index(self, thing: str) -> int | None:
        """The product a listing is, by its category (None for a one-product request, or when unknown)."""
        if not self.needs.items():
            return None
        shelf = self.shelf_known(thing)
        for k, home in enumerate(self.home_shelves()):
            if shelf and shelf in home:
                return k
        return None

    def walk_home(self, k: int | None) -> set[str]:
        homes = self.home_shelves()
        home = set(homes[k if k is not None else 0]) if homes and (k or 0) < len(homes) else set()
        return home or (set(self.shelf_labels()) if k is None else set())

    @staticmethod
    def fig_amount(text: Any) -> tuple[str, str] | None:
        """(v605f FA) an amount led by a whole figure and no limit's words ('4 pieces', '350 ml'): (the figure, its unit word);
        None otherwise (a limit, a decimal figure, no unit word). A store indexes '4-Piece' and '4 pieces' as the tokens '4' and
        'piece', so a search holding the figure as its own token and a word of the unit lists exactly the records writing it."""
        low = str(text).lower()
        if re.search(CH_BOUND, low):
            return None
        found = re.match(r"\s*(\d+)\s+([a-z]+)", low)
        return (found.group(1), found.group(2)) if found else None

    @staticmethod
    def amount_part(text: Any) -> bool:
        """EB4: a part of a value no keyword carries: a figure with a limit's words ('at most 350 ml') or an amount led by its
        figure ('350 ml', '6 cores') - a search word matches only that one written token."""
        low = str(text).lower()
        return bool(re.search(r"\d", low)) and bool(re.search(CH_BOUND, low) or re.match(r"\s*\d+(?:[.,]\d+)?\s+[a-z]", low))

    def walk_carries(self, args: dict[str, Any], k: int | None, alts: bool = False) -> set[Any]:
        """Which requirements in force for product k (k None: all; numbered in props_left order) have a value word among a
        search's leading words; with alts, which (requirement, alternative of a closed list) pairs. An amount bound or an
        amount led by its figure is never carried (EB4)."""
        def near(a: str, b: str) -> bool:
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
                    # (v605f FA: '4 pieces' is carried by the figure as its own token together with a word of its unit)
                    if fig[0] in toks and any(w == fig[1] or near(w, fig[1]) for w in toks):
                        out.add((n, a) if alts else n)
                    continue
                if not self.amount_part(part) and any(near(w, v) for w in words for v in re.findall(r"[a-z0-9]{3,}", part)):
                    out.add((n, a) if alts else n)
        return out

    def walk_score(self, args: dict[str, Any], k: int | None) -> int:
        """How many requirements in force for product k (k None: all) have a value word among a search's leading words."""
        return len(self.walk_carries(args, k))

    def walk_options(self, k: int | None) -> list[tuple[int, str, dict[str, Any]]]:
        """(pages left, key, walk) of the searches the cheapest walk may read on for product k: in id order, leading words
        carrying a value of k's requirements, a row of k's category among those seen, 1 to 2 x WALK_PAGES pages unread, and
        not covered by a search read to its end that showed every match (one whose words are a part of its own, or are all
        values of k and words of k's category label while carrying every value - every alternative of a closed list - it
        carries: all-words results hold no more); of twins (same total and first page) the one with fewer pages left; in an
        all-words store none whose words hold another option's words that one walk can read (EB4: its rows are among that
        one's); fewest pages first. Words are compared as whole sets (the words the store reads, when that is learned)."""
        home = self.walk_home(k)
        if not home:
            return []
        cut = self.lead_words or None

        def lead(walk: dict[str, Any]) -> set[str]:
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
                continue      # (a search of the same mode read to its end that showed every match holds every fit this one can show)
            pages = -(-walk["total"] // max(1, walk["size"])) - len(walk["read"])
            twin = (walk["total"], walk["first"])
            if 0 < pages <= 2 * self.pg(WALK_PAGES) and (twin not in best or pages < best[twin][0]):
                best[twin] = (pages, key, walk)
        opts = list(best.values())
        if self.all_words():
            # (EB4: of two options whose words nest, the longer lists only rows the shorter lists: when its extra words are
            # values of k or words of k's category label it loses no fit and the shorter is dropped; else the shorter, when
            # one walk can read it, holds every fit the longer holds, and the longer is dropped)
            def nested(p: tuple, o: tuple) -> bool:
                a, b = lead(p[2]), lead(o[2])
                return (a < b and not (b - a) <= values | label and p[0] <= self.pg(WALK_PAGES)) or (b < a and (a - b) <= values | label)
            opts = [o for o in opts if not any(nested(p, o) for p in opts if p is not o)]
        return sorted(opts, key=lambda p: (p[0], -self.walk_score(p[2]["args"], k), p[1]))

    def walk_plan(self, budget: int, only: set | None = None) -> list[tuple[int | None, str, int]]:
        """(product, walk key, pages) to read under a cheapest rule in an id-ordered store: per product not walked yet, its
        search with the fewest pages left; the products needing the fewest first, each at most WALK_PAGES, within `budget`."""
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
            n = min(pages, self.pg(WALK_PAGES), left)
            if n <= 0:
                break
            plan.append((k, key, n))
            left -= n
        return plan

    def walk_pick(self, k: int | None) -> tuple[str, float] | None:
        """(id, price) of product k's pick: its cart line, else its cheapest listing the item check passed."""
        gone = self.dead_lines()       # (v605f FF: a sold-out line is no pick)
        cands = [i for i in self.trolley.units if (k is None or self.product_index(i) == k) and i not in gone]
        if not cands:
            cands = [key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == str(self.needs.version)]
            cands = [i for i in cands if i not in self.refused_lines and i not in gone and (k is None or self.product_index(i) == k)]
        priced = [(i, self.archive.price(i)) for i in cands]
        priced = [(i, float(p)) for i, p in priced if is_number(p)]
        return min(priced, key=lambda ip: (ip[1], ip[0])) if priced else None

    def line_query(self, k: int | None) -> dict[str, Any] | None:
        """A ready search for the product line of product k's pick: its maker and the first word of its title after the
        maker that is no maker, kind, link or value word and no figure (the line's name), when that search was not run."""
        pick = self.walk_pick(k)
        template = self.search_template()
        if not pick or not template:
            return None
        rec: dict[str, Any] = {}

        def visit(d: dict[str, Any]) -> None:
            nonlocal rec
            if self.row_mark(d) == pick[0] and self.title_of(d):
                rec = d
        for item in self.archive.items:
            walk_dicts(item, visit)
        maker = next((v for key, v in rec.items() if isinstance(v, str) and v.strip() and re.search(r"brand|make|manufactur", str(key).lower())
                      and "id" not in str(key).lower()), "")
        maker_words = re.findall(r"[a-z0-9]+", maker.lower())
        if not maker_words:
            return self.title_line(k, rec, template)     # (CG5: a pick whose record names no maker: its line by its title's first words)
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
        """Unread rows on product k's category that its value-led id-ordered searches (and the searches `keys`) showed: in
        stock as last seen, within the budget for its units as last priced, not in a refused cart, priced below its pick
        (with none, below the cheapest of those rows already read); only rows shown by the searches carrying the most of k's values (an all-words search shows only rows carrying its
        words); cheapest first."""
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
            # (no pick yet: the cheapest of these rows already read stands in for it)
            seen = sorted((float(self.archive.price(i)), i) for i in ids if i in self.archive.read and is_number(self.archive.price(i)))
            pick = (seen[0][1], seen[0][0]) if seen else None
        qty = self.needs.items()[k]["quantity"] if k is not None and k < len(self.needs.items()) else self.needs.quantity()
        cap = self.needs.spend_cap
        sold: dict[str, bool] = {}

        def visit(d: dict[str, Any]) -> None:
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
        """One compact line per row of a search page: id, price, stock, title, category."""
        rows: list[dict[str, Any]] = []

        def visit(d: dict[str, Any]) -> None:
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
        """One scripted search of the walk; the actor reads a compact digest of its rows (and any shopper news)."""
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
        """Reads the planned pages (scripted, one search a step: each page's token comes from the page before), one line
        search per pick, then opens the cheapest unread rows below each pick; the walked products' lines are rank-checked
        again before the next cart call or order."""
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
                    # (a short line, up to 3 pages, is read to its end)
                    if not self.walk_call({**line["args"], key: self.next_page(line)}, "I read the next page of the line search."):
                        break
                    more += 1
                report.append(f"the pick's line '{' '.join(self.free_text(args))}'")
        ceil_rows: dict[Any, list[str]] = {}
        for k in dict.fromkeys(ceil or []):
            # (CG3: the category below each pick with the store's price filter, when one was learned)
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
            # (one list read, taken in turn from each walked product's cheapest rows)
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
        """Listings a decline counts as fits: the item check passed them under the requirements now in force, or the last
        decline audit named them and no item check failed them since; none from a refused cart."""
        ver = str(self.needs.version)
        live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == ver}
        failed = {k.split("|")[0] for k, v in self.rulings.items() if v and k.split("|")[1:2] == [ver]}
        return (live | (set(self.__dict__.get("audit_fits") or []) - failed)) - self.refused_lines

    def decline_lacking(self) -> list[int | None]:
        """The products (None: the one product asked for) with no fit (decline_fits) on their own category."""
        got = {self.shelf_known(i) for i in self.decline_fits()} - {""}
        if not self.needs.items():
            return [] if set(self.shelf_labels()) & got else [None]
        return [k for k, home in enumerate(self.home_shelves()) if home and not home & got]

    def decline_walks(self) -> list[str]:
        """For each product lacking a fit (at most 2 searches in all): the search the cheapest walk could read on for it
        (walk_options: id order, leading words carrying its values, rows of its category, pages unread, not covered by a
        search read to its end) that carries the most of its values given in words (a requirement given by reference
        counts for none: its words name the listing to read, not the value), then the most values in all, then the one
        with fewest pages left."""
        reqs = self.needs.props_left()

        def stated(args: dict[str, Any], k: int | None) -> int:
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
        """Per (categories, units): unread rows seen (only `ids` when given) on those categories, in stock as last seen,
        within the budget for the units as last priced, not in a refused cart (below: priced under that product's figure);
        cheapest first."""
        cap = self.needs.spend_cap
        shelf: dict[str, str] = {}
        sold: dict[str, bool] = {}

        def visit(d: dict[str, Any]) -> None:
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
        """Several products under no cheapest rule, each with a fit (decline_fits), whose cheapest fits together cost more
        than the budget: per product, the unread rows below its cheapest fit (row_pools); [] otherwise."""
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
            # (the product whose fit takes the biggest share of the order first: a cheaper one there helps most)
            pools = [pools[k] for k in sorted(range(len(pools)), key=lambda k: -shares[k])]
            return pools if any(pools) else []
        except Exception as err:
            self.log.append({"kind": "over_budget_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

    def decline_reads(self) -> str:
        """Once per errand, at a decline: queues the code's own reads of the unread pages of searches carrying required values
        (only in a store whose results come in a fixed id order, where later pages hold fits as often as the first; up to 3
        pages, keeping 6 steps for the order) and, for several products over the budget, of the cheapest rows below each
        pick; returns the hold note ('' when there is nothing to read)."""
        try:
            batch = self.batch_tool()
            if self.__dict__.get("decline_read_done") or self.wire.spare() < CLOSING + 60 or not (self.role("details") or batch) \
                    or self.trolley.units:
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
        """Runs the reads decline_reads queued: the pages (one search a step), then one list read (else up to 2 details
        reads) of the cheapest rows, taken in turn from the rows below each pick and the rows the pages showed; the decline
        audit then runs on the new records."""
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

    # ----------------------------------------------------------------------------------------------- references (BT)
    def ref_register(self) -> None:
        """BT1: one open reference per stated requirement whose value points at another listing; the reading's
        reference_targets give its seller, kind and details, else the request's own words."""
        reading = self.needs.reading
        infos = reading.get("reference_targets") if isinstance(reading.get("reference_targets"), list) else []
        infos = [t for t in infos if isinstance(t, dict)]
        listed = {str(r) for r in reading.get("references") or []} if isinstance(reading.get("references"), list) else set()
        for req in list(self.needs.written):
            value = str(req).partition(":")[2]
            if ref_text(value) or (str(req) in listed and re.search(DIFFER_WORDS, value.lower())):
                self.ref_add(str(req), infos, self.request, "request")

    def ref_untracked(self, refs: list[str], tracked: list[dict[str, Any]]) -> list[str]:
        """The reading's references that no reference record follows (BW3: or that one follows without having found the
        listing it points at - its chain is empty and it is unresolved - so the records the request points at still go to
        the check)."""
        try:
            return [r for r in refs if not any(self.ref_match(t, r) and (t.get("status") == "resolved" or t.get("chain")) for t in tracked)]
        except Exception:
            return refs

    def ref_take(self) -> None:
        """BT1: a need the shopper revealed by pointing at another listing opens a reference too (its seller a name the
        shopper's line writes) and joins the reading's references; a resolved value that a rebuilt requirement list lost
        to the reference's words again is put back."""
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
        """Opens the reference record of one requirement given by reference (None when its listing has no usable seller
        or a reference of that property and product is open already)."""
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
        # (CE2: the source's own words count only between the property's mention and the listing's seller name - a
        # differ word after the listing ('...; apart from that I'm easy') is about something else)
        span = ref_span(prop, source).lower()
        cut = span.find(str(tgt.get("seller") or "").lower()) if str(tgt.get("seller") or "").strip() else -1
        if rel == "same" and (re.search(r"differ|not the same|exclu|apart|other than|unlike", told)
                              or (cut > 0 and re.search(DIFFER_WORDS, span[:cut]))):
            rel, tol, unit = "differ", None, ""       # (CE2: the reader's or listener's differ, or the source's own words)
        elif rel == "same" and (re.search(r"partner|counterpart|pair|compatib|together", told)
                                or (cut > 0 and re.search(MATE_WORDS, span[:cut]))):
            rel, tol, unit = "partner", None, ""      # (R9a: the reader's or listener's pairing, or the source's own words)
        elif rel == "same" and told in ("at least", "at most"):
            rel = told                                # (R9a: a bound the reader or listener read in other words)
        elif rel == "same" and cut > 0 and re.search(LEAST_WORDS, span[:cut]):
            rel = "at least"
        elif rel == "same" and cut > 0 and re.search(MOST_WORDS, span[:cut]):
            rel = "at most"
        ref = {"req": str(req), "head": head.strip(), "prop": prop, "tag": tag, "relation": rel, "tol": tol, "unit": unit,
               "target": tgt, "first": dict(tgt), "chain": [], "seen": {self.ref_key(tgt)}, "judged": set(), "queries": [],
               "pages": 0, "hops": 0, "tries": 0, "fails": 0, "status": "open", "value": "", "goes_with": [], "now": "",
               "origin": origin, "unindexed": False}
        self.refs.append(ref)
        self.log.append({"kind": "ref_open", "req": str(req)[:200], "target": tgt, "relation": rel, "origin": origin})
        return ref

    def ref_target(self, info: Any, value: str, source: str, prop: str) -> dict[str, Any] | None:
        """{seller, kind, details} of the listing a reference points at: the reading's (or listener's) entry when its seller
        is a name the source writes, else read from the source's words after the property's mention, else from the
        requirement's own words; details the source does not write are left out. None without a seller."""
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
        """A seller name the source writes (whole words, a possessive allowed) that is not just a product or category word."""
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
        return (ref_name(tgt.get("seller") or ""), tuple(sorted({x for d in tgt.get("details") or [] for x in ref_toks(d)})))

    @staticmethod
    def ref_desc_of(tgt: dict[str, Any]) -> str:
        return (f"the {tgt.get('kind') or 'listing'} sold by {tgt.get('seller')}"
                + (" (" + "; ".join(tgt.get("details") or []) + ")" if tgt.get("details") else ""))

    def ref_desc(self, ref: dict[str, Any]) -> str:
        return self.ref_desc_of(ref["target"])

    def ref_index(self) -> dict[str, dict[str, Any]]:
        """id -> every field seen for that listing (its search rows and records merged, the newest winning), in the order
        first seen; ref_seen_at keeps the newest observation that showed each."""
        memo = self.__dict__.get("ref_memo")
        if memo and memo[0] == len(self.archive.items):
            return memo[1]
        rows: dict[str, dict[str, Any]] = {}
        seen_at: dict[str, int] = {}
        for n, rec in enumerate(self.archive.items):
            def visit(d: dict[str, Any]) -> None:
                ident = self.row_mark(d)
                # (CE3: a record with a property map counts whatever its size - a spec sheet holds only id, title and map)
                if ident in self.archive.ids and (len(d) >= 4 or any(isinstance(v, dict) and len(v) >= 2 for v in d.values())):
                    rows.setdefault(ident, {}).update(d)
                    seen_at[ident] = n
            walk_dicts(rec, visit)
        self.ref_memo = (len(self.archive.items), rows)
        self.ref_seen_at = seen_at
        return rows

    def ref_sellers(self, row: dict[str, Any]) -> set[str]:
        """The seller names a record or row carries, reduced (ref_name): its fields named like a seller (SELLER_KEYS: shop,
        brand ...) and (CE1) the short text fields learned to name a seller (seller_keys)."""
        learned = self.seller_keys()
        return {ref_name(v) for k, v in row.items() if isinstance(v, str) and v.strip() and len(v) <= 60
                and (re.search(SELLER_KEYS, str(k).lower()) or str(k) in learned) and "id" not in str(k).lower()
                and not re.search(r"note|text|desc|comment|detail|info|message|section", str(k).lower())} - {""}

    def seller_keys(self) -> set[str]:
        """CE1: the fields that name a listing's seller, learned from the records: a short text field (no id, title,
        name, label, category or text field) whose value is a reference's seller name on some record read or row seen,
        or a capitalised name that some record's notes write inside a sentence (a note naming another listing by its
        seller) and that no property map of the store carries as a value."""
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
        """A record's own text fields (its seller notes; with desc, the description when there are no notes)."""
        notes, other = [], []
        for k, v in row.items():
            name = str(k).lower()
            if not isinstance(v, str) or len(v.split()) < 3 or name == "id" or name.endswith("id") \
                    or re.search(r"title|heading|headline|name|label|caption|categor|shelf|department|url|image", name):
                continue
            (other if re.search(r"desc|overview|summary|about", name) else notes).append(v)
        return "\n".join(notes or (other if desc else []))

    def ref_parts(self, row: dict[str, Any]) -> tuple[set[str], set[str]]:
        """(the tokens of a record's field names, the tokens of its title, field values and notes - in the notes, the words
        a negation governs ('and not black', 'not the green one') left out)."""
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
        """A record may show a detail: one of its value words (field names left out) is among the record's title, field
        values and notes; and for each figure with a unit word it names, the record's own title and fields (not its notes,
        which describe other listings too) write that figure in that unit, or no figure in that unit at all (a figure may be
        given in another unit)."""
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
        """A record's value of the field named for the property ('' when none)."""
        for strict in (True, False):
            for val in row.values():
                if isinstance(val, dict):
                    for k, x in val.items():
                        if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip() \
                                and (strict_prop(k, prop) if strict else same_prop(k, prop)):
                            return str(x).strip()
        return ""

    def ref_ids_add(self, ids: list[str]) -> None:
        self.ref_ids = set(self.__dict__.get("ref_ids") or set()) | set(ids)

    def ref_foreign(self) -> set[str]:
        """Listings read only as reference candidates whose category is no product's own (BT8)."""
        try:
            ids = self.__dict__.get("ref_ids") or set()
            if not ids:
                return set()
            shelves = self.home_shelves()
            if not shelves or not all(shelves):
                return set()      # (BW4: a product whose own category is not known yet: no read is called foreign)
            homes = set().union(*shelves)
            rows = self.ref_index()
            mine = set()
            for k in self.trolley.units:
                mine.add(k)
            return {i for i in ids if i not in mine and self.shelf_of(rows.get(i) or {}) and self.shelf_of(rows.get(i) or {}) not in homes}
        except Exception:
            return set()

    def ref_cands(self, ref: dict[str, Any]) -> list[str]:
        """Listings seen (search rows or records) sold by the current target's seller, best first: on a category named like
        its kind, then showing more of its details in the title or overview, then shown by the newest search."""
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
        """BW1: (the keyword-search templates booked so far, best first; the modes that ignore the words). Searches with one
        free-text argument are grouped by their other arguments (an optional one holding a category label left out); a
        group in which two searches with no word in common showed the same first rows (2+) ignores the words and is left
        out; then the group used with the most different texts comes first, the earliest used on a tie."""
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
        """(other arguments, text argument) for the code's reference searches: those of the keyword-search mode the
        errand used most (ref_modes; a mode shown to ignore the words never), less an optional argument holding a category
        label seen on the rows; else, when the search tool has one free-text argument and nothing else is required, that
        argument alone."""
        jschema = self.shapes.get(self.role("search") or "") or {}
        needed = set(jschema.get("required") or []) if isinstance(jschema.get("required"), list) else set()
        labels = {self.shelf_of(r) for r in self.ref_index().values()} - {""}
        modes, blind = self.ref_modes(needed, labels)
        tpl = modes[0] if modes else None
        paging = self.page_arg()
        for entry in reversed(self.log) if not tpl else []:
            # (a search that showed no row is no walk, but its arguments still show how a keyword search is called)
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
        """The details' words a search for the target may carry, likeliest indexed first: a figure with its unit word
        written on it ('4gb'), then a plain figure of 2+ digits, then words of 3+ letters that no field name read so far
        (nor a requirement's property name) carries, the longest first."""
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
                # (CE3: the word right after a figure is its unit, no search word of its own: '2.43 inches')
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
        # (CE3: in a detail written "<field> <value>" its last word is the value, searched before the other words: when its
        # leading words name a field the store's records carry, or when the shopper's own words list every detail of the
        # listing pointed at as such field-value pairs)
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
        """CE: the property field names on the records read so far, lower case with separators as spaces (camelCase split)."""
        mark = len(self.archive.items)
        memo = self.__dict__.get("store_keys_memo")
        if memo and memo[0] == mark:
            return memo[1]
        keys: set[str] = set()

        def visit(d: dict[str, Any]) -> None:
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
        """The reference searches still to run for the current target (in order): the seller with one detail word, the
        seller alone; in a store that does not index sellers, the detail word with the kind word."""
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
        """The next unread page of a search that showed the seller's rows (up to 3 further pages per target)."""
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
        """Steps, time and model budget for one more of the code's reference reads or searches (BT5; CE: a smaller share
        while every open reference is a differ, an exclusion of one value that stays soft when unresolved - and only
        REF_DIFFER_BLIND steps while none of them has found a listing of its chain yet)."""
        blind = [r for r in self.__dict__.get("refs") or [] if r.get("status") == "open"]
        if blind and all(r.get("relation") == "differ" and not r.get("chain") for r in blind) \
                and self.__dict__.get("ref_steps", 0) >= REF_DIFFER_BLIND:
            return False      # (CE, review: a soft exclusion whose listing no search found is not worth the first add's steps)
        cap = max(8, int(REF_STEPS * self.step_cap))
        floor = self.ref_floor()
        opens = [r for r in self.__dict__.get("refs") or [] if r.get("status") == "open"]
        if opens and all(r.get("relation") == "differ" for r in opens):
            cap = max(6, int(REF_DIFFER_STEPS * self.step_cap))
        if not (self.__dict__.get("ref_steps", 0) < cap and self.wire.spare() > CLOSING + 60 and not self.wire.spent_out()
                and not self.wire.done and not self.wire.broke):
            return False
        if self.ref_waits(floor):
            return False      # (ED2: several products - the chase waits while another product has no candidate yet)
        return self.moves_left() > floor or self.ref_hop_room()      # (ED1: a pending named hop's last steps)

    def ref_floor(self) -> int:
        """The steps the code's reference reads and searches leave (v605d M1: the floor the base reserve rule gave, never
        within 7 steps of the order's reserve - CF's larger reserve would otherwise cut a differ chain's last notes read,
        and a short chain's last steps)."""
        return max(max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, self.step_cap // 4 + 1)) + 10, self.order_due() + 7)

    def ref_product(self, ref: dict[str, Any]) -> int | None:
        """The product a reference is about (its tag), None when unknown or one product."""
        return self.product_of(f"x ({ref['tag']})") if self.needs.items() and ref.get("tag") else None

    def ref_waits(self, floor: int | None = None) -> bool:
        """ED2 (MC4b): in a several-product errand whose open references are all differ (a soft exclusion of one value) and
        past their source listing, the code's chase waits while a product other than the references' own has no candidate
        yet (ed_pool: a listing the item check passed, or a read one whose own record shows every plain value), so that
        product's value pages and walks go first; the wait ends while the chase still keeps 4 steps above its floor. One
        product, a same or within reference (its value binds the product's own search), or a differ one whose source
        is not in its chain yet: never."""
        rows = self.needs.items()
        opens = [r for r in self.__dict__.get("refs") or [] if r.get("status") == "open"]
        if not rows or not opens or not all(r.get("relation") == "differ" and r.get("chain") for r in opens):
            return False      # (a same or within reference binds its own product's search, and a differ one's source
            # listing is found first - CE5's pointer guard needs it in the chain: neither waits)
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
        """ED1 (v509c: seller + details): an open reference whose newest hop named the next listing's seller, kind and
        details, with no search for that target run yet, may still take up to 3 scripted reference steps (search, read,
        notes) while more than order_due() + 2 steps are left - in a one-product errand, or once every other product has
        a verified pick. The 3 steps count from the first one granted, across later hops."""
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
        """ED1: one product, or every product but the reference's own has a verified pick (all but one, when its own is
        unknown)."""
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
        # (BW6: the hop calls stop at 80% of the errand's spend cap, so they never take the actor's finishing budget)
        return (self.__dict__.get("ref_calls", 0) < REF_CALLS and self.wire.spare() > CLOSING + 40 and not self.wire.spent_out()
                and self.wire.spend < 0.8 * SPEND_CAP * self.wire.__dict__.get("scale", 1.0))

    def ref_match(self, ref: dict[str, Any], req: Any) -> bool:
        """`req` is the reference's own requirement: the same text, or a reference-shaped value of the same property (and
        product) - or, once resolved, its resolved text."""
        head, _, value = str(req).partition(":")
        if str(req) in (ref["req"], ref.get("now")):
            return True
        pointing = ref_text(value) or (ref.get("relation") == "differ" and re.search(DIFFER_WORDS, value.lower()))
        if not pointing or not same_prop(re.split(r"\s*\(", head)[0], ref["prop"]):
            return False      # (CE2: a differ reference's own requirement may be written with its differ words only)
        tags = re.findall(r"\(([^)]*)\)", head)
        return not tags or not ref["tag"] or bool(kind_words(tags[-1]) & kind_words(ref["tag"]))

    def ref_live(self, ref: dict[str, Any]) -> bool:
        """The reference still binds: no requirement of its property was dropped while none is in force."""
        left = self.needs.props_left()
        if any(self.ref_match(ref, r) or same_prop(re.split(r"\s*\(", str(r).partition(":")[0])[0], ref["prop"]) for r in left):
            return True
        return not any(same_prop(re.split(r"\s*\(", str(r).partition(":")[0])[0], ref["prop"]) for r in self.needs.let_go)

    def ref_step(self) -> bool:
        """BT: one turn of the code's reference resolver - hops judged on records already read (no step), then at most one
        scripted read or search; True when it acted."""
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
            return True         # (CE3: a notes read went out while judging: one scripted step a turn)
        opens = [r for r in opens if r["status"] == "open"]
        if not opens or not self.ref_room():
            return acted
        batch = self.ref_batch()
        most = batch[2] if batch else 2
        pools = []
        for ref in opens:
            pool = [i for i in self.ref_cands(ref) if i not in self.archive.read and i not in ref["judged"]]
            shows = ref.get("shows") or {}
            # (rows are opened once a search led by a detail ran (or none can), when they are few, or when a title shows a detail)
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
        # (another reference with nothing to read searches first, so one list read serves both)
        needy = empty if queue and len(queue) < most else []
        if queue and not needy:
            return self.ref_read(queue, opens) or acted
        if not tpl:
            return acted            # (no keyword search to copy yet: the actor's first search gives one)
        for ref in needy or sorted(opens, key=lambda r: (r["hops"], len(r["queries"]))):
            for plan in self.ref_plans(ref):
                ref["queries"].append(plan.lower())
                if self.ran_query(plan):
                    return True     # (that search ran already: its rows are the candidates, read next)
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

    def ref_batch(self) -> tuple[str, str, int] | None:
        """R9c: the list read for the resolver's candidates: only a list tool whose answers carried the listings' property
        maps (or one not tried yet) - a list of offer pages shows no details to match, so the spec sheets are read one by
        one instead."""
        batch = self.batch_tool()
        caps = ((self.__dict__.get("tool_caps") or {}).get(batch[0]) or {}) if batch else {}
        if batch and caps.get("id", 0) and not caps.get("props", 0):
            return None
        return batch

    def ref_read(self, queue: list[str], opens: list[dict[str, Any]]) -> bool:
        """One scripted read of the candidate rows (one list read, else details reads)."""
        batch = self.ref_batch() if len(queue) >= 2 else None
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
            if batch and not (((self.__dict__.get("tool_caps") or {}).get(batch[0]) or {}).get("props")):
                unread = []      # (R9c: the list showed offer pages without property maps: the spec sheets are read next)
            for ref in opens:
                ref["judged"].update(unread)     # (a listing the read did not return is set aside, never read again)
        return self.wire.tick > before

    def ref_search(self, ref: dict[str, Any], args: dict[str, Any]) -> bool:
        """One scripted search for the reference's current target."""
        before = self.wire.tick
        got = self.walk_call(args, f"The request takes {ref['head']} from another listing ({self.ref_desc(ref)}): I search for it.")
        self.ref_steps = self.__dict__.get("ref_steps", 0) + (self.wire.tick - before)
        rows: list[dict[str, Any]] = []
        if got:
            walk_dicts(self.archive.items[-1], lambda d: rows.append(d) if self.row_mark(d) in self.archive.ids and len(d) >= 4 else None, ceiling=3)
        name = ref_name(ref["target"]["seller"])
        text = " ".join(self.free_text(args))
        if rows and not any(name in self.ref_sellers(r) for r in rows) and ref_name(text) == name:
            ref["unindexed"] = True      # (a seller-only search shows no row of that seller: sellers are not searchable here)
        self.log.append({"kind": "ref_search", "prop": ref["prop"], "query": text[:120], "rows": len(rows)})
        return self.wire.tick > before

    def ref_judge(self, ref: dict[str, Any]) -> bool:
        """BT3/BT4: judges the read records that may be the current target (no step): a plain record by the code, else one
        hop call over the best four; True when it judged."""
        tgt = ref["target"]
        cands = [i for i in self.ref_cands(ref) if i in self.archive.read and i not in ref["judged"]]
        if not cands:
            return False
        rows = self.ref_index()
        seller_read = [i for i in self.ref_cands(ref) if i in self.archive.read]
        parts = {i: self.ref_parts(rows.get(i) or {}) for i in set(cands) | set(seller_read)}
        loose = ref.setdefault("loose", set())

        def shows(i: str, d: str) -> bool:
            return self.ref_shows(rows.get(i) or {}, d, parts[i])
        if len(seller_read) >= 4:
            # (a detail given in words that none of 4+ records of that seller shows is not one the code can test - a word
            # the reading kept from the request's phrasing: the hop call still reads it)
            loose |= {d for d in tgt["details"] if not any(shows(i, d) for i in seller_read)}

        def words_ok(i: str) -> bool:
            return all(shows(i, d) for d in tgt["details"] if d not in loose)
        full = {i: sum(1 for d in tgt["details"] if set(ref_toks(d)) and set(ref_toks(d)) <= parts[i][0] | parts[i][1]) for i in cands}
        ranked = sorted(cands, key=lambda i: (not words_ok(i), -full[i], cands.index(i)))
        if not words_ok(ranked[0]) and ((self.ref_room() or self.ref_waits()) and ([i for i in self.ref_cands(ref) if i not in self.archive.read
                                                                                    and i not in ref["judged"]] or self.ref_plans(ref) or self.ref_page(ref))):
            return False         # (none of these shows the details: the next rows are read first - ED2: once the wait ends)
        pick = [i for i in ranked if words_ok(i)][:4] or ranked[:4]     # (the records that may be it; else the closest)
        exact = [i for i in pick if tgt["details"] and full[i] == len(tgt["details"])]
        missing = self.ref_notes_missing(ref, exact if len(exact) == 1 else ([i for i in pick if words_ok(i)][:2] or pick[:1]))
        if missing and self.ref_room() and self.ref_notes_read(ref, missing):
            return True          # (CE3: their notes are read first, through the tool that shows them: a note may send the reader on)
        if missing and self.ref_waits():
            return False         # (ED2: their notes are read once another product has a candidate - never stopped for it)
        if missing and not self.ref_room() and ref.get("relation") == "differ":
            # (CE3: a listing whose notes the store shows apart and that were never read may send the property elsewhere:
            # its own value is no resolution of an exclusion - the reference stays unresolved, soft; another relation
            # takes the record's own value as before, since unresolved it would bind nothing)
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
            ref["judged"].update(i for i in cands if not words_ok(i))     # (records that miss a detail given in words are not it)
        return True

    def ref_plain(self, ref: dict[str, Any], ident: str) -> bool:
        """A record that needs no hop call: a field named for the property, and notes that neither name the property (a
        note about it may send the reader elsewhere, even to the same seller) nor mention the field's value nor name
        another seller (a seller seen on another row, or a capitalised name past a sentence's first word that none of the
        record's other fields carries)."""
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
        """CE3: the lookup tool that shows the sellers' notes when the details tool's records do not: the learned notes
        tool; None while the details tool's own answers carried notes; else a lookup tool other than the details tool
        whose answers carried a text (or empty) field neither the details tool's answers nor the search rows carry (a
        notes field, empty on the listings read so far), the most used first; else the view tool, else the stock tool,
        when it is another single-listing lookup tool than the details tool and has not answered for a listing yet (one
        that answered showed nothing of the kind)."""
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

    def ref_notes_missing(self, ref: dict[str, Any], ids: list[str]) -> list[str]:
        """CE3: the listings among `ids` (at most two) whose notes the store shows apart (ref_notes_tool) and whose record
        from that tool was neither seen nor asked for this reference yet."""
        tool = self.ref_notes_tool()
        if not tool or not ids:
            return []
        seen = set((self.__dict__.get("read_with") or {}).get(tool) or set())
        for other, keys in (self.__dict__.get("note_keys") or {}).items():
            if keys and other != self.role("details"):
                seen |= set((self.__dict__.get("read_with") or {}).get(other) or set())   # (R9c: any tool that shows notes)
        tried = ref.setdefault("noted", set())
        return [i for i in dict.fromkeys(ids) if i not in seen and i not in tried][:2]

    def ref_notes_read(self, ref: dict[str, Any], need: list[str]) -> bool:
        """CE3: one scripted read of those listings' notes through ref_notes_tool (each asked once); True when a step went out."""
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
        """CE5: the listings a live differ reference's chain went through (the source included): each carries the excluded
        value by definition (its own value, or the end of its notes' redirect) - for the reference's own product only."""
        out: set[str] = set()
        for ref in self.__dict__.get("refs") or []:
            if ref.get("relation") == "differ" and ref.get("status") in ("open", "stuck", "resolved") and self.ref_live(ref):
                out.update(i for i in ref.get("chain") or [] if self.ref_own_product(ref, i))
        return out

    def ref_own_product(self, ref: dict[str, Any], ident: str) -> bool:
        """CE5 (review): a chain listing counts against the reference's own product only - with several products, one
        shelved under another product's category is that product's candidate, not a pointer (an unknown shelf counts)."""
        if not self.needs.items():
            return True
        shelf = self.shelf_known(ident)
        k = self.product_of(f"x ({shelf})") if shelf else None
        return k is None or not ref.get("tag") or k == self.product_of(f"x ({ref['tag']})")

    def ref_pointer_note(self, thing: str) -> str:
        """CE5: why a listing of a differ reference's chain cannot meet it ('' for any other listing)."""
        for ref in self.__dict__.get("refs") or []:
            if ref.get("relation") == "differ" and thing in (ref.get("chain") or []) and self.ref_live(ref) and self.ref_own_product(ref, thing):
                what = f" ({ref['value']})" if ref.get("status") == "resolved" and ref.get("value") else ""
                return (f"{thing} is the listing the shopper's reference for {ref['head']} points at, or one its notes lead "
                        f"through, so its {ref['prop']} is the very value{what} the shopper's {ref['prop']} must differ from: it cannot "
                        "meet that requirement")
        return ""

    def ref_excluded(self, fields: dict[str, str], k: int | None) -> bool:
        """CE5: a listing's fields show a resolved differ reference's excluded value (or a paired one) in the property's
        field, for product k (any product for a reference without one)."""
        rows = self.needs.items()
        for ref in self.__dict__.get("refs") or []:
            if ref.get("relation") != "differ" or ref.get("status") != "resolved" or not self.ref_live(ref):
                continue
            if rows and ref.get("tag") and self.product_of(f"x ({ref['tag']})") not in (k, None):
                continue
            bad = {" ".join(ref_toks(v)) for v in [ref.get("value")] + list(ref.get("goes_with") or []) if str(v or "").strip()}
            for key, val in (fields or {}).items():
                if same_prop(key, ref["prop"]) and " ".join(ref_toks(val)) in bad:
                    return True
        return False

    def own_redirects(self, thing: str) -> list[tuple[str, str, str]]:
        """CE6: (requirement, property, sentence) for each requirement in force (of the listing's own product) whose
        property the listing's own notes send to another listing: a notes sentence that names the property and, after
        it, a seller name the listing does not carry (another record's seller, or a capitalised name past the sentence's
        first word that none of its fields carries), not under a negation ('it is not the X')."""
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
                    # (review: a word right before 'by <Name>' is the other listing's kind ('... the lamp by Name'), no
                    # mention of the property; blanked for the property test only)
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
        """CE6: after a lookup, a cart line or a verified candidate whose own notes send a required property to another
        listing: the remark (once per listing and requirement version); its verified mark goes."""
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
                         + "; ".join(f"'{r}'" for r, _, _ in own) + " (the grader takes the other listing's value), so it meets "
                         + ("them" if len(own) > 1 else "it") + " only when that listing's record shows it; the item check runs again "
                         "before it is ordered" + (". If they do, take it out of the cart and put another verified fit in."
                                                   if ident in self.trolley.units else "."))
        return " ".join(parts)

    # ----------------------------------------------------------------------------------------------- CE2b the shopper's pointing lines
    def ref_phrase(self, line: str) -> tuple[str, int, int, dict[str, Any]] | None:
        """CE2b: (phrase, start, end, listing) of the part of a shopper's line that gives a value by pointing at another
        listing: from the first pointer words before the seller's name, in its clause (relation words, then words that
        point at a listing or page, then the seller) to the end of the listing's description - its '(or ...)' aside when
        one follows before the sentence ends, else the sentence's end; the listing is read from the line (ref_parse:
        seller, kind, details). None when the line points at no listing or page (a differ word alone - 'anything but
        <Name>' - excludes a value or a seller, it points at no listing) or at none with a seller."""
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
            return None       # (a name with no pointer words before it in its clause points at nothing: 'nowt but Nike will do')
        start = cut + 1 + min(marks)
        tail = text[hit.end():]
        aside = re.search(r"\(\s*or\b[^)]*\)", tail)
        stop = re.search(r"[.!?;](?:\s|$)", tail)
        if aside and (not stop or aside.start() < stop.start()):
            end = hit.end() + aside.end()
        else:
            end = hit.end() + (stop.start() if stop else len(tail))
        phrase = text[start:end].strip(" ,;")
        if not phrase:
            return None
        return phrase, start, end, parsed

    def ref_line_words(self, qtext: str | None, outside: str) -> list[str]:
        """CE2b: the runs of words a shopper's line (outside its pointing phrase) shares with the question asked - three
        or more letters, no link word and no product's kind word - the longest first: the property the reply is about."""
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
        """CE2b (Needs.hear's hook, on the listener's answer before it is applied): for each shopper line that points at
        another listing (ref_phrase), ONE need of the property the line names outside its pointing phrase - the listener's
        reference property, else a property of a pointing need, else of any need, else a field name of the store or a
        stated property, the longest such name - whose value keeps the line's pointing words (unless the listener's value
        already points with the same relation at the same seller), tagged with the product the line names outside the
        phrase (else the listener's tag, else the question's product); a reference entry read from the line's words; the
        other listing's facts (a need, drop or freed property named only inside the phrase, or a need whose value words
        all lie inside it) leave the answer. Returns the lines with the pointing phrases blanked."""
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
                return re.split(r"\s*\(", str(req).partition(":")[0])[0].strip()

            def inside_only(prop: str) -> bool:
                return bool(prop) and prop_named(prop, inside) and not prop_named(prop, outside)

            def detail(req: str) -> bool:
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
                        continue      # (the other listing's facts)
                    if same_prop(name(req), prop) and same_item(req) and req not in self.needs.written:
                        continue      # (replaced by the need below)
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
        """A text names the property: one of its words (3+ letters) is a word of the text (or shares its first 4 letters)."""
        words = [w for w in prop_words(prop) if len(w) >= 3]
        toks = set(re.findall(r"[a-z]+", str(text).lower()))
        return not words or any(t == w or (len(w) >= 4 and len(t) >= 4 and t[:4] == w[:4]) for t in toks for w in words)

    def ref_card(self, ident: str) -> str:
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
        """The hop call (haiku): which record is the target, and its go_to or value."""
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
        """BT4/BT5: applies a checked hop answer (go_to: the next target; value: resolved); a refused one is asked once more
        with the reason."""
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
            # (none of these is the listing - or the one named lacks a detail the description names, so it is not it: the
            # next ones are read)
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
                ref["judged"].add(target)    # (that record is set aside; the others are judged again)
                return
            ref["judged"].update(pick)
            hint = self.__dict__.pop("redirect_hint", None)
            if hint and isinstance(hint.get("go"), dict) and hint["go"].get("seller"):
                self.log.append({"kind": "ref_code_go", "prop": ref["prop"], "from": target, "to": hint["go"]})
                self.ref_go(ref, target, dict(hint["go"]))      # (R9a: the code follows the redirect its notes write)
                return
            ref["chain"].append(target)      # (the listing was found; what its notes say could not be read)
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
        """(where, problem, go_to, value, goes_with) of a hop answer checked against the target's own record: the target
        shows each detail given in words, and no other record shown shows more of the details word for word; a go_to
        seller is written in its notes, in the sentence naming the property, and is not the record itself; go_to details
        its notes do not write are dropped; a value is written in the record; a goes_with value in its notes."""
        row = self.ref_index().get(target) or {}
        keys, vals = self.ref_parts(row)
        for d in [d for d in ref["target"]["details"] if d not in (ref.get("loose") or set())]:
            if not self.ref_shows(row, d, (keys, vals)):
                return "target", f"{target} does not show '{d}', so it is not the listing described.", None, "", []

        def shown(ident: str) -> int:
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
        sent_on = self.note_redirect(target, ref["prop"])
        if sent_on and self.ref_field(row, ref["prop"]) and not re.search(
                r"\b(?:not|rather than|instead)\b[^.]{0,30}\bthis\b|\bthat one\b", sent_on["sentence"], re.I):
            sent_on = None      # (a field of its own and a note that does not set it aside: the record's value stands)
        if sent_on:
            # (R9a: the notes send the reader on for the property - the record's own value, if any, is set aside)
            self.redirect_hint = sent_on
            return "answer", (f"{target} has no field for {ref['prop']}, and its notes say: \"{sent_on['sentence']}\". That "
                              "listing is the go_to (its seller, kind and the values named for it); give no value of this record."), None, "", []
        if not value or not set(ref_toks(value)) or not set(ref_toks(value)) <= vals | keys:
            return "answer", f"the value '{value}' is not written in {target}'s record.", None, "", []
        raw = answer.get("goes_with") if isinstance(answer.get("goes_with"), list) else []
        pairs = [str(p).strip() for p in raw if isinstance(p, (str, int, float)) and not isinstance(p, bool) and str(p).strip()]
        pairs = [p for p in dict.fromkeys(pairs) if set(ref_toks(p)) and set(ref_toks(p)) <= ntoks and set(ref_toks(p)) != set(ref_toks(value))][:3]
        return "", "", None, value, pairs

    def ref_go(self, ref: dict[str, Any], target: str, go: dict[str, Any]) -> None:
        """The target's notes send the reader on: the next target (a cycle or too many hops stop it)."""
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
        """The requirement's value from the value read: a range for 'within', a bound, else the value (or the closed list of
        it and its paired values)."""
        value, pairs = str(ref["value"]), list(ref.get("goes_with") or [])
        if ref["relation"] == "within" and is_number(ref.get("tol")):
            hit = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)", value.lower().replace(",", ""))
            if hit:
                base, tol = float(hit.group(1)), float(ref["tol"])
                unit = hit.group(2) or str(ref.get("unit") or "")
                return f"{round(base - tol, 6):g} to {round(base + tol, 6):g} {unit}".strip() + f" (within {tol:g} of {value})"
        if ref["relation"] in ("at most", "at least"):
            return f"{ref['relation']} {value}"
        if ref["relation"] == "partner":
            # (R9a: the counterpart - the value the store pairs with that listing's own value, never that value itself)
            mates = [p for p in pairs if p.lower() != value.lower()]
            return " or ".join(mates) if mates else value
        if ref["relation"] == "differ":
            # (CE2: an exclusion - any other value meets it)
            return " and ".join(f"not {v}" for v in [value] + [p for p in pairs if p.lower() != value.lower()])
        return " or ".join([value] + [p for p in pairs if p.lower() != value.lower()])

    def ref_resolve(self, ref: dict[str, Any], target: str, value: str, pairs: list[str]) -> None:
        """BT6: the value read becomes the requirement (needs.version bumped)."""
        if not str(value or "").strip():
            self.ref_stop(ref, f"{target} shows no value for it")
            return
        if ref["relation"] == "partner":
            mates = [p for p in pairs if str(p).strip() and str(p).strip().lower() != str(value).strip().lower()]
            if not mates:
                try:
                    mates = self.ref_mates(ref, target, str(value))
                except Exception as err:
                    self.log.append({"kind": "ref_mates_error", "error": f"{type(err).__name__}: {err}"[:300]})
                    mates = []
            if not mates:
                ref["own_value"] = str(value).strip()
                self.ref_stop(ref, f"{target} shows {value}, but no note says which value the store pairs with it (that value "
                                   "itself is not the counterpart)")
                return
            pairs = mates
        ref["chain"].append(target)
        ref.update(status="resolved", value=str(value).strip(), goes_with=list(pairs))
        new_req = f"{ref['head']}: {self.ref_value(ref)}"
        ref["now"] = new_req
        self.ref_swap(ref, new_req)
        self.needs.resolved.append(new_req)
        self.needs.version += 1
        self.log.append({"kind": "ref_resolved", "req": ref["req"][:200], "now": new_req, "chain": list(ref["chain"])})
        via = f", reached from {ref['chain'][0]} through the listings' notes" if len(ref["chain"]) > 1 else ""
        pair = (f"; its notes say {ref['value']} goes together with {' and '.join(pairs)}, so each of them meets it" if pairs else "")
        if ref["relation"] == "differ":
            pair = (f"; its notes pair {' and '.join(pairs)} with it" if pairs else "") + (
                ". The shopper's value MUST DIFFER from that one: any other value meets it, that value does not")
        elif ref["relation"] == "partner":
            pair = (f"; the store pairs {ref['value']} with {' and '.join(pairs)}. The shopper wants the COUNTERPART: "
                    f"{' or '.join(pairs)} meets it, {ref['value']} itself does NOT")
        self.ref_tell(f"RESOLVED: {ref['head']} is given by another listing, and {target} shows {ref['value']}{via}{pair}. The "
                      f"requirement is now: {new_req}.")

    def note_redirect(self, ident: str, prop: str) -> dict[str, Any] | None:
        """R9a: a redirect the code reads in a listing's notes - a sentence that names the property and sends the reader on
        (go by, taken from, refer to, look to, check, use ...) to a listing of another seller: {sentence, go: {seller, kind,
        details}}; None when no sentence does."""
        row = self.ref_index().get(ident) or {}
        notes = self.ref_texts(row, desc=False)
        own = self.ref_sellers(row)
        for sentence in re.split(r"(?<=[.!?])\s+", notes):
            if not (prop_named(prop, sentence) or self.ref_names_prop(prop, sentence)):
                continue
            hit = re.search(SEND_ON, sentence, re.I)
            if not hit:
                continue
            tail = re.split(r",?\s+(?:not|rather than|instead(?:\s+of)?|since|because)\b", sentence[hit.end():])[0]
            via = re.search(r"\bfrom\s+([A-Z][a-z]{3,})(?:'s)?\b", tail)
            got = ref_parse("the " + tail.strip(), lambda n: ref_name(n) not in own and self.ref_seller_ok(n, sentence))
            seller = via.group(1) if via and via.group(1).lower() != "this" else (got or {}).get("seller", "")
            if not seller or ref_name(seller) in own:
                continue
            kind, details = (got or {}).get("kind", ""), list((got or {}).get("details") or [])
            if via:
                lead = re.split(r"\b(?:the|a|an)\s+", tail[:via.start()])[-1]
                kind = " ".join(w for w in re.findall(r"[A-Za-z&'-]+", lead) if w.lower() not in ("item", "product", "listing", "piece", "entry", "one", "from"))
                inner = re.search(r"\(([^)]*)\)", tail[via.end():])
                details = [d.strip() for d in re.split(r",|\band\b", inner.group(1))] if inner else details
            details = [re.sub(r"^(?:(?:which|that|who)\s+)?(?:has|have|is|are|comes?\s+with|with)\s+", "", d.strip()) for d in details
                       if d.strip() and ref_name(seller) not in ref_name(d)]
            return {"sentence": sentence.strip(), "go": {"seller": seller, "kind": kind.strip()[:80], "details": [d for d in details if d][:6]}}
        return None

    def ref_mates(self, ref: dict[str, Any], target: str, value: str) -> list[str]:
        """R9a: the values the store pairs with `value` for the reference's property, read from the notes by the code: the
        target's own notes first (where 'its' or 'that' points at the target's value), then the chain's, then every other
        record read (only sentences that write the value)."""
        rows = self.ref_index()
        order = [target] + [i for i in reversed(ref.get("chain") or []) if i != target] + [i for i in rows if i != target]
        for ident in dict.fromkeys(order):
            text = self.ref_texts(rows.get(ident) or {}, desc=False)
            if not text:
                continue
            mates = rule_mates(text, value, ref["prop"], own=(ident == target))
            if mates:
                self.log.append({"kind": "ref_mates", "prop": ref["prop"], "value": value, "mates": mates, "from": ident})
                return mates
        return []

    def ref_swap(self, ref: dict[str, Any], new_req: str) -> None:
        """Puts the resolved requirement in the reference requirement's place: in the stated, revealed and rebuilt lists, the
        reading's requirements and its product's entry."""
        nd = self.needs
        seller = ref_name(ref["first"]["seller"])

        def mine(req: Any) -> bool:
            if self.ref_match(ref, req):
                return True
            head, _, value = str(req).partition(":")
            tags = re.findall(r"\(([^)]*)\)", head)
            if tags and ref["tag"] and not kind_words(tags[-1]) & kind_words(ref["tag"]):
                return False      # (another product's requirement)
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
                continue      # (another product's entry: its requirements carry no product tag)
            row["requirements"] = [f"{str(r).partition(':')[0].strip()}: {value}" if mine(r) else r for r in row["requirements"]]

    def ref_stop(self, ref: dict[str, Any], why: str) -> None:
        """BT7: the reference stays unresolved (soft: never a veto, never a reason to decline)."""
        if ref["status"] != "open":
            return
        ref["status"], ref["why"] = "stuck", why
        self.log.append({"kind": "ref_stuck", "req": ref["req"][:200], "why": why[:300], "chain": list(ref["chain"])})
        self.ref_tell(f"NOT RESOLVED: {ref['head']} ({why}). Its value stays unknown: if a listing's notes you read show where it "
                      "ends, use that value; otherwise never decline or reject a listing over it.")

    def ref_tell(self, text: str) -> None:
        """Tells the actor once, in the newest result it reads."""
        if self.chat and self.chat[-1].get("role") in ("tool", "user") and isinstance(self.chat[-1].get("content"), str) and len(self.chat) > 2:
            self.chat[-1]["content"] += "\n[reference] " + text
        else:
            self.ref_told = list(self.__dict__.get("ref_told") or []) + [text]

    def ref_line(self) -> str:
        """The status line's references: each one's state."""
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
        """BT6/BT7: the references for the item check, the decline audit and the rank check: each resolved value with the
        listing it was read on (and its pairing); an unresolved one as unknown, with the last records its chain reached."""
        try:
            lines, ends = [], []
            for ref in self.__dict__.get("refs") or []:
                if ref["status"] == "resolved":
                    via = f" (reached from {ref['chain'][0]} through the listings' notes)" if len(ref["chain"]) > 1 else ""
                    pair = (f"; its notes say {ref['value']} goes together with {' and '.join(ref['goes_with'])}, so each of them meets it"
                            if ref["goes_with"] else "")
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

    # ----------------------------------------------------------------------------------------------- CG cheapest coverage
    def ceil_learn(self, call_args: dict[str, Any], rows: list, total: int, obs: Any) -> None:
        """CG1: what this store's searches show of their filters and depth, learned from use. Every search's total is kept
        by its arguments (paging aside). A numeric limit argument HOLDS in a mode when a call's rows are all priced at or
        below it and all sit on the one category label one of its text arguments names; a mode whose rows ever break its
        limit is never used for it. The listing keeps the call's other arguments but a text value every row carries in a
        field (a maker filter); a call with fewer free words replaces it. Depth: a full page that offered no next page
        while its total said more rows exist shows how many rows a search can reach."""
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
            bad.add(sig)      # (this mode ignores the limit)
            return
        if sig in bad or skey is None or len(labels) != 1 or "" in labels or any(p is None for p in prices):
            return

        def every(val: str) -> bool:
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
        cost = next((v for key, v in row.items() if is_number(v) and not isinstance(v, bool) and re.search(r"price|cost", str(key).lower())), None)
        return float(cost) if cost is not None else None

    def ceil_spec(self) -> dict[str, Any] | None:
        """CG1: the learned price-filter listing, unless its mode broke a limit since."""
        spec = self.__dict__.get("ceil_mode")
        return spec if spec and spec["sig"] not in self.__dict__.get("ceil_bad", set()) else None

    def cg_listing(self, args: dict[str, Any]) -> bool:
        """CG: a search call in the learned listing mode (its mode arguments, or the limit argument when it has none)."""
        spec = self.ceil_spec()
        if not spec or not isinstance(args, dict):
            return False
        if spec["mode"]:
            return all(args.get(k) == v for k, v in spec["mode"].items())
        return spec["bound"] in args

    def page_size(self) -> int:
        return max([w.get("size", 0) for w in self.__dict__.get("page_book", {}).values()] or [0])

    def pg(self, base: int) -> int:
        """R9d: a page budget for this store's page size - the constants count pages of 10 or more rows; a store showing
        fewer rows a page gets proportionally more pages (at most 2.5 times as many)."""
        size = self.page_size()
        if not size or size >= 10:
            return base
        return int(-(-(base * min(2.5, 10.0 / size)) // 1))

    def visible_depth(self) -> int | None:
        """CG1: how many rows of one search can be reached: the schema's last page (or offset) with the page size, else the
        depth a capped page showed; None while unknown."""
        found = self.page_arg()
        size = self.page_size()
        if found and size > 0 and found[1] != "token" and found[3] is not None:
            return int((found[3] - found[2] + 1) * size) if found[1] else int(found[3] - found[2] + size)
        return self.__dict__.get("page_cap")

    def raw_label(self, labels: set[str]) -> str:
        """CG3: of the category labels `labels` (lower case), the one most rows seen carry, spelled as the rows write it."""
        if not labels:
            return ""
        tally: dict[str, int] = {}

        def visit(d: dict[str, Any]) -> None:
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
        """CG3: the highest limit that leaves `price` out: one unit less with whole prices (or a whole-number limit), else
        a hundredth less; None when that leaves nothing."""
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
        spec = self.ceil_spec() or {}
        if spec.get("int") or (float(a).is_integer() and float(b).is_integer()):
            return (int(a) + int(b)) // 2
        return round((float(a) + float(b)) / 2, 2)

    def ceil_args(self, spec: dict[str, Any], label: str, bound: float | int) -> dict[str, Any]:
        whole = spec.get("int") or float(bound).is_integer()
        return {**spec["tmpl"], spec["shelf"]: label, spec["bound"]: int(bound) if whole else bound}

    def cg_products(self) -> list[int | None]:
        return list(range(len(self.needs.items()))) or [None]

    def ceiling_ready(self, k: int | None) -> bool:
        """CG3: product k can be listed below its pick: a cheapest rule, a learned price-filter listing, k's category label
        seen on rows, and k not listed so yet."""
        if self.needs.items() and k is None:
            return False      # (several products: a listing whose product is unknown names none of them)
        return (self.needs.priority() == "cheapest" and k not in self.__dict__.get("ceil_walked", set()) and self.ceil_spec() is not None
                and self.page_arg() is not None and bool(self.raw_label(self.walk_home(k))))

    def ceil_aim(self, k: int | None) -> float | None:
        """CG3: the price a cheaper listing of product k must beat: the lowest of its pick and of a held put's listing; v605f FG:
        several products, none of those, and an amount bound among k's requirements: the cheapest in-stock row within the
        budget seen on its category."""
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
        """(v605f FG) several products: a requirement in force of product k is an amount bound (a figure with a limit's words),
        which no keyword carries."""
        if not self.needs.items() or k is None:
            return False
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            if self.product_of(head) == k and re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
                return True
        return False

    def home_rows(self, home: set[str]) -> set[str]:
        """(v605f FG) ids of the rows seen on the categories `home`."""
        out: set[str] = set()

        def visit(d: dict[str, Any]) -> None:
            ident = self.row_mark(d)
            if ident in self.archive.ids and self.shelf_of(d) in home:
                out.add(ident)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return out

    def ceil_clash(self, home: set[str], bound: float) -> bool:
        """CG3: a row seen on the categories `home`, in stock, priced at or below `bound` (a listing below it cannot be
        empty then)."""
        hit = False

        def visit(d: dict[str, Any]) -> None:
            nonlocal hit
            if hit or self.shelf_of(d) not in home or not any(isinstance(v, str) and v in self.archive.ids and "id" in str(key).lower() for key, v in d.items()):
                return
            cost = self.row_cost(d)
            hit = cost is not None and cost <= bound and self.stock_state(d) is True
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return hit

    def cheap_unread(self, ids: Any, k: int | None, below: float | None = None) -> list[tuple[float, str]]:
        """CG: (price, id) of the rows `ids` not read yet: in stock as last seen, within the budget for product k's units as
        last priced, not in a refused cart, priced below `below` when given; cheapest first."""
        ids = set(ids)
        if not ids:
            return []
        qty = self.needs.items()[k]["quantity"] if k is not None and k < len(self.needs.items()) else self.needs.quantity()
        cap = self.needs.spend_cap
        sold: dict[str, bool] = {}

        def visit(d: dict[str, Any]) -> None:
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
        """CG: this store's keyword search needs every word read (learned; a store that reads only its first words and
        was not seen to take any of them counts too, as search_note does)."""
        return self.word_kind == "all" or (bool(self.lead_words) and self.word_kind != "any")

    def valued_ids(self, k: int | None) -> set[str]:
        """CG: rows on product k's category that its id-ordered searches led by a required value showed, in a store whose
        keyword search needs every word (each such row carries those words somewhere in its record)."""
        if not self.all_words():
            return set()
        home = self.walk_home(k)
        out: set[str] = set()
        for walk in self.__dict__.get("page_book", {}).values():
            if walk.get("id_order") is True and not self.cg_listing(walk["args"]) and self.walk_carries(walk["args"], k):
                out |= {i for i, s in (walk.get("row_shelf") or {}).items() if s in home}
        return out

    def valued_complete(self, k: int | None) -> bool:
        """(v605f FH) in an all-words store, an id-ordered search led by a required value of product k whose requirement has one
        alternative was read to its end and showed every match: a row outside it does not carry that value."""
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
        """CG3: a search row of `ident` whose own text writes every word of one required value (one alternative of a closed
        list) of product k."""
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
                    figs.append(fig)      # (v605f FA: written as the figure right before its unit - '4-piece', '4 pieces')
                    continue
                words = [self.stem(w) for w in re.findall(r"[a-z0-9]{3,}", part)] if not self.amount_part(part) else []
                if words:
                    wants.append(words)
        if not wants and not figs:
            return False
        text: list[str] = []

        def visit(d: dict[str, Any]) -> None:
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
        """EB: (messages left while the question burst lasts, the steps beyond order_due() and two per such message)."""
        left = max(0, self.ask_limit() - self.msgs_sent) if self.asking and self.role("message") else 0
        return left, self.moves_left() - self.order_due() - 2 * left

    def vl_values(self, k: int | None) -> list[list[list[str]]]:
        """EB1: the query words of each firm word value of product k (code_values), in order, as its alternatives: a closed
        list ('or', 'either', '/') by its first three alternatives, any other value by its first part ('new' of 'new, boxed
        and never used'); words of 2+ characters, link and comparison words and the property's own name words left out,
        integer figures of a name-like value kept ('windows 10'), a decimal figure left out (stores split it); no
        alternative from a reference, an exclusion, an amount bound or an amount led by its figure."""
        out: list[list[list[str]]] = []
        for head, value in self.code_values(k):
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
                # (v605f FA: an amount led by a whole figure keeps its figure and its unit word - '4 pieces')
                words = [w for w in re.findall(r"\d+(?:[.,]\d+)+|[a-z0-9]+", piece)
                         if (len(w) >= 2 or (fig and w == fig[0])) and not re.search(r"[.,]", w) and w not in LINK_WORDS
                         and (w not in name or (fig and w == fig[1]))]
                if words and words not in alts:
                    alts.append(words)
            if alts and alts not in out:
                out.append(alts)
        return out

    def vl_words(self, k: int | None, label: str) -> list[str]:
        """EB1: product k's category label's words as the rows write it ('&', link words and a possessive 's left out); once
        a composed search showed this store does not index the label, the label's words at least half of k's row titles
        hold, else the request's kind word for k."""
        words = list(dict.fromkeys(w for w in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", label.lower().replace("’", "'")))
                                   if len(w) >= 2 and w not in LINK_WORDS))
        if self.__dict__.get("label_indexed") is not False:
            return words
        home = self.walk_home(k)
        titles: list[set[str]] = []

        def visit(d: dict[str, Any]) -> None:
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
        """EB1: product k's composed searches as (all-values forms, first-value forms): with 2+ values, form (i) every firm
        value and the category label's words, then form (ii) the first value and the label's words; with one value, its
        forms only (first list). Under a cheapest rule a closed list gives one form per alternative (the first closed list
        among the values: every alternative's fits are needed), else its first alternative. Values first, so a store that
        reads only its first words still reads them; cut to the words the store reads (when learned), else to the
        schema's length."""
        vals = self.vl_values(k)
        if not vals:
            return [], []
        label_words = self.vl_words(k, label)
        many = self.needs.priority() == "cheapest"

        def build(parts: list[list[str]]) -> str:
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
            return v if many else v[:1]
        lone = [build([a]) for a in alts(vals[0])]
        if len(vals) < 2:
            return list(dict.fromkeys(f for f in lone if f)), []
        j = next((n for n, v in enumerate(vals) if len(alts(v)) > 1), 0)
        both = [build([a if n == j else v[0] for n, v in enumerate(vals)]) for a in alts(vals[j])]
        return list(dict.fromkeys(f for f in both if f)), list(dict.fromkeys(f for f in lone if f))

    def vl_next(self, first: list[str], second: list[str], tkey: str, mode: dict[str, Any]) -> str | None:
        """EB1: the next composed form to send: an unsent all-values form; once every one came back empty, an unsent
        first-value form; '' when a sent form showed rows (or there is none), None when every form came back empty."""
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
        """EB: a search call's mode: its choice arguments of at most four values (a category or other long choice list aside)."""
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        props = props if isinstance(props, dict) else {}
        return tuple(sorted((key, val) for key, val in args.items() if isinstance(val, str) and isinstance(props.get(key), dict)
                            and isinstance(props[key].get("enum"), list) and len(props[key]["enum"]) <= 4))

    def vl_set(self, text: str) -> frozenset:
        """EB: a query's words as a set of stems (only the words the store reads, when that is learned)."""
        return frozenset(self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[: self.lead_words or None])

    def vl_seen(self, text: str, tkey: str, mode: dict[str, Any]) -> int | None:
        """EB1: the total of a search already sent in the same mode whose text argument holds the words of `text` (vl_set);
        None when none was sent."""
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
        """EB1: after a composed search: the store indexes the category label when 2+ rows on it that the search returned
        carry a label word in no other text field of theirs."""
        if self.__dict__.get("label_indexed") is not None or not self.archive.items:
            return
        words = {self.stem(w) for w in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", label.lower())) if len(w) >= 2 and w not in LINK_WORDS}
        hits = 0

        def visit(d: dict[str, Any]) -> None:
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
        """EB2/EB7: the cheapest unread rows of `ids` (rows on product k's category): in stock as last seen, within the budget, below
        k's pick under a cheapest rule (CEIL_OPENS opens per product in all there); at most `most`, never into floor."""
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
        """EB: opens the queued rows (scripted reads; the rank check of those categories runs again)."""
        self.rank_checked = {mark for mark in self.rank_checked if self.shelf_known(mark.split("|")[0]) not in homes}
        self.order_vetted = False
        self.open_queue = queue
        self.open_recheck = True
        self.open_say = say
        self.open_purpose, self.open_floor = "walk", floor
        self.open_named()

    def vl_search(self) -> bool:
        """EB1 (value_pages' first branch, every objective): once per set of firm word values of a product whose category label
        the rows show, its composed value + label search (vl_queries: form (i), then form (ii) when (i) came back empty or
        it has one value; when every form came back empty while the value alone had rows on its category, once, the label's
        words in the row titles instead), built on the schema-built keyword call (code_queries), while more steps than
        order_due() + 2 per message left + 4 remain; LABEL_TRIES per product. A search with the same words sent before
        counts. One page showing every match: under a cheapest rule its cheapest unread rows are opened as the value pages'
        are (several products: later, vl_flush); else (EB7), when it holds every value, its unread in-stock rows within the
        budget, LABEL_OPENS at most."""
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
            # (a keyword search alone: no price limit or other filter, so it compares with the actor's own searches)
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
                    # (every form came back empty while the value alone had rows on the category: this store does not index
                    # the category label; its words in the row titles, else the kind word, stand in for it)
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
        """EB3: several products under a cheapest rule, once no product's value search waits to be read: the cheapest unread
        in-stock rows within the budget below each pick that its read value searches showed, 1-2 per product (2 while the
        steps allow), taken in turn."""
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
        """EB3: several products under a cheapest rule: per product the value-search pages the code read for it and whether it
        has a pick (a cart line or a listing the item check passed), and the first product without one."""
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
        """CG2 + EB1-EB3: first the composed value + label search (vl_search, every objective); then, under a cheapest rule in
        a store that lists results in id order and whose keyword search is not seen to take any word: a search led by a
        required value of a product, with a next page and a total the visible pages can show (while the AND rule is unknown:
        only such a search, then every listing it matched is seen; a composed search above the depth is read to its
        reachable end), is read to its end by the code (max(VALUE_PAGES, the visible pages - 1) further pages per product,
        max(VALUE_TOTAL, VALUE_TOTAL_PER per product) in all; several products: the one with the fewest pages read first,
        each within its share of the steps); pages wait for the question burst unless 2 or fewer are left or the steps come
        down to them; then the cheapest unread in-stock rows within the budget on the product's category (below its pick)
        are opened (several products: once no value search waits, vl_flush). Stops at order_due() + 2 steps."""
        try:
            if any(r.get("status") == "open" for r in self.__dict__.get("refs") or []) and self.ref_room():
                return False      # (v605d M5: an open reference's resolver goes first; the pages wait until its room ends)
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
            per = max(self.pg(VALUE_PAGES), -(-depth // size) - 1) if depth else self.pg(VALUE_PAGES)
            products = self.cg_products()
            several = len(products) > 1
            cap = max(self.pg(VALUE_TOTAL), self.pg(VALUE_TOTAL_PER) * len(products))
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
                        continue      # (EB2: the rule unknown - only a search whose every match the visible pages show)
                    if known and depth is not None and walk["total"] > depth and not composed:
                        continue
                    pairs = self.walk_carries(walk["args"], k, alts=True)
                    how = self.vl_mode(walk["args"])
                    if known and any(mode == how and (seen <= words or (seen <= values | label and pairs <= got)) for seen, got, mode in whole):
                        continue      # (a search read to its end that showed every match holds every fit this one can show)
                    reach = walk["total"] if depth is None else min(walk["total"], depth)
                    pages = -(-reach // max(1, walk["size"])) - len(walk["read"])
                    allowed = min(per - used.get(k, 0), cap - sum(used.values()))
                    if several:
                        allowed = min(allowed, share - used.get(k, 0) - opened.get(k, 0) - (0 if opened.get(k) else opens))
                    take = pages if pages <= allowed else (allowed if composed else 0)
                    if take <= 0 or self.moves_left() - take - opens <= floor:
                        continue
                    if (take > 2 or several) and left and room > take + opens + 4:
                        pending = True      # (EB2: the question burst first - a reveal narrows the search before its pages are spent;
                        continue            # v605f FG: several products - whatever the number of pages)
                    cands.append((take, -self.walk_score(walk["args"], k), key, words))
                if known:
                    # (EB4: of two candidates whose words nest, the longer lists only rows the shorter lists: when its extra words
                    # are values of k or words of k's category label the shorter is dropped, else the longer)
                    cands = [c for c in cands if not any((o[3] < c[3] and not (c[3] - o[3]) <= values | label)
                                                         or (c[3] < o[3] and (o[3] - c[3]) <= values | label) for o in cands)]
                options += [(used.get(k, 0), take, score, n, key, k) for take, score, key, _ in cands]
            if options:
                _, take, _, _, key, k = min(options, key=lambda o: o[:5])
                done.add(key)
                return self.read_value_search(k, key, floor, take, several)
            if wait and not pending and not (left and room > opens * len(wait) + 4):
                return self.vl_flush(floor)     # (v605f FG: the opens wait for the question burst too)
            return False
        except Exception as err:
            self.log.append({"kind": "value_pages_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def read_value_search(self, k: int | None, key: str, floor: int, most: int = VALUE_PAGES, defer: bool = False) -> bool:
        """CG2: reads the rest of one value-led search (scripted, one page a step, at most `most` pages), then opens its
        cheapest unread rows (EB3 defer, several products: once no product's value search waits - vl_flush)."""
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
                break      # (an empty page: the search shows no more)
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
        """CG3: lists product k's category below its pick's price with the learned price filter: the limit one unit below
        the pick; while the total is more than the calls left can read (or than the visible depth), the limit comes down
        to the midpoint of the cheapest price seen and the limit, or lower where an even spread of the total over the
        prices says the pages left would hold it (back up halfway when nothing is left); a total that fits
        is read to its end; CEIL_CALLS calls in all. (report, rows to open: rows a value-led all-words search also showed,
        then rows whose own text carries a required value, then the rest, cheapest first, CEIL_OPENS at most.) Books k as
        listed; a zero total while a known cheaper in-stock row exists, or a row above the limit, retires the mode. Stops at
        order_due() + 2 steps."""
        spec, found = self.ceil_spec(), self.page_arg()
        home = self.walk_home(k)
        label = self.raw_label(home)
        aim = self.ceil_aim(k)
        if not spec or not found or not label or aim is None:
            return None
        self.__dict__.setdefault("ceil_walked", set()).add(k)
        pick_now = self.walk_pick(k)
        self.__dict__.setdefault("ceil_aimed", {})[k] = (float(aim), pick_now[0] if pick_now else "")     # (v605f FF)
        bound = self.below(aim)
        if bound is None:
            return None
        floor = max(floor, self.order_due() + 2) if not self.orders_tried and not repair else floor + (2 if repair else 0)
        # (v605f FF: a repair walk's listing calls stop two steps above the floor its opens keep)
        depth = self.visible_depth()
        book = self.__dict__.setdefault("page_book", {})
        keys: list[str] = []
        calls, top, complete, ran, last = 0, None, False, False, bound
        what = "the pick" if pick_now or is_number((self.__dict__.get("ceil_aims") or {}).get(k)) else "the cheapest row seen"
        # (v605f FG: a walk with no pick yet - nor a held put - is aimed at the cheapest row seen)
        say = f"The cheapest is asked for: I list '{label}' priced below {what} ({aim:g}) with the store's price filter."
        while calls < self.pg(CEIL_CALLS) and self.moves_left() > floor + 1 and not self.wire.done and not self.wire.broke \
                and self.wire.spare() >= CLOSING + 30:
            args = self.ceil_args(spec, label, bound)
            key = json.dumps(args, sort_keys=True, default=str)
            self.__dict__.setdefault("ceil_keys", set()).add(key)
            calls += 1
            reuse = key in book and bool(book[key].get("first")) and key in (self.__dict__.get("ceil_seen") or {})
            # (v605f FF: any identical call already booked - the EB8 seed or the actor's own - is used, no step spent)
            if not reuse and not self.walk_call(args, say):
                break      # (EB8: when the seed listing was this very call, its page is used and no step spent)
            ran, last = True, bound
            keys.append(key)
            walk = book.get(key)
            if walk and any(is_number(self.archive.price(i)) and self.archive.price(i) > bound for i in (walk.get("row_shelf") or {})):
                self.__dict__.setdefault("ceil_bad", set()).add(spec["sig"])
            if self.ceil_spec() is None:
                break      # (a row above the limit: this mode ignores it)
            total = self.__dict__.get("ceil_seen", {}).get(key)
            if total is None:
                break      # (no listing came back - a fault: nothing is concluded)
            if not total or not walk:
                if self.ceil_clash(home, bound):
                    self.__dict__.setdefault("ceil_bad", set()).add(spec["sig"])
                    break
                if top is None:
                    complete = True      # (nothing on the category is priced below the pick)
                    break
                nb = self.ceil_mid(bound, top)
                if not bound < nb < top:
                    break
                bound = nb
                continue
            pages = -(-total // max(1, walk["size"]))
            if (depth is None or total <= depth) and pages - len(walk["read"]) <= self.pg(CEIL_CALLS) - calls:
                while self.next_page(walk) is not None and calls < self.pg(CEIL_CALLS) and self.moves_left() > floor + 1 \
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
            # (down to the midpoint of the cheapest price seen and the limit, or lower: to where the total, spread evenly
            # over the prices, would fill the pages the calls left can read)
            fill = max(1, self.pg(CEIL_CALLS) - calls) * max(1, walk["size"])
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
        bare = [] if self.valued_complete(k) else [i for i in rest if i not in texty]     # (v605f FH)
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
        """CG3, the shelf sweep's first source: (the next price-filter listing call, the sweep proven done). The limit is one
        unit below the cheapest in-stock row within the budget seen on the requested category; its further pages are read
        while it shows nothing cheaper in stock; done when that listing is empty or read to its end. One product only."""
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
        """CG4: under a cheapest rule in a store whose keyword search needs every word: a details read of an unread listing
        that no search led by a required value showed is held once per product and requirement version while rows such
        searches showed, on its category, cheaper than both the pick and that listing, in stock and within the budget, are
        unread (those carry the value's words somewhere in their records). Not in the reserve, not for a listing a check
        named, not when the request points at a listing for a value."""
        try:
            if call_name != self.role("details") or self.needs.priority() != "cheapest" or not self.all_words() or self.in_reserve():
                return ""
            named = [v for v in call_args.values() if isinstance(v, str) and v in self.archive.ids]
            if len(named) != 1 or named[0] in self.archive.read or named[0] in self.trolley.units:
                return ""
            ident = named[0]
            if ident in self.named_open() or self.__dict__.get("refs") or self.needs.reading.get("references"):
                return ""      # (a listing a check named, or a request giving a value by pointing at a listing: no hold)
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
        """CG5: a ready search for the product line of a pick whose record names no maker: the first title word of 3+
        letters that is no kind, link or value word, with the title's next word when that is no such word either (a model
        number, say), when that search was not run."""
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
        """(v605f FF) under a cheapest rule: a product whose CG3 ceiling walk was aimed at a listing that has since died (sold out
        or refused) and whose new pick costs more gets the walk once more, below the new pick, inside the repair reserve (down to
        FIRST_ORDER_BY steps); None when none is due."""
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
            (self.__dict__.get("ceil_aims") or {}).pop(k, None)      # (a held put's price - the dead listing's - aims no more)
            self.log.append({"kind": "ceil_rearm", "product": k, "dead": ident, "aim": pick[1]})
            return self.run_walk([], FIRST_ORDER_BY, [k], repair=True)
        return None

    def walk_step(self) -> bool:
        """Runs the cheapest walk a held cart call or order queued; else the price-filter listing of a product with a
        verified pick (CG3) while the steps allow it; else the walk due as the reserve nears: when the steps left come
        down to the reserve plus the pages and reads planned (CG5: at any loop, and only a plan that fits the steps is
        booked). (Also the reads a held decline queued.)"""
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
                return False      # (several products, one not found yet: the steps before the reserve are for its search)
            opens = 1 if self.batch_tool() else 2
            spare = self.moves_left()
            ceil = [k for k in self.cg_products() if (self.walk_pick(k) or self.bound_need(k)) and self.ceiling_ready(k)
                    and self.ceil_aim(k) is not None]
            if ceil and spare > self.order_due() + 2 + self.pg(CEIL_CALLS) + opens:
                # (EB3: the product with the fewest value pages read first, not the first in order; v605f FG: a product with
                # no pick - an amount bound no keyword carries - before those holding one)
                used = self.__dict__.get("valued_by") or {}
                ceil.sort(key=lambda k: (bool(self.walk_pick(k)), used.get(k, 0), -1 if k is None else k))
                return self.run_walk([], self.order_due() + 2, ceil[:1])
            if not ceil and spare > self.order_due() + 2 + self.pg(CEIL_CALLS) + opens and self.ceil_seed():
                return True
            plan = self.walk_plan(self.pg(WALK_TOTAL))
            if not plan:
                return False
            pages = sum(n for _, _, n in plan) + sum(1 for k, _, _ in plan if self.walk_pick(k))
            if spare > self.order_due() + 2 + pages + opens:
                return False
            plan = self.walk_plan(min(self.pg(WALK_TOTAL), spare - self.order_due() - 1 - opens))
            if not plan:
                return False      # (CG5: nothing fits the steps now; a later loop may still run a smaller plan)
            self.walk_early = True
            return self.run_walk(plan, self.order_due())
        except Exception as err:
            self.log.append({"kind": "walk_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def seed_args(self, label: str, bound: Any) -> dict[str, Any] | None:
        """EB8: a category listing with a price limit, built from the search schema alone: the argument the schema names for a
        category (a free-text one, or a choice holding `label`), a numeric price argument, and the mode value the tool's
        description pairs with the category argument; None when the schema has no such call."""
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
                return None      # (a mode the description never pairs with the category argument: no guess)
            args[key] = hit
        for key in jschema.get("required") or []:
            if key not in args:
                return None
        fixed, faults = ArgCheck.execute(args, jschema)
        return fixed if not faults and isinstance(fixed, dict) else None

    def ceil_seed(self) -> bool:
        """EB8: under a cheapest rule, no price-filter listing learned (nor one retired): once per product with a pick and its
        category label seen, the schema-built listing of that label (seed_args) with the limit one unit below the pick, so
        ceil_learn can learn the filter from its rows; the ceiling walk (CG3) follows and uses this page."""
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
        """Several products: one has none of its own category's listings read (none opened, or its category never seen),
        or one has a verified fit and another none."""
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
        """At a cart put (or, call_args None, an order) under the cheapest rule: when a product it puts or holds was not
        walked yet and the steps allow (the put, a re-read, the order and a repair kept), the walk is queued and the call
        held (no step used)."""
        try:
            if self.needs.priority() != "cheapest" or self.__dict__.get("walk_queue") or self.wire.spare() < CLOSING + 60 \
                    or (self.refusal_n and self.moves_left() <= 15):
                return ""     # (and not while a refused order is being repaired late)
            ids = [i for i, c in self.trolley.lines(call_args) if c is None or c > 0] if call_args is not None else list(self.trolley.units)
            only = {self.product_index(i) for i in ids if i}
            floor = FIRST_ORDER_BY + 1 if self.orders_tried == 0 else 4
            # (at most half the steps beyond those kept and the held call, for pages, line search and opens together: the
            # walk's own floor stops it at that half; the pages get the half less the opens, the line search what is left)
            half = (self.moves_left() - floor - 1) // 2
            room = half - (1 if self.batch_tool() else 2)
            ceil = []
            aims = self.__dict__.setdefault("ceil_aims", {})
            for k in sorted(only, key=lambda x: -1 if x is None else x):
                # (CG3: a product put or ordered for the first time is also listed below its price with the learned price
                # filter, CEIL_CALLS calls kept for it before the pages of its value searches)
                if room >= 2 and self.ceiling_ready(k):
                    ceil.append(k)
                    room -= self.pg(CEIL_CALLS)
                    costs = [float(p) for p in (self.archive.price(i) for i in ids if i and self.product_index(i) == k) if is_number(p)]
                    if costs:
                        aims[k] = min(costs + ([float(aims[k])] if is_number(aims.get(k)) else []))
            plan = self.walk_plan(min(self.pg(WALK_TOTAL), room), only)
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
            # (an all-words store showing every match: each row carries the words it read, often in fields its title hides)
            hits = []
            told = bool((kind == "all" or self.lead_words) and not totals and rows)
            if told or (kind != "any" and not totals and rows):
                # (EB5: a fully shown search's rows are booked whatever the learned rule, unless the store takes any word;
                # the remarks stay with a store known to read every word)
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
                if isinstance(piece, list) and piece and all(isinstance(val, str) for val in piece):
                    long_strings(piece, lns)
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
        if not scripted and call_name not in permitted:
            if call_name == self.role("message") and self.asking:
                self.want_ask(raw_args)
                return "Not run (no step used): questions to the shopper are planned and sent for you after you have opened a few listings' details.", call_name, {}
            if call_name == self.role("message"):
                return ("Not run (no step used): messages to the shopper are sent for you - the planned questions, and the basket "
                        "statement right before the order. Do not message the shopper yourself."), call_name, {}
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
        if not scripted and call_name == self.role("cart_view") and call_name != self.role("cart") and not self.__dict__.get("batch_cart") \
                and self.trolley.units and self.__dict__.get("cart_shown", -1) == self.__dict__.get("cart_ver", -2) \
                and not self.__dict__.get("since_cart", 1):
            # (CB6: the newest cart call's reply listed the lines and no call of the actor's own came since - the code's
            # scripted reads aside: a view shows nothing new, and no stock)
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
        if held:
            return held, call_name, call_args
        if call_name == self.role("cart"):
            refusal = self.screen_cart(call_args, sig)
            if refusal:
                return refusal, call_name, call_args
        return None, call_name, call_args

    def screen_cart(self, call_args: dict[str, Any], sig: str) -> str:
        if self.trolley.is_put(call_args) and self.moves_left() > 4:
            tied = self.fit_hold_text([line for line, count in self.trolley.lines(call_args) if count is None or count > 0])
            if tied:
                return tied
        thing = self.trolley.aim(call_args)
        point = self.ref_pointer_note(thing) if thing and self.trolley.is_put(call_args) else ""
        pointed = self.__dict__.setdefault("pointer_told", {})
        if point and pointed.get(thing, 0) < 2:
            # (CE5: the reference's source, or a listing its chain went through, carries the excluded value)
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
            # (a line the check passed, taken out with no change or refusal since: held once per time it was put in)
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
        walk = "" if point else self.walk_hold(call_args)     # (v605d M2: no cheapest walk below a listing the reference points at)
        if walk:
            return walk
        for other, count in self.trolley.lines(call_args)[1:]:
            # (a call naming several lines: every new line is checked too)
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
            # (every line of the call is rank-checked, once per line and requirement version - again after a cheapest walk)
            if count is not None and count <= 0:
                continue
            better = self.rank_check(line_id)
            if better:
                cheaper = self.rank_verify(line_id)      # (v605f FD: the named read listings are checked at once)
                return "Not run (no step used). " + better.replace("before ordering", "before putting it in the cart").replace(
                    "order the current pick", "send the same call again") + (
                    f" The independent item check found that {cheaper} meets every requirement now in force and costs less than "
                    f"{line_id}: put {cheaper} in instead." if cheaper else "")
        for line_id, count in self.trolley.lines(call_args) if self.needs.priority() == "cheapest" else []:
            # (v605f FD: a put that a cheaper item-check pass undercuts is held once per pair)
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
        """A held-back answer for an order or decline (None lets it go), and the args to send."""
        if self.call_kind(call_name, call_args) == "order" and self.trolley.units and self.moves_left() > 4:
            tied = self.fit_hold_text(list(self.trolley.units))
            if tied:
                return tied.replace("Not run (no step used)", "Not run yet (no step used)") + " Fix that line, then order.", call_args
        if self.call_kind(call_name, call_args) == "order" and not self.trolley.units and self.role("cart") and self.moves_left() > 1:
            # (R9b: an order of an empty cart only spends a step: held twice per cart state)
            mark = ("empty", self.__dict__.get("basket_ver", 0))
            told = self.__dict__.setdefault("empty_told", {})
            told[mark] = told.get(mark, 0) + 1
            if told[mark] >= 3 and self.moves_left() >= 4 and self.wire.spare() > CLOSING:
                # (R9e: the actor keeps ordering an empty cart - the code puts its own picks in, as its own finish does)
                try:
                    filled = self.code_cart(set(), None)
                except Exception as err:
                    filled = False
                    self.log.append({"kind": "empty_fill_error", "error": f"{type(err).__name__}: {err}"[:300]})
                self.log.append({"kind": "empty_fill", "done": filled, "cart": dict(self.trolley.units)})
                if filled and self.trolley.units:
                    return ("Not run yet (no step used). The cart was empty, so the code put in a candidate for every product asked "
                            "for: " + self.trolley.summary() + " Check it against the request, then order."), call_args
            if told[mark] <= 4 or self.moves_left() > 3:
                return ("Not run yet (no step used). The cart is empty as counted, so an order would fail. Put the chosen item"
                        + (" of every product" if self.needs.items() else "") + " in the cart first (one line each, at its "
                        "quantity), then order."), call_args
        if self.call_kind(call_name, call_args) == "order" and self.moves_left() > 2:
            # (CB5: a cart line whose newest record shows it sold out: an order holding it fails - held twice per cart state,
            # then the code swaps the line itself, keeping the other lines, when the steps allow)
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
            cheaper = self.cheaper_fit(thing) if thing else ""
            for line in ([] if cheaper else [i for i in self.trolley.units if i != thing]):
                # (v605f FD: every cart line, not only the newest)
                cheaper = self.cheaper_fit(line)
                if cheaper:
                    thing = line
                    break
            told = self.__dict__.setdefault("cheaper_told", {})
            if cheaper and told.get((thing, cheaper), 0) < 2:
                told[(thing, cheaper)] = told.get((thing, cheaper), 0) + 1
                return (f"Not run yet (no step used). The independent item check found that {cheaper} meets every requirement, and "
                        f"it costs less than {thing}. The cheapest fit is asked for and the check's reading stands over yours: take "
                        f"{thing} out of the cart, put {cheaper} in, re-read its stock and price, and order it."), call_args
            if cheaper and told.get((thing, cheaper), 0) == 2 and thing in self.trolley.units and self.role("cart") \
                    and self.moves_left() >= 6 and self.wire.spare() > CLOSING:
                # (R9e: the actor kept the dearer line through two holds - the code swaps it for the cheaper verified fit itself,
                # keeping every other line and the line's units)
                told[(thing, cheaper)] = 3
                units = int(self.trolley.held(thing) or 1)
                picks = [(i, int(q)) for i, q in self.trolley.units.items() if i != thing] + [(cheaper, units)]
                try:
                    bad = self.code_put(picks)
                except Exception as err:
                    bad = None
                    self.log.append({"kind": "cheaper_swap_error", "error": f"{type(err).__name__}: {err}"[:300]})
                self.log.append({"kind": "code_cheaper", "old": thing, "new": cheaper, "refused": bad, "now": dict(self.trolley.units)})
                swapped = bad == [] and cheaper in self.trolley.units
                if swapped and thing not in self.trolley.units:
                    return (f"Not run yet (no step used). The code took {thing} out of the cart and put {cheaper} in at the same units: "
                            "the item check found it meets every requirement now in force, and the cheapest fit is asked for. "
                            + self.trolley.summary() + " Order this cart now."), call_args
        if self.call_kind(call_name, call_args) == "order" and self.trolley.units:
            walk = self.walk_hold(None)
            if walk:
                return walk, call_args
        reader = self.stock_tool() if self.call_kind(call_name, call_args) == "order" else None
        if reader and self.moves_left() > len(self.trolley.units) + 2 and self.wire.spare() > CLOSING:
            # (CB3: the re-read goes through the tool whose answers carry the stock; a line counts as read only from a record
            # with its stock after the newest cart change)
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
                    cheaper = self.rank_verify(thing)      # (v605f FD)
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
                # (a decline with a verified line in the cart: in two races 2 of 472 declines were right)
                self.decline_kept = True
                return ("Not run yet (no step used). The cart holds " + ", ".join(passed_lines) + ", which the independent check "
                        "found meets every requirement now in force. A decline would throw that away: re-read its stock and price and "
                        "order it" + (" with every product's line in the cart" if self.needs.items() else "") + ". Decline only if a "
                        "change since then clearly rules it out."), call_args
            slack = self.slack_rows() if not self.__dict__.get("slack_done") and (self.role("details") or self.batch_tool()) else []
            spread = self.batch_tool()
            reads = -(-len(slack) // spread[2]) if spread and len(slack) >= 2 else len(slack)
            if slack and self.moves_left() > reads + 4 and self.wire.spare() > CLOSING + 40:
                # (ED5: every product has a fit but the cheapest fits together pass the budget: cheaper rows within what the
                # budget leaves each product are read before any decline)
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
            if fill and self.moves_left() > reads + 3 and self.wire.spare() > CLOSING + 40:
                # (a product with no verified fit whose in-budget rows were never opened: they are read before any decline)
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
                    and self.wire.spare() > CLOSING + 40:
                # (named listings nobody opened: their records are read by the code before any decline)
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
                # (listings were read since the audit, most often the rows it named: their records are checked too)
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
            best = self.unrefuted() if not self.__dict__.get("best_held") and self.moves_left() > 4 else []
            for _ in range(3):
                # (the item check runs on an unverified pick first: a failing one drops out and the next one is tried)
                unchecked = [i for i in best if f"{i}|{self.needs.version}" not in self.passed]
                if not unchecked or self.wire.spare() < CLOSING + 30:
                    break
                self.check_item(unchecked[0], "ORDER")
                best = self.unrefuted()
            if best and self.moves_left() > len(best) + 3 and self.wire.spare() > CLOSING + 20:
                # (a decline is right only when nothing can fit, never because a budget ran out)
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
                # (CE4: an unresolved differ reference's value is unknown - it never rules a listing out or justifies a
                # decline; held once, after the decline's other checks)
                self.ref_decline_held = True
                heads = ", ".join(dict.fromkeys(r["head"] for r in soft))
                return ("Not run yet (no step used). " + heads + " is given by another listing that the code could not resolve, so its "
                        "value is unknown: it never rules a listing out and never justifies a decline. Put the listing that meets every "
                        "other requirement (in stock, within the budget) in the cart" + (", one line per product" if self.needs.items() else "")
                        + ", re-read its stock and price, and order; decline only if no listing meets the other requirements."), call_args
            repair = self.decline_repair()
            if repair:
                return repair, call_args
            near = self.near_check()        # (ED6: one meaning check before a decline after news)
            if near:
                return near, call_args
            return None, self.scrub_decline(call_args, call_name)
        if self.call_kind(call_name, call_args) == "order":
            try:
                held = self.state_gate()        # (R9b: the basket is stated to the shopper right before the order)
            except Exception as err:
                held = ""
                self.log.append({"kind": "state_gate_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if held:
                return (held, call_args)
        return None, call_args

    def dispatch(self, proposals: list[dict[str, Any]], scripted: bool = False) -> list[tuple[str, str]]:
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
        for begin in range(0, len(to_send), self.turn_cap):
            slab = to_send[begin: begin + self.turn_cap]
            if self.wire.done or self.wire.broke or self.moves_left() <= 0:
                responses += [(cid, "Not run: the episode is over or no steps are left.") for cid, _, _ in slab]
                continue
            slab = slab[: self.moves_left()]
            responses += self.send_chunk(slab)
        call_rank = {c["id"]: i for i, c in enumerate(proposals)}
        responses.sort(key=lambda req: call_rank.get(req[0], 0))
        return responses

    def send_chunk(self, slab: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str]]:
        cap = max(1, int(self.__dict__.get("turn_cap") or 16))
        if len(slab) > cap:
            # (R9g: never more calls in one request than the task allows per turn)
            out: list[tuple[str, str]] = []
            for begin in range(0, len(slab), cap):
                if self.wire.done or self.wire.broke or self.moves_left() <= 0:
                    out += [(cid, "Not run: the episode is over or no steps are left.") for cid, _, _ in slab[begin:]]
                    break
                out += self.send_chunk(slab[begin: begin + cap])
            return out
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
        """Books one answered call and returns the text the actor reads for it."""
        shown, obs = obs, lensed(obs)
        msg_tool, basket_tool = self.role("message"), self.role("cart")
        if call_name == msg_tool:
            self.msgs_sent += 1
        if call_name == basket_tool and not self.code_fault(obs):
            # (R9b: every answered cart call is a cart change the basket statement must follow)
            self.basket_ver = self.__dict__.get("basket_ver", 0) + 1
            self.calls_since_cart = 0
        else:
            self.calls_since_cart = self.__dict__.get("calls_since_cart", 0) + 1
        if call_name and call_name == self.quota_tool() and call_name not in (msg_tool, basket_tool):
            allowed = self.quota_from(obs, self.msgs_sent)
            if allowed is not None and allowed >= self.msgs_sent:
                self.msg_cap = allowed         # (R9b: the world's own count of the messages allowed)
                self.log.append({"kind": "msg_quota", "cap": allowed})
        self.last_kind = self.call_kind(call_name, call_args)
        if self.last_kind == "order":
            self.orders_tried += 1
        if call_name in (basket_tool, msg_tool) or call_name in self.finishers():
            self.last_calls = []
        else:
            self.last_calls = (self.last_calls + [call_name + as_text(call_args)])[-2:]
        self.archive.add(obs)
        self.absorb_n = self.__dict__.get("absorb_n", 0) + 1
        if call_name != basket_tool and not self.__dict__.get("covering"):
            self.since_cart = self.__dict__.get("since_cart", 0) + 1      # (CB6: the actor's own calls since the newest cart call)
        try:
            gone_now = self.learn_caps(call_name, call_args, obs)
        except Exception as err:
            gone_now = []
            self.log.append({"kind": "caps_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if self.trolley.units:
            # (when each cart line's own record was last seen: a line not seen since the last cart change is re-read; CB3:
            # once any record showed a stock field, only a record holding the line's id and a stock field counts - a search
            # row only once rows were seen to show the live stock and never seen to lag the lookups)
            fresh: set[str] = set()
            stock_rule = bool(self.__dict__.get("stock_seen"))
            asked = self.asked_ids(call_args)
            rows_ok = bool(self.__dict__.get("rows_live")) and not self.__dict__.get("rows_lag")
            lookups_only = not rows_ok and call_name not in (basket_tool, self.role("cart_view"))

            def seen_line(d: dict[str, Any]) -> None:
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
        # (v605d M8, CG5: another single-listing lookup counts as a details read does - the stock re-read of a cart line or a
        # verified fit aside)
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
            # (v605e F5: a removal has the put's form too - the remark only for a call that puts units in, while the line is in
            # the cart and not sold out)
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
            self.cart_seq = self.absorb_n      # (CB3: a refused checkout - every line's stock is read again before the next one)
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
            # (nothing verified with a third of the steps left: the decline audit runs now, while its rows can be opened)
            self.early_audit = True
            ahead = self.review_decline(early=True)
            if ahead:
                remarks.append("Nothing verified fits yet. " + ahead)
        qtext = " ".join(str(val) for val in call_args.values() if isinstance(val, str)) if call_name == msg_tool else None
        if qtext is not None and self.__dict__.get("stating"):
            qtext = "(The basket was stated to the shopper before the order; no property was asked.) " + qtext
        caught = self.shopper_lines(call_name, obs, refused)
        if caught:
            remarks.append(self.needs.hear(caught, qtext))
            self.ref_take()
        text_body = as_text(clipped(shown, SEARCH_TEXT if call_name == self.role("search") else RECORD_TEXT))
        if last and resp.get("finished"):
            text_body += " [the episode is finished]"
        if self.last_kind == "order" and self.__dict__.get("state_note"):
            text_body = self.state_note + "\n" + text_body
            self.state_note = ""
        summary = " ".join(qty_n for qty_n in remarks if qty_n)
        return text_body + ("\n[note] " + summary if summary else "")

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
            opener += "\n\nOPENING OBSERVATION:\n" + as_text(opener_obs)
            self.archive.add(lensed(opener_obs))
        overview = self.needs.overview(self.step_cap, self.msg_cap)
        if overview:
            opener += "\n\n" + overview
        opener += "\n\n" + self.status_line()
        self.chat = [{"role": "system", "content": ACT_RULES}, {"role": "user", "content": opener}]
        idle_n = fails = seq_no = 0
        turn_ok, missed = monotonic(), 0
        for _ in range(TURN_LIMIT):
            if self.wire.done or self.wire.broke or self.moves_left() <= 0:
                break
            if self.wire.spare() < LAST_CALL:
                break
            self.catch_up()
            try:
                self.name_guards()        # (R9k)
            except Exception as err:
                self.log.append({"kind": "name_guard_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if self.needs.reply_due:
                text_body, self.needs.reply_due = self.needs.reply_due, ""
                if self.reply_back(text_body):
                    continue
            if self.open_named():
                continue
            if (not self.asking or self.__dict__.get("empty_plans")) and self.sweep_step():
                continue
            if self.value_pages():
                continue
            if self.asking and self.pose_planned():
                continue
            if self.ref_step():
                continue
            try:
                hunted = self.fit_step()
            except Exception as err:
                hunted = False
                self.log.append({"kind": "fit_step_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if hunted:
                continue
            if self.walk_step():
                continue
            model_list = ACTOR if self.stuck < STUCK_LIMIT else ACTOR[1:] + ACTOR[:1]
            msg = self.wire.query_model(self.chat, model_list, 2500, self.toolset)
            if msg is None:
                # (TA4: replies without content or calls refused outright still end the loop at the second; a call load or
                # the network lost does not - the models count as gone only when none can be called, or no actor reply
                # came for DOWN_AFTER seconds over two failed turns or over one that tried two models)
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
