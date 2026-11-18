from __future__ import annotations

import json
import re
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

# ===== AOI AUDIT BLOCK BEGIN (no-op: this build records nothing and writes no file) =====
_AOI_LAST = None


def _aoi_answered(*_: Any) -> None:
    return None


def _aoi_roles(*_: Any) -> None:
    return None


def _aoi_finish_before(*_: Any) -> None:
    return None


def _aoi_finish_after(*_: Any) -> None:
    return None


def _aoi_block(*_: Any) -> None:
    return None


def _aoi_screened(*_: Any) -> None:
    return None


def _aoi_deferred(*_: Any) -> None:
    return None


def _aoi_end(*_: Any) -> None:
    return None


def _aoi_unconfirmed(*_: Any) -> None:
    return None


def _aoi_partial_view(*_: Any) -> None:
    return None
# ===== AOI AUDIT BLOCK END =====

ACTOR = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-haiku-5.5"]
READER = ["anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"]
CHECKER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
AUDITOR = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
LISTENER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
PLANNER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]

DEADLINE = 265.0
CLOSING = 70.0
LAST_CALL = 28.0
CONFIRM_KEEP = 8.0        # (v2.3 H3; v2.4 J3) the order goes with at least this much left after a confirmation: its timeout int(spare - 1) >= 7 s
CONFIRM_SLACK = 1.5       # (v2.4 J3) a confirmation's world calls end with CONFIRM_KEEP + this left: the pace wait before its send and the processing before the order's
CONFIRM_LEAST = 2         # (v2.4 J3) the confirmation message's least timeout (pack-6 run local-94404f06: 0.18 s p50, 0.74 s p90, 2.51 s at most, n=63)
CONFIRM_SEND = 11.5       # (v2.4 J3) the confirmation message goes with this much left: CONFIRM_KEEP + CONFIRM_SLACK + CONFIRM_LEAST (a store read: 3 s more, from 14.5 s)
CONFIRM_READ = 38.0       # (v2.2) its reply is read when the listener starts with this much left: LAST_CALL, 9 s for the listener, 1 s gap
CONFIRM_FAILS = 3         # (v2.2) failed confirmation sends per episode; each is sent again at once while the guards allow
CLOCK_STEP = 30
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
FIRST_ORDER_BY = 7
FIRST_ORDER_DUE = 10
IDLE_LIMIT = 3
STUCK_LIMIT = 4
SWEEP_READS = 8
WALK_PAGES = 6
WALK_TOTAL = 8
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
- The grader reads each requirement from the listing's field for that property, by the value's own words as this store writes them in that field. A field whose name says the same property counts whatever its spelling: separators, a prefix, a shorter or longer name, or a usual alternative name ("fit" for a pants fit type, "neckline" for a collar type, "dial size" for a watch dial size, "color family" for a colour). When a field carrying the property's own name exists, it decides, and so does a field of that name with one more leading word (pants length for length, clothing material for material): when either one writes the value, the requirement is met. Two fields whose names differ only by separators (size class, size_class) are one field: the requirement is met when either one writes the value. Otherwise such an alternative field decides; when no field names the property at all, the options list or the title may show the value. The description is never evidence.
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
"references": requirements whose value the request gives only by pointing at ANOTHER listing or page ("the same colour as the jumpsuit sold by X", "within 5 of the figure on the page for Y", "a colour other than the one on Z's page"), each "property (product): as shown on <how the request names that listing>" ("property (product): different from the one shown on <how the request names that listing>" when the value must differ from that listing's); also put each in "requirements" in that form. Empty when none.
"reference_targets": for each entry of "references", the listing it points at: {"property": its property, "product": its product, "seller": the shop or brand name the request gives for that listing, exactly as written, "kind": the kind of product that listing is, "details": [each other value the request names for that listing, short and in the request's words ("blue", "4gb", "net weight 167.262186 g")], "relation": "same", "differ" (the value must NOT be that listing's: set apart from, different from, other than), "within", "at most" or "at least" (how the value must relate to that listing's), "tolerance": the number for "within", else null, "unit": its unit, else ""}. Empty when none.
"decline_no_names": true when the request's decline rule forbids offering, naming or suggesting substitutes or alternatives; false when it asks you to name something; null when it gives no rule.
"roles": an object naming the tool for each role, with the exact tool names from the list, or null when no tool fits: "search", "details", "view", "message", "cart", "cart_view", "order", "decline". "message" sends a message to the shopper; "cart" adds to or takes from the cart; "order" buys or checks out; "decline" ends the errand without buying. Decide each role by what the tool DOES (its description and arguments), not by words in its name: a tool that ends the errand without buying is "decline" even when its name holds an order word.
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
"references": for each need in "needs" whose value the shopper gives only by pointing at another listing or page, the listing it points at: {"property": its property, "seller": the shop or brand name the line gives for that listing, exactly as written, "kind": the kind of product that listing is, "details": [each other value the line names for that listing, short and in the line's words], "relation": "same", "differ" (the value must NOT be that listing's: set apart from, different from, other than), "within", "at most" or "at least", "tolerance": the number for "within", else null, "unit": its unit, else ""}; [] when none.
One line can carry several changes; report each. A need on another property, even a related one (a seat height beside an overall height, a strap width beside a width), is a new requirement that replaces nothing: an earlier requirement is dropped or replaced only when the line says so about that same property. A replaced requirement is both dropped (the old) and a new need (the new). When the shopper says which of two wishes matters more, the preferred one is a need. A stated budget is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and kindly: folksy, figurative or roundabout words that name a value mean the shopper firmly wants that value; a value is excluded only when the shopper clearly says to avoid it. Words that only stress how firmly the shopper holds a value (only, nothing but, nothing short of, strictly, stays, sticking with, locked to, holding out for) are not part of it: write just the value words ("material: jute", "color: white"), never "<value> only"; only a shopper who also rules out mixes or other values in so many words adds an exclusion. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit. A new budget replaces the old one whether higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference or anything works, put the property asked about in no_preference (when the message asked about two properties, only the FIRST one). A reply that only says the shopper did not follow or understand the question ("that was rather dense", "you've lost me", "could you put it another way") is this shopper's way of having no preference when the question named the product the property belongs to: then put the property asked about in no_preference too; when several products are asked for and the question did not say which one, the property is still open (put nothing in no_preference). In a request for several products, write each need "property (product): constraint", the product taken from the line (a line about "toe style of the women's boots" gives "toe style (women's boots): ..."). The assistant's own question is never a requirement: only what the shopper says is; values named as examples in the question do not become needs. A "don't mind" answer frees only the property the assistant's message asked about; it never drops or changes a need the shopper stated earlier, and when the message asked nothing (it only told something), such an answer changes nothing. A need given by pointing at another listing keeps the line's own pointing words as its value ("finish: unlike the one shown on Y's kettle page"); the values the line names for that other listing describe that listing, not the product: they are no needs, drops or freed properties."""

READ_RULES = """- Read each requirement from the listing's field for that property, by the value's own words as this store writes them in that field: a field whose name says the same property counts whatever its spelling (separators, a prefix, a shorter or longer name, or a usual alternative name such as "fit" for a pants fit type, "neckline" for a collar type, "color family" for a colour). When a field carrying the property's own name exists, it decides, and so does a field of that name with one more leading word (pants length for length, clothing material for material): when either one writes the value, the requirement is met. Two fields whose names differ only by separators (size class, size_class) are one field: the requirement is met when either one writes the value. Otherwise such an alternative field decides; only when no field names the property may the options list or the title show the value. The description is never evidence.
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
"goes_with": the values the target's text says go together with, pair with or count the same as that value ([] when none).
A listing the text names only as what the target is not ("not the green one from X"), or as a version, size or colour sold elsewhere, is no go_to; nor is one it names for another property. A record that does not show the described details is not the target, whatever its text says."""

DECLINE_WORDS = r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without"
ORDER_WORDS = r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed"
# (v1 P1.1: generic effect grammar for reading the finishing tools - a no-purchase phrase beats an order word; finishing
# words, strong ones anywhere and weak ones only with the errand as their object in a description; the words of tools that
# read, browse, list or talk, which never finish)
NO_BUY = (r"\bwithout\s+(?:\w+\s+){0,2}?(?:buy\w*|purchas\w*|order\w*|placing|paying|check\w*\s*out)\b|"
          r"\bno\s+(?:\w+\s+)?(?:purchases?|orders?|buy\w*|sale)\b|\bnothing\s+(?:\w+\s+){0,2}?(?:bought|ordered|purchased)\b|"
          r"\b(?:buy\w*|purchas\w*|order\w*)\s+nothing\b|(?:\bnot|n't)\s+(?:\w+\s+)?(?:buy\w*|purchas\w*|order\w*|plac\w*)\b|"
          r"\bstand\w*\s*down\b|\babstain\w*|\bdeclin\w*|\bgiv\w*\s*up\b|\bempty[\s-]*handed\b|\bcancel\w*|\bforgo\w*|\brefus\w*|"
          r"\babandon\w*|\babort\w*|\bwithdraw\w*|\breject\w*|\bwalk\w*\s+away\b|\bskip\w*\s+(?:the\s+)?(?:buy\w*|purchas\w*|order\w*)")
BUY_WORDS = r"\b(?:buy\w*|purchas\w*|order\w*|check\w*\s*out\w*|checkout\w*|pay\w*|plac(?:e|es|ed|ing)|submit\w*|commit\w*)\b"
FINISH_STRONG = r"\b(?:finish\w*|final\w*|conclud\w*|terminat\w*|wrap\w*\s*up|checkout)\b"
FINISH_WEAK = r"\b(?:ends?|ending|exit\w*|quit\w*|complet\w*|clos(?:e|es|ing)|leav(?:e|es|ing)|stop\w*|done)\b"
ERRAND_OBJ = r"(?:\W+\w+){0,3}?\W+(?:errand|task|session|episode|shopping|store|shop|run|job|request|conversation|visit|here)\b"
BUSY_WORDS = (r"\b(?:view\w*|show\w*|list\w*|status|see|look\w*|track\w*|histor\w*|read\w*|inspect\w*|detail\w*|info\w*|"
              r"search\w*|find\w*|brows\w*|explor\w*|sift\w*|filter\w*|weigh\w*|compar\w*|usage|budget|account\w*|report\w*|"
              r"messag\w*|chat\w*|ask\w*|say\w*|tell\w*|talk\w*|repl\w*|question\w*|check)\b(?!\s*out)")
FINISH_CHECK = """These tools may end a shopping errand. Decide by what each tool DOES - its description and its arguments - never by words in its name.
Answer with one JSON object only: {"order": the exact name of the tool that places the current cart as the order (buys it), "decline": the exact name of the tool that ends the errand WITHOUT buying anything}. Use null when no tool listed does that. Name one tool for both only when a choice argument of that tool picks between buying and not buying."""


def name_words(name: Any) -> str:
    """v1: a tool or argument name as lower-case words (camelCase, PascalCase, snake_case, kebab-case, digits)."""
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", str(name or ""))
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return " ".join(w for w in re.split(r"[^A-Za-z0-9]+", text) if w).lower()
SORRY = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
ID_LIKE = r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}"
CODE_LIKE = r"[A-Za-z0-9_\-]+"
KEPT_LINE = ("The last message is kept: the order call sends the basket confirmation first (the code shows the shopper the "
             "basket and asks for the yes before the order goes).")      # (v2.5 V1: told while kept_open())


def as_text(piece: Any, ceiling: int = VIEW_CHARS) -> str:
    try:
        text_body = json.dumps(piece, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text_body = str(piece)
    return text_body if len(text_body) <= ceiling else text_body[:ceiling] + "...(cut)"


def clipped(piece: Any, chars: int, lvl: int = 0) -> Any:
    """Returns a copy of nested data with every long string shortened.
    Strings beyond chars get cut and end in an ellipsis; dicts and lists recurse.
    Recursion stops past depth 8 and yields None there; other values pass through."""
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
    """Pulls a single JSON dict out of free-form model text, or None on failure.
    Strips a fenced code wrapper, then tries the whole text and the outer braces span.
    Only a parsed dict counts; non-string input or unparsable text gives None."""
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
    """Gathers wordy strings (four or more words) from a nested structure.
    Appends matches to the collected list in place, looking at only ten list items per level.
    Stops descending below depth 4 and ignores dict values entirely."""
    if lvl > 4:
        return
    if isinstance(piece, str):
        if len(piece.split()) >= 4:
            collected.append(piece)
    elif isinstance(piece, list):
        for thing in piece[:10]:
            long_strings(thing, collected, lvl + 1)


def walk_dicts(piece: Any, visit, lvl: int = 0, ceiling: int = 6) -> None:
    """Calls a visitor on every dict found inside a nested dict and list tree.
    Depth is capped by ceiling, so deeper content is never visited.
    Only dict or list children are descended into; the visitor result is unused."""
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
    """Tells whether a field name is generic bookkeeping rather than a product trait.
    Splits camel case, drops any parenthesised tail, then matches words like brand or price.
    Also flags bare names such as id, stock, make or links; returns a bool."""
    part = re.split(r"\s*\(", str(key))[0]
    words = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower()))
    return bool(re.search(r"\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|"
                          r"heading|listing|description|overview|price|cost|amount|availability|sku|url)\b", words)
                or re.fullmatch(r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)", words))


def prop_key(name: Any) -> str:
    """Reduces a property name to a compact canonical form for comparison.
    Drops text after a parenthesis, splits camel case, lowercases and glues the words.
    Trailing plural s is removed from words longer than three letters (not ss)."""
    part = re.split(r"\s*\(", str(name))[0]
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower())
    return "".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words)


def edits_within(a: str, b: str, most: int) -> bool:
    """Checks whether two strings are within a given number of single-character edits.
    Runs a row-by-row edit distance and bails out early once a whole row exceeds most.
    Returns False at once when the lengths differ by more than most."""
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
    """Decides whether two property names probably denote the same trait.
    Compares canonical keys: equal, near-equal with a shared three-letter start, or prefix.
    Fuzzy edit tolerance grows from one to two with key length; empty keys never match."""
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
    """Splits a property name into singularised lowercase words.
    Ignores text after an opening parenthesis and separates camel case first.
    Words over three letters lose a single trailing s, unless it ends in ss."""
    part = re.split(r"\s*\(", str(name))[0]
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]


def spelled_alike(lo: str, hi: str) -> bool:
    """Detects a likely typo between a shorter and a longer lowercase word.
    Equal lengths must differ by one adjacent swap; longer by one or two dropped letters.
    Only vowels or doubled letters in the longer word are tried for removal."""
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
    """A stricter test that two property names match, used when fuzzy matching is risky.
    Accepts equal keys, typo variants sharing three starting letters, or one short extra word.
    The extra trailing word must be three letters or fewer; returns a bool."""
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
    """Tokenizes text into lowercase words and decimal numbers for matching.
    Filler words and, the, a, an and of are dropped; None becomes empty.
    Returns the token list in original order."""
    return [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text or "").lower()) if w not in CH_SKIP]


def ch_word_in(word: str, pool: list[str]) -> bool:
    """Checks whether a word appears in a pool of tokens, allowing near forms.
    Accepts exact matches, singular and plural variants, or one typo in long words.
    The typo rule needs six or more letters, a shared first pair and edit distance one."""
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
    """Turns a stated value into alternatives, each one a list of words.
    Removes provenance tags, quotes and emphasis words, then splits on the word or.
    Alternatives with no words are discarded; returns a list of token lists."""
    plain = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", str(value or "").lower()).replace('"', " ")
    plain = re.sub(r"\b(?:only|just|exactly|strictly)\b", " ", plain)
    return [ws for ws in (ch_words(part) for part in re.split(r"\bor\b", plain)) if ws]


def ch_writes(value: Any, text: Any) -> bool:
    """Says whether the given text spells out the stated value.
    Expands the value into alternatives and matches them against the text tokens.
    Returns True when any alternative is fully present."""
    return ch_match(ch_alts(value), ch_words(text))


def ch_match(alts: list[list[str]], pool: list[str]) -> bool:
    """Tests whether any alternative word list is fully present in a token pool.
    First requires every word to be found; otherwise tries gluing neighbouring tokens.
    Gluing applies for multi-word or numeric targets; an empty pool gives False."""
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
    """Looser check that the text mentions at least part of the stated value.
    True on a shared word, a standalone matching number, or a three-letter prefix overlap.
    Used where partial contact matters; walks every alternative and word in turn."""
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
    """Splits text into clauses on commas, semicolons or vertical bars.
    Pieces are trimmed and lowercased; blanks are dropped.
    If nothing splits off but the text is non-empty, the whole text is returned alone."""
    low = str(text or "").strip().lower()
    return [p.strip() for p in re.split(r"[,;|]", low) if p.strip()] or ([low] if low else [])


def ch_sides(part: Any) -> list[str]:
    """Splits a clause on plus signs into its sides.
    Returns the trimmed sides only when there are at least two.
    A clause without a plus sign gives an empty list."""
    sides = [s.strip() for s in re.split(r"\s*\+\s*", str(part)) if s.strip()]
    return sides if len(sides) >= 2 else []


def figure_written(fig: str, said: str) -> bool:
    """Checks that a number is stated in some text as a standalone figure.
    Looks for the figure without adjoining digits; decimals count immediately.
    A whole number counts if no word follows or the next word is short (three letters)."""
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
    """Decides whether a requirement value asks for something rather than excluding it.
    Plain yes or no style answers count, and so do known literal values.
    Otherwise bounds are stripped; rich content fails, and exclusion words need a leading cue."""
    low = str(value).lower().strip()
    if re.fullmatch(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*", low):
        return True
    if LITERAL_VALUES and " ".join(re.findall(r"[a-z0-9]+", re.sub(r"\((?:stated|revealed[^)]*)\)", " ", low))) in LITERAL_VALUES:
        return True

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
    """Checks whether a requirement value is backed by the sample text.
    Returns True on a content word stem found in the samples, or on a numeric run in order.
    Empty samples give False; connector words are removed first."""
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
    """Builds the set of singularised content words in some text.
    Takes words of three or more letters, skipping connector words, and trims a final s.
    The result is a set of singular lowercase strings, useful for overlap tests."""
    return {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", str(text).lower()) if w not in LINK_WORDS}


def kind_tokens(text: Any) -> list[str]:
    """Lists the content words of a product name in order of appearance.
    Possessive endings are removed and connector words are skipped.
    Words are lowercase and at least three letters long."""
    return [w for w in re.findall(r"[a-z]{3,}", re.sub(r"'s\b", "", str(text).lower().replace("’", "'"))) if w not in LINK_WORDS]


def plural_forms(word: Any) -> set[str]:
    """Generates likely singular and plural spellings of a word.
    Adds s and es, handles y to ies, f and fe to ves, and the reverse endings.
    Returns the set with anything under three letters removed."""
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
    """Says whether two words are the same or only differ by plural form.
    Compares equality, then membership in the other word's generated variants.
    The result says whether the two words count as one kind of thing."""
    return a == b or str(b) in plural_forms(a) or str(a) in plural_forms(b)


def kind_in(word: Any, pool: Any) -> bool:
    """Checks whether a word matches any member of a pool, ignoring plural form.
    Delegates each comparison to same_kind.
    Returns True on the first hit."""
    return any(same_kind(word, other) for other in pool)


def kinds_within(words: Any, pool: Any) -> bool:
    """Checks that every word in a collection is present in a pool.
    Each word is looked up with plural-tolerant matching via kind_in.
    Returns True only when none is missing; an empty collection passes."""
    return all(kind_in(w, pool) for w in words)


def kind_score(own: Any, words: Any, shared: Any) -> float:
    """Computes a weighted overlap between own words and a word set.
    Each own word found among words adds 1.0, or 0.5 if it is also in shared.
    Returns the float total; words not found add nothing."""
    return sum(0.5 if kind_in(w, shared) else 1.0 for w in own if kind_in(w, words))


def name_pattern(name: Any) -> str:
    """Builds a regex source that matches a product name in free text.
    Each content word becomes an alternation of plural forms, longest first, plus possessive.
    Words are joined by spaces, apostrophes or hyphens; returns the pattern string."""
    pats = []
    for w in kind_tokens(name):
        forms = sorted(plural_forms(w), key=lambda x: (-len(x), x))
        pats.append(r"(?:" + "|".join(re.escape(x) for x in forms) + r")(?:'s\b|'|\b)")
    return r"[\s'-]+".join(pats)


VALUE_LINKS = r"at least|at most|no more than|no less than|more than|less than|up to|or more|or less|or larger|or smaller|" \
              r"or higher|or lower|minimum|maximum|between|within|under|over|above|below|exactly|only|either|neither|nor|than|plus"


def one_prop_hint(hint: str) -> str:
    """Rewrites a hint naming several slash-separated properties into a single form.
    Keeps the first name and any trailing parenthesised tag, listing the others as alternatives.
    Returns the input unchanged when there is no slash, no colon or a failure."""
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
    """Finds a leading adjective from a detail that the request words before the product.
    Reads the first word of detail, then looks for it right before the product noun in request.
    Returns that word, or an empty string if absent, equal to the product or on error."""
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
    """Lists parts of a requirement that the shopper request text never states.
    Skips references, restated yes or no values and numbers that are clearly written.
    Returns missing content words and unwritten numeric runs; empty means it is covered."""
    req = str(req)
    head, _, value = req.partition(":")
    low = value.strip().lower()
    if not low or req in refs or re.search(r"as shown|same as|the page|listing|sold by|redirect|refer", low):
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


REF_WORDS = r"\bas shown\b|\bshown on\b|\bsame as\b|\bpage\b|\bsold by\b|\bredirect|\brefer(?:s|red|ring)?\b|\blisting points\b|\bpoints you\b"
REF_STEPS = 0.4
REF_CALLS = 12
REF_HOPS = 8
DIFFER_WORDS = (r"\bset apart from\b|\bapart from\b|\bdistinct from\b|\bdifferent (?:from|to|than)\b|\bdiffer(?:s|ent|ing)?\s+from\b"
                r"|\bother than\b|\bunlike\b|\bnot the same as\b|\banything but\b")
POINT_WORDS = r"\bwithin\b|\bat (?:most|least)\b|\bno (?:more|less) than\b|\bsame as\b|\bas shown\b|\bthe value\b|\bthe one\b"
REF_DIFFER_STEPS = 0.3
REF_DIFFER_BLIND = 5
SELLER_KEYS = r"shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by"


def ref_text(value: Any) -> bool:
    """Tells whether a value points elsewhere instead of stating something itself.
    Searches lowercased text for pointer phrases such as same as or sold by.
    The result is True when any pointer phrase is present in the value."""
    return bool(re.search(REF_WORDS, str(value or "").lower()))


def ref_toks(text: Any) -> list[str]:
    """Tokenizes text into normalised comparison tokens, merging numbers with units.
    Drops connector words and single letters, and singularises longer plural tokens.
    Returns the list of tokens in order."""
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
    """Extracts number and unit pairs from text as a set.
    Handles slash-separated figures sharing one unit, and singularises the unit.
    Connector-word units are skipped; any error returns what was gathered so far."""
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
    """Normalises a name for loose comparison.
    Lowercases, removes possessive endings and keeps only letters and digits.
    The result is a compact string used as a loose comparison key."""
    return re.sub(r"[^a-z0-9]", "", re.sub(r"'s\b", "", str(text or "").lower()))


def ref_relation(value: Any) -> tuple[str, float | None, str]:
    """Classifies how a value relates to a referenced one.
    Returns a triple: differ, within with a number and unit, at most, at least, or same.
    Phrases are detected by pattern; the default is same with no number."""
    low = str(value or "").lower()
    if re.search(DIFFER_WORDS, low):
        return "differ", None, ""
    hit = re.search(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?", low)
    if hit:
        return "within", float(hit.group(1)), (hit.group(2) or "").strip()
    if re.search(r"\b(?:at most|no more than|up to|or less|or lower|or smaller)\b", low):
        return "at most", None, ""
    if re.search(r"\b(?:at least|no less than|or more|or higher|or larger)\b", low):
        return "at least", None, ""
    return "same", None, ""


def ref_parse(text: Any, ok: Any = None) -> dict[str, Any] | None:
    """Parses a requirement sentence into seller, product kind and detail phrases.
    Finds a capitalised name past the start, optionally approved by the ok callback.
    Returns a dict with seller, kind (80 chars max) and up to six details, or None."""
    try:
        body = re.split(r"\(\s*or\b", str(text or ""))[0]
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
    """Checks whether a property name appears as consecutive words in some text.
    Splits camel case and ignores any parenthesised suffix; very short names never match.
    Unless exact is set, a word may differ from its plural spelling."""
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
    """Returns the text that follows a property name inside a source passage.
    Looks for the name words in order, then takes up to 500 characters after them.
    Cuts at an alternative marker or sentence end; returns empty on no match or error."""
    try:
        words = [w for w in re.findall(r"[a-z0-9]+", re.split(r"\s*\(", str(prop))[0].lower()) if w not in LINK_WORDS]
        hit = re.search(r"\b" + r"\W+".join(map(re.escape, words)) + r"\b", str(source).lower()) if words else None
        if not hit:
            return ""
        return re.split(r"\(\s*or\b|(?<=[.!?])\s", str(source)[hit.end(): hit.end() + 500])[0]
    except Exception:
        return ""


def lensed(piece: Any) -> Any:
    """Adds common field aliases to record dicts so later code finds them.
    Works on a deep copy, only for dicts carrying an id-like string value.
    Fills price, in_stock or title from a single unambiguous candidate field when missing."""
    try:
        piece = json.loads(json.dumps(piece))
    except (TypeError, ValueError):
        return piece

    def visit(d: dict[str, Any]) -> None:
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


class ArgCheck:

    """Validates and gently repairs tool-call arguments against a JSON schema.
    Holds type tests, coercions (numeric strings, case-fixed enums) and a rule table.
    Rules cover bounds, lengths, item counts, nested items and properties."""
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


class Wire:

    """Client-side session manager that talks to the model proxy and the environment.
    Tracks elapsed time, spend, step counter, pacing, banned models and a log of failures.
    Offers retrying model queries, prompt cache marking and environment action posts."""
    def __init__(self, bind: dict[str, Any]):
        self.t0 = monotonic()
        self.bind = dict(bind)
        self.model_list = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
        self.world = ProxyClient(timeout=95, max_retries=1, retry_delay=1)
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
        return DEADLINE - (monotonic() - self.t0)

    def spent_out(self) -> bool:
        return self.spend >= SPEND_CAP * self.__dict__.get("scale", 1.0)

    def send(self, client: Any, path: str, body: dict[str, Any]) -> Any:
        """Posts a request through a client after honouring pacing delays.
        Waits for the normal or slowed gap, and for model resume time on inference paths.
        A 429 or 503 failure switches to slower pacing; returns the raw response."""
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
        """Computes a retry delay in seconds with exponential growth and jitter.
        Doubles per try up to eight seconds, with a larger base for status 429.
        Jitter is derived deterministically from the seed text, tries and step count."""
        mix = (len(seed) * 7 + sum(ord(ch) for ch in seed[-64:]) + 31 * tries + 13 * self.tick) % 101 / 100.0
        return min(8.0, (2.0 if status == 429 else 1.0) * 2 ** tries) * (0.75 + 0.5 * mix)

    def down(self, at: float | None = None) -> bool:
        """Reports whether the model service looks unavailable.
        True if marked absent, or if misses exist and no reply came within the allowed window.
        The optional at argument sets the time used for the check."""
        return self.no_model or (self.misses > 0 and (monotonic() if at is None else at) - self.last_reply > DOWN_AFTER)

    def usable(self, model_list: list[str], tools: bool = False) -> bool:
        return not self.no_model and not self.mute and any(
            mdl not in self.banned and not (tools and mdl in self.tool_banned) for mdl in list(model_list) + self.spare_models)

    @staticmethod
    def plain(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flattens structured message content into plain text.
        For each message whose content is a list, joins the text pieces with blank lines.
        Returns new message dicts and leaves the inputs unchanged."""
        out = []
        for msg in msgs:
            msg = dict(msg)
            if isinstance(msg.get("content"), list):
                msg["content"] = "\n\n".join(str(part.get("text") or "") for part in msg["content"] if isinstance(part, dict))
            out.append(msg)
        return out

    def query_model(self, msgs: list[dict[str, Any]], model_list: list[str], token_cap: int,
            toolset: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
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
                end = self.__dict__.get("model_end", 0.0)
                if end:
                    # (v2.2 F2: the basket confirmation's listener ends with `end` left, after its pause and the request gap,
                    # and gets 9 s at least - live listener calls: p99 7.5 s, max 11.8 s - else it is not made)
                    room = spare - max(0.0, self.model_resume - monotonic()) - SLOW_GAP - end
                    if room < 9:
                        return self.failed(kinds)
                    self.model_list.timeout = int(min(self.model_list.timeout, room))
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
                        _aoi_answered(mdl)
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
        """Records why a model query ended without an answer and returns None.
        Sets call_fail to empty if an empty reply was seen, or refused if every failure was hard.
        Otherwise it clears the reason."""
        self.call_fail = "empty" if "empty" in kinds else "refused" if kinds == {"hard"} else ""
        return None

    @staticmethod
    def cached(msgs: list[dict[str, Any]], mdl: str) -> list[dict[str, Any]]:
        """Marks messages for prompt caching on Anthropic models.
        Needs a model name with the anthropic prefix and more than two messages.
        Wraps the first and last string contents as text blocks with an ephemeral cache flag."""
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
        """Sends one or several actions to the environment and returns its reply.
        Handles step bookkeeping, retries, resending the last request and shrinking timeouts.
        Sets done on finish or broke on payment failure; returns the data dict or None."""
        env_body: dict[str, Any] = dict(self.bind)
        env_body["step"] = self.tick + 1
        if len(acts) == 1:
            env_body["action"] = acts[0]
        else:
            env_body["actions"] = acts
        again = bool(self.__dict__.get("again")) and isinstance(self.__dict__.get("last_post"), dict)
        if again:
            # (v2.4 J2: the newest request, whose outcome is unknown, once more - the same step and the same actions: the world
            # answers it from its record if it ran it, or runs it once if it never did)
            env_body = dict(self.last_post)
        tries = reused = 0
        keep = self.__dict__.get("world_end", 0.0)      # (v2.2 F2: a basket confirmation's call ends with this much left)
        least = self.__dict__.get("world_least", 3)     # (v2.4 J3: the least timeout of a call that keeps time for another)
        self.posted = 0                                 # (v2.3 H2: requests this call sent; with none, nothing reached the world)
        for _ in range(3 + WORLD_TRIES):
            spare = self.spare()
            if spare < 2 or (keep and spare - keep < least):
                return None
            self.world.timeout = int(max(least, min(95, spare - 1, spare - keep if keep else 95)))
            self.posted += 1
            self.last_post = dict(env_body)
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
                if again:
                    return None      # (v2.4 J2: never another step - a new step would be a second message)
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
            if self.spare() - wait < (keep + least if keep else 2):
                return None      # (v2.4 J3: a pause never takes the time a call keeps for another - the order's)
            sleep(wait)
        return None


class Records:

    """Store of every observation seen, with the ids and labels found inside them.
    Tracks which listings were opened in detail and which shelf label each id carries.
    Provides price lookup, listing cards, shelf stats and a property menu."""
    def __init__(self):
        self.items: list[Any] = []
        self.ids: set[str] = set()
        self.read: set[str] = set()
        self.label_of: dict[str, str] = {}

    def add(self, obs: Any) -> None:
        self.items.append(obs)

        def visit(d: dict[str, Any]) -> None:
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
        """Looks up the latest price recorded for a given listing id.
        Scans stored observations for dicts holding that id and a numeric price or cost field.
        Returns the last match as a float, or None."""
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
        """Collects raw snippets of observations that mention a given listing id.
        Finds dicts having the id as a value, clips them and removes duplicates.
        Returns the newest hold snippets, after gathering at most eight."""
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

    def cards(self, char_cap: int = 30000, cheaper_than: str = "", skip: set[str] | None = None, first: Any = ()) -> list[str]:
        """Builds compact text cards for the known listings within a size budget.
        Merges dicts per id, can skip ids or drop pricier ones than a reference listing.
        Favors first ids and opened listings, deprioritises out-of-stock ones when space is short."""
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        def visit(d: dict[str, Any]) -> None:
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
        """Finds the shelf or category label that belongs to a record dict.
        Uses its own category-like short text field, otherwise a label remembered for its id.
        Returns an empty string if nothing is found."""
        own = next((val.strip() for key, val in d.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                    and re.search(r"categor|shelf|department|aisle", str(key).lower()) and "id" not in str(key).lower()), "")
        if own:
            return own
        labels = self.__dict__.get("label_of") or {}
        return next((labels[val] for key, val in d.items() if isinstance(val, str) and val in labels and "id" in str(key).lower()), "")

    def rich_reads(self, shelf: str = "") -> int:
        """Counts opened listings that carry rich detail.
        A listing is rich with a nested map of three or more entries, or five short text fields.
        An optional shelf restricts the count to that label, ignoring case."""
        rich: set[str] = set()

        def visit(d: dict[str, Any]) -> None:
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
        """Ranks shelf labels by how many opened listings each has.
        Only listings with a nested dict and a known label are tallied.
        Returns labels sorted by descending count."""
        tally: dict[str, int] = {}

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.read), None)
            shelf = self.label_for(d) if ident else ""
            if ident and shelf and any(isinstance(val, dict) for val in d.values()):
                tally[shelf] = tally.get(shelf, 0) + 1
        for rec in self.items:
            walk_dicts(rec, visit)
        return sorted(tally, key=lambda s: -tally[s])

    def prop_menu(self, shelf: str = "") -> tuple[str, int]:
        """Builds a text menu of the commonly seen property fields in stored listings.
        Tallies short text fields on detailed dicts, optionally limited to one shelf.
        Returns the top 28 lines with example values, plus the number of dict sets found."""
        tallies: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        sets = 0
        mapped, row_tally = 0, {}

        def on_shelf(d: dict[str, Any]) -> bool:
            return any(isinstance(val, str) and val.strip().lower() == shelf.lower() and re.search(r"categor|shelf|department|aisle", str(key).lower())
                       for key, val in d.items()) or self.label_for(d).strip().lower() == shelf.strip().lower()

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
            for fkey in [fkey for fkey in tallies if row_tally.get(fkey, 0) >= tallies[fkey]]:
                tallies.pop(fkey)
        top_props = sorted(tallies, key=lambda fkey: -tallies[fkey])[:28]
        return "\n".join(f"- {fkey} (in {tallies[fkey]} listings): " + " | ".join(examples.get(fkey, [])) for fkey in top_props), sets

    def label_lines(self) -> str:
        """Summarises small-vocabulary text fields across stored records.
        Counts values per field in dicts with four or more keys, ignoring code-like values.
        Keeps fields with two to eight distinct values and returns up to twelve lines."""
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


class Needs:

    """Holds what is known about the shopper request and what it still requires.
    Keeps the parsed reading, stated and revealed requirements, dropped ones and exchanges.
    Offers accessors for items, quantities, priority and the requirement text."""
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
        self.sweepable = False
        self.resolved: list[str] = []
        self.ref_news: list[dict[str, Any]] = []

    def items(self) -> list[dict[str, Any]]:
        """Returns the cleaned list of products when the request names several.
        Each row has product text, an integer quantity of at least one and colon-bearing requirements.
        Fewer than two valid rows yields an empty list."""
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
            return 1
        need_n = self.reading.get("quantity")
        return need_n if isinstance(need_n, int) and not isinstance(need_n, bool) and need_n >= 1 else 1

    def counts(self) -> list[int]:
        """Lists the quantity wanted for each product.
        Uses each item row quantity, or the single overall quantity if there are no item rows.
        The result is a list of integer quantities, one per product line."""
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
        """Lists property requirements that remain open.
        Starts from the current set or the stated plus revealed ones, minus those let go.
        Excludes price, budget, quantity, stock and cents wording; returns strings."""
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
        """Builds a note telling the model which requirements the shopper withdrew.
        Returns an empty string when let_go holds nothing to report.
        Otherwise lists each dropped item and says they must stop influencing choices."""
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
        """Finds needs whose property name only appears in the shopper's lines as a longer form.
        Matches each word-run in the lines by prefix against the need head, skipping exact hits.
        Returns a dict mapping the original need to a rewritten need using the spoken wording."""
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
        """Decides whether a dropped requirement was really released by the shopper's words.
        Looks up the stated need among written and uncovered ones, then compares phrases from lns.
        Can also confirm via store field values; returns True when unsure or on any error."""
        try:
            name = str(old_need).split(":", 1)[0]
            tag = re.findall(r"\(([^)]*)\)", name)

            def fits_tag(w: str) -> bool:
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
        """Rewrites needs in the listener answer so values use the store's own spelling.
        For each needs value without digits, asks field_hook for the property's observed values.
        Swaps matches into needs and requirements_now in place, and logs the change."""
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
        """Reports whether a property already has a firm, stated value in the written needs.
        Skips released lines, mismatched kind tags and values merely guessed from the story.
        Returns True on a same-property match; returns False on no match or any error."""
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
        """Checks whether any need in the list is actually voiced in the shopper's lines.
        Requires every property word and every value word of one need to occur in lns.
        Returns True on the first such need, else False; non-list input counts as empty."""
        said = " ".join(str(ln) for ln in lns).lower()
        for need in needs if isinstance(needs, list) else []:
            head, _, value = str(need).partition(":")
            names = [w for w in prop_words(head) if len(w) >= 3]
            values = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in LINK_WORDS]
            if names and values and all(w in said for w in names + values):
                return True
        return False

    def hear(self, lns: list[str], qtext: str | None) -> str:
        """Feeds the shopper's new reply lines to the listener model and merges what it learned.
        Updates uncovered needs, indifferent props, let_go drops, budget cap and the version counter.
        Returns a meaning summary with remarks, or an empty string if nothing was read."""
        new_lines = [ln for ln in lns if ln not in self.caught]
        if not new_lines:
            self.read_miss = "the reply repeats a line already read"      # (v2.2 F2: why the newest reply went unread, '' once read)
            return ""
        self.caught.update(new_lines)
        if self.wire.spare() < self.__dict__.get("read_floor", CLOSING):     # (v2.1: the confirmation's reply sets its own floor)
            self.read_miss = f"{self.wire.spare():.1f} s left when the listener would start"
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
        self.read_miss = "" if answer else "the listener got no answer" if self.wire.call_models else "the listener did not run"
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

    """Tracks the shopping cart as the agent believes it stands, from tool calls and replies.
    Holds the cart tool name, its argument schema, per-item unit counts and pending additions.
    Learns whether calls add or replace, and flags lines that disagree with the request."""
    def __init__(self, the_tool: str | None, jschema: dict[str, Any]):
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
        return as_text({fkey: val for fkey, val in call_args.items() if isinstance(self.prop_specs.get(fkey), dict) and "enum" in self.prop_specs[fkey]})

    def line_spec(self) -> tuple[str, str, str] | None:
        """Finds an array argument in the cart schema whose rows carry an id and a count.
        Prefers string id fields containing id, and takes the first numeric field as the count.
        Returns the triple of array key, id field, count field, or None if no such shape."""
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
        """Extracts the item lines a cart call targets, as pairs of id and quantity.
        Reads the array argument when the schema has one, otherwise a single flat id argument.
        Quantity is None when the row gives no number; blank ids are skipped."""
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
        """Picks the single item id a cart call is mainly about.
        Prefers lines with a missing or positive count, falling back to any line found.
        Returns None when the call names no item."""
        found = self.lines(call_args)
        put = [ln for ln in found if ln[1] is None or ln[1] > 0]
        return (put or found)[0][0] if found else None

    def aim_flat(self, call_args: dict[str, Any]) -> str | None:
        """Looks for a single bare item id among a cart call's plain arguments.
        Accepts the first non-empty string value with no spaces whose schema is not an enum.
        Returns that string, or None when no argument qualifies."""
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
        """Updates the tracked cart after a cart tool call and its observation.
        Handles array-style replacement and single-item add, put or remove, using the prior units.
        Reconciles with any cart view in the reply; returns a summary string, empty on error."""
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
        if seen is not None and set(seen) == {thing} and not self.listed(obs, thing):
            # (v2.2 F3: one record naming only the line this call changed, in no list, acknowledges that line: its units
            # now, the other lines as counted - a cart listing is a list of line records)
            self.learn(seen, thing)
        elif seen is not None:
            self.learn(seen)
        elif (count <= 0 or not self.is_put(call_args)) and self.shows_empty(obs, thing):
            self.units.pop(thing, None)
            self.pending.pop(thing, None)
        return self.summary()

    def shows_empty(self, obs: Any, thing: str) -> bool:
        """Tells whether a reply for a removed item looks like an empty cart listing.
        Needs every list field empty, all numeric fields zero, and no other held item mentioned.
        Returns False when the reply is not a dict or the item is not tracked."""
        if not isinstance(obs, dict) or thing not in self.units:
            return False
        lists = [v for v in obs.values() if isinstance(v, list)]
        nums = [v for v in obs.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        text_body = as_text(obs)
        return bool(lists) and all(not v for v in lists) and bool(nums) and all(v == 0 for v in nums) \
            and not any(other in text_body for other in self.units if other != thing)

    @staticmethod
    def listed(piece: Any, thing: str, inside: bool = False, lvl: int = 0) -> bool:
        """v2.2 (F3): a record naming `thing` sits in a list in the answer (a cart listing, not one line's acknowledgement)."""
        if lvl > 6:
            return False
        if isinstance(piece, dict):
            if inside and thing in piece.values():
                return True
            return any(CartBook.listed(val, thing, inside, lvl + 1) for val in piece.values() if isinstance(val, (dict, list)))
        if isinstance(piece, list):
            return any(CartBook.listed(val, thing, True, lvl + 1) for val in piece)
        return False

    @staticmethod
    def whole(val: Any) -> int | None:
        """(v2.3 H1) a whole number as a cart row may write its units: an int, a float without a fraction or a string of up
        to six digits (a '.0' tail allowed); None for anything else (a bool, a fraction, words, a list)."""
        if isinstance(val, bool):
            return None
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val) if val.is_integer() else None
        if isinstance(val, str) and re.fullmatch(r"\s*\d{1,6}(?:\.0+)?\s*", val):
            return int(float(val))
        return None

    @staticmethod
    def ident(d: dict[str, Any], known: set[str]) -> str:
        """(v2.3 H1) the line a cart row names: a known line id in any field, else a one-word id-like string in a field whose
        name holds 'id'; '' when it names none."""
        got = next((val for val in d.values() if isinstance(val, str) and val in known), "")
        return got or next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and " " not in val
                            and re.fullmatch(ID_LIKE, val)), "")

    @staticmethod
    def rows_left(obs: Any, known: set[str], done: dict[int, str], listings: Any = ()) -> list[str]:
        """(v2.4 J1) the listing ids of an answer that have no units of their own, by each place one appears. An id
        appears where a dict holds it: as a field value - a counted line (known), or one id-like word in a field whose name
        holds 'id' (Records.add's listing-id test); as a dict key or an item of a list held in a field - a counted line, or
        an id the store wrote as a listing id (listings: the archive's ids, by that test). It has units of its own only when
        a dict that gave THAT id its units (view_units: done) is the dict holding it, a dict holding that one (up to the
        nearest list item or the top), or a dict inside it (not through a list); a key also by its own value. One decoded
        dict never excuses another id - a sibling's, a parent's or a second one of its own. (v2.3 H1, kept: in a list
        item, a dict holding no such id where nothing on its path decoded, with a one-word id-like string next to a whole
        number in a field not named for a price - a row naming no line: its first such string.) (v2.4.1 N1: an item of a
        list counts at any nesting - a list inside a list, at the top or held in a field - by the same test as a list item;
        a list row such as [id, units] is never decoded (view_units reads dicts only), so its id makes the answer partial.
        Anything left to read past the depth limit (6) makes the answer partial too - DEEP stands for it.)"""
        left: list[str] = []
        DEEP = "(an entry nested deeper than 6 levels)"

        def is_id(val: Any, key: Any = None) -> bool:
            if not isinstance(val, str):
                return False
            if val in known:
                return True
            if key is None:
                return val in listings
            return "id" in str(key).lower() and re.fullmatch(ID_LIKE, val) is not None

        def note(thing: str) -> int:
            if thing not in left:
                left.append(thing)
            return 1

        def leaves(seq: list[Any], lvl: int) -> list[str]:
            """(v2.4.1 N1) the strings of a list at `lvl` and of every list inside it, at any nesting (not inside a dict:
            its own walk reads those); a list inside it past the depth limit makes the answer partial."""
            out: list[str] = []
            for item in seq:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, list):
                    if lvl + 1 > 6:
                        note(DEEP)
                    else:
                        out += leaves(item, lvl + 1)
            return out

        def walk(piece: Any, above: list[dict[str, Any]], row: bool, lvl: int, held: bool = False) -> set[str]:
            """The lines decoded in this dict and the dicts inside it (not through a list). (v2.4.1 N1: `held` - a list
            whose strings the dict holding it reads, with its lists inside.)"""
            if lvl > 6:
                # (v2.4.1 N1: past the depth limit nothing is read - a dict, or a list with a dict or a list in it (or a
                # string no dict reads), makes the answer partial)
                if (isinstance(piece, dict) and piece) or (isinstance(piece, list) and any(
                        isinstance(val, (dict, list)) or (not held and isinstance(val, str)) for val in piece)):
                    note(DEEP)
                return set()
            if isinstance(piece, list):
                if not held:
                    # (v2.4.1 N1: a list no dict holds - the answer itself, or a list inside one: its strings at any nesting)
                    for item in leaves(piece, lvl):
                        if is_id(item):
                            note(item)
                for val in piece:
                    walk(val, [], True, lvl + 1, True)
                return set()
            if not isinstance(piece, dict):
                return set()
            inner: set[str] = set()
            for val in piece.values():
                if isinstance(val, dict):
                    inner |= walk(val, [piece] + above, row, lvl + 1)
                elif isinstance(val, list):
                    walk(val, [], row, lvl + 1, True)
            path = {done[id(d)] for d in [piece] + above if id(d) in done}
            got = path | inner
            mine = 0
            for key, val in piece.items():
                if is_id(key) and key not in got and done.get(id(val)) != key:
                    mine += note(key)
                if is_id(val, key) and val not in got:
                    mine += note(val)
                for item in (leaves(val, lvl + 1) if isinstance(val, list) else []):      # (v2.4.1 N1: at any nesting)
                    if is_id(item):      # (v2.4.2 Q2: a list gives no units of its own - a decoded dict never excuses it)
                        mine += note(item)
            if row and not mine and not path and not inner \
                    and any(CartBook.whole(val) is not None and not re.search(r"price|cost|due|sum|fee|charge", str(key).lower())
                            for key, val in piece.items()):
                words = [val for val in piece.values() if isinstance(val, str) and " " not in val and re.fullmatch(ID_LIKE, val)]
                if words:
                    note(words[0])
            return inner | ({done[id(piece)]} if id(piece) in done else set())
        walk(obs, [], False, 0)
        return left

    def view_units(self, obs: Any) -> dict[str, float] | None:
        """Line -> units read from a cart-like observation: dicts naming a line (v2.3 H1: ident) and holding one whole-number
        field (the units). None when the observation shows no such rows. (v2.3 H1: a units field may be a whole float or a
        digit string, never a field named for a price, a sum or the stock; `undecoded` - the row-like entries it read no
        units from (rows_left), and the rows it read when they name no counted line. v2.4 J1: every listing id without units
        of its own, wherever it appears)"""
        rows: dict[str, float] = {}
        known = set(self.units) | set(self.pending)
        keys: set[str] = set()                     # (v2.2 F3: the fields that held the units in a row)
        unsure: list[tuple[str, list, int]] = []   # (v2.2 F3: rows with two or more such fields)
        done: dict[int, str] = {}                  # (v2.3 H1: the dicts that gave a line its units; v2.4 J1: and that line)

        def visit(d: dict[str, Any]) -> None:
            ident = CartBook.ident(d, known)
            pairs = [(fkey, num) for fkey, num, val in ((key, CartBook.whole(val), val) for key, val in d.items())
                     if num is not None and 0 < num < 1000 and not (isinstance(val, str) and (val in known or "id" in str(fkey).lower()))
                     and not re.search(r"price|cost|due|sum|fee|charge|stock|avail|inventor|remain|left|on_hand", str(fkey).lower())]
            if len(pairs) > 1:
                # (a row holding a count and another number: the count-named field is the units)
                named = [(fkey, num) for fkey, num in pairs if re.search(r"qty|quant|count|units|number|total|held|ordered", str(fkey).lower())]
                pairs = named if len(named) == 1 else pairs
            if len(pairs) == 1 and ident:
                rows[ident] = rows.get(ident, 0) + pairs[0][1]
                keys.add(str(pairs[0][0]))
                done[id(d)] = ident
            elif pairs and ident:
                unsure.append((ident, pairs, id(d)))
        walk_dicts(obs, visit)
        for ident, pairs, did in unsure:
            # (v2.2 F3: else the field that held the units in another row of the same answer)
            named = [num for fkey, num in pairs if str(fkey) in keys]
            if len(named) == 1:
                rows[ident] = rows.get(ident, 0) + named[0]
                done[did] = ident
        self.undecoded = CartBook.rows_left(obs, known, done, self.__dict__.get("listings") or ())
        if not rows or not (set(rows) & (known or set(rows))):
            self.undecoded += [ident for ident in rows if ident not in self.undecoded]      # (v2.3 H1: none of them taken)
            return None
        return rows

    def learn(self, seen: dict[str, float], only: str | None = None) -> None:
        """A cart view is the truth: it replaces the code's count and settles whether a put adds or sets. (v2.2 F3: with
        `only`, a one-line acknowledgement - that line's units, the other lines kept. v2.3 H1: after an answer that left a
        row-like entry undecoded, the lines it read take its units and every other counted line is kept. v2.4.1 N2: and a
        counted line with an undecoded entry of its own in that answer keeps its count - the rows read may be a part of it;
        when an undecoded entry names no line the code knows (counted, read or listed), every counted line keeps its count;
        a line not counted before takes what was read.)"""
        for thing, expected_if_set in self.pending.items():
            if thing in seen:
                if abs(seen[thing] - expected_if_set) < 1e-9 and abs(self.units.get(thing, 0) - expected_if_set) > 1e-9:
                    self.adds = False
                elif seen[thing] > expected_if_set:
                    self.adds = True
        self.pending = {}
        if only is not None:
            self.units = {**self.units, only: seen[only]}
            return
        if self.__dict__.get("undecoded"):
            hidden = set(self.undecoded)
            blind = any(thing not in self.units and thing not in seen and thing not in (self.__dict__.get("listings") or ())
                        for thing in hidden)      # (v2.4.1 N2: an entry no line can be told from)
            if blind:
                # (v2.4.2 Q3: every counted line keeps its count, as in v2.4.1; a counted line the view did not show at that
                # count is unsure - the confirmation names its quantity as the code's count)
                self.unsure = sorted(t for t, n in self.units.items() if seen.get(t) != n)
                self.units = {**self.units, **{fkey: val for fkey, val in seen.items() if val > 0 and fkey not in self.units}}
                return
            # (v2.4.2 Q3: no entry can hide another line's units: a counted line with an undecoded entry of its own keeps its
            # count, a line read takes what was read - a reduction too - and a counted line the view shows nowhere is out)
            self.unsure = sorted(t for t, n in self.units.items() if t in hidden and seen.get(t) != n)      # (v2.4.3 S3: a
            # count kept for a line with an undecoded entry of its own, which the view does not show at that number, is unsure)
            self.units = {**{t: n for t, n in self.units.items() if t in hidden or seen.get(t, 0) > 0},      # (the order kept)
                          **{fkey: val for fkey, val in seen.items() if val > 0 and not (fkey in self.units and fkey in hidden)}}
            return
        self.unsure = []
        self.units = {fkey: val for fkey, val in seen.items() if val > 0}

    def viewed(self, obs: Any, tick: int, fault: bool = False) -> bool:
        """(v2.2 F3: True when the view showed the cart - its lines, or none; v2.3 H1: never when it left a row-like entry
        undecoded - the lines it read are learnt, no other line is dropped; `fault` - an answer with an error in it shows no
        cart, so it never empties the count)"""
        self.last_view = obs
        self.viewed_at = tick
        seen = self.view_units(obs)
        if seen is not None:
            self.learn(seen)
            return not self.undecoded
        if self.undecoded or fault:
            return False
        if isinstance(obs, (dict, list)) and not self.ids_in(obs):
            self.units = {}
            return True
        return False

    def ids_in(self, obs: Any) -> bool:
        text_body = as_text(obs)
        return any(thing in text_body for thing in self.units)

    def summary(self) -> str:
        if not self.units:
            return "[cart as counted] empty."
        lns = ", ".join(f"{thing} x{int(qty_n) if float(qty_n).is_integer() else qty_n}" for thing, qty_n in self.units.items())
        return f"[cart as counted] {lns}."

    def faults(self, need_n: int, counts: list[int] | None = None) -> list[str]:
        """Lists mismatches between the tracked cart and what the request needs.
        With several product counts it compares line count and sorted quantities against counts.
        For a single product it flags extra lines and a wrong unit count versus need_n."""
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
        self.needs.ref_hook = self.ref_fix
        self.role_map: dict[str, str] = {}
        self.trolley = CartBook(None, {})
        self.trolley.listings = self.archive.ids      # (v2.4 J1: the store's listing ids - a cart view's keys and list items)
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

    @property
    def log(self) -> list[dict[str, Any]]:
        return self.wire.log

    def moves_left(self) -> int:
        return self.step_cap - self.wire.tick

    def role(self, call_name: str) -> str | None:
        return self.role_map.get(call_name)

    def ask_limit(self) -> int:
        return self.msg_cap if self.msg_cap is not None else ASK_DEFAULT

    def confirm_due(self) -> bool:
        """v2: the last message is kept for the basket confirmation while it is unsent: a message tool, a cart tool and an
        order tool, and a cap N >= 2 (the request's message budget, else ASK_DEFAULT)."""
        return bool(self.role("message") and self.role("cart") and self.role("order")) and self.ask_limit() >= 2 \
            and not self.__dict__.get("confirm_sent")

    def question_limit(self) -> int:
        """v2: the messages questions may use - the code's planned ones, its replies and the actor's: N-1 while the
        confirmation's message is kept, else N."""
        return self.ask_limit() - 1 if self.confirm_due() else self.ask_limit()

    def call_kind(self, call_name: str, call_args: dict[str, Any] | None) -> str | None:
        """'order', 'decline' or None for a call. When one tool does both, its mode value tells which: v1.1 - the per-value
        reading of the value sent (choice_reading) first, then HW's words in any short text argument, else 'order'."""
        order, decline = self.role("order"), self.role("decline")
        if call_name not in (order, decline) or not call_name:
            return None
        if order != decline:
            return "order" if call_name == order else "decline"
        reading = self.finish_values(call_name)        # (v1.2: with elimination)
        for key, piece in (call_args or {}).items():
            if isinstance(piece, str):
                kind = next((got for val, got in (reading.get(key) or {}).items() if val.lower() == piece.strip().lower()), None)
                if kind:
                    return kind
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
        """Gives a rough remaining-time phrase once less than half the budget is left.
        Computes spare time minus the last-call reserve and rounds it down to the clock step.
        Returns an empty string while time is plentiful, else about N s or under one step."""
        left = max(0.0, self.wire.spare() - LAST_CALL)
        if left > (DEADLINE - LAST_CALL) / 2:
            return ""
        step = int(left // CLOCK_STEP) * CLOCK_STEP
        return f"about {step} s left" if step > 0 else f"under {CLOCK_STEP} s left"

    def status_line(self) -> str:
        spare = self.moves_left()
        pieces = [f"[status] steps used {self.wire.tick} of {self.step_cap}, {spare} left",
                 f"messages sent {self.msgs_sent}" + (f" of {self.ask_limit()} (the last one is kept: the order call sends the basket "
                                                      "confirmation first)" if self.kept_open()      # (v2.5 V1b)
                                                      else f" of {self.msg_cap}" if self.msg_cap is not None else "")]
        clock = self.clock_text()
        if clock:
            pieces.append(clock)
        text_body = "; ".join(pieces) + ".\n[requirements now] " + self.needs.text_body() + self.ref_line() + self.product_steps()
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
        _aoi_roles(self, "read_request", reading)

    def use_reading(self, reading: dict[str, Any], again: bool = False) -> None:
        """Installs the first or repeated request reading into the agent's working state.
        Maps tool roles, rebuilds the cart tracker, stores written requirements and loose wants.
        Also sets the spend cap, message and step budgets, and bumps the needs version on rereads."""
        if not again:
            self.steps_given = self.step_cap
        old_cart = self.role("cart")
        role_map = reading.get("roles") if isinstance(reading.get("roles"), dict) else {}
        self.role_map = {fkey: val for fkey, val in role_map.items() if isinstance(val, str) and val in self.shapes}
        if not reading.get("code_reading"):
            self.settle_finishers(reading)       # (v1.1: a VALID reader answer wins; see settle_finishers)
        if again and self.trolley.used and old_cart:
            self.role_map["cart"] = old_cart
        if not (again and self.trolley.used):
            self.trolley = CartBook(self.role("cart"), self.shapes.get(self.role("cart") or "", {}))
            self.trolley.listings = self.archive.ids      # (v2.4 J1)
        self.needs.sweepable = bool(self.role("search")) and self.page_arg() is not None
        spend_cap = reading.get("message_budget")
        if is_number(spend_cap) and spend_cap >= 0:
            self.msg_cap = int(spend_cap)
        self.needs.reading = reading
        try:
            self.wire.scale = 1.0 + 0.3 * min(2, max(0, len(self.needs.items()) - 1))
        except Exception:
            self.wire.scale = 1.0
        named = [str(req) for req in reading.get("requirements") or [] if req]
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
            self.needs.version += 1

    def story_values(self, reading: dict[str, Any]) -> None:
        """Normalises the hints from a reading and moves story-guessed needs into hints.
        A need only inferred from the story is removed from written ones and from item rows.
        Writes at most twelve hints back into the reading and logs any moved needs."""
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
        """Promotes hints whose value is plainly spelled in the request into real requirements.
        Skips a hint when it clashes with an existing requirement of the same property or value.
        Returns the hints it promoted, appending the new need to the matching item row."""
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
        """Collects the hints that still deserve a question to the shopper.
        A hint stays when its property is open by samples and was not asked about already.
        Returns alternates for no-preference hints first, then the plain open hints."""
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
        """Finds the menu line describing one property for a given product shelf.
        Builds and memoises the property menu from the shelves read so far for that product.
        Returns the matching line, or an empty string if none matches or on error."""
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
        """Returns the set of values seen so far for one property on a product's shelf.
        Splits the menu line after its count marker on the bar separator and normalises each value.
        The result is lowercase alphanumeric tokens with empty entries removed."""
        line = self.prop_line(k, prop)
        values = line.split("):", 1)[1].split("|") if "):" in line else []
        return {" ".join(re.findall(r"[a-z0-9]+", v.lower())) for v in values} - {""}

    def same_field(self, k: int | None, a: str, b: str) -> bool:
        """Decides if two property names refer to the same store field.
        True for a strict name match, or for similar names whose observed value sets overlap.
        Any exception while reading menu values makes it answer False for the pair."""
        if strict_prop(a, b):
            return True
        try:
            return same_prop(a, b) and bool(self.field_values(k, a) & self.field_values(k, b))
        except Exception:
            return False

    def hinted_prop(self, k: int | None, prop: str) -> bool:
        """Tells whether a property name appears as the head of any reading hint.
        Ignores any parenthesised tag, and for multi-item requests restricts to the product k.
        Returns True on a strict name match with a hint head."""
        name = str(prop).split("(")[0].strip()
        for hint in self.needs.reading.get("hints") or []:
            head = str(hint).split(":", 1)[0]
            if strict_prop(head, name) and (k is None or not self.needs.items() or self.product_of(head) in (k, None)):
                return True
        return False

    def product_reads(self, k: int | None) -> int:
        """Counts how many detailed listing reads exist for a product's shelves.
        Uses the home shelves of item k, or the single own shelf for a one-product request.
        Returns 0 on any error or when no shelf is known."""
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
        """Detects a hinted property that the store's listings do not seem to name as a field.
        Needs at least two rich reads, no menu line for it, and a hint or hinted name for it.
        Returns True only when no menu line carries that name; False on error."""
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            if not name or not (hinted or self.hinted_prop(k, name)) or self.prop_line(k, name) or self.product_reads(k) < 2:
                return False
            menu = (self.__dict__.get("menu_memo", {}).get(k) or (0, ""))[1]
            return not any(same_prop(line.strip("- ").split(" (in ")[0].strip(), name) for line in menu.splitlines())
        except Exception:
            return False

    def hint_mark(self, hint: str) -> str:
        """Annotates a hint string when its property has no field key in the store.
        Looks up the product for the hint head and asks plain_hint with the hinted flag set.
        Returns the hint with a no-key suffix, or the original text unchanged."""
        try:
            prop = str(hint).split(":", 1)[0].strip()
            k = self.product_of(prop) if self.needs.items() else None
            return f"{hint} [no key]" if self.plain_hint(k, prop, hinted=True) else str(hint)
        except Exception:
            return str(hint)

    @staticmethod
    def hint_alts_of(detail: str) -> list[str]:
        """Pulls alternate property names out of the parenthesised or-clauses in hint detail.
        Splits each clause on commas, the word or, or slashes and trims the pieces.
        Returns the non-empty alternatives as a list of strings."""
        return [a.strip() for m in re.findall(r"\(or ([^()]+)\)", str(detail)) for a in re.split(r",|\bor\b|/", m) if a.strip()]

    @staticmethod
    def name_inside(a: Any, b: Any) -> bool:
        """Checks whether one property name's words are contained in the other's.
        Ignores short words and link words, then tests subset in either direction.
        Returns False if either name has no usable words."""
        wa = {w for w in prop_words(a) if len(w) >= 3 and w not in LINK_WORDS}
        wb = {w for w in prop_words(b) if len(w) >= 3 and w not in LINK_WORDS}
        return bool(wa) and bool(wb) and (wa <= wb or wb <= wa)

    def asked_inside(self, k: int | None, head: str) -> bool:
        """Tells whether the shopper was already asked about a property in earlier questions.
        Scans the questions log, skipping unanswered ones, and resolves which product each covers.
        Returns True on a name-containment match for product k; False on error."""
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
        """Builds an alternate-property hint when the first property drew no preference.
        Needs a quiet earlier ask on that field, then finds an or-alternative that has a menu line.
        Returns a rewritten hint using that key, or empty if spent, not plan-open, or on error."""
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
        """Checks whether a property is covered by some reading hint for the product.
        Compares the hint head and its alternates strictly, then falls back to all words appearing.
        Returns True on a match, False otherwise or on error."""
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
        """Decides if two property names look like near-twins without being identical.
        Compares head words (all but the last) of one name with words of the other, by prefix.
        Returns False for strictly equal names; prefix matches need at least four letters."""
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
        """Wraps the twin test for two property names within the context of product k.
        True when the names differ strictly and are either name twins or share the same field values.
        A failure inside the comparison is swallowed and treated as not twins."""
        try:
            return not strict_prop(a, b) and (self.name_twin(a, b) or self.same_field(k, a, b))
        except Exception:
            return False

    def twin_key(self, k: int | None, prop: str) -> str:
        """Searches the property menu for a sibling field worth asking about instead.
        Candidates must be twins of prop, unspent, plan-open, and mentioned in the story or a hint.
        Returns the first such key, or an empty string if none or on error."""
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
        """Writes a short either-or question offering a property's distinct observed values.
        Takes up to three clean, short values on the menu line that other fields do not repeat.
        Returns the question text, or empty if fewer than two values qualify or on error."""
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
        """Chooses a one-off follow-up question for a property, once per product.
        Prefers a values question, otherwise a twin key question about a sibling property.
        Returns a pair of label and question text, or None; records the product as done."""
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
        """Tells whether a property was already asked about for product k.
        Walks the questions log, splits labels into parts, and matches the head strictly.
        Returns True on a hit, False otherwise or on error."""
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
        """Finds a hinted alternative property key still worth asking in place of prop.
        Looks through hints related to prop, and tries their or-alternatives against the menu.
        Skips junk, spent or already asked keys; returns the first usable key or empty string."""
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
        """Offers an alternative property to ask about when the first one collided.
        Runs at most once per product, using alt_key, and records that in the done set.
        Returns the key, with the product in parentheses for multi-item requests, else empty."""
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
        """Checks that a text contains all content words of a plain question.
        Drops frame words and link words, stems the rest, and tests each against the text.
        Returns True only if there is at least one needed word and all are present."""
        frame = {"which", "would", "you", "like", "for", "the"}
        have = {self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())}
        need = [self.stem(w) for w in re.findall(r"[a-z0-9]+", str(plain_q).lower()) if w not in frame and w not in LINK_WORDS]
        return bool(need) and all(w in have for w in need)

    def alt_untried(self, k: int) -> bool:
        """Reports whether a product has a hinted alternative property not yet asked.
        Looks for a hint head already asked for that still yields an alt_key.
        Returns True on the first one found; False otherwise or on error."""
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
        """Picks a sibling property to quietly ask again after a no-preference answer.
        Gated by one use per product, a mapped hint, no pending no-preference hints and covered items.
        Returns the twin key, with the product tag when several are asked, else empty."""
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
        """Drops rarely seen properties from a menu once enough listings were read.
        With fewer than four reads or an empty menu the text is returned as is.
        Otherwise filters out lines seen on zero or one listings, unless nothing would remain."""
        if reads < 4 or not menu:
            return menu
        kept = [ln for ln in menu.splitlines() if not re.search(r"\(in [01] listings?\)", ln)]
        return "\n".join(kept) if kept else menu

    def first_reads(self) -> bool:
        """Plans an initial batch of listing openings to learn each product's fields.
        Runs once, sizes the room from remaining moves and ask reserve, and picks menu rows per shelf.
        Sets the open queue and a spoken note, returning True if anything was queued."""
        try:
            if self.__dict__.get("first_reads_done") or self.batch_tool() or not self.role("details"):
                return False
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
                if home and have < 2 and room > 0:
                    picks = self.menu_rows(home, 2 - have, taken, qtys[min(k, len(qtys) - 1)])
                    taken += picks
                    if picks:
                        wants.append((have, k, picks))
            queue = [picks[n] for n in range(2) for _, _, picks in sorted(wants) if n < len(picks)][: max(0, room)]
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
        """Asks whether a property is still open to question for product k.
        Uses the menu line as evidence when the property is hinted, sampling when it is.
        Falls back to a plain open_prop check if anything fails."""
        try:
            hinted = self.hinted_prop(k, prop)
            return self.open_prop(k, prop, self.prop_line(k, prop) if hinted else "", by_samples=hinted)
        except Exception:
            return self.open_prop(k, prop)

    def settles(self, req: str, samples: str = "") -> bool:
        """Decides whether a requirement is already settled without asking the shopper.
        True for reference-like values, or positive values not merely guessed from the story.
        With samples, non-numeric value words must appear in them, allowing one edit."""
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
        """Computes the number of moves that must remain before the order is placed.
        Derives a base from the step cap and fixed floors, raising it after an event or dead item.
        Returns an integer capped at half of the step cap and not below the base."""
        base = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, self.step_cap // 4 + 1))
        due = max(base, min(16, self.step_cap // 3))
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
        """Tells whether the run is in the reserved stretch saved for ordering.
        Needs order and cart roles, more than three moves left, and the due point or a refusal.
        Used to hold back lookup tools so the final moves go to the order step."""
        spare = self.moves_left()
        return bool(self.role("order") and self.role("cart")) and spare > 3 and (
            (spare <= self.order_due() and self.orders_tried == 0) or (spare <= FIRST_ORDER_BY and self.refusal_n > 0))

    def unverified(self) -> bool:
        """Reports whether the cart still lacks a verified, live pick for the current needs.
        Considers passed items of this needs version that are not refused or dead.
        For multi-item requests also checks every home shelf has one such pick."""
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
        """Returns the ids suggested by checks that are known but not yet opened.
        Starts from the named set, keeps ids in the archive, drops read and refused lines.
        Result is a set of id strings."""
        return {i for i in self.__dict__.get("named", set()) if i in self.archive.ids} - self.archive.read - self.refused_lines

    def widened_late(self) -> bool:
        """Tells whether a requirement was dropped recently enough to matter now.
        True when the widening happened at the current needs version and within the order-due ticks.
        Compares the wire tick now against the tick recorded when the drop was heard."""
        return self.needs.widened_at == self.needs.version and self.wire.tick - self.needs.widened_tick <= self.order_due()

    def begin_batch(self) -> None:
        self.late_q = 0
        self.late_ids = set()

    def late_hold(self, call_name: str, call_args: dict[str, Any]) -> str:
        """Gates detail and search calls late in the run so steps stay for ordering.
        Allows a limited number of extra reads for unverified picks, named rows, backups or a swap.
        Returns an empty string to allow the call, else a text explaining it was not run."""
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
        """Lists cheaper unread rows that could back up a verified cheapest pick.
        Only applies for cheapest priority after an event, dead line or named opening exists.
        Returns up to six ids sorted by price, or empty if disabled or on error."""
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
        """Produces a prompt section listing property names found on the store's listings.
        Takes the property menu from the archive; counts show how many listings carry each.
        Returns an empty string when the menu is empty."""
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
        """Returns the structured listing rows used by the field checks, memoised.
        The cache is keyed on the number of archive items and rebuilt via code_rows when it grows.
        Returns a dict from listing id to its row data."""
        memo = self.__dict__.get("ch_rows_memo")
        if memo and memo[0] == len(self.archive.items):
            return memo[1]
        rows = self.code_rows()
        self.ch_rows_memo = (len(self.archive.items), rows)
        return rows

    def store_fields(self, label: Any = "") -> dict[str, list[str]]:
        """Collects every field name with the values it takes across relevant listings.
        Restricts rows to the shelf of the labelled product, or the own or first home shelf.
        Skips junk keys and blank values; returns a dict of lists, empty on error."""
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
        """Detects whether a listing's own notes send a property elsewhere.
        Uses an own_redirects hook if present, else scans sentences of the listing's text.
        Looks for the property words near wording like refer or go by; returns a bool."""
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
        """Returns the field evidence for a listing, memoised per state of the search.
        Wraps field_facts, falling back to empty lines, rules and fails if it raises.
        Appends meaning-check confirmations to matching lines and removes failures they cover."""
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
        """Compares one listing's fields against how peers on its shelf write each required value.
        Builds a table of field values over peer listings, needing at least three peers.
        Returns lines, rules and fails text lists saying where the listing meets or misses a need."""
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
        """Asks the checking model whether other listings could beat the chosen one.
        Builds a prompt from request, needs, cards and field evidence; runs once per state key.
        Returns advice naming better candidates, or empty; stores them as named for checking."""
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
        better = [x for x in better if not self.field_view(x)["fails"]]
        self.rank_better = list(better)
        self.__dict__.setdefault("named", set()).update(better)
        if not better:
            return ""
        return (f"A rank check thinks these listings may also fit and beat {thing} on the {what}: " + ", ".join(better)
                + (f" ({answer.get('reason')})" if answer.get("reason") else "")
                + ". Check them (details, stock, price) before ordering; if each fails, order the current pick.")

    def review_decline(self, early: bool = False) -> str:
        """Runs a last audit before the agent declines, to find overlooked fitting listings.
        Sends request, needs and listing cards to the auditor, then adds search pages and combos.
        Returns advice naming fits, unopened rows and searches to run, or empty if nothing found."""
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
        """Lists named or hit listings that were never opened and still look affordable.
        Orders by hit rank, skips read, refused, unknown ids, over-budget prices and other shelves.
        Returns the remaining ids as a list."""
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
        """Builds arguments for a details call on a given listing id.
        Reuses the shape of an earlier logged details call by swapping in the new id.
        Otherwise picks the required or sole string parameter from the schema; None if unclear."""
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
        """Issues scripted detail reads for a queue of listing ids and returns the replies.
        Uses one batch call per slice when batch is given, else one call per id via reader.
        Appends a synthetic assistant turn, dispatches with cover mode on, records last_reads."""
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
        """Runs any queued listing reads (from open_queue or put_queue) before the agent moves on.
        Trims the queue to the step budget, picks batch or single reads, falls back on failure.
        Posts replies, may attach a decline review note, and returns True when something was read."""
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
        batch = self.batch_tool() if len(queue) >= 2 else None
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
        """Finds the newest stock state recorded for one listing id in the archive.
        Walks every stored record and keeps the last in-stock or sold-out reading it finds.
        Returns True, False, or None when no record reported stock for that id."""
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
        """For cheapest-first requests, runs item checks on up to two better-ranked candidates.
        Skips unread, refused, dead, passed or over-budget lines and stops when time is short.
        Returns the cheaper verified alternative to thing, or an empty string on error."""
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
        """Looks for a cheaper listing that already passed checks and could replace thing.
        Only considers the current requirements version, same shelf kind, live and non-failing lines.
        Returns the id of the lowest-priced such line, or empty when none beats the current price."""
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
        """Works out which shelf or category a given listing id belongs to.
        Scans archived records for a dict whose id field matches, then falls back to stored labels.
        Returns the shelf name in lowercase, or an empty string if it is unknown."""
        found = ""

        def visit(d: dict[str, Any]) -> None:
            nonlocal found
            if any(isinstance(val, str) and val == thing and "id" in str(key).lower() for key, val in d.items()):
                found = self.shelf_of(d) or found
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return found or str((self.archive.__dict__.get("label_of") or {}).get(thing) or "").strip().lower()

    def chosen(self) -> str | None:
        """Returns the id of the line most recently added to the cart.
        Reads the last key of the cart units mapping.
        Gives None when the cart holds nothing."""
        if self.trolley.units:
            return list(self.trolley.units)[-1]
        return None

    def home_shelves(self) -> list[set[str]]:
        """Matches each requested product to the shelf names that best fit its wording.
        Collects shelves seen next to known ids, then keeps those with the top token match per product.
        Returns a list of shelf-name sets, cached until the archive grows."""
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
                    score = kind_score(own, words, shared)
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
        """Names the single shelf the agent should be browsing for a one-product request.
        Returns empty for multi-product requests, or if every shelf read is already in the home set.
        Otherwise prefers a read shelf in the home set, else the first home shelf alphabetically."""
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
        """Finds how the request words two product kinds together under one shared category label.
        Tries the label halves joined by and, or, slash or plus, then a loose scan of both names.
        Returns the matched phrase and whether it came from the label, or None if none is found."""
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
        """Tells whether a logged shopper question was about one of the requested products.
        Parses label and question from the log line, then compares bracketed tags and question text.
        Returns True on a kind-token or name-pattern match with any row."""
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
        """Checks whether two product rows state conflicting values for the same property.
        Splits each requirement into property and normalised value, comparing same-named properties.
        Returns True if any shared property has different values, so the rows must stay separate."""
        def pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
            out = []
            for req in row.get("requirements") or []:
                head, _, value = str(req).partition(":")
                out.append((re.split(r"\s*\(", head)[0].strip(), " ".join(re.findall(r"[a-z0-9]+", value.lower()))))
            return out
        return any(same_prop(pa, pb) and va != vb for pa, va in pairs(a) for pb, vb in pairs(b))

    def merge_joined(self, obs: Any) -> str:
        """Detects two requested products that the store sells as one joined category and merges them.
        Collects two-part category labels from records, then pairs equal-quantity rows with no clash.
        Returns the merge remarks joined together, after rewriting the needs through merge_items."""
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
        """Collapses requirement rows i and j into a single product line named by phrase.
        Rewrites stored requirements, hints, references and tags, dropping tags the merged item owns.
        Bumps the needs version, clears caches, and returns a note describing the single combined line."""
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

    def lookup_key(self, tool: str | None) -> str | None:
        """Finds the one text argument a tool needs for looking something up.
        Inspects the tool schema for string properties without enums and required ones.
        Returns that argument name, or None when the choice is ambiguous or other args are required."""
        jschema = self.shapes.get(tool or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        needed = [k for k in jschema.get("required") or [] if k in props] if isinstance(jschema.get("required"), list) else []
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
        key = next((k for k in texts if k in needed), None) or (texts[0] if len(texts) == 1 else None)
        return key if key and all(k == key for k in needed) else None

    def lookup_tools(self) -> list[str]:
        """Lists tools that could serve as single-listing readers.
        Excludes tools already holding a search, cart, order, message or decline role.
        Keeps those with a usable lookup key that are the details or view role, or whose shape fits."""
        taken = {self.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        reads = {self.role("details"), self.role("view")} - {None}
        return [name for name in self.shapes if name not in taken and self.lookup_key(name) and (name in reads or self.lookup_shape(name))]

    def lookup_shape(self, tool: str) -> bool:
        """v1 (plan v2 Phase 1 item 6): a tool with no role reads one listing only when its shape says so - its one text
        argument names an id (an id, code, sku, number or reference, or a listing / item / product key that is no class,
        category, kind or query), and its name and description's main clause are no browse, category, usage or budget
        read, no finish, no no-buy and do not name the shopper. A category browse or a status read is never probed as a
        stock read."""
        key = self.lookup_key(tool)
        if not key:
            return False
        props = (self.shapes.get(tool) or {}).get("properties") or {}
        spec = props.get(key) if isinstance(props.get(key), dict) else {}
        key_text = name_words(key) + " " + str(spec.get("description") or "").lower()
        desc = next((str(tdef["function"].get("description") or "") for tdef in self.toolset if tdef["function"]["name"] == tool), "")
        clauses = self.main_clauses(desc)
        head = name_words(tool) + " | " + (clauses[0] if clauses else "")
        if re.search(r"\b(?:brows\w*|explor\w*|categor\w*|class\w*|department\w*|aisle\w*|shelf|shelves|section\w*|usage|budget)\b", head):
            return False
        if re.search(FINISH_STRONG, head) or re.search(NO_BUY, head) or re.search(r"\b(?:shoppers?|customers?|buyers?|clients?|users?)\b", head):
            return False
        if re.search(r"\b(?:class\w*|categor\w*|department\w*|kinds?|types?|groups?|section\w*|shelf|aisle\w*|range|quer\w*|keywords?|"
                     r"terms?|text|phrase|note|message|body|reason)\b", key_text):
            return False
        return bool(re.search(r"\b(?:ids?|identifier\w*|codes?|sku\w*|refs?|reference\w*|numbers?|listings?|items?|products?|records?|"
                              r"offers?|handles?|slugs?|keys?)\b", key_text))

    def lookup_call(self, ident: str, tool: str | None) -> dict[str, Any] | None:
        """Builds the argument dict for reading listing ident with the given tool.
        Uses the details builder for that role, else copies a past call shape swapping in the new id.
        Falls back to a single-key dict from lookup_key, or None if the tool is missing or unusable."""
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
        """Pulls an identifier string out of a record dict.
        Tries a key named exactly id first, then any key ending in id, needing an id-shaped string.
        Returns the id, or None when no field qualifies."""
        for exact in (True, False):
            for key, val in d.items():
                name = str(key).lower()
                if (name == "id" if exact else name.endswith("id")) and isinstance(val, str) and re.fullmatch(ID_LIKE, val):
                    return val
        return None

    @staticmethod
    def asked_ids(call_args: Any) -> set[str]:
        """Extracts the listing ids that a tool call asked about.
        Takes id-shaped strings from argument values and from list-valued arguments.
        Returns them as a set, empty when the arguments are not a dict."""
        args = call_args if isinstance(call_args, dict) else {}
        asked = {v for v in args.values() if isinstance(v, str) and re.fullmatch(ID_LIKE, v)}
        asked |= {x for v in args.values() if isinstance(v, list) for x in v if isinstance(x, str) and re.fullmatch(ID_LIKE, x)}
        return asked

    def learn_caps(self, call_name: str, call_args: Any, obs: Any) -> list[str]:
        """Learns what a tool reveals by inspecting one reply, tracking its capabilities per tool name.
        Updates counters for ids, stock, properties, notes and price, and tracks sold-out lines.
        Returns the cart lines newly found sold out; ignores error replies."""
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
                if isinstance(v, str) and len(v.split()) >= 8 and (row.get(str(k)) != v if row is not None else str(k) not in row_keys):
                    note_keys.add(str(k))
                    hit["notes"] = True
        if not (isinstance(obs, dict) and obs.get("error")):
            walk_dicts(obs, visit)
        for key, got in hit.items():
            caps[key] += int(got)
        return newly

    def stock_tool(self) -> str | None:
        """Chooses the tool best suited for re-reading stock of a listing.
        Prefers tools already seen returning stock, then view or details roles, then an idle one.
        Returns the tool name, or None when no candidate looks suitable."""
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
        """Chooses a lookup tool that exposes free-text note fields beyond the details tool.
        Compares learned note keys per tool with the details tool keys and picks the most productive.
        Returns the tool name or None."""
        caps = self.__dict__.get("tool_caps") or {}
        details = self.role("details")
        dkeys = (self.__dict__.get("tool_keys") or {}).get(details or "", set())
        nkeys = self.__dict__.get("note_keys") or {}
        shown = [t for t in self.lookup_tools() if t != details and set(nkeys.get(t) or set()) - set(dkeys)]
        return max(shown, key=lambda t: ((caps.get(t) or {}).get("notes", 0), t)) if shown else None

    def dead_lines(self) -> set[str]:
        """Returns a copy of the set of listing ids judged sold out.
        Reads the dead set stored on the instance, defaulting to empty.
        The copy keeps callers from mutating the internal record."""
        return set(self.__dict__.get("dead") or set())

    def stale_lines(self) -> list[str]:
        """Lists cart lines whose stock was not re-read since the cart last changed.
        Compares each line sequence number with the current cart sequence mark.
        Returns the ids still lacking a fresh reading."""
        seqs = self.__dict__.get("line_seq") or {}
        mark = self.__dict__.get("cart_seq", 0)
        return [i for i in self.trolley.units if seqs.get(i, -1) < mark]

    def dead_swap(self, line: str) -> tuple[str, str]:
        """Looks for a replacement for a cart line that went out of stock.
        Prefers a verified fit within shelf and budget, else ranks in-stock rows by price.
        Returns the chosen id plus fallback candidate text, and stores unread picks and a page to try."""
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
        """Finds search arguments for the next unread results page relevant to a product.
        Picks the stored search walk carrying most stated values for index k, then advances its page.
        Returns the new argument dict, or None when no page argument or walk exists."""
        found = self.page_arg()
        if not found:
            return None
        walks = [w for w in self.__dict__.get("page_book", {}).values() if self.walk_carries(w["args"], k) and self.next_page(w) is not None]
        if not walks:
            return None
        best = max(walks, key=lambda w: (len(self.walk_carries(w["args"], k)), -len(w["read"])))
        return {**best["args"], found[0]: self.next_page(best)}

    def swap_open(self) -> str:
        """Returns the next candidate replacement that still needs its listing read.
        Does nothing once the swap is marked done; skips ids already read, dead, or in the cart.
        Gives the id, or an empty string when none remains."""
        if self.__dict__.get("swap_done"):
            return ""
        dead = self.dead_lines()
        return next((i for i in self.__dict__.get("swap_unread") or [] if i not in self.archive.read and i not in dead
                     and i not in self.trolley.units), "")

    def dead_note(self, lines: list[str]) -> str:
        """Writes advice text for cart lines that are out of stock.
        For each line it runs dead_swap and tells the agent to swap in a verified pick or check rows.
        Resets swap state, appends a next-page hint if present, and returns the combined text."""
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
        """Builds follow-up text after the order tool refused the cart.
        First checks whether the refusal reply shows the one cart line became unavailable.
        Otherwise queues stock re-reads of the cart lines; returns the combined message."""
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
        """Decides if a refusal means the single cart line sold out.
        Feeds the refusal shopper text to the needs reader and sees whether requirements stayed put.
        If so marks the line dead and returns heard text plus swap advice; otherwise empty strings."""
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
        """Queues stock re-reads for up to four cart lines after a refused order.
        Bails out when muted, covering, a queue exists, or steps and time are low.
        Sets the open_queue fields and returns a short note naming the reading tool."""
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
        """Tells whether code can judge every outstanding requirement without asking anyone.
        Counts plain comparable values plus requirements settled by resolved differ references.
        Returns True if that count covers all properties still left."""
        items = self.needs.items()
        vals = [plain for k in (range(len(items)) if items else [None]) for _, plain in self.code_values(k)]
        plain = [v for v in vals if not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
                                                  v.lower())]
        resolved = self.differ_done(None)
        judged = sum(1 for req in self.needs.props_left() if any(self.ref_match(ref, req) for ref in resolved))
        return len(plain) + judged >= len(list(self.needs.props_left()))

    def differ_done(self, k: int | None) -> list[dict[str, Any]]:
        """Lists resolved differ-type references that still apply to open requirements.
        Keeps live references with a value that match a remaining property and the product index k.
        Returns the matching reference dicts."""
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
        """Tries to repair a doomed decline by building a fitting cart in code.
        Runs after a refusal or when steps run low, once per mark; verifies picks, mutes to put.
        Returns advice to order the repaired cart, or empty if nothing fit; undoes state on failure."""
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
        """Tells whether a requirement value is a plain single value.
        Rejects links, numeric ranges, comparison signs and approximate wording.
        Returns True when none of those patterns appear."""
        return not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
                             str(value).lower())

    @staticmethod
    def ed_word(value: str) -> bool:
        """Tells whether a value is a plain word-like value rather than a number or reference.
        Needs three or more letters, no digits, plain form, no excluded words and no reference text.
        Returns True when all those conditions hold."""
        low = str(value).lower()
        return bool(re.search(r"[a-z]{3,}", low)) and not re.search(r"\d", low) and Errand.ed_plain(low) \
            and not re.search(EXCLUDE_WORDS, low) and not ref_text(low)

    def ed_reqs(self, k: int | None) -> list[tuple[str, str, bool]]:
        """Lists remaining property requirements for product index k.
        Splits each as head and value, drops stated or revealed tags, and skips other products.
        Returns triples of property, plain value, and whether it was revealed rather than written."""
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
        """Builds an annotated pool of candidate listings for product index k.
        Per row it flags gone, over budget, pointer, ruled out, clash, misses and fit status.
        Returns a list of dicts, memoised on a state key so repeated calls are cheap."""
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
        """Finds products for which every read candidate is ruled out.
        Looks at the pool for each product index, needing at least one read listing, all marked out.
        Returns the list of such product indexes."""
        out = []
        for k in range(len(self.needs.items())):
            reads = [e for e in self.ed_pool(k) if e["read"]]
            if reads and all(e["out"] for e in reads):
                out.append(k)
        return out

    def slack_rows(self) -> list[str]:
        """Picks unread cheap listings that could still fit once other products take their floors.
        Only for multi-product requests with a cap that the cheapest fits exceed; ranks by search lead.
        Returns up to three ids interleaved across products and records the floor total."""
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
        """Fetches the problems from the latest failed item check for a listing.
        Scans the log backward for the verify entry on that id and reads its result.
        Returns problem strings if it did not fit, else None."""
        for entry in reversed(self.log):
            if isinstance(entry, dict) and entry.get("kind") == "verify" and entry.get("item") == ident:
                res = entry.get("result")
                if isinstance(res, dict) and res.get("fits") is False:
                    return [str(p) for p in res.get("problems") or [] if str(p).strip()]
                return None
        return None

    def ed_news(self, k: int | None) -> bool:
        """Tells whether new information could change the pool for product index k.
        True after any refusal or when uncovered or let-go requirements concern that product.
        Returns a bool: True means asking again may be worthwhile."""
        if self.refusal_n > 0:
            return True
        rows = self.needs.items()
        return any(not rows or self.product_of(str(req).partition(":")[0]) in (k, None)
                   for req in list(self.needs.uncovered) + list(self.needs.let_go))

    def near_hold(self) -> str:
        """Orders the cart directly when a meaning check confirmed its lines fit in other words.
        Requires unchanged needs version and refusals, intact live lines and enough steps and time.
        Calls finish_now, restoring mute state, and returns an explanatory note or empty."""
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
        """Guarded entry to the near-match basket check.
        Returns empty if already done, roles are missing, steps or time are low, or the wire is idle.
        Otherwise returns near_basket output, logging any error."""
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
        """Records bookkeeping after the cart contents changed.
        Bumps the cart version, resets the since-cart counter, sets the cart sequence for new lines.
        Drops freshness marks for lines whose unit count grew compared with before."""
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
        """Handles a successful add-to-cart call by scheduling stock reads for new lines.
        Notes when the cart view was shown, and finds first-time lines that are stale.
        Stores up to four ids in put_queue with the reading tool, if steps and time allow."""
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
        """Marks cart lines as freshly read when a reply shows their stock.
        Walks the reply for records matching cart ids with a stock state.
        Stores the current absorb counter for each such line; skips if stock was never seen."""
        if not self.__dict__.get("stock_seen"):
            return
        seqs = self.__dict__.setdefault("line_seq", {})

        def visit(d: dict[str, Any]) -> None:
            ident = self.rec_id(d)
            if ident in self.trolley.units and self.stock_state(d) is not None:
                seqs[ident] = self.__dict__.get("absorb_n", 0)
        walk_dicts(obs, visit)

    def cart_gone(self, lines: list[str]) -> set[str]:
        """Works out which of the given lines are sold out.
        Starts from known dead lines, then re-checks stale ones if a stock tool and steps exist.
        Returns the set of sold-out lines."""
        gone = set(lines) & self.dead_lines()
        stale = [i for i in self.stale_lines() if i in lines and i not in gone]
        if stale and self.stock_tool() and self.moves_left() >= len(stale) + 2:
            _, sold = self.code_recheck(stale)
            gone |= set(sold) | (set(lines) & self.dead_lines())
        return gone

    def batch_tool(self) -> tuple[str, str, int] | None:
        """Detects a tool that reads many listings in one call, caching the result.
        Needs one required string-array argument with a size limit and no cart or order purpose.
        Returns the tool name, argument name and slice size capped at ten, or None."""
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
                    if name not in taken and len(arrays) == 1 and not [r for r in needed if r not in arrays] and is_number(most) and most >= 2:
                        spec = (name, arrays[0], int(min(most, 10)))
                        break
            except Exception:
                spec = None
            self.batch_spec = spec
        return self.batch_spec

    @staticmethod
    def batch_ok(found: list[Any], ids: list[str]) -> bool:
        """Checks that a batch read actually returned usable detail for the requested ids.
        Walks replies for records that are nested or flat enough and collects ids that were asked for.
        Returns True if at least half of the requested ids appear."""
        got: set[str] = set()
        wanted = set(ids)

        def visit(d: dict[str, Any]) -> None:
            nested = any(isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2 for v in d.values())
            flat = not any(isinstance(v, dict) for v in d.values()) and sum(1 for v in d.values() if isinstance(v, str) and 0 < len(v) <= 70) >= 5
            if nested or flat:
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
        """Chooses unread listings of a shelf set to open next, to learn what fields they show.
        Skips read, sold-out, taken, refused and over-budget ones; favours search hits and new titles.
        Returns up to most ids, possibly led by a row from a search carrying stated values."""
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
        """Plans an early batch of listing reads before the agent asks the shopper anything.
        Needs a batch tool and spare steps; opens listings for thinly read shelves, in two rounds.
        Sets the open queue fields and returns True when a read is queued."""
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
        """Picks unread listings for products that have no verified fit yet.
        Finds home shelves with no passed line, then pools unread, unsold, in-budget rows newest first.
        Returns ids drawn round-robin across the lacking products, up to the batch limit."""
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
        """Maps a piece of text to the index of the requested product it concerns.
        Compares kind tokens of a parenthesised tag, then the whole text, against each product name.
        Returns the index with a unique best match, or None."""
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
        return None

    def plan_product(self, qitem: dict[str, Any]) -> int | None:
        """Finds which product a planned question item is about.
        Tries its property text first, then falls back to its question text.
        Returns a product index or None."""
        k = self.product_of(str(qitem.get("property") or ""))
        return k if k is not None else self.product_of(str(qitem.get("question") or ""))

    def quota_first(self, scripted: list[dict[str, Any]], ok: Any = None) -> list[dict[str, Any]]:
        """Reorders planned questions so every requested product gets asked about fairly.
        Counts questions asked and sent per product, then ranks products by need, neediest first.
        Returns the scripted list with a chosen or generated question moved to the front."""
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
        unasked = [j for j in range(len(rows)) if not asked[j]]
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

        def level_of(k: int) -> tuple[int, int, int, int, int]:
            return (1 if found[k] and any(j != k for j in unasked) else 0, 0 if k in hinted else 1, asked[k] + found[k],
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
                    pick = self.quota_pick(k, asked[k] == 0, ok)
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
        """Chooses one question about a property for product index k.
        Reads that shelf menu, trying a planned pick, hints, then ranked menu lines as fallback.
        Returns a property and question dict, or None; may queue listing reads first."""
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
        for weigh in (False, True):
            # value none of its sample values carry: story words written in the property's place; v2.5 V2: the lines in
            # ask_rank's order first)
            for line in self.rank_menu(k, menu):
                prop = line.strip("- ").split(" (in ")[0].strip()
                if prop and self.open_prop(k, prop, line, by_samples=weigh) and good(f"{prop} ({product})"):
                    return {"property": f"{prop} ({product})", "question": f"Which {prop} would you like for the {product}?"}
        return None

    def product_plan(self, k: int, menu: str) -> dict[str, Any] | None:
        """Safe wrapper that asks the planner for a question about product k from menu.
        Delegates to product_plan_call and swallows any exception.
        Returns its dict result, or None on failure."""
        try:
            return self.product_plan_call(k, menu)
        except Exception:
            return None

    def product_plan_call(self, k: int, menu: str) -> dict[str, Any] | None:
        if not menu or self.__dict__.get("product_plans", 0) >= 1 or self.wire.spare() < CLOSING + 40:
            return None
        self.product_plans = self.__dict__.get("product_plans", 0) + 1
        product = self.needs.items()[k]["product"]
        remaining = max(1, self.question_limit() - self.msgs_sent)      # (v2: the confirmation's message is kept)
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
        ask_text += self.rank_hint(k)          # (v2.5 V2: one ranked property of this product, after its menu)
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

    def rank_listings(self, k: int | None, every: bool = False) -> dict[str, dict[str, str]]:
        """v2.5 (V2): {listing id: {field name: value}} of the listings whose details were read - for product k (several
        products: the records of its own categories read, quota_pick's read_home; one product: its own category when
        another was read too, own_shelf), or every listing read (`every`). A listing's fields are the short text fields (at
        most 70 characters) of the property maps (2+ fields) its records hold, else a flat record's own when it has 5+ such
        fields (a search row has fewer); named and filtered as the question menu does it (prop_menu: underscores and dots
        as spaces, lower case, the menu's own field filter, id-like codes left out); the newest record of a field wins."""
        shelves = None
        rows = self.needs.items()
        if not every and rows:
            homes = self.home_shelves()
            home = {s.strip().lower() for s in (homes[k] if k is not None and k < len(homes) else set())}
            shelves = {s.strip().lower() for s in self.archive.shelves_read() if s.strip().lower() in home}
            if not shelves:
                return {}
        elif not every:
            own = self.own_shelf()
            shelves = {own.strip().lower()} if own else None
        found: dict[str, dict[str, str]] = {}

        def short(d: dict[str, Any]) -> dict[str, str]:
            out = {}
            for key, val in d.items():
                if not isinstance(val, str) or not 0 < len(val.strip()) <= 70 or junk_key(key):
                    continue
                if re.search(r"\d", val) and re.fullmatch(CODE_LIKE, val) and ("-" in val or "_" in val):
                    continue
                out[re.sub(r"[_.]+", " ", str(key)).strip().lower()] = val.strip()
            return out

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.archive.read), None)
            if not ident or (shelves is not None and self.archive.label_for(d).strip().lower() not in shelves):
                return
            maps = [val for val in d.values() if isinstance(val, dict) and sum(1 for v in val.values() if isinstance(v, (str, int, float))) >= 2]
            got: dict[str, str] = {}
            for val in maps:
                got.update(short(val))
            if not maps:
                own = {key: val for key, val in short(d).items() if "id" not in key.split()}
                got = own if len(own) >= 5 else {}
            if got:
                found.setdefault(ident, {}).update(got)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return found

    def ask_rank(self, k: int | None) -> list[dict[str, Any]]:
        """v2.5 (V2, the hidden-need study's rule): product k's property keys from its listings read (rank_listings), most
        promising first. Two names whose words (prop_words) contain one another and whose values agree on at least half of
        the listings carrying both are one key, named by its most carried name. Left out: a CONSTRAINED key - a requirement
        in force for the product names it (all the requirement property's words among the key's, or all the key's words
        among the requirement property's, or the same reduced name, prop_key), or open_prop finds it asked, settled or not
        minded; a VARIANT LIST - a name ending in a figure, or more than half of its values listing 3+ items (2+ commas or
        semicolons) or joining parts with " / " (one choice of each of the listing's variant dimensions, not one property's
        value); a CATCH-ALL - a per-listing label: over every
        listing read in the errand, 4+ carry it, its value differs on each and most of its values hold no figure. Ranked by
        coverage (the share of the product's listings read that carry the key), then by its number of distinct values, then
        the order first seen. Every name and value it uses comes from the records read and the requirements in force."""
        recs = self.rank_listings(k)
        if not recs:
            return []
        names: list[str] = []
        for facts in recs.values():
            names += [nm for nm in facts if nm not in names]
        words = {nm: set(prop_words(nm)) for nm in names}
        parent = {nm: nm for nm in names}

        def root(nm: str) -> str:
            while parent[nm] != nm:
                nm = parent[nm]
            return nm
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if not words[a] or not words[b] or not (words[a] <= words[b] or words[b] <= words[a]):
                    continue
                both = [f for f in recs.values() if a in f and b in f]
                agree = sum(1 for f in both if " ".join(f[a].lower().split()) == " ".join(f[b].lower().split()))
                if both and agree * 2 >= len(both) and root(a) != root(b):
                    parent[root(b)] = root(a)
        groups: dict[str, list[str]] = {}
        for nm in names:
            groups.setdefault(root(nm), []).append(nm)
        every = self.rank_listings(None, every=True)
        rows = self.needs.items()
        reqs = []
        for req in self.needs.props_left():
            head = str(req).split(":", 1)[0]
            if rows and k is not None and self.product_of(head) not in (k, None):
                continue
            reqs.append(head)
        ranked = []
        for order, members in enumerate(groups.values()):
            carry = list(members)
            vals = []
            for facts in recs.values():
                got = next((facts[nm] for nm in carry if nm in facts), None)
                if got is not None:
                    vals.append(" ".join(got.lower().split()))
            if not vals:
                continue
            label = max(carry, key=lambda nm: (sum(1 for f in recs.values() if nm in f), -len(nm), -names.index(nm)))
            if any(set(prop_words(head)) and words[nm] and (set(prop_words(head)) <= words[nm] or words[nm] <= set(prop_words(head)))
                   or prop_key(head) == prop_key(nm) for head in reqs for nm in carry):
                continue        # (constrained by a requirement in force)
            if not all(self.open_prop(k, nm) for nm in carry):
                continue        # (asked, settled or not minded - open_prop)
            if any(re.search(r"\d$", nm) for nm in carry) or sum(1 for v in vals if v.count(",") + v.count(";") >= 2 or " / " in v) * 2 > len(vals):
                continue        # (a variant list, or a combination of variant choices)
            wide = [next((" ".join(f[nm].lower().split()) for nm in carry if nm in f)) for f in every.values() if any(nm in f for nm in carry)]
            if len(wide) >= 4 and len(set(wide)) == len(wide) and sum(1 for v in wide if not re.search(r"\d", v)) * 2 > len(wide):
                continue        # (a catch-all: a per-listing label)
            ranked.append({"name": label, "cov": len(vals) / len(recs), "nval": len(set(vals)), "order": order, "names": carry,
                           "carry": len(vals), "of": len(recs)})
        ranked.sort(key=lambda g: (-g["cov"], -g["nval"], g["order"]))
        return ranked

    def rank_due(self) -> int | None:
        """v2.5 (V2): the product the next question goes to, round-robin: several products - the fewest questions asked so
        far (a question counts for the product its line names), the request's order breaking a tie; one product: None."""
        rows = self.needs.items()
        if not rows:
            return None
        asked = [0] * len(rows)
        for ln in self.questions_log:
            j = self.product_of(ln.split(":", 1)[0])
            if j is not None:
                asked[j] += 1
        return min(range(len(rows)), key=lambda j: (asked[j], j))

    def rank_hint(self, k: int | None = None, due: bool = True) -> str:
        """v2.5 (V2): one line for the question planner naming ONE property - the best-ranked key (ask_rank) of product k,
        else (`due`) of the product rank_due names; '' when no question is left or no key qualifies. A hint, not an
        override: the planner still chooses, under ASK_PLAN unchanged."""
        try:
            if not self.role("message") or self.msgs_sent >= self.question_limit():
                return ""
            if k is None and due:
                k = self.rank_due()
            ranked = self.ask_rank(k)
            if not ranked:
                return ""
            rows = self.needs.items()
            name = ranked[0]["name"] + (f" ({rows[k]['product']})" if rows and k is not None and k < len(rows) else "")
            return ("\n\nRANKED FROM THE LISTINGS READ (a hint, not a rule; story hints come first): of the fields no requirement "
                    f"or earlier question covers, the one most listings carry, with the most different values: {name}.")
        except Exception:
            return ""

    def rank_menu(self, k: int, menu: str) -> list[str]:
        """v2.5 (V2): quota_pick's menu lines for its code-picked question, the keys ask_rank ranks first in its order, every
        other line after them in the menu's own order (v2.4's order when nothing ranks)."""
        lines = menu.splitlines()
        try:
            ranked = self.ask_rank(k)
        except Exception:
            return lines
        first: list[str] = []
        for grp in ranked:
            for line in lines:
                prop = line.strip("- ").split(" (in ")[0].strip()
                if prop in grp["names"] and line not in first:
                    first.append(line)
        return first + [line for line in lines if line not in first]

    def open_prop(self, k: int | None, prop: str, line: str = "", by_samples: bool = False) -> bool:
        """Decides whether a property is still open to ask the shopper about.
        Rejects junk names, properties already asked, and ones settled by requirements or samples.
        Returns True when it is worth asking; True also on any error."""
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
        """Builds extra prompt text about unfinished question plans for product k.
        Lists second properties of earlier pairs still open, and wanted questions still relevant.
        Returns the assembled text, or an empty string on error."""
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
        """Remembers a question the assistant wanted to ask, from raw call arguments.
        Joins the string argument values, trimmed to 300 characters, and drops case-duplicates.
        Keeps only the latest three in wanted_asks; errors are ignored."""
        try:
            args = raw_args if isinstance(raw_args, dict) else (json_object(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {})
            text = " ".join(str(v).strip() for v in (args or {}).values() if isinstance(v, str) and v.strip())[:300]
            if text:
                kept = [w for w in self.__dict__.get("wanted_asks") or [] if w.lower() != text.lower()]
                self.wanted_asks = (kept + [text])[-3:]
        except Exception:
            pass

    def drop_wanted(self, label: str) -> None:
        """Removes remembered wanted questions that a newly asked label covers.
        Splits the label into properties and drops wanted texts containing all stemmed words of one.
        Replaces wanted_asks with the rest; errors are ignored."""
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
        """Tells whether a drafted question mentions some other property than prop.
        Compares its words against the second property, menu keys and menu sample values.
        Returns True if another property or value is named; False on error."""
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
        """Tells whether two properties can be asked together in a single paired question.
        Requires the same product, different fields, no shared words or values, no twin on the menu.
        Returns True if the pair is safe, else False (also on error)."""
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
        """Remembers a second property to ask about once the current question is answered.
        Appends prop to the pending_seconds list unless the same text is already queued.
        Comparison ignores letter case; empty text is skipped and nothing is returned."""
        queue = self.__dict__.setdefault("pending_seconds", [])
        if prop and prop.lower() not in [p.lower() for p in queue]:
            queue.append(prop)

    def pose_planned(self, urge: bool = False) -> bool:
        """Decides whether to ask the shopper a planned clarifying question now, and asks it.
        Gates on message tool, question ceiling, wait counters, step budget and listings read so far.
        Builds a planner prompt, trims it to fresh properties, sends one message, and logs the reply.
        Handles unanswered or paired asks by requeueing; returns True only when a turn was consumed."""
        the_tool = self.role("message")
        ceiling = self.question_limit()       # (v2: questions stop at N-1 while the confirmation's message is kept)
        if not the_tool or self.msgs_sent >= ceiling or self.__dict__.get("confirm_sent"):
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
            order = (mine or read) if all(homes) else mine + [s for s in read if s not in mine]
            menus = [(shelf, self.archive.prop_menu(shelf)[0]) for shelf in order[:4]]
            prop_list = "\n".join(f"CATEGORY {shelf}:\n{menu}" for shelf, menu in menus if menu) or prop_list
            shown = "\n".join(f"CATEGORY {shelf}:\n{self.menu_trim(menu, self.archive.rich_reads(shelf))}"
                              for shelf, menu in menus if menu) or shown
        ask_text += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n" + (shown or "(none seen yet)")
        ask_text += self.rank_hint(None)       # (v2.5 V2: one ranked property, after the menu)
        asked_props = {p.strip().lower() for ln in self.questions_log for p in ln.split(":", 1)[0].strip("- ").split(";")}

        def fresh(prop: Any) -> bool:
            prop = str(prop or "").strip()
            if not prop or prop.lower() in asked_props or junk_key(prop) or prop.lower() in self.__dict__.get("spent_asks", set()):
                return False
            j = self.product_of(prop) if self.needs.items() else None
            return not self.plain_hint(j, prop) and self.plan_open(j, prop)
        queue = [p for p in self.__dict__.get("pending_seconds") or [] if fresh(p)]
        self.pending_seconds = queue
        pending = queue[0] if queue else ""
        if pending:
            tied = re.search(r"\(([^)]+)\)", pending)
            product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
            text = (self.__dict__.get("pending_text") or {}).pop(pending.lower(), "")
            answer = {"questions": [{"property": pending, "question": text or f"Which {pending.split('(')[0].strip()} would you like for the {product}?"}]}
        else:
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
        if self.needs.items() and not pending:
            scripted = self.quota_first(scripted, fresh)
            if self.__dict__.pop("quota_wait", False):
                return True
        proposal = None
        split_second = ""
        for thing in scripted[:2]:
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
            proposal = self.message_call(str(thing.get("question")).strip(), f"ask_{self.msgs_sent + 1}_{self.wire.tick}")
            if proposal:
                label = (str(thing.get("property")), str(thing.get("question")).strip())
                second = str(thing.get("second") or "").strip()
                if second and second.lower() != label[0].lower() and not re.search(r"none|^-$|n/a", second.lower()):
                    if remaining == 1 and not self.__dict__.get("no_pairs") and self.pair_fits(label[0], second) \
                            and self.hint_mapped(self.product_of(label[0]) if self.needs.items() else None, label[0]):
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
                        single = f"Which {prop} would you like for the {product}?"
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
            self.asking = False
            return False
        self.asked_at_sets, self.asked_at_tick = sets, self.wire.tick
        self.chat.append({"role": "assistant", "content": f"I will ask the shopper about {label[0]}.", "tool_calls": [proposal]})
        found_before, freed_before = len(self.needs.uncovered), len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0)
        self.asking_label = label[0]
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
                redo = self.distinct_ask(first) if first.lower() not in again else None
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
            second = label[0].split(";", 1)[1].strip()
            found_text = " ".join(self.needs.uncovered[found_before:]).lower()
            if not any(w in found_text for w in re.findall(r"[a-z]{4,}", second.split("(")[0].lower())):
                self.queue_second(second)
                open_note = f" [{second}: not answered, still open]"
            label = (label[0].split(";")[0].strip(), label[1])
        elif ";" in label[0] and len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0) > freed_before:
            second = label[0].split(";", 1)[1].strip()
            label = (label[0].split(";")[0].strip(), label[1])
            self.needs.indifferent = [p for p in self.needs.indifferent if p.strip().lower() not in (second.lower(), second.split("(")[0].strip().lower())]
            pairs = self.__dict__.setdefault("open_pair", [])
            if second.lower() not in [p.lower() for p in pairs]:
                pairs.append(second)
            open_note = f" [{second}: not answered, still open]"
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
        if not self.role("message") or self.msgs_sent >= self.question_limit() or self.moves_left() <= 4 or self.wire.spare() < CLOSING \
                or (self.asking and self.msgs_sent >= self.question_limit() - 1 and self.moves_left() > 10 and self.wire.spare() >= CLOSING + 40):
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

    def plain_args(self, call_name: str, text_body: str, kind: str = "") -> dict[str, Any]:
        jschema = self.shapes.get(call_name) or {}
        prop_specs = jschema.get("properties") or {}
        call_args: dict[str, Any] = {}
        words = DECLINE_WORDS if kind == "decline" else ORDER_WORDS if kind == "order" else ""
        for key in jschema.get("required") or []:
            pspec = prop_specs.get(key) or {}
            if pspec.get("enum"):
                read = (self.finish_values(call_name).get(key) or {}) if kind else {}      # (v1.1/v1.2: the per-value reading first)
                fit = [val for val in pspec["enum"] if isinstance(val, str) and read.get(val) == kind] or \
                    [val for val in pspec["enum"] if words and isinstance(val, str) and re.search(words, val.lower())]
                if not fit and kind == "decline" and "order" in read.values():
                    # (v1.2: a decline never sends a value read as the order)
                    fit = [val for val in pspec["enum"] if read.get(val) != "order"]
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

    def code_reading(self) -> dict[str, Any]:
        """Builds a rule-based reading of the request when no model reading is available.
        Extracts price cap, stock wish, cheapest priority, message and step limits from the wording.
        Detects tool roles and ordered items; returns a dict with product, quantity, requirements."""
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
        """Finds the spending limit stated in the request text and the unit attached to it.
        Splits into sentences and rates each number by currency sign, spending words and limit words.
        Skips numbers that count messages or items; returns the best figure and unit, or None and None."""
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
        """Reads a numeric allowance such as a cap on a counted noun from lowercase request text.
        Matches a digit or number word before the noun regex, then needs a limiting phrase nearby.
        Returns the integer found, or None when no number is phrased as a limit."""
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
        """Reads how many units of the product the request wants, defaulting to one.
        Looks for a number followed by units, copies, or of them in the lowercase text.
        Returns the value only when it is between 1 and 99; otherwise returns 1."""
        figures = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        said = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:units?|of (?:them|those|these)|copies)\b", low)
        if not said:
            return 1
        n = int(said.group(1)) if said.group(1).isdigit() else figures[said.group(1)]
        return n if 1 <= n <= 99 else 1

    @staticmethod
    def code_needs(seg: str) -> list[tuple[str, str]]:
        """Pulls property and value pairs out of phrases saying a property must or should be a value.
        Applies a sentence pattern, then drops pairs with reference text, long or pronoun-like parts.
        Returns a list of (property, value) tuples taken from the segment seg."""
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
        """Finds the separate products with counts that the request asks for, using a search tool enum.
        Matches enum labels against request words, reads a nearby count, and orders hits by position.
        Each result holds product, label, quantity and the requirements found in the text after it."""
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
        """Guesses which tool plays each role by studying tool names, descriptions and argument schemas.
        Finds order and decline from enum values, cart and cart view from basket wording, then others.
        Cues cover search, message, details and view; returns a dict mapping role name to tool name."""
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
        # (v1 P1.1: the finishing tools by their primary effect - finish_reading - in place of the first finishing word of
        # a name, which took 'order' from a name that ends the errand without buying; the enum pass above stays first)
        try:
            fin = self.finish_reading()
        except Exception:
            fin = {}
        for role in ("order", "decline"):
            if role not in roles and fin.get(role):
                roles[role] = fin[role]
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
        speak_more = r"\b(?:send\w*|repl\w*|notes?|tell\w*|messag\w*|chat\w*|writ\w*|talk\w*|ask\w*|say\w*|said|question\w*)\b"   # (v1 P1.5)
        cues = [("search", r"\b(?:search\w*|find\w*|look\w*|locat\w*|hunt\w*|brows\w*|quer\w*)\b", False),
                ("message", speak_more, True),
                ("details", r"\b(?:details?|records?|spec\w*|propert\w*|attribut\w*|facts?|full)\b", True),
                ("view", r"\b(?:offer|price|stock|availab\w*)\b", True)]
        for role, cue, one_text in cues:
            for name in said:
                # (v1 P1.5: for the message tool its argument names count too - Ask(ChatText) with no description)
                hay = said[name] + (" | " + " ".join(name_words(key) for key in props(name)) if role == "message" else "")
                if role in roles or name in roles.values() or not re.search(cue, hay):
                    continue
                if role == "search" and re.search(person, said[name]) and re.search(speak, said[name]):
                    continue
                if role == "message" and not (re.search(person, hay) or re.search(r"\b(?:messag\w*|chat\w*)\b", hay)):
                    continue
                if one_text and not (len(needed(name)) == 1 and needed(name)[0] in texts(name)):
                    continue
                if not one_text and not (texts(name) or any(isinstance(spec, dict) and spec.get("enum") for spec in props(name).values())):
                    continue
                roles[role] = name
        if "message" not in roles:
            # (v1 P1.5: no speaking word - a tool whose one required argument is free text, no id or query, and whose name or
            # description names the shopper is the message tool; the first by name when several fit)
            fits = [name for name in said if name not in roles.values() and len(needed(name)) == 1 and needed(name)[0] in texts(name)
                    and not re.search(r"\b(?:ids?|codes?|sku\w*|refs?|numbers?|quer\w*|keywords?|terms?|search\w*)\b", name_words(needed(name)[0]))
                    and re.search(person, said[name])]
            if fits:
                roles["message"] = min(fits)
        return roles

    @staticmethod
    def main_clauses(text_body: Any) -> list[str]:
        """v1 (P1.1): a tool description's main clauses in order, lower case: each sentence or ';' / ':' part up to a
        condition or an aside (if, unless, when, otherwise, except, but, before, after, once, until, '('); a part that
        opens with a condition keeps only what follows its first comma."""
        found: list[str] = []
        for part in re.split(r"[.;:!?\n]+", str(text_body or "").lower()):
            part = part.strip()
            if re.match(r"(?:if|unless|when|whenever|once|after|before|until|in case|otherwise)\b", part):
                part = part.split(",", 1)[1].strip() if "," in part else ""
            part = re.split(r"\(|\s(?:if|unless|when|whenever|otherwise|except|but|before|after|once|until)\s", " " + part + " ")[0].strip()
            if part:
                found.append(part)
        return found

    def finish_reading(self) -> dict[str, Any]:
        """v1 (plan v2 Phase 1 item 1): the order and decline tools by their PRIMARY effect, with 'sure' / 'unsure' per role.
        A tool whose required choice holds an order word and a decline word does both (HW's enum pass, first). Otherwise a
        candidate is a tool that is no cart change (put / take choice, or an item argument with a put or take verb) and no
        read, list, search or message tool unless it says it finishes, and that finishes or states a buy / no-buy effect in
        its name or its description's first sentence (a weak finishing word in a name only for a tool with no required
        argument but a choice; in a description only with the errand as its object). Its effect: the first of the description's main clauses that states
        one (a no-purchase phrase beats an order word), else its name's words (the same rule). 'sure' when the description
        states the effect and the name does not say the other; a name alone, or a name against its description, is unsure.
        Several candidates for a role: the surest, then the description's word, the name's, a finishing word, fewer
        required arguments, then the name. With exactly two candidates, one role taken and the other candidate stating no
        effect, that one takes the free role (unsure); a lone candidate stating no effect takes both roles (unsure); never
        a pairing by elimination otherwise. Returns {"order", "decline": tool or None, "sure": {role: bool}, "cands":
        {tool: effect or None}}."""
        out: dict[str, Any] = {"order": None, "decline": None, "sure": {"order": False, "decline": False}, "cands": {}}

        def props(name: str) -> dict[str, Any]:
            found = (self.shapes.get(name) or {}).get("properties")
            return found if isinstance(found, dict) else {}

        def needed(name: str) -> list[str]:
            found = (self.shapes.get(name) or {}).get("required")
            return [key for key in found if key in props(name)] if isinstance(found, list) else []

        def effect(text_body: str) -> str | None:
            return "decline" if re.search(NO_BUY, text_body) else ("order" if re.search(BUY_WORDS, text_body) else None)
        for tdef in self.toolset:
            name = tdef["function"]["name"]
            for key in needed(name):
                vals = [str(v).lower() for v in (props(name)[key].get("enum") or [])] if isinstance(props(name)[key], dict) else []
                if (any(re.search(ORDER_WORDS, v) for v in vals) and any(re.search(DECLINE_WORDS, v) for v in vals)) \
                        or {"order", "decline"} <= set((self.choice_reading(name).get(key) or {}).values()):    # (v1.1)
                    out.update({"order": name, "decline": name, "sure": {"order": True, "decline": True}, "cands": {name: "both"}})
                    return out
        rows: list[tuple[str, str | None, bool, bool, bool, bool, int]] = []
        for tdef in self.toolset:
            name = tdef["function"]["name"]
            head = name_words(name)
            desc = str(tdef["function"].get("description") or "").strip()
            mains = self.main_clauses(desc)
            first = self.main_clauses(re.split(r"(?<=[.!?])\s+|\n", desc)[0]) if desc else []
            lead = first[0] if first else ""
            name_way = effect(head)
            desc_way = next((way for way in map(effect, mains) if way), None)
            # (a weak finishing word - end, leave, complete, close ... - counts in a name only for a tool that takes no
            # required argument but a choice: LeaveNote(text) or CompleteRecord(id) does not finish)
            free_args = [key for key in needed(name) if not (isinstance(props(name)[key], dict) and props(name)[key].get("enum"))]
            fin = bool(re.search(FINISH_STRONG, head) or (re.search(FINISH_WEAK, head) and not free_args)
                       or re.search(FINISH_STRONG, lead) or re.search(FINISH_WEAK + ERRAND_OBJ, lead))
            no_buy = bool(re.search(NO_BUY, head) or any(re.search(NO_BUY, c) for c in first))
            vals = {w for spec in props(name).values() if isinstance(spec, dict) for v in spec.get("enum") or [] for w in name_words(v).split()}
            item_arg = any(isinstance(spec, dict) and (spec.get("type") in ("array", "integer", "number")
                           or re.search(r"\b(?:ids?|codes?|sku\w*|items?|listings?|products?|lines?)\b", name_words(key)))
                           for key, spec in props(name).items())
            cart_like = bool(vals & PUT_WORDS and vals & DROP_WORDS) or (item_arg and not fin and name_way != "order" and bool(re.search(
                r"\b(?:add|put|insert|remove|drop|release|delete|discard|hold|unhold|increase|decrease|append|clear|set)\w*\b", head + " " + lead)))
            busy = cart_like or (not fin and not no_buy and bool(re.search(BUSY_WORDS, head + " | " + lead)))
            if busy or not (name_way or fin or any(effect(c) for c in first)):
                continue
            way = desc_way or name_way
            rows.append((name, way, bool(desc_way) and name_way in (None, desc_way), bool(way) and desc_way == way,
                         bool(way) and name_way == way, fin, len(needed(name))))
            out["cands"][name] = way
        for role in ("order", "decline"):
            mine = sorted((row for row in rows if row[1] == role), key=lambda row: (not row[2], not row[3], not row[4], not row[5], row[6], row[0]))
            if mine:
                out[role], out["sure"][role] = mine[0][0], mine[0][2]
        loose = sorted(row[0] for row in rows if row[1] is None)
        free = [role for role in ("order", "decline") if not out[role]]
        if len(rows) == 1 and loose:
            out["order"] = out["decline"] = loose[0]
        elif len(rows) == 2 and len(free) == 1 and len(loose) == 1:
            out[free[0]] = loose[0]
        return out

    def choice_reading(self, name: str) -> dict[str, dict[str, str]]:
        """v1.1 (enum finishers): each choice (enum) argument of a tool as {value: 'order' or 'decline'}, read per value:
        the value's own words first (HW's decline words, then its order words, then the effect grammar on the value's
        words), else the part of the description that names that value (quoted, or as a whole word of 3+ characters). A
        part naming one value is read as a whole; a part naming several gives each value the stretch after its mention
        when the part opens with no effect, else the stretch before it when the part ends with no effect, else nothing.
        A value neither way decides is left out. Cached per tool."""
        kept = self.__dict__.setdefault("choice_kept", {})
        if name in kept:
            return kept[name]
        shape = self.shapes.get(name) or {}
        props = shape.get("properties") if isinstance(shape.get("properties"), dict) else {}
        desc = next((str(tdef["function"].get("description") or "") for tdef in self.toolset if tdef["function"]["name"] == name), "")
        parts = [part.strip() for part in re.split(r"[.;!?\n]+", desc.lower()) if part.strip()]
        quote = "[\"'`\u2018\u2019\u201c\u201d]"

        def way(text_body: str) -> str | None:
            return "decline" if re.search(NO_BUY, text_body) else ("order" if re.search(BUY_WORDS, text_body) else None)

        def by_words(val: str) -> str | None:
            low = val.lower()
            if re.search(DECLINE_WORDS, low):
                return "decline"
            if re.search(ORDER_WORDS, low):
                return "order"
            return way(name_words(val))

        def spot(val: str) -> str:
            low = re.escape(val.lower())
            if len(val) >= 3 and re.fullmatch(r"[A-Za-z0-9_\- ]+", val):
                return quote + low + quote + r"|(?<![\w-])" + low + r"(?![\w-])"
            return quote + low + quote
        out: dict[str, dict[str, str]] = {}
        for key, spec in props.items():
            vals = [val for val in (spec.get("enum") or []) if isinstance(val, str) and val.strip()] if isinstance(spec, dict) else []
            if len(vals) < 2:
                continue
            read = {val: by_words(val) for val in vals}
            pats = {val: spot(val) for val in vals}
            for part in parts:
                hits = sorted((m.start(), m.end(), val) for val in vals for m in re.finditer(pats[val], part))
                named = {val for _, _, val in hits}
                if len(named) == 1:
                    val = hits[0][2]
                    read[val] = read[val] or way(re.sub(pats[val], " ", part))
                    continue
                if not named:
                    continue
                head, tail = part[: hits[0][0]], part[hits[-1][1]:]
                for idx, (start, end, val) in enumerate(hits):
                    if read[val]:
                        continue
                    if not way(head):
                        read[val] = way(part[end: hits[idx + 1][0] if idx + 1 < len(hits) else len(part)])
                    elif not way(tail):
                        read[val] = way(part[hits[idx - 1][1] if idx else 0: start])
            got = {val: kind for val, kind in read.items() if kind}
            if got:
                out[key] = got
        kept[name] = out
        return out

    def finish_values(self, name: str) -> dict[str, dict[str, str]]:
        """v1.2: choice_reading for a finishing call. For the enum finisher (one tool that is both the order and the decline
        tool), a two-value choice with one value read as order and the other unread: the other value is the decline
        (elimination). Any other tool: choice_reading as it is."""
        reading = {key: dict(got) for key, got in self.choice_reading(name).items()}
        if not name or self.role("order") != name or self.role("decline") != name:
            return reading
        shape = self.shapes.get(name) or {}
        props = shape.get("properties") if isinstance(shape.get("properties"), dict) else {}
        for key, spec in props.items():
            vals = [val for val in (spec.get("enum") or []) if isinstance(val, str) and val.strip()] if isinstance(spec, dict) else []
            got = reading.get(key) or {}
            if len(vals) == 2 and list(got.values()) == ["order"]:
                reading[key] = {**got, next(val for val in vals if val not in got): "decline"}
        return reading

    def both_ways(self, name: str | None) -> bool:
        """v1.1: the tool has a required choice that picks between buying and not buying - HW's test (an order word and a
        decline word among its values) or the per-value reading (one value read as order and one as decline)."""
        shape = self.shapes.get(name or "") or {}
        props = shape.get("properties") if isinstance(shape.get("properties"), dict) else {}
        need = [key for key in shape.get("required") or [] if key in props] if isinstance(shape.get("required"), list) else []
        for key in need:
            vals = [str(val).lower() for val in (props[key].get("enum") or [])] if isinstance(props[key], dict) else []
            if any(re.search(ORDER_WORDS, val) for val in vals) and any(re.search(DECLINE_WORDS, val) for val in vals):
                return True
            if {"order", "decline"} <= set((self.choice_reading(name or "").get(key) or {}).values()):
                return True
        return False

    def may_choose(self, name: str | None) -> bool:
        """v1.1: one tool may be both the order and the decline tool - both_ways, or it has a required choice of 2+ values
        that the per-value reading does not put wholly in one kind (a value nothing reads may be the other kind). False
        for a tool with no such choice."""
        if self.both_ways(name):
            return True
        shape = self.shapes.get(name or "") or {}
        props = shape.get("properties") if isinstance(shape.get("properties"), dict) else {}
        need = [key for key in shape.get("required") or [] if key in props] if isinstance(shape.get("required"), list) else []
        reading = self.choice_reading(name or "")
        for key in need:
            vals = [val for val in (props[key].get("enum") or []) if isinstance(val, str) and val.strip()] if isinstance(props[key], dict) else []
            kinds = reading.get(key) or {}
            if len(vals) >= 2 and not (len(kinds) == len(vals) and len(set(kinds.values())) == 1):
                return True
        return False

    def clause_way(self, name: str | None) -> str | None:
        """v1.1: a tool's primary effect as its description states it - the first main clause carrying an explicit
        no-purchase signal ('decline') or buy signal ('order'); None without one (a name is never read here)."""
        desc = next((str(tdef["function"].get("description") or "") for tdef in self.toolset if tdef["function"]["name"] == name), "")
        for clause in self.main_clauses(desc):
            if re.search(NO_BUY, clause):
                return "decline"
            if re.search(BUY_WORDS, clause):
                return "order"
        return None

    def settle_finishers(self, reading: dict[str, Any] | None = None) -> None:
        """v1.1 (precedence; replaces v1's): a VALID reader answer for the order or the decline tool wins. The code replaces
        it only when (a) the answer is invalid - the role is missing, names no tool in the toolset, or names one tool for
        both roles where no choice of that tool can pick between buying and not buying (may_choose) - or (b) the tool's own
        primary-effect clause directly contradicts the role (the order tool's states no purchase, the decline tool's a
        purchase). Such a role takes the code's tool when the code is sure, else the finish_check model answer, else the
        code's guess, else none - never the tool the reader validly gave the other role, unless one tool can do both. The code-only path (no reader) never comes here. Provenance: self.finish_note."""
        try:
            fin = self.finish_reading()
        except Exception as err:
            self.log.append({"kind": "finish_reading_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return
        raw = (reading or {}).get("roles")
        raw = raw if isinstance(raw, dict) else {}
        said = {role: self.role_map.get(role) for role in ("order", "decline")}
        why: dict[str, str] = {}
        for role in ("order", "decline"):
            if not said[role]:
                named = raw.get(role)
                why[role] = "not_a_tool" if isinstance(named, str) and named.strip().lower() not in ("", "null", "none") else "missing"
        same = bool(said["order"]) and said["order"] == said["decline"]
        both = same and self.may_choose(said["order"])
        if same and not both:
            why["order"] = why["decline"] = "same_tool"

        def clash(role: str, tool: Any) -> bool:
            # (a replacement never takes the tool the reader validly gave the other role, unless one tool can do both)
            other = "decline" if role == "order" else "order"
            return other not in why and tool == self.role_map.get(other) and not self.may_choose(tool)
        # (v1.2: a contradiction replaces a valid reader role only when the code holds a SURE, DIFFERENT tool for that role
        # that is free to take; otherwise the reader stands)
        hit = [role for role, other in (("order", "decline"), ("decline", "order"))
               if role not in why and not both and self.clause_way(said[role]) == other
               and fin.get(role) and fin["sure"][role] and fin[role] != said[role]]
        for role in hit:
            why[role] = "contradicted"
        for role in hit:
            if clash(role, fin[role]):
                why.pop(role, None)
        how = {role: "reader" for role in ("order", "decline") if role not in why}
        doubt: list[str] = []
        for role in ("order", "decline"):
            if role not in why:
                continue
            if fin.get(role) and fin["sure"][role] and not clash(role, fin[role]):
                self.role_map[role], how[role] = fin[role], "code_sure"
            else:
                doubt.append(role)
        check = self.finish_check(fin) if doubt else None
        for role in doubt:
            pick = (check or {}).get(role)
            if pick and not clash(role, pick):
                self.role_map[role], how[role] = pick, "model_check"
            elif fin.get(role) and not clash(role, fin[role]):
                self.role_map[role], how[role] = fin[role], "code_guess"
            else:
                self.role_map.pop(role, None)
                how[role] = "none"
        self.finish_note = {"reader": said, "code": {role: fin.get(role) for role in ("order", "decline")},
                            "code_sure": dict(fin["sure"]), "why": why, "how": how, "check": check}

    def finish_check(self, fin: dict[str, Any]) -> dict[str, Any] | None:
        """v1 (P1.2): one small model call (READER ladder, 300 tokens) on the finishing tools alone: which places the order,
        which ends without buying. Valid only when each answer is null or one of the tools shown, and one tool is named for
        both only when it is the only one shown; None when no model answers or the answer is not valid."""
        names = sorted(set(fin.get("cands") or {}) | {val for val in (self.role_map.get("order"), self.role_map.get("decline")) if val})
        if not names or self.wire.mute or self.wire.no_model:
            return None
        rows = [f"- {tdef['function']['name']}: {tdef['function'].get('description', '')} | arguments: {as_text(tdef['function'].get('parameters') or {})}"
                for tdef in self.toolset if tdef["function"]["name"] in names]
        msg = self.wire.query_model([{"role": "user", "content": FINISH_CHECK + "\n\nTOOLS:\n" + "\n".join(rows)}], READER, 300)
        got = json_object((msg or {}).get("content") or "") or {}
        picks = {role: got.get(role) for role in ("order", "decline")}
        if any(val is not None and val not in names for val in picks.values()):
            return None
        if picks["order"] and picks["order"] == picks["decline"] and len(names) > 1:
            return None
        return picks

    def actor_tools(self) -> list[dict[str, Any]]:
        """v1 (plan v2 Phase 1 item 3): the toolset as the actor sees it - each finishing tool's description ends with a
        short note of its effect; names and argument schemas unchanged. Kept while the roles stay, so the prompt stays
        the same from turn to turn."""
        order, decline = self.role("order"), self.role("decline")
        kept = self.__dict__.get("actor_set")
        if kept and kept[0] == (order, decline):
            return kept[1]
        shown: list[dict[str, Any]] = []
        for tdef in self.toolset:
            fn = tdef["function"]
            note = ""
            if fn["name"] == order and order == decline:
                note = "Effect: ends the errand; its choice argument says whether the current cart is placed as the order or nothing is bought."
            elif fn["name"] == order:
                note = "Effect: places the current cart as the order and ends the errand."
            elif fn["name"] == decline:
                note = "Effect: ends the errand WITHOUT buying anything."
            if note:
                tdef = {**tdef, "function": {**fn, "description": (str(fn.get("description") or "").rstrip() + " " + note).strip()}}
            shown.append(tdef)
        self.actor_set = ((order, decline), shown)
        return shown

    def decline_hold(self, call_name: str, call_args: dict[str, Any]) -> str:
        """v1 (plan v2 Phase 1 item 3): an actor's decline while the cart holds lines is held once per set of cart lines (no
        step used), with a note naming the order tool; sent again, it goes on to screen_finish's decline checks."""
        order = self.role("order")
        if not order or self.call_kind(call_name, call_args) != "decline":
            return ""
        lines = sorted(line for line in self.trolley.units if self.trolley.held(line) > 0)
        told = self.__dict__.setdefault("cart_decline_told", set())
        if not lines or tuple(lines) in told:
            return ""
        told.add(tuple(lines))
        how = f"{order} with its order choice" if order == call_name else order
        return (f"Not run yet (no step used). The cart holds {', '.join(lines)}, and {call_name} ends the errand WITHOUT buying "
                f"anything: the cart would be thrown away. To buy the cart, call {how}. Send the decline again only if you mean "
                "to leave without buying.")

    def code_label(self, tool: str | None, want: set[str]) -> str:
        """Picks the catalog category label that best matches the product words in want.
        Takes the largest enum on the search tool schema and counts shared words per label.
        Returns the label with most overlap, ties broken by share of its words; empty if none."""
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
        """Lists the property values still needed that are safe to use as search words.
        Keeps requirements for product k, strips status tags, drops references or long excluding text.
        Returns (property name, plain value) pairs for values that are positive or short."""
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
        """Builds ready-to-send search calls for one product by reading the search tool schema.
        Picks text, category, mode and price-cap arguments, fills required fields, trims to max length.
        Each call is checked with ArgCheck; returns the distinct valid argument dicts, possibly none."""
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
        """Summarises every listing record in the archive into one info dict per listing id.
        Collects category, title, price, stock state, word set and field values by walking records.
        Returns a mapping from id to those details; only ids already known to the archive are kept."""
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
        """Tells whether a listing row displays the wanted value for a property.
        Uses fields whose name matches prop; else checks the title, then search words only if unread.
        Returns True when the value is visible by any of those routes, otherwise False."""
        named = [val for key, val in row["fields"].items() if same_prop(key, prop)]
        if named:
            return any(Errand.code_seen(value, val) for val in named)
        if Errand.code_seen(value, row.get("title") or ""):
            return True
        own = kind_words(value)
        return not read and bool(own) and own <= hit_words

    @staticmethod
    def code_seen(value: str, text: str) -> bool:
        """Checks whether text contains the value, allowing alternatives and number matching.
        Splits value on or, slash, comma or semicolon; every word stem and every figure must appear.
        Returns True if any alternative is fully present in text, False for empty text."""
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
        """Chooses which listings to put in the cart, one per wanted product with quantities.
        Filters by shelf, stock, failed checks, field clashes, budget; ranks by evidence and price.
        Respects avoid and keep sets, tries a second ranking if over budget, sets code_fit on success.
        Returns a list of (id, quantity) pairs, or an empty list when any product has no candidate."""
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
        """Reports whether a row states a different value for a property than the one wanted.
        Reads the row fields whose names match prop, then compares each against value with value_seen.
        Returns True only if such fields exist and none of them shows the value."""
        named = [val for key, val in fields.items() if same_prop(key, prop)]
        return bool(named) and not any(value_seen(value, val) for val in named)

    def cart_form(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None, list[str]]:
        """Works out how the cart tool must be called from its argument schema and past usage.
        Chooses put and drop values for required enum fields, an id text key and numeric count keys.
        Returns the schema, put dict, drop dict, id key (or None) and the list of count keys."""
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
        """Turns chosen picks into validated argument dicts for the cart tool.
        Uses one call with an array of lines when the cart takes lines, else one call per pick.
        Every call goes through ArgCheck; returns an empty list on any fault or missing id key."""
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
        """Builds cart calls that remove the given lines from the basket.
        Uses the drop enum values with the held quantity, or a zero count when the schema allows it.
        Returns the validated calls, an empty list if none needed, or None when removal is impossible."""
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
        """Fetches the newest tool observations from the log after a batch was sent.
        Finds the latest environment entry and reads its observations list or single observation.
        Returns at most n observations, or an empty list when the reply has an unexpected shape."""
        entry = next((e for e in reversed(self.log) if isinstance(e, dict) and e.get("kind") == "env"), None)
        resp = (entry or {}).get("reply")
        if not isinstance(resp, dict):
            return []
        obs_list = resp.get("observations") if isinstance(resp.get("observations"), list) else [resp.get("observation")]
        return list(obs_list)[:n]

    @staticmethod
    def code_fault(obs: Any) -> bool:
        """Detects whether a tool observation reports an error condition.
        Walks nested dicts to a shallow depth looking for keys that name faults with a non-empty value.
        Returns True when at least one such key is found, otherwise False."""
        found = []

        def visit(d: dict[str, Any]) -> None:
            if any(re.search(FAULT_KEYS, str(key).lower()) and val not in (None, False, "", [], {}) for key, val in d.items()):
                found.append(d)
        walk_dicts(obs, visit, 0, 2)
        return bool(found)

    def code_search(self, products: int, more: bool = False) -> None:
        """Runs the scripted search phase: sends searches, records hits, and follows result pages.
        Builds calls from code_queries (or queued next pages when more is set) within the move reserve.
        Stores row hits per search words, sets follow-up pages in code_more, and stops early on a fit.
        Returns nothing; progress shows in code_searched, code_hits and the archive."""
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
        """Computes the arguments for the next page of a search from its latest observation.
        Finds the result page dict, a continuation token or page counter, and any total count.
        Returns validated args for the following page, or None when results are exhausted or invalid."""
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
        """Opens full detail records for picked listings that have not been read yet.
        Uses a batch detail tool when two or more are unread, else one details call each in budget.
        Re-picks after each read round and stops when picks stop changing; returns the final picks."""
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
        """Brings the cart in line with picks by removing extra lines and adding missing ones.
        Sends the cart calls, inspects replies for faults, and may view the cart to confirm contents.
        Removes failed lines from the local cart record; returns sorted bad ids, or None if impossible."""
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
        """Re-reads stock for the given cart lines using the stock, view or details tool.
        Needs enough moves and spare time; sends one lookup per line and rebuilds the row info.
        Returns (whether the check ran, set of ids now found out of stock)."""
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
        """Scans shopper reply lines for a new spending limit and adopts it.
        Looks at the given lines plus any waiting ones, each only once, using code_budget on the text.
        A positive figure replaces needs.spend_cap and a note is added to the log; returns nothing."""
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
        """Builds the final cart without the model: search, read, put, then recheck stock.
        Loops up to four times, adding avoided ids for lines that failed or sold out and picking again.
        Logs each attempt outcome; returns True if the cart holds lines, False otherwise or on error."""
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
        """Forces the run to end by sending the order call, or a decline if no order is possible.
        Checks which lines are still available, builds the cart with code_cart, retries within a limit.
        After each order it rechecks stock and the budget and reorders or swaps lines when needed.
        Falls back to the decline tool when an order fails; logs every forced step and returns nothing."""
        if self.wire.done or self.wire.broke or self.moves_left() <= 0 or self.wire.spare() < 2:
            return
        self.change_back()      # (v2.4.2 Q4: an unfinished change after a change hold is put back first)
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
                        gone = self.kept_gone(lines0)      # (v2.5.1 U2)
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
                    _aoi_block(self, None, None, "finish_now: no order to send and no decline tool", "finish_now")
                    return
            if self.call_kind(move["name"], move["args"]) == "order":
                self.confirm_order(final=True)      # (v2.1 E3: the code's own order shows its basket first, reply unread)
            _aoi_t = _aoi_finish_before(self, [move])
            resp = self.wire.act([move])
            _aoi_finish_after(self, _aoi_t, resp)
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
            if again:
                self.store_cart, self.cart_touched, self.store_partial = None, True, None      # (v2.2 F3: the refusal above was not absorbed; the store may have changed the cart)
                self.confirm_order(final=True)      # (v2.1 E3: as above, for the order sent again)
            _aoi_t = _aoi_finish_before(self, [move])
            self.log.append({"kind": "forced_finish", "action": move, "reply": self.wire.act([move])})
            _aoi_finish_after(self, _aoi_t, _AOI_LAST)

    def again_ok(self, avoid: set[str]) -> bool:
        """Decides whether sending the order again is worthwhile after a refusal.
        True when the cart has lines, none of them are in avoid or the known dead lines.
        Also requires code_picks to still find a valid set of picks once those lines are excluded."""
        units = set(self.trolley.units)
        gone = set(avoid) | self.dead_lines()
        return bool(units) and not units & gone and bool(self.code_picks(gone))

    def catch_up(self) -> None:
        """Folds late information into the chat before the run closes.
        May ask the reader model to re-read the request if replies arrived after the first reading.
        Also processes waiting shopper lines and appends the notes to the last message with a status.
        Skipped when muted or short on time; errors are logged and never raised."""
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
                    _aoi_roles(self, "catch_up", reading)
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
        """An order or decline call whose episode did not finish, or any result holding a refusal field. (v1 P1.4: a decline
        too - a finish never counts as the end until the reply says the episode finished.)"""
        if isinstance(obs, dict) and "refusal" in obs:
            return True
        return call_name in self.finishers() and not resp.get("finished") and not (isinstance(obs, dict) and obs.get("error") in ("invalid_args", "unknown_tool"))

    def notes_for(self, call_name: str, obs: Any, refused: bool) -> str:
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
        """Reads the category name of a listing row from a field named like a shelf.
        Takes the first short string under a key matching category, department or aisle, not an id key.
        Returns it stripped and lowercased, or an empty string when nothing fits."""
        return next((val.strip().lower() for key, val in row.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                     and re.search(r"categor|shelf|department|aisle", str(key).lower()) and "id" not in str(key).lower()), "")

    @staticmethod
    def key_stock(key: Any, val: Any) -> bool | None:
        """Interprets one field of a row as an availability signal.
        Only keys named like stock, availability or on hand count; booleans, numbers and text are read.
        Returns True if in stock, False if gone, and None when the key says nothing about stock."""
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
        """Combines every availability signal in a row into one state.
        Runs key_stock on each field and ignores fields that carry no stock signal.
        Returns True if all signals are positive, False if any is negative, None if there are none."""
        states = [s for s in (Errand.key_stock(key, val) for key, val in row.items()) if s is not None]
        return all(states) if states else None

    @staticmethod
    def sold_out(row: dict[str, Any]) -> bool:
        """Reports whether any field of a listing row says the item is unavailable.
        Applies key_stock to each key and value pair of the row.
        Returns True as soon as one field yields False, otherwise False."""
        return any(Errand.key_stock(key, val) is False for key, val in row.items())

    def shelf_labels(self) -> set[str]:
        """Finds which category labels the read listings mostly belong to.
        Tallies shelves of read rows, skipping foreign references, and takes the most common ones.
        With several products it matches each home shelf against the tally; returns the set of labels."""
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
        """Gives a stable identifier for a record dict.
        Prefers a key named exactly id, then any key ending in id, holding a string or integer.
        Falls back to a sorted JSON dump of the whole row, so the result is always a string."""
        for exact in (True, False):
            for key, val in row.items():
                name = str(key).lower()
                if (name == "id" if exact else name.endswith("id")) and isinstance(val, (str, int)) and not isinstance(val, bool):
                    return str(val)
        return json.dumps(row, sort_keys=True, default=str)

    def search_keys(self, call_args: dict[str, Any]) -> tuple[list[tuple], tuple, bool, set]:
        """Splits the arguments of a search call into text words, mode values and paging info.
        Text fields become lowercase word tuples; enum strings are modes; page-like numbers set paging.
        Returns the text list, sorted mode tuple, a flag for first page, and the set of paging keys."""
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
        """Learns whether the search tool requires all query words or any of them to match.
        Compares totals of searches whose words extend or shorten earlier ones in the same context.
        Updates and returns word_kind; only first-page calls count, history is kept in word_seen."""
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
        """Learns how many leading query words the search tool actually uses.
        Compares totals and row ids across searches sharing a prefix and records lower and upper bound.
        Updates lead_words and lead_ctx from the tightest claim; returns the current lead_words."""
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
        """Finds the search argument that carries a continuation token, if the tool has one.
        Looks at the search schema for a free string property named like cursor, bookmark or next.
        Returns that property name, or None when no such argument exists."""
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        for key, pspec in (props if isinstance(props, dict) else {}).items():
            if isinstance(pspec, dict) and pspec.get("type") == "string" and not pspec.get("enum") \
                    and re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(key).lower()):
                return key
        return None

    @staticmethod
    def next_token(obs: Any) -> str | None:
        """Extracts a continuation token from a search observation.
        Scans the top-level keys for a non-empty string under a name like next, cursor or token.
        Returns the string, or None when obs is not a dict or has no such key."""
        if not isinstance(obs, dict):
            return None
        for key, val in obs.items():
            if isinstance(val, str) and val.strip() and re.search(r"next|onward|cursor|bookmark|token|continu|more", str(key).lower()):
                return val
        return None

    def page_arg(self) -> tuple[str, Any, Any, float | None] | None:
        """Describes how the search tool pages through results.
        Prefers a numeric page or offset property with its minimum and maximum, else the token key.
        Returns (key, page-or-offset flag, first value, upper limit), or None if paging is absent."""
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
        """Collects price, title and maker for in-stock listings on the chosen shelves.
        Walks archive records, skips refused or sold-out rows, uses the newest archive price per id.
        Drops entries whose cost times quantity exceeds the spend cap; returns a dict keyed by id."""
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
        """Turns the shelf labels into short search phrases made of their meaningful words.
        Splits each label at filler words, using two-word phrases, head words and any remaining words.
        Returns the phrases in order with duplicates removed."""
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
        """Suggests search words from titles that repeat across the shelf rows.
        Takes words or word pairs of titles, skipping shelf words and maker names, and counts them.
        Keeps those on two or more titles, ordered by cheapest price, then count; returns phrases."""
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
        """Picks the next search call for sweeping a shelf to find the cheapest listing.
        Weighs continuing a listing walk, or new queries from kind words, label words, and a ceiling.
        Skips calls already tried; returns the args and a flag saying it continues a listing, or None."""
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
            return ceil, True
        if ended:
            return None, False
        for pick in [0] + [1 + (turn + shift) % 3 for shift in range(3)]:
            picks = [a for a in sources[pick]() if json.dumps(a, sort_keys=True, default=str) not in tried]
            if picks:
                return picks[0], pick == 0
        return None, False

    def sweep_step(self) -> bool:
        """Runs one extra shelf-reading search when cheapest wins and no property is open.
        Respects sweep read limit, spare moves and time; records the call as tried and dispatches it.
        On the last slice it appends a note listing the cheapest rows; returns True if a call ran."""
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
        """Lists the free text values in a set of search arguments.
        Keeps strings whose schema property has no enum and which are not the paging token key.
        Returns those values in argument order."""
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        props = props if isinstance(props, dict) else {}
        tok = self.token_key()
        return [val for key, val in call_args.items() if isinstance(val, str) and not (props.get(key) or {}).get("enum") and key != tok]

    def book_page(self, call_args: dict[str, Any], rows: list, total: int, obs: Any = None) -> None:
        """Records which result page of a search was read, building a walk record per query.
        Keys the record by the arguments minus the paging value, noting rows, size, total and shelves.
        Also tracks first page ids, full results, tokens, and whether ids come back sorted."""
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
        """Works out the next unread page value for a recorded search walk.
        For tokens it returns an unread token seen earlier; for numbers it steps by page or page size.
        Returns None past the upper limit or the total; otherwise the page value to request next."""
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
        """Counts how many still-needed values a search call appears to target.
        Compares value words of open requirements with the free text words, with prefix matching.
        Considers only leading words when lead_words is known; returns the match count."""
        def near(a: str, b: str) -> bool:
            return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))
        vals = [re.findall(r"[a-z0-9]{3,}", req.partition(":")[2].lower()) for req in self.needs.props_left() if not ref_text(req.partition(":")[2])]
        words = [w for text in self.free_text(call_args) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
        words = words[: self.lead_words] if self.lead_words else words
        return sum(1 for value_words in vals if any(near(w, v) for w in words for v in value_words))

    def row_hits(self, rows: list, words: list[str] | None = None) -> list[str]:
        """Picks result rows worth reading in full and notes pointer listings.
        Skips rows already read, over the spend cap, sold out, or off the target shelves.
        Remembers each hit shelf and may add maker-pointing ids to ref_hits; returns ids of new hits."""
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
        """Gathers words from references and from read listings that point to other sellers.
        Reads reference texts plus non-title string fields of read rows, minus the row own maker words.
        The set is cached by archive size; returns it, tolerating any failure silently."""
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
        """Chooses unread result pages of earlier searches that look most promising.
        Considers walks with a next page on matching shelves and ranks them by value_score then total.
        Returns up to most argument dicts with the paging value set to the next page."""
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
        """Proposes an extra search when the query drops a needed value for a property.
        Compares the leading query words with requirement values and product words using near matches.
        Returns (property name, probe text), or two empty strings if nothing is missing or already run."""
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

    def ran_query(self, text: str) -> bool:
        """Checks whether an equivalent free-text search was already run.
        Compares stemmed leading words of text against each recorded walk, regardless of order.
        Returns True on a match, else False."""
        width = self.lead_words or 3
        mine = sorted(self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[:width])
        for walk in self.__dict__.get("page_book", {}).values():
            ran = sorted(self.stem(w) for w in re.findall(r"[a-z0-9]+", " ".join(self.free_text(walk["args"])).lower())[:width])
            if ran == mine:
                return True
        return False

    def search_template(self) -> tuple[dict[str, Any], str] | None:
        """Finds a past search call to copy as a template for new text queries.
        Goes through recorded walks newest first, skips listing walks, needs exactly one text value.
        Returns the other arguments and the name of the text key, or None if none qualifies."""
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
        """Spots query words that would wrongly narrow a search under all-words matching.
        Flags words that are property names or dropped requirements, and builds a cleaned query.
        Returns an advice note plus the new query, or two empty strings; errors are logged."""
        try:
            if self.word_kind == "any":
                return "", ""
            toks = [t.strip("\"'()[]{},.;!?|").lower() for t in " ".join(raw).split()]
            toks = [t for t in toks if re.search(r"[a-z0-9]", t)]
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
        """Builds search calls joining values of several open properties into one query.
        Needs a search template; skips values with digits or negations and already satisfied shelves.
        Adds the product kind word when width allows; returns at most two calls not yet run."""
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
        """Settles a conflict between the decline audit and an earlier item check.
        Reruns the item check with a note giving both readings and the audit reason.
        Returns the new ruling, clears the stored rulings if it passes, or the old ruling on error."""
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
        """Lists read listings that nothing has ruled out, one per wanted product.
        Excludes failed, audited, refused, sold-out and pointer rows, then takes cheapest per shelf.
        Returns the ids only if the total stays within the spend cap; otherwise an empty list."""
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
        """Builds a note urging a reconsideration when the audit named a possible fit.
        Runs the item check on up to two listed fits and arbitrates conflicts; it runs only once.
        Returns advice text naming the verified items, or an empty string when none apply."""
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
        """Tells whether the store lists search results in a fixed id order.
        Looks at every recorded search walk that carries an id_order flag.
        True only when at least two walks were id ordered and none were not."""
        orders = [w.get("id_order") for w in self.__dict__.get("page_book", {}).values() if "id_order" in w]
        return orders.count(True) >= 2 and False not in orders

    def product_index(self, thing: str) -> int | None:
        """Maps a product name to the index of the wanted product it belongs to.
        Finds the shelf of thing and returns the position of the home shelf set holding it.
        Returns None when there are no itemised needs or no home shelf matches."""
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
        """Parses a leading quantity and unit word from a text fragment.
        Returns the digits and the first word after them as a pair of strings.
        Returns None if the fragment hits the boundary pattern or has no such start."""
        low = str(text).lower()
        if re.search(CH_BOUND, low):
            return None
        found = re.match(r"\s*(\d+)\s+([a-z]+)", low)
        return (found.group(1), found.group(2)) if found else None

    @staticmethod
    def amount_part(text: Any) -> bool:
        """Decides whether a text fragment looks like a measured amount.
        True with a digit plus the boundary pattern, or a number then a word at the start.
        Used to keep amount fragments out of plain word matching."""
        low = str(text).lower()
        return bool(re.search(r"\d", low)) and bool(re.search(CH_BOUND, low) or re.match(r"\s*\d+(?:[.,]\d+)?\s+[a-z]", low))

    def walk_carries(self, args: dict[str, Any], k: int | None, alts: bool = False) -> set[Any]:
        """Finds which unmet requirements a search request already carries in its free text.
        Splits each requirement value on separators, matching amounts and near words to the query.
        Returns requirement indexes, or index and alternative pairs when alts is set."""
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
                    if fig[0] in toks and any(w == fig[1] or near(w, fig[1]) for w in toks):
                        out.add((n, a) if alts else n)
                    continue
                if not self.amount_part(part) and any(near(w, v) for w in words for v in re.findall(r"[a-z0-9]{3,}", part)):
                    out.add((n, a) if alts else n)
        return out

    def walk_score(self, args: dict[str, Any], k: int | None) -> int:
        """Counts how many open requirements a search request covers for product k.
        A thin wrapper around walk_carries that returns only the size of its result set.
        Higher values mean the query already names more of the needed properties."""
        return len(self.walk_carries(args, k))

    def walk_options(self, k: int | None) -> list[tuple[int, str, dict[str, Any]]]:
        """Lists unfinished id-ordered search walks worth paging through for product k.
        Skips walks whose lead words or modes were already covered by a fully read walk.
        Returns (pages left, key, walk) tuples, cheapest first, with nested queries pruned."""
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
                continue
            pages = -(-walk["total"] // max(1, walk["size"])) - len(walk["read"])
            twin = (walk["total"], walk["first"])
            if 0 < pages <= 2 * WALK_PAGES and (twin not in best or pages < best[twin][0]):
                best[twin] = (pages, key, walk)
        opts = list(best.values())
        if self.all_words():
            def nested(p: tuple, o: tuple) -> bool:
                a, b = lead(p[2]), lead(o[2])
                return (a < b and not (b - a) <= values | label and p[0] <= WALK_PAGES) or (b < a and (a - b) <= values | label)
            opts = [o for o in opts if not any(nested(p, o) for p in opts if p is not o)]
        return sorted(opts, key=lambda p: (p[0], -self.walk_score(p[2]["args"], k), p[1]))

    def walk_plan(self, budget: int, only: set | None = None) -> list[tuple[int | None, str, int]]:
        """Builds the paging plan for the cheapest-first request across products.
        Needs a positive budget, search tool, page argument and id-ordered store, else nothing.
        Returns (product, walk key, page count) steps, each capped by the walk limit and budget."""
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
        """Chooses the cheapest known candidate item for product k.
        Prefers cart units, else items that passed this needs version, minus dead or refused ones.
        Returns (id, price) of the lowest priced one, or None if nothing is priced."""
        gone = self.dead_lines()
        cands = [i for i in self.trolley.units if (k is None or self.product_index(i) == k) and i not in gone]
        if not cands:
            cands = [key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == str(self.needs.version)]
            cands = [i for i in cands if i not in self.refused_lines and i not in gone and (k is None or self.product_index(i) == k)]
        priced = [(i, self.archive.price(i)) for i in cands]
        priced = [(i, float(p)) for i, p in priced if is_number(p)]
        return min(priced, key=lambda ip: (ip[1], ip[0])) if priced else None

    def line_query(self, k: int | None) -> dict[str, Any] | None:
        """Builds a follow-up search request aimed at the picked item's product line.
        Finds the maker field of the pick, then the first non-generic title word after it.
        Returns the search template filled with maker and word, or None if unusable or already run."""
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
        """Picks the cheapest unread, in-stock, affordable rows seen in walks for product k.
        Collects ids from the best matching walks and the named keys that sit on home shelves.
        Drops sold out, refused, over-cap, or dearer-than-pick rows and returns up to most ids."""
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
        """Condenses a search result page into a short one-line text summary.
        Takes up to twelve rows with an id and lists price, sold-out flag, title and shelf.
        Returns the summary, or a none marker when no rows were found."""
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
        """Runs one scripted search call and records it in the chat as a tool exchange.
        Appends an assistant proposal, dispatches it in covering mode, swaps the reply for a digest.
        Returns True only if the call added a new item to the archive."""
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
        """Walks extra result pages and product-line searches, then queues cheap rows to open.
        Respects the move floor and wire budget; reads pages per plan, line queries, and ceiling walks.
        Sets the open queue, appends a summary to the last tool reply, logs, returns whether it ran."""
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
        """Collects the item ids that currently fit the needs and were not refused.
        Merges ids that passed the current needs version with audited fits, minus those ruled failing.
        Returns the set used to decide whether a decline is justified."""
        ver = str(self.needs.version)
        live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == ver}
        failed = {k.split("|")[0] for k, v in self.rulings.items() if v and k.split("|")[1:2] == [ver]}
        return (live | (set(self.__dict__.get("audit_fits") or []) - failed)) - self.refused_lines

    def decline_lacking(self) -> list[int | None]:
        """Lists the products that still have no fitting item before a decline.
        Compares shelves of the fitting ids against each product's home shelves.
        Returns product indexes, or a single None entry for the unitemised request."""
        got = {self.shelf_known(i) for i in self.decline_fits()} - {""}
        if not self.needs.items():
            return [] if set(self.shelf_labels()) & got else [None]
        return [k for k, home in enumerate(self.home_shelves()) if home and not home & got]

    def decline_walks(self) -> list[str]:
        """Selects up to two search walks worth paging before declining.
        For each product lacking a fit, takes the best walk whose query states a required value.
        Ignores requirements that refer to other listings and returns the walk keys."""
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
        """Builds per-product lists of unread in-stock row ids sorted by ascending price.
        Reads shelf and stock fields from archive items; filters by home shelf, spend cap, ids.
        An optional below list demands a price under each product's limit."""
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
        """Finds cheaper unread rows when the cheapest fits together exceed the spend cap.
        Only applies to non-cheapest priority; sums each product's lowest fit price times quantity.
        Returns pools below each pick, largest share first, or an empty list on any error."""
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

    def decline_reads(self) -> str:
        """Decides whether to read more pages and open cheaper rows before declining.
        Checks budget, tools and remaining moves, then plans walks and over-budget row pools.
        Stores a decline queue and returns an explanatory note, or an empty string when not needed."""
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
        """Carries out the queued pre-decline reading: pages first, then opening rows.
        Pops the stored queue, reads follow-up pages within limits, and interleaves candidate pools.
        Starts opening the chosen rows and returns True if pages were read or rows queued."""
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
        """Registers reference requirements that point to another seller's listing.
        Scans written requirements for reference text or listed requirements with differ wording.
        Hands each match to ref_add together with the request text, tagged as coming from it."""
        reading = self.needs.reading
        infos = reading.get("reference_targets") if isinstance(reading.get("reference_targets"), list) else []
        infos = [t for t in infos if isinstance(t, dict)]
        listed = {str(r) for r in reading.get("references") or []} if isinstance(reading.get("references"), list) else set()
        for req in list(self.needs.written):
            value = str(req).partition(":")[2]
            if ref_text(value) or (str(req) in listed and re.search(DIFFER_WORDS, value.lower())):
                self.ref_add(str(req), infos, self.request, "request")

    def ref_untracked(self, refs: list[str], tracked: list[dict[str, Any]]) -> list[str]:
        """Filters references that no tracked entry has resolved or chained yet.
        A reference counts as tracked when a matching entry is resolved or has a chain.
        Returns the untracked ones, or all of them if matching fails."""
        try:
            return [r for r in refs if not any(self.ref_match(t, r) and (t.get("status") == "resolved" or t.get("chain")) for t in tracked)]
        except Exception:
            return refs

    def ref_take(self) -> None:
        """Syncs tracked references with the current needs and any newly reported ones.
        Swaps in the resolved value for matching requirements, and adds references queued from replies.
        Appends each new requirement to the reading references list; errors are only logged."""
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
        """Creates and stores a new tracked reference from a requirement string.
        Splits property and tag, finds matching info, builds the target and works out the relation.
        Returns the new record, or None for duplicates, junk keys, or an unusable target."""
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
        span = ref_span(prop, source).lower()
        cut = span.find(str(tgt.get("seller") or "").lower()) if str(tgt.get("seller") or "").strip() else -1
        if rel == "same" and (re.search(r"differ|not the same|exclu|apart|other than|unlike", told)
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
        """Works out which seller and listing a reference points at.
        Combines the hint info with parsing of the nearby request text and the requirement value.
        Returns a dict with seller, kind and trimmed details, or None if no valid seller is found."""
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
        """Checks that a candidate seller name is plausible and appears in the source text.
        Needs three letters, a whole word match in source, and a name that is not just a product kind.
        Product kinds come from the needs and from enum values of the search tool."""
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
        """Builds and caches a map from item id to its merged row fields.
        Walks every archive item, keeping dicts that carry a known id and enough fields.
        Reuses the cache while the item count is unchanged and records the item index of each id."""
        memo = self.__dict__.get("ref_memo")
        if memo and memo[0] == len(self.archive.items):
            return memo[1]
        rows: dict[str, dict[str, Any]] = {}
        seen_at: dict[str, int] = {}
        for n, rec in enumerate(self.archive.items):
            def visit(d: dict[str, Any]) -> None:
                ident = self.row_mark(d)
                if ident in self.archive.ids and (len(d) >= 4 or any(isinstance(v, dict) and len(v) >= 2 for v in d.values())):
                    rows.setdefault(ident, {}).update(d)
                    seen_at[ident] = n
            walk_dicts(rec, visit)
        self.ref_memo = (len(self.archive.items), rows)
        self.ref_seen_at = seen_at
        return rows

    def ref_sellers(self, row: dict[str, Any]) -> set[str]:
        """Extracts the normalised seller names that a row names.
        Reads short string fields whose key looks like a seller key or was learned from data.
        Skips id and note-like fields and returns the set of names."""
        learned = self.seller_keys()
        return {ref_name(v) for k, v in row.items() if isinstance(v, str) and v.strip() and len(v) <= 60
                and (re.search(SELLER_KEYS, str(k).lower()) or str(k) in learned) and "id" not in str(k).lower()
                and not re.search(r"note|text|desc|comment|detail|info|message|section", str(k).lower())} - {""}

    def seller_keys(self) -> set[str]:
        """Learns which row field names hold seller names in this store.
        Uses known reference sellers plus capitalised names in notes to test short string fields.
        Caches by item count and names; returns the set of keys, or an empty set on error."""
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
        """Joins the free-text note fields of a row into one string.
        Skips short values, ids, titles and labels; sorts the rest into notes and descriptions.
        Returns notes, falling back to descriptions only when desc is true."""
        notes, other = [], []
        for k, v in row.items():
            name = str(k).lower()
            if not isinstance(v, str) or len(v.split()) < 3 or name == "id" or name.endswith("id") \
                    or re.search(r"title|heading|headline|name|label|caption|categor|shelf|department|url|image", name):
                continue
            (other if re.search(r"desc|overview|summary|about", name) else notes).append(v)
        return "\n".join(notes or (other if desc else []))

    def ref_parts(self, row: dict[str, Any]) -> tuple[set[str], set[str]]:
        """Splits a row into field-name tokens and value tokens for matching.
        Cleans negated phrases from notes and gathers title, notes and nested detail fields.
        Returns a pair of token sets: key names and values."""
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
        """Tests whether a row displays a detail described in a reference.
        Checks that the detail's words appear in the row's values and that numbers with units agree.
        Returns False on any word miss or numeric mismatch, otherwise True."""
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
        """Reads the value of a property from the nested detail dicts of a row.
        Tries a strict property-name match across all nested fields first, then a looser match.
        Returns the stripped value as a string, or an empty string."""
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
        """Finds referenced ids that sit on shelves outside the wanted products.
        Requires home shelves for every product, then checks each id against the index rows.
        Excludes cart units and returns an empty set on error."""
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
        """Ranks stored listings that may be the one a reference points to.
        Keeps rows whose seller matches, preferring matching kind and most shown details.
        Records the shown detail counts on the reference and returns ids best first."""
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
        """Groups recorded searches by their non-text arguments to find reusable query modes.
        Detects modes whose differing queries return the same first rows, marking them blind.
        Returns the (other args, text key) templates ranked by variety, plus the blind mode set."""
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
        """Chooses a search request shape for looking up referenced sellers.
        Tries the top ranked mode, then past logged actions, then the schema's only free text field.
        Returns (fixed args, text key), or None when no such template can be found."""
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
        """Extracts search words from a reference target's details.
        Drops link words, seller words and store field names; keeps codes, figures and rare words.
        Orders them so last words of multi-word details come first and deduplicates."""
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
        """Gathers field names used inside nested detail dicts across all archive items.
        Normalises camel case and punctuation into lowercase phrases.
        Caches the result by archive length and returns the key set."""
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
        """Proposes search query strings for locating a reference's listing.
        Combines the seller name with the best word and kind, depending on whether it is unindexed.
        Excludes queries already tried and returns the remaining candidates."""
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
        """Finds the next result page to fetch for a reference's seller search.
        Gives up after three pages; looks backwards for a walk whose query names the seller.
        Returns that walk's arguments with the next page set, or None."""
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
        """Decides whether the step budget allows more work on open references.
        Stops when all open ones are differ and blind, or steps, wire spare, or moves run out.
        Returns True if moves exceed the reference floor or a late hop is allowed."""
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
        """Computes the number of moves that must stay in reserve for references.
        Takes the larger of a first-order-based figure plus ten and the order due plus seven.
        Returns the integer floor used by the reference room check."""
        return max(max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, self.step_cap // 4 + 1)) + 10, self.order_due() + 7)

    def ref_product(self, ref: dict[str, Any]) -> int | None:
        """Maps a reference to the product index it belongs to.
        Only applies when needs are itemised and the reference has a tag.
        Returns the product index via product_of, otherwise None."""
        return self.product_of(f"x ({ref['tag']})") if self.needs.items() and ref.get("tag") else None

    def ref_waits(self, floor: int | None = None) -> bool:
        """Decides if differ references should hold back while other products lack fits.
        Applies only when every open reference is a differ one with a chain and moves exceed the floor.
        Logs the lacking products once per change and returns True when there are enough to wait on."""
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
        """Allows one late extra hop for a reference that has followed hops already.
        Needs moves beyond order due and a complete target; marks the start step on first use.
        Returns True while within three steps of that start, else False."""
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
        """Checks that products other than the reference's own already have verified fits.
        Returns True for unitemised needs; otherwise lists products without a verified fit.
        True if only the reference's own product (or at most one) lacks one."""
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
        return (self.__dict__.get("ref_calls", 0) < REF_CALLS and self.wire.spare() > CLOSING + 40 and not self.wire.spent_out()
                and self.wire.spend < 0.8 * SPEND_CAP * self.wire.__dict__.get("scale", 1.0))

    def ref_match(self, ref: dict[str, Any], req: Any) -> bool:
        """Tests whether a requirement string corresponds to a tracked reference.
        Matches exact req or current text, or a reference-style value with the same property.
        Also requires the parenthesised kind tags to overlap when both exist."""
        head, _, value = str(req).partition(":")
        if str(req) in (ref["req"], ref.get("now")):
            return True
        pointing = ref_text(value) or (ref.get("relation") == "differ" and re.search(DIFFER_WORDS, value.lower()))
        if not pointing or not same_prop(re.split(r"\s*\(", head)[0], ref["prop"]):
            return False
        tags = re.findall(r"\(([^)]*)\)", head)
        return not tags or not ref["tag"] or bool(kind_words(tags[-1]) & kind_words(ref["tag"]))

    def ref_live(self, ref: dict[str, Any]) -> bool:
        """Tells whether a reference still matters to the current needs.
        True when a remaining requirement matches it or names its property.
        Otherwise True only if the property was not let go."""
        left = self.needs.props_left()
        if any(self.ref_match(ref, r) or same_prop(re.split(r"\s*\(", str(r).partition(":")[0])[0], ref["prop"]) for r in left):
            return True
        return not any(same_prop(re.split(r"\s*\(", str(r).partition(":")[0])[0], ref["prop"]) for r in self.needs.let_go)

    def ref_step(self) -> bool:
        """Runs one reference-resolution turn guarded against errors.
        Returns False when there are no references or the wire is done or broken.
        On an exception it logs it, marks open references stuck, and returns False."""
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
        """Opens a queue of candidate listings to check against open references.
        Uses a batch tool when available, validates it on first use, and posts the replies.
        Updates step counts, id sets and judged lists, and returns True if the tick advanced."""
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
        """Runs a search for a reference's target listing and records what came back.
        Calls walk_call with a note naming the reference, then counts the rows in the new result.
        Marks the reference unindexed if rows lack its seller; returns True if the tick advanced."""
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
        """Judges the read candidate listings for one reference and acts on the best match.
        Ranks candidates by detail coverage, may read notes first, resolves one exact plain match.
        Otherwise asks for a judgement, applies it, or stops the reference; returns whether it acted."""
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
            loose |= {d for d in tgt["details"] if not any(shows(i, d) for i in seller_read)}

        def words_ok(i: str) -> bool:
            return all(shows(i, d) for d in tgt["details"] if d not in loose)
        full = {i: sum(1 for d in tgt["details"] if set(ref_toks(d)) and set(ref_toks(d)) <= parts[i][0] | parts[i][1]) for i in cands}
        ranked = sorted(cands, key=lambda i: (not words_ok(i), -full[i], cands.index(i)))
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
        """Decides whether a listing's own field value can be trusted without asking.
        Rejects it if notes name the property, repeat the value, or mention another seller.
        Returns True only when the value is present and notes show no redirect."""
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
        """Picks a tool that can show a listing's notes.
        Prefers the known notes tool, then a lookup tool with extra text, then view or stock tools.
        Returns the tool name, or None if none fits or notes already come with details."""
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
        """Selects up to two ids whose notes have not yet been read for a reference.
        Needs a notes tool; skips ids already seen with it or already tried for this reference.
        Returns the list of ids still lacking notes."""
        tool = self.ref_notes_tool()
        if not tool or not ids:
            return []
        seen = set((self.__dict__.get("read_with") or {}).get(tool) or set())
        tried = ref.setdefault("noted", set())
        return [i for i in dict.fromkeys(ids) if i not in seen and i not in tried][:2]

    def ref_notes_read(self, ref: dict[str, Any], need: list[str]) -> bool:
        """Reads the notes of the given listings using the notes tool.
        Marks the ids as tried, opens them with an explanatory message, and posts the replies.
        Adds to ref steps and ids, logs it, and returns True if the tick advanced."""
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
        """Collects ids that act as pointer listings for differ references.
        Looks at live differ references that are open, stuck or resolved, and takes their chain ids.
        Keeps only ids that belong to the reference's own product."""
        out: set[str] = set()
        for ref in self.__dict__.get("refs") or []:
            if ref.get("relation") == "differ" and ref.get("status") in ("open", "stuck", "resolved") and self.ref_live(ref):
                out.update(i for i in ref.get("chain") or [] if self.ref_own_product(ref, i))
        return out

    def ref_own_product(self, ref: dict[str, Any], ident: str) -> bool:
        """Checks that an id belongs to the same product as a reference.
        Unitemised needs always pass; otherwise compares the product index of its shelf with the tag.
        Unknown product or missing tag also counts as a match."""
        if not self.needs.items():
            return True
        shelf = self.shelf_known(ident)
        k = self.product_of(f"x ({shelf})") if shelf else None
        return k is None or not ref.get("tag") or k == self.product_of(f"x ({ref['tag']})")

    def ref_pointer_note(self, thing: str) -> str:
        """Builds the explanation that an id is the listing a reference points at.
        Finds a live differ reference whose chain holds the id and the same product.
        Returns text saying it cannot meet the requirement, or an empty string."""
        for ref in self.__dict__.get("refs") or []:
            if ref.get("relation") == "differ" and thing in (ref.get("chain") or []) and self.ref_live(ref) and self.ref_own_product(ref, thing):
                what = f" ({ref['value']})" if ref.get("status") == "resolved" and ref.get("value") else ""
                return (f"{thing} is the listing the shopper's reference for {ref['head']} points at, or one its notes lead "
                        f"through, so its {ref['prop']} is the very value{what} the shopper's {ref['prop']} must differ from: it cannot "
                        "meet that requirement")
        return ""

    def ref_excluded(self, fields: dict[str, str], k: int | None) -> bool:
        """Tells whether given fields hold the value a differ reference must avoid.
        Uses resolved live differ references for matching products, comparing normalised tokens.
        Returns True on a property match whose value equals the reference value or companions."""
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
        """Finds notes in an item that may redirect a requirement to another listing.
        Scans note sentences after the first mention of each needed property for other sellers' names.
        Skips negated mentions; returns (requirement, property, sentence) tuples, empty on error."""
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
        """Builds a warning when a chosen item's own notes may redirect a requirement.
        Skips search, cart, message and finishing calls; checks passed or carted ids in the call.
        Tells once per id and version, drops the pass, logs it, and returns the joined text."""
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

    def ref_phrase(self, line: str) -> tuple[str, int, int, dict[str, Any]] | None:
        """Finds the span of a shopper line that points at another seller's listing.
        Parses the line, locates the seller name, then widens left to the cue word and right to a stop.
        Returns (phrase, start, end, parsed) or None when no seller or cue is found."""
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
        phrase = text[start:end].strip(" ,;")
        if not phrase:
            return None
        return phrase, start, end, parsed

    def ref_line_words(self, qtext: str | None, outside: str) -> list[str]:
        """Builds candidate property names from the words of a question text.
        Slides 3, 2 and 1 word windows, skipping link words, short words and product kind words.
        Keeps windows that prop_named finds in the outside text and returns them as a list."""
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
        """Rewrites the reader answer so pointer phrases become explicit reference needs.
        For each pointer phrase it picks the property, product tag and relation, then edits needs,
        requirements_now, dropped and references, and returns the lines with the phrase masked."""
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
        """Static check on whether a text mentions the words of a property name.
        Compares lowercase tokens to property words of 3+ letters, allowing a shared 4-letter stem.
        Returns True when the property has no usable words or any token matches."""
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
        """Sends one hop question to the checker about the candidate records in pick.
        Builds the prompt from the property, description, record cards and an optional failure reason.
        Counts the call, logs the exchange and returns the parsed answer, or None when no calls remain."""
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
        """Acts on a hop answer for a reference: judges, follows, resolves or gives up.
        Rejects empty or partial answers, validates the target, and retries once with the problem.
        Updates the ref judged set and chain, then calls ref_go, ref_resolve or ref_stop."""
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
        """Checks a hop answer against the record of its chosen target.
        Verifies the details shown, that no better candidate exists, and go_to or value is in notes.
        Returns (where, problem, next_target, value, pairs); empty problem means it is usable."""
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
        if not value or not set(ref_toks(value)) or not set(ref_toks(value)) <= vals | keys:
            return "answer", f"the value '{value}' is not written in {target}'s record.", None, "", []
        raw = answer.get("goes_with") if isinstance(answer.get("goes_with"), list) else []
        pairs = [str(p).strip() for p in raw if isinstance(p, (str, int, float)) and not isinstance(p, bool) and str(p).strip()]
        pairs = [p for p in dict.fromkeys(pairs) if set(ref_toks(p)) and set(ref_toks(p)) <= ntoks and set(ref_toks(p)) != set(ref_toks(value))][:3]
        return "", "", None, value, pairs

    def ref_go(self, ref: dict[str, Any], target: str, go: dict[str, Any]) -> None:
        """Moves a reference to the next listing named in the notes of the current one.
        Stops on a loop or after the hop limit; otherwise resets the search state for the new target.
        Logs the hop and tells the model the code is following the notes onward."""
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
        """Turns a resolved reference into requirement text according to its relation.
        Handles within-tolerance ranges, at most or at least bounds, differ lists, alternatives.
        Returns the formatted value string, with paired values joined by or, or negated for differ."""
        value, pairs = str(ref["value"]), list(ref.get("goes_with") or [])
        if ref["relation"] == "within" and is_number(ref.get("tol")):
            hit = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)", value.lower().replace(",", ""))
            if hit:
                base, tol = float(hit.group(1)), float(ref["tol"])
                unit = hit.group(2) or str(ref.get("unit") or "")
                return f"{round(base - tol, 6):g} to {round(base + tol, 6):g} {unit}".strip() + f" (within {tol:g} of {value})"
        if ref["relation"] in ("at most", "at least"):
            return f"{ref['relation']} {value}"
        if ref["relation"] == "differ":
            return " and ".join(f"not {v}" for v in [value] + [p for p in pairs if p.lower() != value.lower()])
        return " or ".join([value] + [p for p in pairs if p.lower() != value.lower()])

    def ref_resolve(self, ref: dict[str, Any], target: str, value: str, pairs: list[str]) -> None:
        """Marks a reference as resolved with the value read on the target listing.
        Builds the new requirement, swaps it in via ref_swap, bumps the needs version and logs it.
        Tells the model what was found, noting chain, paired values and the must-differ case."""
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
        pair = (f"; its notes say {ref['value']} goes together with {' and '.join(pairs)}, so each of them meets it" if pairs else "")
        if ref["relation"] == "differ":
            pair = (f"; its notes pair {' and '.join(pairs)} with it" if pairs else "") + (
                ". The shopper's value MUST DIFFER from that one: any other value meets it, that value does not")
        self.ref_tell(f"RESOLVED: {ref['head']} is given by another listing, and {target} shows {ref['value']}{via}{pair}. The "
                      f"requirement is now: {new_req}.")

    def ref_swap(self, ref: dict[str, Any], new_req: str) -> None:
        """Replaces the old pointer requirement with the resolved one across the needs state.
        Rewrites the written, uncovered and now-requirements lists and per-product requirement rows.
        Matches lines by reference match or seller and property; adds the text if none matched."""
        nd = self.needs
        seller = ref_name(ref["first"]["seller"])

        def mine(req: Any) -> bool:
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
        """Marks an open reference as stuck, recording the reason.
        Does nothing when the reference is no longer open; otherwise logs it with the chain so far.
        Tells the model the value stays unknown and that a listing must not be rejected over it."""
        if ref["status"] != "open":
            return
        ref["status"], ref["why"] = "stuck", why
        self.log.append({"kind": "ref_stuck", "req": ref["req"][:200], "why": why[:300], "chain": list(ref["chain"])})
        self.ref_tell(f"NOT RESOLVED: {ref['head']} ({why}). Its value stays unknown: if a listing's notes you read show where it "
                      "ends, use that value; otherwise never decline or reject a listing over it.")

    def ref_tell(self, text: str) -> None:
        """Delivers a reference status message to the model.
        Appends it to the last tool or user chat message when one exists after the opening turns.
        Otherwise queues the text in ref_told for the next reference line."""
        if self.chat and self.chat[-1].get("role") in ("tool", "user") and isinstance(self.chat[-1].get("content"), str) and len(self.chat) > 2:
            self.chat[-1]["content"] += "\n[reference] " + text
        else:
            self.ref_told = list(self.__dict__.get("ref_told") or []) + [text]

    def ref_line(self) -> str:
        """Builds the one-line reference summary added to the next tool message.
        Lists resolved, open and stuck references with their status, then drains any queued messages.
        Returns the text, or an empty string if nothing applies or an error occurs."""
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
        """Builds the longer requirements-by-reference note for a checker prompt.
        Describes each resolved reference and its chain, and flags open or stuck ones as unresolved.
        Appends up to two archive records of the last chain listings; empty string on failure."""
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

    def ceil_learn(self, call_args: dict[str, Any], rows: list, total: int, obs: Any) -> None:
        """Learns from a search result how paging depth and a price filter behave.
        Records the total per call shape, detects a reachable-page cap, finds one numeric bound.
        Sets ceil_mode when rows stay under the bound with a shelf filter, or marks that shape as bad."""
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
        """Returns the learned price-filter specification when it is still trusted.
        Reads ceil_mode and discards it if its signature was marked bad.
        Returns the spec dict or None."""
        spec = self.__dict__.get("ceil_mode")
        return spec if spec and spec["sig"] not in self.__dict__.get("ceil_bad", set()) else None

    def cg_listing(self, args: dict[str, Any]) -> bool:
        """Tells whether a search call is a category listing under the learned price filter.
        Compares the call to the stored mode values, or checks that the bound argument is present.
        Returns False when no spec is available or args is not a dict."""
        spec = self.ceil_spec()
        if not spec or not isinstance(args, dict):
            return False
        if spec["mode"]:
            return all(args.get(k) == v for k, v in spec["mode"].items())
        return spec["bound"] in args

    def page_size(self) -> int:
        return max([w.get("size", 0) for w in self.__dict__.get("page_book", {}).values()] or [0])

    def visible_depth(self) -> int | None:
        """Estimates how many result rows can be reached by paging in the current search.
        Combines page size with the first, current and last page numbers, or an offset style position.
        Falls back to the stored page cap; returns an int or None when unknown."""
        found = self.page_arg()
        size = self.page_size()
        if found and size > 0 and found[1] != "token" and found[3] is not None:
            return int((found[3] - found[2] + 1) * size) if found[1] else int(found[3] - found[2] + size)
        return self.__dict__.get("page_cap")

    def raw_label(self, labels: set[str]) -> str:
        """Finds how the store itself spells the category name for a set of lowercase labels.
        Walks archived records with a known id, tallying category field values matching a label.
        Returns the most frequent original spelling, or an empty string when none matches."""
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
        """Computes the largest price strictly under a given price.
        Uses whole numbers when the filter is integer typed or the price is whole, else minus a cent.
        Returns the bound, or None for a non-number or a result that is not positive."""
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
        """Decides whether a price-ceiling walk can run for product k.
        Requires cheapest priority, a trusted spec, a paging argument, a category label, no past walk.
        Returns False for a multi-product request with k None; otherwise a bool."""
        if self.needs.items() and k is None:
            return False
        return (self.needs.priority() == "cheapest" and k not in self.__dict__.get("ceil_walked", set()) and self.ceil_spec() is not None
                and self.page_arg() is not None and bool(self.raw_label(self.walk_home(k))))

    def ceil_aim(self, k: int | None) -> float | None:
        """Picks the price that a ceiling walk should go under for product k.
        Takes the lowest of the pick price and any stored aim, or the cheapest read in-stock row.
        Falls back to rows in the home shelves when a bound need exists; returns a float or None."""
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
        """Tells whether product k has a remaining requirement that is a numeric limit.
        Scans unmet properties for that product whose value has a digit and matches the bound pattern.
        Returns False when the request has no product rows or k is None."""
        if not self.needs.items() or k is None:
            return False
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            if self.product_of(head) == k and re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
                return True
        return False

    def home_rows(self, home: set[str]) -> set[str]:
        """Collects ids of archived rows whose category is among the given home shelves.
        Walks every archived record and keeps rows with a known id on a matching shelf.
        Returns the matching ids as a set of strings."""
        out: set[str] = set()

        def visit(d: dict[str, Any]) -> None:
            ident = self.row_mark(d)
            if ident in self.archive.ids and self.shelf_of(d) in home:
                out.add(ident)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return out

    def ceil_clash(self, home: set[str], bound: float) -> bool:
        """Detects a contradiction in the price filter for a bound.
        Looks for an archived in-stock row on the home shelves priced at or under the bound.
        Returns True when such a row exists, meaning an empty filter result cannot be trusted."""
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
        """Lists unread candidate rows from ids, cheapest first.
        Skips read, refused, sold-out and unpriced rows, and ones over the cap or not below limit.
        Returns a sorted list of (price, id) pairs."""
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
        """Tells whether the store search is known to require every query word.
        True when the word kind is all, or lead words are set and the kind is not any.
        Returns a bool used to decide how queries and value searches are built."""
        return self.word_kind == "all" or (bool(self.lead_words) and self.word_kind != "any")

    def valued_ids(self, k: int | None) -> set[str]:
        """Collects row ids shown by value-led searches that sit on the product's home shelves.
        Only applies when every word is required; uses id-ordered searches, not category listings.
        Returns a set of ids, empty otherwise."""
        if not self.all_words():
            return set()
        home = self.walk_home(k)
        out: set[str] = set()
        for walk in self.__dict__.get("page_book", {}).values():
            if walk.get("id_order") is True and not self.cg_listing(walk["args"]) and self.walk_carries(walk["args"], k):
                out |= {i for i, s in (walk.get("row_shelf") or {}).items() if s in home}
        return out

    def valued_complete(self, k: int | None) -> bool:
        """Tells whether a value search has already shown every matching listing for product k.
        Needs a fully paged id-order search carrying a single-valued requirement on the home shelves.
        Returns True or False; any error counts as False."""
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
        """Checks whether an archived record contains the remaining requirement values for product k.
        Splits each unmet value into figures with units and stemmed words, then scans the record text.
        Returns True if a figure matches or all words of one alternative are present."""
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
        """Computes how much step budget is left for value-search work.
        Reserves moves for planned shopper questions and the pending order.
        Returns (questions_left, spare_moves)."""
        left = max(0, self.ask_limit() - self.msgs_sent) if self.asking and self.role("message") else 0
        return left, self.moves_left() - self.order_due() - 2 * left

    def vl_values(self, k: int | None) -> list[list[list[str]]]:
        """Extracts search-ready word alternatives from the code-read requirement values of product k.
        Drops reference text, exclusions and bare amounts, and strips link words and property name.
        Returns a list per requirement of alternative word lists."""
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
                words = [w for w in re.findall(r"\d+(?:[.,]\d+)+|[a-z0-9]+", piece)
                         if (len(w) >= 2 or (fig and w == fig[0])) and not re.search(r"[.,]", w) and w not in LINK_WORDS
                         and (w not in name or (fig and w == fig[1]))]
                if words and words not in alts:
                    alts.append(words)
            if alts and alts not in out:
                out.append(alts)
        return out

    def vl_words(self, k: int | None, label: str) -> list[str]:
        """Derives the category words to add to a value query from a label.
        If the store does not index the label, keeps only words common to the home shelf titles.
        Falls back to the last word of the product name; returns a list of words."""
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
        """Builds value query strings for product k within a length cap.
        Combines value words with category words, trims to lead words and cap_len, with alternatives.
        Returns (combined_queries, single_value_queries) as de-duplicated lists."""
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
        """Picks the next query form not yet searched.
        Checks both lists in order using the totals already seen for the text key and mode.
        Returns a query, an empty string when done or a match exists, or None if nothing is available."""
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
        """Extracts the mode arguments of a search call.
        Keeps string arguments whose schema property is an enum of at most four values.
        Returns a sorted tuple of key and value pairs."""
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        props = props if isinstance(props, dict) else {}
        return tuple(sorted((key, val) for key, val in args.items() if isinstance(val, str) and isinstance(props.get(key), dict)
                            and isinstance(props[key].get("enum"), list) and len(props[key]["enum"]) <= 4))

    def vl_set(self, text: str) -> frozenset:
        """Normalizes query text to a comparable set of stems.
        Tokenizes lowercase text, limited to the store's lead word count if set, and stems each token.
        Returns a frozenset used to compare whether two queries are alike."""
        return frozenset(self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[: self.lead_words or None])

    def vl_seen(self, text: str, tkey: str, mode: dict[str, Any]) -> int | None:
        """Looks up the total from an earlier search with the same words and mode.
        Scans recorded call shapes whose text argument has the same stem set and matching mode values.
        Returns the stored total as an int, or None when no such search was made."""
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
        """Learns whether the store indexes category names in its keyword search.
        Checks the latest archived records for rows whose label words appear in no other field.
        Sets label_indexed True after at least two such rows; otherwise leaves it undecided."""
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
        """Builds the list of unread rows to open after a value search.
        Uses cheapest unread rows under the pick, limited by batch size, open budget and moves.
        Updates the per-product open counter when cheapest is asked; returns the id list."""
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
        """Starts an open pass over the queued rows.
        Clears stale rank checks for the home shelves, resets order vetting, sets the open queue.
        Marks the purpose as a walk with a floor of moves, then calls open_named."""
        self.rank_checked = {mark for mark in self.rank_checked if self.shelf_known(mark.split("|")[0]) not in homes}
        self.order_vetted = False
        self.open_queue = queue
        self.open_recheck = True
        self.open_say = say
        self.open_purpose, self.open_floor = "walk", floor
        self.open_named()

    def vl_search(self) -> bool:
        """Runs one category-name keyword search for a product's required values.
        Picks a text argument from past queries, validates args, runs the call and annotates the reply.
        Queues cheap rows to open when every match is visible; returns True if it acted."""
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
        """Opens the cheapest unread rows deferred while several products waited for value searches.
        Splits the move budget across waiting products and interleaves their pools into one queue.
        Launches the open pass, adds a summary to the last tool message and returns True if it ran."""
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
        """Produces a short progress line for multi-product cheapest requests.
        Per product reports value pages read and whether a pick exists, naming the next one lacking it.
        Returns an empty string for single product requests, other priorities or on error."""
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
        """Decides whether to read more pages of an id-ordered value search and does so.
        Guards on references, orders, budget, role; tries label searches first, then ranks candidates.
        Reads the best via read_value_search, flushes deferred opens, or returns False; logs errors."""
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
        """Reads further pages of one stored value search in order.
        Advances through next pages within move and spare limits, counting pages used per product.
        Queues or defers cheap rows to open, appends a summary to the last tool message; returns bool."""
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
        """Lists a category under a price filter, narrowing the bound until cheaper rows fit the budget.
        Bisects between bounds, reads pages when the total is reachable, flags a misbehaving filter.
        Returns (report text, ranked ids to open) or None when it cannot run."""
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
        """Chooses the next price-filter call for a request without product rows.
        Bounds just below the cheapest known shelf row, skipping tried calls and following next pages.
        Returns (call args or None, flag that the filter result is complete and empty)."""
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
        """Holds back a details call for an id that lacks the searched value words.
        Applies when cheapest is asked and all words are required; compares with unread cheaper rows.
        Returns a not-run message naming those rows, once per product and needs version, else empty."""
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
        """Builds a fallback search call from a distinctive word in a record title.
        Skips link words, shelf words, requirement words and product names; takes up to two words.
        Returns the template call with that text, or None if no word exists or the query already ran."""
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
        """Re-runs a price ceiling walk after the row it was aimed under became unusable.
        Finds an aimed product whose row is dead or refused and whose new pick costs more than the aim.
        Clears the old aim, logs it and returns the repair walk result, or None if none applies."""
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
        """Chooses the next scheduled walk action while cheapest is asked.
        Prefers queued decline reads and walks, then re-arming, then ceilings, seeds or page plans.
        Returns whether a step ran; any error is logged and returns False."""
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
        """Builds trial arguments for a search with a category filter and a price bound.
        Uses the search tool schema and description to pick category field, bound field and modes.
        Returns arguments validated by ArgCheck, or None if a required part cannot be filled."""
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
        """Probes once whether the store's price filter holds by running a seed call.
        Only runs without a learned spec, using a product's pick and category label.
        Logs the outcome and adds a status line when nothing was learned; returns True if a call ran."""
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
        """Tells whether some product of a multi-product request is still not located.
        True if a product's home shelves are unknown or unread, or live passes exist yet unverified.
        Returns False for single product requests and on any error."""
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
        """Pauses an order or cart call so unread cheaper pages can be walked first.
        Works out affected products, a move budget, ceiling targets and a page plan, then queues them.
        Returns a not-run message for the model, or an empty string when no hold applies."""
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
        """Writes advice notes after a search result and records its pages.
        Books the page, learns the price filter, tracks hits, flags paging limits and ignored filters.
        Returns the joined remarks, or an empty string when the result has no rows."""
        if not isinstance(obs, dict):
            return ""

        def rows_of(piece: dict[str, Any]) -> list | None:
            lists = [val for val in piece.values() if isinstance(val, list) and all(isinstance(r, dict) for r in val)]
            return next((val for val in lists if val), lists[0] if lists else None)
        rows = rows_of(obs)
        if rows is None:
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
        """(refusal text or None, tool name, checked args) for one proposed call. v2.5 (V1a): an actor's message refused for
        ANY reason while the kept message can still go with the order (kept_open) is told so: the generic too-few-steps
        refusal becomes v2.4's kept-message text (kept_text), any other refusal not already saying so gets KEPT_LINE."""
        refusal, call_name, call_args = self.screen_base(proposal, permitted, scripted, batch, queued_messages)
        if refusal and not scripted and call_name and call_name == self.role("message") and self.kept_open():
            if refusal.startswith("Not run (no step used): too few steps are left for this;"):
                refusal = self.kept_text()
            elif "kept for the basket confirmation" not in refusal and KEPT_LINE not in refusal:
                refusal += " " + KEPT_LINE
        return refusal, call_name, call_args

    def screen_base(self, proposal: dict[str, Any], permitted: set[str], scripted: bool, batch: int, queued_messages: int) -> tuple[str | None, str, dict[str, Any]]:
        """(refusal text or None, tool name, checked args) for one proposed call (v2.4's screen, unchanged)."""
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
        if not scripted and call_name == self.role("message") and self.confirm_due() \
                and self.msgs_sent + queued_messages >= self.question_limit():
            # (v2: the actor's message would take the one kept for the basket confirmation - held, no step used)
            first = not self.__dict__.get("keep_told")
            self.keep_told = True
            return ("Not run (no step used): the last message is kept for the basket confirmation" + (
                ": when you call the order tool with the final basket in the cart, the code shows the shopper each line and the "
                "total and asks for a yes before the order goes. Do not message the shopper now: finish the cart, re-read stock "
                "and price, then call the order tool." if first else ", which the code sends when you call the order tool.")), \
                call_name, call_args
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
        if held:
            return held, call_name, call_args
        if call_name == self.role("cart"):
            refusal = self.screen_cart(call_args, sig)
            if refusal:
                return refusal, call_name, call_args
        return None, call_name, call_args

    def screen_cart(self, call_args: dict[str, Any], sig: str) -> str:
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

    def line_title(self, thing: str) -> str:
        """v2: the newest title seen on a record that carries the line's id."""
        found: list[str] = []

        def visit(d: dict[str, Any]) -> None:
            if any(isinstance(val, str) and val == thing and "id" in str(key).lower() for key, val in d.items()):
                title = self.title_of(d)
                if title.strip():
                    found.append(title.strip())
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return found[-1] if found else ""

    def basket_text(self, basket: dict[str, float] | None = None) -> str:
        """v2: the confirmation message, written by the code and kept short (one question): each cart line (quantity x title,
        else id; unit price as last read and the line total), the basket total against the budget, and the request for a yes.
        (v2.2 F3: the lines of `basket`, the store's cart as read, else the cart as counted.)"""
        cur = str(self.needs.reading.get("currency") or "").strip()
        parts, total, priced = [], 0.0, True
        lines = basket if basket is not None else {t: self.trolley.held(t) for t in self.trolley.units}
        unsure = set(self.trolley.__dict__.get("unsure") or ()) if self.__dict__.get("store_cart") is None else set()
        for thing, qty in [(t, q) for t, q in lines.items() if q > 0]:
            shown = int(qty) if float(qty).is_integer() else qty
            name = self.line_title(thing)[:50] or thing
            if thing in unsure:
                name += " (my count: the cart view did not show this line in full)"      # (v2.4.2 Q3)
            unit = self.archive.price(thing)
            if unit is None:
                priced = False
                parts.append(f"{shown} x {name}")
            else:
                total += unit * qty
                parts.append(f"{shown} x {name} {unit:.2f}" if qty == 1 else f"{shown} x {name} at {unit:.2f} = {unit * qty:.2f}")
        cap = self.needs.spend_cap
        tail = f"total {total:.2f}" + (f" {cur}" if cur else "") + ("" if priced else " for the priced lines")
        if is_number(cap):
            tail += (", within" if total <= float(cap) + 1e-9 else ", over") + f" your budget of {float(cap):g}"
        return "Your basket: " + "; ".join(parts) + f", {tail}. Shall I place the order? Please reply yes."

    def confirm_basket(self, read: bool = True) -> str:
        """v2: sends the basket confirmation when it is due - (v2.1) owed (the cart is not the basket the shopper was last
        shown) and not blocked (confirm_block: a message, two steps and CONFIRM_SEND seconds left) - as a scripted message
        (no model call writes it); its reply is read like any shopper line (the listener) when `read` and CONFIRM_READ
        seconds allow, else it goes unread and counts as approval. Returns the note for the actor, '' when nothing was
        delivered. A reply that brings no change and no new need counts as approval (live shoppers acknowledge rather than
        say yes) and the order may go; one that changes or restates a need makes the pre-order checks run again before any
        order. (v2.1 E1: it counts as sent only once delivered - the message count rose and the answer holds no error; a
        failed send keeps the kept message.) (v2.2 F1: a failed send goes again at once while the guards allow, CONFIRM_FAILS
        failed sends per episode; F2: the reply counts as read only when the listener ran and answered, its floor checked
        when it would start; F3: the basket shown is the store's cart, read after the newest cart call - store_read.)"""
        for _ in range(CONFIRM_FAILS):
            if not self.confirm_owed() or self.confirm_block(viewing=True):
                return ""
            basket = self.cart_truth()
            proposal = self.message_call(self.basket_text(basket), f"confirm_{self.wire.tick}")
            if proposal and self.__dict__.get("store_cart") is None:
                self.store_read()      # (v2.2 F3: only for a message that can be built)
                basket = self.cart_truth()
                proposal = self.message_call(self.basket_text(basket), f"confirm_{self.wire.tick}")
            if not self.confirm_owed() or self.confirm_block():
                return ""
            self.asking = False
            if not proposal:
                # (v2.2 F1: built again at once, within the bound; not counted as a failed send and no block - the next
                # order call builds it again)
                self.confirm_why = "no message call"
                self.log.append({"kind": "confirm", "sent": False, "why": "no message call"})
                continue
            version, sent_before, mark = self.needs.version, self.msgs_sent, len(self.log)
            self.needs.read_floor = CONFIRM_READ if read else float("inf")      # (v2.2 F2: hear checks it when the listener would start)
            self.needs.read_miss = "no shopper line in the reply"
            # (v2.2 F2: the order's time, the actor's turn; v2.4 J3: the confirmation's calls end with CONFIRM_KEEP + CONFIRM_SLACK
            # left and may take CONFIRM_LEAST seconds - the order keeps CONFIRM_KEEP after up to CONFIRM_SLACK of overhead)
            self.wire.world_end, self.wire.model_end = CONFIRM_KEEP + CONFIRM_SLACK, LAST_CALL
            self.wire.world_least = CONFIRM_LEAST
            try:
                responses = self.dispatch([proposal], scripted=True)
                env = next((e for e in reversed(self.log[mark:]) if e.get("kind") == "env"), {})
                posted = bool(env) and not isinstance(env.get("reply"), dict) and bool(self.wire.__dict__.get("posted"))
                for _ in range(CONFIRM_FAILS - 1 if posted else 0):
                    # (v2.4 J2: the request went out and no answer came back - it may have reached the shopper. The same step
                    # with the same actions goes again while the time allows (Wire.act `again`): the world answers it from its
                    # record if it ran it - an answered confirmation, counted once and read as any - or runs it once if it never
                    # did. Never a new step: that would be a second message. Not a failed send: confirm_fails stays.)
                    if isinstance(env.get("reply"), dict) or self.wire.done or self.wire.broke:
                        break
                    self.wire.again = True
                    try:
                        responses = self.send_chunk([(proposal["id"], act["name"], act["args"]) for act in env.get("actions") or []])
                    finally:
                        self.wire.again = False
                    env = next((e for e in reversed(self.log[mark:]) if e.get("kind") == "env"), {})
                    if not self.wire.__dict__.get("posted"):
                        break
            finally:
                self.needs.read_floor, self.wire.world_end, self.wire.model_end, self.wire.world_least = CLOSING, 0.0, 0.0, 3
            heard = (responses[0][1] if responses else "").strip()
            env = next((e for e in reversed(self.log[mark:]) if e.get("kind") == "env"), {})
            resp = env.get("reply") if isinstance(env.get("reply"), dict) else {}
            obs_list = resp.get("observations") if isinstance(resp.get("observations"), list) else [resp.get("observation")]
            obs = obs_list[0] if obs_list else None
            if env and not resp and posted:
                # (v2.3 H2: the request went out and no answer came back, its replays included (v2.4 J2) - it may have reached
                # the shopper. Never sent at a new step (a second message); it counts as spent, so the cap is never passed; the
                # order may go, logged unconfirmed with this why)
                self.msgs_sent = sent_before + 1
                self.confirm_unknown = basket
                self.confirm_why = "delivery_unknown: " + (heard[:120] or "no answer")
                self.log.append({"kind": "confirm", "sent": False, "delivery_unknown": True, "basket": basket, "why": self.confirm_why})
                return ""
            if self.msgs_sent == sent_before or self.code_fault(obs):
                # (v2.1 E1: not delivered - the message is not counted and stays kept; v2.2 F1: sent again at once; v2.3 H2: only
                # after an answer that says so - an error in it, also inside a wrapper - or when no request went out)
                self.msgs_sent = sent_before
                self.confirm_fails = self.__dict__.get("confirm_fails", 0) + 1
                self.confirm_why = "the confirmation was not delivered: " + (heard[:120] or "no answer")
                self.log.append({"kind": "confirm", "sent": False, "why": heard[:200]})
                continue
            self.confirm_sent = True      # (v2.1 E1: only once delivered)
            self.confirm_cart, self.cart_touched = basket, False
            self.confirm_said = "change" if self.needs.version != version else "yes"
            miss = self.needs.__dict__.get("read_miss", "") if read else "the code's own finish reads no reply"
            self.log.append({"kind": "confirm", "sent": True, "basket": basket, "said": self.confirm_said, "read": not miss,
                             "unread": miss, "from": "store" if self.__dict__.get("store_cart") is not None else "count",
                             "reply": heard[:300], **({"partial_view": self.store_partial} if self.__dict__.get("store_partial") else {})})
            break
        else:
            return ""
        mark = f"message {self.msgs_sent} of {self.ask_limit()}"
        if self.confirm_said == "yes":
            return (f"Before the order, the code showed the shopper this basket and asked for a yes ({mark}); the shopper "
                    f"replied: {heard[:500]} - no change and no new need in it, so it counts as approval of exactly this "
                    "basket: the basket confirmation is done.")
        self.order_vetted = False
        return (f"Before the order, the code showed the shopper this basket and asked for a yes ({mark}); the shopper replied: "
                f"{heard[:500]}. It changes or adds a need, which is binding: fix the cart to meet it, re-read the lines' stock "
                "and price, then call the order tool (the checks run again). Decline only if nothing can meet the request now"
                + ("; no message is left." if self.msgs_sent >= self.ask_limit() else "."))

    def cart_truth(self) -> dict[str, float]:
        """v2.2 (F3): the cart's lines and units: the store's, as a cart view after the newest cart call showed them, else
        as counted."""
        got = self.__dict__.get("store_cart")
        if got is not None:
            return dict(got)
        return {t: self.trolley.held(t) for t in self.trolley.units if self.trolley.held(t) > 0}

    def store_read(self) -> None:
        """v2.2 (F3): reads the store's cart (the cart-view tool, one step) when no cart view came after the newest cart call,
        while three steps (the read, the message, the order) and CONFIRM_SEND + 3 seconds are left; the read ends with
        CONFIRM_SEND left. (v2.3 H1: not after a view since the newest cart call left rows undecoded - one more would too.)"""
        if self.__dict__.get("store_cart") is not None or self.__dict__.get("store_partial") or not self.role("cart_view") or self.moves_left() < 3 \
                or self.wire.spare() < CONFIRM_SEND + 3 or self.wire.done or self.wire.broke:
            return
        self.wire.world_end, self.covering = CONFIRM_SEND, True
        try:
            self.send_chunk([(f"confirm_view_{self.wire.tick}", self.role("cart_view"), {})])
        finally:
            self.wire.world_end, self.covering = 0.0, False

    def confirm_owed(self) -> bool:
        """v2.1 (E3): the basket about to be ordered still needs its confirmation: a message tool, a cart tool and an order
        tool and N >= 2 (as confirm_due, sent or not), and the cart is not the basket the shopper was last shown (by listing
        and quantity - after a change reply the fixed cart is shown again). (v2.2 F3: the cart as the store showed it -
        cart_truth; and a cart call or a refused order after the confirmation owes it again until a store read shows the
        same basket. v2.3 H1: a view that left rows undecoded is no such read - store_cart stays unknown.)"""
        if not (self.role("message") and self.role("cart") and self.role("order")) or self.ask_limit() < 2:
            return False
        shown = self.__dict__.get("confirm_cart")
        if shown is None or shown != self.cart_truth():
            return True
        return self.__dict__.get("store_cart") is None and bool(self.__dict__.get("cart_touched"))

    def confirm_block(self, viewing: bool = False) -> str:
        """v2.1 (E2): why an owed confirmation cannot be sent now; '' when it can - lines in the cart, a message left, two
        steps (the message and the order), CONFIRM_SEND seconds, the episode on, (v2.2) fewer than CONFIRM_FAILS failed sends.
        (v2.2 F3: lines in the cart as cart_truth gives it; `viewing` - a store read comes first and settles the lines. v2.3 H2:
        not the basket whose confirmation went out with no answer.)"""
        unknown = self.__dict__.get("confirm_unknown")
        if unknown is not None and unknown == self.cart_truth():
            return self.__dict__.get("confirm_why") or "delivery_unknown"
        if not any(q > 0 for q in self.cart_truth().values()) \
                and not (viewing and self.__dict__.get("store_cart") is None and self.role("cart_view")):
            return "the cart is empty"
        if self.wire.done or self.wire.broke:
            return "the episode is over"
        if self.msgs_sent >= self.ask_limit():
            return "no message left"
        if self.moves_left() < 2:
            return "fewer than 2 steps left"
        if self.wire.spare() < CONFIRM_SEND:
            return f"less than {CONFIRM_SEND:g} s left"
        if self.__dict__.get("confirm_fails", 0) >= CONFIRM_FAILS:
            return self.__dict__.get("confirm_why") or f"the send failed {CONFIRM_FAILS} times"
        return ""

    def kept_open(self) -> bool:
        """v2.5 (V1): the kept message can still go with the order: it is kept (confirm_due), the cart holds lines and the
        order path can send it now (confirm_block() == ''). Nothing of V1 acts otherwise."""
        try:
            return self.confirm_due() and any(q > 0 for q in self.cart_truth().values()) and self.confirm_block() == ""
        except Exception:
            return False

    def kept_text(self) -> str:
        """v2.5 (V1a): v2.4's kept-message refusal (screen), its long form the first time it is told, then the short one."""
        first = not self.__dict__.get("keep_told")
        self.keep_told = True
        return ("Not run (no step used): the last message is kept for the basket confirmation" + (
            ": when you call the order tool with the final basket in the cart, the code shows the shopper each line and the "
            "total and asks for a yes before the order goes. Do not message the shopper now: finish the cart, re-read stock "
            "and price, then call the order tool." if first else ", which the code sends when you call the order tool."))

    def kept_hold(self, call_name: str, call_args: dict[str, Any], held: str) -> str:
        """v2.5 (V1c): a held decline of the actor's while the cart holds a line the independent check passed (under the
        requirements in force, not refused, not sold out) and the kept message can still go with the order (kept_open): the
        hold text adds KEPT_LINE. Which holds fire, and how often, is unchanged."""
        try:
            if not held or KEPT_LINE in held or self.call_kind(call_name, call_args) != "decline" or not self.kept_open():
                return held
            dead = self.dead_lines()
            fits = [i for i in self.trolley.units if self.trolley.held(i) > 0 and f"{i}|{self.needs.version}" in self.passed
                    and i not in self.refused_lines and i not in dead]
            return held + " " + KEPT_LINE if fits else held
        except Exception:
            return held

    def kept_keep(self, to_send: list[tuple[str, str, dict[str, Any]]], scripted: bool) -> tuple[int, str] | None:
        """v2.5.1 (U2, red team v25 F3): the kept-message promise holds its two steps. While the kept message can still go
        with the order (kept_open), the actor's non-order calls may use only the steps above the last two - the basket
        confirmation, then the order; (how many may run, the hold text) for the calls past that, which are not run (no step
        used). Once per step count and set of held calls: the same calls sent again at that step run as before (the order
        then goes without the confirmation, logged unconfirmed as before). None when nothing is held."""
        try:
            room = self.moves_left() - 2
            if scripted or len(to_send) <= room or not self.kept_open():
                return None
            mark = (self.wire.tick, tuple(name + as_text(args) for _, name, args in to_send[max(0, room):]))
            marks = self.__dict__.setdefault("kept_marks", set())
            if mark in marks:
                return None
            marks.add(mark)
            return max(0, room), (f"Not run (no step used): {self.moves_left()} steps are left, and the order call needs the last "
                                  "two: the basket confirmation (the kept message) goes to the shopper first, then the order. Order "
                                  "now with the cart as it is - the code shows the shopper this basket and asks for the yes. If this "
                                  "call must still run, send it again: the order then goes without the confirmation.")
        except Exception:
            return None

    def kept_gone(self, lines: list[str]) -> set[str]:
        """v2.5.1 (U2): finish_now's CB5 stock re-read (cart_gone) keeps the kept message's time while it can still go with
        the order (kept_open): the re-read runs only with CONFIRM_SEND + 3 s left and its call ends with CONFIRM_SEND left
        (store_read's floor); with less, the lines already known sold out and no re-read. Otherwise cart_gone as before."""
        if not self.kept_open():
            return self.cart_gone(lines)
        if self.wire.spare() < CONFIRM_SEND + 3:
            return set(lines) & self.dead_lines()
        self.wire.world_end = CONFIRM_SEND
        try:
            return self.cart_gone(lines)
        finally:
            self.wire.world_end = 0.0

    def confirm_order(self, final: bool = False) -> str:
        """v2.1 (E3): every order path calls this just before the order goes to the world: an owed confirmation of the
        CURRENT basket is sent first. Returns the note to hold the order with - the reply changes a need, or the send failed
        and the next order call may send it again - else '' and the order goes. With `final` (the code's own finish) it never
        holds and the reply is not read. An order that goes with the confirmation still owed is logged with why: it is
        never declined for that (unconfirmed, both score nothing under H2; the order may still pass if H2 is wrong). (v2.2 F1:
        a failed send is sent again inside confirm_basket, so only a change reply holds the order.) (v2.4.1 N3: after a
        reply that changed a need, the basket shown before it is no confirmed basket unless the item checks ran again under
        the reply - screen_finish runs them only above CLOSING; see below.)"""
        if final and self.__dict__.get("change_restored") and not self.__dict__.get("change_restored_told") \
                and self.cart_truth() == self.__dict__.get("change_from"):
            # (v2.4.3 S2: only for the first order after the put-back, while the cart is the restored basket - a cart swapped
            # or repaired after it, or one a refused order may have changed, takes the normal path: the store's cart read first)
            # (v2.4.2 Q4: the code's own finish put back the basket shown before a change reply (change_back): it goes
            # as it is - never shown again unread - logged unconfirmed, once)
            if not self.__dict__.get("change_restored_told"):
                self.change_restored_told = True
                why = "change_unmet: the change was not done by the code's own finish; the basket shown before the reply was put back"
                self.log.append({"kind": "unconfirmed_order", "why": why, "shown": self.__dict__.get("confirm_cart"),
                                 "cart": self.cart_truth(), "store": self.__dict__.get("store_cart") is not None})
                _aoi_unconfirmed(self, why)
            return ""
        told = self.confirm_basket(read=not final)
        if final and self.__dict__.get("change_back_tried") and not self.__dict__.get("change_restored"):
            # (v2.4.3 S4: the put-back failed - its cart calls timed out, were refused or the time ran short: the cart goes
            # as it stands, logged change_unmet with the cart as the store last showed it, else as counted)
            self.log.append({"kind": "change_unmet", "why": "change_unmet: the code's own finish could not put back the basket "
                             "shown before the reply; the cart goes as it stands", "cart": self.cart_truth(),
                             "shown": self.__dict__.get("confirm_cart"), "store": self.__dict__.get("store_cart") is not None})
        if told and self.__dict__.get("confirm_said") != "yes" and not final:
            return told
        if not self.confirm_owed():
            # (v2.4.2 Q1: "vetted" only when screen_finish's pre-order check passed under the reply - vetted_pass)
            if self.__dict__.get("confirm_said") != "change" or (self.__dict__.get("order_vetted") and self.__dict__.get("vetted_pass")):
                return ""
            # (v2.4.1 N3: the reply changed a need, the cart is the basket shown before it, and the item checks did not
            # run again under the reply. The order is held once while the change fits: two steps per cart line (out, in),
            # the store read, the message and the order; and per cart line CONFIRM_READ - LAST_CALL (10 s, about three
            # actor turns: find the new line, take the old out, put the new in) before the code's own finish (LAST_CALL),
            # whose floors (CONFIRM_SEND, the store read's 3 s, CONFIRM_KEEP) then keep the order's reserve. Else, or when
            # the order comes again, it goes logged unconfirmed - never presented as confirmed; never declined for it (E3):
            # a decline scores nothing for sure, the old basket still can if the reply was misread.)
            # (v2.4.2 Q4: and only with a message left - the changed basket needs a fresh confirmation, and without one it
            # would be ordered unconfirmed, while the old basket is still right if the reply was misread -
            # and with the steps to undo the change (two a line more): the code's own finish puts an unfinished change back,
            # change_back)
            lines = len(self.cart_truth())
            if not final and self.__dict__.get("change_held") != self.msgs_sent and self.msgs_sent < self.ask_limit() \
                    and self.moves_left() >= 4 * lines + 3 and self.wire.spare() >= LAST_CALL + (CONFIRM_READ - LAST_CALL) * lines:
                self.change_held = self.msgs_sent
                self.change_from, self.change_offer = self.cart_truth(), None      # (v2.4.2 Q4: the basket before the change)
                self.log.append({"kind": "change_hold", "cart": self.cart_truth(), "spare": round(self.wire.spare(), 1),
                                 "moves_left": self.moves_left()})
                return ("The shopper's reply to the basket confirmation changed a need, and the cart is still the basket "
                        "shown before that reply: change the cart to meet the reply, then call the order tool - the code shows "
                        "the new basket to the shopper before the order. If this cart already meets the reply, call the order "
                        "tool again.")
            why = "change_unmet: the reply to the confirmation changed a need and the cart is the basket shown before it"
            self.log.append({"kind": "unconfirmed_order", "why": why, "shown": self.__dict__.get("confirm_cart"),
                             "cart": self.cart_truth(), "store": self.__dict__.get("store_cart") is not None})
            _aoi_unconfirmed(self, why)
            return ""
        why = self.confirm_block() or self.__dict__.get("confirm_why") or "not sent"
        # (v2.2 F1: a failed send was sent again at once in confirm_basket - no hold for it)
        self.log.append({"kind": "unconfirmed_order", "why": why, "shown": self.__dict__.get("confirm_cart"),
                         "cart": self.cart_truth(), "store": self.__dict__.get("store_cart") is not None})
        _aoi_unconfirmed(self, why)
        return ""

    def change_back(self) -> bool:
        """v2.4.2 (Q4): at the code's own finish after an N3 change hold, the cart goes as changed only when the actor itself
        ordered it (change_offer: the change done in its view) or it is still the basket shown before the reply; any other
        cart - empty or half-changed, the finish floor came first - is put back to that basket by the code (code_put: cart
        calls only, no model call), which then goes logged unconfirmed change_unmet (confirm_order). True when put back."""
        was = self.__dict__.get("change_from")
        if not was or self.__dict__.get("change_back_tried"):
            return False
        now = self.cart_truth()
        if now == was or now == self.__dict__.get("change_offer"):
            return False
        self.change_back_tried = True
        was_end = self.wire.__dict__.get("world_end", 0.0)
        try:
            self.wire.mute = True
            # (v2.4.3 S4: each cart call of the put-back ends with the store read, the confirmation and the order's time left -
            # a timed-out or slow call never takes the order's reserve)
            self.wire.world_end = CONFIRM_SEND + 3 + CONFIRM_SLACK
            bad = self.code_put(sorted(was.items()))
        except Exception as err:
            bad = None
            self.log.append({"kind": "change_back_error", "error": f"{type(err).__name__}: {err}"[:300]})
        finally:
            self.wire.world_end = was_end
        self.change_restored = bad == [] and self.cart_truth() == was
        self.log.append({"kind": "change_back", "from": now, "to": was, "done": self.change_restored, "moves_left": self.moves_left()})
        return self.change_restored

    def confirm_mark(self) -> str:
        """v2: item (5) of the pre-order check counts the basket confirmation as done when it was sent while the cart held
        exactly this basket (after a reply that changed a need, the checks before this text ran again under it)."""
        kept = self.__dict__.get("confirm_cart")
        now = self.cart_truth()      # (v2.2 F3: the store's cart as read)
        said = self.__dict__.get("confirm_said")
        if not kept or kept != now:
            return ""
        if said == "yes":
            return " (the basket confirmation counts as done: the shopper was shown exactly this basket and approved it)"
        return (" (the basket confirmation counts as done: it was sent with exactly this basket, and the item checks above ran "
                "again under the needs in the shopper's reply)")

    def screen_finish(self, cid: str, call_name: str, call_args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """Runs the pre-finish checks on an order or decline call.
        Holds the call with a not-run message for dead cart lines, cheaper fits, stale stock or checks.
        For declines, forces final reads and reviews; returns (message or None, cleaned args)."""
        if self.__dict__.get("change_from") is not None and self.call_kind(call_name, call_args) == "order":
            self.change_offer = None      # (v2.4.3 S1: an order try after a change hold voids an earlier pass: the cart is the
            # actor's change only once a check lets it through - the pre-order check's end, or the order going on below)
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
            cheaper = self.cheaper_fit(thing) if thing else ""
            for line in ([] if cheaper else [i for i in self.trolley.units if i != thing]):
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
        if self.call_kind(call_name, call_args) == "order" and self.trolley.units:
            walk = self.walk_hold(None)
            if walk:
                return walk, call_args
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
            self.vetted_pass = False      # (v2.4.2 Q1: true only when this check reaches its end with no fault)
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
            told = self.confirm_basket()      # (v2: the basket confirmation goes now, with the final basket, before the check)
            if told and self.__dict__.get("confirm_said") != "yes":
                return "Not run yet (no step used). " + told, call_args
            self.vetted_pass = not faults      # (v2.4.2 Q1: no fault, no ruling, no better item, no change reply held it)
            if self.vetted_pass and self.__dict__.get("change_from") is not None and self.cart_truth():
                self.change_offer = self.cart_truth()      # (v2.4.3 S1: every check passed this cart after a change hold;
                # an empty cart is never the change)
            return ("Not run yet (no step used). " + (told + " " if told else "") + "Pre-order check: (1) the cart holds exactly one line " + ("per product asked for, each at its quantity" if self.needs.items() else "the chosen item at the required quantity") + ", and nothing else; "
                    "(2) after your last cart change you re-read this item's stock and price with a tool that shows them, and it is in stock and price times quantity is within the current budget; "
                    "(3) its own properties meet every stated requirement, every need the shopper revealed and every change announced; "
                    "(4) if the cheapest or best was asked for, no better fitting item exists among everything you could find; "
                    "(5) every procedural rule in the request has been followed" + self.confirm_mark() + ". If anything is unchecked, check it now; "
                    "otherwise call the order tool again."), call_args
        if self.call_kind(call_name, call_args) == "order":
            told = self.confirm_order()      # (v2.1 E3: the order goes now and its basket was not shown yet: it is shown first)
            if told:
                return "Not run yet (no step used). " + told, call_args
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
            if slack and self.moves_left() > reads + 4 and self.wire.spare() > CLOSING + 40:
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
            best = self.unrefuted() if not self.__dict__.get("best_held") and self.moves_left() > 4 else []
            for _ in range(3):
                unchecked = [i for i in best if f"{i}|{self.needs.version}" not in self.passed]
                if not unchecked or self.wire.spare() < CLOSING + 30:
                    break
                self.check_item(unchecked[0], "ORDER")
                best = self.unrefuted()
            if best and self.moves_left() > len(best) + 3 and self.wire.spare() > CLOSING + 20:
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
            return None, self.scrub_decline(call_args, call_name)
        if self.__dict__.get("change_from") is not None and self.call_kind(call_name, call_args) == "order" and self.cart_truth():
            self.change_offer = self.cart_truth()      # (v2.4.3 S1: the order goes: no check held this cart)
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
                responses.append((proposal["id"], "Not run (no step used): an earlier cart change in this reply was not run; read why, "
                                                  "then send the cart changes again."))
                continue
            refusal, call_name, call_args = self.screen(proposal, permitted, scripted, len(proposals), queued_messages)
            if refusal:
                _aoi_block(self, call_name, call_args, refusal, "screen")
                cart_held = cart_held or (call_name == self.role("cart") and isinstance(call_args, dict) and bool(call_args)
                                          and not self.trolley.is_put(call_args))
                responses.append((proposal["id"], refusal))
                continue
            if call_name == self.role("message"):
                queued_messages += 1
            to_send.append((proposal["id"], call_name, call_args))
        if to_send and to_send[0][1] in self.finishers():
            cid, call_name, call_args = to_send[0]
            _aoi_args0 = call_args
            held = "" if scripted else self.decline_hold(call_name, call_args)      # (v1 P1.3)
            if not held:
                held, call_args = self.screen_finish(cid, call_name, call_args)
            if held and not scripted:
                held = self.kept_hold(call_name, call_args, held)       # (v2.5 V1c)
            _aoi_screened(self, call_name, _aoi_args0, call_args, held)
            if held:
                responses.append((cid, held))
                return responses
            to_send[0] = (cid, call_name, call_args)
        if to_send and to_send[0][1] not in self.finishers() and self.finishers():
            hold = max(0, self.moves_left() - 1)
            kept = self.kept_keep(to_send, scripted)      # (v2.5.1 U2)
            if kept:
                hold = kept[0]
            for cid, _, _ in to_send[hold:]:
                responses.append((cid, kept[1] if kept else "Not run (no step used): only the last step is left, and it must be the order (or the decline)."))
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
        return responses

    def send_chunk(self, slab: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str]]:
        acts = [{"name": call_name, "args": call_args} for _, call_name, call_args in slab]
        prior = self.wire.tick
        _aoi_t = _aoi_finish_before(self, acts)
        resp = self.wire.act(acts)
        _aoi_finish_after(self, _aoi_t, resp)
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
        """Folds one tool call and its result into the run state.
        Updates counters, archive, cart and stock knowledge, refusals; collects remarks from helpers.
        Returns the clipped result text with an appended note line of remarks."""
        shown, obs = obs, lensed(obs)
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
        self.absorb_n = self.__dict__.get("absorb_n", 0) + 1
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
            self.store_cart, self.cart_touched, self.store_partial = None, True, None      # (v2.2 F3: unknown until a cart view after this call)
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
            shows = self.trolley.viewed(obs, self.wire.tick, self.code_fault(obs))
            # (v2.2 F3: the store's cart as this view shows it: what a basket confirmation shows and is checked against)
            self.store_cart = dict(self.trolley.units) if shows and not self.code_fault(obs) else None
            # (v2.3 H1: a view that left row-like entries undecoded is no store truth - the store's cart stays unknown, the
            # confirmation stays owed after a cart call and shows the count with the lines this view read; no store read again
            # until the next cart call)
            self.store_partial = list(self.trolley.undecoded) if not self.code_fault(obs) and self.trolley.undecoded else None
            if self.store_partial:
                self.log.append({"kind": "partial_view", "rows": self.store_partial[:12], "step": self.wire.tick})
                _aoi_partial_view(self, self.store_partial)
            remarks.append(self.trolley.summary())
        refused = self.refused(call_name, obs, resp if last else {"finished": False})
        if refused and call_name in self.finishers():
            # (v1 P1.4: as finish_now's TA6 - the refusal's lines are also read by the code for a budget; once per line)
            self.code_news(self.shopper_lines(call_name, obs, True))
        if refused and self.last_kind == "order":
            self.cart_seq = self.absorb_n
            self.store_cart, self.cart_touched, self.store_partial = None, True, None      # (v2.2 F3: the store may have changed the cart)
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
            self.archive.add(opener_obs)
        overview = self.needs.overview(self.step_cap, self.msg_cap)
        if overview:
            opener += "\n\n" + overview
        if self.confirm_due():
            opener += ("\n" if overview else "\n\n") + "- the last message allowed is kept for a basket confirmation, which the code sends when you call the order tool"
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
            if self.walk_step():
                continue
            model_list = ACTOR if self.stuck < STUCK_LIMIT else ACTOR[1:] + ACTOR[:1]
            msg = self.wire.query_model(self.chat, model_list, 2500, self.actor_tools())     # (v1 P1.3: effect notes)
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
                _aoi_deferred(self, proposals)
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
        _aoi_end(self, "execute")
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
        _aoi_end(errand, "crash")
        return errand.log


__all__ = ["agent_main"]
