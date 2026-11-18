from __future__ import annotations
import json
import re
from time import monotonic, sleep
from typing import Any
from src.agent.proxy_client import ProxyClient
ACTOR = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-haiku-5.5"]
READER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"]
FIT_MODELS = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"]
CHECKER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
AUDITOR = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
LISTENER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
PLANNER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
FINISHER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
DEADLINE = 265.0
CLOSING = 70.0
LAST_CALL = 28.0
CLOCK_STEP = 30
CH_SYNONYM_MIN = 4
VIEW_CHARS = 20000
SEARCH_TEXT = 600
RECORD_TEXT = 2500
SOFT_SPEND = 0.03
SPEND_CAP = 0.045
HARD_SPEND = 0.10
FINISH_STEPS = 2
FINISH_TURNS = 2
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
LIST_SIZE = 4
# how reliably an answer covers a property by its place in the list (1st, 2nd, 3rd, 4th, 5th, later): a property
# named only further down a list may still be open, so later lists lead with those properties
LIST_REACH = (1.0, 0.68, 0.64, 0.39, 0.33, 0.2)
# weight of a category's most carried key, second most carried, ... when choosing what to ask (0.01 each beyond)
KEY_PRIOR = (0.28, 0.2, 0.12, 0.12, 0.1, 0.08, 0.03, 0.03)
# listings of each product read before the first question (one product / two / more): off two or three listings key
# counts are mostly ties, off six to eight they separate
READ_FIRST_BATCH = (6, 4, 3)     # with a tool that reads several listings in one step
READ_FIRST_SINGLY = (6, 3, 2)    # one step per listing
ASK_DEFAULT = 3
FIRST_ORDER_BY = 7
FIRST_ORDER_DUE = 10
IDLE_LIMIT = 3
STUCK_LIMIT = 4
SWEEP_READS = 12
WALK_PAGES = 6
WALK_TOTAL = 8
VALUE_PAGES = 4
VALUE_TOTAL = 8
VALUE_TOTAL_PER = 5
LABEL_TRIES = 4
LABEL_OPENS = 3
CEIL_CALLS = 4
CEIL_OPENS = 3
FIT_PAGES = 30
LINK_WORDS = {"and", "or", "the", "for", "with", "of", "a", "an", "in", "on", "to", "by", "from", "at", "as", "is", "are", "be",
              "this", "that", "it", "its", "your", "our", "you", "all", "any", "not", "no"}
ASK_AFTER_SETS = 2
ASK_GRACE_STEPS = 3
ASK_AFTER_STEPS = 5
LIST_MOST = 5
SWEEP_ALL = 60
SWEEP_MOST = 10
COVER_PAGES = 3
COVER_LISTS = 2
VALUE_BLOCKS = 2
ACT_RULES = """You are a careful shopping assistant working for one shopper in a simulated store; the supplied tools are your only way to act. The errand scores 1 or 0, and it scores 1 only when the finish is right in every respect: the ordered item meets every requirement the shopper wrote, every hidden need the shopper reveals when asked, and every change the shopper announces along the way; the quantity and the budget are right; the cheapest (or best) item is chosen when the request asks for that; and every rule of procedure in the request is obeyed. When nothing in the store meets everything, the right finish is a decline, done exactly as the request says.

HOW THE STORE WORKS
- Each tool call costs one step, failed and invalid calls included, and steps are limited. Thinking costs nothing.
- One reply may carry several tool calls when none needs another's result (for example opening several listings' details); they run in order. An order or a decline always goes alone.
- A details tool that takes a LIST of listing ids shows all their full records for ONE step: when you open two or more listings, use it, with at most five ids a call (a longer list is refused in some stores and reads nothing).
- Tool names, argument names and result shapes are different in every store. Read each tool's description and JSON schema and use exactly the names and allowed values given. Learn from the first results which tool shows which facts (price, stock, detailed properties).
- Early on, learn which category labels the store puts on listings of the requested kind. When neighbouring labels exist, the right one is the label that matches the request's wording as a whole; search and buy inside it.
- Cover the shelf before you judge. The code opens the errand for you: it searches each product's stated value with its kind word, reads the next pages, lists the category under the budget and opens up to ten listings (five each for several products); their results are in the conversation when your turn begins, so build on them rather than repeating them. A listing that never appeared in any result cannot be bought, and most wrong declines come from never searching a revealed value: after a reply the code searches the revealed value with the kind word and opens the rows carrying it; search any other value yourself the same way (the value first) and read a search's second page when its total is larger than the rows shown; over the errand open five or six candidates of the requested kind, not two.
- Searches differ from store to store; learn this one's from its first results. Rows may come best match first or in a fixed id order; a keyword search may need every word (adding a word shrinks the total) or take any word (adding a word grows the total, and listings matching more of the words rank higher); only a few rows or pages may be visible; a filter may work in one mode only (a keyword mode may ignore the category or price filter, a filter mode may ignore the words). Compare a reported total with the rows you can see: when it is larger, the fitting or cheapest listing may sit on a later page or beyond the visible ones. When every word is required, narrow with more exact words (or the category and price filters with a lower price limit) until the total fits on the visible pages. Keep queries short (one to three words) with the most distinctive words FIRST: some stores read only the first word or two of a query and ignore the rest, so a long query may search only its opening words (compare a longer query's total and rows with a shorter one that starts the same way; the notes report what they show). When every word is required, a required value with the item kind ("blue chair", the value first) lists exactly the listings that carry it. When any word matches, common words (the item kind, a unit) match most of the shelf and bury the rest: a required value's own distinctive word or figure alone ("herringbone", "30m", "0.4") lists just the listings that carry it, so search it alone and in the listings' other spellings ("30 m", "3 atm"), read every page of such a search when its total fits on them, and put the item kind with the values together only to rank the closest matches first. A keyword search may also match category names and property values. Notes after each search say what the totals so far show about this store's search.

THE SHOPPER AND HIDDEN NEEDS
- The shopper holds needs they did not write down and reveals one only when a message asks about that property. The FIRST thing a message asks is answered most reliably, so a message asks about one property, or two at most with the likelier first: a short, plain, friendly question that names the property (one or two example values are fine). No instructions, no lists, labels, positions or rules.
- "Don't mind", "no preference", "anything is fine" means the property is free. A stated preference or limit is a hard requirement from then on, exactly as worded, numbers included.
- When questions are planned for you, they go out once you have opened the details of two or three listings of the requested kind, so open those in your first moves. The answers arrive in the conversation with their meaning; apply each revealed need as a hard requirement and search again with it and the stated values together.
- If you ask yourself: first open two or three listings' details, then ask about what the story hints at (who it is for, how it must look, where or how it is used), then about properties the listings carry that differ among candidates. Never ask what the request already settles and never go over the message budget. Your own messages are how you carry out the steps the instructions state (a confirmation, the basket, the total, an answer to the shopper): the message tool is yours whenever the allowance has a message left. While hidden-need questions are planned for you, leave the asking to the plan (it picks properties from the listings you opened) and keep your own messages for what the instructions ask of you; before an order, the basket is told to the shopper for you when you have not told it after your last cart change.

RULES OF PROCEDURE
- The request may set rules for how to work: how many listings to look at or compare, what to check, what to tell the shopper, what a remark must say, how to decline. They are graded like requirements; follow each one literally and fully. "Look at" or "open" N listings means N fully fitting listings, each opened with the details tool.
- When the shopper asks you something (for example to confirm a change), it is answered for you while messages remain; the change itself is binding.
- The status line lists the INSTRUCTIONS IN FORCE: the request's rules and every instruction the shopper or the store adds later (confirm with me before buying, tell me the total, a new message or step allowance). They can change at any time, so re-read them, the requirements and the cart after every reply, notice or refusal, and decide the next move from what they say now, never from habit. A stated step still undone when you finish is carried out for you right before the finish goes, and one message is kept for it; a message you send yourself after the last cart change counts too.
- What the shopper would merely like ("ideally", "would be nice", "if possible") is not a requirement: it only breaks ties between items that already meet everything, and never beats a cheapest rule.
- When the request leaves something only the shopper can settle (for example no price limit, or two wishes that may not both be met and the shopper must say which matters more), ask before you choose.

CHANGES WHILE YOU WORK
- The shopper may change things: a new budget (higher or lower), a requirement dropped, replaced or added, a price rise, an item selling out. One line can carry several changes. Such lines can arrive inside any tool result (an extra field holding sentences) or with an order attempt. Each change binds from that moment: re-check the pick and the backups, and when the cheapest or best is asked for, choose again over the new acceptable set.
- An order attempt can be REFUSED: the episode goes on and the cart is as it was. Read the refusal, apply every change it states, take out of the cart whatever no longer fits (for example a sold-out item), put in the next verified fit, re-check its stock and price, and order again.
- Items can also sell out or change price without notice, so stock and price are re-checked right before every order attempt; an item whose current price times quantity is over the current budget no longer fits.

SEVERAL PRODUCTS
- A request may ask for several different products, each with its own quantity, requirements and hidden needs. Work on every product: search each one's category, verify a fit for each, and order them together, one line per product at its quantity. A cart tool that takes a list of lines may replace the whole cart with that list, so send every product's line in each such call. The cheapest or best rule applies within each product.
- A requirement may give its value by pointing at another listing or page ("the same colour as the jumpsuit sold by X"): find that listing first (search the words that name it, such as its shop or brand with its kind, and the properties it is described by), then open it with every tool that shows one listing (a store can split a listing between a facts view and a page view, and the page's notes can say which listing really holds the value). The listing meant shows EVERY value the description names (its shop, its kind and each value); a listing of that shop whose page says a version with one of those values is sold elsewhere is a near twin, and its notes are never followed. When the shop's name with a value finds nothing, search the shop's name alone and read its rows page by page (a title can write a value differently, such as joined to other characters). When the page meant sends you to a further listing for the property asked (a note about another property does not count), find and open that one the same way: the last page of the chain gives the value, and every page of the chain is opened before the order. Take the value as the requirement relates it (the same value, anything but that value, or the value the page's own rule or note pairs with it), then search the requested product with that value.

THE CART
- The cart tool may ADD units to a line that is already in the cart rather than set them: putting 1 more of a held line makes 2. Never put a line in again that the cart already holds; to change units, take the difference out or in. A note after each cart call says what the code counts in the cart.

HOW TO JUDGE A LISTING (this decides the score)
- Each requirement is read from the listing's field for that property, by the value's own words as this store writes them in that field. A field whose name says the same property counts whatever its spelling: separators, a prefix, a shorter or longer name, or a usual alternative name ("fit" for a pants fit type, "neckline" for a collar type, "dial size" for a watch dial size, "color family" for a colour). When a field carrying the property's own name exists, it decides, and so does a field of that name with one more leading word (pants length for length, clothing material for material): when either one writes the value, the requirement is met. Two fields whose names differ only by separators (size class, size_class) are one field: the requirement is met when either one writes the value. Otherwise such an alternative field decides; when no field names the property at all, the options list, the title or the description may show the value. A field decides only when this store writes values of the required value's own kind there (colours for a colour, model codes for a model code): when no listing of the category writes a word or figure of the required value in that field, the field names the property another way (car names where a model code is required) and is silent on it, and the listing's own text then decides. The description counts only for what it states about this listing itself, not for a pairing, a comparison or another product.
- Values match by their words, as this store writes them: case, separators, singular and plural and spelling variants are the same; a more specific value that contains the required one meets it ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060); one element of a list is enough. A synonym, an abbreviation, a related word or a shade ("tech wood" for technical wood, "win11" for windows 11, "embroidered" for embroidery, "solid" for plain, ivory or cream for off-white) meets a value only when no listing of the category writes the value itself in that field: when other listings write "matte" as their finish, a "satin" finish does not meet matte. A requirement marked as revealed by the shopper takes its value from the shopper's answer, which may put the store's value in other words: a field value that names the same thing in other words (another everyday name or wording of that same thing) meets it when no listing that meets every other requirement writes the answer's own words (prefer one that does); a different value of the property never does, however close (a nearby kind, a related feature, an amount where the requirement describes a quality). A value joined with another by a plus ("wool + nylon") does not meet one of its parts when the field also carries the other part alone ("nylon"). FIELD EVIDENCE lines, when given, are computed from this store's records and say which case holds. A different value of the property does not (quartz is not mechanical, rosewood is not maple), nor does a word that only looks alike. A value that claims to cover every value of the property without naming any ("suitable for all occasions", "all seasons", "universal") names none of them and does not meet a required value; a list that also names the required value still meets it. A family, series or range written as a pattern does name its members: "every 48xx model", "the 48xx series" or "4800-4899" includes 4820 and meets it, while a family or range that leaves the value out (every 61xx model for 4820) fails it. A field that names another value of the property does not meet the required value just because the two often go together or one tends to imply the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); only a shade usually sold as the required colour, or a more specific value that contains it, counts.
- A field of a different property is not the requirement's property (a lining material is not the material, a length is not a style, a strap length is not a strap type): it never stands in for a property whose own field the category's listings carry.
- Only what the shopper stated or revealed counts. Never rule out a listing for something the shopper did not ask for: an age group, gender, size, occasion, condition, quality or use you guess from the story or the description. The story is context; if it seems to matter, ask the shopper about that property instead of assuming. Your own questions are not requirements: only the shopper's answers are.
- Numbers are compared as amounts after converting units (4.5 cm is 45 mm); limits include the edge; a range counts only when all of it meets the limit. When no field names the property and the title is the evidence, an exact required value is met when the title shows its words and figures in the same order, even with more detail after them (9' x 12' 4" shows 9' x 12'; 9' x 13' does not).
- Figures in a title ("55 x 94 Inch", "117 x 228 cm") do not say which dimension each one is: never rule a listing out, or skip opening it, for the figures in its title; read the fields of its details.
- Respect exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
- The requested kind of product is met by the listing's category alone: a listing on the requested category IS that kind, even when its title or description names another object (a container, a case, a refill, a set, an accessory), so never skip, reject or demote it for what it seems to be; judge it only on the requirements, price and stock. No property has to repeat the category name. When the store labels categories, the item must sit in the one whose label matches the request's wording as a whole (a request for "A or B" matches a category "A & B", not a narrower neighbour "B"). It must be in stock when the request wants things on hand, and price times quantity must be within the budget (equal is fine). A listing's price is in the budget's unit unless its record names another (45 against a budget of 1000 US cents is 45 cents); never rescale either.
- A decline is rarely the right finish. When no listing meets the strict wording but one meets each requirement under these rules (a more specific value, another name of the same field, or a synonym or shade where no listing of the category writes the value itself in that field), that listing is the fit; decline only when the search really found nothing that meets every requirement under these rules.

CHEAPEST OR BEST
- For the cheapest, the order must be the lowest-priced item among ALL items meeting every requirement; one fitting item is not enough. When the keyword search needs every word and matches property values, the value-and-kind search lists the candidates that carry the value: read all its pages and open them by price; a category listing with a price limit shows the whole category's prices but not its values: use it to see which cheaper rows exist (lower the limit until the total fits), then check those rows' properties. Search each required value with the item kind as a short query, the value first ("blue chair", "30m watch"), and read every page of it when its total fits on the readable pages (when it does not, add a second required value or use a narrower kind word); do the same with the kind's other names ("stool", "bench") and with the value alone; when a value cannot be searched (a figure the store drops), also search the product line of each fit or near-fit you find, its maker with one distinctive word of its title as two words, every page: a line's other sizes and colours are separate listings that share its property values and may cost less; sort the visible candidates by price yourself and open their details from the cheapest up; the first whose property fields meet everything, with every cheaper one opened and rejected, is the answer. Keep the next fitting ones as backups. Before each order the code lists the category below the cart line's price itself: no rows means nothing cheaper exists on that shelf; rows mean cheaper candidates, which it opens for you to judge under the value rules.
- For the best by some measure (most, largest, longest, highest...), compare ALL fitting items on that measure the same way and take the extreme. An item stating no value for the measure cannot win.

FINISHING
- Keep a ranked list of verified fits with at least one backup: the first pick may sell out at checkout.
- Before every order attempt: the cart holds exactly one line per product asked for (one line when one product is asked), each the chosen item at its required quantity, and nothing else; after your last cart change you have re-read the chosen item's stock and price, and it is in stock and within budget.
- Decline only after a thorough search finds nothing that meets every requirement, following the request's decline rule exactly. When the request says not to offer substitutes, the decline remark names, describes or points to no listing: no ids, titles or prices.
- Budget your steps: the first order attempt goes out while at least seven steps remain, so a refusal can still be handled. With 7 steps left only cart, stock-check and finishing tools remain. Decide once; do not swap the cart back and forth.
- Every reply carries at least one tool call. Keep the text short: the requirements now, the verified fits with price and stock, the next move."""
KIT_RULES = """

PRODUCTS THAT MUST WORK TOGETHER
- This request asks for a main product and other products that must work with it (fit it, suit it, be compatible with it). The set counts as a whole: each other product's line must fit the main product's line, and every line must also meet its own requirements, the shopper's answers and, together, the budget.
- Listings state what they fit in their own text, in whatever form this store uses: read it as written, in the search rows when they show it, before opening pages. An exception always wins, and a statement about ANOTHER part speaks for that part, not for the listing it is printed on. The main product's listing can write several codes (its own model number and a family or series it belongs to): the code that links is the one the other products' listings carry (in their titles, their lists of what they work with, or notes naming other parts), so match on that one, never on a code no other listing repeats.
- Work in this order: (1) find main-product listings that meet the main product's own requirements within the budget and note the codes they share with the other products' listings; (2) for each other product, search the linking code itself (alone, then with the product's own stated value) and its category with its stated values, and keep the rows whose statements cover that code (when a search for a code finds nothing, read the category's rows page by page); every line of the set carries the same linking code; (3) the [works with] line gives the code's reading of which listings fit which, and the sets that fit together within the budget, cheapest first; (4) by the middle of the errand put one whole set in the cart with a single cart call holding every product's line, then check each line against its requirements and the shopper's answers, and swap a line only for one that fits the same main listing.
- When no listing of another product that meets its requirements fits the main listing you chose, change the main listing to one such a listing fits; never decline while that change is untried.
- Under a cheapest rule the set's total decides: a cheaper main listing that only dear parts fit can cost more in all.
- When the products must all work with one device the order does not include (the shopper's own), every line must name that same device or code; when the request does not say which device, choose the code that fitting listings of every product share."""
READ_REQUEST = """Read the shopper's request and the store's tool list below, then answer with ONE compact JSON object (no indentation), nothing else, holding these keys:

"product": the kind of product asked for, in the request's words (for several products, all of them joined with "and").
"items": one entry per DIFFERENT product the request asks to buy, in its order: {"product": its kind in the request's words, "quantity": units of one listing of it (as for "quantity" below), "requirements": its firm requirements (as for "requirements" below), "depends_on": the 1-based number of the other entry this product must work with, fit, suit or be compatible with as the request says, else null}. One entry when one product is asked. A product the request names only as the source of a reference (another listing whose value is to be matched or avoided) or as something the shopper already has is not an entry, and the questions never go to it.
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
"instructions": the procedural instructions again, one object each: {"text": the instruction in the request's words, at most 20 words, "before_finish": true when it must be carried out before or at the finish (before an order or a decline: confirming with the shopper first, telling them what is in the basket or the total, a closing remark, a last check), else false, "needs_message": true when carrying it out means sending the shopper a message that TELLS them something (the basket, the total, a confirmation, a closing remark), else false, "invites_question": true when it only offers to answer questions or wants the assistant to put a question to the shopper (it asks the assistant for no message that informs the shopper), else false, "before_change": true when it asks the assistant to check with the shopper before changing, replacing or removing something already in the cart, else false, "on_order": false when it applies only when nothing is bought (what to do or say when no listing fits, a decline), else true, "conditional": true when it applies only if something happens first (a line turns out unavailable, the price passes a limit, a choice comes up), else false}. Also an instruction that only says how to finish or what to say when finishing. How many messages or steps the assistant has is message_budget or step_budget, not an instruction. Empty when the request sets none.
"ask_first": what only the shopper can tell and should be asked before choosing: for example the budget when no limit is given, which of two wishes matters more when the request says both may not be possible (then put both wishes in soft_wants, not in requirements), or anything the request says to check with the shopper. Empty when none.
"hints": the story details that point at a property of a product without stating its value (what it is used for or with, where, by whom, what it must fit, match or survive), each "property (product): the detail in a few words" ("property: detail" for one product), the property named as listings would likely name it. When a detail may concern more than one property, name the likeliest first and the next after a slash ("colour / style (chair): must suit a room with one painted wall"): what the item must match, suit or sit well with (a room, a wall, an outfit, things it is used with) and how it looks to the eye point first to its colour, then its pattern or style. Leave out a detail that only explains a requirement the request already states (why it must be the stated material or size). Weather, a job, a schedule, travel, or a place where the item is not worn or used is background, never a hint; so is a purpose every listing of the kind serves (storage for a dresser, warmth for a coat). Empty when none.
Every sentence of the request lands somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or plain story. When unsure whether a sentence is a rule, put it in "rules".
"references": requirements whose value the request gives only by pointing at ANOTHER listing or page ("the same colour as the jumpsuit sold by X", "within 5 of the figure on the page for Y", "a colour other than the one on Z's page"), each "property (product): as shown on <how the request names that listing>" ("property (product): different from the one shown on <how the request names that listing>" when the value must differ from that listing's); also put each in "requirements" in that form. Empty when none.
"reference_targets": for each entry of "references", the listing it points at: {"property": its property, "product": its product, "seller": the shop or brand name the request gives for that listing, exactly as written, "kind": the kind of product that listing is, "details": [each other value the request names for that listing, short and in the request's words ("blue", "4gb", "net weight 167.262186 g")], "relation": "same", "differ" (the value must NOT be that listing's: set apart from, different from, other than; but words refusing everything set apart from it - "don't show me anything where it comes in other than <relation> ...", "anything with it apart from <relation> ... gets binned" - give the relation after them, never differ), "pairs" (the value must be the one that goes with, works with or pairs with that listing's value, not that value itself), "within", "at most" or "at least" (how the value must relate to that listing's), "tolerance": the number for "within", else null, "unit": its unit, else ""}. Empty when none.
"decline_no_names": true when the request's decline rule forbids offering, naming or suggesting substitutes or alternatives; false when it asks you to name something; null when it gives no rule.
"roles": an object naming the tool for each role, with the exact tool names from the list, or null when no tool fits: "search", "details", "view", "message", "cart", "cart_view", "order", "decline". "message" sends a message to the shopper; "cart" adds to or takes from the cart; "order" buys or checks out; "decline" ends the errand without buying.
"search_page": {"arg": the search tool's argument that moves on to further results, copied exactly from its schema, or null when it has none, "kind": "offset" when it counts results to skip, "page" when it counts pages, "token" when it takes a value a result hands back}.
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
"instructions": procedural instructions these lines state for the assistant from now on (confirm with me before buying, tell me the total or the basket, do or do not do something, how to finish or what to say when finishing), one object each: {"text": the instruction in the lines' words, "before_finish": true when it must be carried out before or at the finish (an order or a decline), else false, "needs_message": true when carrying it out means sending the shopper a message that TELLS them something (the basket, the total, a confirmation), else false, "invites_question": true when it only invites the assistant to ask the shopper something, else false, "before_change": true when it asks the assistant to check with the shopper before changing, replacing or removing something already in the cart, else false, "on_order": false when it applies only when nothing is bought (a decline), else true, "conditional": true when it applies only if something happens first, else false}; [] when the lines state none. A need, a budget or a dropped requirement is not an instruction.
"allowance": when the lines state a new limit on the assistant's messages, questions, steps, moves or calls: {"messages_total": the new total number of messages allowed, or null, "messages_left": how many messages may still be sent from now, or null, "steps_total": the new total number of steps allowed, or null, "steps_left": how many steps are left from now, or null}; null when no such limit is stated.
"references": for each need in "needs" whose value the shopper gives only by pointing at another listing or page, the listing it points at: {"property": its property, "seller": the shop or brand name the line gives for that listing, exactly as written, "kind": the kind of product that listing is, "details": [each other value the line names for that listing, short and in the line's words], "relation": "same", "differ" (the value must NOT be that listing's: set apart from, different from, other than; but words refusing everything set apart from it - "don't show me anything where it comes in other than <relation> ...", "anything with it apart from <relation> ... gets binned" - give the relation after them, never differ), "pairs" (the value must be the one that goes with, works with or pairs with that listing's value, not that value itself), "within", "at most" or "at least", "tolerance": the number for "within", else null, "unit": its unit, else ""}; [] when none.
One line can carry several changes; report each. A need on another property, even a related one (a seat height beside an overall height, a strap width beside a width), is a new requirement that replaces nothing: an earlier requirement is dropped or replaced only when the line says so about that same property. A replaced requirement is both dropped (the old) and a new need (the new). When the shopper says which of two wishes matters more, the preferred one is a need. A stated budget is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and kindly: folksy, figurative or roundabout words that name a value mean the shopper firmly wants that value; a value is excluded only when the shopper clearly says to avoid it. Words that only stress how firmly the shopper holds a value (only, nothing but, nothing short of, strictly, stays, sticking with, locked to, holding out for) are not part of it: write just the value words ("material: jute", "color: white"), never "<value> only"; only a shopper who also rules out mixes or other values in so many words adds an exclusion. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit. A new budget replaces the old one whether higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference or anything works, put the property asked about in no_preference (when the message asked about two or more properties, only the FIRST one it names). A reply that only says the shopper did not follow or understand the question ("that was rather dense", "you've lost me", "could you put it another way") is this shopper's way of having no preference when the question named the product the property belongs to: then put the property asked about (the FIRST one when the message named several) in no_preference too; when several products are asked for and the question did not say which one, the property is still open (put nothing in no_preference). In a request for several products, write each need "property (product): constraint", the product taken from the line (a line about "toe style of the women's boots" gives "toe style (women's boots): ..."). The assistant's own question is never a requirement: only what the shopper says is; values named as examples in the question do not become needs. A "don't mind" answer frees only the property the assistant's message asked about; it never drops or changes a need the shopper stated earlier, and when the message asked nothing (it only told something), such an answer changes nothing. A need given by pointing at another listing keeps the line's own pointing words as its value ("finish: unlike the one shown on Y's kettle page"); the values the line names for that other listing describe that listing, not the product: they are no needs, drops or freed properties."""
READ_RULES = """- Read each requirement from the listing's field for that property, by the value's own words as this store writes them in that field: a field whose name says the same property counts whatever its spelling (separators, a prefix, a shorter or longer name, or a usual alternative name such as "fit" for a pants fit type, "neckline" for a collar type, "color family" for a colour). When a field carrying the property's own name exists, it decides, and so does a field of that name with one more leading word (pants length for length, clothing material for material): when either one writes the value, the requirement is met. Two fields whose names differ only by separators (size class, size_class) are one field: the requirement is met when either one writes the value. Otherwise such an alternative field decides; only when no field names the property may the options list, the title or the description show the value. A field decides only when this store writes values of the required value's own kind there (colours for a colour, model codes for a model code): when no listing of the category writes a word or figure of the required value in that field, the field names the property another way (car names where a model code is required) and is silent on it, and the listing's own text then decides. The description counts only for what it states about this listing itself, not for a pairing, a comparison or another product.
- Values match by their words, as this store writes them: case, separators, singular and plural and spelling variants are the same; a more specific value that contains the required one meets it ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060); one element of a list is enough. A synonym, an abbreviation, a related word or a shade ("tech wood" for technical wood, "win11" for windows 11, "embroidered" for embroidery, "solid" for plain, ivory or cream for off-white) meets a value only when no listing of the category writes the value itself in that field: when other listings write "matte" as their finish, a "satin" finish does not meet matte. A requirement marked as revealed by the shopper takes its value from the shopper's answer, which may put the store's value in other words: a field value that names the same thing in other words (another everyday name or wording of that same thing) meets it when no listing that meets every other requirement writes the answer's own words (prefer one that does); a different value of the property never does, however close (a nearby kind, a related feature, an amount where the requirement describes a quality). A value joined with another by a plus ("wool + nylon") does not meet one of its parts when the field also carries the other part alone ("nylon"). FIELD EVIDENCE lines, when given, are computed from this store's records and say which case holds. A different value of the property does not (quartz is not mechanical, rosewood is not maple), nor a word that only looks alike. A value that claims to cover every value of the property without naming any ("suitable for all occasions", "all seasons", "universal") names none of them and does not meet a required value; a list that also names the required value still meets it. A family, series or range written as a pattern does name its members: "every 48xx model", "the 48xx series" or "4800-4899" includes 4820 and meets it, while a family or range that leaves the value out (every 61xx model for 4820) fails it. A field that names another value of the property does not meet the required value just because the two often go together or one tends to imply the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); only a shade usually sold as the required colour, or a more specific value that contains it, counts.
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
Answer with ONE JSON object, nothing else: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement the records show NOT met], "unshown": [short strings, each naming a requirement no record of the item states either way]}. Give fits=false only for a listed requirement that the item's fields clearly fail under these rules (a FIELD EVIDENCE or COMPATIBILITY line saying the item does NOT meet a requirement is such a failure; a COMPATIBILITY line saying it fits settles whether it works with that product): a problem; or that neither a field nor (when no field of that kind exists for it) the options list, the title or the description states: unshown. Do not fail an item for a missing record of stock or price."""
RANK_CHECK = """A shopping assistant is about to order an item for a request that asks for the PRIORITY. Below are the requirements, the chosen item and the listing records seen. Using the reading rules below (price times quantity within the budget; not shown out of stock), name the listings that may meet every requirement AND beat the chosen item on the priority (a lower price for the cheapest; a better value of the measure for the best). Leave out listings the records show failing a requirement, out of stock or over budget. Figures in a ROW's title (such as 55 x 94) do not say which dimension each one is, so they never rule a ROW out.
READING RULES:
""" + READ_RULES + """
Answer with ONE JSON object, nothing else: {"better": [ids, best first, at most 4], "reason": "one sentence"}."""
ASK_PLAN = """You plan the questions a shopping assistant sends to a shopper. The shopper holds hidden needs: each is a value of one property of the product, revealed only when a message asks about that exact property; the first thing a message asks is answered most reliably, and questions about anything else get a polite "don't mind". Choose up to QUESTION_COUNT different properties, most promising first:
0. first, anything listed below as MUST BE LEARNED FROM THE SHOPPER that has not been asked; ask it plainly. For the budget, ask the most they want to spend. For a choice between two wishes, name both values with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?";
1. then properties the listings' details below actually carry. Read the story first (STORY HINTS below, when given, name its details and the property each concerns: they come first): the hidden need is usually the property its concrete details concern without naming the value. What the user will touch or press, see up close or in photos, hear, measure or fit it against, read on the label, or how it was made points at that part or aspect of the item and its property: tags, labels or the fine print point to the material, fabric or ingredients; photos and looks to the pattern, colour or the visible part; a body part or a space it must fit to that part's shape or size; buying for someone else, or someone barely known, to whom it is made for (gender, age group); how it must feel to the hand, skin or foot first to the material or fabric; where, when or for what activity it is worn or used (a gym class, the office, outdoors) to its style or type, the property whose values name such uses; something done up or adjusted again and again to its closure or mechanism; what does the work inside it to its core component. Weather, a job, a schedule or a place that is not where the item is worn or used are background. Ask first about the property the story's details point at, by the name the listings use for it. Whom the item is made for is a candidate like any other: ask it when the story points at someone else, or when nothing else is hinted. The keys come best first (most listings carry them; among equally carried ones, the one whose values differ most); among the listed keys, prefer one marked as differing among the listings read (a hidden need tells listings apart; a key every listing writes the same cannot), give a key marked as figures with a unit (a size, width, height, length, weight) a slot of its own whenever one exists, prefer a key most of that product's opened listings carry, and on a request for several products the product whose fitting listings are still many (a hidden need must cut the set), and never spend a slot on an option or variant picker or on a numbered duplicate of another key.
Never pick a property the request already settles (its requirements, the price, stock, quantity), never repeat one already asked, never spend two questions on near-duplicates (two kinds of style, two spellings of one property), and never ask about brand, maker, a product or model name, price, stock or delivery. Ask only about a property that appears as a key in the listings' details below, by the name the listings use for it; a property under STORY HINTS that no key below matches may be asked by its plain name, unless the hint is marked [no key] (enough of its product's listings were read without that field): then ask the key below whose values differ the way its detail describes, or leave it. Each step-1 question is one short, plain, friendly sentence (at most 15 words) asking which <property> they would like for the <item kind>, naming the property by the listings' own name for it (example values alone never reveal a need), with no example values (a list of values can read as several things asked). No lead-in, no status report, no instructions or lists, and never a catch-all key such as options, variants, a description, a title or a name. Name the property by the exact key one of that product's listings below writes (copied as written). Each question asks about ONE property of ONE product (the step-0 choice between two wishes aside): never two properties in one question and never properties of two products, since a reply reveals at most one need and a question naming two subjects may get no answer at all. In "second", always give the next most promising property of the SAME product (another key of that product's listings, not another name for the first), or leave it empty only when none is left: the assistant asks it next, in a message of its own.
When the request asks for SEVERAL products, the hidden needs may sit on any of them: spread the questions so every product gets one before any gets a second (a product with no property left aside); when the other products must work with, fit or suit one of them, that main product gets the first question only, then each other product gets one. Take each product's properties from its own category's list below, write the property as "property (product)", and name the product in the question ("Which toe style would you like for the boots?").
Answer with ONE JSON object, nothing else: {"questions": [{"property": "...", "second": "... or empty", "question": "..."}, ...]}."""
LIST_NOTE = """
This message may name several properties of ONE product: the shopper answers about the first one named most reliably, so the first question's property stays the single most promising. In each question's object also give "more": up to 7 other properties of the SAME product that might hold its hidden need, most promising first, each a key of that product's listings below copied as written (or a story hint's plain name), none settled by the request or asked before, and none the rules above keep out."""
NEXT_ONE = """
This time only ONE question is sent; the shopper's answer is read before the next is chosen. REMAINING question(s) are left in all and every one should be used: put the single most promising property first in the list (one or two more may follow as spares); after a revealed need, the next question goes to a different property and story detail."""
BEFORE_DECLINE = """A shopping assistant is about to tell the shopper that nothing in the store fits; that is rarely the right finish. Check the listing records below against the requirements by these reading rules (the price must be within the budget and the item must not be shown out of stock):
""" + READ_RULES + """
Some listings below are known only from search rows (no details read, so their property fields are unknown). Such a row on the requested category (whatever object its title names) whose title, price and stock do not show it failing a requirement has not been checked (figures in a title, such as 55 x 94, do not say which dimension each one is, so they never show a row failing): list it under "unopened", the likeliest first (its title naming a required value, or the requested kind with nothing against it).
Answer with ONE JSON object, nothing else: {"possible_fits": [ids of listings that seem to meet every requirement, best first, at most 5], "unopened": [ids of rows-only listings to open before declining, at most 6], "reason": "one sentence"}."""
REF_HOP = """A shopper's request takes the PROPERTY of the product it asks for from another listing in the store, described as: DESCRIPTION. Below are the records of listings that may be that listing. Answer with ONE JSON object, nothing else, holding:
"target": the id of the record that IS the described listing: sold by that seller, of that kind, and showing every detail named in its fields, title or its own text, read by meaning (a figure given in another unit counts when it is the same amount); null when none of them is.
"missing": the details the target does not show ([] when it shows them all).
"go_to": when the target's own text sends the reader to a DIFFERENT listing for the PROPERTY (take it from, use, refer to, look at or go by another listing), that listing as {"seller": its shop or brand name as written, "kind": its kind of product, "details": [each value the text names for it, short and as written: a colour, a size with its unit, a material]}; else null.
"value": when there is no go_to, the target's own value of the PROPERTY, as its record writes it: its field named for the PROPERTY; when no field names it, the words its own text gives for the PROPERTY ("a frame with a brushed steel finish" gives "brushed steel" for finish), never another field's value.
"goes_with": the values the target's text says go together with, pair with or count the same as that value ([] when none).
A listing the text names only as what the target is not ("not the green one from X"), or as a version, size or colour sold elsewhere, is no go_to; nor is one it names for another property. A record that shows the described details nowhere is not the target, whatever its text says; one whose own text describes it with those details and then sends the reader elsewhere for the PROPERTY ("go by the X one, not this item") IS the target, with that go_to."""
CHANGE_REVIEW = """A shopping errand's conditions just changed: a store notice, a shopper line or a refused finish announced a new budget, a dropped or added requirement, or an item gone. Below are the request, what changed, the requirements now in force, the priority, the cart and one card per listing seen so far (a DETAILS card carries the full record). Re-judge the OPENED listings against the conditions as they stand NOW, by these reading rules:
""" + READ_RULES + """
Answer with ONE JSON object, nothing else: {"fits": [ids of opened listings that meet every requirement now in force, are within the budget and are not shown out of stock, best first (the cheapest first when the cheapest is asked for)], "pick": the id to order now, or null when none fits, "cart_still_right": true when the cart's line is among the fits and is the right pick (a basket that still fits is kept), false when the cart holds a line that no longer fits or a better pick exists under a cheapest or best rule, null when the cart is empty, "reason": "one sentence"}."""
WORLD_READ = """Below are listing records exactly as one store returned them (each store names its fields its own way). Name the key of these records that holds each role, copied exactly as written, or null when no key holds it. Answer with ONE compact JSON object, nothing else: {"id": the listing's own id or code (one per listing, what a tool takes to look it up or buy it), "title": the listing's own title or headline, "price": its price (a number), "stock": the key that says whether it can be bought now, "stock_means": "in_stock" when true or a count above zero means it can be bought now, "sold_out" when true means it cannot, "text" when the value is words, "category": the store's category or shelf label of the listing, "seller": the shop or seller offering it, "brand": its maker or brand}. Never a key of a nested object."""
FIT_READ = """Reply with the JSON object at once, without deliberating. Below are store listings, each under its id: its title, then the sentences of its text and fields that hold codes. Read which products each listing says it fits, works with or is compatible with. A code the listing gives as the name of its own product, with no words saying it fits something, goes in "own" and is no statement of fit. Answer with ONE compact JSON object, nothing else, keyed by listing id: {"<id>": {"own": [the codes that name this listing's own product - its model, part, item or series number - copied exactly as written; [] when none], "claims": [one object per statement of fit: {"about": "self" when the statement is about this listing's own product, else the code of the other product it is about, as written, "fits": [each model or code it names as fitting, as written], "families": [each group it names as fitting by a pattern or a range, with "?" for each character the pattern leaves open ("every KX-7xx unit" gives "KX-7??") or as "FIRST..LAST" for a range ("models 210 to 260" gives "210..260")], "except": [each model or code it names as NOT fitting, as written], "see": [each code of another product it sends the reader to for more of its list, as written]}]}}. A statement that names another product's code and says what that product fits is about that product, not this listing. A listing that states no fit has "claims": []. Never write a code the text does not hold."""
FIT_CODES = """Reply with ONE compact JSON object at once, without deliberating. Below is a shopper's request. List each code - a model, part or series number written with letters and digits - that the request says a product it asks for must fit, suit or be compatible with: {"fits": [{"product": that product as the request names it, "code": the code copied exactly as the request writes it}]}. Give {"fits": []} when the request names no such code. A quantity, price, size, year, budget or other measurement is not a code, and neither is a value the product itself must have."""
FINISH_CHECK = """You are the last check before a shopping assistant finishes an errand with an order or a decline. (Before an order, when no message went to the shopper after the last cart change, the code tells the shopper the basket with its total by itself; list that only when a stated step asks for something more or different, such as a question to answer or a wait for a yes.) The request, the shopper's later lines and the store's notices may state steps the assistant must carry out, some of them before or at the finish: confirming with the shopper first, telling them what is in the basket or the total, a closing remark, a last look at something. Read the instructions in force, the newest shopper lines and notices, and what was done after the last cart change, then decide whether any stated step is still undone before THIS finish. A step about the basket done before the last cart change does not count as done. Invent nothing: only a step the request, the shopper or the store actually stated, and only one of the listed tools can carry it out. Answer with ONE JSON object, nothing else: {"pending": [one entry per undone step, in the order to run them: {"why": the stated step in a few words, "tool": the exact name of the tool that carries it out, "args": its arguments exactly as that tool's schema asks; when the step is a message to the shopper, write it in full as one or two short plain sentences carrying the facts the step asks for (for the basket: each cart line with its id, its count and its price, and the total)}], "ready": true when nothing is pending, else false}. At most two entries; a step no listed tool can carry out is left out."""
DECLINE_WORDS = (r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without|skip|stop|quit|"
                 r"exit|abort|forgo|halt|bail|walk.?away|\bend\b|\bpass\b")
ORDER_WORDS = r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed"
SORRY = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
ID_LIKE = r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}"
ID_KEYS: set[str] = set()
ID_HINTS = r"sku|code|ref|key|handle|slug|uid|ean|upc|asin|serial|number|\bno\b"
def id_rank(key: Any) -> int | None:
    name = str(key)
    low = name.lower()
    if low in ("id", "_id"):
        return 0
    if low.endswith("id") or name in ID_KEYS:
        return 1
    return 2 if "id" in re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower()) else None
def is_id_key(key: Any) -> bool:
    return "id" in str(key).lower() or str(key) in ID_KEYS
def learn_id_keys(obs: Any) -> None:
    def shape(val: str) -> str:
        return re.sub(r"[A-Za-z0-9]", "x", val)
    def judge(rows: list[dict[str, Any]]) -> None:
        common = set(rows[0])
        for row in rows[1:]:
            common &= set(row)
        good = []
        for key in common:
            vals = [row[key] for row in rows]
            if all(isinstance(v, str) and re.fullmatch(ID_LIKE, v) and re.search(r"\d", v) for v in vals) and len(set(vals)) == len(vals):
                good.append(str(key))
        if not good or any(id_rank(k) is not None and k not in ID_KEYS for k in good):
            return
        if len(good) > 1:
            same = [k for k in good if len({shape(str(row[k])) for row in rows}) == 1]
            good = same if len(same) == 1 else [k for k in (same or good) if re.search(ID_HINTS, k.lower())]
        if len(good) == 1:
            ID_KEYS.add(good[0])
    def visit(piece: Any, lvl: int) -> None:
        if lvl > 6:
            return
        if isinstance(piece, list):
            rows = [x for x in piece if isinstance(x, dict)]
            if len(rows) >= 3 and len(rows) == len(piece):
                judge(rows)
            for x in piece[:60]:
                visit(x, lvl + 1)
        elif isinstance(piece, dict):
            for val in piece.values():
                visit(val, lvl + 1)
    try:
        visit(obs, 0)
    except Exception:
        return
def shelf_key(key: Any) -> bool:
    low = str(key).lower()
    return bool(re.search(r"categor|shelf|department|aisle|group|section|collection", low)) \
        and not re.search(r"\b(?:age|size|price|user|customer|target|blood)\b|(?:age|size|price|user|customer|target|blood)[_\s-]", low)
def record_id(d: dict[str, Any]) -> str | None:
    best: tuple[int, str] | None = None
    for key, val in d.items():
        rank = id_rank(key)
        if rank is not None and isinstance(val, str) and re.fullmatch(ID_LIKE, val) and (best is None or rank < best[0]):
            best = (rank, val)
    return best[1] if best else None
CODE_LIKE = r"[A-Za-z0-9_\-]+"
def as_text(piece: Any, ceiling: int = VIEW_CHARS) -> str:
    try:
        text_body = json.dumps(piece, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text_body = str(piece)
    return text_body if len(text_body) <= ceiling else text_body[:ceiling] + "...(cut)"
def clipped(piece: Any, chars: int, lvl: int = 0) -> Any:
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
    if lvl > 4:
        return
    if isinstance(piece, str):
        if len(piece.split()) >= 4:
            collected.append(piece)
    elif isinstance(piece, list):
        for thing in piece[:10]:
            long_strings(thing, collected, lvl + 1)
def walk_dicts(piece: Any, visit, lvl: int = 0, ceiling: int = 6) -> None:
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
    part = re.split(r"\s*\(", str(key))[0]
    words = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower()))
    return bool(re.search(r"\b(?:brand|shop|seller|store|vendor|maker|manufacturer|category|categories|department|title|headline|"
                          r"heading|listing|description|overview|price|cost|amount|availability|sku|url)\b", words)
                or re.fullmatch(r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)", words)
                or shelf_key(part))
def prose(value: Any) -> bool:
    text = str(value or "")
    return len(text.split()) >= 4 and bool(re.search(r"[.!?;:](?:\s|$)", text))
def prop_key(name: Any) -> str:
    part = re.split(r"\s*\(", str(name))[0]
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower())
    return "".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words)
def edits_within(a: str, b: str, most: int) -> bool:
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
    part = re.split(r"\s*\(", str(name))[0]
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]
def spelled_alike(lo: str, hi: str) -> bool:
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
def base_name(prop: Any) -> str:
    return re.split(r"\s*\(", str(prop))[0].strip()
def strict_prop(a: Any, b: Any) -> bool:
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
    return [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text or "").lower()) if w not in CH_SKIP]
def ch_word_in(word: str, pool: list[str]) -> bool:
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
    plain = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", str(value or "").lower()).replace('"', " ")
    plain = re.sub(r"\b(?:only|just|exactly|strictly)\b", " ", plain)
    return [ws for ws in (ch_words(part) for part in re.split(r"\bor\b", plain)) if ws]
def ch_writes(value: Any, text: Any) -> bool:
    return ch_match(ch_alts(value), ch_words(text))
def ch_match(alts: list[list[str]], pool: list[str]) -> bool:
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
    low = str(text or "").strip().lower()
    return [p.strip() for p in re.split(r"[,;|]", low) if p.strip()] or ([low] if low else [])
def ch_sides(part: Any) -> list[str]:
    sides = [s.strip() for s in re.split(r"\s*\+\s*", str(part)) if s.strip()]
    return sides if len(sides) >= 2 else []
def figure_written(fig: str, said: str) -> bool:
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
    return {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", str(text).lower()) if w not in LINK_WORDS}
def kind_tokens(text: Any) -> list[str]:
    return [w for w in re.findall(r"[a-z]{3,}", re.sub(r"'s\b", "", str(text).lower().replace("\u2019", "'"))) if w not in LINK_WORDS]
def plural_forms(word: Any) -> set[str]:
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
    return a == b or str(b) in plural_forms(a) or str(a) in plural_forms(b)
def kind_in(word: Any, pool: Any) -> bool:
    return any(same_kind(word, other) for other in pool)
def kinds_within(words: Any, pool: Any) -> bool:
    return all(kind_in(w, pool) for w in words)
def kind_score(own: Any, words: Any, shared: Any) -> float:
    return sum(0.5 if kind_in(w, shared) else 1.0 for w in own if kind_in(w, words))
def name_pattern(name: Any) -> str:
    pats = []
    for w in kind_tokens(name):
        forms = sorted(plural_forms(w), key=lambda x: (-len(x), x))
        pats.append(r"(?:" + "|".join(re.escape(x) for x in forms) + r")(?:'s\b|'|\b)")
    return r"[\s'-]+".join(pats)
VALUE_LINKS = r"at least|at most|no more than|no less than|more than|less than|up to|or more|or less|or larger|or smaller|" \
              r"or higher|or lower|minimum|maximum|between|within|under|over|above|below|exactly|only|either|neither|nor|than|plus"
def one_prop_hint(hint: str) -> str:
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
PAIR_WORDS = (r"\bworks?\s+(?:alongside|together\s+with|with)\b|\bgo(?:es)?\s+(?:together\s+)?with\b|\bpairs?\s+(?:up\s+)?with\b"
              r"|\bpaired\s+with\b|\bteams?\s+up\s+with\b|\balongside\b")
MOST_WORDS = (r"\bat most\b|\bno (?:more|higher|greater|bigger|larger|longer|heavier|wider|taller) than\b|\bnot (?:more|higher|above|over)\b"
              r"|\bup to\b|\bor (?:less|lower|smaller)\b")
LEAST_WORDS = (r"\bat least\b|\bno (?:less|lower|smaller|shorter|lighter|fewer|narrower) than\b|\bnot (?:less|lower|below|under)\b"
               r"|\bor (?:more|higher|larger)\b")
REFUSE_WORDS = (r"\b(?:binned|rejected|refused|ruled out|out of the question|a no-?go|off the table|turned away|thrown out|no room for|"
                r"deal-?breaker|won'?t do|will not do|not (?:wanted|welcome|acceptable|an option))\b")
POINT_WORDS = (r"\bwithin\b|" + MOST_WORDS + "|" + LEAST_WORDS + "|" + PAIR_WORDS
               + r"|\bsame as\b|\bas shown\b|\bthe value\b|\bthe one\b|\bthe (?:type|kind|sort) (?:that|which)\b")
REF_DIFFER_STEPS = 0.3
REF_DIFFER_BLIND = 5
SELLER_KEYS = r"shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by"
def ref_text(value: Any) -> bool:
    return bool(re.search(REF_WORDS, str(value or "").lower()))
def ref_toks(text: Any) -> list[str]:
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
    return re.sub(r"[^a-z0-9]", "", re.sub(r"'s\b", "", str(text or "").lower()))
def ref_double(text: Any, at: int) -> bool:
    low = str(text or "").lower()
    heads = [m.end() for m in re.finditer(r"[.!?](?:\s|$)", low[:at])]
    tail = re.search(r"[.!?](?:\s|$)", low[at:])
    start = heads[-1] if heads else 0
    before, after = low[start: at], low[at: at + tail.start()] if tail else low[at:]
    if re.search(r"\bnothing\b|\bno other\b|\bnone other\b", before):
        return True
    if not re.search(r"\b(?:any|anything|something|everything)\b", low[start: at + 9]):
        return False
    return bool(re.search(r"\b(?:don'?t|do not|never|won'?t|will not|can'?t|cannot|refuse|no way)\b", before) or re.search(REFUSE_WORDS, after))
def ref_relation(value: Any) -> tuple[str, float | None, str]:
    low = str(value or "").lower()
    neg = re.search(DIFFER_WORDS, low)
    if neg and ref_double(low, neg.start()):
        low, neg = low[neg.end():], None
    body = low[neg.end():] if neg else low
    point = re.search(REF_WORDS + r"|\bwhatever\b", body)
    body = body[:point.start()] if point else body
    rel, tol, unit = "same", None, ""
    hit = re.search(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?", body)
    if hit:
        rel, tol, unit = "within", float(hit.group(1)), (hit.group(2) or "").strip()
        unit = "" if unit in LINK_WORDS or unit in ("either", "each") else unit
    elif re.search(MOST_WORDS, body):
        rel = "at most"
    elif re.search(LEAST_WORDS, body):
        rel = "at least"
    elif re.search(PAIR_WORDS, body):
        rel = "pairs"
    if neg:
        return ("differ", None, "") if rel == "same" else ("not " + rel, tol, unit)
    return rel, tol, unit
def ref_negation(prop: Any, source: Any, seller: Any) -> str:
    try:
        text = str(source or "")
        words = [w for w in re.findall(r"[a-z0-9]+", re.split(r"\s*\(", str(prop))[0].lower()) if w not in LINK_WORDS]
        hit = re.search(r"\b" + r"\W+".join(map(re.escape, words)) + r"\b", text.lower()) if words else None
        name = str(seller or "").strip().lower()
        if not hit or not name:
            return ""
        span = ref_span(prop, text).lower()
        cut = span.find(name)
        neg = re.search(DIFFER_WORDS, span[:cut]) if cut > 0 else None
        if not neg:
            return ""
        return "double" if ref_double(text, hit.end() + neg.start()) else "differ"
    except Exception:
        return ""
def ref_parse(text: Any, ok: Any = None) -> dict[str, Any] | None:
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
    try:
        words = [w for w in re.findall(r"[a-z0-9]+", re.split(r"\s*\(", str(prop))[0].lower()) if w not in LINK_WORDS]
        hit = re.search(r"\b" + r"\W+".join(map(re.escape, words)) + r"\b", str(source).lower()) if words else None
        if not hit:
            return ""
        return re.split(r"\(\s*or\b|(?<=[.!?])\s", str(source)[hit.end(): hit.end() + 500])[0]
    except Exception:
        return ""
LENS: dict[str, Any] = {}
def lens_alias(d: dict[str, Any]) -> None:
    if not LENS:
        return
    names = {str(key).lower(): key for key in d}
    def has(pattern: str, test: Any = None) -> bool:
        return any(re.search(pattern, low) and (test is None or test(d[key])) for low, key in names.items())
    def take(role: str, test: Any) -> Any:
        key = LENS.get(role)
        return key if isinstance(key, str) and key in d and test(d[key]) else None
    def numeric(val: Any) -> bool:
        return is_number(val) and not isinstance(val, bool)
    key = take("title", lambda v: isinstance(v, str))
    if key and not has(r"title|heading|headline") and "title" not in d:
        d["title"] = d[key]
    key = take("price", numeric)
    if key and not has(r"price|cost", numeric) and "price" not in d:
        d["price"] = d[key]
    key = take("stock", lambda v: v is not None)
    if key and not has(r"stock|avail|on.?hand"):
        mode = LENS.get("stock_means")
        alias = "out_of_stock" if mode == "sold_out" else ("availability" if mode == "text" else "in_stock")
        d.setdefault(alias, d[key])
    key = take("category", lambda v: isinstance(v, str) and 0 < len(v.strip()) <= 80)
    if key and not any(shelf_key(k) for k in d) and "category" not in d:
        d["category"] = d[key]
    key = take("seller", lambda v: isinstance(v, str))
    if key and not has(r"shop|seller|store|vendor|merchant|retailer") and "seller" not in d:
        d["seller"] = d[key]
    key = take("brand", lambda v: isinstance(v, str))
    if key and not has(r"brand|maker|manufactur") and "brand" not in d:
        d["brand"] = d[key]
def lensed(piece: Any) -> Any:
    try:
        piece = json.loads(json.dumps(piece))
    except (TypeError, ValueError):
        return piece
    def visit(d: dict[str, Any]) -> None:
        if not any(isinstance(val, str) and is_id_key(key) and re.fullmatch(ID_LIKE, val) for key, val in d.items()):
            return
        lens_alias(d)
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
class SendFault:
    def __init__(self, why: str):
        self.data = None
        self.error = {"status": None, "body": why[:300]}
class Wire:
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
        gap = SLOW_GAP if monotonic() < self.slow_until else PACE_GAP
        wait = self.last_send + gap - monotonic()
        if "/inference/" in path:
            wait = max(wait, self.model_resume - monotonic())
        wait = min(wait, self.spare() - 1)
        if wait > 0:
            sleep(wait)
        self.last_send = monotonic()
        try:
            answer = client.post_verbose(path, body)
        except Exception as err:
            answer = SendFault(f"{type(err).__name__}: {err}")
        if answer.data is None and (answer.error or {}).get("status") in (429, 503):
            self.slow_until = monotonic() + SLOW_FOR
        return answer
    def pause(self, seed: str, tries: int, status: Any) -> float:
        mix = (len(seed) * 7 + sum(ord(ch) for ch in seed[-64:]) + 31 * tries + 13 * self.tick) % 101 / 100.0
        return min(8.0, (2.0 if status == 429 else 1.0) * 2 ** tries) * (0.75 + 0.5 * mix)
    def down(self, at: float | None = None) -> bool:
        return self.no_model or (self.misses > 0 and (monotonic() if at is None else at) - self.last_reply > DOWN_AFTER)
    def usable(self, model_list: list[str], tools: bool = False) -> bool:
        return not self.no_model and not self.mute and any(
            mdl not in self.banned and not (tools and mdl in self.tool_banned) for mdl in list(model_list) + self.spare_models)
    @staticmethod
    def plain(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            if self.spend >= HARD_SPEND * max(self.__dict__.get("scale", 1.0), self.__dict__.get("long", 1.0)) and not self.no_model:
                self.no_model = True
                self.log.append({"kind": "spend_stop", "spend": round(self.spend, 4)})
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
        self.call_fail = "empty" if "empty" in kinds else "refused" if kinds == {"hard"} else ""
        return None
    @staticmethod
    def cached(msgs: list[dict[str, Any]], mdl: str) -> list[dict[str, Any]]:
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
    def __init__(self):
        self.items: list[Any] = []
        self.ids: set[str] = set()
        self.read: set[str] = set()
        self.label_of: dict[str, str] = {}
    def add(self, obs: Any) -> None:
        self.items.append(obs)
        learn_id_keys(obs)
        def visit(d: dict[str, Any]) -> None:
            for key, piece in d.items():
                if isinstance(piece, str) and is_id_key(key) and re.fullmatch(ID_LIKE, piece):
                    self.ids.add(piece)
            ident = record_id(d)
            if ident and any(isinstance(val, dict) and sum(1 for v in val.values() if isinstance(v, (str, int, float))) >= 2
                             for val in d.values()):
                self.read.add(ident)
            own = next((piece.strip() for key, piece in d.items() if isinstance(piece, str) and 0 < len(piece.strip()) <= 60
                        and shelf_key(key) and not is_id_key(key)), "")
            if ident and own:
                self.__dict__.setdefault("label_of", {}).setdefault(ident, own)
        walk_dicts(obs, visit)
    def relens(self) -> None:
        old = self.items
        self.items, self.ids, self.read, self.label_of = [], set(), set(), {}
        for obs in old:
            self.add(lensed(obs))
    def price(self, thing: str) -> float | None:
        found: list[float] = []
        def visit(d: dict[str, Any]) -> None:
            if not any(isinstance(val, str) and val == thing and is_id_key(key) for key, val in d.items()):
                return
            for key, val in d.items():
                if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower()):
                    found.append(float(val))
                    return
        for rec in self.items:
            walk_dicts(rec, visit)
        return found[-1] if found else None
    def about(self, thing: str, hold: int = 4) -> list[str]:
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
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and is_id_key(key)
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
        own = next((val.strip() for key, val in d.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                    and shelf_key(key) and not is_id_key(key)), "")
        if own:
            return own
        labels = self.__dict__.get("label_of") or {}
        return next((labels[val] for key, val in d.items() if isinstance(val, str) and val in labels and is_id_key(key)), "")
    def rich_reads(self, shelf: str = "") -> int:
        rich: set[str] = set()
        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and is_id_key(key) and val in self.read), None)
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
        tally: dict[str, int] = {}
        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and is_id_key(key) and val in self.read), None)
            shelf = self.label_for(d) if ident else ""
            if ident and shelf and any(isinstance(val, dict) for val in d.values()):
                tally[shelf] = tally.get(shelf, 0) + 1
        for rec in self.items:
            walk_dicts(rec, visit)
        return sorted(tally, key=lambda s: -tally[s])
    def prop_menu(self, shelf: str = "") -> tuple[str, int]:
        tallies: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        counts: dict[str, dict[str, int]] = {}
        said: dict[str, str] = {}
        sets = 0
        mapped, row_tally = 0, {}
        def on_shelf(d: dict[str, Any]) -> bool:
            return any(isinstance(val, str) and val.strip().lower() == shelf.lower() and shelf_key(key)
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
            brief_vals = [(fkey, val) for fkey, val in d.items() if isinstance(val, str) and 0 < len(val) <= 70 and not junk_key(fkey)
                          and not prose(val)]
            if len(brief_vals) < 5 and not (shelf and len(brief_vals) >= 2 and not any(isinstance(v, dict) for v in d.values())):
                return
            sets += 1
            rowlike = len(brief_vals) < 5 and any(isinstance(val, str) and "id" in str(fkey).lower() for fkey, val in d.items())
            mapped += 0 if rowlike else 1
            for key, piece in brief_vals:
                if re.search(r"\d", piece) and re.fullmatch(CODE_LIKE, piece) and ("-" in piece or "_" in piece):
                    continue
                call_name = re.sub(r"[_.]+", " ", str(key)).strip().lower()
                said.setdefault(call_name, re.sub(r"\s+", " ", re.sub(r"([a-z])([A-Z])", r"\1 \2", re.sub(r"[_.]+", " ", str(key)))).strip().lower())
                tallies[call_name] = tallies.get(call_name, 0) + 1
                if rowlike:
                    row_tally[call_name] = row_tally.get(call_name, 0) + 1
                held = counts.setdefault(call_name, {})
                held[piece.lower()] = held.get(piece.lower(), 0) + 1
                value_counts = examples.setdefault(call_name, [])
                if piece.lower() not in value_counts and len(value_counts) < 5:
                    value_counts.append(piece.lower())
        for rec in self.items:
            seek(rec, False, 0)
        if mapped:
            for fkey in [fkey for fkey in tallies if row_tally.get(fkey, 0) >= tallies[fkey]]:
                tallies.pop(fkey)
        for fkey in list(tallies):
            words = fkey.split()
            held = counts.get(fkey) or {}
            # (a key whose first words are another key is that key spelled longer only when they share a value: a key with
            # its own values, such as "<word> <other word>" beside "<word>", is a property of its own)
            if any(" ".join(words[:n]) in tallies and tallies[" ".join(words[:n])] >= max(2, tallies[fkey] // 2)
                   and set(examples.get(" ".join(words[:n])) or []) & set(examples.get(fkey) or []) for n in range(1, len(words))) \
                    or (examples.get(fkey) and all(re.fullmatch(r"\d{4,}", v) for v in examples[fkey])) \
                    or (tallies[fkey] >= 3 and len(held) == sum(held.values()) and sum(len(v.split()) for v in held) >= 4 * len(held)):
                tallies.pop(fkey, None)
        def spread(fkey: str) -> int:
            return len(examples.get(fkey, []))
        def figured(fkey: str) -> bool:
            vals = examples.get(fkey, [])
            return bool(vals) and all(re.search(r"\d", v) for v in vals) and any(re.search(r"\d\s*[a-z\"']", v) for v in vals)
        def score(fkey: str) -> float:
            held = counts.get(fkey) or {}
            seen = sum(held.values())
            top = max(held.values()) / seen if seen else 1.0
            return min(1.0, tallies[fkey] / max(1, sets)) * (1.0 - top)
        # most carried first, then the one whose values differ most (a preference has to tell listings apart)
        top_props = sorted(tallies, key=lambda fkey: (-tallies[fkey], -len(counts.get(fkey) or {}), fkey))[:28]
        rows = []
        for fkey in top_props:
            marks = f" [values differ among the listings: {spread(fkey)} seen]" if spread(fkey) >= 2 and tallies[fkey] >= 2 else ""
            marks += " [figures with a unit]" if figured(fkey) else ""
            rows.append(f"- {said.get(fkey) or fkey} (in {tallies[fkey]} listings){marks}: " + " | ".join(examples.get(fkey, [])))
        return "\n".join(rows), sets
    def label_lines(self) -> str:
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
NO_CARE_LABEL = "shopper does not care about"
NO_LONGER_LABEL = "no longer required"
PRICE_LABEL = "total price at most"
class Needs:
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
        self.told: list[dict[str, Any]] = []
        self.told_version = 0
        self.allowance_news: dict[str, Any] | None = None
    def add_told(self, items: Any, source: str) -> list[str]:
        added: list[str] = []
        if not isinstance(items, list):
            return added
        known = {" ".join(str(t.get("text") or "").lower().split()) for t in self.told}
        for item in items[:12]:
            if isinstance(item, str):
                item = {"text": item, "before_finish": None, "needs_message": None}
            if not isinstance(item, dict):
                continue
            if item.get("invites_question") is True and item.get("needs_message") is not False:
                # an invitation to ask the shopper something is no instruction to tell them anything: kept, it would hold
                # back one of the few messages for a basket report nobody asked for
                continue
            words = " ".join(str(item.get("text") or "").split())
            if not words or words.lower() in known:
                continue
            known.add(words.lower())
            flags = {k: (item.get(k) if isinstance(item.get(k), bool) else None)
                     for k in ("before_finish", "needs_message", "before_change", "on_order", "conditional")}
            self.told.append({"text": words[:400], "source": source, **flags})
            added.append(words[:120])
        if added:
            self.told_version += 1
        return added
    def finish_told(self) -> list[dict[str, Any]]:
        return [t for t in self.told if t.get("before_finish") is not False]
    def told_text(self) -> str:
        if not self.told:
            return ""
        rows = []
        for n, t in enumerate(self.told[-10:]):
            tags = [tag for tag, on in (("before the finish", t.get("before_finish")), ("by a message", t.get("needs_message"))) if on]
            rows.append(f"{n + 1}) {t['text']}" + (f" [{', '.join(tags)}; from the {t.get('source')}]" if tags else f" [from the {t.get('source')}]"))
        return "\n[instructions in force] " + " ".join(rows)
    def items(self) -> list[dict[str, Any]]:
        rows = []
        for row in self.reading.get("items") or []:
            if isinstance(row, dict) and str(row.get("product") or "").strip():
                qty = row.get("quantity")
                reqs = row.get("requirements") if isinstance(row.get("requirements"), list) else []
                dep = row.get("depends_on")
                rows.append({"product": str(row["product"]).strip(), "quantity": qty if isinstance(qty, int) and not isinstance(qty, bool) and qty >= 1 else 1,
                             "requirements": [str(r) for r in reqs if r and ":" in str(r)]})
                rows[-1]["depends_on"] = dep - 1 if isinstance(dep, int) and not isinstance(dep, bool) and dep >= 1 else None
        return rows if len(rows) >= 2 else []
    def quantity(self) -> int:
        if self.items():
            return 1
        need_n = self.reading.get("quantity")
        return need_n if isinstance(need_n, int) and not isinstance(need_n, bool) and need_n >= 1 else 1
    def counts(self) -> list[int]:
        return [row["quantity"] for row in self.items()] or [self.quantity()]
    def own_labels(self) -> list[str]:
        def plain(text: str) -> list[str]:
            return re.findall(r"[a-z0-9]+", text.lower())
        labels = [" ".join(plain(x)) for x in (NO_CARE_LABEL, NO_LONGER_LABEL, PRICE_LABEL)]
        head = re.split(r":\s*1\)", self.items_text())[0]
        labels += [" ".join(plain(part)[:5]) for part in re.split(r"[;:]", head) if len(plain(part)) >= 4]
        return labels + [" ".join(plain(f"{row['product']} x{row['quantity']}")) for row in self.items()]
    def items_text(self) -> str:
        rows = self.items()
        if not rows:
            return ""
        kit = any(r.get("depends_on") is not None for r in rows)
        return ("THE REQUEST ASKS FOR SEVERAL PRODUCTS; the order holds exactly one line per product, at its quantity, and nothing "
                "else; each listing is judged only against the requirements of the product it is (its category decides), and "
                + ("the products must work together as the request says, and a cheapest or best rule applies to the whole set "
                   "(its total), not to each product alone: " if kit else
                   "the cheapest or best rule applies within each product: ") + " | ".join(
                    f"{n + 1}) {row['product']} x{row['quantity']}" for n, row in enumerate(rows)))
    def priority(self) -> str:
        return self.reading.get("priority") if self.reading.get("priority") in ("cheapest", "best") else "none"
    def props_left(self) -> list[str]:
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
            collected += f"; {PRICE_LABEL} {self.spend_cap} {self.reading.get('currency') or ''}".rstrip()
        collected += f"; quantity {self.quantity()}" if not self.items() else "; " + self.items_text()
        if self.let_go:
            collected += f"; {NO_LONGER_LABEL}: " + "; ".join(self.let_go)
        if self.indifferent:
            collected += f"; {NO_CARE_LABEL}: " + ", ".join(self.indifferent)
        return collected
    def changes(self) -> str:
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
        said = " ".join(str(ln) for ln in lns).lower()
        for need in needs if isinstance(needs, list) else []:
            head, _, value = str(need).partition(":")
            names = [w for w in prop_words(head) if len(w) >= 3]
            values = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in LINK_WORDS]
            if names and values and all(w in said for w in names + values):
                return True
        return False
    def hear(self, lns: list[str], qtext: str | None) -> str:
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
        labels = self.own_labels()
        def own_label(req: str) -> bool:
            text = " ".join(re.findall(r"[a-z0-9]+", str(req).lower()))
            return any(lab in text for lab in labels)
        if isinstance(now_list, list) and all(isinstance(req, str) for req in now_list):
            copied = [req for req in now_list if own_label(req)]
            if copied:
                self.wire.log.append({"kind": "own_label_dropped", "lines": copied[:4]})
            self.now_reqs = [req for req in now_list if req.strip() and grounded(req) and not own_label(req)]
        for new_need in answer.get("needs") or []:
            if isinstance(new_need, str) and new_need and new_need not in self.uncovered and grounded(new_need) \
                    and not own_label(new_need):
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
        try:
            added = self.add_told(answer.get("instructions"), "shopper" if qtext is not None else "store")
            if added:
                remarks.append("new instruction: " + "; ".join(added))
            allowance = answer.get("allowance")
            if isinstance(allowance, dict) and any(is_number(v) and not isinstance(v, bool) for v in allowance.values()):
                self.allowance_news = {k: v for k, v in allowance.items() if is_number(v) and not isinstance(v, bool)}
                remarks.append("new allowance: " + json.dumps(self.allowance_news))
        except Exception as err:
            self.wire.log.append({"kind": "told_error", "error": f"{type(err).__name__}: {err}"[:300]})
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
        found = self.lines(call_args)
        put = [ln for ln in found if ln[1] is None or ln[1] > 0]
        return (put or found)[0][0] if found else None
    def aim_flat(self, call_args: dict[str, Any]) -> str | None:
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
        if seen is not None and set(seen) <= {thing}:
            self.settle(thing, seen.get(thing, 0.0))
        elif seen is not None:
            self.learn(seen)
        elif (count <= 0 or not self.is_put(call_args)) and self.shows_empty(obs, thing):
            self.units.pop(thing, None)
            self.pending.pop(thing, None)
        return self.summary()
    def shows_empty(self, obs: Any, thing: str) -> bool:
        if not isinstance(obs, dict) or thing not in self.units:
            return False
        lists = [v for v in obs.values() if isinstance(v, list)]
        nums = [v for v in obs.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        text_body = as_text(obs)
        return bool(lists) and all(not v for v in lists) and bool(nums) and all(v == 0 for v in nums) \
            and not any(other in text_body for other in self.units if other != thing)
    def view_units(self, obs: Any) -> dict[str, float] | None:
        rows: dict[str, float] = {}
        known = set(self.units) | set(self.pending)
        def visit(d: dict[str, Any]) -> None:
            ids = [val for val in d.values() if isinstance(val, str) and (val in known or re.fullmatch(ID_LIKE, val)) and " " not in val]
            pairs = [(fkey, val) for fkey, val in d.items() if isinstance(val, int) and not isinstance(val, bool) and 0 < val < 1000
                     and str(fkey) != "price"]
            if len(pairs) > 1:
                named = [(fkey, val) for fkey, val in pairs if re.search(r"qty|quant|count|units|number|total|held|ordered", str(fkey).lower())
                         and not re.search(r"price|cost|due|sum|fee|charge", str(fkey).lower())]
                pairs = named if len(named) == 1 else pairs
            if len(pairs) > 1 and ids:
                ident0 = next((i for i in ids if i in known), ids[0])
                held = self.units.get(ident0, self.pending.get(ident0))
                same = [(fkey, val) for fkey, val in pairs if held is not None and abs(val - held) < 1e-9]
                pairs = same[:1] if len(same) == 1 else [min(pairs, key=lambda kv: kv[1])]
            ints = [val for _, val in pairs]
            if len(ints) == 1 and ids:
                ident = next((i for i in ids if i in known), ids[0])
                rows[ident] = rows.get(ident, 0) + ints[0]
        walk_dicts(obs, visit)
        if not rows or not (set(rows) & (known or set(rows))):
            return None
        return rows
    def settle(self, thing: str, units: float) -> None:
        expected_if_set = self.pending.pop(thing, None)
        if expected_if_set is not None:
            if abs(units - expected_if_set) < 1e-9 and abs(self.units.get(thing, 0) - expected_if_set) > 1e-9:
                self.adds = False
            elif units > expected_if_set:
                self.adds = True
        if units > 0:
            self.units[thing] = units
        else:
            self.units.pop(thing, None)
    def learn(self, seen: dict[str, float]) -> None:
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
LIT_FUNCTION_WORDS = frozenset("a an the of in on at to for with and or must be is are should it its as than any only exactly has have had would need needs want wants that which one all".split())
LIT_EXCLUDES = re.compile(r"^(?:but )?(?:not|no|without|nothing with|none with|anything but|anything except|except|exclude|excluding|avoid|cross off|leave out|never|rule out|skip|rather than|instead of)\s+(?:a |an |the |any )?(.+)$", re.I)
LIT_SPLIT_NOT = re.compile(r"\s*[,;]\s*(?=(?:but )?(?:not|no|without|nothing|none|never|exclude|excluding|avoid|rather than|instead of)\b)", re.I)
LIT_AT_MOST = re.compile(r"\b(?:at most|up to|no more than|not more than|maximum|max|under|below|less than|or less|or under|or below|within)\b", re.I)
LIT_AT_LEAST = re.compile(r"\b(?:at least|no less than|not less than|minimum|min|over|above|more than|or more|or over|or above|or longer|or wider|or higher|or larger|or bigger)\b", re.I)
LIT_NUMBER = re.compile(r"(?<![a-z])[-+]?\d+(?:[.,]\d+)?")
def lit_stem(w: str) -> str:
    for suffix in ("sses", "shes", "ches", "xes"):
        if w.endswith(suffix):
            return w[:-2]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "ie"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w
def lit_words(text: Any) -> list[str]:
    return [lit_stem(w) for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", re.sub(r"'s\b", "", str(text if text is not None else "").lower()).replace("_", " ").replace("-", " "))]
def lit_content(text: Any) -> list[str]:
    return [w for w in lit_words(text) if w not in LIT_FUNCTION_WORDS]
LIT_RELAX_LEAD = re.compile(r"\b(?:scrap|drop|ditch|forget(?: about)?|skip|lose|waive|ignore|relax|loosen|never mind|giving up on|give up on|gave up on|"
                            r"let go of|no need for|don't need|do not need|not fussed about|not bothered about|stop worrying about|stop fussing about|"
                            r"leave out|leave off|do without|strike|set aside|let)\s+"
                            r"(?:the |about |my |that |any )?([a-z][a-z /&'-]{1,40}?)(?=\s*(?:,|\.|;|!|$|\bfrom\b|\baltogether\b|\bgo\b|\bslide\b|\bbe\b|\bwait\b|"
                            r"\bentirely\b|\bcompletely\b|\bfor now\b|\bthen\b|\bnow\b|\bplease\b|\bafter all\b|\boff\b|\bout\b))", re.I)
LIT_RELAX_TAIL = re.compile(r"\b([a-z][a-z /&'-]{1,40}?)\s+(?:is|are|isn't|aren't|was|were)\s+(?:no longer|not)\s+(?:a |an |the )?(?:condition|requirement|must|need|needed|required|"
                            r"necessary|essential|dealbreaker|deal breaker|issue|concern|factor|priority|thing)\b|"
                            r"\b([a-z][a-z /&'-]{1,40}?)\s+(?:can|may|could)\s+(?:slide|go|wait|drop|be dropped|be anything)\b|"
                            r"\b([a-z][a-z /&'-]{1,40}?)\s+(?:doesn't|does not|don't|do not|no longer|needn't)\s+(?:matter|count)\b", re.I)
LIT_RELAX_NOT = re.compile(r"\b(?:don't|do not|never|not|won't|can't|cannot)\s+(?:let|drop|scrap|forget|skip|give up|ditch|ignore|relax|loosen)\b", re.I)
def lit_label_words(key: Any) -> list[str]:
    words = lit_words(key)
    while len(words) > 1 and len(words[0]) <= 2:
        words = words[1:]
    return words
def lit_near(a: str, b: str) -> bool:
    if a == b:
        return True
    lo, hi = sorted((a, b), key=len)
    if len(lo) < 4 or lo[0] != hi[0]:
        return False
    if len(lo) == len(hi):
        diff = [i for i in range(len(lo)) if lo[i] != hi[i]]
        if len(diff) == 1 and {lo[diff[0]], hi[diff[0]]} in ({"a", "e"}, {"i", "y"}, {"a", "y"}):
            return True
    return spelled_alike(lo, hi)
def lit_several(text: Any) -> bool:
    return bool(re.search(r"[,;/+&|]|\band\b", str(text).lower()))
def lit_states(value: str, text: str) -> bool:
    have = set(lit_words(text))
    joined = re.sub(r"[^a-z0-9]+", "", str(text).lower())
    for alternative in re.split(r"\s+or\s+|\s*/\s*", str(value)):
        wanted = lit_content(alternative)
        if wanted and all(any(lit_near(w, h) for h in have) for w in wanted):
            return True
        flat = re.sub(r"[^a-z0-9]+", "", alternative.lower())
        if len(flat) >= 6 and flat in joined:
            return True
    return False
def lit_number(text: str) -> float | None:
    found = LIT_NUMBER.search(str(text).replace(",", ""))
    return float(found.group(0)) if found else None
def lit_split(constraint: str) -> tuple[str, list[str]]:
    text = constraint.replace("(", ", ").replace(")", " ").strip(" ,")
    parts = [p.strip(" ,;") for p in LIT_SPLIT_NOT.split(text) if p.strip(" ,;")]
    positive, excluded = "", []
    for index, part in enumerate(parts):
        found = LIT_EXCLUDES.match(part)
        if found:
            excluded.append(found.group(1))
        elif index == 0:
            positive = part
    return positive, excluded
def lit_judge(requirement: str, title: str, pairs: list[tuple[str, str]], kind: str = "") -> tuple[str, str]:
    prop, _, constraint = requirement.partition(":")
    prop, constraint = prop.strip(), constraint.strip()
    if not prop or not constraint:
        return "ok", ""
    prop_w = [w for w in lit_label_words(prop) if w not in LIT_FUNCTION_WORDS]
    if not prop_w:
        return "ok", ""
    positive, excluded = lit_split(constraint)
    def without_property(text: str) -> str:
        return " ".join(w for w in text.split() if lit_stem(re.sub(r"[^a-z0-9]", "", w.lower())) not in prop_w)
    value = without_property(positive)
    value_words = lit_content(value)
    naming = [(k, v) for k, v in pairs if set(prop_w) <= set(lit_label_words(k))]
    if not naming and not value_words and len(prop_w) >= 2:
        prop_w, value_words = prop_w[-1:], prop_w[:-1]
        value = " ".join(value_words)
        naming = [(k, v) for k, v in pairs if set(prop_w) <= set(lit_label_words(k))]
    for bad in excluded:
        bad = without_property(bad)
        if not lit_content(bad):
            continue
        hit = next(((k, v) for k, v in naming if lit_states(bad, v)), None)
        if hit:
            return "fail", f"{hit[0]}: {hit[1][:60]} states the excluded value"
        if not naming and lit_states(bad, title):
            return "fail", "the title states the excluded value"
    if not value:
        return "ok", "excluded value not stated" if excluded else ""
    number = lit_number(value)
    if number is not None and (LIT_AT_MOST.search(positive) or LIT_AT_LEAST.search(positive)):
        at_most = bool(LIT_AT_MOST.search(positive))
        for k, v in naming:
            found = lit_number(v)
            if found is None:
                continue
            fits = found <= number if at_most else found >= number
            return ("ok" if fits else "fail"), f"{k}: {v[:60]}"
        return "unproven", "no fact for it gives a figure" if naming else "no fact names it"
    for k, v in naming:
        if lit_states(value, v):
            if len(re.findall(r"\d\s*%", v)) >= 2 and "%" not in value:
                return "unproven", f"{k}: {v[:60]} states it only as part of a blend"
            other = next(((k2, v2) for k2, v2 in naming if (k2, v2) != (k, v) and not lit_several(v2) and len(v2.split()) <= 4
                          and not lit_states(value, v2) and set(lit_label_words(k2)) == set(prop_w)), None)
            if other and lit_several(v):
                return "unproven", f"{other[0]}: {other[1][:60]} states another value while {k} lists several"
            return "ok", f"{k}: {v[:60]}"
    if kind and lit_states(value, kind):
        return "ok", f"stated by the kind label ({kind[:40]})"
    if naming:
        wanted = set(value_words)
        single = [(k, v) for k, v in naming if not lit_several(v) and len(v.split()) <= 4]
        if single and len(single) == len(naming) and not any(lit_near(w, h) for _, v in single for w in wanted for h in lit_words(v)):
            return "fail", f"{single[0][0]}: {single[0][1][:60]} states another value"
        return "unproven", f"{naming[0][0]}: {naming[0][1][:60]} does not state it plainly"
    if lit_states(value, title):
        return "ok", "stated in the title (no fact names it)"
    return "unproven", "no fact names it and the title does not state it"
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
        self.list_mode = True    # a message names up to LIST_SIZE properties of one product until the shopper asks for one at a time
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
    @property
    def log(self) -> list[dict[str, Any]]:
        return self.wire.log
    def moves_left(self) -> int:
        return self.step_cap - self.wire.tick
    def role(self, call_name: str) -> str | None:
        return self.role_map.get(call_name)
    def ask_limit(self) -> int:
        return self.msg_cap if self.msg_cap is not None else ASK_DEFAULT
    def msg_reserve(self) -> int:
        if not self.role("message"):
            return 0
        return 1 if any(t.get("needs_message") and t.get("before_finish") is not False and t.get("on_order") is not False
                        and t.get("conditional") is not True for t in self.needs.told) else 0
    def report_asked(self) -> bool:
        return any(t.get("needs_message") and t.get("on_order") is not False for t in self.needs.told)
    def ask_room(self) -> int:
        return max(0, self.ask_limit() - self.msg_reserve())
    def step_reserve(self) -> int:
        return FINISH_STEPS if self.role("message") or self.needs.finish_told() else 0
    def apply_allowance(self, news: dict[str, Any]) -> None:
        before = (self.msg_cap, self.step_cap)
        total, left = news.get("messages_total"), news.get("messages_left")
        if is_number(total):
            self.msg_cap = max(self.msgs_sent, int(total))
        elif is_number(left):
            self.msg_cap = self.msgs_sent + max(0, int(left))
        total, left = news.get("steps_total"), news.get("steps_left")
        if is_number(total) and int(total) > self.wire.tick:
            self.step_cap = min(self.step_cap, int(total))
        elif is_number(left) and int(left) >= 0:
            self.step_cap = min(self.step_cap, self.wire.tick + int(left))
        if (self.msg_cap, self.step_cap) != before:
            self.asking = self.asking and self.msgs_sent < self.ask_room()
            self.log.append(dict(kind="allowance", news=news, caps=[self.msg_cap, self.step_cap]))
    def call_kind(self, call_name: str, call_args: dict[str, Any] | None) -> str | None:
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
        text_body = "; ".join(pieces) + ".\n[requirements now] " + self.needs.text_body() + self.ref_line() + self.product_steps()
        text_body += self.needs.told_text()
        try:
            text_body += self.bound_hint()
            text_body += self.dependency_hint()
            text_body += self.basket_ledger()
            text_body += self.kit_push()
        except Exception:
            pass
        if self.msg_reserve() and self.msgs_sent < self.ask_limit():
            text_body += " (the planned questions leave one message for the stated finishing step: yours to send, or sent for you at the finish)"
        usable = [tdef["function"]["name"] for tdef in self.offered()]
        if len(usable) < len(self.toolset):
            text_body += "\n[tools usable now] " + ", ".join(usable) + " (any other tool call is refused without using a step" + (
                "; hidden-need questions are also planned and sent for you" if self.asking else "") + ")."
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
    def free_text_keys(self, tool: str | None) -> list[str]:
        props = (self.shapes.get(tool or "") or {}).get("properties") or {}
        return [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")
                and not re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(k).lower())]
    def fix_search_role(self) -> None:
        search = self.role("search")
        if search and len(self.free_text_keys(search)) == 1:
            return
        # a search tool with several text fields stays unless another tool's one text field reads as a query; a field
        # that takes a listing's id or code never searches, and a tool holding another role is never the search tool
        has_text = bool(search and self.free_text_keys(search))
        taken = {self.role(r) for r in ("details", "view", "message", "cart", "cart_view", "order", "decline")} - {None}
        best = None
        for name, jschema in self.shapes.items():
            if name in taken or not isinstance(jschema, dict):
                continue
            props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
            if any(isinstance(p, dict) and p.get("type") == "array" for p in props.values()):
                continue
            texts = self.free_text_keys(name)
            needed = [k for k in (jschema.get("required") or []) if k not in texts]
            if len(texts) != 1 or needed or is_id_key(texts[0]) or re.search(r"code|number|sku|listing|ident|\bref", texts[0].lower()):
                continue
            score = (texts[0] in (jschema.get("required") or []), bool(re.search(r"^q$|query|text|keyword|needle|term|word|phrase|string|search", texts[0].lower())))
            if has_text and not score[1]:
                continue
            if best is None or score > best[0]:
                best = (score, name)
        if best is None and not search:
            # no reading named a search tool and none has a single text field: a tool whose text field reads as a query
            for name in self.shapes:
                if name not in taken and any(re.search(r"^q$|query|text|keyword|needle|term|word|phrase|search", k.lower())
                                             and not is_id_key(k) for k in self.free_text_keys(name)):
                    best = ((False, True), name)
                    break
        if best and best[1] != search:
            self.wire.log.append({"kind": "search_role_fixed", "from": search, "to": best[1]})
            self.role_map["search"] = best[1]
    def use_reading(self, reading: dict[str, Any], again: bool = False) -> None:
        if not again:
            self.steps_given = self.step_cap
        old_cart = self.role("cart")
        role_map = reading.get("roles") if isinstance(reading.get("roles"), dict) else {}
        self.role_map = {fkey: val for fkey, val in role_map.items() if isinstance(val, str) and val in self.shapes}
        try:
            self.fix_search_role()
        except Exception as err:
            self.wire.log.append({"kind": "search_role_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if again and self.trolley.used and old_cart:
            self.role_map["cart"] = old_cart
        if not (again and self.trolley.used):
            self.trolley = CartBook(self.role("cart"), self.shapes.get(self.role("cart") or "", {}))
        self.needs.sweepable = bool(self.role("search")) and self.page_arg() is not None
        spend_cap = reading.get("message_budget")
        if not is_number(spend_cap):
            # the reading can miss a stated message limit; the request's own figure then holds (else the default applies)
            try:
                said = self.code_limit(self.request.lower(), r"messages?|questions?|notes?|repl(?:y|ies)|texts?")
                if said is not None:
                    spend_cap = reading["message_budget"] = said
                    self.wire.log.append({"kind": "message_budget_code", "messages": said})
            except Exception as err:
                self.wire.log.append({"kind": "message_budget_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if is_number(spend_cap) and spend_cap >= 0:
            self.msg_cap = int(spend_cap)
        self.needs.reading = reading
        try:
            self.needs.add_told(reading.get("instructions"), "request")
            self.needs.add_told([r for r in reading.get("rules") or [] if isinstance(r, str)], "request")
        except Exception as err:
            self.wire.log.append({"kind": "told_seed_error", "error": f"{type(err).__name__}: {err}"[:300]})
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
        # a long errand (several products need their steps) keeps at least the full model budget
        self.wire.scale = max(1.0, self.wire.__dict__.get("scale", 1.0) * max(0.5, min(1.0, 40.0 / max(1, self.step_cap))))
        # the hard stop grows a little with a long step allowance, so a long errand is not cut off midway
        self.wire.long = max(1.0, min(1.4, self.step_cap / 80.0))
        if again:
            self.needs.version += 1
    def story_values(self, reading: dict[str, Any]) -> None:
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
        line = self.prop_line(k, prop)
        # the values follow the count and any marks: "- <key> (in N listings) [mark] [mark]: v1 | v2"
        hit = re.match(r"\s*-\s*.+?\(in \d+ listings?\)(?:\s*\[[^\]]*\])*:\s?(.*)$", line)
        values = hit.group(1).split("|") if hit else []
        return {" ".join(re.findall(r"[a-z0-9]+", v.lower())) for v in values} - {""}
    def same_field(self, k: int | None, a: str, b: str) -> bool:
        if strict_prop(a, b):
            return True
        try:
            return same_prop(a, b) and bool(self.field_values(k, a) & self.field_values(k, b))
        except Exception:
            return False
    def hinted_prop(self, k: int | None, prop: str) -> bool:
        name = str(prop).split("(")[0].strip()
        for hint in self.needs.reading.get("hints") or []:
            head = str(hint).split(":", 1)[0]
            if strict_prop(head, name) and (k is None or not self.needs.items() or self.product_of(head) in (k, None)):
                return True
        return False
    def product_reads(self, k: int | None) -> int:
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
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            if not name or not (hinted or self.hinted_prop(k, name)) or self.prop_line(k, name) or self.product_reads(k) < 2:
                return False
            menu = (self.__dict__.get("menu_memo", {}).get(k) or (0, ""))[1]
            return not any(same_prop(line.strip("- ").split(" (in ")[0].strip(), name) for line in menu.splitlines())
        except Exception:
            return False
    def hint_mark(self, hint: str) -> str:
        try:
            prop = str(hint).split(":", 1)[0].strip()
            k = self.product_of(prop) if self.needs.items() else None
            return f"{hint} [no key]" if self.plain_hint(k, prop, hinted=True) else str(hint)
        except Exception:
            return str(hint)
    @staticmethod
    def hint_alts_of(detail: str) -> list[str]:
        return [a.strip() for m in re.findall(r"\(or ([^()]+)\)", str(detail)) for a in re.split(r",|\bor\b|/", m) if a.strip()]
    @staticmethod
    def name_inside(a: Any, b: Any) -> bool:
        wa = {w for w in prop_words(a) if len(w) >= 3 and w not in LINK_WORDS}
        wb = {w for w in prop_words(b) if len(w) >= 3 and w not in LINK_WORDS}
        return bool(wa) and bool(wb) and (wa <= wb or wb <= wa)
    def asked_inside(self, k: int | None, head: str) -> bool:
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
        try:
            return not strict_prop(a, b) and (self.name_twin(a, b) or self.same_field(k, a, b))
        except Exception:
            return False
    def twin_key(self, k: int | None, prop: str) -> str:
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
        frame = {"which", "would", "you", "like", "for", "the"}
        have = {self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())}
        need = [self.stem(w) for w in re.findall(r"[a-z0-9]+", str(plain_q).lower()) if w not in frame and w not in LINK_WORDS]
        return bool(need) and all(w in have for w in need)
    def alt_untried(self, k: int) -> bool:
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
    @staticmethod
    def menu_trim(menu: str, reads: int) -> str:
        if reads < 4 or not menu:
            return menu
        kept = [ln for ln in menu.splitlines() if not re.search(r"\(in [01] listings?\)", ln)]
        return "\n".join(kept) if kept else menu
    def first_reads(self) -> bool:
        try:
            batch = self.batch_tool()
            if self.__dict__.get("first_reads_done") or not (batch or self.role("details")):
                return False
            self.first_reads_done = True
            rows = self.needs.items()
            homes = self.home_shelves()
            qtys = [row["quantity"] for row in rows] or [self.needs.quantity()]
            room = self.moves_left() - self.order_due() - 2 * max(0, self.ask_room() - self.msgs_sent) - 8
            target = (READ_FIRST_BATCH if batch else READ_FIRST_SINGLY)[min(2, max(0, len(rows) - 1))]
            wants: list[tuple[int, int, list[str]]] = []
            taken: list[str] = []
            for k in range(len(rows) or 1):
                home = homes[k] if k < len(homes) else set()
                have = self.product_reads(k if rows else None)
                if home and have < target and room > 0:
                    picks = self.menu_rows(home, target - have, taken, qtys[min(k, len(qtys) - 1)])
                    taken += picks
                    if picks:
                        wants.append((have, k, picks))
            # (room counts steps: one side-by-side call reads several listings)
            queue = [picks[n] for n in range(target) for _, _, picks in sorted(wants) if n < len(picks)][: max(0, room * (batch[2] if batch else 1))]
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
        try:
            hinted = self.hinted_prop(k, prop)
            return self.open_prop(k, prop, self.prop_line(k, prop) if hinted else "", by_samples=hinted)
        except Exception:
            return self.open_prop(k, prop)
    def settles(self, req: str, samples: str = "") -> bool:
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
        base = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, self.step_cap // 4 + 1))
        due = max(base, min(16, self.step_cap // 3))
        if self.__dict__.get("event_seen") or (self.__dict__.get("dead") and set(self.trolley.units) & self.__dict__["dead"]):
            try:
                due = max(due, 8 + max(1, len(self.needs.items())))
            except Exception:
                due = max(due, 9)
        return max(base, min(due, self.step_cap // 2)) + self.step_reserve()
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
            if self.role("message") and spare >= 2 and (self.msg_cap is None or self.msgs_sent < self.msg_cap):
                name_set.add(self.role("message"))
            trimmed = [tdef for tdef in self.toolset if tdef["function"]["name"] in name_set]
            toolset = trimmed or toolset
        return toolset
    def in_reserve(self) -> bool:
        spare = self.moves_left()
        return bool(self.role("order") and self.role("cart")) and spare > 3 and (
            (spare <= self.order_due() and self.orders_tried == 0) or (spare <= FIRST_ORDER_BY and self.refusal_n > 0))
    def unverified(self) -> bool:
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
        return {i for i in self.__dict__.get("named", set()) if i in self.archive.ids} - self.archive.read - self.refused_lines
    def widened_late(self) -> bool:
        return self.needs.widened_at == self.needs.version and self.wire.tick - self.needs.widened_tick <= self.order_due()
    def begin_batch(self) -> None:
        self.late_q = 0
        self.late_ids = set()
    def late_hold(self, call_name: str, call_args: dict[str, Any]) -> str:
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
        try:
            fit_found, fit_text = self.fit_line(thing)
        except Exception as err:
            fit_found, fit_text = None, ""
            self.log.append({"kind": "fit_line_error", "error": f"{type(err).__name__}: {err}"[:300]})
        memo = key + (f"|{len(self.archive.read)}" if refs or any(r.get("status") != "resolved" for r in tracked) else "") + ("|second" if second else "") \
            + (f"|own{len(own)}" if own else "") + (f"|fit {fit_found} {fit_text[:80]}" if fit_found else "")
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
        if fit_found == "no":
            ruling = f"An independent check says {thing} does NOT meet every requirement: {fit_text}. Choose another listing."
            self.rulings[memo] = ruling
            self.log.append({"kind": "verify", "item": thing, "result": {"item": thing, "fits": False, "problems": [fit_text], "by": "fit"}})
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
                f"{prop} from, refer to, go by or look to another listing), {prop} is taken from that listing, so this item's own "
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
        if fit_text:
            ask_text += "\n\nCOMPATIBILITY (computed by code from the listings' own statements of fit, exceptions applied): " + fit_text
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
        hidden = [str(p) for p in answer.get("unshown") or []][:5]
        if hidden and not faults:
            self.__dict__.setdefault("unshown", set()).add(key)
            ruling = (f"An independent check says {thing} does not SHOW: " + "; ".join(hidden) + ". Prefer a listing whose record "
                      "shows it; when none does after the candidates are read, this one may still be bought (nothing shown fails).")
        else:
            ruling = f"An independent check says {thing} does NOT meet every requirement: " + "; ".join(faults + hidden) + ". Unless you are certain the check is wrong, choose another fully verified fit."
        self.rulings[memo] = ruling
        return ruling
    def ch_rows(self) -> dict[str, dict[str, Any]]:
        memo = self.__dict__.get("ch_rows_memo")
        if memo and memo[0] == len(self.archive.items):
            return memo[1]
        rows = self.code_rows()
        self.ch_rows_memo = (len(self.archive.items), rows)
        return rows
    def store_fields(self, label: Any = "") -> dict[str, list[str]]:
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
            if positive and Errand.code_like(value) and not any(
                    set(ch_words(value)) & set(words) for key in (named or list(carry)) for _, _, words in table.get(key, [])):
                said = (me.get("title") or "") + " " + (me.get("text") or "")
                lead = next((w for w in ch_words(value) if w.isalpha() and len(w) >= 3), "")
                hint = next((m.group(0).strip() for m in re.finditer(r"[^.;\n]*\b" + re.escape(lead) + r"\b[^.;\n]*", said.lower())), "") if lead else ""
                self.fit_read([thing])
                found, _, left = self.fit_verdict(thing, self.value_codes(value))
                verdict = ("its statement of fit covers it: it meets it." if found == "yes"
                           else "its statements of fit do not cover it: it does NOT meet it." if found == "no"
                           else f"its statements send the reader to part {' / '.join(left)} for the rest of their list, never read: "
                                "not shown either way." if found == "unknown"
                           else "it covers it: it meets it." if Errand.code_family(value, said)
                           else "it names it otherwise: it does NOT meet it." if hint else "it does not state it.")
                where = (f"the field named for it ({' / '.join(repr(x) for x in named[:2])}) holds values of another kind" if named
                         else "no field is named for it")
                head_txt = (f"- {prop} = '{clip(value)}': {where}; no read listing of this category writes '{clip(value)}' or any "
                            "word or figure of it in a field, so the listing's own text decides (a family or range written as a "
                            "pattern covers its members).")
                out["rules"].append(head_txt)
                out["lines"].append(head_txt + " This listing's text" + (f" says '{clip(hint, 90)}': " if hint else ": ") + verdict)
                continue
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
        if priority == "cheapest" and self.moves_left() > 6:
            better = list(dict.fromkeys(self.cheaper_rows(thing) + better))[:4]
        self.rank_better = list(better)
        self.__dict__.setdefault("named", set()).update(better)
        if not better:
            return ""
        return (f"A rank check thinks these listings may also fit and beat {thing} on the {what}: " + ", ".join(better)
                + (f" ({(answer or {}).get('reason')})" if (answer or {}).get("reason") else "")
                + ". Check them (details, stock, price) before ordering; if each fails, order the current pick.")
    def review_decline(self, early: bool = False) -> str:
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
        conforms = self.known_ids((answer or {}).get("possible_fits"))[:5]
        conforms = [x for x in conforms if not self.field_view(x)["fails"]]
        self.audit_fits = conforms
        unopened = [x for x in self.known_ids((answer or {}).get("unopened")) if x not in self.archive.read][:6]
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
    def known_ids(self, raw: Any) -> list[str]:
        out: list[str] = []
        rows = None
        for x in raw if isinstance(raw, list) else []:
            tok = str(x).strip().strip("[]()")
            if not tok or isinstance(x, bool):
                continue
            if tok in self.archive.ids:
                got = tok
            else:
                rows = rows if rows is not None else self.code_rows()
                pat = re.compile(r"(?<![\w-])" + re.escape(tok.lower()) + r"(?![\w-])")
                hits = [i for i, r in rows.items() if pat.search((r.get("title") or "").lower() + " " + (r.get("text") or "").lower())]
                got = hits[0] if len(hits) == 1 else ""
            if got and got not in out:
                out.append(got)
        return out
    def named_unread(self) -> list[str]:
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
                half = max(2, len(queue) // 2)
                keep0 = int(floor) if is_number(floor) else 3
                if len(queue) >= 3 and half < batch[2] and not self.__dict__.get("batch_shrunk") and not self.wire.done \
                        and self.moves_left() > keep0 + 1 and (reader or self.role("details")):
                    self.batch_shrunk = True
                    self.batch_spec = (batch[0], batch[1], half)
                    self.log.append({"kind": "batch_shrunk", "tool": batch[0], "most": half})
                    self.post_replies(responses)
                    responses = self.open_reads(queue[:half], say, self.batch_spec, reader)
                    if responses and self.batch_ok(self.__dict__.get("last_reads") or [], queue[:half]):
                        self.batch_proven = True
                if not self.__dict__.get("batch_proven"):
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
    def cheaper_rows(self, thing: str, most: int = 2) -> list[str]:
        mine, shelf = self.archive.price(thing), self.shelf_known(thing)
        if mine is None or not shelf:
            return []
        ver = str(self.needs.version)
        failed = {k.split("|")[0] for k, v in self.rulings.items() if v and k.split("|")[1:2] == [ver]}
        dead = self.dead_lines()
        cap, qty = self.needs.spend_cap, (1 if self.needs.items() else self.needs.quantity())
        k = self.product_index(thing) if self.needs.items() else None
        stated = [(prop, value) for prop, value in self.code_values(k)]
        pool = []
        for ident, row in self.code_rows().items():
            cost = row["price"] if row["price"] is not None else self.archive.price(ident)
            if ident == thing or not is_number(cost) or cost >= mine or row["stock"] is False or ident in failed or ident in dead \
                    or ident in self.refused_lines or (row["shelf"] or self.shelf_known(ident)) != shelf \
                    or (is_number(cap) and cost * qty > cap) or any(self.code_clash(row["fields"], p, v) for p, v in stated):
                continue
            pool.append((cost, ident))
        return [ident for _, ident in sorted(pool)[:most]]
    def cheaper_fit(self, thing: str) -> str:
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
        found = ""
        def visit(d: dict[str, Any]) -> None:
            nonlocal found
            if any(isinstance(val, str) and val == thing and is_id_key(key) for key, val in d.items()):
                found = self.shelf_of(d) or found
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return found or str((self.archive.__dict__.get("label_of") or {}).get(thing) or "").strip().lower()
    def chosen(self) -> str | None:
        if self.trolley.units:
            return list(self.trolley.units)[-1]
        return None
    def home_shelves(self) -> list[set[str]]:
        memo = self.__dict__.get("home_memo")
        if memo and memo[0] == len(self.archive.items):
            return [set(h) for h in memo[1]]
        homes: list[set[str]] = []
        try:
            rows = self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}]
            seen: set[str] = set()
            def visit(d: dict[str, Any]) -> None:
                shelf = self.shelf_of(d)
                if shelf and any(isinstance(v, str) and v in self.archive.ids and is_id_key(key) for key, v in d.items()):
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
        low = re.sub(r"\s+", " ", self.request.lower().replace("\u2019", "'"))
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
        hit = re.match(r'-\s*(.*?): asked "(.*?)" -> ', str(line), re.S)
        label, question = (hit.group(1), hit.group(2)) if hit else (str(line), str(line))
        kinds = [kind_tokens(r["product"]) for r in rows]
        for tag in re.findall(r"\(([^()]*)\)", label):
            words = kind_tokens(tag)
            if words and any(kind_in(w, k) for w in words for k in kinds):
                return True
        low = question.lower().replace("\u2019", "'")
        return any(name_pattern(r["product"]) and re.search(r"\b" + name_pattern(r["product"]), low) for r in rows)
    @staticmethod
    def joined_clash(a: dict[str, Any], b: dict[str, Any]) -> bool:
        def pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
            out = []
            for req in row.get("requirements") or []:
                head, _, value = str(req).partition(":")
                out.append((re.split(r"\s*\(", head)[0].strip(), " ".join(re.findall(r"[a-z0-9]+", value.lower()))))
            return out
        return any(same_prop(pa, pb) and va != vb for pa, va in pairs(a) for pb, vb in pairs(b))
    def merge_joined(self, obs: Any) -> str:
        seen = self.__dict__.setdefault("joined_seen", {})
        def visit(d: dict[str, Any]) -> None:
            if any(isinstance(v, str) and is_id_key(k) and re.fullmatch(ID_LIKE, v) for k, v in d.items()):
                label = next((val.strip() for key, val in d.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                              and shelf_key(key) and not is_id_key(key)), "")
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
        self.wire.scale = max(1.0, (1.0 + 0.3 * min(2, max(0, len(needs.items()) - 1))) * max(0.5, min(1.0, 40.0 / max(1, self.step_cap))))
        self.log.append({"kind": "merge_joined", "label": label, "products": [rows[i]["product"], rows[j]["product"]],
                         "into": phrase, "label_words": in_order, "products_left": max(1, len(needs.items()))})
        return (f"The store sells '{label}' as ONE category: the request asks for one product of it ('{phrase}'), so the order "
                + ("holds one line for it" if left >= 2 else "holds ONE line") + ", not one line for each kind its name joins. "
                "Requirements now: " + needs.text_body())
    def lookup_key(self, tool: str | None) -> str | None:
        jschema = self.shapes.get(tool or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        needed = [k for k in jschema.get("required") or [] if k in props] if isinstance(jschema.get("required"), list) else []
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
        key = next((k for k in texts if k in needed), None) or (texts[0] if len(texts) == 1 else None)
        return key if key and all(k == key for k in needed) else None
    def lookup_tools(self) -> list[str]:
        taken = {self.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        return [name for name in self.shapes if name not in taken and self.lookup_key(name)]
    def lookup_call(self, ident: str, tool: str | None) -> dict[str, Any] | None:
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
        return record_id(d)
    @staticmethod
    def asked_ids(call_args: Any) -> set[str]:
        args = call_args if isinstance(call_args, dict) else {}
        asked = {v for v in args.values() if isinstance(v, str) and re.fullmatch(ID_LIKE, v)}
        asked |= {x for v in args.values() if isinstance(v, list) for x in v if isinstance(x, str) and re.fullmatch(ID_LIKE, x)}
        return asked
    def learn_caps(self, call_name: str, call_args: Any, obs: Any) -> list[str]:
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
        caps = self.__dict__.get("tool_caps") or {}
        details = self.role("details")
        dkeys = (self.__dict__.get("tool_keys") or {}).get(details or "", set())
        nkeys = self.__dict__.get("note_keys") or {}
        shown = [t for t in self.lookup_tools() if t != details and set(nkeys.get(t) or set()) - set(dkeys)]
        return max(shown, key=lambda t: ((caps.get(t) or {}).get("notes", 0), t)) if shown else None
    def dead_lines(self) -> set[str]:
        return set(self.__dict__.get("dead") or set())
    def stale_lines(self) -> list[str]:
        seqs = self.__dict__.get("line_seq") or {}
        mark = self.__dict__.get("cart_seq", 0)
        return [i for i in self.trolley.units if seqs.get(i, -1) < mark]
    def dead_swap(self, line: str) -> tuple[str, str]:
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
            rows.insert(0, rows.pop(rows.index(low)))
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
        found = self.page_arg()
        if not found:
            return None
        walks = [w for w in self.__dict__.get("page_book", {}).values() if self.walk_carries(w["args"], k) and self.next_page(w) is not None]
        if not walks:
            return None
        best = max(walks, key=lambda w: (len(self.walk_carries(w["args"], k)), -len(w["read"])))
        return {**best["args"], found[0]: self.next_page(best)}
    def swap_open(self) -> str:
        if self.__dict__.get("swap_done"):
            return ""
        dead = self.dead_lines()
        return next((i for i in self.__dict__.get("swap_unread") or [] if i not in self.archive.read and i not in dead
                     and i not in self.trolley.units), "")
    def dead_note(self, lines: list[str]) -> str:
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
        items = self.needs.items()
        vals = [plain for k in (range(len(items)) if items else [None]) for _, plain in self.code_values(k)]
        plain = [v for v in vals if not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
                                                  v.lower())]
        resolved = self.differ_done(None)
        judged = sum(1 for req in self.needs.props_left() if any(self.ref_match(ref, req) for ref in resolved))
        return len(plain) + judged >= len(list(self.needs.props_left()))
    def differ_done(self, k: int | None) -> list[dict[str, Any]]:
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
            if self.moves_left() < len(self.cart_args(picks) or picks) + 1 + len(self.cart_drops([i for i in self.trolley.units if i not in {p for p, _ in picks}]) or []):
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
        return not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
                             str(value).lower())
    @staticmethod
    def ed_word(value: str) -> bool:
        low = str(value).lower()
        return bool(re.search(r"[a-z]{3,}", low)) and not re.search(r"\d", low) and Errand.ed_plain(low) \
            and not re.search(EXCLUDE_WORDS, low) and not ref_text(low)
    def ed_reqs(self, k: int | None) -> list[tuple[str, str, bool]]:
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
        out = []
        for k in range(len(self.needs.items())):
            reads = [e for e in self.ed_pool(k) if e["read"]]
            if reads and all(e["out"] for e in reads):
                out.append(k)
        return out
    def slack_rows(self) -> list[str]:
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
        for entry in reversed(self.log):
            if isinstance(entry, dict) and entry.get("kind") == "verify" and entry.get("item") == ident:
                res = entry.get("result")
                if isinstance(res, dict) and res.get("fits") is False:
                    return [str(p) for p in res.get("problems") or [] if str(p).strip()]
                return None
        return None
    def ed_news(self, k: int | None) -> bool:
        if self.refusal_n > 0:
            return True
        rows = self.needs.items()
        return any(not rows or self.product_of(str(req).partition(":")[0]) in (k, None)
                   for req in list(self.needs.uncovered) + list(self.needs.let_go))
    def near_hold(self) -> str:
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
                    # (a need the shopper revealed is checked too: the shopper's answer can put the store's value in other words)
                    if not says or not self.ed_word(value) or any(re.search(r"\d", s) for s in says):
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
               "the words of the request or of the shopper's answers. THE SHOPPER'S REQUEST (context):\n" + self.request[:1500] + "\n\nEach line: a listing, "
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
                "such listing meets every other requirement in force, is in stock and within the budget, so the code put the basket "
                "in the cart and read its stock: " + self.trolley.summary() + " Order this cart now; decline only if a record "
                "you read shows one of these lines failing a requirement now in force.")
    def cart_moved(self, before: dict[str, float]) -> None:
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
        if not self.__dict__.get("stock_seen"):
            return
        seqs = self.__dict__.setdefault("line_seq", {})
        def visit(d: dict[str, Any]) -> None:
            ident = self.rec_id(d)
            if ident in self.trolley.units and self.stock_state(d) is not None:
                seqs[ident] = self.__dict__.get("absorb_n", 0)
        walk_dicts(obs, visit)
    def cart_gone(self, lines: list[str]) -> set[str]:
        gone = set(lines) & self.dead_lines()
        stale = [i for i in self.stale_lines() if i in lines and i not in gone]
        if stale and self.stock_tool() and self.moves_left() >= len(stale) + 2:
            _, sold = self.code_recheck(stale)
            gone |= set(sold) | (set(lines) & self.dead_lines())
        return gone
    def batch_tool(self) -> tuple[str, str, int] | None:
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
                        spec = (name, arrays[0], int(min(most, LIST_MOST)))
                        break
            except Exception:
                spec = None
            self.batch_spec = spec
        return self.batch_spec
    @staticmethod
    def batch_ok(found: list[Any], ids: list[str]) -> bool:
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
        k = self.product_of(str(qitem.get("property") or ""))
        return k if k is not None else self.product_of(str(qitem.get("question") or ""))
    def quota_first(self, scripted: list[dict[str, Any]], ok: Any = None) -> list[dict[str, Any]]:
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
                asked[k] += 1
        for k in self.__dict__.get("group_asked") or set():
            if isinstance(k, int) and 0 <= k < len(rows):
                sent[k] += 1
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
        main = self.main_product()
        focus = False
        main_due = main is not None and (focus or asked[main] == 0)
        kinds = self.__dict__.get("reply_kind") or {}
        def level_of(k: int) -> tuple[int, int, int, int, int, int, int]:
            # (in a basket of unrelated products, one whose answer said the shopper does not mind comes before one whose
            # answer followed nothing: "no preference" on the asked properties leaves its other properties open)
            return (1 if main_due and k != main else 0, 1 if found[k] and any(j != k for j in unasked) else 0,
                    0 if k in hinted else 1, asked[k] + found[k], 0 if main is None and kinds.get(k) == "free" else 1,
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
                    pick = self.quota_pick(k, asked[k] == 0 or (focus and k == main), ok)
                except Exception:
                    pick = None
                if pick and ok is not None and (not ok(pick.get("property")) or self.asked_for(k, pick.get("property"))):
                    pick = None
                if pick:
                    return [pick] + scripted
                if self.__dict__.get("quota_wait"):
                    return scripted
                if not found[k]:
                    pick = self.turn_pick(k, ok)
                    if pick:
                        self.log.append({"kind": "turn_pick", "product": rows[k]["product"], "property": pick["property"]})
                        return [pick] + scripted
        return scripted
    def turn_pick(self, k: int, ok: Any = None) -> dict[str, Any] | None:
        rows = self.needs.items()
        product = rows[k]["product"]
        homes = self.home_shelves()
        home = homes[k] if k < len(homes) else set()
        read_home = [shelf for shelf in self.archive.shelves_read() if shelf.strip().lower() in home]
        def good(label: Any) -> bool:
            return ok is None or (bool(ok(label)) and not self.asked_for(k, label))
        pick = None
        if read_home:
            try:
                pick = self.product_plan_call(k, self.menu_trim(self.archive.prop_menu(read_home[0])[0], self.archive.rich_reads(read_home[0])), True)
            except Exception as err:
                self.log.append({"kind": "turn_plan_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if pick and good(pick.get("property")):
            return pick
        for prop in (base_name(p) for p in (self.__dict__.get("list_left") or {}).get(k, [])):
            label = f"{prop} ({product})"
            if good(label):
                return {"property": label, "question": f"Which {prop} would you like for the {product}?"}
        return None
    def linked_products(self) -> bool:
        return any(r.get("depends_on") is not None for r in self.needs.items())
    def main_product(self) -> int | None:
        rows = self.needs.items()
        mains = {r["depends_on"] for k, r in enumerate(rows) if r.get("depends_on") is not None and r["depends_on"] != k
                 and r["depends_on"] < len(rows)}
        return next(iter(mains)) if len(mains) == 1 else None
    def quota_pick(self, k: int, fallback: bool = True, ok: Any = None) -> dict[str, Any] | None:
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
            for line in menu.splitlines():
                prop = line.strip("- ").split(" (in ")[0].strip()
                if prop and self.open_prop(k, prop, line, by_samples=weigh) and good(f"{prop} ({product})"):
                    return {"property": f"{prop} ({product})", "question": f"Which {prop} would you like for the {product}?"}
        return None
    def product_plan(self, k: int, menu: str) -> dict[str, Any] | None:
        try:
            return self.product_plan_call(k, menu)
        except Exception:
            return None
    def product_plan_call(self, k: int, menu: str, turn: bool = False) -> dict[str, Any] | None:
        if not menu or self.__dict__.get("product_plans", 0) >= (2 if turn else 1) or self.wire.spare() < CLOSING + 40:
            return None
        self.product_plans = self.__dict__.get("product_plans", 0) + 1
        product = self.needs.items()[k]["product"]
        remaining = max(1, self.ask_room() - self.msgs_sent)
        ask_text = ASK_PLAN.replace("QUESTION_COUNT", str(min(3, remaining)))
        if self.__dict__.get("list_mode") is not False:
            ask_text += LIST_NOTE
        ask_text += "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
        ask_text += "\n\nREQUIREMENTS SO FAR:\n" + self.needs.text_body()
        if self.questions_log:
            ask_text += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.questions_log)
        named_before = sorted((self.__dict__.get("list_named") or {}).get(k, set()))
        ask_text += (f"\n\nTHIS TIME EVERY QUESTION IS ABOUT THE {product.upper()}: "
                     + (f"no need of it is known yet. Already named to the shopper for it: {', '.join(named_before) or 'nothing'}; properties "
                        "never named go first. " if turn else "every other product asked for already had a question and this one had none. ")
                     + f"Choose properties of the {product} only, from its own list below, the one the story's details point at first.")
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
                        "more": qitem.get("more") if isinstance(qitem.get("more"), list) else [],
                        "question": question if named else f"Which {first} would you like for the {product}?"}
            if second and self.plan_open(k, second):
                return {"property": f"{second} ({product})", "question": f"Which {second} would you like for the {product}?"}
        return None
    def open_prop(self, k: int | None, prop: str, line: str = "", by_samples: bool = False) -> bool:
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
    def drop_wanted(self, label: str) -> None:
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
    def list_question(self, scripted: list[dict[str, Any]], prop_list: str, fresh: Any) -> dict[str, Any] | None:
        if self.__dict__.get("list_mode") is False or not scripted:
            return None
        if self.needs.reading.get("ask_first") and not self.questions_log:
            return None
        head = scripted[0]
        first = str(head.get("property") or "").strip()
        if not first or ";" in first or not fresh(first):
            return None
        several = bool(self.needs.items())
        k = self.plan_product(head) if several else None
        if several and k is None:
            return None
        product = self.needs.items()[k]["product"] if several else str(self.needs.reading.get("product") or "item")
        named_before = (self.__dict__.get("list_named") or {}).get(k, set())
        lead_before = (self.__dict__.get("list_lead") or {}).get(k, set())
        names = [first]
        later: list[str] = []
        fresh_names: list[str] = []
        def base(p: str) -> str:
            return re.split(r"\s*\(", str(p))[0].strip()
        own_words = set(prop_words(product))

        def same(a: str, b: str) -> bool:
            # (one property under two spellings; a key whose words hold another key's words is a different property,
            # unless the extra words are only one- or two-letter marks or the product's own name: "pattern" and "fa
            # pattern", or "length" and "<product> length", spend two places of a short list on one property)
            wa, wb = set(prop_words(a)), set(prop_words(b))
            if strict_prop(a, b) or (bool(wa) and wa == wb):
                return True
            if re.sub(r"[^a-z0-9]", "", base(a).lower()) == re.sub(r"[^a-z0-9]", "", base(b).lower()):
                return True
            small, big = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
            return bool(small) and small < big and all(len(w) <= 2 or w in own_words for w in big - small)
        def add(p: Any) -> None:
            p = str(p or "").strip()
            if not p or junk_key(base(p)) or ";" in p:
                return
            tagged = f"{base(p)} ({product})" if several else base(p)
            if any(same(base(tagged), base(n)) for n in names + later + fresh_names):
                return
            if several and self.product_of(tagged) not in (k, None):
                return
            if not fresh(tagged) or self.asked_for(k, tagged) or base(tagged).lower() in lead_before:
                return
            if base(tagged).lower() in named_before:
                later.append(tagged)
            else:
                fresh_names.append(tagged)
        add(head.get("second"))
        for p in head.get("more") if isinstance(head.get("more"), list) else []:
            add(p)
        for p in (self.__dict__.get("list_left") or {}).get(k, []):
            add(p)
        for qitem in scripted[1:]:
            if not several or self.plan_product(qitem) == k:
                add(qitem.get("property"))
                add(qitem.get("second"))
        if several:
            homes = self.home_shelves()
            home = homes[k] if k < len(homes) else set()
            menu = "\n".join(self.archive.prop_menu(shelf)[0] for shelf in self.archive.shelves_read() if shelf.strip().lower() in home)
        else:
            menu = prop_list
        keyed = []
        for n, line in enumerate(menu.splitlines()):
            hit = re.match(r"\s*-\s*(.+?) \(in (\d+) listings?\)", line)
            if hit:
                keyed.append((-int(hit.group(2)), n, hit.group(1).strip()))
        top = -min(keyed)[0] if keyed else 0
        places = (self.__dict__.get("list_places") or {}).get(k, {})
        for neg, _, key in sorted(keyed):
            # a first list stops at LIST_SIZE; a later list of this product weighs every key the listings carry
            if (places or len(names) + len(later) + len(fresh_names) < LIST_SIZE) and -neg >= max(1, 0.2 * top):
                add(key)
        # (a planner's first pick that led an earlier list of this product gives way to the others)
        pool = [n for n in names if base(n).lower() not in lead_before] + later + fresh_names
        if places:
            # a later list: the planner's new first pick, then every other property by its weight (KEY_PRIOR by its rank
            # among the product's most carried keys) times the chance that each earlier place in a list left it uncovered
            # (a never-named key keeps its whole weight)
            ranked = [key for _, _, key in sorted(keyed)]

            def missed(p: str) -> float:
                chance = 1.0
                for place in places.get(base(p).lower(), []):
                    chance *= 1.0 - LIST_REACH[min(place, len(LIST_REACH) - 1)]
                return chance

            def prior(p: str) -> float:
                rank = next((n for n, key in enumerate(ranked) if same(base(p), key)), len(KEY_PRIOR))
                return KEY_PRIOR[rank] if rank < len(KEY_PRIOR) else 0.01
            # (the planner's first pick, when no list named it yet, keeps the lead: it usually follows the request's story)
            pick = pool[0] if pool and base(pool[0]).lower() not in places and same(pool[0], first) else None
            rest = sorted((p for p in pool if p is not pick), key=lambda p: -prior(p) * missed(p))
            pool = ([pick] if pick else []) + rest
        # the request's sentences that name this product: only these settle a property by naming its value (a story
        # sentence's adjective names no value of the product); none found, nothing is moved
        own = {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", str(product).lower())}
        lines = [line for line in re.split(r"(?<=[.!?;])\s+", self.request)
                 if own & {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", line.lower())}]
        req_words = (" " + " ".join(re.findall(r"[a-z0-9]+", " ".join(lines).lower())) + " ") if lines else ""

        def told(p: str) -> bool:
            # a property one of whose listing values those sentences name ("<value> <product>") is settled by the request,
            # whether or not the plan wrote it down as a requirement: it goes last
            try:
                return any(len(v) >= 3 and re.search(r"[a-z]", v) and f" {v} " in req_words for v in self.field_values(k, base(p)))
            except Exception as err:
                self.log.append({"kind": "list_told_error", "error": f"{type(err).__name__}: {err}"[:300]})
                return False
        settled_here = [p for p in pool if told(p)]
        if settled_here:
            self.log.append({"kind": "list_told", "names": settled_here[:8]})
        pool = [p for p in pool if p not in settled_here] + settled_here
        # more products still unasked than messages left: this message also names the next ones, a few keys each, so no
        # product is left without a question (a reply discloses a named hidden need about as often with two products)
        group = self.group_products(k) if several else []
        share = max(2, LIST_SIZE // (1 + len(group))) if group else LIST_SIZE
        # (with two places a product, each goes to a key this product's own listings carry)
        own_keys = {frozenset(prop_words(key)) for _, _, key in keyed} - {frozenset()}
        mine = [n for n in pool if frozenset(prop_words(base(n))) in own_keys] if group and own_keys else pool
        segs: list[tuple[str, list[str]]] = [(product, (mine or pool)[:share])]
        for j in group:
            try:
                got = self.group_names(j, scripted, fresh, share)
            except Exception as err:
                got = []
                self.log.append({"kind": "group_names_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if got:
                segs.append((self.needs.items()[j]["product"], got))
        if len(segs) > 1 and segs[0][1]:
            def phrase(ns: list[str]) -> str:
                said_ = [base(n) for n in ns]
                return said_[0] if len(said_) == 1 else f"{', '.join(said_[:-1])} or {said_[-1]}"
            question = (f"Which {phrase(segs[0][1])} would you like for the {segs[0][0]}, "
                        + ", ".join(f"and which {phrase(ns)} for the {prod}" for prod, ns in segs[1:]) + "?")
            names = [n for _, ns in segs for n in ns]
            self.log.append({"kind": "list_question", "names": names, "grouped": [prod for prod, _ in segs]})
            return {"property": "; ".join(names), "question": question, "list": True}
        names = pool[:LIST_SIZE]
        if len(names) < 2:
            return None
        said = [base(n) for n in names]
        question = f"Which {', '.join(said[:-1])} or {said[-1]} would you like for the {product}?"
        self.log.append({"kind": "list_question", "names": names})
        return {"property": "; ".join(names), "question": question, "list": True}
    def products_asked(self) -> set[int]:
        out = {j for j in self.__dict__.get("group_asked") or set() if isinstance(j, int)}
        for ln in self.questions_log:
            j = self.product_of(ln.split(":", 1)[0])
            if j is None:
                said = re.search(r'asked "(.*?)" -> ', ln)
                j = self.product_of(said.group(1)) if said else None
            if j is not None:
                out.add(j)
        return out
    def group_products(self, k: int | None) -> list[int]:
        rows = self.needs.items()
        if k is None or len(rows) < 2:
            return []
        asked = self.products_asked()
        free = self.ask_room() - self.msgs_sent
        unasked = [j for j in range(len(rows)) if j not in asked and j != k]
        if k in asked or free <= 0 or len(unasked) + 1 <= free:
            return []
        size = -(-(len(unasked) + 1) // free)
        return unasked[: size - 1]
    def group_names(self, j: int, scripted: list[dict[str, Any]], fresh: Any, cap: int) -> list[str]:
        product = self.needs.items()[j]["product"]
        named = (self.__dict__.get("list_named") or {}).get(j, set())
        out: list[str] = []
        held: list[tuple[set[str], str]] = []
        def base(p: Any) -> str:
            return re.split(r"\s*\(", str(p or ""))[0].strip()
        def add(p: Any) -> None:
            name = base(p)
            if not name or len(out) >= cap or junk_key(name) or ";" in name or name.lower() in named:
                return
            words, flat = set(prop_words(name)), re.sub(r"[^a-z0-9]", "", name.lower())
            if not words or any(words == w or flat == f for w, f in held):
                return
            tagged = f"{name} ({product})"
            if self.product_of(tagged) not in (j, None) or not fresh(tagged) or self.asked_for(j, tagged):
                return
            if own_keys and frozenset(words) not in own_keys:
                return
            held.append((words, flat))
            out.append(tagged)
        homes = self.home_shelves()
        home = homes[j] if j < len(homes) else set()
        menu = "\n".join(self.archive.prop_menu(shelf)[0] for shelf in self.archive.shelves_read() if shelf.strip().lower() in home)
        keyed = []
        for n, line in enumerate(menu.splitlines()):
            hit = re.match(r"\s*-\s*(.+?) \(in (\d+) listings?\)", line)
            if hit:
                keyed.append((-int(hit.group(2)), n, hit.group(1).strip()))
        # (only keys this product's own listings carry: a product never read gets no places)
        own_keys = {frozenset(prop_words(key)) for _, _, key in keyed} - {frozenset()}
        if not own_keys:
            return []
        for qitem in scripted:
            if self.plan_product(qitem) == j:
                add(qitem.get("property"))
                add(qitem.get("second"))
                for more in qitem.get("more") if isinstance(qitem.get("more"), list) else []:
                    add(more)
        for left in (self.__dict__.get("list_left") or {}).get(j, []):
            add(left)
        top = -min(keyed)[0] if keyed else 0
        for neg, _, key in sorted(keyed):
            if -neg >= max(1, 0.2 * top):
                add(key)
        return out
    def list_reply(self, label: tuple[str, str], heard_meta: Any, found_before: int) -> tuple[str, tuple[str, str]]:
        parts = [p.strip() for p in label[0].split(";") if p.strip()]
        first, rest = parts[0], parts[1:]
        several = bool(self.needs.items())
        k = self.product_of(first) if several else None
        def plain(text: Any) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
        found_text = " ".join(self.needs.uncovered[found_before:]).lower()
        def settled(p: str) -> bool:
            words = re.findall(r"[a-z]{4,}", p.split("(")[0].lower())
            return bool(words) and all(w in found_text for w in words)
        left = [p for p in rest if not settled(p)]
        self.needs.indifferent = [p for p in self.needs.indifferent
                                  if not any(p.strip().lower() in (q.lower(), q.split("(")[0].strip().lower()) for q in rest)]
        # (a grouped message names several products: each keeps its own names, leads and places)
        segs: dict[Any, list[str]] = {}
        for p in parts:
            j = self.product_of(p) if several else None
            segs.setdefault(k if j is None else j, []).append(p)
        store = self.__dict__.setdefault("list_left", {})
        for j, seg in segs.items():
            mine = [p for p in seg if p != first and not settled(p)]
            store[j] = list(dict.fromkeys(mine + [q for q in store.get(j, []) if q not in parts]))
            self.__dict__.setdefault("list_named", {}).setdefault(j, set()).update(base_name(p).lower() for p in seg)
            self.__dict__.setdefault("list_lead", {}).setdefault(j, set()).update(base_name(p).lower() for p in seg[:1])
            places = self.__dict__.setdefault("list_places", {}).setdefault(j, {})
            for p in seg:
                places.setdefault(base_name(p).lower(), []).append(parts.index(p))
        others = {j for j in segs if j is not None and j != k}
        if others:
            self.__dict__.setdefault("group_asked", set()).update(others)
        if heard_meta is not None and plain(label[1]) and plain(label[1]) in plain(heard_meta):
            # the shopper asked for one property per message: no more lists for any product; the lead goes out again
            # alone next, then one product alone takes the rest one by one, several products each take their own turn
            if not several:
                pairs = self.__dict__.setdefault("open_pair", [])
                for p in left:
                    if p.lower() not in [x.lower() for x in pairs]:
                        pairs.append(p)
            self.list_mode = False
            self.no_pairs = True
            if others:
                # nothing was answered: the other products of the message still wait for their own question
                self.__dict__.setdefault("group_asked", set()).difference_update(others)
            self.pending_seconds = [first] + [p for p in self.__dict__.get("pending_seconds") or [] if p.lower() != first.lower()]
            self.log.append({"kind": "list_refused", "names": parts, "several": several})
            if several:
                return (f" [one at a time: nothing answered; {first} asked again alone next, then one property per message, "
                        "each product in its turn]", (f"{first} [not answered] [one at a time]", label[1]))
            return (f" [one at a time: nothing answered; {first} asked again alone next, then "
                    f"{', '.join(base_name(p) for p in left)} one by one]", (f"{first} [not answered] [one at a time]", label[1]))
        self.list_mode = True
        if left:
            return (f" [also named in this message: {', '.join(base_name(p) for p in left)} - the reply may concern any one of "
                    "them; ask properties not named yet first]", (first, label[1]))
        return "", (first, label[1])
    def queue_second(self, prop: str) -> None:
        queue = self.__dict__.setdefault("pending_seconds", [])
        if prop and prop.lower() not in [p.lower() for p in queue]:
            queue.append(prop)
    def pose_planned(self, urge: bool = False) -> bool:
        the_tool = self.role("message")
        ceiling = self.ask_room()
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
        if self.__dict__.get("list_mode") is not False and not (self.needs.reading.get("ask_first") and not self.questions_log):
            ask_text += LIST_NOTE
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
        if pending and self.needs.items():
            sent = {self.product_of(ln.split(":", 1)[0]) for ln in self.questions_log} | set(self.__dict__.get("group_asked") or set())
            if any(j not in sent for j in range(len(self.needs.items()))):
                pending = ""
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
        is_list = False
        if not pending:
            try:
                listed = self.list_question(scripted, prop_list, fresh)
            except Exception as err:
                listed = None
                self.log.append({"kind": "list_question_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if listed:
                scripted = [listed] + scripted
        for thing in scripted[:2]:
            if thing.get("list"):
                proposal = self.message_call(str(thing.get("question")).strip(), f"ask_{self.msgs_sent + 1}_{self.wire.tick}")
                if proposal:
                    label = (str(thing.get("property")), str(thing.get("question")).strip())
                    is_list = True
                    break
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
            try:
                thing = self.key_named(thing, prop_list)
            except Exception as err:
                self.log.append({"kind": "key_named_error", "error": f"{type(err).__name__}: {err}"[:300]})
            proposal = self.message_call(str(thing.get("question")).strip(), f"ask_{self.msgs_sent + 1}_{self.wire.tick}")
            if proposal:
                label = (str(thing.get("property")), str(thing.get("question")).strip())
                second = str(thing.get("second") or "").strip()
                if second and second.lower() != label[0].lower() and not re.search(r"none|^-$|n/a", second.lower()):
                    split_second = second
                break
        if not proposal:
            if sets < ASK_AFTER_SETS and self.moves_left() > 12 and not urge:
                self.plan_wait = sets
                return False
            try:
                unread = [k for k in range(len(self.needs.items()) or 1) if self.product_reads(k if self.needs.items() else None) < 2]
            except Exception:
                unread = []
            if unread and self.moves_left() > 12 and not urge:
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
        if len(self.needs.uncovered) > found_before:
            try:
                self.value_block()
            except Exception as err:
                self.log.append({"kind": "value_block_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if pending and label[0].split(";")[0].strip().lower() == pending.lower():
            self.pending_seconds = [p for p in self.__dict__.get("pending_seconds") or [] if p.lower() != pending.lower()]
        open_note = ""
        heard_meta = self.needs.__dict__.pop("meta_last", None)
        try:
            # what this product's last answer did: named a need, said the shopper does not mind, or followed nothing
            if self.needs.items():
                k_asked = self.product_of(label[0].split(";")[0].strip())
                if k_asked is not None:
                    freed = len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0) > freed_before
                    self.__dict__.setdefault("reply_kind", {})[k_asked] = (
                        "need" if len(self.needs.uncovered) > found_before else "free" if freed and heard_meta is None else "none")
        except Exception as err:
            self.log.append({"kind": "reply_kind_error", "error": f"{type(err).__name__}: {err}"[:300]})
        def plain(text: Any) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
        if is_list:
            open_note, label = self.list_reply(label, heard_meta, found_before)
        elif heard_meta is not None and plain(label[1]) and plain(label[1]) in plain(heard_meta) and self.needs.items():
            parts = [p.strip() for p in label[0].split(";") if p.strip()] or [label[0]]
            self.__dict__.setdefault("spent_asks", set()).update(p.lower() for p in parts)
            open_note = " [no need named: the next product's turn]"
            label = (parts[0], label[1])
        elif heard_meta is not None and plain(label[1]) and plain(label[1]) in plain(heard_meta):
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
            twin = ""
            if twin:
                self.pending_seconds = [twin] + [p for p in self.__dict__.get("pending_seconds") or [] if p.lower() != twin.lower()]
                open_note += f" [{twin}: its story twin, asked next]"
        for _, text_body in responses[:1]:
            self.questions_log.append(f"- {label[0]}: asked \"{label[1]}\" -> {text_body.split(chr(10) + '[status]')[0][:400]}{open_note}")
        self.drop_wanted(asked_label)
        self.asking = self.msgs_sent < ceiling
        return True
    def reply_back(self, text_body: str) -> bool:
        if not self.role("message") or self.msgs_sent >= self.ask_room() or self.moves_left() <= 4 or self.wire.spare() < CLOSING \
                or (self.asking and self.msgs_sent >= self.ask_room() - 1 and self.moves_left() > 10 and self.wire.spare() >= CLOSING + 40):
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
    def code_reading(self) -> dict[str, Any]:
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
                             and len(w.strip(".")) >= 2 and not re.search(r"\d", w)]
                    sign = (m.group(1) or "").strip()
                    found = (int(figure) if figure.is_integer() else figure, " ".join(words) or sign or None)
        return found
    @staticmethod
    def code_limit(low: str, nouns: str) -> int | None:
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
        figures = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        said = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:units?|of (?:them|those|these)|copies)\b", low)
        if not said:
            return 1
        n = int(said.group(1)) if said.group(1).isdigit() else figures[said.group(1)]
        return n if 1 <= n <= 99 else 1
    @staticmethod
    def code_needs(seg: str) -> list[tuple[str, str]]:
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
        out: dict[str, dict[str, Any]] = {}
        def visit(d: dict[str, Any]) -> None:
            ident = self.row_mark(d)
            if ident not in self.archive.ids:
                return
            info = out.setdefault(ident, {"shelf": "", "title": "", "price": None, "stock": None, "words": set(), "fields": {}, "text": ""})
            info["shelf"] = self.shelf_of(d) or info["shelf"]
            info["title"] = next((str(val) for key, val in d.items() if isinstance(val, str) and re.search(r"title|headline|heading|name", str(key).lower())
                                  and not is_id_key(key)
                                  and not re.search(r"shop|brand|seller|store|vendor|maker|manufactur|producer|company|user|file", str(key).lower())), "") or info["title"]
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
            for key, val in d.items():
                if isinstance(val, str) and len(val.split()) >= 3 and val != info["title"] and val[:80] not in info["text"] \
                        and id_rank(key) is None and not re.search(r"shop|brand|seller|store|vendor|maker|manufactur|categor|url|image|"
                                                                   r"link|title|heading|name|label", str(key).lower()):
                    info["text"] = (info["text"] + " " + val)[:3000]
            for v in d.values():
                if isinstance(v, dict):
                    info["fields"].update({str(key): str(x) for key, x in v.items() if isinstance(x, (str, int, float)) and not isinstance(x, bool)})
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return out
    @staticmethod
    def code_shows(row: dict[str, Any], prop: str, value: str, hit_words: set[str], read: bool) -> bool:
        named = [val for key, val in row["fields"].items() if same_prop(key, prop)]
        coded = Errand.code_like(value)
        if named:
            if any(Errand.code_seen(value, val) for val in named):
                return True
            if not coded or any(set(ch_words(val)) & set(ch_words(value)) for val in named):
                return False
        if Errand.code_seen(value, row.get("title") or ""):
            return True
        if coded and Errand.code_family(value, (row.get("title") or "") + " " + (row.get("text") or "")):
            return True
        if named:
            return False
        own = kind_words(value)
        return not read and bool(own) and own <= hit_words
    @staticmethod
    def code_like(value: str) -> bool:
        body = str(value or "").lower()
        return bool(re.search(r"\b[a-z]{3,}\b", body)) and bool(re.search(r"(?<![\d.])\d{2,}(?![\d.])", body)) \
            and not re.search(r"\d\s*(?:-|to)\s*\d|\b(?:at least|at most|or more|or less|up to|under|over)\b", body)
    @staticmethod
    def code_family(value: str, text: str) -> bool:
        pool = str(text or "").lower()
        body = str(value or "").lower()
        words = [w for w in re.findall(r"[a-z]{3,}", body) if w not in LINK_WORDS]
        figs = re.findall(r"(?<![\d.])\d{2,}(?![\d.])", body)
        if not words or not figs or not all(re.search(r"\b" + re.escape(w), pool) for w in words):
            return False
        def covered(fig: str) -> bool:
            if re.search(r"(?<!\d)(?<!\d\.)" + re.escape(fig) + r"(?!\d)(?!\.\d)", pool):
                return True
            if any(len(head) + len(tail) == len(fig) and fig.startswith(head)
                   for head, tail in re.findall(r"(?<![\w.])(\d+)([x*]+)(?![\w*])", pool)):
                return True
            return any(len(lo) == len(hi) == len(fig) and int(lo) <= int(fig) <= int(hi)
                       for lo, hi in re.findall(r"(?<!\d)(?<!\d\.)(\d+)\s*(?:-|to)\s*(\d+)(?!\d)(?!\.\d)", pool))
        return all(covered(fig) for fig in figs)
    @staticmethod
    def code_seen(value: str, text: str) -> bool:
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
        return self.code_picks_once(avoid, keep) or (self.code_picks_once(avoid, keep, True) if self.__dict__.get("unshown") else [])
    def code_picks_once(self, avoid: set[str] | None = None, keep: set[str] | None = None, lenient: bool = False) -> list[tuple[str, int]]:
        rows = self.needs.items()
        counts = self.needs.counts()
        homes = self.home_shelves() if rows else [set(self.shelf_labels()) or (self.home_shelves() or [set()])[0]]
        info = self.code_rows()
        ver = str(self.needs.version)
        failed = {key.split("|")[0] for key, val in self.rulings.items() if val and key.split("|")[1:2] == [ver]}
        loose = {key.split("|")[0] for key in self.__dict__.get("unshown") or set() if key.split("|")[1:2] == [ver]} if lenient else set()
        failed -= loose
        live = {key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == ver}
        reviewed = self.__dict__.get("audit_ids")
        if self.__dict__.get("audit_done") and (reviewed is None or len(self.archive.read) <= self.__dict__.get("audit_reads", -1)):
            reviewed = set(self.archive.read)
        answered = next((isinstance(e.get("result"), dict) for e in reversed(self.log) if isinstance(e, dict) and e.get("kind") == "audit"), False)
        audited = (set(reviewed or set()) - set(self.__dict__.get("audit_fits") or []) - live - loose) if answered else set()
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
                              and not (e[-1] in self.archive.read and self.field_view(e[-1])["fails"])
                              and not self.fit_clash(k, e[-1], picks)), None)
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
        named = [val for key, val in fields.items() if same_prop(key, prop)]
        return bool(named) and not any(value_seen(value, val) for val in named)
    def cart_form(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None, list[str]]:
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
        entry = next((e for e in reversed(self.log) if isinstance(e, dict) and e.get("kind") == "env"), None)
        resp = (entry or {}).get("reply")
        if not isinstance(resp, dict):
            return []
        obs_list = resp.get("observations") if isinstance(resp.get("observations"), list) else [resp.get("observation")]
        return list(obs_list)[:n]
    @staticmethod
    def code_fault(obs: Any) -> bool:
        found = []
        def visit(d: dict[str, Any]) -> None:
            if any(re.search(FAULT_KEYS, str(key).lower()) and val not in (None, False, "", [], {}) for key, val in d.items()):
                found.append(d)
        walk_dicts(obs, visit, 0, 2)
        return bool(found)
    def code_search(self, products: int, more: bool = False) -> None:
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
        cart = self.role("cart")
        want = {ident for ident, _ in picks}
        spec_lines = self.trolley.line_spec()
        extra = [line for line in self.trolley.units if line not in want]
        stated = any(t.get("before_change") is True for t in self.needs.told)
        unread = any(t.get("before_change") is None for t in self.needs.told)
        if extra and (stated or unread):
            if not self.code_asks_swap(extra, [ident for ident, _ in picks]) and stated:
                return None
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
        def whole_cart(x: Any) -> bool:
            # a reply listing cart lines shows the whole cart; one about a single line shows only that line
            found: list[bool] = []
            walk_dicts(x, lambda d: found.extend([True for v in d.values() if isinstance(v, list) and any(isinstance(r, dict) for r in v)]))
            return bool(found) or (isinstance(x, list) and any(isinstance(r, dict) for r in x))
        for call_args, obs in zip(calls, replies):
            seen = self.trolley.view_units(obs)
            if seen is not None:
                shown = seen if whole_cart(obs) or shown is None else {**shown, **seen}
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
            try:
                dropped = self.code_relax(line)
            except Exception as err:
                dropped = []
                self.log.append({"kind": "code_relax_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if dropped:
                self.needs.let_go.extend(r for r in dropped if r not in self.needs.let_go)
                self.needs.version += 1
                self.log.append({"kind": "code_news", "dropped": dropped})
    def code_relax(self, line: str) -> list[str]:
        text = str(line)
        if LIT_RELAX_NOT.search(text):
            return []
        phrases = [m.group(1) for m in LIT_RELAX_LEAD.finditer(text)]
        phrases += [g for m in LIT_RELAX_TAIL.finditer(text) for g in m.groups() if g]
        if not phrases:
            return []
        rows_ = self.needs.items()
        products = {w for row in rows_ for w in lit_content(row["product"])} | set(lit_content(self.needs.reading.get("product") or ""))
        lines_now = [r for r in self.needs.props_left() if ":" in str(r)]
        def core(req: str) -> set[str]:
            head = set(lit_label_words(str(req).split(":")[0]))
            return (head - products) or head
        out: list[str] = []
        for phrase in phrases:
            words = set(lit_content(phrase)) - products
            if not words:
                continue
            targets = [r for r in lines_now if core(r) == words]
            if not targets:
                found = [r for r in lines_now if words <= set(lit_label_words(str(r).split(":")[0])) or core(r) <= words]
                first = (lit_content(phrase) or [""])[0]
                leading = [r for r in found if first in set(lit_label_words(str(r).split(":")[0]))]
                targets = leading or found
            out.extend(r for r in targets if r not in out)
        return out
    def code_cart(self, avoid: set[str] | None = None, keep: set[str] | None = None) -> bool:
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
                if call_rank and turn == 0:
                    try:
                        self.restore_full()
                    except Exception as err:
                        self.log.append({"kind": "restore_full_error", "error": f"{type(err).__name__}: {err}"[:300]})
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
                if call_rank and turn == 0 and not gone and self.trolley.units and self.needs.items():
                    try:
                        have = {self.product_index(line) for line in self.trolley.units}
                        if any(k not in have for k in range(len(self.needs.items()))):
                            filled = self.code_cart(avoid, set(self.trolley.units))
                            self.log.append({"kind": "cart_filled", "done": filled, "cart": dict(self.trolley.units)})
                    except Exception as err:
                        self.log.append({"kind": "cart_filled_error", "error": f"{type(err).__name__}: {err}"[:300]})
                if call_rank and turn == 0 and not gone and self.trolley.units:
                    try:
                        fix = self.kit_repair()
                    except Exception as err:
                        fix = ""
                        self.log.append({"kind": "kit_repair_error", "error": f"{type(err).__name__}: {err}"[:300]})
                    if fix == "misfit" and decl:
                        move = {"name": decl, "args": self.plain_args(decl, SORRY, "decline")}
                if move is not None:
                    pass
                elif call_rank and turn == 0 and not gone and self.trolley.used and (self.trolley.units or self.trolley.view_units(self.trolley.last_view) is None):
                    move = {"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")}
                elif call_rank and self.code_cart(avoid, keep):
                    move = {"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")}
                elif call_rank and turn > 0 and self.again_ok(avoid):
                    move = {"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")}
                elif decl:
                    move = {"name": decl, "args": self.plain_args(decl, SORRY, "decline")}
                else:
                    return
            kind = self.call_kind(move["name"], move["args"])
            try:
                self.finish_steps(kind or "order")
            except Exception as err:
                self.log.append({"kind": "finish_steps_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if self.wire.done or self.wire.broke or self.moves_left() <= 0:
                return
            resp = self.wire.act([move])
            self.log.append({"kind": "forced_finish", "action": move, "reply": resp})
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
            self.log.append({"kind": "forced_finish", "action": move, "reply": self.wire.act([move])})
    def again_ok(self, avoid: set[str]) -> bool:
        units = set(self.trolley.units)
        gone = set(avoid) | self.dead_lines()
        return bool(units) and not units & gone and bool(self.code_picks(gone))
    def catch_up(self) -> None:
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
        if isinstance(obs, dict) and "refusal" in obs:
            return True
        return call_name == self.role("order") and not resp.get("finished") and not (isinstance(obs, dict) and obs.get("error") in ("invalid_args", "unknown_tool"))
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
            lns: list[str] = self.notice_lines(obs)
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
        return next((val.strip().lower() for key, val in row.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                     and shelf_key(key) and not is_id_key(key)), "")
    @staticmethod
    def key_stock(key: Any, val: Any) -> bool | None:
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
        states = [s for s in (Errand.key_stock(key, val) for key, val in row.items()) if s is not None]
        return all(states) if states else None
    @staticmethod
    def sold_out(row: dict[str, Any]) -> bool:
        return any(Errand.key_stock(key, val) is False for key, val in row.items())
    def shelf_labels(self) -> set[str]:
        tally: dict[str, int] = {}
        counted: set[str] = set()
        skip = self.ref_foreign()
        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and is_id_key(key) and val in self.archive.read), None)
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
        best: tuple[int, str] | None = None
        for key, val in row.items():
            rank = id_rank(key)
            if rank is None or isinstance(val, bool) or not isinstance(val, (str, int)) \
                    or (rank == 2 and not (isinstance(val, str) and re.fullmatch(ID_LIKE, val))):
                continue
            if best is None or rank < best[0]:
                best = (rank, str(val))
        return best[1] if best else json.dumps(row, sort_keys=True, default=str)
    def search_keys(self, call_args: dict[str, Any]) -> tuple[list[tuple], tuple, bool, set]:
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
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        for key, pspec in (props if isinstance(props, dict) else {}).items():
            if isinstance(pspec, dict) and pspec.get("type") == "string" and not pspec.get("enum") \
                    and re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(key).lower()):
                return key
        hint = self.page_hint()
        return hint[0] if hint and hint[1] == "token" else None
    def page_hint(self) -> tuple[str, Any, Any, float | None] | None:
        got = (self.needs.reading or {}).get("search_page")
        if not isinstance(got, dict) or not isinstance(got.get("arg"), str):
            return None
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties") or {}
        pspec, kind = props.get(got["arg"]), str(got.get("kind") or "").lower()
        if not isinstance(pspec, dict):
            return None
        if pspec.get("type") in ("integer", "number") and kind in ("offset", "page"):
            low = pspec.get("minimum", pspec.get("default"))
            low = low if is_number(low) else (1 if kind == "page" else 0)
            return got["arg"], kind == "page", low, (pspec.get("maximum") if is_number(pspec.get("maximum")) else None)
        if pspec.get("type") == "string" and not pspec.get("enum") and kind == "token":
            return got["arg"], "token", "", None
        return None
    @staticmethod
    def next_token(obs: Any) -> str | None:
        if not isinstance(obs, dict):
            return None
        lens_next = LENS.get("next")
        if isinstance(lens_next, str) and isinstance(obs.get(lens_next), str) and obs[lens_next].strip():
            return obs[lens_next]
        for key, val in obs.items():
            if isinstance(val, str) and val.strip() and re.search(r"next|onward|cursor|bookmark|token|continu|more", str(key).lower()):
                return val
        return None
    def page_arg(self) -> tuple[str, Any, Any, float | None] | None:
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
        return (tok, "token", "", None) if tok else self.page_hint()
    @staticmethod
    def stem(word: str) -> str:
        return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word
    def shelf_rows(self) -> dict[str, tuple[float, str, str]]:
        shelves, cap, qty = self.shelf_labels(), self.needs.spend_cap, self.needs.quantity()
        found: dict[str, tuple[float, str, str]] = {}
        if not shelves:
            return found
        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and is_id_key(key) and val in self.archive.ids), None)
            if not ident or ident in self.refused_lines or self.shelf_of(d) not in shelves:
                return
            if self.sold_out(d):
                found.pop(ident, None)
                return
            cost = next((val for key, val in d.items() if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower())), None)
            title = d.get("title") if isinstance(d.get("title"), str) and 0 < len(d["title"]) <= 300 else next(
                (val for key, val in d.items() if isinstance(val, str) and 0 < len(val) <= 300 and re.search(r"title|heading|headline|name", str(key).lower())
                 and not is_id_key(key) and not self.shelf_of({key: val}) and not junk_key(key)), "")
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
            ones = [self.stem(h) for h in heads]
            out += ones + ([" ".join(words[:2])] if len(words) >= 2 else []) + heads + [w for w in words if w not in heads]
        return list(dict.fromkeys(out))
    def kind_queries(self, pairs: bool) -> list[str]:
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
        heads = {self.stem(q) for q in self.label_queries() if " " not in q}
        def second_pages() -> list[dict[str, Any]]:
            out = []
            for w in book.values():
                said = [self.stem(t.strip().lower()) for t in self.free_text(w["args"])]
                if len(w["read"]) == 1 and said and said[0] in heads and w["shelves"] & shelves and self.next_page(w) is not None:
                    out.append({**w["args"], key: self.next_page(w)})
            return out
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
        extra = [a for a in second_pages() if json.dumps(a, sort_keys=True, default=str) not in tried]
        if extra and turn % 2 == 1:
            return extra[0], False
        for pick in [0] + [1 + (turn + shift) % 3 for shift in range(3)]:
            picks = [a for a in sources[pick]() if json.dumps(a, sort_keys=True, default=str) not in tried]
            if picks:
                return picks[0], pick == 0
        return None, False
    def sweep_step(self) -> bool:
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
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        props = props if isinstance(props, dict) else {}
        tok = self.token_key()
        return [val for key, val in call_args.items() if isinstance(val, str) and not (props.get(key) or {}).get("enum") and key != tok]
    def book_page(self, call_args: dict[str, Any], rows: list, total: int, obs: Any = None) -> None:
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
        def near(a: str, b: str) -> bool:
            return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))
        vals = [re.findall(r"[a-z0-9]{3,}", req.partition(":")[2].lower()) for req in self.needs.props_left() if not ref_text(req.partition(":")[2])]
        words = [w for text in self.free_text(call_args) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
        words = words[: self.lead_words] if self.lead_words else words
        return sum(1 for value_words in vals if any(near(w, v) for w in words for v in value_words))
    def row_hits(self, rows: list, words: list[str] | None = None) -> list[str]:
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
        width = self.lead_words or 3
        mine = sorted(self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[:width])
        for walk in self.__dict__.get("page_book", {}).values():
            ran = sorted(self.stem(w) for w in re.findall(r"[a-z0-9]+", " ".join(self.free_text(walk["args"])).lower())[:width])
            if ran == mine:
                return True
        return False
    def search_template(self) -> tuple[dict[str, Any], str] | None:
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
        return self.unrefuted_once() or (self.unrefuted_once(True) if self.__dict__.get("unshown") else [])
    def unrefuted_once(self, lenient: bool = False) -> list[str]:
        try:
            rows = self.needs.items()
            homes = self.home_shelves() if rows else [set(self.shelf_labels())]
            if not homes or any(not home for home in homes):
                return []
            ver = str(self.needs.version)
            failed = {k.split("|")[0] for k, v in self.rulings.items() if v and k.split("|")[1:2] == [ver]}
            loose = {k.split("|")[0] for k in self.__dict__.get("unshown") or set() if k.split("|")[1:2] == [ver]} if lenient else set()
            failed -= loose
            live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == ver}
            reviewed = self.__dict__.get("audit_ids")
            if self.__dict__.get("audit_done") and (reviewed is None or len(self.archive.read) <= self.__dict__.get("audit_reads", -1)):
                reviewed = set(self.archive.read)
            reviewed = set(reviewed or set())
            audited = reviewed - set(self.__dict__.get("audit_fits") or []) - loose
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
        orders = [w.get("id_order") for w in self.__dict__.get("page_book", {}).values() if "id_order" in w]
        return orders.count(True) >= 2 and False not in orders
    def product_index(self, thing: str) -> int | None:
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
        low = str(text).lower()
        if re.search(CH_BOUND, low):
            return None
        found = re.match(r"\s*(\d+)\s+([a-z]+)", low)
        return (found.group(1), found.group(2)) if found else None
    @staticmethod
    def amount_part(text: Any) -> bool:
        low = str(text).lower()
        return bool(re.search(r"\d", low)) and bool(re.search(CH_BOUND, low) or re.match(r"\s*\d+(?:[.,]\d+)?\s+[a-z]", low))
    def walk_carries(self, args: dict[str, Any], k: int | None, alts: bool = False) -> set[Any]:
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
        return len(self.walk_carries(args, k))
    def walk_options(self, k: int | None) -> list[tuple[int, str, dict[str, Any]]]:
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
        gone = self.dead_lines()
        cands = [i for i in self.trolley.units if (k is None or self.product_index(i) == k) and i not in gone]
        if not cands:
            cands = [key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == str(self.needs.version)]
            cands = [i for i in cands if i not in self.refused_lines and i not in gone and (k is None or self.product_index(i) == k)]
        priced = [(i, self.archive.price(i)) for i in cands]
        priced = [(i, float(p)) for i, p in priced if is_number(p)]
        return min(priced, key=lambda ip: (ip[1], ip[0])) if priced else None
    def line_query(self, k: int | None) -> dict[str, Any] | None:
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
                      and not is_id_key(key)), "")
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
            if any(isinstance(v, str) and v in ids and is_id_key(key) for key, v in d.items()):
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
        rows: list[dict[str, Any]] = []
        def visit(d: dict[str, Any]) -> None:
            if len(d) >= 4 and any(isinstance(v, str) and v in self.archive.ids and is_id_key(key) for key, v in d.items()):
                rows.append(d)
        walk_dicts(obs, visit, ceiling=3)
        parts = []
        for r in rows[:12]:
            cost = next((v for key, v in r.items() if is_number(v) and re.search(r"price|cost", str(key).lower())), None)
            parts.append(f"{self.row_mark(r)} {cost if cost is not None else '?'}{' SOLD OUT' if self.sold_out(r) else ''} "
                         f"'{self.title_of(r)[:70]}' [{self.shelf_of(r)}]")
        return "Page rows (id, price, title, category): " + ("; ".join(parts) if parts else "none")
    def walk_call(self, args: dict[str, Any], say: str) -> bool:
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
        ver = str(self.needs.version)
        live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == ver}
        failed = {k.split("|")[0] for k, v in self.rulings.items() if v and k.split("|")[1:2] == [ver]}
        return (live | (set(self.__dict__.get("audit_fits") or []) - failed)) - self.refused_lines
    def decline_lacking(self) -> list[int | None]:
        got = {self.shelf_known(i) for i in self.decline_fits()} - {""}
        if not self.needs.items():
            return [] if set(self.shelf_labels()) & got else [None]
        return [k for k, home in enumerate(self.home_shelves()) if home and not home & got]
    def decline_walks(self) -> list[str]:
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
        reading = self.needs.reading
        infos = reading.get("reference_targets") if isinstance(reading.get("reference_targets"), list) else []
        infos = [t for t in infos if isinstance(t, dict)]
        listed = {str(r) for r in reading.get("references") or []} if isinstance(reading.get("references"), list) else set()
        for req in list(self.needs.written):
            value = str(req).partition(":")[2]
            if ref_text(value) or (str(req) in listed and re.search(DIFFER_WORDS, value.lower())):
                self.ref_add(str(req), infos, self.request, "request")
    def ref_untracked(self, refs: list[str], tracked: list[dict[str, Any]]) -> list[str]:
        try:
            return [r for r in refs if not any(self.ref_match(t, r) and (t.get("status") == "resolved" or t.get("chain")) for t in tracked)]
        except Exception:
            return refs
    def ref_take(self) -> None:
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
        neg = ref_negation(prop, source, tgt.get("seller"))
        if neg == "double" and (rel == "differ" or rel.startswith("not ")):
            rel, tol, unit = ref_relation(re.sub(DIFFER_WORDS, " ", str(value).lower(), count=1))
        elif neg != "double" and rel == "same" and (re.search(r"differ|not the same|exclu|apart|other than|unlike", told) or neg == "differ"):
            rel, tol, unit = "differ", None, ""
        elif neg == "differ" and rel in ("within", "at most", "at least", "pairs"):
            rel = "not " + rel
        elif rel == "same" and (re.search(r"pair", told) or (cut > 0 and re.search(PAIR_WORDS, span[:cut]))):
            rel = "pairs"
        ref = {"req": str(req), "head": head.strip(), "prop": prop, "tag": tag, "relation": rel, "tol": tol, "unit": unit,
               "target": tgt, "first": dict(tgt), "chain": [], "seen": {self.ref_key(tgt)}, "judged": set(), "queries": [],
               "pages": 0, "hops": 0, "tries": 0, "fails": 0, "status": "open", "value": "", "goes_with": [], "now": "",
               "origin": origin, "unindexed": False}
        self.refs.append(ref)
        self.log.append({"kind": "ref_open", "req": str(req)[:200], "target": tgt, "relation": rel, "origin": origin})
        return ref
    def ref_target(self, info: Any, value: str, source: str, prop: str) -> dict[str, Any] | None:
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
        learned = self.seller_keys()
        return {ref_name(v) for k, v in row.items() if isinstance(v, str) and v.strip() and len(v) <= 60
                and (re.search(SELLER_KEYS, str(k).lower()) or str(k) in learned) and "id" not in str(k).lower()
                and not re.search(r"note|text|desc|comment|detail|info|message|section", str(k).lower())} - {""}
    def seller_keys(self) -> set[str]:
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
                            r"title|heading|name|label|caption|categor|shelf|department|aisle|group|section|collection|note|text|desc|comment|detail|info|"
                            r"message|section|url|image|summary|overview", low) and (
                            ref_name(v) in names or (v.strip()[:1].isupper() and len(v.split()) <= 3 and ref_name(v) in noted)):
                        keys.add(str(k))
            self.seller_memo = (mark, keys)
            return keys
        except Exception:
            return set()
    @staticmethod
    def ref_texts(row: dict[str, Any], desc: bool = True) -> str:
        notes, other = [], []
        for k, v in row.items():
            name = str(k).lower()
            if not isinstance(v, str) or len(v.split()) < 3 or id_rank(k) is not None \
                    or re.search(r"title|heading|headline|name|label|caption|categor|shelf|department|url|image", name):
                continue
            (other if re.search(r"desc|overview|summary|about", name) else notes).append(v)
        return "\n".join(notes or (other if desc else []))
    def ref_parts(self, row: dict[str, Any]) -> tuple[set[str], set[str]]:
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
    def ref_elsewhere(self, ref: dict[str, Any], row: dict[str, Any]) -> str:
        """A described detail whose figures the listing's own notes write while its own field for that property holds other
        figures: the notes speak of another listing (a near twin with that value sold elsewhere), so this listing is not the
        one described. A listing with no field for the property is never ruled out here."""
        try:
            notes = " ".join(re.findall(r"[a-z0-9.]+", self.ref_texts(row, desc=False).lower()))
            if not notes:
                return ""
            fields = [(str(k), str(x)) for v in row.values() if isinstance(v, dict) for k, x in v.items()
                      if isinstance(x, (str, int, float)) and not isinstance(x, bool)]

            def has(fig: str, text: str) -> bool:
                return bool(re.search(r"(?<![0-9.])" + re.escape(fig) + r"(?![0-9])", text))
            for d in ref["target"]["details"]:
                figs = re.findall(r"\d+(?:\.\d+)?", str(d))
                words = {w for w in re.findall(r"[a-z]{3,}", str(d).lower()) if w not in LINK_WORDS}
                if not figs or not words or not all(has(f, notes) for f in figs):
                    continue
                for k, x in fields:
                    kw = {w for w in re.findall(r"[a-z]{3,}", re.sub(r"([a-z])([A-Z])", r"\1 \2", k).lower().replace("_", " "))}
                    low = str(x).lower()
                    if words & kw and re.search(r"\d", low) and not all(has(f, low) for f in figs):
                        return str(d)
            return ""
        except Exception:
            return ""
    @staticmethod
    def ref_field(row: dict[str, Any], prop: str) -> str:
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
        tgt = ref["target"]
        name = ref_name(tgt["seller"])
        kinds = kind_words(tgt["kind"])
        rows = self.ref_index()
        seen_at = self.__dict__.get("ref_seen_at") or {}
        names = set(ref_toks(" ".join(sorted(self.ref_key_words()))))
        values = [set(ref_toks(d)) - names or set(ref_toks(d)) for d in tgt["details"]]
        found = []
        for ident, row in rows.items():
            if name not in self.ref_sellers(row):
                continue
            hay = set(ref_toks(self.title_of(row) + " " + " ".join(str(v) for k, v in row.items()
                                                                  if isinstance(v, str) and re.search(r"desc|overview|summary", str(k).lower()))))
            found.append((ident, row, [n for n, v in enumerate(values) if v and v <= hay]))
        share = {n: sum(1 for _, _, hits in found if n in hits) for n in range(len(values))}
        scored = []
        for ident, row, hits in found:
            weight = sum(1.0 / share[n] for n in hits)
            scored.append((-(2 if kinds & kind_words(self.shelf_of(row)) else 0) - len(hits), -weight, -seen_at.get(ident, 0), ident, len(hits)))
        scored.sort()
        ref["shows"] = {ident: shown for _, _, _, ident, shown in scored}
        return [ident for _, _, _, ident, _ in scored]
    def ref_modes(self, needed: set[str], labels: set[str]) -> tuple[list[tuple[dict[str, Any], str]], set[str]]:
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
    def ref_key_words(self) -> set[str]:
        keys: set[str] = set()
        for row in self.ref_index().values():
            for val in row.values():
                if isinstance(val, dict):
                    keys |= {w for k in val for w in re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(k)).lower())}
        keys |= {w for req in self.needs.written + self.needs.uncovered for w in re.findall(r"[a-z]+", re.split(r"\s*\(", str(req).partition(":")[0])[0].lower())}
        return keys
    def ref_words(self, ref: dict[str, Any]) -> list[str]:
        keys = self.ref_key_words()
        seller = set(re.findall(r"[a-z0-9]+", str(ref["target"]["seller"]).lower()))
        joined, figures, words = [], [], []
        for d in ref["target"]["details"]:
            prev = ""
            for whole in str(d).lower().split():
                unit = bool(re.fullmatch(r"\d+(?:[.,]\d+)?", prev))
                if unit and re.fullmatch(r"[a-z]{1,3}", whole) and whole not in LINK_WORDS and prev.isdigit() and len(prev) >= 2:
                    joined.append(prev + whole)
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
        return max(max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, self.step_cap // 4 + 1)) + 10, self.order_due() + 7)
    def ref_product(self, ref: dict[str, Any]) -> int | None:
        return self.product_of(f"x ({ref['tag']})") if self.needs.items() and ref.get("tag") else None
    def ref_waits(self, floor: int | None = None) -> bool:
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
                if steps - start < (5 if self.moves_left() > self.order_due() + 8 else 3):
                    return True
                continue
            if ref.get("queries") or not self.ed_others_verified(ref):
                continue
            ref["late_from"] = steps
            self.log.append({"kind": "ref_late", "prop": ref.get("prop"), "to": str(tgt.get("seller"))[:80]})
            return True
        return False
    def ed_others_verified(self, ref: dict[str, Any]) -> bool:
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
        head, _, value = str(req).partition(":")
        if str(req) in (ref["req"], ref.get("now")):
            return True
        pointing = ref_text(value) or (ref.get("relation") == "differ" and re.search(DIFFER_WORDS, value.lower()))
        if not pointing or not same_prop(re.split(r"\s*\(", head)[0], ref["prop"]):
            return False
        tags = re.findall(r"\(([^)]*)\)", head)
        return not tags or not ref["tag"] or bool(kind_words(tags[-1]) & kind_words(ref["tag"]))
    def ref_live(self, ref: dict[str, Any]) -> bool:
        left = self.needs.props_left()
        if any(self.ref_match(ref, r) or same_prop(re.split(r"\s*\(", str(r).partition(":")[0])[0], ref["prop"]) for r in left):
            return True
        return not any(same_prop(re.split(r"\s*\(", str(r).partition(":")[0])[0], ref["prop"]) for r in self.needs.let_go)
    def ref_step(self) -> bool:
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
        twins = {i for i in cands if self.ref_elsewhere(ref, rows.get(i) or {})}
        if twins:
            ref["judged"].update(twins)
            self.log.append({"kind": "ref_twins", "prop": ref["prop"], "ids": sorted(twins)[:6]})
            cands = [i for i in cands if i not in twins]
            if not cands:
                return False
        def words_ok(i: str) -> bool:
            return all(shows(i, d) for d in tgt["details"] if d not in loose)
        full = {i: self.ref_full(ref, i, parts[i]) for i in cands}
        ranked = sorted(cands, key=lambda i: (not words_ok(i), -full[i], cands.index(i)))
        near = [i for i in ranked[:2] if sum(1 for d in tgt["details"] if shows(i, d)) >= max(1, len(tgt["details"]) - 1)]
        if not words_ok(ranked[0]) and near and self.ref_room() and self.ref_notes_read(ref, self.ref_notes_missing(ref, near)):
            return True
        if not words_ok(ranked[0]) and ((self.ref_room() or self.ref_waits()) and ([i for i in self.ref_cands(ref) if i not in self.archive.read
                                                                                    and i not in ref["judged"]] or self.ref_plans(ref) or self.ref_page(ref))):
            return False
        pick = [i for i in ranked if words_ok(i)][:4] or ranked[:4]
        exact = [i for i in pick if tgt["details"] and full[i] == len(tgt["details"])]
        missing = self.ref_notes_missing(ref, exact if len(exact) == 1 else ([i for i in pick if words_ok(i)][:2] or pick[:1]))
        room = self.ref_room() or (bool(missing) and self.moves_left() > self.order_due() + 2 and self.wire.spare() > CLOSING + 30)
        if missing and room and self.ref_notes_read(ref, missing):
            return True
        if missing and self.ref_waits():
            return False
        if missing and not self.ref_room() and ref.get("relation") == "differ":
            ref["judged"].update(pick)
            self.ref_stop(ref, "the notes of the listing it points at could not be read in the steps left")
            return True
        if len(exact) == 1 and not missing and ref["relation"] not in ("pairs", "not pairs") and self.ref_plain(ref, exact[0]):
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
    def ref_full(self, ref: dict[str, Any], ident: str, parts: tuple[set[str], set[str]] | None = None) -> int:
        keys, vals = parts or self.ref_parts(self.ref_index().get(ident) or {})
        return sum(1 for d in ref["target"]["details"] if set(ref_toks(d)) and set(ref_toks(d)) <= keys | vals)
    def ref_plain(self, ref: dict[str, Any], ident: str) -> bool:
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
        return next((t for t in tools if not (caps.get(t) or {}).get("id", 0)), None)
    def ref_notes_missing(self, ref: dict[str, Any], ids: list[str]) -> list[str]:
        tool = self.ref_notes_tool()
        if not tool or not ids:
            return []
        seen = set((self.__dict__.get("read_with") or {}).get(tool) or set())
        tried = ref.setdefault("noted", set())
        return [i for i in dict.fromkeys(ids) if i not in seen and i not in tried][:2]
    def ref_notes_read(self, ref: dict[str, Any], need: list[str]) -> bool:
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
        out: set[str] = set()
        for ref in self.__dict__.get("refs") or []:
            if ref.get("relation") == "differ" and ref.get("status") in ("open", "stuck", "resolved") and self.ref_live(ref):
                out.update(i for i in ref.get("chain") or [] if self.ref_own_product(ref, i))
        return out
    def ref_own_product(self, ref: dict[str, Any], ident: str) -> bool:
        if not self.needs.items():
            return True
        shelf = self.shelf_known(ident)
        k = self.product_of(f"x ({shelf})") if shelf else None
        return k is None or not ref.get("tag") or k == self.product_of(f"x ({ref['tag']})")
    def ref_pointer_note(self, thing: str) -> str:
        for ref in self.__dict__.get("refs") or []:
            if ref.get("relation") == "differ" and thing in (ref.get("chain") or []) and self.ref_live(ref) and self.ref_own_product(ref, thing):
                what = f" ({ref['value']})" if ref.get("status") == "resolved" and ref.get("value") else ""
                return (f"{thing} is the listing the shopper's reference for {ref['head']} points at, or one its notes lead "
                        f"through, so its {ref['prop']} is the very value{what} the shopper's {ref['prop']} must differ from: it cannot "
                        "meet that requirement")
        return ""
    def ref_excluded(self, fields: dict[str, str], k: int | None) -> bool:
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
                         + "; ".join(f"'{r}'" for r, _, _ in own) + " (the value is taken from the other listing), so it meets "
                         + ("them" if len(own) > 1 else "it") + " only when that listing's record shows it; the item check runs again "
                         "before it is ordered" + (". If they do, take it out of the cart and put another verified fit in."
                                                   if ident in self.trolley.units else "."))
        return " ".join(parts)
    def ref_phrase(self, line: str) -> tuple[str, int, int, dict[str, Any]] | None:
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
        neg = re.match(DIFFER_WORDS, text[start:].lower())
        if neg and ref_double(text, start):
            start += neg.end() + len(text[start + neg.end():]) - len(text[start + neg.end():].lstrip())
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
        if not isinstance(answer, dict):
            ref["fails"] += 1
            if ref["fails"] >= 2:
                ref["judged"].update(pick)
                self.ref_stop(ref, "the hop check gave no answer")
            return
        target = answer.get("target")
        target = str(target).strip() if isinstance(target, (str, int)) and not isinstance(target, bool) else ""
        missing = [str(x) for x in answer.get("missing") or [] if str(x).strip()] if isinstance(answer.get("missing"), list) else []
        sure = [i for i in pick if ref["target"]["details"] and self.ref_full(ref, i) == len(ref["target"]["details"])]
        if target not in pick and len(sure) == 1 and again and self.ref_calls_ok():
            self.ref_apply(ref, pick, self.ref_ask(ref, pick, f"{sure[0]} is sold by that seller and its record or its own text "
                                                   f"shows every detail named ({'; '.join(ref['target']['details'])}): it is the "
                                                   f"described listing unless it is of another kind. When its text sends the reader "
                                                   f"to another listing for the {ref['prop']}, give that listing as go_to."), again=False)
            return
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
        row = self.ref_index().get(target) or {}
        keys, vals = self.ref_parts(row)
        for d in [d for d in ref["target"]["details"] if d not in (ref.get("loose") or set())]:
            if not self.ref_shows(row, d, (keys, vals)):
                return "target", f"{target} does not show '{d}', so it is not the listing described.", None, "", []
        twin = self.ref_elsewhere(ref, row)
        if twin:
            return "target", (f"{target}'s own notes write '{twin}' as another listing's while its fields do not hold it: it is a near "
                              f"twin, not the listing described."), None, "", []
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
        vt, own = set(ref_toks(value)), self.ref_field(row, ref["prop"])
        if not (own and vt <= set(ref_toks(own))):
            said = set(ref_toks(self.title_of(row) + " " + " ".join(s for s in re.split(r"(?<=[.!?;])\s+", notes) if self.ref_names_prop(ref["prop"], s))))
            other = next((str(k) for v in row.values() if isinstance(v, dict) for k, x in v.items()
                          if isinstance(x, (str, int, float)) and vt <= set(ref_toks(x)) and not same_prop(k, ref["prop"])), "")
            if other and not vt <= said:
                return "answer", (f"'{value}' is {target}'s {other}, not its {ref['prop']}: give the value its record or its own text writes "
                                  f"for the {ref['prop']}."), None, "", []
        raw = answer.get("goes_with") if isinstance(answer.get("goes_with"), list) else []
        pairs = [str(p).strip() for p in raw if isinstance(p, (str, int, float)) and not isinstance(p, bool) and str(p).strip()]
        pairs = [p for p in dict.fromkeys(pairs) if set(ref_toks(p)) and set(ref_toks(p)) <= ntoks and set(ref_toks(p)) != set(ref_toks(value))][:3]
        return "", "", None, value, pairs
    def ref_go(self, ref: dict[str, Any], target: str, go: dict[str, Any]) -> None:
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
        value, pairs = str(ref["value"]), list(ref.get("goes_with") or [])
        if ref["relation"] == "within" and is_number(ref.get("tol")):
            hit = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)", value.lower().replace(",", ""))
            if hit:
                base, tol = float(hit.group(1)), float(ref["tol"])
                unit = hit.group(2) or str(ref.get("unit") or "")
                return f"{round(base - tol, 6):g} to {round(base + tol, 6):g} {unit}".strip() + f" (within {tol:g} of {value})"
        if ref["relation"] in ("at most", "at least"):
            return f"{ref['relation']} {value}"
        if ref["relation"] == "not within" and is_number(ref.get("tol")):
            hit = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)", value.lower().replace(",", ""))
            if hit:
                base, tol = float(hit.group(1)), float(ref["tol"])
                unit = hit.group(2) or str(ref.get("unit") or "")
                return f"below {round(base - tol, 6):g} {unit} or above {round(base + tol, 6):g} {unit}".replace("  ", " ").strip() \
                    + f" (more than {tol:g} away from {value})"
        if ref["relation"] in ("not at most", "not at least"):
            return f"{'more' if ref['relation'] == 'not at most' else 'less'} than {value}"
        if ref["relation"] == "not pairs":
            return " and ".join(f"not {p}" for p in pairs if p.lower() != value.lower())
        if ref["relation"] == "differ":
            return " and ".join(f"not {v}" for v in [value] + [p for p in pairs if p.lower() != value.lower()])
        if ref["relation"] == "pairs":
            others = [p for p in pairs if p.lower() != value.lower()]
            return " or ".join(others) if others else value
        return " or ".join([value] + [p for p in pairs if p.lower() != value.lower()])
    def ref_resolve(self, ref: dict[str, Any], target: str, value: str, pairs: list[str]) -> None:
        if not str(value or "").strip():
            self.ref_stop(ref, f"{target} shows no value for it")
            return
        if ref["relation"] in ("pairs", "not pairs") and not [p for p in pairs if p.lower() != str(value).strip().lower()]:
            ref["chain"].append(target)
            self.ref_stop(ref, f"{target}'s notes name no value that goes with its {value}")
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
        if ref["relation"] == "pairs":
            pair = (f"; its notes say {ref['value']} goes together with {' and '.join(pairs)}: the shopper wants the value that goes with "
                    f"it, so {ref['value']} itself does not meet it" if pairs else "; its notes name no value that goes with it")
        if ref["relation"] == "differ":
            pair = (f"; its notes pair {' and '.join(pairs)} with it" if pairs else "") + (
                ". The shopper's value MUST DIFFER from that one: any other value meets it, that value does not")
        if ref["relation"].startswith("not "):
            pair = self.ref_against(ref)
        self.ref_tell(f"RESOLVED: {ref['head']} is given by another listing, and {target} shows {ref['value']}{via}{pair}. The "
                      f"requirement is now: {new_req}.")
    @staticmethod
    def ref_against(ref: dict[str, Any]) -> str:
        rel, value, pairs = str(ref.get("relation")), str(ref.get("value")), list(ref.get("goes_with") or [])
        if rel == "not pairs":
            return (f"; its notes say {value} goes together with {' and '.join(pairs)}: the shopper's value must NOT be one that goes "
                    f"with it ({value} itself meets it)")
        if rel == "not within":
            return f". The shopper's value must be MORE than {ref.get('tol')} {ref.get('unit') or ''} away from that one".replace("  ", " ")
        return f". The shopper's value must be {'MORE' if rel == 'not at most' else 'LESS'} than that one"
    def ref_swap(self, ref: dict[str, Any], new_req: str) -> None:
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
        if ref["status"] != "open":
            return
        ref["status"], ref["why"] = "stuck", why
        self.log.append({"kind": "ref_stuck", "req": ref["req"][:200], "why": why[:300], "chain": list(ref["chain"])})
        self.ref_tell(f"NOT RESOLVED: {ref['head']} ({why}). Its value stays unknown: if a listing's notes you read show where it "
                      "ends, use that value; otherwise never decline or reject a listing over it.")
    def ref_tell(self, text: str) -> None:
        if self.chat and self.chat[-1].get("role") in ("tool", "user") and isinstance(self.chat[-1].get("content"), str) and len(self.chat) > 2:
            self.chat[-1]["content"] += "\n[reference] " + text
        else:
            self.ref_told = list(self.__dict__.get("ref_told") or []) + [text]
    def ref_line(self) -> str:
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
        try:
            lines, ends = [], []
            for ref in self.__dict__.get("refs") or []:
                if ref["status"] == "resolved":
                    via = f" (reached from {ref['chain'][0]} through the listings' notes)" if len(ref["chain"]) > 1 else ""
                    pair = (f"; its notes say {ref['value']} goes together with {' and '.join(ref['goes_with'])}, so each of them meets it"
                            if ref["goes_with"] else "")
                    if ref.get("relation") == "pairs":
                        pair = (f"; its notes say {ref['value']} goes together with {' and '.join(ref['goes_with'])}: the value that goes "
                                f"with it is required, {ref['value']} itself is not" if ref["goes_with"] else "")
                    if ref.get("relation") == "differ":
                        pair = (f"; its notes pair {' and '.join(ref['goes_with'])} with it" if ref["goes_with"] else "")
                        lines.append(f"- {ref['now']}: the shopper's {ref['prop']} MUST DIFFER from the value shown on {ref['chain'][-1]} "
                                     f"({ref['value']}){via}{pair}: any other value meets it, that value does not; the listings of its "
                                     f"chain ({', '.join(ref['chain'])}) carry it.")
                    elif str(ref.get("relation")).startswith("not "):
                        lines.append(f"- {ref['now']}: set from the value shown on {ref['chain'][-1]} ({ref['value']}){via}{self.ref_against(ref)}.")
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
                           if not self.shelf_of({kk: x})) for r in dicts)
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
        spec = self.__dict__.get("ceil_mode")
        return spec if spec and spec["sig"] not in self.__dict__.get("ceil_bad", set()) else None
    def cg_listing(self, args: dict[str, Any]) -> bool:
        spec = self.ceil_spec()
        if not spec or not isinstance(args, dict):
            return False
        if spec["mode"]:
            return all(args.get(k) == v for k, v in spec["mode"].items())
        return spec["bound"] in args
    def page_size(self) -> int:
        return max([w.get("size", 0) for w in self.__dict__.get("page_book", {}).values()] or [0])
    def visible_depth(self) -> int | None:
        found = self.page_arg()
        size = self.page_size()
        if found and size > 0 and found[1] != "token" and found[3] is not None:
            return int((found[3] - found[2] + 1) * size) if found[1] else int(found[3] - found[2] + size)
        return self.__dict__.get("page_cap")
    def raw_label(self, labels: set[str]) -> str:
        if not labels:
            return ""
        tally: dict[str, int] = {}
        def visit(d: dict[str, Any]) -> None:
            if not any(isinstance(v, str) and v in self.archive.ids and is_id_key(key) for key, v in d.items()):
                return
            raw = next((val.strip() for key, val in d.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                        and shelf_key(key) and not is_id_key(key)), "")
            if raw and raw.lower() in labels:
                tally[raw] = tally.get(raw, 0) + 1
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return max(sorted(tally), key=lambda s: tally[s], default="")
    def below(self, price: Any) -> float | int | None:
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
        if self.needs.items() and k is None:
            return False
        return (self.needs.priority() == "cheapest" and k not in self.__dict__.get("ceil_walked", set()) and self.ceil_spec() is not None
                and self.page_arg() is not None and bool(self.raw_label(self.walk_home(k))))
    def ceil_aim(self, k: int | None) -> float | None:
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
        if not self.needs.items() or k is None:
            return False
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            if self.product_of(head) == k and re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
                return True
        return False
    def home_rows(self, home: set[str]) -> set[str]:
        out: set[str] = set()
        def visit(d: dict[str, Any]) -> None:
            ident = self.row_mark(d)
            if ident in self.archive.ids and self.shelf_of(d) in home:
                out.add(ident)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return out
    def ceil_clash(self, home: set[str], bound: float) -> bool:
        hit = False
        def visit(d: dict[str, Any]) -> None:
            nonlocal hit
            if hit or self.shelf_of(d) not in home or not any(isinstance(v, str) and v in self.archive.ids and is_id_key(key) for key, v in d.items()):
                return
            cost = self.row_cost(d)
            hit = cost is not None and cost <= bound and self.stock_state(d) is True
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return hit
    def cheap_unread(self, ids: Any, k: int | None, below: float | None = None) -> list[tuple[float, str]]:
        ids = set(ids)
        if not ids:
            return []
        qty = self.needs.items()[k]["quantity"] if k is not None and k < len(self.needs.items()) else self.needs.quantity()
        cap = self.needs.spend_cap
        sold: dict[str, bool] = {}
        def visit(d: dict[str, Any]) -> None:
            if any(isinstance(v, str) and v in ids and is_id_key(key) for key, v in d.items()):
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
        return self.word_kind == "all" or (bool(self.lead_words) and self.word_kind != "any")
    def valued_ids(self, k: int | None) -> set[str]:
        if not self.all_words():
            return set()
        home = self.walk_home(k)
        out: set[str] = set()
        for walk in self.__dict__.get("page_book", {}).values():
            if walk.get("id_order") is True and not self.cg_listing(walk["args"]) and self.walk_carries(walk["args"], k):
                out |= {i for i, s in (walk.get("row_shelf") or {}).items() if s in home}
        return out
    def valued_complete(self, k: int | None) -> bool:
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
            if any(isinstance(v, str) and v == ident and is_id_key(key) for key, v in d.items()):
                text.extend(str(v).lower() for v in d.values() if isinstance(v, str))
        for item in self.archive.items:
            walk_dicts(item, visit)
        have = {self.stem(w) for w in re.findall(r"[a-z0-9]{3,}", " ".join(text))}
        body = " ".join(text)
        if any(re.search(r"(?<![\d.,])" + re.escape(fig) + r"\s*-?\s*" + re.escape(self.stem(unit)), body) for fig, unit in figs):
            return True
        return any(all(w in have for w in words) for words in wants)
    def vl_room(self) -> tuple[int, int]:
        left = max(0, self.ask_room() - self.msgs_sent) if self.asking and self.role("message") else 0
        return left, self.moves_left() - self.order_due() - 2 * left
    def vl_values(self, k: int | None) -> list[list[list[str]]]:
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
        words = list(dict.fromkeys(w for w in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", label.lower().replace("\u2019", "'")))
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
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        props = props if isinstance(props, dict) else {}
        return tuple(sorted((key, val) for key, val in args.items() if isinstance(val, str) and isinstance(props.get(key), dict)
                            and isinstance(props[key].get("enum"), list) and len(props[key]["enum"]) <= 4))
    def vl_set(self, text: str) -> frozenset:
        return frozenset(self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[: self.lead_words or None])
    def vl_seen(self, text: str, tkey: str, mode: dict[str, Any]) -> int | None:
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
        if self.__dict__.get("label_indexed") is not None or not self.archive.items:
            return
        words = {self.stem(w) for w in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", label.lower())) if len(w) >= 2 and w not in LINK_WORDS}
        hits = 0
        def visit(d: dict[str, Any]) -> None:
            nonlocal hits
            if self.row_mark(d) not in self.archive.ids or self.shelf_of(d) != label.strip().lower():
                return
            other = {self.stem(w) for key, v in d.items() if isinstance(v, str) and not shelf_key(key)
                     for w in re.findall(r"[a-z0-9]+", v.lower())}
            hits += bool(words - other)
        walk_dicts(self.archive.items[-1], visit, ceiling=3)
        if hits >= 2:
            self.label_indexed = True
    def vl_queue(self, k: int | None, ids: Any, floor: int, most: int) -> list[str]:
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
        self.rank_checked = {mark for mark in self.rank_checked if self.shelf_known(mark.split("|")[0]) not in homes}
        self.order_vetted = False
        self.open_queue = queue
        self.open_recheck = True
        self.open_say = say
        self.open_purpose, self.open_floor = "walk", floor
        self.open_named()
    def vl_search(self) -> bool:
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
        tool = self.role("search")
        jschema = self.shapes.get(tool or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        found = self.page_arg()
        if not props or not found or not is_number(bound):
            return None
        told = next((str(tdef["function"].get("description") or "") for tdef in self.toolset if tdef["function"]["name"] == tool), "")
        parts = [part.lower() for part in re.split(r"[.;]", told)]
        tok = self.token_key()
        cat_key, cat_val = None, None
        for key, spec in props.items():
            if not isinstance(spec, dict) or key in (tok, found[0]) or not shelf_key(key):
                continue
            if isinstance(spec.get("enum"), list):
                hit = next((v for v in spec["enum"] if isinstance(v, str) and v.strip().lower() == label.strip().lower()), None)
                if hit is not None:
                    cat_key, cat_val = key, hit
                    break
            elif spec.get("type") == "string":
                cat_key, cat_val = key, label
                break
        bound_key = next((key for key, spec in props.items() if isinstance(spec, dict) and spec.get("type") in ("integer", "number")
                          and not spec.get("enum") and key != found[0] and re.search(r"ceil|max|price|cap$|budget|upper|most", str(key).lower())
                          and not re.search(r"page|first|offset|skip|count|size|num", str(key).lower())), None)
        if not cat_key or not bound_key:
            return None
        args: dict[str, Any] = {cat_key: cat_val, bound_key: int(bound) if props[bound_key].get("type") == "integer" or float(bound).is_integer() else bound}
        for key, spec in props.items():
            if key == cat_key or not isinstance(spec, dict) or not isinstance(spec.get("enum"), list) or not 2 <= len(spec["enum"]) <= 4:
                continue
            hit = next((v for v in spec["enum"] if any(re.search(r"\b" + re.escape(str(v).lower()) + r"\b", part)
                                                       and re.search(r"\b" + re.escape(cat_key.lower()) + r"\b", part) for part in parts)), None)
            if hit is None:
                return None
            args[key] = hit
        for key in jschema.get("required") or []:
            if key not in args:
                return None
        fixed, faults = ArgCheck.execute(args, jschema)
        return fixed if not faults and isinstance(fixed, dict) else None
    def ceil_seed(self) -> bool:
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
                def visit(d: dict[str, Any]) -> None:
                    long_strings([val for val in d.values() if isinstance(val, (str, list))], lns)
                walk_dicts(obs, visit, ceiling=4)
        elif isinstance(obs, dict):
            lns = self.notice_lines(obs, refused)
        return list(dict.fromkeys(lns))
    @staticmethod
    def notice_lines(obs: Any, refused: bool = False) -> list[str]:
        lns: list[str] = []
        if not isinstance(obs, dict):
            return lns
        def visit(d: dict[str, Any]) -> None:
            if d is not obs and len(d) > 6:
                return
            for piece in d.values():
                if isinstance(piece, list) and piece and all(isinstance(val, str) for val in piece):
                    long_strings(piece, lns)
                elif isinstance(piece, str) and len(piece.split()) >= 4 and (refused or (len(d) <= 3 and len(piece) <= 600 and not any(
                        isinstance(v, str) and is_id_key(k) and re.fullmatch(ID_LIKE, v) for k, v in d.items()))):
                    lns.append(piece)
        walk_dicts(obs, visit, ceiling=4)
        return list(dict.fromkeys(lns))
    def screen(self, proposal: dict[str, Any], permitted: set[str], scripted: bool, batch: int, queued_messages: int) -> tuple[str | None, str, dict[str, Any]]:
        func = proposal.get("function") or {}
        call_name, raw_args = func.get("name"), func.get("arguments")
        if call_name not in self.shapes:
            return f"Not run (no step used): '{call_name}' is not one of the tools. Use one of: {', '.join(sorted(permitted))}.", call_name, {}
        if not scripted and call_name not in permitted:
            return "Not run (no step used): too few steps are left for this; only these tools can be used now: " + ", ".join(sorted(permitted)) + ".", call_name, {}
        call_args = raw_args if isinstance(raw_args, dict) else (json_object(raw_args) if (raw_args or "").strip() else {})
        if call_args is None:
            return "Not run (no step used): the arguments were not a valid JSON object.", call_name, {}
        call_args, faults = ArgCheck.execute(call_args, self.shapes[call_name])
        if faults:
            return "Not run (no step used): invalid arguments: " + "; ".join(faults[:6]), call_name, call_args
        if call_name in self.finishers() and batch > 1:
            return "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls.", call_name, call_args
        lister = self.batch_tool() if not scripted else None
        if lister and call_name == lister[0] and isinstance(call_args.get(lister[1]), list) and len(call_args[lister[1]]) > lister[2]:
            return (f"Not run (no step used): {call_name} takes at most {lister[2]} ids a call here (a longer list is refused and reads "
                    f"nothing): send the first {lister[2]} now and the rest in your next reply."), call_name, call_args
        if call_name == self.role("cart") and not scripted and self.role("message") and self.trolley.units:
            try:
                left = (self.msg_cap if self.msg_cap is not None else ASK_DEFAULT) - self.msgs_sent - queued_messages
                new = {line: units for line, units in self.trolley.lines(call_args)}
                changes = new != {k: float(v) for k, v in self.trolley.units.items()} and bool(new)
                dead = set(self.trolley.units) & (self.dead_lines() | self.refused_lines)
                mark = tuple(sorted((k, v) for k, v in new.items() if v is not None))
                held_once = self.__dict__.setdefault("told_cart_held", set())
                if left <= 0 and changes and not dead and self.basket_told_ok() and mark not in held_once:
                    held_once.add(mark)
                    self.log.append({"kind": "told_cart_hold", "cart": dict(self.trolley.units), "proposed": new})
                    return ("Not run (no step used): the shopper was told this basket and no message is left to tell them another; "
                            "an order of a different basket than the one told fails. Keep the cart as told and order it, unless a line "
                            "is sold out or breaks a requirement in force (then send the change again)."), call_name, call_args
            except Exception as err:
                self.log.append({"kind": "told_cart_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if call_name == self.role("message") and self.msg_cap is not None and self.msgs_sent + queued_messages >= self.msg_cap:
            return "Not run (no step used): the message budget is used up. Do not message the shopper again.", call_name, call_args
        if call_name == self.role("message") and not scripted:
            said = " ".join(str(v) for v in call_args.values() if isinstance(v, str)).strip()
            newest = str(self.needs.exchanges[-1]).split("shopper:", 1)[-1] if self.needs.exchanges else ""
            if said and "?" not in said and not self.report_asked() and "?" not in newest:
                self.log.append({"kind": "report_refused", "text": said[:200]})
                return ("Not run (no step used): this message asks the shopper nothing, and no instruction in force calls for a "
                        "message. Each message is kept for a question about a property."), call_name, call_args
        if call_name == self.role("message") and not scripted and not self.trolley.units and self.msg_reserve() \
                and self.msgs_sent + queued_messages >= self.ask_room():
            return ("Not run (no step used): the messages left are kept for telling the shopper the basket once something is in "
                    "the cart (and for any finishing step the instructions state). Put your pick in the cart first; after that "
                    "the message tool is yours again."), call_name, call_args
        if call_name == self.role("message") and not scripted and self.trolley.units and self.msg_reserve() \
                and self.msgs_sent + queued_messages >= self.ask_limit() - 1:
            said = " ".join(str(v) for v in call_args.values() if isinstance(v, str))
            if "?" in said and not re.search(r"\b(?:ok|okay|shall i|should i|go ahead|proceed|confirm|happy with|alright|all right|"
                                             r"good to go|approve)\b", said.lower()):
                self.log.append({"kind": "kept_last_message", "text": said[:200]})
                return ("Not run (no step used): this is your last message, kept for the stated finishing step. Send it once the cart "
                        "holds the final pick: the basket and its total, with no new question in it (an answer could change the cart "
                        "after the total was told, with no message left to tell the new one)."), call_name, call_args
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
        if not scripted and call_name == self.role("search"):
            try:
                again = self.repeat_search_note(call_args)
            except Exception as err:
                again = ""
                self.log.append({"kind": "repeat_search_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if again:
                return again, call_name, call_args
        held = "" if scripted else self.late_hold(call_name, call_args)
        held = held or ("" if scripted else self.value_guard(call_name, call_args))
        if held:
            return held, call_name, call_args
        if call_name == self.role("cart"):
            refusal = self.screen_cart(call_args, sig)
            if refusal:
                return refusal, call_name, call_args
        return None, call_name, call_args
    def search_walk(self, call_args: dict[str, Any]) -> tuple[dict[str, Any] | None, Any]:
        found = self.page_arg()
        if not found:
            return None, None
        key, mode, low, _ = found
        at = call_args.get(key, low)
        at = (at.strip() if isinstance(at, str) else "") if mode == "token" else at
        def shape(args: dict[str, Any]) -> tuple:
            free = " ".join(self.free_text(args)).lower().split()
            fixed = {k: v for k, v in args.items() if k != key and not (isinstance(v, str) and v in self.free_text(args))}
            return (" ".join(free), json.dumps(fixed, sort_keys=True, default=str))
        mine = shape({k: v for k, v in call_args.items() if k != key})
        for walk in self.__dict__.get("page_book", {}).values():
            if shape(walk["args"]) == mine and at in walk["read"]:
                return walk, at
        return None, at
    def repeat_search_note(self, call_args: dict[str, Any]) -> str:
        walk, at = self.search_walk(call_args)
        if not walk:
            return ""
        mark = (json.dumps(walk["args"], sort_keys=True, default=str), str(at))
        told = self.__dict__.setdefault("repeat_told", set())
        if mark in told:
            return ""
        told.add(mark)
        key = (self.page_arg() or ("",))[0]
        nxt = self.next_page(walk)
        phrase = " ".join(self.free_text(walk["args"])).strip()
        what = f"'{phrase}'" if phrase else "this search"
        shown = (f"{walk['total']} matches, " if walk.get("total") else "") + f"{len(walk['read'])} page(s) read"
        self.log.append({"kind": "repeat_search", "phrase": phrase, "at": str(at), "next": nxt})
        if nxt is None:
            return (f"Not run (no step used): {what} was already searched ({shown}) and every page of it was read; its rows are "
                    "above. Search another phrase: the kind word with a required value, or another kind word.")
        return (f"Not run (no step used): {what} was already searched ({shown}); that page is above. To see more of its matches, "
                f"send the same call with {key} = {json.dumps(nxt)}; otherwise search another phrase.")
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
    def lit_rows(self) -> dict[str, dict[str, Any]]:
        memo = self.__dict__.get("lit_memo")
        if memo and memo[0] == len(self.archive.items):
            return memo[1]
        rows = self.code_rows()
        self.lit_memo = (len(self.archive.items), rows)
        return rows
    def literal_lines(self, thing: str) -> list[str]:
        reqs = self.needs.props_left()
        if not self.needs.items():
            return reqs
        k = self.product_index(thing)
        out = []
        for req in reqs:
            j = self.product_of(str(req).partition(":")[0])
            if j is None or k is None or j == k:
                out.append(req)
        return out
    def literal_verdicts(self, thing: str) -> list[tuple[str, str, str]]:
        info = self.lit_rows().get(thing) or {}
        pairs = [(str(k), str(v)) for k, v in (info.get("fields") or {}).items() if str(v).strip() and not junk_key(k)]
        title = str(info.get("title") or "")
        kind = str(info.get("shelf") or self.shelf_known(thing) or "")
        out: list[tuple[str, str, str]] = []
        for req in self.literal_lines(thing)[:12]:
            body = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", str(req))
            head, _, raw = body.partition(":")
            prop = re.split(r"\s*\(", head)[0].strip()
            value = " ".join(raw.split())
            if not prop or not value:
                continue
            if ref_text(value) or re.search(r"as shown|same as|the page|sold by|refer|pointed|counterpart", value.lower()):
                out.append((f"{prop}: {value}", "unproven", "its value is given by another listing"))
                continue
            verdict, why = lit_judge(f"{prop}: {value}", title, pairs, kind)
            if verdict == "fail" and why.startswith(("the title",)) is False:
                field = why.split(":")[0].strip()
                if not same_prop(field, prop):
                    verdict, why = "unproven", why + " (a field of another name)"
            out.append((f"{prop}: {value}", verdict, why))
        return out
    def cart_literal(self) -> tuple[list[str], list[str], int]:
        fails, unproven, judged = [], [], 0
        for thing in list(self.trolley.units):
            if thing not in self.archive.read:
                continue
            judged += 1
            for req, verdict, why in self.literal_verdicts(thing):
                if verdict == "fail":
                    fails.append(f"{thing}: '{req}' - {why}")
                elif verdict == "unproven":
                    unproven.append(f"{thing}: '{req}' - {why}")
        return fails, unproven, judged
    def literal_fits(self) -> list[str]:
        rows_ = self.needs.items()
        cap = self.needs.spend_cap
        homes = self.home_shelves() if rows_ else [set(self.shelf_labels())]
        if not homes or any(not home for home in homes):
            return []
        info = self.lit_rows()
        dead = self.dead_lines()
        picks: list[str] = []
        total = 0.0
        for k, home in enumerate(homes):
            qty = rows_[k]["quantity"] if rows_ else self.needs.quantity()
            pool = []
            for ident in sorted(self.archive.read):
                if ident in self.refused_lines or ident in dead or ident in picks or self.shelf_known(ident) not in home:
                    continue
                if (info.get(ident) or {}).get("stock") is False:
                    continue
                cost = self.archive.price(ident)
                if not is_number(cost) or (is_number(cap) and cost * qty > cap):
                    continue
                verdicts = self.literal_verdicts(ident)
                if not verdicts or any(verdict != "ok" for _, verdict, _ in verdicts):
                    continue
                pool.append((float(cost), ident))
            if not pool:
                return []
            cost, ident = min(pool)
            picks.append(ident)
            total += cost * qty
        if is_number(cap) and total > cap:
            return []
        if any(row.get("depends_on") is not None and row["depends_on"] != k and row["depends_on"] < len(picks)
               and self.fit_pair(picks[k], picks[row["depends_on"]])[0] == "no" for k, row in enumerate(rows_ or [])):
            return []
        return picks
    def literal_cheaper(self, line: str) -> str:
        mine = self.archive.price(line)
        if not is_number(mine):
            return ""
        rows_ = self.needs.items()
        k = self.product_index(line) if rows_ else None
        home = set(self.home_shelves()[k]) if rows_ and k is not None and k < len(self.home_shelves()) else {self.shelf_known(line)} - {""}
        qty = rows_[k]["quantity"] if rows_ and k is not None else self.needs.quantity()
        cap = self.needs.spend_cap
        info = self.lit_rows()
        dead = self.dead_lines()
        best: tuple[float, str] | None = None
        for ident in sorted(self.archive.read):
            if ident == line or ident in self.trolley.units or ident in self.refused_lines or ident in dead:
                continue
            if home and self.shelf_known(ident) not in home:
                continue
            if (info.get(ident) or {}).get("stock") is False:
                continue
            cost = self.archive.price(ident)
            if not is_number(cost) or cost >= mine or (is_number(cap) and cost * qty > cap):
                continue
            verdicts = self.literal_verdicts(ident)
            if not verdicts or any(verdict != "ok" for _, verdict, _ in verdicts):
                continue
            if not self.fit_swap_ok(line, ident):
                continue
            if best is None or cost < best[0]:
                best = (float(cost), ident)
        return best[1] if best else ""
    def basket_told_ok(self) -> bool:
        edits = self.__dict__.get("cart_edits", 0)
        said = [s for _, s, at in self.__dict__.get("msg_log", []) if at == edits]
        if not said or not self.trolley.units:
            return False
        text = str(said[-1]).lower().replace(",", "")
        info = self.lit_rows()
        def title_words(thing: str) -> list[str]:
            row = info.get(thing) or {}
            return list(dict.fromkeys(w for w in re.findall(r"[a-z0-9]{3,}", str(row.get("title") or "").lower()) if w not in LINK_WORDS))[:8]
        def price_forms(thing: str) -> set[str]:
            price = (info.get(thing) or {}).get("price")
            if not is_number(price):
                return set()
            qty = self.trolley.units.get(thing) or 1
            out = set()
            for figure in {float(price), float(price) * float(qty)}:
                out.add(f"{figure:g}")
                out.add(f"{figure / 100:.2f}")
            return out
        def named(thing: str, apart_from: set[str], other_prices: set[str]) -> bool:
            if thing.lower() in text:
                return True
            for price in price_forms(thing) - other_prices:
                if re.search(r"(?<![\d.])" + re.escape(price) + r"(?![\d.]*\d)", text):
                    return True
            words = [w for w in title_words(thing) if w not in apart_from]
            hits = sum(1 for w in words if re.search(r"\b" + re.escape(w), text))
            return bool(words) and hits >= min(2, len(words))
        if not all(named(thing, set(), set()) for thing in self.trolley.units):
            return False
        current_words = {w for thing in self.trolley.units for w in title_words(thing)}
        current_prices = {form for thing in self.trolley.units for form in price_forms(thing)}
        gone = [i for i in self.__dict__.get("ever_held", set()) if i not in self.trolley.units]
        return not any(named(i, current_words, current_prices) for i in gone)
    def paged_ids(self) -> set[str]:
        """Listings whose own page a details, view or side-by-side call has read (a search row is not a page)."""
        tools = {self.role("details"), self.role("view")} - {None}
        batch = self.batch_tool()
        if batch:
            tools.add(batch[0])
        out: set[str] = set()
        for entry in self.log:
            if not isinstance(entry, dict) or entry.get("kind") != "env":
                continue
            for act in entry.get("actions") or []:
                if not isinstance(act, dict) or act.get("name") not in tools:
                    continue
                for val in (act.get("args") or {}).values():
                    for v in (val if isinstance(val, list) else [val]):
                        if isinstance(v, str):
                            out.add(v.strip())
        return out
    def viewed_ids(self, tool: str) -> set[str]:
        """Listings one tool has opened."""
        out: set[str] = set()
        for entry in self.log:
            if not isinstance(entry, dict) or entry.get("kind") != "env":
                continue
            for act in entry.get("actions") or []:
                if isinstance(act, dict) and act.get("name") == tool:
                    for val in (act.get("args") or {}).values():
                        for v in (val if isinstance(val, list) else [val]):
                            if isinstance(v, str):
                                out.add(v.strip())
        return out
    def read_unpaged_lines(self) -> None:
        """A listing is bought only after its own page was read: a search row can leave out what the page states, so a cart
        line whose page was never opened is read before the order, in one side-by-side call when the store has one. When the
        store splits a listing between a details view and a separate page view, a line read only in the details view also
        gets its page view opened (the page can carry notes the details leave out)."""
        unread = [i for i in self.trolley.units if i not in self.paged_ids()]
        details, view = self.role("details"), self.role("view")
        unviewed = [i for i in self.trolley.units if i not in unread and details and view and view != details
                    and i not in self.viewed_ids(view)] if not self.batch_tool() else []
        if unviewed and self.__dict__.get("unviewed_done") != tuple(sorted(unviewed)) and self.moves_left() >= len(unviewed) + 2:
            self.unviewed_done = tuple(sorted(unviewed))
            slab_v = [(f"unviewed_{n}_{self.wire.tick}", view, args) for n, ident in enumerate(unviewed)
                      for args in [self.lookup_call(ident, view)] if args]
            if slab_v:
                self.send_chunk(slab_v)
                self.log.append({"kind": "unviewed_lines", "ids": unviewed})
        if not unread or self.__dict__.get("unpaged_done") == tuple(sorted(unread)):
            return
        self.unpaged_done = tuple(sorted(unread))
        batch = self.batch_tool() if len(unread) >= 2 else None
        slab: list[tuple[str, str, dict[str, Any]]] = []
        if batch:
            tool, arg, most = batch
            slab = [(f"unpaged_{n}_{self.wire.tick}", tool, {arg: unread[i: i + most]}) for n, i in enumerate(range(0, len(unread), most))]
        else:
            tool = self.role("details") or self.role("view")
            for n, ident in enumerate(unread):
                call_args = self.lookup_call(ident, tool) if tool else None
                if call_args:
                    slab.append((f"unpaged_{n}_{self.wire.tick}", tool, call_args))
        if not slab or self.moves_left() < len(slab) + 1:
            return
        self.send_chunk(slab)
        self.log.append({"kind": "unpaged_lines", "ids": unread})
    def screen_finish(self, cid: str, call_name: str, call_args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        if self.call_kind(call_name, call_args) == "order" and self.trolley.units and self.moves_left() >= 2 and self.wire.spare() > 12:
            try:
                self.read_unpaged_lines()
            except Exception as err:
                self.log.append({"kind": "unpaged_error", "error": f"{type(err).__name__}: {err}"[:300]})
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
        if self.call_kind(call_name, call_args) == "order" and self.trolley.units and self.moves_left() > 6:
            try:
                fails, _, _ = self.cart_literal()
            except Exception as err:
                fails = []
                self.log.append({"kind": "literal_error", "error": f"{type(err).__name__}: {err}"[:300]})
            mark = (tuple(sorted((i, float(q)) for i, q in self.trolley.units.items())), self.needs.version)
            told = self.__dict__.setdefault("literal_told", set())
            if fails and mark not in told:
                told.add(mark)
                self.log.append({"kind": "literal_hold", "fails": fails[:6]})
                return ("Not run yet (no step used). Read literally from the records, the cart does not meet the requirements in force: "
                        + "; ".join(fails[:6]) + ". A listing's own field for a property decides, and a field that writes another "
                        "value fails it. Replace the failing line with a listing whose own fields state every requirement (search "
                        "again if none was read), re-read its stock and price, and order. Order again only if the field's value "
                        "does meet the requirement under this store's spellings of it. This is the code's reading of the record, "
                        "not a question for the shopper: spend no message on it."), call_args
        if self.call_kind(call_name, call_args) == "order" and self.moves_left() > 6 and self.needs.priority() == "cheapest":
            try:
                swaps = [(line, self.literal_cheaper(line)) for line in self.trolley.units]
                swaps = [(line, other) for line, other in swaps if other]
            except Exception as err:
                swaps = []
                self.log.append({"kind": "literal_cheaper_error", "error": f"{type(err).__name__}: {err}"[:300]})
            mark = (tuple(sorted(self.trolley.units)), self.needs.version)
            told_cheaper = self.__dict__.setdefault("literal_cheaper_told", set())
            if swaps and mark not in told_cheaper:
                told_cheaper.add(mark)
                self.log.append({"kind": "literal_cheaper_hold", "swaps": swaps})
                said = []
                for line, other in swaps:
                    proof = "; ".join(f"{req} ({why})" for req, _, why in self.literal_verdicts(other)[:5] if why)
                    said.append(f"{other} at {self.archive.price(other):g} states every requirement in force ({proof}) and costs less than "
                                f"{line} at {self.archive.price(line):g}")
                return ("Not run yet (no step used). The cheapest fit is asked for, and read literally from the records " + "; ".join(said)
                        + ". Put the cheaper listing in as that product's line instead, re-read its stock and price, and order. Order "
                        "the cart as it is only if a record shows the cheaper listing failing a requirement in force under this "
                        "store's spellings."), call_args
        if self.call_kind(call_name, call_args) == "order" and self.trolley.units and self.moves_left() > 7 and self.needs.priority() == "cheapest":
            try:
                proof = self.cheap_proof()
            except Exception as err:
                proof = ""
                self.log.append({"kind": "cheap_proof_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if proof:
                return proof, call_args
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
        if self.call_kind(call_name, call_args) == "order" and self.trolley.units and self.moves_left() > 3:
            try:
                viewed = self.view_before_order()
            except Exception as err:
                viewed = ""
                self.log.append({"kind": "view_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if viewed:
                return viewed, call_args
        if self.call_kind(call_name, call_args) == "order":
            try:
                fit_held = self.fit_gate()
            except Exception as err:
                fit_held = ""
                self.log.append({"kind": "fit_gate_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if fit_held:
                return fit_held, call_args
            try:
                swept = self.value_sweep()
            except Exception as err:
                swept = ""
                self.log.append({"kind": "value_sweep_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if swept:
                return swept, call_args
            try:
                cheap = self.cheap_sweep()
            except Exception as err:
                cheap = ""
                self.log.append({"kind": "cheap_sweep_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if cheap:
                return cheap, call_args
            try:
                twins = self.twin_lines()
            except Exception as err:
                twins = []
                self.log.append({"kind": "twin_lines_error", "error": f"{type(err).__name__}: {err}"[:300]})
            mark = ("twins", tuple(sorted(self.trolley.units)))
            told_twins = self.__dict__.setdefault("twin_told", set())
            if twins and mark not in told_twins and self.moves_left() > 3:
                told_twins.add(mark)
                self.log.append({"kind": "twin_hold", "lines": twins})
                return ("Not run yet (no step used). The cart holds two or more listings of one kind: " + "; ".join(twins)
                        + ". The order takes one listing per product asked for, at its quantity as one line, and nothing else; "
                        "an extra listing of a kind fails it. Keep the one that fits best (the cheaper when both fit), set its "
                        "quantity, take the other out, and order. If the request truly asks for different listings of that kind, "
                        "send the order again."), call_args
        if self.call_kind(call_name, call_args) == "order" and self.moves_left() > 2 and self.wire.spare() > LAST_CALL + 25:
            try:
                deps = self.dependency_faults()
            except Exception:
                deps = []
            if self.moves_left() <= 5 or self.wire.spare() <= CLOSING:
                deps = [d for d in deps if " is not known" not in d]
            mark = (tuple(sorted(self.trolley.units)), tuple(deps))
            told = self.__dict__.setdefault("dep_told", {})
            if deps and told.get(mark, 0) < 2 and len(told) < 4:
                told[mark] = told.get(mark, 0) + 1
                return ("Not run yet (no step used). The request says one product must work with another: " + "; ".join(deps)
                        + ". Put in a line whose own statement of fit, or another listing's statement naming its part's code, covers "
                        "the other product's code (a family covers its members unless it excepts that code), or take another "
                        "listing of the other product that the line fits; then order. Never decline while such a set is in budget."), call_args
            try:
                cheaper = self.kit_cheaper() if self.moves_left() > 5 and self.wire.spare() > CLOSING else ""
            except Exception as err:
                cheaper = ""
                self.log.append({"kind": "kit_cheaper_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if cheaper:
                self.log.append({"kind": "kit_cheaper", "cart": dict(self.trolley.units)})
                return cheaper, call_args
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
            try:
                kit = self.kit_hold()
            except Exception as err:
                kit = ""
                self.log.append({"kind": "kit_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if kit:
                self.log.append({"kind": "kit_hold", "cart": dict(self.trolley.units)})
                return kit, call_args
            try:
                fits = self.literal_fits() if self.moves_left() > 2 else []
            except Exception as err:
                fits = []
                self.log.append({"kind": "literal_fits_error", "error": f"{type(err).__name__}: {err}"[:300]})
            told_fits = self.__dict__.setdefault("literal_decline_told", {})
            key = (tuple(fits), self.needs.version)
            if fits and told_fits.get(key, 0) < 2:
                told_fits[key] = told_fits.get(key, 0) + 1
                shown = []
                for fit in fits:
                    proof = "; ".join(f"{req} ({why})" for req, _, why in self.literal_verdicts(fit)[:5] if why) or "no property requirement is left"
                    cost = self.archive.price(fit)
                    shown.append(f"{fit} at {cost:g} ({proof})" if is_number(cost) else f"{fit} ({proof})")
                self.log.append({"kind": "literal_decline_hold", "fits": fits})
                if told_fits[key] == 2 and self.role("cart") and self.moves_left() >= len(fits) + 4 and self.wire.spare() > CLOSING:
                    picks = [(fit, (rows_[n]["quantity"] if (rows_ := self.needs.items()) and n < len(rows_) else self.needs.quantity())) for n, fit in enumerate(fits)]
                    was = self.wire.mute
                    try:
                        bad = self.code_put(picks)
                    except Exception as err:
                        bad = None
                        self.log.append({"kind": "decline_cart_error", "error": f"{type(err).__name__}: {err}"[:300]})
                    finally:
                        self.wire.mute = was
                    self.log.append({"kind": "decline_cart", "picks": picks, "bad": bad, "cart": dict(self.trolley.units)})
                    if bad == [] and all(fit in self.trolley.units for fit, _ in picks):
                        return ("Not run yet (no step used). A decline is wrong while a read listing states every requirement in force "
                                "in its own fields, is in stock and is within the budget: " + " | ".join(shown) + ". The code has put "
                                + ("each in the cart as its product's line, at the quantity asked" if self.needs.items() else "it in the cart at the quantity asked")
                                + ": " + self.trolley.summary() + " Re-read the lines' stock and price, tell the shopper the basket if a "
                                "message is left, and order. Decline only if a record shows a line failing a requirement now in force."), call_args
                return ("Not run yet (no step used). A decline is wrong while a read listing states every requirement in force in "
                        "its own fields, is in stock and is within the budget. The records show " + ("one for every product asked for: "
                        if self.needs.items() else "one: ") + " | ".join(shown) + ". Put " + ("each in the cart as its product's line, "
                        "at the quantity asked" if self.needs.items() else "it in the cart at the quantity asked") + ", re-read stock "
                        "and price, and order. Decline only if a record shows it failing a requirement now in force."), call_args
            try:
                scan = self.decline_scan()
            except Exception as err:
                scan = ""
                self.log.append({"kind": "decline_scan_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if scan:
                return scan, call_args
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
            best = self.unrefuted() if not self.__dict__.get("best_put") and self.moves_left() > 4 else []
            for _ in range(3 if not self.__dict__.get("best_held") else 0):
                unchecked = [i for i in best if f"{i}|{self.needs.version}" not in self.passed]
                if not unchecked or self.wire.spare() < CLOSING + 30:
                    break
                self.check_item(unchecked[0], "ORDER")
                best = self.unrefuted()
            if best and self.__dict__.get("best_held") and self.role("cart") and self.moves_left() >= len(best) + 4 and self.wire.spare() > CLOSING:
                self.best_put = True
                rows_ = self.needs.items()
                picks = [(ident, (rows_[n]["quantity"] if rows_ and n < len(rows_) else self.needs.quantity())) for n, ident in enumerate(best)]
                was = self.wire.mute
                try:
                    bad = self.code_put(picks)
                except Exception as err:
                    bad = None
                    self.log.append({"kind": "best_cart_error", "error": f"{type(err).__name__}: {err}"[:300]})
                finally:
                    self.wire.mute = was
                self.log.append({"kind": "best_cart", "picks": picks, "bad": bad, "cart": dict(self.trolley.units)})
                if bad == [] and all(ident in self.trolley.units for ident, _ in picks):
                    return ("Not run yet (no step used). A decline is right only when no listing can meet the requirements, and the records "
                            "hold " + ("a candidate for every product asked for" if rows_ else "a candidate") + " that is in stock, within the "
                            "budget and that no check ruled out. The code has put " + ("each in the cart as its product's line, at the quantity "
                            "asked" if rows_ else "it in the cart at the quantity asked") + ": " + self.trolley.summary() + " Re-read the lines' "
                            "stock and price, tell the shopper the basket if a message is left, and order. Decline only if a record shows a "
                            "line failing a requirement now in force."), call_args
            if best and not self.__dict__.get("best_held") and self.moves_left() > len(best) + 3 and self.wire.spare() > CLOSING + 20:
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
            waited = self.finish_steps("decline")
            if waited:
                return waited, call_args
            return None, self.scrub_decline(call_args, call_name)
        if self.call_kind(call_name, call_args) == "order":
            waited = self.finish_steps("order")
            if waited:
                return waited, call_args
        return None, call_args
    def change_review(self) -> str:
        if self.__dict__.get("reviews", 0) >= 3 or not self.archive.read or self.wire.mute or self.wire.spare() < CLOSING + 20 \
                or not self.wire.usable(CHECKER) or self.moves_left() < 3:
            return ""
        self.reviews = self.__dict__.get("reviews", 0) + 1
        cards = self.archive.cards(14000, "", self.ref_foreign())
        if not cards:
            return ""
        ask_text = CHANGE_REVIEW + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request + self.needs.changes()
        ask_text += "\n\nWHAT JUST CHANGED: " + " / ".join(self.needs.exchanges[-2:])
        ask_text += "\n\nREQUIREMENTS NOW: " + self.needs.text_body()
        ask_text += f"\n\nPRIORITY: {self.needs.priority()} {self.needs.reading.get('priority_detail') or ''}".rstrip()
        ask_text += "\n\nTHE CART: " + self.trolley.summary()
        ask_text += "\n\nLISTINGS SEEN (DETAILS = its full record was read, ROW = seen only in search rows):\n" + "\n".join(cards)
        ask_text += self.field_names()
        answer = self.wire.consult(ask_text, 2500, CHECKER)
        self.log.append(dict(kind="change_review", answer=answer))
        if not answer:
            return ""
        fits = [str(x) for x in answer.get("fits") or [] if isinstance(x, (str, int)) and str(x) in self.archive.ids][:6]
        pick = answer.get("pick")
        pick = str(pick) if isinstance(pick, (str, int)) and str(pick) in self.archive.ids else ""
        self.__dict__.setdefault("named", set()).update(fits)
        parts = ["[re-check after the change]"]
        parts.append(("opened listings that meet every requirement now in force, best first: " + ", ".join(fits)) if fits
                     else "none of the opened listings is shown to meet every requirement now in force")
        if pick:
            parts.append(f"the one to take now: {pick}")
        if answer.get("cart_still_right") is True and self.trolley.units:
            parts.append("the cart's line still fits: keep it, so the basket already told stays the one ordered")
        elif answer.get("cart_still_right") is False and self.trolley.units:
            parts.append("the cart's line is no longer the right pick under the conditions now: replace it (when the instructions "
                         "ask you to check with the shopper before replacing anything, ask first)")
        if answer.get("reason"):
            parts.append(str(answer["reason"])[:300])
        parts.append("Act on this before anything else: re-pick among what fits now (the cheapest or best of them when the request "
                     "asks for that), fix the cart, re-read its stock and price, then finish; never decline while an opened listing fits.")
        return " ".join(parts)
    def code_asks_swap(self, old: list[str], new: list[str]) -> bool:
        tool = self.role("message")
        aim = self.text_arg(tool) if tool else None
        if not aim or self.ask_limit() - self.msgs_sent < 1 + self.msg_reserve() or self.moves_left() < 4:
            return False
        text = ("Before I change the basket: I would like to take out " + ", ".join(old[:3]) + " and put in " + ", ".join(new[:3])
                + " instead, since the first no longer fits. Is that all right with you?")
        answers = self.send_chunk([(f"code_swap_ask_{self.wire.tick}", tool, {aim: text})])
        self.log.append({"kind": "code_swap_ask", "old": old[:3], "new": new[:3]})
        return bool(answers) and not str(answers[0][1]).startswith("Not run")
    def line_seller(self, thing: str) -> str:
        learned = self.seller_keys()
        found: list[str] = []
        def visit(d: dict[str, Any]) -> None:
            if thing not in [v for k, v in d.items() if isinstance(v, str) and is_id_key(k)]:
                return
            for k, v in d.items():
                if isinstance(v, str) and v.strip() and len(v) <= 60 and "id" not in str(k).lower() and (
                        re.search(SELLER_KEYS, str(k).lower()) or str(k) in learned):
                    found.append(v.strip())
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return found[-1] if found else ""
    @staticmethod
    def squash(text: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(text).lower())
    @staticmethod
    def fit_sentences(row: dict[str, Any]) -> str:
        code = r"[A-Za-z]-?\d{2,}|\d{2,}[xX*?]{1,4}\b|\b\d{3,}\b"
        text = str(row.get("text") or "")
        pieces = [p.strip() for p in re.split(r"(?<=[.;!?])\s+|\n+", text) if re.search(code, p)]
        pieces += [f"{k}: {v}" for k, v in (row.get("fields") or {}).items() if re.search(code, str(v))]
        return f"{row.get('title') or ''} | " + (" ".join(pieces)[:400] or text[:240])
    @staticmethod
    def fam_form(pattern: Any) -> str:
        low = re.sub(r"[^a-z0-9?*]", "", str(pattern).lower())
        return re.sub(r"(?<=\d)x+$", lambda m: "?" * len(m.group(0)), low)
    @staticmethod
    def fam_covers(pattern: Any, code: str) -> bool:
        low = str(pattern).lower()
        if ".." in low:
            ends = [re.fullmatch(r"([a-z]*)(\d+)", re.sub(r"[^a-z0-9]", "", end)) for end in low.split("..", 1)]
            mine = re.fullmatch(r"([a-z]*)(\d+)", code)
            if not mine or not ends[0] or not ends[1]:
                return False
            (head, lo), (top_head, hi) = ends[0].groups(), ends[1].groups()
            return mine.group(1) == head and top_head in ("", head) and len(lo) == len(hi) == len(mine.group(2)) \
                and int(lo) <= int(mine.group(2)) <= int(hi)
        form = Errand.fam_form(low)
        if not re.search(r"[?*]", form):
            return False
        return bool(re.fullmatch(re.escape(form).replace(r"\?", "[a-z0-9]").replace(r"\*", "[a-z0-9]*"), code))
    @staticmethod
    def value_codes(value: Any) -> set[str]:
        return {re.sub(r"[^a-z0-9]", "", c) for c in re.findall(r"[a-z]+-?\d+[a-z0-9]*|\b\d{3,}[a-z0-9]*", str(value).lower())}
    def fit_keep(self, answer: Any, texts: dict[str, str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for ident, text in texts.items():
            got = answer.get(ident) if isinstance(answer, dict) else None
            if not isinstance(got, dict):
                continue
            pool = self.squash(text)
            def listed(src: dict[str, Any], key: str) -> list[Any]:
                vals = src.get(key)
                return vals if isinstance(vals, list) else [vals] if isinstance(vals, (str, int)) and not isinstance(vals, bool) else []
            def held(code: Any) -> bool:
                body = self.squash(code)
                return len(body) >= 2 and body in pool
            def fam_ok(pattern: Any) -> bool:
                low = str(pattern).lower()
                if ".." in low:
                    return all(len(self.squash(end)) >= 2 and (self.squash(end) in pool or len(re.sub(r"\D", "", end)) >= 2
                                                              and re.sub(r"\D", "", end) in pool) for end in low.split("..", 1))
                form = self.fam_form(low)
                fixed = re.split(r"[?*]", form, 1)[0]
                return len(fixed) >= 2 and fixed != form and fixed in pool
            def codes(src: dict[str, Any], key: str) -> list[str]:
                return [str(x).strip() for x in listed(src, key) if isinstance(x, (str, int)) and not isinstance(x, bool) and held(x)][:12]
            claims = []
            for claim in listed(got, "claims"):
                if not isinstance(claim, dict):
                    continue
                about = str(claim.get("about") or "self").strip()
                if about.lower() != "self" and not held(about):
                    continue
                kept = {"about": "self" if about.lower() == "self" else about, "fits": codes(claim, "fits"),
                        "families": [str(x) for x in listed(claim, "families") if isinstance(x, str) and fam_ok(x)][:6],
                        "except": codes(claim, "except"), "see": codes(claim, "see")}
                if kept["fits"] or kept["families"] or kept["see"]:
                    claims.append(kept)
            out[ident] = {"own": codes(got, "own")[:6], "claims": claims}
        return out
    def fit_main_codes(self, facts: dict[str, dict[str, Any]]) -> list[str]:
        rows = self.needs.items()
        targets = {r["depends_on"] for k, r in enumerate(rows) if r.get("depends_on") is not None and r["depends_on"] != k
                   and r["depends_on"] < len(rows)}
        if not targets or not facts:
            return []
        homes = self.home_shelves()
        shelves = set().union(*(homes[j] for j in targets if j < len(homes))) if homes else set()
        info = self.lit_rows()
        changed = []
        for i, f in facts.items():
            if f["own"] or not shelves or (info.get(i) or {}).get("shelf") not in shelves:
                continue
            mine = [c for c in f["claims"] if c["about"] == "self"]
            named = list(dict.fromkeys(x for c in mine for x in c["fits"]))
            if mine and 1 <= len(named) <= 2 and not any(c["families"] or c["except"] or c["see"] for c in mine):
                f["own"] = named
                f["claims"] = [c for c in f["claims"] if c["about"] != "self"]
                changed.append(i)
        return changed
    def fit_open(self) -> set[str]:
        facts = self.__dict__.get("fit_facts") or {}
        known = {self.squash(c) for f in facts.values() for c in f["own"]}
        return {self.squash(p) for f in facts.values() for c in f["claims"] for p in c["see"]} - known - {""}
    def fit_read(self, first: Any = (), least: int = 1, soft: bool = False) -> None:
        facts = self.__dict__.setdefault("fit_facts", {})
        self.fit_main_codes(facts)
        tries = self.__dict__.get("fit_tries", 0)
        if tries >= 5 or (soft and tries >= 2) or not self.wire.usable(FIT_MODELS) or self.wire.spare() < CLOSING + 30 \
                or self.wire.spent_out():
            return
        info = self.lit_rows()
        rows = self.needs.items()
        linked = {k for k, r in enumerate(rows) if r.get("depends_on") is not None}
        linked |= {r["depends_on"] for r in rows if r.get("depends_on") is not None}
        homes = self.home_shelves() if linked else []
        shelves = set().union(*(homes[k] for k in linked if k < len(homes))) if homes else set()
        given = [str(t) for t in first if t]
        wanted = self.fit_open()
        pointed = [i for i, row in info.items() if wanted and any(p in self.squash(row.get("title")) for p in wanted)]
        cheap = sorted((i for i in info if is_number(info[i].get("price"))), key=lambda i: info[i]["price"])
        todo = [i for i in dict.fromkeys(given + list(self.trolley.units) + pointed + sorted(self.archive.read) + cheap)
                if i in info and i not in facts and (i in given or info[i].get("shelf") in shelves)]
        if not todo or (len(todo) < least and not (set(given) | set(self.trolley.units) | set(pointed)) & set(todo)):
            return
        texts = {i: self.fit_sentences(info[i]) for i in todo[:14]}
        self.fit_tries = tries + 1
        answer = self.wire.consult(FIT_READ + "\n\nLISTINGS:\n" + "\n".join(f"{i}: {t}" for i, t in texts.items()), 5000, FIT_MODELS)
        kept = self.fit_keep(answer, texts) if answer else {}
        if answer:
            for i in texts:
                facts[i] = kept.get(i) or {"own": [], "claims": []}
        self.log.append({"kind": "fit_read", "try": self.fit_tries, "asked": list(texts), "kept": kept,
                         "as_own": self.fit_main_codes(facts)})
    def fit_verdict(self, thing: str, codes: set[str]) -> tuple[str | None, str, list[str]]:
        facts = self.__dict__.get("fit_facts") or {}
        mine = facts.get(thing)
        codes = {c for c in codes if c}
        if not mine or not codes:
            return None, "", []
        own = {self.squash(c) for c in mine["own"]} - {""}
        title = self.squash((self.lit_rows().get(thing) or {}).get("title"))
        def me(code: str) -> bool:
            body = self.squash(code)
            return body in own or (len(body) >= 4 and body in title)
        claims = [c for c in mine["claims"] if c["about"] == "self" or me(c["about"])]
        claims += [c for i, f in facts.items() if i != thing for c in f["claims"] if c["about"] != "self" and me(c["about"])]
        if not claims:
            return None, "", []
        for c in claims:
            if any(self.squash(x) in codes for x in c["except"]):
                continue
            hit = next((x for x in c["fits"] if self.squash(x) in codes), "") or next(
                (p for p in c["families"] if any(self.fam_covers(p, code) for code in codes)), "")
            if hit:
                return "yes", hit, []
        unread = self.fit_open()
        left = list(dict.fromkeys(p for c in claims for p in c["see"] if self.squash(p) in unread))
        if left:
            return "unknown", "", left
        barred = [x for c in claims for x in c["except"] if self.squash(x) in codes]
        named = list(dict.fromkeys(x for c in claims for x in c["fits"] + c["families"]))
        return "no", (f"they name {barred[0]} as one it does not fit" if barred else "they name " + ", ".join(named[:6])), []
    def fit_pair(self, one: str, other: str) -> tuple[str | None, str, list[str]]:
        facts = self.__dict__.get("fit_facts") or {}
        def codes(thing: str) -> set[str]:
            return {self.squash(c) for c in (facts.get(thing) or {}).get("own") or []} - {""}
        found = self.fit_verdict(one, codes(other))
        return found if found[0] is not None else self.fit_verdict(other, codes(one))
    def fit_code(self, thing: str) -> str:
        return " / ".join(((self.__dict__.get("fit_facts") or {}).get(thing) or {}).get("own", [])[:2])
    def fit_cost(self, thing: str) -> str:
        cost = (self.lit_rows().get(thing) or {}).get("price")
        return f"{thing} ({cost:g})" if is_number(cost) else thing
    def fit_pool(self, k: int, anchor: str) -> tuple[list[str], list[tuple[str, list[str]]], int]:
        facts = self.__dict__.get("fit_facts") or {}
        info = self.lit_rows()
        homes = self.home_shelves()
        home = homes[k] if k < len(homes) else set()
        fits, waits, nos = [], [], 0
        for i in facts:
            row = info.get(i) or {}
            if i == anchor or not row or row.get("shelf") not in home or row.get("stock") is False or i in self.refused_lines:
                continue
            found, _, left = self.fit_pair(i, anchor)
            if found == "yes":
                fits.append(i)
            elif found == "unknown":
                waits.append((i, left))
            elif found == "no":
                nos += 1
        fits.sort(key=lambda i: info[i]["price"] if is_number(info[i].get("price")) else 10 ** 12)
        return fits, waits, nos
    def fit_line(self, thing: str) -> tuple[str | None, str]:
        rows = self.needs.items()
        if not rows or not any(r.get("depends_on") is not None for r in rows):
            return None, ""
        k = self.product_index(thing)
        if k is None:
            return None, ""
        for line in self.trolley.units:
            j = self.product_index(line)
            if line == thing or j is None or j == k or not (rows[k].get("depends_on") == j or rows[j].get("depends_on") == k):
                continue
            self.fit_read([thing, line])
            found, why, left = self.fit_pair(thing, line)
            name = f"the {rows[j]['product']} in the cart ({line}" + (f", {self.fit_code(line)})" if self.fit_code(line) else ")")
            if found == "yes":
                return found, f"it fits {name}: a statement of fit names {why}, so the requirement that the two work together is met."
            if found == "no" and rows[k].get("depends_on") == j:
                return found, f"it must work with {name}, and the statements of fit do not cover it ({why})"
            if found == "no":
                return "clash", (f"{name} must work with it, and the statements of fit do not cover it ({why}): one of the two "
                                 "must change, judge this item on its own requirements")
            if found == "unknown":
                return found, (f"whether it fits {name} waits on part {' / '.join(left)}, which its statements send the reader to "
                               "and which was never read: not shown either way")
        return None, ""
    def fit_swap_ok(self, line: str, other: str) -> bool:
        rows = self.needs.items()
        if not rows or not any(r.get("depends_on") is not None for r in rows):
            return True
        k = self.product_index(line)
        for m in self.trolley.units:
            j = self.product_index(m)
            if m == line or k is None or j is None or j == k or not (rows[k].get("depends_on") == j or rows[j].get("depends_on") == k):
                continue
            new = self.fit_pair(other, m)[0]
            if new == "no" or (new != "yes" and self.fit_pair(line, m)[0] == "yes"):
                return False
        return True
    def dependency_faults(self) -> list[str]:
        rows = self.needs.items()
        if not rows or not any(r.get("depends_on") is not None for r in rows):
            return []
        held: dict[int, str] = {}
        for line in self.trolley.units:
            k = self.product_index(line)
            if k is not None:
                held.setdefault(k, line)
        self.fit_read(list(held.values()))
        faults = []
        for k, row in enumerate(rows):
            j = row.get("depends_on")
            if j is None or j == k or j >= len(rows) or k not in held or j not in held:
                continue
            anchor, dep = held[j], held[k]
            found, why, left = self.fit_pair(dep, anchor)
            name = f"the {rows[j]['product']} {anchor}" + (f" ({self.fit_code(anchor)})" if self.fit_code(anchor) else "")
            if found == "no":
                others = [self.fit_cost(i) for i in self.fit_pool(k, anchor)[0] if i != dep][:3]
                faults.append(f"the {row['product']} {dep} must work with {name}, but the statements of fit do not cover it ({why})"
                              + (f"; read {row['product']} listings that fit it: " + ", ".join(others) if others else ""))
            elif found == "unknown":
                faults.append(f"whether the {row['product']} {dep} fits {name} is not known: its statements send the reader to part "
                              f"{' / '.join(left)} for the rest of their list, never read - search that code, open the listing and "
                              "read what it says about this part")
        return faults
    def dependency_hint(self) -> str:
        rows = self.needs.items()
        if not rows or not any(r.get("depends_on") is not None for r in rows):
            return ""
        self.fit_read(least=6, soft=True)
        facts = self.__dict__.get("fit_facts") or {}
        info = self.lit_rows()
        homes = self.home_shelves()
        out = []
        for k, row in enumerate(rows):
            j = row.get("depends_on")
            if j is None or j == k or j >= len(rows) or j >= len(homes):
                continue
            held = next((line for line in self.trolley.units if self.product_index(line) == j), "")
            anchors = [held] if held else sorted(
                (i for i in facts if facts[i]["own"] and (info.get(i) or {}).get("shelf") in homes[j]
                 and (info.get(i) or {}).get("stock") is not False and i not in self.refused_lines),
                key=lambda i: info[i]["price"] if is_number(info[i].get("price")) else 10 ** 12)[:3]
            parts = []
            for a in anchors:
                fits, waits, nos = self.fit_pool(k, a)
                code = self.fit_code(a)
                bits = ["fit it: " + ", ".join(self.fit_cost(i) for i in fits[:4]) if fits else f"no read {row['product']} fits it yet"
                        + (f" (search the {row['product']} with {code} as the phrase; when that finds nothing, read the rows of "
                           f"the {row['product']}'s category page by page)" if code and held else "")]
                if waits:
                    bits.append("known once a part they send you to is read: " + ", ".join(f"{i} (part {' / '.join(p)})" for i, p in waits[:3]))
                if nos:
                    bits.append(f"{nos} read do not fit it")
                parts.append(f"{rows[j]['product']} {a}" + (f" ({code})" if code else "") + ": " + "; ".join(bits))
            if parts:
                out.append(f"{row['product']} for the {rows[j]['product']}" + (" in the cart" if held else "") + " - " + " | ".join(parts))
        sets = self.kit_sets()
        if sets:
            out.append("sets that fit together, none shown sold out or failing a requirement by its own record, within the budget, "
                       "cheapest first: " + "; ".join(f"{a} + {' + '.join(p)} = {t:g}" for t, a, p in sets[:3]))
        return ("\n[works with] (the code's reading of the listings' own statements of fit: a family covers its members, an exception "
                "wins, a statement naming another part's code counts for that part) " + "; ".join(out)) if out else ""
    def kit_rank(self, thing: str) -> int:
        return 0 if all(verdict == "ok" for _, verdict, _ in self.literal_verdicts(thing)) else 1
    def kit_sets(self) -> list[tuple[float, str, list[str]]]:
        rows = self.needs.items()
        main = self.main_product()
        if main is None:
            return []
        facts = self.__dict__.get("fit_facts") or {}
        key = (len(facts), len(self.archive.items), self.needs.version, len(self.refused_lines))
        memo = self.__dict__.get("kit_memo")
        if memo and memo[0] == key:
            return memo[1]
        info = self.lit_rows()
        homes = self.home_shelves()
        cap = self.needs.spend_cap
        dead = self.dead_lines()
        deps = [k for k, r in enumerate(rows) if r.get("depends_on") == main and k != main]
        ruled: dict[str, bool] = {}
        def out(i: str) -> bool:
            if i not in ruled:
                ruled[i] = i in self.refused_lines or i in dead or (info.get(i) or {}).get("stock") is False or any(
                    verdict == "fail" for _, verdict, _ in self.literal_verdicts(i))
            return ruled[i]
        sets = []
        if deps and main < len(homes):
            for a in facts:
                held = info.get(a) or {}
                if held.get("shelf") not in homes[main] or not is_number(held.get("price")) or out(a):
                    continue
                total, picks = held["price"] * rows[main]["quantity"], []
                for k in deps:
                    fits = sorted((i for i in self.fit_pool(k, a)[0] if is_number(info[i].get("price")) and not out(i)),
                                  key=self.kit_rank)
                    if not fits:
                        break
                    picks.append(fits[0])
                    total += info[fits[0]]["price"] * rows[k]["quantity"]
                else:
                    if not is_number(cap) or total <= cap:
                        sets.append((total, a, picks))
        sets.sort(key=lambda st: (sum(self.kit_rank(i) for i in [st[1]] + st[2]), st[0], st[1]))
        self.kit_memo = (key, sets)
        return sets
    def kit_cheaper(self) -> str:
        rows = self.needs.items()
        main = self.main_product()
        if main is None or self.needs.priority() != "cheapest":
            return ""
        deps = [k for k, r in enumerate(rows) if r.get("depends_on") == main and k != main]
        held: dict[int, str] = {}
        for line in self.trolley.units:
            k = self.product_index(line)
            if k is not None:
                held.setdefault(k, line)
        if not deps or main not in held or any(k not in held for k in deps):
            return ""
        info = self.lit_rows()
        mine = [held[main]] + [held[k] for k in deps]
        costs = [(info.get(i) or {}).get("price") for i in mine]
        if not all(is_number(c) for c in costs):
            return ""
        total = sum(c * self.trolley.held(i) for c, i in zip(costs, mine))
        told = self.__dict__.setdefault("kit_cheaper_told", set())
        mark = tuple(sorted(mine))
        sets = self.kit_sets() if mark not in told and len(told) < 2 else []
        if not sets or sets[0][0] >= total - 1e-9 or [sets[0][1]] + sets[0][2] == mine:
            return ""
        told.add(mark)
        t, a, p = sets[0]
        return (f"Not run yet (no step used). The request asks for the cheapest, and by the listings' own statements of fit a "
                f"cheaper set fits together: {a} with {', '.join(p)} (total {t:g}, against {total:g} for the cart's lines). Check "
                "its lines against every requirement now in force, the shopper's answers included; if they meet them, put that "
                "set in the cart and order; if one fails, order the cart again.")
    def fit_clash(self, k: int, thing: str, picks: list[tuple[str, int]]) -> bool:
        rows = self.needs.items()
        if not rows or k >= len(rows):
            return False
        return any(j < len(rows) and j != k and (rows[k].get("depends_on") == j or rows[j].get("depends_on") == k)
                   and self.fit_pair(thing, pick)[0] == "no" for j, (pick, _) in enumerate(picks))
    def basket_ledger(self) -> str:
        rows = self.needs.items()
        if not rows or not self.trolley.units:
            return ""
        info = self.lit_rows()
        held: dict[int | None, list[tuple[str, float]]] = {}
        for line, units in self.trolley.units.items():
            held.setdefault(self.product_index(line), []).append((line, units))
        parts, total = [], 0.0
        for k, row in enumerate(rows):
            bits = []
            for line, units in held.get(k) or []:
                cost = (info.get(line) or {}).get("price")
                total += cost * units if is_number(cost) else 0.0
                bits.append(f"{line} x{int(units)}" + (f" at {cost:g}" if is_number(cost) else ""))
            parts.append(f"{row['product']}: " + (", ".join(bits) if bits else "none yet"))
        if held.get(None):
            parts.append("lines matched to no product: " + ", ".join(line for line, _ in held[None]))
        if all(held.get(k) for k in range(len(rows))) and not held.get(None):
            self.full_cart = dict(self.trolley.units)
        cap = self.needs.spend_cap
        return ("\n[basket] " + "; ".join(parts) + f"; total {total:g}"
                + (f" of the budget {cap:g} ({cap - total:g} left)" if is_number(cap) else ""))
    def restore_full(self) -> None:
        rows = self.needs.items()
        full = self.__dict__.get("full_cart")
        if not rows or not full or full == dict(self.trolley.units) or not self.role("cart") or self.moves_left() < 3:
            return
        have = {self.product_index(line) for line in self.trolley.units}
        if all(k in have for k in range(len(rows))):
            return
        dead = self.dead_lines()
        if any(i in dead or i in self.refused_lines for i in full):
            return
        index = {i: self.product_index(i) for i in full}
        order = [(next((x for x in full if index[x] == j), ""), 1) for j in range(len(rows))]
        if any(index[i] is not None and self.fit_clash(index[i], i, order) for i in full):
            return
        was = self.wire.mute
        try:
            self.wire.mute = True
            bad = self.code_put([(i, int(u)) for i, u in full.items()])
        finally:
            self.wire.mute = was
        self.log.append({"kind": "full_restore", "basket": full, "bad": bad, "cart": dict(self.trolley.units)})
    def kit_hold(self) -> str:
        rows = self.needs.items()
        if not rows or not any(r.get("depends_on") is not None for r in rows) or self.__dict__.get("kit_held", 0) >= 2 \
                or self.moves_left() < 4 or self.wire.spare() < CLOSING + 30:
            return ""
        self.fit_read()
        facts = self.__dict__.get("fit_facts") or {}
        info = self.lit_rows()
        homes = self.home_shelves()
        cap = self.needs.spend_cap
        sets, waits = [], {}
        for j in sorted({r["depends_on"] for r in rows if r.get("depends_on") is not None}):
            deps = [k for k, r in enumerate(rows) if r.get("depends_on") == j and k != j]
            if j >= len(rows) or j >= len(homes) or not deps:
                continue
            for a in facts:
                held = info.get(a) or {}
                if held.get("shelf") not in homes[j] or held.get("stock") is False or a in self.refused_lines or not is_number(held.get("price")):
                    continue
                total, picks = held["price"] * rows[j]["quantity"], []
                for k in deps:
                    fits, wait, _ = self.fit_pool(k, a)
                    waits.update(dict(wait))
                    fits = [i for i in fits if is_number(info[i].get("price"))]
                    if not fits:
                        break
                    picks.append(fits[0])
                    total += info[fits[0]]["price"] * rows[k]["quantity"]
                else:
                    if not is_number(cap) or total <= cap:
                        sets.append((total, a, picks))
        if self.main_product() is not None:
            sets = list(self.kit_sets())
        if not sets and not waits:
            return ""
        self.kit_held = self.__dict__.get("kit_held", 0) + 1
        sets.sort()
        text = "Not run yet (no step used). The products asked for must work together, and a decline is wrong while they can."
        if sets:
            text += (" By the listings' own statements of fit (a family covers its members, an exception wins, a statement naming "
                     "another part's code counts for that part) these sets fit together, none shown sold out, within the budget: "
                     + "; ".join(f"{a} with {', '.join(p)} (total {t:g})" for t, a, p in sets[:3])
                     + ". Judge each line's other requirements and buy a set that meets them.")
        if waits:
            text += (" These listings send the reader to a part never read for the rest of their list: "
                     + ", ".join(f"{i} (part {' / '.join(p)})" for i, p in list(waits.items())[:4]) + "; search each code and open it.")
        return text + " Decline only if every set fails a requirement now in force."
    def kit_repair(self) -> str:
        rows = self.needs.items()
        if not rows or not any(r.get("depends_on") is not None for r in rows) or not self.trolley.units or not self.role("cart"):
            return ""
        held: dict[int, str] = {}
        for line in self.trolley.units:
            k = self.product_index(line)
            if k is not None:
                held.setdefault(k, line)
        self.fit_read(list(held.values()))
        bad = [k for k, row in enumerate(rows) if row.get("depends_on") is not None and row["depends_on"] != k
               and k in held and row["depends_on"] in held and self.fit_pair(held[k], held[row["depends_on"]])[0] == "no"]
        if not bad:
            return ""
        info = self.lit_rows()
        cap = self.needs.spend_cap
        dead = self.dead_lines()
        def usable(i: str) -> bool:
            return i not in dead and i not in self.refused_lines and (info.get(i) or {}).get("stock") is not False \
                and is_number((info.get(i) or {}).get("price")) and not any(v == "fail" for _, v, _ in self.literal_verdicts(i))
        kept = {held[k]: self.trolley.held(held[k]) or rows[k]["quantity"] for k in held}
        swap_lines: dict[str, Any] | None = dict(kept)
        swapped = []
        for k in bad:
            fits = sorted((i for i in self.fit_pool(k, held[rows[k]["depends_on"]])[0] if usable(i)), key=self.kit_rank)
            if not fits:
                swap_lines = None
                break
            swap_lines.pop(held[k], None)
            swap_lines[fits[0]] = rows[k]["quantity"]
            swapped.append(fits[0])
        if swap_lines is not None:
            total = sum(info[i]["price"] * n for i, n in swap_lines.items() if is_number((info.get(i) or {}).get("price")))
            if is_number(cap) and total > cap:
                swap_lines = None
        set_lines: dict[str, Any] | None = None
        main = self.main_product()
        sets = self.kit_sets() if main is not None else []
        if sets:
            _, a, picks = sets[0]
            deps = [k for k, r in enumerate(rows) if r.get("depends_on") == main and k != main]
            set_lines = {a: rows[main]["quantity"]}
            set_lines.update({p: rows[k]["quantity"] for k, p in zip(deps, picks)})
            set_lines.update({held[k]: kept[held[k]] for k in held if k != main and k not in deps})
        swap_ok = swap_lines is not None and all(self.kit_rank(i) == 0 for i in swapped)
        set_ok = set_lines is not None and all(self.kit_rank(i) == 0 for i in set_lines)
        lines = next((c for c, good in ((swap_lines, swap_ok), (set_lines, set_ok), (swap_lines, True), (set_lines, True))
                      if c is not None and good), None)
        if not lines:
            self.log.append({"kind": "kit_repair", "bad": bad, "result": "misfit"})
            return "misfit"
        was = self.wire.mute
        try:
            self.wire.mute = True
            failed = self.code_put([(i, int(n)) for i, n in lines.items()])
        finally:
            self.wire.mute = was
        self.log.append({"kind": "kit_repair", "bad": bad, "lines": lines, "failed": failed, "cart": dict(self.trolley.units)})
        return "fixed" if failed == [] else ""
    def kit_push(self) -> str:
        if not self.linked_products():
            return ""
        rows = self.needs.items()
        have = {self.product_index(line) for line in self.trolley.units}
        if all(k in have for k in range(len(rows))):
            return ""
        used = max(self.wire.tick / max(1, self.step_cap or 1), (monotonic() - self.wire.t0) / DEADLINE)
        if used < 0.45:
            return ""
        sets = self.kit_sets()
        text = ("\n[kit] Half of the errand is used and the cart does not hold a whole set yet: put one set in the cart now with "
                "ONE cart call holding every product's line, then check each line against its requirements and order")
        if sets:
            t, a, picks = sets[0]
            return text + (f"; by the statements of fit the cheapest set within the budget is {a} with {', '.join(picks)} "
                           f"(total {t:g}): check it against the requirements now in force.")
        return text + ": choose the main listing now, and for each other product a listing whose statement of fit covers it."
    def cheap_sweep(self) -> str:
        if self.needs.priority() != "cheapest" or self.needs.items() or self.orders_tried or self.__dict__.get("cheap_swept"):
            return ""
        search = self.role("search")
        if not search or not self.trolley.units or self.moves_left() < 12 or self.wire.spare() < CLOSING + 40:
            return ""
        self.cheap_swept = True
        props = (self.shapes.get(search) or {}).get("properties") or {}
        need = set((self.shapes.get(search) or {}).get("required") or [])
        texts = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and not v.get("enum")]
        if len(texts) != 1 or need - {texts[0]}:
            return ""
        ran = {" ".join(self.free_text(w["args"])).strip().lower() for w in self.__dict__.get("page_book", {}).values()}
        queries = [q for q in self.label_queries() if " " not in q and q not in ran][:2]
        if not queries:
            return ""
        self.send_chunk([(f"cheap_{n}_{self.wire.tick}", search, {texts[0]: q}) for n, q in enumerate(queries)])
        line = next(iter(self.trolley.units))
        price = self.archive.price(line)
        if price is None:
            return ""
        rows = sorted((c, i, t) for i, (c, t, _) in self.shelf_rows().items()
                      if c < price and i not in self.trolley.units and i not in self.archive.read)[:4]
        self.log.append({"kind": "cheap_sweep", "queries": queries, "rows": [i for _, i, _ in rows]})
        if not rows:
            return ""
        self.__dict__.setdefault("named", set()).update(i for _, i, _ in rows)
        return ("Not run yet (no step used). The cheapest fit is asked for, so the code searched " + ", ".join(queries)
                + ": these in-stock rows of the requested category cost less than the cart's line and were never opened: "
                + "; ".join(f"{i} '{t[:70]}' {c:g}" for c, i, t in rows) + ". Open them (several in one reply), and if one meets "
                "every requirement, make it the cart's line, re-read its stock and price, and order; otherwise order the cart as it is.")
    def value_sweep(self) -> str:
        if self.orders_tried or self.__dict__.get("value_swept") or not self.trolley.units:
            return ""
        search = self.role("search")
        if not search or self.moves_left() < 12 or self.wire.spare() < CLOSING + 60:
            return ""
        props = (self.shapes.get(search) or {}).get("properties") or {}
        need = set((self.shapes.get(search) or {}).get("required") or [])
        texts = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and not v.get("enum")]
        tok = self.token_key()
        texts = [k for k in texts if k != tok]
        if len(texts) != 1 or need - {texts[0]}:
            return ""
        self.value_swept = True
        text_key = texts[0]
        page = self.page_arg()
        rows_ = self.needs.items()
        products = [row["product"] for row in rows_] if rows_ else [str(self.needs.reading.get("product") or "")]
        homes = self.home_shelves()
        cheapest = self.needs.priority() == "cheapest"
        held: dict[int, str] = {}
        for line in self.trolley.units:
            k = self.product_index(line) if rows_ else 0
            if k is not None:
                held.setdefault(k, line)
        def norm(text: str) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
        def ran() -> set[str]:
            return {norm(" ".join(self.free_text(w["args"]))) for w in self.__dict__.get("page_book", {}).values()}
        def walk_of(phrase: str) -> dict[str, Any] | None:
            return next((w for w in self.__dict__.get("page_book", {}).values() if norm(" ".join(self.free_text(w["args"]))) == norm(phrase)), None)
        sent: list[str] = []
        values_of: dict[int, list[str]] = {}
        for k, product in enumerate(products[:2]):
            kind, vals = self.value_terms(k if rows_ else None, product)
            if not kind:
                continue
            values_of[k] = vals[:3]
            phrase = " ".join([kind] + vals[:3])
            if len(phrase.split()) > 7:
                phrase = " ".join(phrase.split()[:7])
            todo: list[dict[str, Any]] = []
            if vals and norm(phrase) not in ran():
                todo.append({text_key: phrase})
            for q in ([phrase] if vals else []) + [kind]:
                if len(sent) + len(todo) >= 4 or not cheapest or not page:
                    break
                walk = walk_of(q)
                if walk is None and q == kind and norm(kind) not in ran() and not any(norm(" ".join(self.free_text(t))) == norm(kind) for t in todo):
                    todo.append({text_key: kind})
                elif walk is not None and self.next_page(walk) is not None and walk.get("total", 0) > walk.get("size", 0):
                    todo.append({**walk["args"], page[0]: self.next_page(walk)})
            for call_args in todo:
                if len(sent) >= 4 or self.moves_left() < 9 or self.wire.spare() < CLOSING + 40:
                    break
                self.send_chunk([(f"vsweep_{len(sent)}_{self.wire.tick}", search, call_args)])
                sent.append(" ".join(self.free_text(call_args)) + (" (next page)" if page and page[0] in call_args else ""))
                if cheapest and page and page[0] not in call_args:
                    walk = walk_of(" ".join(self.free_text(call_args)))
                    nxt = self.next_page(walk) if walk else None
                    if walk and nxt is not None and walk.get("total", 0) > walk.get("size", 0) and len(sent) < 4 \
                            and self.moves_left() >= 9 and self.wire.spare() >= CLOSING + 40:
                        self.send_chunk([(f"vsweep_{len(sent)}_{self.wire.tick}", search, {**call_args, page[0]: nxt})])
                        sent.append(" ".join(self.free_text(call_args)) + " (next page)")
        if cheapest and page:
            names = self.ref_key_words()
            def room() -> bool:
                return len(sent) < SWEEP_MOST and self.moves_left() >= 9 and self.wire.spare() >= CLOSING + 40
            for k, product in enumerate(products[:2]):
                kind_set = set(re.findall(r"[a-z0-9]+", self.value_terms(k if rows_ else None, product)[0]))
                for value in values_of.get(k, [])[:2]:
                    words = [w for w in re.findall(r"[a-z0-9]+", value) if w not in LINK_WORDS and w not in names and w not in kind_set]
                    phrase = " ".join(words)
                    if not words or not room():
                        continue
                    if norm(phrase) not in ran():
                        self.send_chunk([(f"vsweep_{len(sent)}_{self.wire.tick}", search, {text_key: phrase})])
                        sent.append(phrase)
                    walk = walk_of(phrase)
                    while walk is not None and 0 < walk.get("total", 0) <= SWEEP_ALL and self.next_page(walk) is not None and room():
                        self.send_chunk([(f"vsweep_{len(sent)}_{self.wire.tick}", search, {**walk["args"], page[0]: self.next_page(walk)})])
                        sent.append(phrase + " (next page)")
                        walk = walk_of(phrase)
        found: list[tuple[float, str, str, str]] = []
        for k, product in enumerate(products[:2]):
            home = set(homes[k]) if k < len(homes) else set()
            line = held.get(k)
            top = self.archive.price(line) if line else None
            words = {w for v in values_of.get(k, []) for w in re.findall(r"[a-z0-9]{3,}", v)}
            for ident, (cost, title, _) in self.shelf_rows().items():
                if ident in self.archive.read or ident in self.trolley.units or (home and self.shelf_known(ident) not in home):
                    continue
                if cheapest:
                    if top is None or cost >= top:
                        continue
                elif not (words and any(re.search(r"\b" + re.escape(w), title.lower()) for w in words)):
                    continue
                found.append((cost, ident, title, product))
        found = sorted(found)[:4]
        self.log.append({"kind": "value_sweep", "sent": sent, "rows": [i for _, i, _, _ in found]})
        if not sent and not found:
            return ""
        self.__dict__.setdefault("named", set()).update(i for _, i, _, _ in found)
        said = ("the code searched " + ", ".join(f"'{q}'" for q in sent)) if sent else "from everything seen"
        if not found:
            return ""
        return ("Not run yet (no step used). Before the first order, " + said + ": these in-stock rows of the requested category "
                + ("cost less than the cart's line for their product" if cheapest else "carry a required value in their title")
                + " and were never opened: " + "; ".join(f"{i} '{t[:70]}' {c:g} ({pr})" for c, i, t, pr in found)
                + ". Open them (several in one reply); if one meets every requirement in force" + (" and costs less" if cheapest else "")
                + ", make it the cart's line for that product, re-read its stock and price, and order; otherwise order the cart as it is.")
    def value_terms(self, k: int | None, product: str) -> tuple[str, list[str]]:
        rows_ = self.needs.items()
        def norm(text: str) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
        kind = " ".join(kind_tokens(product)) or norm(product)
        vals: list[str] = []
        if not kind:
            return "", vals
        for req in self.needs.props_left():
            prop, _, value = str(req).partition(":")
            if rows_ and k is not None and self.product_of(prop) not in (k, None):
                continue
            value = re.sub(r"\s+", " ", value.strip().lower().replace("(", " ").replace(")", " "))
            words = re.findall(r"[a-z0-9]+", value)
            if not words or len(words) > 4 or ref_text(value) or not re.fullmatch(r"[a-z0-9 .\-/%\"']+", value) \
                    or re.search(r"\b(?:at least|at most|under|over|below|above|no more|no less|not|than|same|different|any|or|without)\b", value) \
                    or all(w in kind.split() for w in words):
                continue
            if norm(value) not in [norm(v) for v in vals]:
                vals.append(value)
        return kind, vals
    def value_rows(self) -> list[tuple[int, float, str, str]]:
        rows_ = self.needs.items()
        products = [row["product"] for row in rows_] if rows_ else [str(self.needs.reading.get("product") or "")]
        homes = self.home_shelves() if rows_ else []
        wanted: dict[int | None, set[str]] = {}
        for k, product in enumerate(products):
            kind, vals = self.value_terms(k if rows_ else None, product)
            skip = set(kind.split())
            wanted[k if rows_ else None] = {w for v in vals for w in lit_content(v) if len(w) >= 3 and w not in skip}
        if not any(wanted.values()):
            return []
        cap = self.needs.spend_cap
        dead = self.dead_lines()
        seen: dict[str, tuple[int, float, str]] = {}
        sold: set[str] = set()
        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and is_id_key(key) and val in self.archive.ids), None)
            if not ident or ident in self.archive.read or ident in self.refused_lines or ident in dead:
                return
            if self.sold_out(d):
                sold.add(ident)
                return
            cost = next((val for key, val in d.items() if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower())), None)
            if cost is None:
                return
            pieces = [str(val) for key, val in d.items() if isinstance(val, (str, int, float)) and not isinstance(val, bool)
                      and not (is_id_key(key) and isinstance(val, str) and re.fullmatch(ID_LIKE, val))]
            pieces += [str(v) for val in d.values() if isinstance(val, dict) for v in val.values() if isinstance(v, (str, int, float))]
            have = set(lit_words(" ".join(pieces)))
            k = self.product_index(ident) if rows_ else None
            if rows_ and k is None:
                shelf = self.shelf_of(d)
                k = next((j for j, home in enumerate(homes) if shelf and shelf in home), None)
            words = wanted.get(k) if (k is not None or not rows_) else set().union(*wanted.values())
            hits = sum(1 for w in words or () if any(lit_near(w, h) for h in have))
            if not hits:
                return
            qty = rows_[k]["quantity"] if rows_ and k is not None and k < len(rows_) else self.needs.quantity()
            if is_number(cap) and float(cost) * qty > cap:
                return
            title = d.get("title") if isinstance(d.get("title"), str) else next(
                (val for key, val in d.items() if isinstance(val, str) and 0 < len(val) <= 300 and re.search(r"title|heading|headline|name", str(key).lower())
                 and not is_id_key(key)), "")
            seen[ident] = (hits, float(cost), str(title or "")[:80])
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return sorted(((hits, cost, ident, title) for ident, (hits, cost, title) in seen.items() if ident not in sold),
                      key=lambda r: (-r[0], r[1], r[2]))
    def decline_scan(self) -> str:
        done = self.__dict__.setdefault("decline_scanned", set())
        if self.needs.version in done or len(done) >= 2 or self.wire.spare() < CLOSING + 60 or self.moves_left() < 8:
            return ""
        if not (self.role("details") or self.batch_tool()):
            return ""
        if [i for i in self.trolley.units if i not in self.refused_lines and i not in self.dead_lines()]:
            return ""
        done.add(self.needs.version)
        search = self.role("search")
        page = self.page_arg()
        rows_ = self.needs.items()
        products = [row["product"] for row in rows_] if rows_ else [str(self.needs.reading.get("product") or "")]
        def norm(text: str) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
        def ran() -> set[str]:
            return {norm(" ".join(self.free_text(w["args"]))) for w in self.__dict__.get("page_book", {}).values()}
        def walk_of(phrase: str) -> dict[str, Any] | None:
            return next((w for w in self.__dict__.get("page_book", {}).values() if norm(" ".join(self.free_text(w["args"]))) == norm(phrase)), None)
        sent: list[str] = []
        before = len(self.archive.items)
        text_key = None
        if search:
            props = (self.shapes.get(search) or {}).get("properties") or {}
            need = set((self.shapes.get(search) or {}).get("required") or [])
            tok = self.token_key()
            texts = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and not v.get("enum") and k != tok]
            text_key = texts[0] if len(texts) == 1 and not need - {texts[0]} else None
        if text_key:
            todo: list[dict[str, Any]] = []
            for k, product in enumerate(products[:2]):
                kind, vals = self.value_terms(k if rows_ else None, product)
                if not kind or not vals:
                    continue
                phrase = " ".join(" ".join([kind] + vals[:3]).split()[:7])
                walk = walk_of(phrase)
                if walk is None:
                    todo.append({text_key: phrase})
                elif page and self.next_page(walk) is not None and walk.get("total", 0) > walk.get("size", 0):
                    todo.append({**walk["args"], page[0]: self.next_page(walk)})
                lead = " ".join((vals[0] + " " + kind).split()[:7])
                if len(vals) >= 1 and walk_of(lead) is None and norm(lead) != norm(phrase) and len(todo) < 2:
                    todo.append({text_key: lead})
            for call_args in todo[:2]:
                if self.moves_left() < 8 or self.wire.spare() < CLOSING + 40:
                    break
                self.send_chunk([(f"dscan_{len(sent)}_{self.wire.tick}", search, call_args)])
                sent.append(" ".join(self.free_text(call_args)) + (" (next page)" if page and page[0] in call_args else ""))
        try:
            ranked = self.value_rows()
        except Exception as err:
            ranked = []
            self.log.append({"kind": "value_rows_error", "error": f"{type(err).__name__}: {err}"[:300]})
        batch = self.batch_tool()
        most = batch[2] if batch else (5 if self.moves_left() >= 18 else 3)
        picks = [ident for _, _, ident, _ in ranked][:most]
        if len(picks) < most:
            fresh: dict[str, float] = {}
            cap, qty = self.needs.spend_cap, self.needs.quantity()
            shown: set[str] = set()
            for walk in self.__dict__.get("page_book", {}).values():
                if self.value_score(walk["args"]):
                    shown |= set(walk.get("row_shelf") or {})
            def visit(d: dict[str, Any]) -> None:
                ident = next((val for key, val in d.items() if isinstance(val, str) and is_id_key(key) and val in self.archive.ids), None)
                cost = next((val for key, val in d.items() if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower())), None)
                if ident and cost is not None and ident not in self.archive.read and ident not in self.refused_lines and not self.sold_out(d) \
                        and not (is_number(cap) and float(cost) * qty > cap):
                    fresh.setdefault(ident, float(cost))
            for rec in self.archive.items[before:]:
                walk_dicts(rec, visit)
            late = dict(fresh)
            for rec in self.archive.items[:before]:
                walk_dicts(rec, visit)
            fresh = {i: c for i, c in fresh.items() if i in late or i in shown}
            picks += [ident for ident, _ in sorted(fresh.items(), key=lambda p: (p[1], p[0])) if ident not in picks][: most - len(picks)]
        reads = 1 if batch and len(picks) >= 2 else len(picks)
        while picks and self.moves_left() <= reads + 4:
            picks.pop()
            reads = 1 if batch and len(picks) >= 2 else len(picks)
        self.log.append({"kind": "decline_scan", "sent": sent, "rows": picks, "ranked": [r[2] for r in ranked[:6]]})
        if not sent and not picks:
            return ""
        said = ("Before a decline, the code searched " + ", ".join(f"'{q}'" for q in sent) + " (a decline after a need was heard "
                "most often comes from never searching its value)") if sent else "Before a decline"
        if not picks:
            return ("Not run yet (no step used). " + said + "; the results showed no unread in-stock row within the budget. Decline only "
                    "if nothing seen meets every requirement in force under the value rules.")
        self.__dict__.setdefault("named", set()).update(picks)
        self.open_queue = picks
        self.open_say = "Before declining, I open unread rows that carry a required value."
        self.open_purpose, self.open_floor = "decline", 4
        shown = "; ".join(f"{ident} '{title}' {cost:g}" if title else f"{ident} {cost:g}" for _, cost, ident, title in ranked if ident in picks) or ", ".join(picks)
        return ("Not run yet (no step used). " + said + "; these unread rows " + ("carry a required value in their row text and "
                if ranked else "") + "are in stock and within the budget, so their records are read for you now: " + shown
                + ". Judge each against the requirements in force under the value rules (where this store writes the value itself in a "
                "field, only those words meet it); put one that fits in the cart" + (" (one line per product, at its quantity)" if rows_ else "")
                + ", re-read its stock and price, and order; decline only if each one fails.")
    def query_key(self) -> str | None:
        search = self.role("search")
        if not search:
            return None
        props = (self.shapes.get(search) or {}).get("properties") or {}
        need = set((self.shapes.get(search) or {}).get("required") or [])
        tok = self.token_key()
        texts = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and not v.get("enum") and k != tok]
        return texts[0] if len(texts) == 1 and not need - {texts[0]} else None
    def shelf_tool(self) -> tuple[str, str, str, str | None, bool] | None:
        if "shelf_spec" in self.__dict__:
            return self.shelf_spec
        spec = None
        try:
            taken = {self.role(r) for r in ("details", "message", "cart", "cart_view", "order", "decline")} - {None}
            batch = self.batch_tool()
            if batch:
                taken.add(batch[0])
            for name, jschema in self.shapes.items():
                if name in taken or not isinstance(jschema, dict):
                    continue
                props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
                shelf = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string"
                              and shelf_key(k) and "id" not in str(k).lower()), None)
                cap = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") in ("integer", "number") and not p.get("enum")
                            and re.search(r"ceil|max|price|cap$|cost|budget|upper|most", str(k).lower())
                            and not re.search(r"page|first|offset|skip|count|size|num|min|low", str(k).lower())), None)
                needed = [k for k in (jschema.get("required") or []) if k not in (shelf, cap)]
                if not shelf or not cap or needed:
                    continue
                tok = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and k != shelf and not p.get("enum")
                            and re.search(r"token|cursor|next|continu|page|bookmark", str(k).lower())), None)
                spec = (name, shelf, cap, tok, (props.get(cap) or {}).get("type") == "integer")
                if name == self.role("search"):
                    break
        except Exception as err:
            spec = None
            self.log.append({"kind": "shelf_tool_error", "error": f"{type(err).__name__}: {err}"[:300]})
        self.shelf_spec = spec
        return spec
    def shelf_call(self, label: str, bound: Any, token: str | None = None) -> tuple[str, dict[str, Any]] | None:
        spec = self.shelf_tool()
        if not spec or not str(label or "").strip():
            return None
        name, skey, ckey, tok, whole = spec
        props = (self.shapes.get(name) or {}).get("properties") or {}
        val: Any = label
        if isinstance((props.get(skey) or {}).get("enum"), list):
            val = next((v for v in props[skey]["enum"] if isinstance(v, str) and v.strip().lower() == label.strip().lower()), None) \
                or self.code_label(name, kind_words(label))
            if not val:
                return None
        args: dict[str, Any] = {skey: val}
        if is_number(bound) and float(bound) > 0:
            args[ckey] = int(bound) if whole or float(bound).is_integer() else round(float(bound), 2)
        if token and tok:
            args[tok] = token
        fixed, faults = ArgCheck.execute(args, self.shapes.get(name) or {})
        return (name, fixed) if not faults and isinstance(fixed, dict) else None
    def scripted_calls(self, say: str, calls: list[tuple[str, dict[str, Any]]], tag: str) -> list[Any]:
        proposals = [{"id": f"{tag}_{n}_{self.wire.tick}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                     for n, (name, args) in enumerate(calls)]
        if not proposals:
            return []
        self.chat.append({"role": "assistant", "content": say, "tool_calls": proposals})
        before = len(self.archive.items)
        self.covering = True
        try:
            responses = self.dispatch(proposals, scripted=True)
        finally:
            self.covering = False
        self.post_replies(responses)
        return self.archive.items[before:]
    def fit_codes(self) -> list[tuple[int | None, str]]:
        if "fit_code_list" in self.__dict__:
            return self.fit_code_list
        self.fit_code_list: list[tuple[int | None, str]] = []
        if not re.search(r"\d", self.request):
            return self.fit_code_list
        answer = self.wire.consult(FIT_CODES + "\n\nREQUEST:\n" + self.request, 700, READER)
        found = answer.get("fits") if isinstance(answer, dict) else None
        rows_ = self.needs.items()
        for entry in found if isinstance(found, list) else []:
            if not isinstance(entry, dict):
                continue
            code = str(entry.get("code") or "").strip()
            parts = re.findall(r"[a-z0-9]+", code.lower())
            # a code holds a digit and at least four letters or digits, and the request writes it
            if not parts or not re.search(r"\d", code) or len("".join(parts)) < 4 or not self.fit_pattern(code).search(self.request):
                continue
            k = self.product_of(str(entry.get("product") or "")) if rows_ else None
            if rows_ and k is None:
                continue
            if (k, code) not in self.fit_code_list:
                self.fit_code_list.append((k, code))
        self.log.append({"kind": "fit_codes", "answer": found if isinstance(found, list) else None, "kept": self.fit_code_list})
        return self.fit_code_list
    @staticmethod
    def fit_pattern(code: str) -> Any:
        parts = re.findall(r"[a-z0-9]+", code.lower())
        return re.compile(r"(?<![a-z0-9])" + r"[\s\-_./]*".join(re.escape(x) for x in parts) + r"(?![a-z0-9])", re.I)
    def fit_rows(self, code: str, home: set[str]) -> tuple[list[dict[str, Any]], int]:
        pattern = self.fit_pattern(code)
        rows: dict[str, dict[str, Any]] = {}
        total: set[str] = set()
        def visit(d: dict[str, Any]) -> None:
            ident = next((v for key, v in d.items() if isinstance(v, str) and is_id_key(key) and v in self.archive.ids), None)
            if not ident or len(d) < 4 or (home and self.shelf_of(d) not in home):
                return
            total.add(ident)
            text = " ".join(str(v) for v in d.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool))
            for v in d.values():
                if isinstance(v, dict):
                    text += " " + " ".join(f"{a} {b}" for a, b in v.items() if isinstance(b, (str, int, float)))
            if pattern.search(text):
                rows[ident] = {**rows.get(ident, {}), **d}
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return list(rows.values()), len(total)
    def fit_sweep(self) -> bool:
        if self.wire.done or self.wire.broke or self.orders_tried or self.wire.spare() < CLOSING + 40:
            return False
        codes = self.fit_codes()
        search, page = self.role("search"), self.page_arg()
        if not codes or not search or not page or page[1] == "token":
            return False
        swept = self.__dict__.setdefault("fit_swept", set())
        todo = [(k, code) for k, code in codes if (k, code) not in swept]
        if not todo:
            return False
        key, by_page, low, high = page
        book = self.__dict__.get("page_book", {})
        homes = self.home_shelves()
        rows_ = self.needs.items()
        room = self.cover_room(2) - 3 * len(todo)
        per = room // max(1, len(todo))
        if per < 2:
            return False
        def rest(w: dict[str, Any]) -> list[Any]:
            read = {v for v in w["read"] if is_number(v)}
            if not read:
                return []
            start = min([low] + list(read))
            size = max(1, w["size"])
            step = 1 if by_page else size
            out = []
            for n in range(1, 400):
                val = start + n * step
                if (high is not None and val > high) or n * size >= w["total"]:
                    break
                if val not in read:
                    out.append(val)
            return out
        names = [set(kind_tokens(row["product"])) for row in rows_] if rows_ else []
        for k, code in todo:
            home = set(homes[k]) if k is not None and k < len(homes) else (set(homes[0]) if homes and not rows_ else set())
            if k is not None and home:
                # a shelf that is also another product's home says little about this product
                alone = home - set().union(*(set(h) for j, h in enumerate(homes) if j != k))
                home = alone or home
            product = rows_[k]["product"] if rows_ and k is not None else str(self.needs.reading.get("product") or "")
            own = set(kind_tokens(product))
            if k is not None and names:
                # words this product shares with another product say little about which product a search is for
                own = (own - set().union(*(names[j] for j in range(len(names)) if j != k))) or own
            best: tuple[int, int, dict[str, Any], list[Any]] | None = None
            for w in book.values():
                if not w.get("first") or not w.get("size"):
                    continue
                said = own & set(kind_tokens(" ".join(self.free_text(w["args"]))))
                if home:
                    # most of the search's rows sit on the product's own shelf (a value searched alone spans many shelves)
                    shelves = list((w.get("row_shelf") or {}).values())
                    share = sum(1 for x in shelves if x in home) / max(1, len(shelves))
                    if share < 0.5 and not (said and share >= 0.25):
                        continue
                elif not said:
                    continue
                vals = rest(w)
                if not vals or len(vals) > min(per, FIT_PAGES):
                    continue
                if best is None or w["total"] > best[0]:
                    best = (w["total"], -len(vals), w, vals)
            if best is None:
                continue
            swept.add((k, code))
            _, _, walk, vals = best
            query = " ".join(self.free_text(walk["args"]))
            calls = [(search, {**walk["args"], key: v}) for v in vals]
            self.scripted_calls(f"The fit code {code} is in the listings' text, which the search does not read: I read every remaining page "
                                f"of '{query}' and keep the rows that name {code}.", calls, "fit_p")
            found, seen = self.fit_rows(code, home)
            buy = sorted((r for r in found if not self.sold_out(r)), key=lambda r: self.archive.price(self.row_mark(r)) or 0.0)
            gone = [self.row_mark(r) for r in found if self.sold_out(r)]
            def line(r: dict[str, Any]) -> str:
                facts = next((v for v in r.values() if isinstance(v, dict) and v), None)
                cost = self.archive.price(self.row_mark(r))
                return (f"{self.row_mark(r)} {cost if cost is not None else '?'} '{self.title_of(r)[:90]}' [{self.shelf_of(r)}]"
                        + (f" {json.dumps(facts, default=str)[:160]}" if facts else ""))
            where = f" on the {'/'.join(sorted(home))} shelf" if home else ""
            if buy:
                summary = (f"[fit sweep] {len(vals)} more pages of '{query}' read ({walk['total']} matches). Rows naming {code}{where}: "
                           f"{len(found)} of {seen} rows seen. In stock, cheapest first: " + "; ".join(line(r) for r in buy[:12])
                           + (f". Sold out: {', '.join(gone[:6])}" if gone else "")
                           + f". Only these rows say they fit {code}; a row's other requirements are on its page - read the cheapest"
                           " ones and choose among them.")
            else:
                summary = (f"[fit sweep] {len(vals)} more pages of '{query}' read ({walk['total']} matches): no in-stock row{where} names "
                           f"{code}" + (f" (sold out: {', '.join(gone[:6])})" if gone else "") + ". Search the "
                           f"{product} under its plain name or other words; the code itself is not searchable.")
            ids: set[str] = set()
            for m in reversed(self.chat):
                if m.get("role") == "assistant" and any(str(c.get("id", "")).startswith("fit_p_") for c in m.get("tool_calls") or []):
                    ids = {c["id"] for c in m["tool_calls"]}
                    break
            replies = [m for m in self.chat if m.get("role") == "tool" and m.get("tool_call_id") in ids]
            for m in replies[:-1]:
                m["content"] = "Page read; its rows are summed up in the last reply."
            if replies:
                replies[-1]["content"] = summary + "\n" + self.status_line()
            self.log.append({"kind": "fit_sweep", "product": k, "code": code, "query": query, "pages": len(vals),
                             "rows": [self.row_mark(r) for r in buy[:12]], "sold_out": gone[:6], "seen": seen})
            return True
        return False
    def fit_gate(self) -> str:
        codes = self.__dict__.get("fit_code_list") or []
        if not codes or not self.trolley.units:
            return ""
        mark = json.dumps(sorted(self.trolley.units), default=str)
        held = self.__dict__.setdefault("fit_held", set())
        if mark in held:
            return ""
        rows_ = self.needs.items()
        bad: list[tuple[str, int | None, list[str]]] = []
        for line in self.trolley.units:
            k = self.product_index(line) if rows_ else None
            need = [c for kk, c in codes if kk == k]
            if not need:
                continue
            texts: list[str] = []
            def visit(d: dict[str, Any]) -> None:
                if any(isinstance(v, str) and v == line and is_id_key(key) for key, v in d.items()):
                    texts.append(json.dumps(d, default=str))
            for rec in self.archive.items:
                walk_dicts(rec, visit)
            if texts and not any(self.fit_pattern(c).search(" ".join(texts)) for c in need):
                bad.append((line, k, need))
        if not bad:
            return ""
        held.add(mark)
        parts = []
        for line, k, need in bad:
            product = rows_[k]["product"] if rows_ and k is not None else "the product"
            home = set(self.home_shelves()[k]) if k is not None and k < len(self.home_shelves()) else set()
            found, _ = self.fit_rows(need[0], home)
            buy = sorted((r for r in found if not self.sold_out(r)), key=lambda r: self.archive.price(self.row_mark(r)) or 0.0)
            seen = "; ".join(f"{self.row_mark(r)} {self.archive.price(self.row_mark(r))} '{self.title_of(r)[:70]}'" for r in buy[:5])
            parts.append(f"the line {line} for the {product} never names {need[0]}, the code the request ties it to"
                         + (f" (rows seen that name it: {seen})" if seen else " (no row seen names it yet)"))
        self.log.append({"kind": "fit_gate", "lines": [b[0] for b in bad]})
        return ("Not run yet (no step used). " + "; ".join(parts) + ". Replace each such line with a listing whose text names the code and"
                " that meets its other requirements, then order; if no such listing can be found, decide whether the request can still be met.")
    def cover_room(self, spare_msgs: int = 3) -> int:
        return self.moves_left() - self.order_due() - 2 * max(0, self.ask_room() - self.msgs_sent) - spare_msgs
    def cover_step(self) -> bool:
        if self.__dict__.get("covered") or self.wire.done or self.wire.broke:
            return False
        self.covered = True
        search, text_key, page = self.role("search"), self.query_key(), self.page_arg()
        if not search or not text_key or not page or self.wire.spare() < CLOSING + 60:
            return False
        rows_ = self.needs.items()
        products = ([row["product"] for row in rows_] if rows_ else [str(self.needs.reading.get("product") or "")])[
            :5]
        products = [p for p in products if str(p).strip()]
        if not products:
            return False
        n = len(products)
        room = self.cover_room()
        if room < 3:
            self.log.append({"kind": "cover_skip", "room": room})
            return False
        refs = bool(self.needs.reading.get("references"))
        plan = [1 if refs else (COVER_PAGES if n == 1 else 2 if n == 2 else 1), 1 if refs else (COVER_LISTS if n == 1 else 1),
                1 if is_number(self.needs.spend_cap) and float(self.needs.spend_cap) > 0 and self.shelf_tool() else 0]
        def cost() -> int:
            return n * (plan[0] + plan[1] + plan[2])
        for cut in ("shelf", "page", "list", "page"):
            if cost() <= room:
                break
            if cut == "shelf":
                plan[2] = 0
            elif cut == "page":
                plan[0] = max(1, plan[0] - 1)
            else:
                plan[1] = max(1, plan[1] - 1)
        while cost() > room and n > 1:
            n -= 1
            products = products[:n]
        if cost() > room:
            self.log.append({"kind": "cover_skip", "room": room, "plan": plan})
            return False
        book = self.__dict__.setdefault("page_book", {})
        def norm(text: Any) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
        def walk_of(args: dict[str, Any]) -> dict[str, Any] | None:
            base = {k: v for k, v in args.items() if k != page[0]}
            found = book.get(json.dumps(base, sort_keys=True, default=str))
            if found:
                return found
            words = norm(" ".join(self.free_text(base)))
            return next((w for w in book.values() if norm(" ".join(self.free_text(w["args"]))) == words), None)
        leads: list[dict[str, Any] | None] = []
        calls: list[tuple[str, dict[str, Any]]] = []
        for k, _ in enumerate(products):
            try:
                lead = next((c for c in self.code_queries(k if rows_ else None) if text_key in c and str(c[text_key]).strip()), None)
            except Exception as err:
                lead = None
                self.log.append({"kind": "cover_query_error", "error": f"{type(err).__name__}: {err}"[:300]})
            leads.append(lead)
            if lead and walk_of(lead) is None and (search, lead) not in calls:
                calls.append((search, lead))
        sent = [str(a.get(text_key)) for _, a in calls]
        if calls:
            self.scripted_calls("I open the errand by searching each product's stated value with its kind word; then I read the next pages, "
                                "list the category under the budget and open the rows.", calls, "cover_q")
        pages = 0
        for lead in leads:
            for _ in range(plan[0] - 1):
                walk = walk_of(lead) if lead else None
                if walk is None or self.next_page(walk) is None or walk.get("total", 0) <= walk.get("size", 0) * len(walk["read"]) \
                        or self.cover_room(1) < 1 or self.wire.done:
                    break
                self.scripted_calls("I read the next page of that search.", [(search, {**walk["args"], page[0]: self.next_page(walk)})], "cover_p")
                pages += 1
        cheapest = self.needs.priority() == "cheapest"
        marks: dict[int, list[str]] = {}
        if cheapest:
            names = self.ref_key_words()
            for k, product in enumerate(products):
                kind, vals = self.value_terms(k if rows_ else None, product)
                kind_set = set(re.findall(r"[a-z0-9]+", kind))
                words = [w for w in re.findall(r"[a-z0-9]+", vals[0]) if w not in LINK_WORDS and w not in names and w not in kind_set] if vals else []
                if not words or self.cover_room(1) < 1 or self.wire.done:
                    continue
                marks[k] = words
                args = {text_key: " ".join(words)}
                if walk_of(args) is None:
                    self.scripted_calls("Under the cheapest rule I search the stated value alone: it shows every listing that writes it.",
                                        [(search, args)], "cover_v")
                    sent.append(args[text_key])
                walk = walk_of(args)
                while walk is not None and 0 < walk.get("total", 0) <= SWEEP_ALL and self.next_page(walk) is not None \
                        and self.cover_room(1) >= 1 and not self.wire.done:
                    self.scripted_calls("I read its next page: results come by relevance, not by price.",
                                        [(search, {**walk["args"], page[0]: self.next_page(walk)})], "cover_v")
                    pages += 1
                    walk = walk_of(args)
        shelves: list[dict[str, Any]] = []
        if plan[2]:
            cap = float(self.needs.spend_cap)
            for k, _ in enumerate(products):
                if self.cover_room(1) < 1 or self.wire.done:
                    break
                home = self.walk_home(k if rows_ else None)
                label = self.raw_label(home) if home else ""
                qty = (rows_[k]["quantity"] if rows_ else self.needs.quantity()) or 1
                call = self.shelf_call(label, cap // qty if float(cap).is_integer() else cap / qty) if label else None
                if call:
                    self.scripted_calls(f"I list the category '{label}' under the budget to see the cheap end of the shelf.", [call], "cover_s")
                    shelves.append(call[1])
        picks: list[str] = []
        batch = self.batch_tool()
        each = (batch[2] if batch else 1) * plan[1]
        for k, _ in enumerate(products):
            home = self.walk_home(k if rows_ else None)
            if not home:
                continue
            qty = (rows_[k]["quantity"] if rows_ else self.needs.quantity()) or 1
            if marks.get(k):
                cap = self.needs.spend_cap
                rows_seen: dict[str, float] = {}
                def cheap_row(d: dict[str, Any], words: list[str] = marks[k]) -> None:
                    ident = self.row_mark(d)
                    cost = self.archive.price(ident) if ident in self.archive.ids else None
                    if ident in self.archive.read or ident in picks or ident in self.refused_lines or not is_number(cost) \
                            or self.shelf_of(d) not in home or self.sold_out(d) or (is_number(cap) and float(cost) * qty > float(cap)):
                        return
                    if all(re.search(r"\b" + re.escape(w), self.title_of(d).lower()) for w in words):
                        rows_seen[ident] = float(cost)
                for rec in self.archive.items:
                    walk_dicts(rec, cheap_row)
                picks += [i for i, _ in sorted(rows_seen.items(), key=lambda kv: (kv[1], kv[0]))[:each]]
            try:
                picks += [i for i in self.menu_rows(home, each, picks, qty) if i not in picks]
            except Exception as err:
                self.log.append({"kind": "cover_rows_error", "error": f"{type(err).__name__}: {err}"[:300]})
        reads = -(-len(picks) // batch[2]) if batch and len(picks) >= 2 else len(picks)
        while picks and reads > max(0, self.cover_room(1)):
            picks.pop()
            reads = -(-len(picks) // batch[2]) if batch and len(picks) >= 2 else len(picks)
        if picks:
            self.open_queue = picks
            self.open_say = "I open the rows of each product's category to learn their fields and prices before anything is asked."
            self.open_purpose = "cover"
            self.open_recheck = True
        self.log.append({"kind": "cover", "products": n, "plan": plan, "queries": sent, "pages": pages, "shelves": shelves, "picks": picks})
        return bool(sent or pages or shelves or picks)
    def value_block(self) -> bool:
        done = self.__dict__.get("value_blocks", 0)
        search, text_key, page = self.role("search"), self.query_key(), self.page_arg()
        if done >= VALUE_BLOCKS or not search or not text_key or self.wire.done or self.wire.broke or self.wire.spare() < CLOSING + 60:
            return False
        if self.cover_room(2) < 2:
            return False
        need = str(self.needs.uncovered[-1]) if self.needs.uncovered else ""
        prop, _, value = need.partition(":")
        if not value.strip():
            return False
        rows_ = self.needs.items()
        k = self.product_of(prop) if rows_ else None
        product = rows_[k]["product"] if rows_ and k is not None and k < len(rows_) else str(self.needs.reading.get("product") or "")
        kind, vals = self.value_terms(k if rows_ else None, product)
        def norm(text: Any) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower().replace("(", " ").replace(")", " ")))
        val = next((v for v in vals if norm(v) == norm(value) or norm(value) in norm(v) or norm(v) in norm(value)), "")
        if not kind or not val:
            return False
        self.value_blocks = done + 1
        phrase = " ".join((val + " " + kind).split()[:7])
        book = self.__dict__.setdefault("page_book", {})
        def walk_of(text: str) -> dict[str, Any] | None:
            return next((w for w in book.values() if norm(" ".join(self.free_text(w["args"]))) == norm(text)), None)
        sent: list[str] = []
        if walk_of(phrase) is None:
            self.scripted_calls(f"The shopper revealed {need.strip()}: I search that value with the kind word, read its next page and open "
                                "the rows that carry it.", [(search, {text_key: phrase})], "value_q")
            sent.append(phrase)
        walk = walk_of(phrase)
        if walk is not None and page and self.next_page(walk) is not None and walk.get("total", 0) > walk.get("size", 0) * len(walk["read"]) \
                and self.cover_room(2) >= 2 and not self.wire.done:
            self.scripted_calls("I read the next page of that search.", [(search, {**walk["args"], page[0]: self.next_page(walk)})], "value_p")
            sent.append(phrase + " (next page)")
        try:
            ranked = self.value_rows()
        except Exception as err:
            ranked = []
            self.log.append({"kind": "value_rows_error", "error": f"{type(err).__name__}: {err}"[:300]})
        batch = self.batch_tool()
        most = batch[2] if batch else 2
        picks = [ident for _, _, ident, _ in ranked if not rows_ or k is None or self.product_index(ident) in (k, None)][:most]
        reads = 1 if batch and len(picks) >= 2 else len(picks)
        while picks and reads > max(0, self.cover_room(1)):
            picks.pop()
            reads = 1 if batch and len(picks) >= 2 else len(picks)
        if picks:
            self.__dict__.setdefault("named", set()).update(picks)
            self.open_queue = picks
            self.open_say = "I open the unread rows that carry the revealed value."
            self.open_purpose = "value"
            self.open_recheck = True
        self.log.append({"kind": "value_block", "need": need, "sent": sent, "picks": picks})
        return bool(sent or picks)
    def cheap_proof(self) -> str:
        if self.needs.priority() != "cheapest" or not self.trolley.units or self.wire.spare() < CLOSING + 40 or not self.shelf_tool():
            return ""
        done = self.__dict__.setdefault("proofs", {})
        for line in list(self.trolley.units)[:2]:
            mark = (line, self.needs.version)
            if mark in done or line in self.dead_lines() or line in self.refused_lines:
                continue
            price = self.archive.price(line)
            label = self.raw_label({self.shelf_known(line)} - {""})
            if not is_number(price) or not label:
                continue
            if self.moves_left() < 8 or self.wire.done:
                return ""
            done[mark] = "listed"
            bound = self.below(price)
            if bound is None:
                done[mark] = "proven"
                continue
            call = self.shelf_call(label, bound)
            if not call:
                continue
            tick0 = self.wire.tick
            fresh = self.scripted_calls(f"The cheapest is asked for: I list '{label}' priced below the cart line {line} ({float(price):g}) to see "
                                        "whether anything cheaper exists.", [call], "proof")
            if self.wire.tick == tick0:
                done[mark] = "refused"
                self.log.append({"kind": "cheap_proof", "line": line, "bound": bound, "refused": True})
                continue
            listed: dict[str, tuple[float, str, bool]] = {}
            token = None
            def visit(d: dict[str, Any]) -> None:
                nonlocal token
                token = token or self.next_token(d)
                ident = next((v for key, v in d.items() if isinstance(v, str) and is_id_key(key) and v in self.archive.ids), None)
                cost = next((v for key, v in d.items() if is_number(v) and not isinstance(v, bool) and re.search(r"price|cost", str(key).lower())), None)
                if ident and cost is not None and ident != line:
                    listed[ident] = (float(cost), self.title_of(d), self.sold_out(d))
            for rec in fresh:
                walk_dicts(rec, visit)
            if token and len(listed) >= max(2, self.page_size() or 2) and self.cover_room(1) >= 2 and not self.wire.done:
                more = self.shelf_call(label, bound, token)
                if more:
                    for rec in self.scripted_calls("I read the next page of that listing.", [more], "proof_p"):
                        walk_dicts(rec, visit)
            cheaper = sorted((c, i, t) for i, (c, t, gone) in listed.items() if c < float(price) and not gone and i not in self.refused_lines)
            if not listed:
                done[mark] = "proven"
                self.log.append({"kind": "cheap_proof", "line": line, "bound": bound, "proven": True})
                continue
            unread = [i for _, i, _ in cheaper if i not in self.archive.read][:LIST_MOST]
            self.log.append({"kind": "cheap_proof", "line": line, "bound": bound, "cheaper": [i for _, i, _ in cheaper][:8], "unread": unread})
            if not unread:
                done[mark] = "read"
                continue
            batch = self.batch_tool()
            reads = 1 if batch and len(unread) >= 2 else len(unread)
            if self.moves_left() < reads + 5:
                return ""
            done[mark] = "held"
            self.__dict__.setdefault("named", set()).update(unread)
            self.open_queue = unread
            self.open_say = "I open the cheaper rows of the shelf before the order."
            self.open_purpose = "proof"
            self.open_recheck = True
            return ("Not run yet (no step used). The cheapest is asked for, and the category listed below the cart line's price shows rows "
                    "costing less that were never opened: " + "; ".join(f"{i} '{t[:60]}' {c:g}" for c, i, t in cheaper if i in unread)
                    + ". Their records are read for you now. If one states every requirement in force in its own fields (under this "
                    "store's spellings) and is in stock, make it the line instead, re-read its stock and price, and order; otherwise order "
                    "the cart as it is.")
        return ""
    def view_before_order(self) -> str:
        tool = self.role("cart_view")
        if not tool or not self.trolley.units or self.moves_left() < 4 or self.wire.spare() < CLOSING + 20 or self.wire.done:
            return ""
        mark = (self.__dict__.get("cart_seq", 0), self.__dict__.get("cart_edits", 0), tuple(sorted(self.trolley.units)))
        if self.__dict__.get("viewed_mark") == mark:
            return ""
        self.viewed_mark = mark
        if self.__dict__.get("cart_ver", -2) == self.__dict__.get("view_ver", -3):
            return ""
        state = (self.needs.version, self.needs.told_version, self.msg_cap, self.step_cap, len(self.trolley.units),
                 tuple(sorted(self.dead_lines())), self.needs.spend_cap)
        self.scripted_calls("Before the order, I view the cart once: a store's notices and the shopper's changes can arrive there.", [(tool, {})], "view")
        changed = state != (self.needs.version, self.needs.told_version, self.msg_cap, self.step_cap, len(self.trolley.units),
                            tuple(sorted(self.dead_lines())), self.needs.spend_cap)
        self.log.append({"kind": "view_before_order", "changed": changed})
        if changed:
            return ("Not run yet (no step used). The cart view brought news: re-read the requirements, the instructions and the cart as "
                    "they stand now, fix what changed, and order again.")
        return ""
    def key_named(self, thing: dict[str, Any], menu: str) -> dict[str, Any]:
        prop = str(thing.get("property") or "")
        name = prop.split("(")[0].strip()
        words = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in LINK_WORDS]
        if not name or not words:
            return thing
        tied = re.search(r"\(([^)]+)\)", prop)
        product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
        asked = " ".join(str(thing.get("question") or "").split())
        if re.fullmatch(r"(?i)(?:which|what)\s+" + re.escape(name) + r"\s+would you like for (?:the|your|a|an)\s+[^?]{1,60}\?", asked):
            return thing
        firsts = [str(x).lower() for x in self.needs.reading.get("ask_first") or []]
        choice = any(all(w in first for w in words) and re.search(r"\bversus\b|\bvs\b|matters? more|rather|trade|priorit", first)
                     for first in firsts)
        fielded = any(strict_prop(ln.strip().lstrip("-").strip().split(" (in ")[0].strip(), name) for ln in str(menu or "").splitlines() if " (in " in ln)
        if (choice and not fielded) or re.search(r"budget|price|spend|cost", name.lower()):
            return thing
        self.log.append({"kind": "key_named", "property": name, "was": str(thing.get("question") or "")[:120]})
        return dict(thing, question=f"Which {name} would you like for the {product}?")
    def twin_lines(self) -> list[str]:
        groups: dict[str, list[str]] = {}
        for line in self.trolley.units:
            k = self.product_index(line) if self.needs.items() else None
            key = f"product {k + 1}" if k is not None else (self.shelf_known(line) or "")
            if key:
                groups.setdefault(key, []).append(line)
        return [f"{', '.join(lines)} ({key})" for key, lines in groups.items() if len(lines) >= 2]
    def bound_hint(self) -> str:
        if self.needs.priority() != "cheapest" or self.wire.spare() < CLOSING:
            return ""
        figs = []
        for req in self.needs.props_left():
            val = str(req).split(":", 1)[-1]
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s?([a-z\"']{1,4})\b", val.lower()):
                figs.append(m.group(1) + m.group(2))
        figs = list(dict.fromkeys(figs))[:3]
        if not figs:
            return ""
        return ("\n[cheapest with a measured limit] the cheapest fit often writes the limit's own figure: search " + ", ".join(figs)
                + " with the kind word (with and without a space before the unit), read two pages of each, and open the cheapest "
                  "in-budget rows before fixing the pick.")
    def basket_text(self) -> str:
        info = {}
        try:
            info = self.code_rows()
        except Exception:
            info = {}
        parts, total, priced, stocked = [], 0.0, True, True
        unit = str(self.needs.reading.get("currency") or "").strip()
        for thing, qty_n in self.trolley.units.items():
            row = info.get(thing) or {}
            cost = self.archive.price(thing)
            cost = cost if cost is not None else row.get("price")
            units = int(qty_n) if float(qty_n).is_integer() else qty_n
            name = " ".join(str(row.get("title") or "").split())[:70] or thing
            try:
                seller = self.line_seller(thing)
            except Exception:
                seller = ""
            name += f" from {seller}" if seller else ""
            stocked = stocked and row.get("stock") is True
            if cost is None:
                priced = False
                parts.append(f"{units} x {name}")
            else:
                total += float(cost) * float(qty_n)
                parts.append(f"{units} x {name} at {cost:g}{(' ' + unit) if unit else ''} each")
        if not parts:
            return "Before I finish: the basket is empty, so nothing will be ordered."
        text_body = "Before I place the order, here is the basket: " + "; ".join(parts) + "."
        if priced:
            text_body += f" The total is {total:g}{(' ' + unit) if unit else ''}."
        if stocked:
            text_body += " Every line is in stock."
        return text_body + " I will place the order now."
    def finish_steps(self, kind: str) -> str:
        quiet = not self.needs.told and not self.needs.exchanges and not self.__dict__.get("event_seen")
        if self.moves_left() < 2 or self.wire.done or self.wire.broke:
            return ""
        edits = self.__dict__.get("cart_edits", 0)
        sig = (kind, tuple(sorted(self.trolley.units.items())), edits, self.needs.told_version, self.refusal_n, self.needs.version)
        checked = self.__dict__.setdefault("finish_checked", set())
        if sig in checked:
            return ""
        checked.add(sig)
        msg_tool = self.role("message")
        msgs_left = max(0, self.ask_limit() - self.msgs_sent) if msg_tool else 0
        since = [f"- message sent at step {tick}: {said}" for tick, said, at in self.__dict__.get("msg_log", [])
                 if at == edits and (edits > 0 or not self.trolley.units)]
        try:
            told_ok = bool(since) and (kind != "order" or not self.trolley.units or self.basket_told_ok())
        except Exception as err:
            told_ok = bool(since)
            self.log.append({"kind": "basket_check_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if since and not told_ok:
            self.log.append({"kind": "basket_retold", "cart": dict(self.trolley.units)})
        told = self.needs.told
        usable = [tdef for tdef in self.toolset if tdef["function"]["name"] not in self.finishers() and tdef["function"]["name"] != self.role("cart")]
        if msgs_left <= 0:
            usable = [tdef for tdef in usable if tdef["function"]["name"] != msg_tool]
        ask_text = FINISH_CHECK
        ask_text += "\n\nINSTRUCTIONS IN FORCE:\n" + ("\n".join(f"- {t['text']} (from the {t.get('source')})" for t in told) or "(none stated)")
        ask_text += "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
        ask_text += "\n\nNEWEST SHOPPER LINES AND STORE NOTICES (oldest first):\n" + ("\n".join(self.needs.exchanges[-8:]) or "(none)")
        ask_text += "\n\nREQUIREMENTS NOW: " + self.needs.text_body()
        ask_text += "\n\nTHE CART: " + self.trolley.summary() + " " + self.basket_text()
        ask_text += "\n\nTHE FINISH ABOUT TO HAPPEN: " + kind
        ask_text += "\n\nDONE AFTER THE LAST CART CHANGE:\n" + ("\n".join(since) or "(nothing was sent or done since the last cart change)")
        ask_text += f"\n\nMESSAGES LEFT: {msgs_left}\nSTEPS LEFT: {self.moves_left()} (the finish itself takes one)"
        ask_text += "\n\nTOOLS THAT MAY CARRY OUT A STEP:\n" + "\n".join(
            f"- {tdef['function']['name']}: {tdef['function'].get('description', '')} | arguments: {as_text(tdef['function'].get('parameters') or {})}" for tdef in usable)
        answer = None
        if not quiet and self.wire.usable(FINISHER) and self.wire.spare() > LAST_CALL + 10:
            answer = self.wire.consult(ask_text, 2500, FINISHER)
        calls: list[tuple[str, dict[str, Any], str]] = []
        if answer is not None:
            names = {tdef["function"]["name"] for tdef in usable}
            for step in (answer.get("pending") or [])[:FINISH_TURNS]:
                if not isinstance(step, dict) or step.get("tool") not in names or not isinstance(step.get("args"), dict):
                    continue
                call_args, faults = ArgCheck.execute(dict(step["args"]), self.shapes[step["tool"]])
                if faults:
                    continue
                calls.append((step["tool"], call_args, str(step.get("why") or "")[:120]))
        aim = self.text_arg(msg_tool) if msg_tool else None
        if kind == "order" and self.trolley.units and aim and not told_ok and msgs_left > sum(1 for c in calls if c[0] == msg_tool) \
                and not any(c[0] == msg_tool for c in calls) and self.report_asked():
            calls.append((msg_tool, {aim: self.basket_text()}, "the basket told before the order"))
        self.log.append(dict(kind="finish_check", finish=kind, answer=answer, calls=[c[0] for c in calls]))
        calls = calls[: max(0, self.moves_left() - 1)]
        if not calls:
            return ""
        proposals = [{"id": f"finish_step_{self.wire.tick}_{n}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args)}} for n, (name, args, _) in enumerate(calls)]
        state = (self.needs.version, self.needs.told_version, self.msg_cap, self.step_cap, self.refusal_n, len(self.trolley.units))
        was = self.__dict__.get("scripted_batch")
        try:
            responses = self.dispatch(proposals, scripted=True)
        finally:
            self.scripted_batch = was
        changed = state != (self.needs.version, self.needs.told_version, self.msg_cap, self.step_cap, self.refusal_n, len(self.trolley.units))
        done = [f"{name} {json.dumps(args, ensure_ascii=False)[:400]} -> {text.split(chr(10) + '[status]')[0][:700]}"
                for (name, args, _), (_, text) in zip(calls, responses)]
        note = "[before this finish the code carried out: " + "; ".join(c[2] for c in calls) + "] " + " | ".join(done)
        self.log.append(dict(kind="finish_steps_done", note=note[:600]))
        if msg_tool and any(name == msg_tool for name, _, _ in calls) and (changed or self.needs.reply_due or self.wire.done):
            return ("Not run yet (no step used). " + note + "\nThe answer changed something: read it against the requirements and "
                    "the instructions in force, fix what it asks for, and finish again.")
        self.finish_note = note
        return ""
    def dispatch(self, proposals: list[dict[str, Any]], scripted: bool = False) -> list[tuple[str, str]]:
        responses: list[tuple[str, str]] = []
        self.scripted_batch = scripted
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
        note = self.__dict__.pop("finish_note", "")
        if note and to_send and to_send[0][1] in self.finishers():
            responses = [(cid, note + "\n" + text if cid == to_send[0][0] else text) for cid, text in responses]
        call_rank = {c["id"]: i for i, c in enumerate(proposals)}
        responses.sort(key=lambda req: call_rank.get(req[0], 0))
        return responses
    def send_chunk(self, slab: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str]]:
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
    @staticmethod
    def listing_lists(piece: Any) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        found: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        def plain(row: dict[str, Any]) -> bool:
            return sum(1 for v in row.values() if v is None or isinstance(v, (str, int, float))) >= 4
        def seek(x: Any, holder: dict[str, Any], lvl: int) -> None:
            if lvl > 6 or len(found) >= 3:
                return
            if isinstance(x, list):
                rows = [r for r in x if isinstance(r, dict)]
                if len(rows) >= 2 and len(rows) == len(x) and all(plain(r) for r in rows):
                    found.append((holder, rows))
                for r in x[:40]:
                    seek(r, {}, lvl + 1)
            elif isinstance(x, dict):
                for v in x.values():
                    seek(v, x, lvl + 1)
        seek(piece, {}, 0)
        return found
    def world_gaps(self, rows: list[dict[str, Any]]) -> list[str]:
        def numeric(v: Any) -> bool:
            return is_number(v) and not isinstance(v, bool)
        gaps = []
        if any(record_id(r) is None for r in rows):
            gaps.append("id")
        if not all(any(isinstance(v, str) and re.search(r"title|heading|headline", str(k).lower()) for k, v in r.items()) for r in rows):
            gaps.append("title")
        if not all(any(numeric(v) and re.search(r"price|cost", str(k).lower()) for k, v in r.items()) for r in rows):
            gaps.append("price")
        if not any(self.key_stock(k, v) is not None for r in rows for k, v in r.items()):
            gaps.append("stock")
        if not all(any(shelf_key(k) and isinstance(v, str) for k, v in r.items()) for r in rows):
            gaps.append("category")
        return gaps
    @staticmethod
    def world_check(answer: dict[str, Any], rows: list[dict[str, Any]], holder: dict[str, Any]) -> dict[str, Any]:
        keys = set().union(*(set(r) for r in rows))
        out: dict[str, Any] = {}
        def vals(k: str) -> list[Any]:
            return [r[k] for r in rows if k in r]
        def named(role: str) -> str | None:
            k = answer.get(role)
            return k if isinstance(k, str) and k in keys and vals(k) else None
        k = named("id")
        if k and len(vals(k)) == len(rows) and all(isinstance(x, str) and re.fullmatch(ID_LIKE, x) for x in vals(k)) \
                and len(set(vals(k))) == len(rows):
            out["id"] = k
        k = named("title")
        if k and all(isinstance(x, str) and x.strip() for x in vals(k)):
            out["title"] = k
        k = named("price")
        if k and all(is_number(x) and not isinstance(x, bool) for x in vals(k)):
            out["price"] = k
        k = named("stock")
        if k and all(isinstance(x, (bool, int, float, str)) for x in vals(k)):
            out["stock"] = k
            means = str(answer.get("stock_means") or "").lower()
            out["stock_means"] = "text" if all(isinstance(x, str) for x in vals(k)) else (
                "sold_out" if means == "sold_out" and all(isinstance(x, bool) for x in vals(k)) else "in_stock")
        k = named("category")
        if k and all(isinstance(x, str) and 0 < len(x.strip()) <= 80 for x in vals(k)):
            out["category"] = k
        for role in ("seller", "brand"):
            k = named(role)
            if k and all(x is None or isinstance(x, str) for x in vals(k)):
                out[role] = k
        nxt = answer.get("next")
        if isinstance(nxt, str) and nxt in holder and not isinstance(holder.get(nxt), (dict, list)):
            out["next"] = nxt
        roles = [r for r in out if r not in ("stock_means", "next")]
        for r in roles:
            if sum(1 for q in roles if out.get(q) == out.get(r)) > 1:
                out.pop(r, None)
        if "stock" not in out:
            out.pop("stock_means", None)
        return out
    def read_world(self, raw: Any, obs: Any) -> Any:
        if self.__dict__.get("world_read"):
            return obs
        raw_lists, seen = self.listing_lists(raw), self.listing_lists(obs)
        if not raw_lists or not seen:
            return obs
        self.world_read = "code"
        gaps = self.world_gaps(seen[0][1])
        if not gaps:
            return obs
        holder, rows = raw_lists[0]
        if not self.wire.usable(READER) or self.wire.spare() < CLOSING + 30 or self.wire.spent_out():
            self.log.append({"kind": "world_gaps", "gaps": gaps})
            return obs
        beside = {k: v for k, v in holder.items() if not isinstance(v, (dict, list))}
        ask_text = (WORLD_READ + "\n\nRECORDS:\n" + as_text([clipped(r, 160) for r in rows[:3]], 6000)
                    + "\n\nKEYS BESIDE THE RECORDS ON THIS PAGE: " + as_text(beside, 600)
                    + '\nAlso name "next": the key beside the records that gives the next page (a token or a row number), or null.')
        answer = self.wire.consult(ask_text, 2000, READER) or {}
        learned = self.world_check(answer, rows, holder)
        self.world_tries = self.__dict__.get("world_tries", 0) + 1
        self.world_read = None if not answer and self.world_tries < 2 else "model"
        self.log.append({"kind": "world_read", "gaps": gaps, "kept": learned, "try": self.world_tries,
                         "answer": {k: answer.get(k) for k in ("id", "title", "price", "stock", "stock_means", "category", "seller", "brand", "next")}})
        if not learned:
            return obs
        LENS.update(learned)
        if learned.get("id"):
            ID_KEYS.add(learned["id"])
        try:
            self.archive.relens()
        except Exception as err:
            self.log.append({"kind": "world_relens_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return lensed(raw)
    def absorb(self, call_name: str, call_args: dict[str, Any], obs: Any, resp: dict[str, Any], last: bool) -> str:
        shown, obs = obs, lensed(obs)
        try:
            obs = self.read_world(shown, obs)
        except Exception as err:
            self.world_read = "error"
            self.log.append({"kind": "world_read_error", "error": f"{type(err).__name__}: {err}"[:300]})
        msg_tool, basket_tool = self.role("message"), self.role("cart")
        if call_name == msg_tool:
            self.msgs_sent += 1
            said = " ".join(str(val) for val in call_args.values() if isinstance(val, str))
            self.__dict__.setdefault("msg_log", []).append((self.wire.tick, said[:600], self.__dict__.get("cart_edits", 0)))
            if not self.__dict__.get("scripted_batch"):
                self.questions_log.append(f"- (the assistant's own message): \"{said[:300]}\" -> {as_text(obs)[:300]}")
        if call_name == basket_tool:
            try:
                sig = tuple(sorted((line, units) for line, units in self.trolley.lines(call_args)))
            except Exception:
                sig = None
            if sig is None or sig != self.__dict__.get("cart_sig") or self.trolley.adds:
                self.cart_edits = self.__dict__.get("cart_edits", 0) + 1
            self.cart_sig = sig
            try:
                self.__dict__.setdefault("ever_held", set()).update(line for line, _ in self.trolley.lines(call_args))
            except Exception:
                pass
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
                hits = {v for k, v in d.items() if isinstance(v, str) and v in self.trolley.units and is_id_key(k)}
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
            self.view_ver = self.__dict__.get("cart_ver", -2)
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
        caught = self.shopper_lines(call_name, obs, refused)
        if caught:
            ver_before = self.needs.version
            loose_before, cap_before = len(self.needs.let_go), self.needs.spend_cap
            remarks.append(self.needs.hear(caught, qtext))
            self.ref_take()
            news = self.needs.__dict__.get("allowance_news")
            if news:
                self.needs.allowance_news = None
                try:
                    self.apply_allowance(news)
                except Exception as err:
                    self.log.append({"kind": "allowance_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if self.needs.told_version != self.__dict__.get("told_seen"):
                self.told_seen = self.needs.told_version
                self.order_vetted = False
            if self.needs.version > ver_before and (qtext is None or len(self.needs.let_go) > loose_before
                                                     or self.needs.spend_cap != cap_before):
                try:
                    review = self.change_review()
                    if review:
                        remarks.append(review)
                except Exception as err:
                    self.log.append({"kind": "change_review_error", "error": f"{type(err).__name__}: {err}"[:300]})
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
        opener += "\n\n" + self.status_line()
        self.chat = [{"role": "system", "content": ACT_RULES + (KIT_RULES if self.linked_products() else "")},
                     {"role": "user", "content": opener}]
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
            if self.cover_step():
                continue
            if self.fit_sweep():
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
